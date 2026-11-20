"""Stage 2: reproduce ColQwen2 (ColPali recipe on Qwen2-VL-2B): LoRA + 128-d projection, MaxSim, in-batch negatives.

torchrun --nproc_per_node 2 -m mmsearch.train.train_colpali --config configs/colpali_vidore.yaml

Data: vidore/colpali_train_set (the paper's 118k query-page pairs; ModelScope mirror), read lazily from parquet.
Dev:  the set's own `test` split (in-domain) — ViDoRe is touched once, at the end, by the benchmark.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DistributedSampler

from ..data.preprocess import download_colpali_train
from ..eval.metrics import evaluate_run
from ..models.colpali import ColQwen2Encoder, build_doc_inputs, build_query_inputs, pad_multivectors
from ..models.losses import colbert_ce_loss, colbert_pairwise_ce_loss, maxsim_scores
from ..utils import dist as D
from ..utils.config import load_config
from ..utils.dataloader import make_loader
from ..utils.misc import get_logger, seed_everything, write_json
from ..utils.paths import cache_dir, runs_dir
from ..utils.tracker import Tracker

log = get_logger("train_colpali")


def load_split(root: str, pattern: str, limit: int | None = None):
    import datasets

    files = sorted(str(p) for p in Path(root).glob(pattern))
    if not files:
        raise FileNotFoundError(f"{pattern} under {root}")
    ds = datasets.load_dataset("parquet", data_files=files, split="train", cache_dir=str(cache_dir() / "hf_datasets"))
    ds = ds.select_columns(["image", "query"])
    ds = ds.filter(lambda q: q is not None and len(q.strip()) > 0, input_columns=["query"], num_proc=8)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def to_pil(x):
    """HF datasets decodes parquet image columns to PIL when feature metadata is present; else {'bytes': ...}."""
    if isinstance(x, dict):
        import io

        from PIL import Image

        x = Image.open(io.BytesIO(x["bytes"]))
    return x.convert("RGB")


class PairDataset(Dataset):
    def __init__(self, hf):
        self.hf = hf

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, i: int):
        r = self.hf[i]
        return to_pil(r["image"]), r["query"]


class PairCollator:
    """Picklable collate: holds only the processor + encoder config, never the model."""

    def __init__(self, processor, cfg):
        self.processor, self.cfg = processor, cfg

    def __call__(self, items):
        images, queries = zip(*items)
        return build_query_inputs(self.processor, self.cfg, queries), build_doc_inputs(self.processor, self.cfg, images)


@torch.no_grad()
def dev_eval(enc: ColQwen2Encoder, dev, batch: int = 16) -> dict[str, float]:
    enc.eval()
    imgs = [to_pil(dev[i]["image"]) for i in range(len(dev))]
    qs = [dev[i]["query"] for i in range(len(dev))]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        D_ = enc.encode_images(imgs, batch_size=batch, num_workers=0, desc=None)
        Q = enc.encode_queries(qs, batch_size=64)
    d, dm = pad_multivectors(D_, enc.device, torch.bfloat16)
    run = {}
    for i in range(0, len(Q), 32):
        q, qm = pad_multivectors(Q[i : i + 32], enc.device, torch.bfloat16)
        sc = maxsim_scores(q, qm, d, dm, doc_chunk=64).cpu()
        for j, row in enumerate(sc):
            top = row.topk(min(10, row.numel()))
            run[f"q{i + j}"] = {f"d{k}": float(v) for k, v in zip(top.indices.tolist(), top.values.tolist())}
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(len(Q))}
    agg, _ = evaluate_run(run, qrels, ["ndcg@5", "recall@1", "recall@5"])
    enc.train()
    return agg


def linear_warmup_decay(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    device = D.init_distributed()
    seed_everything(int(cfg.seed) + D.rank())
    run_dir = Path(cfg.get("out_dir") or runs_dir() / "colpali" / cfg.name)
    tracker = Tracker(run_dir, cfg, enabled=D.is_main())

    # data (rank 0 downloads/converts first, others reuse the cache)
    if D.is_main():
        root = download_colpali_train(cfg.data.get("shards", 35))
    D.barrier()
    root = download_colpali_train(cfg.data.get("shards", 35))
    if D.is_main():
        train_hf = load_split(root, "data/train-*.parquet", cfg.data.get("max_train"))
        dev_hf = load_split(root, "data/test-*.parquet", int(cfg.data.get("dev_size", 500)))
    D.barrier()
    if not D.is_main():
        train_hf = load_split(root, "data/train-*.parquet", cfg.data.get("max_train"))
        dev_hf = load_split(root, "data/test-*.parquet", int(cfg.data.get("dev_size", 500)))
    log.info("train pairs %d, dev %d", len(train_hf), len(dev_hf))

    enc_spec = {"model": cfg.base_model, "backend": "custom", "dim": int(cfg.get("dim", 128)),
                "max_pixels": int(cfg.max_pixels), "min_pixels": int(cfg.get("min_pixels", 4 * 28 * 28)),
                "query_augmentation": int(cfg.get("query_augmentation", 10))}
    if cfg.get("init_checkpoint"):
        enc_spec = {"checkpoint": cfg.init_checkpoint}
    enc = ColQwen2Encoder(enc_spec, device=device, for_training=True)
    if not enc._peft:
        lc = cfg.lora
        enc.enable_lora(int(lc.r), int(lc.alpha), float(lc.dropout))
    enc.enable_gradient_checkpointing()
    enc.train()
    model = DDP(enc, device_ids=[device.index], static_graph=bool(cfg.get("ddp_static_graph", False))) if D.is_dist() else enc

    ds = PairDataset(train_hf)
    sampler = DistributedSampler(ds, shuffle=True, seed=int(cfg.seed)) if D.is_dist() else None
    bs = int(cfg.per_device_batch)
    loader = make_loader(ds, bs, collate_fn=PairCollator(enc.processor, enc.cfg), num_workers=int(cfg.get("num_workers", 8)),
                         shuffle=sampler is None, sampler=sampler, drop_last=True,
                         batch_bytes=bs * 3 * 2**20)  # ~3 MB per decoded page image
    accum = int(cfg.get("grad_accum", 1))
    total = int(cfg.epochs) * len(loader) // accum
    params = enc.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.get("wd", 0.0)))
    warm = int(cfg.get("warmup_steps", 100))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: linear_warmup_decay(s, total, warm))
    loss_kind = cfg.get("loss", "pairwise")
    gather = bool(cfg.get("gather_negatives", True))
    eval_every, save_every = int(cfg.get("eval_every", 500)), int(cfg.get("save_every", 1000))

    def maybe_eval(step: int) -> None:
        if D.is_main():
            res = dev_eval(enc, dev_hf)
            tracker.log({f"dev/{k}": v for k, v in res.items()}, step)
            log.info("step %d dev %s", step, {k: round(v, 4) for k, v in res.items()})
        D.barrier()

    step, micro = 0, 0
    if cfg.get("eval_at_start", True):
        maybe_eval(0)
    t0, seen = time.time(), 0
    for epoch in range(int(cfg.epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        it, wait = iter(loader), 0.0
        while True:
            tw = time.time()
            try:
                q_in, d_in = next(it)
            except StopIteration:
                break
            wait += time.time() - tw
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q, qm, d, dm = model(q_in, d_in)
            if gather:
                d_all, dm_all = D.gather_padded(d, dm)
                pos = torch.arange(q.shape[0], device=device) + D.rank() * q.shape[0]
            else:
                d_all, dm_all, pos = d, dm, torch.arange(q.shape[0], device=device)
            scores = maxsim_scores(q, qm, d_all, dm_all, doc_chunk=64)
            if loss_kind == "pairwise":
                loss = colbert_pairwise_ce_loss(scores, pos)
            else:
                loss = colbert_ce_loss(scores, pos, qm.sum(1).float(), float(cfg.get("temperature", 0.02)))
            (loss / accum).backward()
            micro += 1
            seen += q.shape[0] * D.world_size()
            if micro % accum:
                continue
            gn = torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % int(cfg.get("log_every", 10)) == 0:
                el = time.time() - t0
                tracker.log({"train/loss": float(D.reduce_mean(loss.detach())), "train/lr": sched.get_last_lr()[0],
                             "train/grad_norm": float(gn), "train/pairs_per_s": seen / el,
                             "train/peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}, step)
                log.info("ep %d step %d/%d loss %.4f %.1f pairs/s mem %.1fGB", epoch, step, total, float(loss),
                         seen / el, torch.cuda.max_memory_allocated() / 2**30)
            if step % eval_every == 0:
                maybe_eval(step)
            if step % save_every == 0 and D.is_main():
                enc.save(run_dir / f"step{step}")
    maybe_eval(step)
    if D.is_main():
        enc.save(run_dir / "final")
        write_json({"steps": step, "pairs_seen": seen, "hours": (time.time() - t0) / 3600}, run_dir / "final" / "train_info.json")
        log.info("saved -> %s ; evaluate on ViDoRe with configs/vidore_benchmark.yaml (encoder.checkpoint=%s)", run_dir / "final", run_dir / "final")
    tracker.finish()
    D.cleanup()


if __name__ == "__main__":
    main()
