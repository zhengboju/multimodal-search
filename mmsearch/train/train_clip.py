"""Stage 1: CLIP image-text retrieval fine-tuning on COCO (Karpathy train) with loss / negative-sampling ablations.

torchrun --nproc_per_node 2 -m mmsearch.train.train_clip --config configs/clip_coco.yaml
torchrun --nproc_per_node 2 -m mmsearch.train.train_clip --config configs/clip_coco.yaml loss=siglip name=clip_siglip
torchrun --nproc_per_node 2 -m mmsearch.train.train_clip --config configs/clip_coco.yaml negatives=mixed name=clip_hn_mixed

E1 negatives: inbatch | hard (positive vs mined negatives only) | mixed (in-batch + mined) | random (control)
E2 loss:      infonce | siglip | triplet
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DistributedSampler

from ..data.hard_negatives import mine_hard_negatives, random_negatives
from ..data.schema import RetrievalDataset, load_dataset
from ..eval.eval_clip import bidirectional_recall
from ..models.clip_wrapper import CLIPWrapper
from ..models.losses import SigLIPHead, infonce_loss, siglip_loss, triplet_loss
from ..utils import dist as D
from ..utils.config import load_config
from ..utils.dataloader import make_loader
from ..utils.misc import get_logger, seed_everything, write_json
from ..utils.paths import runs_dir
from ..utils.tracker import Tracker

log = get_logger("train_clip")


# ----------------------------------------------------------------------------- data
class ImageCaptionData:
    """Groups a processed caption->image dataset by image; holds out `dev_holdout` images for model selection."""

    def __init__(self, ds: RetrievalDataset, dev_holdout: int, seed: int):
        caps: dict[str, list[str]] = {}
        for qid, rels in ds.qrels.items():
            img = next(iter(rels))
            caps.setdefault(img, []).append(ds.queries[qid].text or "")
        ids = sorted(caps)
        random.Random(seed).shuffle(ids)
        self.dev_ids, self.train_ids = ids[:dev_holdout], ids[dev_holdout:]
        self.paths = {d: ds.corpus[d].image for d in ids}
        self.caps = caps

    def flat(self, ids: list[str]) -> tuple[list[str], list[str], torch.Tensor]:
        texts, t2i = [], []
        for i, d in enumerate(ids):
            for c in self.caps[d]:
                texts.append(c)
                t2i.append(i)
        return [self.paths[d] for d in ids], texts, torch.tensor(t2i)


class TrainSet(Dataset):
    def __init__(self, data: ImageCaptionData, transform, n_hard: int):
        self.ids = data.train_ids
        self.data = data
        self.transform = transform
        self.n_hard = n_hard
        self.hard: list[np.ndarray] | None = None
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.ids)

    def _img(self, i: int) -> torch.Tensor:
        return self.transform(Image.open(self.data.paths[self.ids[i]]).convert("RGB"))

    def __getitem__(self, i: int):
        rng = random.Random(hash((i, self.epoch)))
        item = {"pixel": self._img(i), "caption": rng.choice(self.data.caps[self.ids[i]]), "img_idx": i}
        if self.hard is not None and self.n_hard > 0:
            negs = self.hard[i]
            pick = [int(negs[rng.randrange(len(negs))]) for _ in range(self.n_hard)] if len(negs) else [rng.randrange(len(self.ids))]
            item["neg_pixel"] = torch.stack([self._img(j) for j in pick])
            item["neg_idx"] = torch.tensor(pick)
        return item


def make_collate(tokenizer):
    def collate(items):
        out = {"pixel": torch.stack([x["pixel"] for x in items]), "img_idx": torch.tensor([x["img_idx"] for x in items])}
        tok = tokenizer([x["caption"] for x in items], padding=True, truncation=True, max_length=77, return_tensors="pt")
        out["input_ids"], out["attention_mask"] = tok["input_ids"], tok["attention_mask"]
        if "neg_pixel" in items[0]:
            out["neg_pixel"] = torch.cat([x["neg_pixel"] for x in items])
            out["neg_idx"] = torch.cat([x["neg_idx"] for x in items])
        return out

    return collate


# ----------------------------------------------------------------------------- distributed encoding (mining / eval)
@torch.no_grad()
def dist_encode(n: int, encode_fn, device) -> torch.Tensor:
    """Each rank encodes indices r::W; results are all-gathered and re-ordered."""
    W, r = D.world_size(), D.rank()
    idx = list(range(r, n, W))
    emb = encode_fn(idx).to(device)
    per = math.ceil(n / W)
    if emb.shape[0] < per:
        emb = torch.cat([emb, emb.new_zeros(per - emb.shape[0], emb.shape[1])])
    allv = D.all_gather_no_grad(emb.contiguous()).cpu()  # [W*per, D]
    out = torch.empty(n, allv.shape[1])
    for rr in range(W):
        ids = list(range(rr, n, W))
        out[ids] = allv[rr * per : rr * per + len(ids)]
    return out


def encode_split(clip: CLIPWrapper, images: list[str], texts: list[str], device, bs: int):
    img = dist_encode(len(images), lambda ids: clip.encode_images([images[i] for i in ids], bs, desc=None), device)
    txt = dist_encode(len(texts), lambda ids: clip.encode_texts([texts[i] for i in ids], bs * 4, desc=None), device)
    return img, txt


# ----------------------------------------------------------------------------- training
class ClipTrainModule(nn.Module):
    def __init__(self, clip: CLIPWrapper, loss: str):
        super().__init__()
        self.clip = clip
        self.head = SigLIPHead() if loss == "siglip" else None
        if loss != "infonce":  # logit_scale unused by siglip/triplet -> freeze so DDP doesn't wait on it
            clip.model.logit_scale.requires_grad_(False)

    def forward(self, pixel, input_ids, attention_mask, neg_pixel=None):
        return self.clip(pixel, input_ids, attention_mask, neg_pixel)


def param_groups(module: nn.Module, wd: float):
    decay, no_decay = [], []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "bias" in n or "norm" in n or "logit" in n or "embedding" in n else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    device = D.init_distributed()
    seed_everything(int(cfg.seed) + D.rank())
    run_dir = Path(cfg.get("out_dir") or runs_dir() / "clip" / cfg.name)
    tracker = Tracker(run_dir, cfg, enabled=D.is_main())

    clip = CLIPWrapper(cfg.model, gradient_checkpointing=bool(cfg.get("grad_ckpt", True))).to(device)
    module = ClipTrainModule(clip, cfg.loss).to(device)
    model = DDP(module, device_ids=[device.index]) if D.is_dist() else module

    data = ImageCaptionData(load_dataset(cfg.train_dataset), int(cfg.dev_holdout), int(cfg.seed))
    negatives = cfg.get("negatives", "inbatch")
    hn = cfg.get("hard_neg", {})
    n_hard = int(hn.get("num_per_sample", 1)) if negatives in ("hard", "mixed", "random") else 0
    train_set = TrainSet(data, clip.transform(train=bool(cfg.get("augment", False))), n_hard)
    sampler = DistributedSampler(train_set, shuffle=True, seed=int(cfg.seed)) if D.is_dist() else None
    bs = int(cfg.per_device_batch)
    loader = make_loader(train_set, bs, collate_fn=make_collate(clip.tokenizer), num_workers=int(cfg.get("num_workers", 12)),
                         shuffle=sampler is None, sampler=sampler, drop_last=True)
    epochs = int(cfg.epochs)
    total = epochs * len(loader)
    opt = torch.optim.AdamW(param_groups(module, float(cfg.wd)), lr=float(cfg.lr), betas=(0.9, 0.98), eps=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, total, int(cfg.get("warmup_ratio", 0.05) * total)))
    dev_imgs, dev_txts, dev_t2i = data.flat(data.dev_ids)
    train_imgs, train_txts, train_t2i = data.flat(data.train_ids)
    eval_bs = int(cfg.get("eval_batch", 256))

    def evaluate(tag: str, step: int) -> dict:
        clip.eval()
        img, txt = encode_split(clip, dev_imgs, dev_txts, device, eval_bs)
        clip.train()
        res = bidirectional_recall(img, txt, dev_t2i)
        tracker.log({f"dev/{k}": v for k, v in res.items()}, step)
        log.info("[%s] dev %s", tag, {k: round(v, 2) for k, v in res.items()})
        return res

    def mine(epoch: int) -> None:
        if negatives == "random":
            train_set.hard = random_negatives(len(train_set), int(hn.get("pool", 20)), int(cfg.seed) + epoch)
            return
        t0 = time.time()
        clip.eval()
        img, txt = encode_split(clip, train_imgs, train_txts, device, eval_bs)
        clip.train()
        train_set.hard = mine_hard_negatives(img, txt, train_t2i, tuple(hn.get("rank_range", [3, 50])), int(hn.get("pool", 20)), str(device))
        tracker.log({"mine/seconds": time.time() - t0}, epoch * len(loader))
        log.info("mined hard negatives in %.1fs", time.time() - t0)

    step, best = 0, -1.0
    if cfg.get("eval_at_start", True):
        best = evaluate("zero-shot", 0)["rsum"]
    mine_every = int(hn.get("mine_every", 1))
    for epoch in range(epochs):
        train_set.epoch = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        elif hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        if n_hard and (epoch % mine_every == 0):
            mine(epoch)
        model.train()
        t_epoch, seen = time.time(), 0
        for batch in loader:
            pixel = batch["pixel"].to(device, non_blocking=True)
            ids, am = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            negp = batch.get("neg_pixel")
            negp = negp.to(device, non_blocking=True) if negp is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                img, txt, extra = model(pixel, ids, am, negp)
            img, txt = img.float(), txt.float()
            img_all, txt_all = D.all_gather_with_grad(img), D.all_gather_with_grad(txt)
            offset = D.rank() * img.shape[0]
            extra_all, invalid = None, None
            if extra is not None:
                extra_all = D.all_gather_with_grad(extra.float())
                # mask mined negatives that coincide with a row's own positive image
                pos_ids = batch["img_idx"].to(device)
                neg_ids = D.all_gather_no_grad(batch["neg_idx"].to(device))
                invalid = neg_ids.unsqueeze(0) == pos_ids.unsqueeze(1)
            if cfg.loss == "infonce":
                loss = infonce_loss(txt, img, txt_all, img_all, clip.logit_scale, offset, extra_all,
                                    hard_only=negatives == "hard", extra_invalid=invalid)
            elif cfg.loss == "siglip":
                loss = siglip_loss(txt, img_all, module.head, offset, extra_all, invalid)
            elif cfg.loss == "triplet":
                loss = triplet_loss(txt, img, txt_all, img_all, float(cfg.get("margin", 0.2)), offset, extra_all, invalid)
            else:
                raise ValueError(cfg.loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(module.parameters(), float(cfg.get("max_grad_norm", 1.0)))
            opt.step()
            sched.step()
            with torch.no_grad():
                clip.logit_scale.clamp_(0, math.log(100))
            step += 1
            seen += pixel.shape[0] * D.world_size()
            if step % int(cfg.get("log_every", 20)) == 0:
                el = time.time() - t_epoch
                tracker.log({"train/loss": float(D.reduce_mean(loss.detach())), "train/lr": sched.get_last_lr()[0],
                             "train/grad_norm": float(gn), "train/logit_scale": float(clip.logit_scale.exp()),
                             "train/img_per_s": seen / el,
                             "train/peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}, step)
                log.info("ep %d step %d/%d loss %.4f %.0f img/s mem %.1fGB", epoch, step, total, float(loss),
                         seen / el, torch.cuda.max_memory_allocated() / 2**30)
        tracker.log({"train/epoch_seconds": time.time() - t_epoch}, step)
        if (epoch + 1) % int(cfg.get("eval_every_epochs", 1)) == 0 or epoch == epochs - 1:
            res = evaluate(f"epoch{epoch}", step)
            if res["rsum"] > best and D.is_main():
                best = res["rsum"]
                clip.save(run_dir / "best")
                write_json({"epoch": epoch, **res}, run_dir / "best" / "dev_metrics.json")
        D.barrier()
    if D.is_main():
        clip.save(run_dir / "last")
        tracker.summary({"best_dev_rsum": best})
        log.info("done. test once with: python -m mmsearch.eval.eval_clip --model %s --dataset coco_test --folds 1 5", run_dir / "best")
    tracker.finish()
    D.cleanup()


if __name__ == "__main__":
    main()
