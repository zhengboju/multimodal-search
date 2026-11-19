"""Stage 4: LoRA / QLoRA SFT of a Qwen2(.5)-VL model as the query-understanding module.

torchrun --nproc_per_node 2 -m mmsearch.train.train_qlora --config configs/query_rewrite.yaml
Output: <run>/adapter (PEFT) and <run>/merged (full bf16 weights; serve with vLLM without LoRA plumbing):
    bash scripts/serve_vlm.sh <run>/merged 8002 rewriter
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ..models.colpali import DEFAULT_LORA_TARGET
from ..models.compat import from_pretrained, pick_attn_impl
from ..query.prompts import REWRITE_SYSTEM
from ..utils import dist as D
from ..utils.config import load_config
from ..utils.hub import resolve_model
from ..utils.misc import get_logger, read_jsonl, seed_everything
from ..utils.paths import runs_dir
from ..utils.tracker import Tracker

log = get_logger("train_qlora")


def build_messages(query: str, target: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": f"Query: {query}"}]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs


class SFTData(Dataset):
    """Tokenises chat examples; loss only on the assistant answer."""

    def __init__(self, rows: list[dict], processor, max_len: int):
        self.items = []
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        for r in rows:
            prompt = processor.apply_chat_template(build_messages(r["query"]), tokenize=False, add_generation_prompt=True)
            full = processor.apply_chat_template(build_messages(r["query"], r["target"]), tokenize=False)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
            labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
            if all(l == -100 for l in labels):
                continue
            self.items.append((f_ids, labels))
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]

    def collate(self, batch):
        L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        am = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, : len(x)] = torch.tensor(x)
            lab[i, : len(y)] = torch.tensor(y)
            am[i, : len(x)] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": am}


def main() -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    tc = cfg.train
    device = D.init_distributed()
    seed_everything(int(cfg.get("seed", 0)))
    run_dir = Path(tc.get("out_dir") or runs_dir() / "query" / f"{cfg.name}_sft")
    tracker = Tracker(run_dir, cfg, enabled=D.is_main())
    data_dir = Path(tc.get("data_dir") or runs_dir() / "query" / cfg.name)

    path = resolve_model(tc.base_model)
    processor = AutoProcessor.from_pretrained(path)
    kw = {"attn_implementation": pick_attn_impl("auto")}
    if tc.get("qlora", False):
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": device.index}
    model = from_pretrained(AutoModelForImageTextToText, path, torch.bfloat16, **kw)
    model.config.use_cache = False
    if tc.get("qlora", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.to(device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lc = tc.lora
    model = get_peft_model(model, LoraConfig(r=int(lc.r), lora_alpha=int(lc.alpha), lora_dropout=float(lc.dropout),
                                             target_modules=DEFAULT_LORA_TARGET, bias="none", task_type="CAUSAL_LM"))
    if D.is_main():
        model.print_trainable_parameters()
    ddp = DDP(model, device_ids=[device.index]) if D.is_dist() else model

    train = SFTData(read_jsonl(data_dir / "sft_train.jsonl"), processor, int(tc.get("max_len", 512)))
    dev = SFTData(read_jsonl(data_dir / "sft_dev.jsonl"), processor, int(tc.get("max_len", 512)))
    sampler = DistributedSampler(train, shuffle=True, seed=int(cfg.get("seed", 0))) if D.is_dist() else None
    loader = DataLoader(train, batch_size=int(tc.per_device_batch), sampler=sampler, shuffle=sampler is None,
                        collate_fn=train.collate, drop_last=True)
    accum = int(tc.get("grad_accum", 1))
    total = int(tc.epochs) * len(loader) // accum
    warm = max(1, int(float(tc.get("warmup_ratio", 0.03)) * total))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(tc.lr), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))

    @torch.no_grad()
    def dev_loss() -> float:
        model.eval()
        losses = []
        for b in DataLoader(dev, batch_size=16, collate_fn=dev.collate):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses.append(float(model(**b).loss))
        model.train()
        return sum(losses) / max(1, len(losses))

    step, micro, t0 = 0, 0, time.time()
    model.train()
    for epoch in range(int(tc.epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ddp(**b).loss
            (loss / accum).backward()
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                tracker.log({"train/loss": float(D.reduce_mean(loss.detach())), "train/lr": sched.get_last_lr()[0]}, step)
                log.info("ep %d step %d/%d loss %.4f (%.0fs)", epoch, step, total, float(loss), time.time() - t0)
        if D.is_main():
            dl = dev_loss()
            tracker.log({"dev/loss": dl}, step)
            log.info("epoch %d dev loss %.4f", epoch, dl)
        D.barrier()

    if D.is_main():
        model.save_pretrained(str(run_dir / "adapter"))
        processor.save_pretrained(str(run_dir / "adapter"))
        if tc.get("merge", True):  # reload bf16 base (also for QLoRA) and merge -> plain checkpoint for vLLM
            from peft import PeftModel

            del ddp, model
            torch.cuda.empty_cache()
            base = from_pretrained(AutoModelForImageTextToText, path, torch.bfloat16)
            merged = PeftModel.from_pretrained(base, str(run_dir / "adapter")).merge_and_unload()
            merged.save_pretrained(str(run_dir / "merged"), safe_serialization=True)
            processor.save_pretrained(str(run_dir / "merged"))
            log.info("merged model -> %s", run_dir / "merged")
    tracker.finish()
    D.cleanup()


if __name__ == "__main__":
    main()
