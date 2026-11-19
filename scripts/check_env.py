"""Environment sanity check (stage 0, D1): GPUs, kernels, optional packages, hub reachability, disk."""

from __future__ import annotations

import importlib
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def ok(flag: bool) -> str:
    return "OK " if flag else "-- "


def main() -> None:
    import torch

    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | cuda {torch.version.cuda}")
    n = torch.cuda.device_count()
    print(f"{ok(n >= 2)}GPUs: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"    [{i}] {p.name}  {p.total_memory / 2**30:.0f} GB  bf16={torch.cuda.is_bf16_supported()}")
    for mod in ["transformers", "vllm", "flash_attn", "peft", "faiss", "matplotlib", "pytrec_eval", "modelscope", "bitsandbytes", "wandb"]:
        try:
            m = importlib.import_module(mod)
            print(f"{ok(True)}{mod} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"{ok(False)}{mod} ({type(e).__name__})")
    try:
        from transformers import ColQwen2ForRetrieval  # noqa: F401

        print(f"{ok(True)}transformers.ColQwen2ForRetrieval")
    except ImportError:
        print(f"{ok(False)}transformers.ColQwen2ForRetrieval (needs transformers>=4.53)")
    for name, url in [("modelscope", "https://www.modelscope.cn/api/v1/models/Qwen/Qwen2-VL-2B-Instruct"),
                      ("huggingface", "https://huggingface.co/api/models/openai/clip-vit-large-patch14"),
                      ("hf-mirror", "https://hf-mirror.com/api/models/openai/clip-vit-large-patch14")]:
        try:
            urllib.request.urlopen(url, timeout=8)
            print(f"{ok(True)}{name} reachable")
        except Exception as e:  # noqa: BLE001
            print(f"{ok(False)}{name} unreachable ({type(e).__name__})")
    from mmsearch.utils.paths import storage_dir

    free = shutil.disk_usage(storage_dir()).free / 2**30
    print(f"{ok(free > 300)}storage {storage_dir()} free {free:.0f} GB (need ~300 GB for COCO+ViDoRe+colpali subset+models)")


if __name__ == "__main__":
    main()
