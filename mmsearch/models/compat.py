"""Transformers version compatibility shims (server: transformers 5.x; code also works on >=4.50)."""

from __future__ import annotations

from typing import Any

import torch
import transformers
from packaging.version import Version

_TF_VERSION = Version(transformers.__version__)


def from_pretrained(cls: Any, path: str, dtype: torch.dtype | None = None, **kw: Any) -> Any:
    """`dtype=` on transformers>=4.56 (and 5.x), `torch_dtype=` before."""
    if dtype is not None:
        kw["dtype" if _TF_VERSION >= Version("4.56") else "torch_dtype"] = dtype
    return cls.from_pretrained(path, **kw)


def pick_attn_impl(preferred: str | None) -> str:
    if preferred in (None, "auto"):
        preferred = "flash_attention_2"
    if preferred == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401

            if torch.cuda.is_available():
                return "flash_attention_2"
        except ImportError:
            pass
        return "sdpa"
    return preferred


def text_hidden_size(config: Any) -> int:
    tc = getattr(config, "text_config", None)
    if tc is not None and getattr(tc, "hidden_size", None):
        return int(tc.hidden_size)
    return int(config.hidden_size)


def parse_dtype(name: str | torch.dtype | None) -> torch.dtype | None:
    if name is None or isinstance(name, torch.dtype):
        return name
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32}[name]
