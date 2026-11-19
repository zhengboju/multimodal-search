from __future__ import annotations

import itertools
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def get_logger(name: str = "mmsearch") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        rank = os.environ.get("RANK")
        prefix = f"[rank{rank}] " if rank is not None else ""
        h.setFormatter(logging.Formatter(f"%(asctime)s {prefix}%(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(os.environ.get("MMS_LOGLEVEL", "INFO"))
        logger.propagate = False
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def batched(items: Iterable[Any], n: int) -> Iterator[list[Any]]:
    it = iter(items)
    while chunk := list(itertools.islice(it, n)):
        yield chunk


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=str)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict], path: str | Path, append: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def _fmt(v: Any, digits: int) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def markdown_table(rows: Sequence[dict], columns: Sequence[str] | None = None, digits: int = 2) -> str:
    """Render list-of-dicts as a GitHub markdown table."""
    if not rows:
        return "(empty)"
    columns = list(columns or rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c, ""), digits) for c in columns) + " |")
    return "\n".join(lines)
