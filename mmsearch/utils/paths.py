"""Filesystem layout. Everything large lives under $MMS_STORAGE (default: <repo>/storage).

storage/
  models/      downloaded checkpoints (ModelScope first, HF fallback)
  raw/         raw dataset files (parquet / webdataset tars)
  processed/   unified retrieval datasets (corpus.jsonl / queries.jsonl / qrels.tsv / images/)
  cache/       embedding caches keyed by encoder tag + dataset
  runs/        training runs, benchmark outputs, reports
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def storage_dir() -> Path:
    p = Path(os.environ.get("MMS_STORAGE", repo_root() / "storage"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(name: str) -> Path:
    p = storage_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    return _sub("models")


def raw_dir() -> Path:
    return _sub("raw")


def processed_dir() -> Path:
    return _sub("processed")


def cache_dir() -> Path:
    return _sub("cache")


def runs_dir() -> Path:
    return _sub("runs")
