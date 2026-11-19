"""Retriever layer: sparse (BM25), dense (CLIP), multi-vector (ColQwen2), fusion, VLM rerank."""

from __future__ import annotations

from typing import Any

from .base import Retriever, Run


def build_retriever(cfg: dict[str, Any]) -> Retriever:
    """Factory for `type: bm25 | dense | multivector` retrievers (fusion / rerank are composed by the runner)."""
    cfg = dict(cfg)
    t = cfg.pop("type")
    name = cfg.pop("name", None)
    if t == "bm25":
        from .sparse import BM25Retriever

        return BM25Retriever(name=name or "bm25", **cfg)
    if t == "dense":
        from .dense import DenseRetriever

        return DenseRetriever(name=name, **cfg)
    if t == "multivector":
        from .multivector import MultiVectorRetriever

        return MultiVectorRetriever(name=name, **cfg)
    raise ValueError(f"unknown retriever type {t}")


__all__ = ["Retriever", "Run", "build_retriever"]
