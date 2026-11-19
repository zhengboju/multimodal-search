"""Run-level fusion: RRF and normalised weighted sum. Fusion is applied to already-computed runs, so
hybrid variants (BM25 ⊕ ColQwen2, …) cost nothing extra to evaluate."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.schema import Query, RetrievalDataset
from .base import Retriever, Run


def rrf(runs: Sequence[Run], k: int = 60, weights: Sequence[float] | None = None, depth: int = 100) -> Run:
    weights = list(weights or [1.0] * len(runs))
    out: Run = {}
    qids = set().union(*[r.keys() for r in runs])
    for qid in qids:
        acc: dict[str, float] = {}
        for w, run in zip(weights, runs):
            ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])[:depth]
            for rank, (did, _) in enumerate(ranked):
                acc[did] = acc.get(did, 0.0) + w / (k + rank + 1)
        out[qid] = acc
    return out


def _normalize(scores: dict[str, float], method: str) -> dict[str, float]:
    if not scores:
        return {}
    v = np.asarray(list(scores.values()), dtype=np.float64)
    if method == "minmax":
        lo, hi = v.min(), v.max()
        n = (v - lo) / (hi - lo) if hi > lo else np.ones_like(v)
    elif method == "zscore":
        sd = v.std()
        n = (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
    else:
        raise ValueError(method)
    return dict(zip(scores.keys(), n.tolist()))


def weighted_sum(runs: Sequence[Run], weights: Sequence[float], norm: str = "minmax", depth: int = 100) -> Run:
    out: Run = {}
    qids = set().union(*[r.keys() for r in runs])
    for qid in qids:
        acc: dict[str, float] = {}
        for w, run in zip(weights, runs):
            top = dict(sorted(run.get(qid, {}).items(), key=lambda x: -x[1])[:depth])
            for did, s in _normalize(top, norm).items():
                acc[did] = acc.get(did, 0.0) + w * s
        out[qid] = acc
    return out


def fuse(runs: Sequence[Run], method: str = "rrf", weights: Sequence[float] | None = None, **kw) -> Run:
    if method == "rrf":
        return rrf(runs, weights=weights, **kw)
    if method in ("weighted", "minmax", "zscore"):
        norm = "minmax" if method == "weighted" else method
        return weighted_sum(runs, weights or [1.0] * len(runs), norm=norm, **kw)
    raise ValueError(f"unknown fusion {method}")


def truncate(run: Run, k: int) -> Run:
    return {q: dict(sorted(d.items(), key=lambda x: -x[1])[:k]) for q, d in run.items()}


class FusionRetriever(Retriever):
    """Online hybrid retriever (used by the service); offline benchmarks fuse cached runs instead."""

    def __init__(self, members: Sequence[Retriever], method: str = "rrf", weights: Sequence[float] | None = None,
                 depth: int = 100, name: str = "hybrid", **kw):
        super().__init__()
        self.members, self.method, self.weights, self.depth, self.kw = list(members), method, weights, depth, kw
        self.name = name

    def index(self, ds: RetrievalDataset) -> None:
        self.dataset = ds
        for m in self.members:
            if m.dataset is not ds:
                m.index(ds)

    def search(self, queries: dict[str, Query], k: int = 100) -> Run:
        runs = [m.search(queries, max(k, self.depth)) for m in self.members]
        return truncate(fuse(runs, self.method, self.weights, depth=self.depth, **self.kw), k)
