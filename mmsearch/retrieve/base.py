"""Retriever interface. A run is {query_id: {doc_id: score}} (BEIR/pytrec_eval format)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from ..data.schema import Query, RetrievalDataset
from ..utils.timer import LatencyRecorder

Run = dict[str, dict[str, float]]


class Retriever(ABC):
    name: str = "retriever"

    def __init__(self) -> None:
        self.build_stats: dict[str, Any] = {}
        self.dataset: RetrievalDataset | None = None

    @abstractmethod
    def index(self, ds: RetrievalDataset) -> None:
        """Build/load the index for `ds.corpus`."""

    @abstractmethod
    def search(self, queries: dict[str, Query], k: int = 100) -> Run:
        """Batched retrieval for many queries."""

    def measure_latency(self, queries: dict[str, Query], k: int = 10, n: int = 100, warmup: int = 5) -> dict[str, float]:
        """End-to-end single-query latency (query encoding + search), batch size 1."""
        pool = list(queries.items())
        items = [pool[i % len(pool)] for i in range(n + warmup)] if pool else []  # cycle small query sets
        rec = LatencyRecorder()
        for i, (qid, q) in enumerate(items):
            if i < warmup:
                self.search({qid: q}, k)
                continue
            with rec.time():
                self.search({qid: q}, k)
        return rec.summary("latency_")

    def stats(self) -> dict[str, Any]:
        return dict(self.build_stats)


def timed(fn):
    def wrap(self, *a, **kw):
        t0 = time.perf_counter()
        out = fn(self, *a, **kw)
        self.build_stats["index_build_s"] = time.perf_counter() - t0
        return out

    return wrap
