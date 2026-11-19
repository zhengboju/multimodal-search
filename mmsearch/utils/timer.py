from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class LatencyRecorder:
    """Collects per-request latencies (ms) and summarises P50/P90/P99 + throughput."""

    def __init__(self, cuda_sync: bool = True):
        self.samples: list[float] = []
        self.cuda_sync = cuda_sync
        self._wall_start: float | None = None
        self._wall_end: float | None = None

    @contextmanager
    def time(self) -> Iterator[None]:
        if self.cuda_sync:
            _sync()
        t0 = time.perf_counter()
        if self._wall_start is None:
            self._wall_start = t0
        yield
        if self.cuda_sync:
            _sync()
        t1 = time.perf_counter()
        self._wall_end = t1
        self.samples.append((t1 - t0) * 1000.0)

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    def summary(self, prefix: str = "") -> dict[str, float]:
        if not self.samples:
            return {}
        a = np.asarray(self.samples)
        out = {
            "n": float(len(a)),
            "mean_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)),
            "p90_ms": float(np.percentile(a, 90)),
            "p99_ms": float(np.percentile(a, 99)),
            # sequential (batch=1) throughput implied by mean latency
            "qps_seq": float(1000.0 / a.mean()) if a.mean() > 0 else float("inf"),
        }
        return {f"{prefix}{k}": v for k, v in out.items()}
