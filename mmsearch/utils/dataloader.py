"""Data loading helpers that survive containers with a tiny /dev/shm (Docker default: 64 MB).

torch DataLoader workers pass every batch through shared memory. With bs=256 and 224×224 images that is
~154 MB per batch, so `RuntimeError: unable to allocate shared memory(shm)` is guaranteed. Two remedies:

* inference (`ThreadedBatcher`) -> order-preserving prefetch in a **thread pool**: no worker processes, no shm,
  and PIL/libjpeg releases the GIL, so decode throughput stays close to process workers.
* training (`make_loader`) -> `tune_workers()` detects the shm budget; if worker processes cannot work it falls
  back to `ThreadedBatcher` (parallel decode, shuffled per epoch) instead of collapsing to a serial loader.
  The real fix is to start the container with `--shm-size=8g`.

Env `MMS_NUM_WORKERS` forces the value everywhere (0 = no parallelism, N = N workers/threads).
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .misc import get_logger

log = get_logger(__name__)
_DEFAULT_BATCH = 3 * 224 * 224 * 4  # one 224² float32 image ≈ 0.6 MB


def shm_available_bytes() -> int | None:
    try:
        st = os.statvfs("/dev/shm")
        return st.f_bavail * st.f_frsize
    except (AttributeError, OSError):
        return None


def describe_shm() -> str:
    avail = shm_available_bytes()
    return "n/a (not Linux)" if avail is None else f"{avail / 2**20:.0f} MB"


def tune_workers(requested: int, batch_bytes: int = _DEFAULT_BATCH) -> int:
    """Return a num_workers value that will not exhaust /dev/shm."""
    if (env := os.environ.get("MMS_NUM_WORKERS")) is not None:
        return max(0, int(env))
    requested = max(0, int(requested))
    avail = shm_available_bytes()
    if requested == 0 or avail is None or avail >= 1 << 30:  # no workers, not Linux, or plenty of room
        return requested
    if avail >= max(2 * batch_bytes, 256 << 20):  # tight but workable: keep workers, spill via files
        torch.multiprocessing.set_sharing_strategy("file_system")
        capped = min(requested, 4)
        log.warning("/dev/shm is only %s; using %d workers with the 'file_system' sharing strategy", describe_shm(), capped)
        return capped
    log.warning(
        "/dev/shm is only %s (need ~%.0f MB per batch): worker processes disabled, falling back to a thread pool. "
        "Start the container with `--shm-size=8g` for maximum speed, or set MMS_NUM_WORKERS to override.",
        describe_shm(),
        batch_bytes / 2**20,
    )
    return 0


class ThreadedBatcher:
    """Batched iterator backed by a thread pool: no worker processes, therefore **no /dev/shm use**.

    Order-preserving by default; `shuffle=True` re-permutes indices every epoch (call `set_epoch(e)`).
    API-compatible with the parts of DataLoader used in this repo (`len`, `iter`, `dataset`, `set_epoch`).
    """

    def __init__(
        self,
        dataset: Sequence[Any],
        batch_size: int = 8,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        workers: int = 8,
        desc: str | None = None,
        lookahead: int = 2,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if (env := os.environ.get("MMS_NUM_WORKERS")) is not None:
            workers = int(env)
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.collate_fn = collate_fn or torch.utils.data.default_collate
        self.workers = max(1, min(int(workers), 32))
        self.desc = desc
        self.lookahead = max(1, int(lookahead))
        self.shuffle, self.drop_last, self.seed = shuffle, drop_last, int(seed)
        self.epoch = 0
        self._n = len(dataset)

    # ------------------------------------------------------------------ shims
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        n = self._n
        if self.drop_last:
            n -= n % self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, i: int):
        return self.dataset[i]

    def _indices(self) -> list[int]:
        n = self._n
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            idx = torch.randperm(n, generator=g).tolist()
        else:
            idx = list(range(n))
        if self.drop_last:
            idx = idx[: n - n % self.batch_size]
        return idx

    # ------------------------------------------------------------------ iteration
    def __iter__(self) -> Iterator[Any]:
        idx = self._indices()
        n = len(idx)
        if n == 0:
            return
        bs = self.batch_size
        pool = ThreadPoolExecutor(max_workers=self.workers)
        window: deque = deque()
        nxt = 0

        def fill() -> None:
            nonlocal nxt
            while nxt < n and len(window) < bs * self.lookahead:
                window.append(pool.submit(self.dataset.__getitem__, idx[nxt]))
                nxt += 1

        try:
            fill()
            for _ in tqdm(range((n + bs - 1) // bs), desc=self.desc, disable=self.desc is None):
                take = min(bs, n - (nxt - len(window)))  # items not yet consumed nor still in flight
                items = [window.popleft().result() for _ in range(take)]
                fill()
                yield self.collate_fn(items)
        finally:
            for f in window:
                f.cancel()
            pool.shutdown(wait=True)


def make_loader(
    dataset: Sequence[Any],
    batch_size: int,
    *,
    collate_fn: Callable | None = None,
    num_workers: int = 12,
    shuffle: bool = False,
    sampler: Any = None,
    drop_last: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    seed: int = 0,
    threaded_below_workers: int | None = None,
) -> Any:
    """DataLoader / ThreadedBatcher factory for the training scripts.

    A `sampler` (DDP sharding) forces a real DataLoader — sharding cannot be emulated with threads.
    """
    workers = tune_workers(num_workers)
    if threaded_below_workers is None:
        switch = workers == 0 and num_workers > 0 and sampler is None
    else:
        switch = workers <= threaded_below_workers and sampler is None
    if switch:
        threads = max(1, min(num_workers, os.cpu_count() or 1))
        return ThreadedBatcher(dataset, batch_size, collate_fn, workers=threads, shuffle=shuffle,
                               drop_last=drop_last, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=prefetch_factor if workers > 0 else None,
    )
