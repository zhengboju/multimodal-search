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
    """Decide how many *worker processes* to use (0 => the caller falls back to a thread pool).

    `MMS_NUM_WORKERS` overrides the parallelism **count** only — never the mechanism. It is the number of
    decoders the user wants (threads in the fallback, processes when /dev/shm allows); letting it force
    processes is how a well-meant `MMS_NUM_WORKERS=16` blew up a 52 MB /dev/shm.
    """
    if (env := os.environ.get("MMS_NUM_WORKERS")) is not None:
        requested = int(env)
    requested = max(0, int(requested))
    avail = shm_available_bytes()
    if requested == 0 or avail is None or avail >= 1 << 30:  # no parallelism, not Linux, or plenty of room
        return requested
    if avail >= max(4 * batch_bytes, 256 << 20):  # tight but workable: keep workers, spill via files
        torch.multiprocessing.set_sharing_strategy("file_system")
        capped = min(requested, 4)
        log.warning("/dev/shm is only %s; using %d workers with the 'file_system' sharing strategy", describe_shm(), capped)
        return capped
    log.warning(
        "/dev/shm is only %s but one batch needs ~%.0f MB (worker processes need ~4x that): using %d decode "
        "THREADS instead of processes — same parallelism, no /dev/shm use. Permanent fix: `--shm-size=8g` at "
        "container start, or `mount -o remount,size=8G /dev/shm` as root.",
        describe_shm(),
        batch_bytes / 2**20,
        requested or 8,
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
        sampler: Any = None,
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
        self.sampler = sampler  # e.g. DistributedSampler: __iter__ gives this rank's shard for the epoch
        self.epoch = 0
        self._n = len(dataset)

    # ------------------------------------------------------------------ shims
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self.epoch)

    def __len__(self) -> int:
        n = self._n
        if self.drop_last:
            n -= n % self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, i: int):
        return self.dataset[i]

    def _indices(self) -> list[int]:
        n = self._n
        if self.sampler is not None:
            idx = [int(i) for i in self.sampler]
            if self.drop_last:
                idx = idx[: len(idx) - len(idx) % self.batch_size]
            return idx
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


class ShmGuardLoader:
    """Last line of defence around a multi-process DataLoader.

    The /dev/shm budget is an estimate (image sizes, prefetch depth, other processes). If worker processes do
    run out anyway, torch raises `RuntimeError: unable to allocate shared memory` mid-epoch; instead of killing
    a training run hours in, switch this loader to decode threads and continue from the next batch.
    """

    def __init__(self, loader: Any, dataset: Sequence[Any], batch_size: int, collate_fn: Callable | None,
                 num_workers: int, sampler: Any, drop_last: bool, shuffle: bool, seed: int):
        self._loader = loader
        self._args = dict(batch_size=batch_size, collate_fn=collate_fn, workers=max(1, num_workers),
                          sampler=sampler, drop_last=drop_last, shuffle=shuffle, seed=seed)
        self._dataset = dataset
        self.dataset = dataset
        self._epoch = 0

    def __len__(self) -> int:
        return len(self._loader)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if hasattr(self._loader, "set_epoch"):
            self._loader.set_epoch(self._epoch)

    def __iter__(self) -> Iterator[Any]:
        try:
            for batch in self._loader:
                yield batch
            return
        except RuntimeError as e:  # noqa: PERF203 - only raised once, on a dead worker pool
            if "shared memory" not in str(e).lower():
                raise
            log.warning("worker processes ran out of /dev/shm mid-epoch (%s); continuing with decode threads", e)
        self._loader = ThreadedBatcher(self._dataset, **self._args)
        self._loader.set_epoch(self._epoch)
        yield from self._loader


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
    batch_bytes: int | None = None,
) -> Any:
    """DataLoader / ThreadedBatcher factory for the training scripts.

    `batch_bytes` is the estimated size of one collated batch (defaults to `batch_size` 224² float32 images);
    it only feeds the /dev/shm budget check, so pass a realistic value for big pages.
    A `sampler` (e.g. DistributedSampler) is honoured by both paths: ThreadedBatcher simply iterates it, so
    DDP keeps its per-rank shard without any worker process (this is what keeps the GPU fed when /dev/shm is tiny).
    """
    if batch_bytes is None:
        batch_bytes = max(1, batch_size) * _DEFAULT_BATCH
    workers = tune_workers(num_workers, batch_bytes)
    if threaded_below_workers is None:
        switch = workers == 0 and num_workers > 0
    else:
        switch = workers <= threaded_below_workers
    if switch:
        # share the cores between DDP ranks: N ranks x M threads must still fit in the CPU budget
        world = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", 1) or 1))
        threads = max(1, min(num_workers or 8, (os.cpu_count() or 8) // world))
        log.info("data pipeline: %d decode threads, lookahead 2 batches (no worker processes -> no /dev/shm use)", threads)
        return ThreadedBatcher(dataset, batch_size, collate_fn, workers=threads, shuffle=shuffle,
                               drop_last=drop_last, seed=seed, sampler=sampler)
    loader = DataLoader(
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
    if workers > 0:
        return ShmGuardLoader(loader, dataset, batch_size, collate_fn, num_workers, sampler, drop_last, shuffle, seed)
    return loader
