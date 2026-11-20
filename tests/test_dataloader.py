import torch
from torch.utils.data import Dataset

from mmsearch.utils import dataloader as dl


class IdxDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor([i, i * 2])


def test_threaded_batcher_order_and_sizes():
    ds = IdxDataset(23)
    batches = list(dl.ThreadedBatcher(ds, batch_size=8, workers=4, desc=None))
    assert [b.shape[0] for b in batches] == [8, 8, 7]
    flat = torch.cat(batches)
    assert torch.equal(flat[:, 0], torch.arange(23))  # exact order preserved
    assert len(dl.ThreadedBatcher(ds, 8)) == 3


def test_threaded_batcher_collate_and_empty():
    ds = IdxDataset(4)
    out = list(dl.ThreadedBatcher(ds, batch_size=4, collate_fn=lambda items: len(items), workers=2))
    assert out == [4]
    assert list(dl.ThreadedBatcher(IdxDataset(0), batch_size=4)) == []


def test_threaded_batcher_surfaces_dataset_errors():
    class Boom(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            raise ValueError("decode failed")

    try:
        list(dl.ThreadedBatcher(Boom(), batch_size=2, workers=2))
    except ValueError as e:
        assert "decode failed" in str(e)
    else:  # pragma: no cover
        raise AssertionError("exception from a worker thread must propagate")


def test_tune_workers_degrades_on_tiny_shm(monkeypatch):
    monkeypatch.delenv("MMS_NUM_WORKERS", raising=False)
    monkeypatch.setattr(dl, "shm_available_bytes", lambda: 64 << 20)  # Docker default
    assert dl.tune_workers(12) == 0
    monkeypatch.setattr(dl, "shm_available_bytes", lambda: 512 << 20)
    assert 1 <= dl.tune_workers(12) <= 4  # file_system strategy, capped
    monkeypatch.setattr(dl, "shm_available_bytes", lambda: 8 << 30)
    assert dl.tune_workers(12) == 12
    monkeypatch.setattr(dl, "shm_available_bytes", lambda: None)  # not Linux
    assert dl.tune_workers(12) == 12
    monkeypatch.setenv("MMS_NUM_WORKERS", "0")
    assert dl.tune_workers(12) == 0


def test_make_loader_respects_budget(monkeypatch):
    monkeypatch.delenv("MMS_NUM_WORKERS", raising=False)
    monkeypatch.setattr(dl, "shm_available_bytes", lambda: 64 << 20)
    it = dl.make_loader(IdxDataset(10), 4, num_workers=8, shuffle=False)
    assert isinstance(it, dl.ThreadedBatcher)
    assert it.workers == min(8, dl.os.cpu_count() or 1)  # decode threads replace worker processes
    seen = {}
    monkeypatch.setattr(dl, "tune_workers", lambda n, b=0: (seen.setdefault("bytes", b), 0)[1])  # record, then 0 workers
    dl.make_loader(IdxDataset(10), 4, num_workers=8, shuffle=False, batch_bytes=300 << 20)
    assert seen["bytes"] == 300 << 20  # the shm budget check gets the caller's realistic estimate
    assert len(list(it)) == 3
    monkeypatch.setattr(dl, "shm_available_bytes", lambda: 8 << 30)
    it2 = dl.make_loader(IdxDataset(10), 4, num_workers=0, shuffle=False)
    assert type(it2).__name__ == "DataLoader" and len(it2) == 3  # no worker spawn: correctness is covered above
    assert torch.equal(torch.cat(list(it2)), torch.cat([torch.tensor([[i, i * 2]]) for i in range(10)]))


def test_threaded_shuffle_covers_every_item_per_epoch():
    ds = IdxDataset(17)
    b = dl.ThreadedBatcher(ds, batch_size=4, workers=3, shuffle=True, drop_last=True, seed=1)
    assert len(b) == 4  # 17 -> 16 items, drop_last
    epoch_ids = []
    for e in range(3):
        b.set_epoch(e)
        ids = torch.cat(list(b))[:, 0].tolist()
        assert len(ids) == 16 and len(set(ids)) == 16
        epoch_ids.append(ids)
    assert epoch_ids[0] != epoch_ids[1]  # re-permuted every epoch


def test_make_loader_keeps_ddp_sharding_when_falling_back_to_threads(monkeypatch):
    """The old fallback refused threads with a sampler and silently used num_workers=0, which serialised JPEG
    decoding and left the GPUs idle. A sampler must survive the fallback: iterate it, do not ignore it."""
    from torch.utils.data import SequentialSampler

    monkeypatch.delenv("MMS_NUM_WORKERS", raising=False)
    monkeypatch.setattr(dl, "shm_available_bytes", lambda: 64 << 20)
    # "this rank" only gets indices 0..3 of 10 -> must be respected exactly
    it = dl.make_loader(IdxDataset(10), 2, num_workers=4, sampler=SequentialSampler(range(4)))
    assert isinstance(it, dl.ThreadedBatcher)
    seen = torch.cat(list(it))[:, 0].tolist()
    assert seen == [0, 1, 2, 3]


def test_threaded_batcher_sampler_epochs(monkeypatch):
    class Sharded:
        """Stand-in for DistributedSampler: epoch e yields shifted windows of one shared index list."""

        def __init__(self, n, world=2, rank=0):
            self.n, self.world, self.rank, self.epoch = n, world, rank, 0

        def set_epoch(self, e):
            self.epoch = e

        def __iter__(self):
            ids = list(range(self.n))[self.rank :: self.world]
            return iter(ids[self.epoch * 2 :] + ids[: self.epoch * 2])

    b = dl.ThreadedBatcher(IdxDataset(8), batch_size=2, workers=2, sampler=Sharded(8, 2, 1))
    assert torch.cat(list(b))[:, 0].tolist() == [1, 3, 5, 7]
    b.set_epoch(1)  # set_epoch must be forwarded to the sampler
    assert torch.cat(list(b))[:, 0].tolist() == [5, 7, 1, 3]

def test_env_override_sets_count_not_mechanism(monkeypatch):
    """`MMS_NUM_WORKERS=16` on a 52 MB /dev/shm must mean 16 decode THREADS, not 16 worker processes
    (that mistake is what blew the shm up: 'Resource temporarily unavailable')."""
    monkeypatch.delenv("MMS_NUM_WORKERS", raising=False)
    monkeypatch.setattr(dl, "shm_available_bytes", lambda: 52 << 20)
    assert dl.tune_workers(12) == 0  # no room for processes
    monkeypatch.setenv("MMS_NUM_WORKERS", "16")
    assert dl.tune_workers(12) == 0  # the override must NOT resurrect worker processes
    it = dl.make_loader(IdxDataset(8), 4, num_workers=4)
    assert isinstance(it, dl.ThreadedBatcher)


def test_shm_guard_switches_to_threads_mid_epoch(monkeypatch):
    monkeypatch.delenv("MMS_NUM_WORKERS", raising=False)

    class Boom:
        def __init__(self):
            self.calls = 0

        def __len__(self):
            return 3

        def __iter__(self):
            self.calls += 1
            yield torch.tensor([[0, 0]])
            raise RuntimeError("unable to allocate shared memory(shm) for file </torch_x>")

        def set_epoch(self, e):
            pass

    boom = Boom()
    guard = dl.ShmGuardLoader(boom, IdxDataset(8), 4, None, 8, None, True, False, 0)
    out = torch.cat(list(guard))
    assert out.shape[0] == 9  # 1 item yielded before the failure + the 8 items replayed from threads
    assert isinstance(guard._loader, dl.ThreadedBatcher)


def test_shm_guard_does_not_swallow_other_errors():
    class Broken:
        def __init__(self):
            pass

        def __iter__(self):
            raise RuntimeError("CUDA error: device-side assert")

        def __len__(self):
            return 1

    guard = dl.ShmGuardLoader(Broken(), IdxDataset(4), 2, None, 4, None, True, False, 0)
    try:
        list(guard)
    except RuntimeError as e:
        assert "device-side assert" in str(e)
    else:  # pragma: no cover
        raise AssertionError("unrelated errors must propagate")