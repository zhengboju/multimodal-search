"""Compact storage for multi-vector (late-interaction) document embeddings.

All token vectors are concatenated into one [T, D] tensor with CSR-style offsets [N+1].
Supports fp16 (default) or int8 payloads (per-dimension symmetric scale) for compression experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import torch


class MultiVectorStore:
    def __init__(self, data: torch.Tensor, offsets: torch.Tensor, doc_ids: Sequence[str], scale: torch.Tensor | None = None):
        assert offsets.numel() == len(doc_ids) + 1, "offsets must have N+1 entries"
        self.data = data
        self.offsets = offsets.long()
        self.doc_ids = list(doc_ids)
        self.scale = scale  # set iff data is int8

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_list(cls, items: Sequence[torch.Tensor], doc_ids: Sequence[str]) -> "MultiVectorStore":
        assert len(items) == len(doc_ids)
        lengths = torch.tensor([int(t.shape[0]) for t in items], dtype=torch.long)
        offsets = torch.zeros(len(items) + 1, dtype=torch.long)
        offsets[1:] = torch.cumsum(lengths, 0)
        data = torch.cat([t.to(torch.float16).cpu() for t in items]) if items else torch.empty(0, 0, dtype=torch.float16)
        return cls(data, offsets, doc_ids)

    # ---------------------------------------------------------------- views
    def __len__(self) -> int:
        return len(self.doc_ids)

    @property
    def dim(self) -> int:
        return int(self.data.shape[1])

    @property
    def num_tokens(self) -> int:
        return int(self.data.shape[0])

    @property
    def lengths(self) -> torch.Tensor:
        return self.offsets[1:] - self.offsets[:-1]

    def dense(self) -> torch.Tensor:
        """Full payload as fp16 (dequantised if int8)."""
        if self.scale is None:
            return self.data
        return (self.data.float() * self.scale.float()).to(torch.float16)

    def get(self, i: int) -> torch.Tensor:
        s, e = int(self.offsets[i]), int(self.offsets[i + 1])
        x = self.data[s:e]
        return x if self.scale is None else (x.float() * self.scale.float()).to(torch.float16)

    def to_list(self) -> list[torch.Tensor]:
        return [self.get(i) for i in range(len(self))]

    def nbytes(self) -> int:
        n = self.data.numel() * self.data.element_size()
        if self.scale is not None:
            n += self.scale.numel() * self.scale.element_size()
        return int(n + self.offsets.numel() * 4)  # offsets as int32 on disk-equivalent

    def padded(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float16,
               idx: Sequence[int] | torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """[n, Lmax, D] padded tensor + bool mask for docs `idx` (all if None)."""
        ids = list(range(len(self))) if idx is None else [int(i) for i in idx]
        lens = [int(self.offsets[i + 1] - self.offsets[i]) for i in ids]
        Lmax = max(lens) if lens else 0
        out = torch.zeros(len(ids), Lmax, self.dim, dtype=dtype, device=device)
        mask = torch.zeros(len(ids), Lmax, dtype=torch.bool, device=device)
        for r, (i, L) in enumerate(zip(ids, lens)):
            out[r, :L] = self.get(i).to(device=device, dtype=dtype)
            mask[r, :L] = True
        return out, mask

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "MultiVectorStore":
        return MultiVectorStore.from_list([fn(x) for x in self.to_list()], self.doc_ids)

    def subset(self, idx: Sequence[int]) -> "MultiVectorStore":
        return MultiVectorStore.from_list([self.get(i) for i in idx], [self.doc_ids[i] for i in idx])

    # ---------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"data": self.data, "offsets": self.offsets, "doc_ids": self.doc_ids, "scale": self.scale}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MultiVectorStore":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return cls(obj["data"], obj["offsets"], obj["doc_ids"], obj.get("scale"))

    def stats(self) -> dict:
        L = self.lengths.float()
        return {
            "docs": len(self),
            "tokens": self.num_tokens,
            "dim": self.dim,
            "avg_tokens_per_doc": float(L.mean()) if len(self) else 0.0,
            "payload_dtype": str(self.data.dtype).replace("torch.", ""),
            "size_mb": self.nbytes() / 2**20,
        }
