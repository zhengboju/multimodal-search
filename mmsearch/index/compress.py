"""Multi-vector compression strategies (E3). Each compressor transforms docs (and, if needed, queries).

  kmeans   per-page k-means: replace ~800 patch vectors by k centroids             (token count ↓)
  pca      linear projection 128 -> d fitted on doc tokens, applied to both sides   (dimension ↓)
  prune    keep top-k tokens by query-agnostic salience (random = control)        (token count ↓)
  int8     per-dimension symmetric scalar quantisation of the payload              (bytes/value ↓)
Compose with a list:  [{type: pca, dim: 64}, {type: int8}]
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .kmeans import batched_kmeans
from .store import MultiVectorStore


class Compressor:
    name = "identity"

    def fit(self, store: MultiVectorStore) -> "Compressor":
        return self

    def docs(self, store: MultiVectorStore) -> MultiVectorStore:
        return store

    def query(self, q: torch.Tensor) -> torch.Tensor:
        return q

    def describe(self) -> str:
        return self.name


def _batches(store: MultiVectorStore, bs: int):
    for s in range(0, len(store), bs):
        idx = list(range(s, min(len(store), s + bs)))
        yield idx


class KMeansPool(Compressor):
    def __init__(self, k: int = 64, iters: int = 10, device: str | None = None, batch: int = 64):
        self.k, self.iters, self.batch = k, iters, batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"kmeans{k}"

    def docs(self, store: MultiVectorStore) -> MultiVectorStore:
        out: list[torch.Tensor] = []
        for idx in _batches(store, self.batch):
            X, M = store.padded(self.device, torch.float32, idx)
            C, valid = batched_kmeans(X, M, self.k, self.iters)
            for i in range(len(idx)):
                L = int(M[i].sum())
                out.append(store.get(idx[i]) if L <= self.k else C[i][valid[i]].half().cpu())
        return MultiVectorStore.from_list(out, store.doc_ids)


class PCAReduce(Compressor):
    def __init__(self, dim: int = 64, sample_tokens: int = 500_000, seed: int = 0):
        self.dim, self.sample_tokens, self.seed = dim, sample_tokens, seed
        self.W: torch.Tensor | None = None
        self.mu: torch.Tensor | None = None
        self.name = f"pca{dim}"

    def fit(self, store: MultiVectorStore) -> "PCAReduce":
        X = store.dense()
        g = torch.Generator().manual_seed(self.seed)
        if X.shape[0] > self.sample_tokens:
            X = X[torch.randperm(X.shape[0], generator=g)[: self.sample_tokens]]
        X = X.float()
        self.mu = X.mean(0)
        # principal directions of the (uncentred-for-IP) token cloud; centring only for the covariance estimate
        _, _, V = torch.linalg.svd(X - self.mu, full_matrices=False)
        self.W = V[: self.dim].t().contiguous()  # [D, d]
        return self

    def _proj(self, x: torch.Tensor) -> torch.Tensor:
        assert self.W is not None, "call fit() first"
        return F.normalize(x.float() @ self.W.to(x.device), dim=-1).half()

    def docs(self, store: MultiVectorStore) -> MultiVectorStore:
        return MultiVectorStore.from_list([self._proj(x) for x in store.to_list()], store.doc_ids)

    def query(self, q: torch.Tensor) -> torch.Tensor:
        return self._proj(q).to(q.dtype)


class TokenPrune(Compressor):
    """Query-agnostic token pruning. salience = 1 − cos(token, page mean): keeps tokens that carry
    page-specific content and drops near-duplicate background/white-space patches."""

    def __init__(self, keep: int | float = 0.5, strategy: str = "salience", seed: int = 0):
        self.keep, self.strategy, self.seed = keep, strategy, seed
        self.name = f"prune_{strategy}{keep}"

    def _n_keep(self, L: int) -> int:
        n = int(round(self.keep * L)) if isinstance(self.keep, float) and self.keep <= 1 else int(self.keep)
        return max(1, min(L, n))

    def docs(self, store: MultiVectorStore) -> MultiVectorStore:
        g = torch.Generator().manual_seed(self.seed)
        out = []
        for x in store.to_list():
            n = self._n_keep(x.shape[0])
            if self.strategy == "salience":
                xf = x.float()
                score = 1 - xf @ F.normalize(xf.mean(0), dim=0)
            elif self.strategy == "random":
                score = torch.rand(x.shape[0], generator=g)
            else:
                raise ValueError(self.strategy)
            out.append(x[score.topk(n).indices.sort().values])
        return MultiVectorStore.from_list(out, store.doc_ids)


class Int8Quant(Compressor):
    name = "int8"

    def docs(self, store: MultiVectorStore) -> MultiVectorStore:
        X = store.dense().float()
        scale = (X.abs().amax(0).clamp(min=1e-6) / 127.0)
        q = torch.round(X / scale).clamp(-127, 127).to(torch.int8)
        return MultiVectorStore(q, store.offsets.clone(), store.doc_ids, scale.half())


class Chain(Compressor):
    def __init__(self, parts: Sequence[Compressor]):
        self.parts = list(parts)
        self.name = "+".join(p.name for p in self.parts) or "identity"

    def fit(self, store: MultiVectorStore) -> "Chain":
        # fit sequentially on the progressively-compressed store
        cur = store
        for p in self.parts:
            p.fit(cur)
            cur = p.docs(cur)
        self._fitted_docs = cur
        return self

    def docs(self, store: MultiVectorStore) -> MultiVectorStore:
        cur = store
        for p in self.parts:
            cur = p.docs(cur)
        return cur

    def query(self, q: torch.Tensor) -> torch.Tensor:
        for p in self.parts:
            q = p.query(q)
        return q


def build_compressor(cfg: Any) -> Compressor:
    """cfg: None | {type: ...} | [ {type: ...}, ... ]"""
    if not cfg:
        return Compressor()
    if isinstance(cfg, (list, tuple)):
        return Chain([build_compressor(c) for c in cfg])
    cfg = dict(cfg)
    t = cfg.pop("type")
    cls = {"kmeans": KMeansPool, "pca": PCAReduce, "prune": TokenPrune, "int8": Int8Quant, "identity": Compressor}[t]
    return cls(**cfg)


def compress(store: MultiVectorStore, cfg: Any) -> tuple[MultiVectorStore, Compressor]:
    """Fit + apply; returns the compressed store and the fitted compressor (for query-side transforms)."""
    comp = build_compressor(cfg)
    if isinstance(comp, Chain):
        comp.fit(store)
        return comp._fitted_docs, comp
    comp.fit(store)
    return comp.docs(store), comp
