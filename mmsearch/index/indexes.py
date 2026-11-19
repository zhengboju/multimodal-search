"""Multi-vector indexes (E3/E4). All return (doc indices, scores) per query, best first.

  ExhaustiveIndex    exact MaxSim over every page (length-bucketed padding, GPU)        — quality reference
  PLAIDLiteIndex     centroid codes + inverted lists -> centroid-interaction approx score
                     -> exact MaxSim rerank of the top candidates (PLAID, minus residual compression)
  HierarchicalIndex  page-level coarse vectors (k-means pooled, e.g. 8/page) -> exact MaxSim on top-N
                     ("页级 + 块级两级索引，先粗后细")
  FaissTokenIndex    (faiss_index.py) token ANN (IVF/HNSW/PQ) candidate generation + exact rerank (ColBERTv1 style)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import torch

from ..models.losses import maxsim_scores
from ..models.colpali import pad_multivectors
from .compress import KMeansPool
from .kmeans import assign_nearest, kmeans
from .store import MultiVectorStore


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _compute_dtype(device: str | torch.device) -> torch.dtype:
    return torch.float16 if str(device).startswith("cuda") else torch.float32


class MultiVectorIndex(ABC):
    name = "base"

    @abstractmethod
    def search(self, queries: Sequence[torch.Tensor], k: int) -> tuple[list[list[int]], list[list[float]]]: ...

    @abstractmethod
    def nbytes(self) -> int: ...

    def stats(self) -> dict:
        return {"index": self.name, "index_mb": self.nbytes() / 2**20}


class ExactScorer:
    """Holds the full store as length-bucketed padded blocks on device; scores all docs or a candidate subset."""

    def __init__(self, store: MultiVectorStore, device: str | None = None, bucket_size: int = 512):
        self.store = store
        self.device = device or _default_device()
        self.dtype = _compute_dtype(self.device)
        lengths = store.lengths
        order = torch.argsort(lengths)
        self.buckets: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for s in range(0, len(store), bucket_size):
            idx = order[s : s + bucket_size]
            pad, mask = store.padded(self.device, self.dtype, idx)
            self.buckets.append((pad, mask, idx.to(self.device)))
        # random-access view for candidate reranking: doc -> (bucket, row)
        self.loc = torch.empty(len(store), 2, dtype=torch.long)
        for b, (_, _, idx) in enumerate(self.buckets):
            self.loc[idx.cpu(), 0] = b
            self.loc[idx.cpu(), 1] = torch.arange(len(idx))

    def _q(self, queries: Sequence[torch.Tensor]):
        return pad_multivectors(queries, self.device, self.dtype)

    def score_all(self, queries: Sequence[torch.Tensor]) -> torch.Tensor:
        q, qm = self._q(queries)
        out = torch.empty(len(queries), len(self.store), device=self.device)
        for pad, mask, idx in self.buckets:
            out[:, idx] = maxsim_scores(q, qm, pad, mask, doc_chunk=128)
        return out

    def score_candidates(self, query: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Exact scores of one query against candidate doc indices (1-D LongTensor)."""
        q, qm = self._q([query])
        cand_cpu = cand.cpu()
        scores = torch.empty(len(cand_cpu), device=self.device)
        locs = self.loc[cand_cpu]
        for b in locs[:, 0].unique().tolist():
            sel = (locs[:, 0] == b).nonzero(as_tuple=True)[0]
            pad, mask, _ = self.buckets[b]
            rows = locs[sel, 1].to(self.device)
            scores[sel.to(self.device)] = maxsim_scores(q, qm, pad[rows], mask[rows], doc_chunk=256)[0]
        return scores


def _topk(scores: torch.Tensor, k: int, ids: torch.Tensor | None = None) -> tuple[list[int], list[float]]:
    k = min(k, scores.numel())
    v, i = scores.topk(k)
    if ids is not None:
        i = ids[i]
    return i.tolist(), v.float().tolist()


class ExhaustiveIndex(MultiVectorIndex):
    name = "exhaustive"

    def __init__(self, store: MultiVectorStore, device: str | None = None, query_batch: int = 32):
        self.store = store
        self.scorer = ExactScorer(store, device)
        self.query_batch = query_batch

    def search(self, queries, k):
        I, S = [], []
        for s in range(0, len(queries), self.query_batch):
            sc = self.scorer.score_all(queries[s : s + self.query_batch])
            for row in sc:
                i, v = _topk(row, k)
                I.append(i)
                S.append(v)
        return I, S

    def nbytes(self) -> int:
        return self.store.nbytes()


class PLAIDLiteIndex(MultiVectorIndex):
    def __init__(self, store: MultiVectorStore, n_centroids: int | None = None, nprobe: int = 4, n_rerank: int = 128,
                 exact_rerank: bool = True, kmeans_iters: int = 10, sample_tokens: int = 400_000,
                 device: str | None = None, seed: int = 0):
        self.store = store
        self.device = device or _default_device()
        self.nprobe, self.n_rerank, self.exact_rerank = nprobe, n_rerank, exact_rerank
        T = store.num_tokens
        if n_centroids is None:  # PLAID heuristic: 2^floor(log2(16*sqrt(T)))
            n_centroids = 2 ** int(math.floor(math.log2(16 * math.sqrt(T))))
        X = store.dense()
        g = torch.Generator().manual_seed(seed)
        sample = X[torch.randperm(T, generator=g)[: min(T, sample_tokens)]].to(self.device).float()
        self.centroids = kmeans(sample, n_centroids, kmeans_iters, seed).half()  # [C, D]
        self.n_centroids = self.centroids.shape[0]
        codes = assign_nearest(X.to(self.device).to(_compute_dtype(self.device)), self.centroids.to(_compute_dtype(self.device)))
        self.codes_dtype = torch.int16 if self.n_centroids <= 32767 else torch.int32
        # per-doc padded code matrix (pad = -1) for centroid interaction
        lens = store.lengths
        Lmax = int(lens.max())
        self.doc_codes = torch.full((len(store), Lmax), -1, dtype=torch.long, device=self.device)
        doc_of_tok = torch.repeat_interleave(torch.arange(len(store)), lens).to(self.device)
        pos = (torch.arange(T, device=self.device) - store.offsets[:-1].to(self.device)[doc_of_tok])
        self.doc_codes[doc_of_tok, pos] = codes
        # inverted lists: centroid -> unique docs (CSR)
        pairs = torch.unique(codes * len(store) + doc_of_tok)
        c_of = pairs // len(store)
        self.ivf_docs = pairs % len(store)
        counts = torch.bincount(c_of, minlength=self.n_centroids)
        self.ivf_off = torch.zeros(self.n_centroids + 1, dtype=torch.long, device=self.device)
        self.ivf_off[1:] = torch.cumsum(counts, 0)
        self.scorer = ExactScorer(store, self.device) if exact_rerank else None
        self.name = f"plaid_c{self.n_centroids}_p{nprobe}_r{n_rerank}" + ("" if exact_rerank else "_approx")

    cand_chunk = 2048

    def _centroid_interaction(self, S: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Approximate MaxSim using each doc token's centroid instead of the token itself. S [Lq, C] -> [n]."""
        codes = self.doc_codes[cand]  # [n, Lmax]
        approx = S[:, codes.clamp(min=0)]  # [Lq, n, Lmax]
        return approx.masked_fill((codes < 0)[None], float("-inf")).amax(-1).sum(0)

    def _candidates(self, S: torch.Tensor) -> torch.Tensor:
        probe = S.topk(min(self.nprobe, S.shape[1]), dim=1).indices.unique()
        starts, ends = self.ivf_off[probe], self.ivf_off[probe + 1]
        parts = [self.ivf_docs[s:e] for s, e in zip(starts.tolist(), ends.tolist())]
        return torch.unique(torch.cat(parts)) if parts else torch.empty(0, dtype=torch.long, device=self.device)

    def search(self, queries, k):
        I, Sc = [], []
        C = self.centroids.to(_compute_dtype(self.device))
        for q in queries:
            qd = q.to(self.device, C.dtype)
            S = (qd @ C.t()).float()  # [Lq, C]
            cand = self._candidates(S)
            if cand.numel() == 0:
                I.append([]); Sc.append([])  # noqa: E702
                continue
            approx = torch.cat([self._centroid_interaction(S, cand[s : s + self.cand_chunk])
                                for s in range(0, cand.numel(), self.cand_chunk)])
            if not self.exact_rerank:
                i, v = _topk(approx, k, cand)
            else:
                top = approx.topk(min(self.n_rerank, approx.numel())).indices
                cand2 = cand[top]
                i, v = _topk(self.scorer.score_candidates(q, cand2), k, cand2)
            I.append(i)
            Sc.append(v)
        return I, Sc

    def nbytes(self) -> int:
        """Compressed index = centroids + per-token codes + inverted lists. Exact rerank additionally needs the
        full-precision store (reported separately in stats())."""
        return int(self.centroids.numel() * 2 + self.store.num_tokens * torch.tensor([], dtype=self.codes_dtype).element_size()
                   + self.ivf_docs.numel() * 4 + self.ivf_off.numel() * 4)

    def stats(self) -> dict:
        s = super().stats()
        s["rerank_store_mb"] = self.store.nbytes() / 2**20 if self.exact_rerank else 0.0
        s["total_mb"] = s["index_mb"] + s["rerank_store_mb"]
        return s


class HierarchicalIndex(MultiVectorIndex):
    def __init__(self, store: MultiVectorStore, coarse_k: int = 8, n_candidates: int = 100, device: str | None = None,
                 coarse_store: MultiVectorStore | None = None):
        self.device = device or _default_device()
        self.store = store
        self.coarse_store = coarse_store or KMeansPool(coarse_k, device=self.device).docs(store)
        self.coarse = ExactScorer(self.coarse_store, self.device)
        self.fine = ExactScorer(store, self.device)
        self.n_candidates = n_candidates
        self.name = f"hier_k{coarse_k}_n{n_candidates}"

    def search(self, queries, k):
        I, S = [], []
        coarse_scores = self.coarse.score_all(queries)
        for q, row in zip(queries, coarse_scores):
            cand = row.topk(min(self.n_candidates, row.numel())).indices
            i, v = _topk(self.fine.score_candidates(q, cand), k, cand)
            I.append(i)
            S.append(v)
        return I, S

    def nbytes(self) -> int:
        return self.coarse_store.nbytes() + self.store.nbytes()

    def stats(self) -> dict:
        return {"index": self.name, "index_mb": self.nbytes() / 2**20, "coarse_mb": self.coarse_store.nbytes() / 2**20}
