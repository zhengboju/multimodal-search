"""FAISS token-level ANN index for multi-vector retrieval (ColBERTv1-style candidate generation + exact rerank).

Optional dependency: `pip install faiss-cpu` (or faiss-gpu-cu12). factory examples: "IVF1024,Flat", "IVF4096,PQ32x8", "HNSW32".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .indexes import ExactScorer, MultiVectorIndex, _topk
from .store import MultiVectorStore


class FaissTokenIndex(MultiVectorIndex):
    def __init__(self, store: MultiVectorStore, factory: str = "IVF1024,Flat", nprobe: int = 16, k_tokens: int = 64,
                 n_rerank: int = 128, exact_rerank: bool = True, train_tokens: int = 300_000, use_gpu: bool = False,
                 device: str | None = None, seed: int = 0):
        try:
            import faiss
        except ImportError as e:  # pragma: no cover
            raise ImportError("FaissTokenIndex needs faiss: pip install faiss-cpu") from e
        self.faiss = faiss
        self.store = store
        X = store.dense().float().numpy()
        index = faiss.index_factory(store.dim, factory, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            rng = np.random.default_rng(seed)
            sample = X[rng.choice(len(X), size=min(len(X), train_tokens), replace=False)]
            index.train(sample)
        index.add(X)
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe
        elif "IVF" in factory:
            faiss.extract_index_ivf(index).nprobe = nprobe
        if "HNSW" in factory:
            faiss.downcast_index(index).hnsw.efSearch = max(64, k_tokens * 2)
        self.cpu_index = index
        self.index = index
        if use_gpu and hasattr(faiss, "StandardGpuResources"):
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        self.tok2doc = torch.repeat_interleave(torch.arange(len(store)), store.lengths).numpy()
        self.k_tokens, self.n_rerank, self.exact_rerank = k_tokens, n_rerank, exact_rerank
        self.scorer = ExactScorer(store, device) if exact_rerank else None
        self.name = f"faiss[{factory}]_p{nprobe}_kt{k_tokens}_r{n_rerank}" + ("" if exact_rerank else "_approx")

    def search(self, queries: Sequence[torch.Tensor], k: int):
        I, S = [], []
        for q in queries:
            qn = q.float().numpy()
            sims, toks = self.index.search(qn, self.k_tokens)  # [Lq, kt]
            valid = toks >= 0
            docs = np.where(valid, self.tok2doc[np.clip(toks, 0, None)], -1)
            # approximate MaxSim restricted to retrieved tokens: per (query token, doc) keep best sim
            cand = np.unique(docs[valid])
            if cand.size == 0:
                I.append([]); S.append([])  # noqa: E702
                continue
            pos = {d: j for j, d in enumerate(cand.tolist())}
            approx = np.full((qn.shape[0], cand.size), 0.0, dtype=np.float32)
            for i in range(qn.shape[0]):
                for s, d in zip(sims[i][valid[i]], docs[i][valid[i]]):
                    j = pos[int(d)]
                    if s > approx[i, j]:
                        approx[i, j] = s
            approx_t = torch.from_numpy(approx.sum(0))
            cand_t = torch.from_numpy(cand)
            if not self.exact_rerank:
                i_, v_ = _topk(approx_t, k, cand_t)
            else:
                top = cand_t[approx_t.topk(min(self.n_rerank, cand.size)).indices]
                sc = self.scorer.score_candidates(q, top.to(self.scorer.device))
                i_, v_ = _topk(sc.cpu(), k, top)
            I.append(i_)
            S.append(v_)
        return I, S

    def nbytes(self) -> int:
        return int(self.faiss.serialize_index(self.cpu_index).nbytes)

    def stats(self) -> dict:
        s = super().stats()
        s["rerank_store_mb"] = self.store.nbytes() / 2**20 if self.exact_rerank else 0.0
        s["total_mb"] = s["index_mb"] + s["rerank_store_mb"]
        return s
