"""Multi-vector (ColQwen2 late-interaction) retriever: cached page embeddings -> compression -> index -> MaxSim."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence

import torch

from ..data.schema import Query, RetrievalDataset
from ..index.build_index import build_with_compression
from ..index.store import MultiVectorStore
from ..models.colpali import ColQwen2Encoder, EncoderConfig
from ..utils.misc import get_logger
from ..utils.paths import cache_dir
from ..utils.timer import LatencyRecorder
from .base import Retriever, Run

log = get_logger(__name__)
_ENCODERS: dict[str, ColQwen2Encoder] = {}


def get_encoder(spec: EncoderConfig, device: str | None = None) -> ColQwen2Encoder:
    """Process-wide encoder registry so many index variants share one loaded model."""
    tag = spec.tag()
    if tag not in _ENCODERS:
        log.info("loading encoder %s", tag)
        _ENCODERS[tag] = ColQwen2Encoder(spec, device=device)
    return _ENCODERS[tag]


def _key(items: Sequence[str]) -> str:
    return hashlib.sha1("\n".join(items).encode("utf-8")).hexdigest()[:12]


def encode_corpus_cached(spec: EncoderConfig, ds: RetrievalDataset, encoder_fn: Callable[[], ColQwen2Encoder],
                         batch_size: int = 16) -> MultiVectorStore:
    doc_ids = list(ds.corpus)
    path = cache_dir() / "emb" / spec.tag() / f"{ds.corpus_name}_docs_{_key(doc_ids)}.pt"
    if path.exists():
        return MultiVectorStore.load(path)
    docs = [ds.corpus[d] for d in doc_ids]
    enc = encoder_fn()
    if all(d.image for d in docs):
        embs = enc.encode_images([d.image for d in docs], batch_size=batch_size, desc=f"encode {ds.name}")
    else:  # text-only corpus (e.g. OCR-only pages) — encode as queries-style text
        embs = enc.encode_queries([d.text or "" for d in docs], batch_size=batch_size, desc=f"encode {ds.name}")
    store = MultiVectorStore.from_list(embs, doc_ids)
    store.save(path)
    return store


class QueryEmbeddingCache:
    """Disk-backed {text: embedding} cache per encoder tag (rewrite/translation experiments reuse it heavily)."""

    def __init__(self, spec: EncoderConfig):
        self.dir = cache_dir() / "emb" / spec.tag() / "queries"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mem: dict[str, torch.Tensor] = {}

    def get_many(self, texts: Sequence[str], encode: Callable[[list[str]], list[torch.Tensor]], persist: bool = True) -> list[torch.Tensor]:
        uniq = list(dict.fromkeys(texts))
        path = self.dir / f"{_key(sorted(uniq))}.pt"
        if path.exists() and not all(t in self.mem for t in uniq):
            self.mem.update(torch.load(path))
        missing = [t for t in uniq if t not in self.mem]
        if missing:
            for t, e in zip(missing, encode(missing)):
                self.mem[t] = e
            if persist and len(uniq) > 1:
                torch.save({t: self.mem[t] for t in uniq}, path)
        return [self.mem[t] for t in texts]


class MultiVectorRetriever(Retriever):
    def __init__(self, encoder: dict | str | EncoderConfig, index: dict | None = None, compress: Any = None,
                 name: str | None = None, device: str | None = None, doc_batch_size: int = 16, query_batch_size: int = 64,
                 cache_queries: bool = True):
        super().__init__()
        self.spec = EncoderConfig.from_any(encoder)
        self.index_cfg, self.compress_cfg = index, compress
        self.device = device
        self.doc_batch_size, self.query_batch_size = doc_batch_size, query_batch_size
        self.name = name or f"mv[{self.spec.tag()}]"
        self.qcache = QueryEmbeddingCache(self.spec) if cache_queries else None
        self.search_latency = LatencyRecorder()
        self._index = None
        self._comp = None
        self.doc_ids: list[str] = []

    @property
    def encoder(self) -> ColQwen2Encoder:
        return get_encoder(self.spec, self.device)

    def index(self, ds: RetrievalDataset) -> None:
        self.dataset = ds
        store = encode_corpus_cached(self.spec, ds, lambda: self.encoder, self.doc_batch_size)
        self.doc_ids = store.doc_ids
        self._index, self._comp, stats = build_with_compression(store, self.index_cfg, self.compress_cfg, self.device)
        self.build_stats.update(stats)
        log.info("%s index stats: %s", self.name, stats)

    def encode_queries(self, texts: list[str], use_cache: bool = True) -> list[torch.Tensor]:
        enc = lambda ts: self.encoder.encode_queries(ts, self.query_batch_size)  # noqa: E731
        if self.qcache is not None and use_cache:
            return self.qcache.get_many(texts, enc)
        return enc(texts)

    def search_embeddings(self, qembs: list[torch.Tensor], k: int) -> tuple[list[list[int]], list[list[float]]]:
        qembs = [self._comp.query(q) for q in qembs]
        return self._index.search(qembs, k)

    def search(self, queries: dict[str, Query], k: int = 100) -> Run:
        assert self._index is not None, "call index() first"
        qids = list(queries)
        embs = self.encode_queries([queries[q].text or "" for q in qids], use_cache=len(qids) > 1)
        I, S = self.search_embeddings(embs, k)
        return {qid: {self.doc_ids[i]: s for i, s in zip(ii, ss)} for qid, ii, ss in zip(qids, I, S)}

    def measure_search_latency(self, queries: dict[str, Query], k: int = 10, n: int = 200, warmup: int = 10) -> dict[str, float]:
        """Index-only latency (embeddings precomputed) — isolates the index structure for E3/E4."""
        pool = list(queries)
        uniq = self.encode_queries([queries[q].text or "" for q in pool[: n + warmup]])
        embs = [uniq[i % len(uniq)] for i in range(n + warmup)] if uniq else []
        rec = LatencyRecorder()
        for i, e in enumerate(embs):
            if i < warmup:
                self.search_embeddings([e], k)
                continue
            with rec.time():
                self.search_embeddings([e], k)
        return rec.summary("search_")
