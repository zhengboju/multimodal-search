"""Single-vector dense retrieval with a CLIP dual encoder (text query -> image/text doc)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from ..data.schema import Query, RetrievalDataset
from ..models.clip_wrapper import CLIPWrapper
from ..utils.misc import get_logger
from ..utils.paths import cache_dir
from .base import Retriever, Run, timed

log = get_logger(__name__)
_CLIP: dict[str, CLIPWrapper] = {}


def get_clip(model: str, device: str | None = None) -> CLIPWrapper:
    if model not in _CLIP:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _CLIP[model] = CLIPWrapper(model, dtype="fp16" if dev.startswith("cuda") else None).to(dev).eval()
    return _CLIP[model]


def _tag(model: str) -> str:
    return Path(model).name if Path(model).exists() else model.replace("/", "__")


class DenseRetriever(Retriever):
    def __init__(self, model: str = "openai/clip-vit-large-patch14", name: str | None = None, device: str | None = None,
                 batch_size: int = 256):
        super().__init__()
        self.model_name, self.device, self.batch_size = model, device, batch_size
        self.name = name or f"dense[{_tag(model)}]"
        self.doc_ids: list[str] = []
        self.doc_emb: torch.Tensor | None = None

    @property
    def encoder(self) -> CLIPWrapper:
        return get_clip(self.model_name, self.device)

    @timed
    def index(self, ds: RetrievalDataset) -> None:
        self.dataset = ds
        self.doc_ids = list(ds.corpus)
        key = hashlib.sha1("\n".join(self.doc_ids).encode()).hexdigest()[:10]
        path = cache_dir() / "emb" / _tag(self.model_name) / f"{ds.corpus_name}_{key}.pt"
        if path.exists():
            self.doc_emb = torch.load(path)
        else:
            docs = [ds.corpus[d] for d in self.doc_ids]
            if all(d.image for d in docs):
                self.doc_emb = self.encoder.encode_images([d.image for d in docs], self.batch_size)
            else:
                self.doc_emb = self.encoder.encode_texts([d.text or "" for d in docs], self.batch_size)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.doc_emb, path)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.doc_emb = self.doc_emb.to(dev, torch.float16 if dev.startswith("cuda") else torch.float32)
        self.build_stats.update({"docs": len(self.doc_ids), "index_mb": self.doc_emb.numel() * 2 / 2**20})

    def search(self, queries: dict[str, Query], k: int = 100) -> Run:
        assert self.doc_emb is not None
        qids = list(queries)
        qs = [queries[q] for q in qids]
        if all(q.image for q in qs):
            qe = self.encoder.encode_images([q.image for q in qs], self.batch_size, desc=None)
        else:
            qe = self.encoder.encode_texts([q.text or "" for q in qs], self.batch_size, desc=None)
        scores = qe.to(self.doc_emb.device, self.doc_emb.dtype) @ self.doc_emb.t()
        v, i = scores.float().topk(min(k, scores.shape[1]), dim=1)
        return {qid: {self.doc_ids[j]: float(s) for j, s in zip(ii.tolist(), vv.tolist())} for qid, ii, vv in zip(qids, i, v)}
