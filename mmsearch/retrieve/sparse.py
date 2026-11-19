"""BM25 over page text (OCR via data/ocr.py, or native text). Multilingual-safe tokenizer:
latin words + CJK character uni/bi-grams."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import scipy.sparse as sp

from ..data.schema import Query, RetrievalDataset
from ..utils.misc import get_logger
from .base import Retriever, Run, timed

log = get_logger(__name__)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
STOPWORDS = set(
    "a an the of and or to in on for with by at from as is are was were be been being this that these those it its "
    "what which who whom whose when where why how do does did can could should would will shall may might must "
    "de la le les des du et en un une est que qui pour dans sur par au aux".split()
)


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for w in _WORD.findall(text.lower()):
        if _CJK.search(w):
            chars = [c for c in w]
            toks.extend(chars)
            toks.extend(a + b for a, b in zip(chars, chars[1:]))
        elif w not in STOPWORDS:
            toks.append(w)
    return toks


class BM25Retriever(Retriever):
    def __init__(self, k1: float = 1.2, b: float = 0.75, name: str = "bm25"):
        super().__init__()
        self.k1, self.b, self.name = k1, b, name
        self.vocab: dict[str, int] = {}
        self.W: sp.csr_matrix | None = None
        self.doc_ids: list[str] = []

    @timed
    def index(self, ds: RetrievalDataset) -> None:
        self.dataset = ds
        self.doc_ids = list(ds.corpus)
        rows, cols, vals, dls = [], [], [], []
        missing = 0
        for r, did in enumerate(self.doc_ids):
            text = ds.corpus[did].text or ""
            missing += not text
            tf = Counter(tokenize(text))
            dls.append(sum(tf.values()))
            for t, c in tf.items():
                rows.append(r)
                cols.append(self.vocab.setdefault(t, len(self.vocab)))
                vals.append(c)
        if missing == len(self.doc_ids):
            raise ValueError(f"{ds.name}: no document text — run `python -m mmsearch.data.ocr --datasets {ds.corpus_name}` first")
        if missing:
            log.warning("%s: %d/%d docs have no text (run `python -m mmsearch.data.ocr`)", ds.name, missing, len(self.doc_ids))
        N, V = len(self.doc_ids), max(1, len(self.vocab))
        tf = sp.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(N, V))
        df = np.bincount(np.asarray(cols, np.int64), minlength=V) if cols else np.zeros(V)
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(dls, np.float32)
        avgdl = max(1e-6, float(dl.mean())) if N else 1.0
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avgdl)
        w = idf[tf.col] * tf.data * (self.k1 + 1) / denom
        self.W = sp.csr_matrix((w, (tf.row, tf.col)), shape=(N, V)).tocsc()
        self.build_stats.update({"docs": N, "vocab": V, "index_mb": (self.W.data.nbytes + self.W.indices.nbytes) / 2**20})

    def search(self, queries: dict[str, Query], k: int = 100) -> Run:
        assert self.W is not None, "call index() first"
        run: Run = {}
        for qid, q in queries.items():
            terms = Counter(t for t in tokenize(q.text or "") if t in self.vocab)
            if not terms:
                run[qid] = {}
                continue
            cols = [self.vocab[t] for t in terms]
            weights = np.asarray(list(terms.values()), np.float32)
            scores = np.asarray(self.W[:, cols] @ weights).ravel()
            kk = min(k, len(scores))
            top = np.argpartition(-scores, kk - 1)[:kk]
            top = top[np.argsort(-scores[top])]
            run[qid] = {self.doc_ids[i]: float(scores[i]) for i in top if scores[i] > 0}
        return run
