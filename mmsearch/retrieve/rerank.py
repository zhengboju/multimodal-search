"""VLM pointwise reranker: P("Yes" | page image, query) from first-token logprobs of a vLLM-served VLM.

An order of magnitude slower than first-stage retrieval (one VLM prefill per (query, page) pair), so it
only rescoring the top-n. Latency/quality trade-off is measured in the benchmark (plan §阶段3).
"""

from __future__ import annotations

from ..data.schema import Query, RetrievalDataset
from ..llm.client import VLMClient, user_message, yes_probability
from .base import Retriever, Run
from .fusion import truncate

RERANK_PROMPT = (
    "Query: {query}\n"
    "Does this document page contain the information needed to answer the query? Answer only Yes or No."
)


class VLMReranker:
    def __init__(self, client: VLMClient | None = None, top_n: int = 10, max_pixels: int = 768 * 28 * 28,
                 interpolate: float = 0.0):
        self.client = client or VLMClient()
        self.top_n, self.max_pixels, self.interpolate = top_n, max_pixels, interpolate

    def _messages(self, query: str, image: str) -> list[dict]:
        return [user_message(RERANK_PROMPT.format(query=query), [image], self.max_pixels)]

    def rerank_run(self, run: Run, queries: dict[str, Query], ds: RetrievalDataset) -> Run:
        pairs: list[tuple[str, str]] = []
        for qid, docs in run.items():
            for did, _ in sorted(docs.items(), key=lambda x: -x[1])[: self.top_n]:
                pairs.append((qid, did))
        msgs = [self._messages(queries[q].text or "", ds.corpus[d].image) for q, d in pairs]
        res = self.client.chat_many(msgs, desc="vlm rerank", max_tokens=1, logprobs=True, top_logprobs=10)
        out: Run = {qid: {} for qid in run}
        for (qid, did), r in zip(pairs, res):
            p = yes_probability(r)
            out[qid][did] = p
        # keep tail (beyond top_n) below every reranked doc, preserving first-stage order
        for qid, docs in run.items():
            tail = sorted(docs.items(), key=lambda x: -x[1])[self.top_n :]
            for rank, (did, _) in enumerate(tail):
                out[qid][did] = -1.0 - rank
            if self.interpolate > 0:  # optional blend with first-stage rank prior
                head = sorted(docs.items(), key=lambda x: -x[1])[: self.top_n]
                for rank, (did, _) in enumerate(head):
                    out[qid][did] = (1 - self.interpolate) * out[qid][did] + self.interpolate / (rank + 1)
        return out


class RerankRetriever(Retriever):
    def __init__(self, base: Retriever, reranker: VLMReranker, name: str | None = None):
        super().__init__()
        self.base, self.reranker = base, reranker
        self.name = name or f"{base.name}+vlm_rerank{reranker.top_n}"

    def index(self, ds: RetrievalDataset) -> None:
        self.dataset = ds
        if self.base.dataset is not ds:
            self.base.index(ds)

    def search(self, queries: dict[str, Query], k: int = 100) -> Run:
        run = self.base.search(queries, max(k, self.reranker.top_n))
        return truncate(self.reranker.rerank_run(run, queries, self.dataset), k)
