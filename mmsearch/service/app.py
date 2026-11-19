"""Online search service: FastAPI + micro-batched ColQwen2 query encoding + PLAID-lite/exhaustive index, BM25 & hybrid,
optional VLM rerank and RAG answer via a vLLM server.

MMS_SERVICE_CONFIG=configs/service.yaml uvicorn mmsearch.service.app:app --host 0.0.0.0 --port 8080

POST /search {"query": "...", "k": 10, "mode": "multivector|bm25|hybrid", "rerank": false}
POST /rag    {"query": "...", "k": 3}
GET  /page/{doc_id}    page image;  GET /health;  GET /metrics (Prometheus, if instrumentator installed)
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..data.schema import Query, load_dataset
from ..retrieve.fusion import fuse, truncate
from ..retrieve.multivector import MultiVectorRetriever
from ..retrieve.sparse import BM25Retriever
from ..utils.config import load_config
from ..utils.misc import get_logger

log = get_logger("service")


class MicroBatcher:
    """Collects concurrent query-encoding requests for up to `max_wait_ms` and encodes them as one GPU batch."""

    def __init__(self, encode_fn, max_batch: int = 32, max_wait_ms: float = 5.0):
        self.encode_fn, self.max_batch, self.max_wait = encode_fn, max_batch, max_wait_ms / 1000
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def encode(self, text: str) -> torch.Tensor:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((text, fut))
        return await fut

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.queue.get()]
            deadline = loop.time() + self.max_wait
            while len(items) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            texts = [t for t, _ in items]
            try:
                embs = await loop.run_in_executor(None, self.encode_fn, texts)
                for (_, fut), e in zip(items, embs):
                    fut.set_result(e)
            except Exception as e:  # noqa: BLE001
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(e)


class State:
    cfg: Any = None
    ds: Any = None
    mv: MultiVectorRetriever | None = None
    bm25: BM25Retriever | None = None
    batcher: MicroBatcher | None = None
    rag = None
    reranker = None
    gpu_lock: asyncio.Lock | None = None


S = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(os.environ.get("MMS_SERVICE_CONFIG", "configs/service.yaml"))
    S.cfg = cfg
    S.ds = load_dataset(cfg.dataset)
    S.mv = MultiVectorRetriever(**cfg.multivector)
    S.mv.index(S.ds)
    if cfg.get("bm25", True) and any(d.text for d in S.ds.corpus.values()):
        S.bm25 = BM25Retriever()
        S.bm25.index(S.ds)
    S.batcher = MicroBatcher(lambda ts: S.mv.encoder.encode_queries(ts, batch_size=len(ts)),
                             int(cfg.get("max_batch", 32)), float(cfg.get("max_wait_ms", 5)))
    S.batcher.start()
    S.gpu_lock = asyncio.Lock()
    if cfg.get("vlm"):
        from ..llm.client import VLMClient
        from ..rag.pipeline import RAGConfig, RAGGenerator
        from ..retrieve.rerank import VLMReranker

        client = VLMClient(cfg.vlm.get("base_url"), cfg.vlm.get("model"))
        S.rag = RAGGenerator(client, RAGConfig(total_tokens=int(cfg.vlm.get("total_tokens", 4096))))
        S.reranker = VLMReranker(client, top_n=int(cfg.vlm.get("rerank_top_n", 10)))
    log.info("service ready: %s, index %s", S.ds, S.mv.build_stats.get("index"))
    yield


app = FastAPI(title="mmsearch", lifespan=lifespan)
try:  # optional Prometheus metrics
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)
except ImportError:  # pragma: no cover
    pass


class SearchRequest(BaseModel):
    query: str
    k: int = Field(10, ge=1, le=100)
    mode: str = "multivector"
    rerank: bool = False


class RAGRequest(BaseModel):
    query: str
    k: int = Field(3, ge=1, le=10)


async def _search(req: SearchRequest) -> tuple[dict[str, float], dict[str, float]]:
    timings: dict[str, float] = {}
    runs = []
    if req.mode in ("multivector", "hybrid"):
        t0 = time.perf_counter()
        emb = await S.batcher.encode(req.query)
        timings["encode_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        async with S.gpu_lock:
            I, Sc = await asyncio.get_running_loop().run_in_executor(None, S.mv.search_embeddings, [emb], max(req.k, 50))
        timings["search_ms"] = (time.perf_counter() - t0) * 1000
        runs.append({"q": {S.mv.doc_ids[i]: s for i, s in zip(I[0], Sc[0])}})
    if req.mode in ("bm25", "hybrid"):
        if S.bm25 is None:
            raise HTTPException(400, "BM25 unavailable: corpus has no OCR text (run mmsearch.data.ocr)")
        t0 = time.perf_counter()
        runs.append(S.bm25.search({"q": Query("q", req.query)}, max(req.k, 50)))
        timings["bm25_ms"] = (time.perf_counter() - t0) * 1000
    if not runs:
        raise HTTPException(400, f"unknown mode {req.mode}")
    run = runs[0] if len(runs) == 1 else fuse(runs, "rrf")
    if req.rerank:
        if S.reranker is None:
            raise HTTPException(400, "rerank needs `vlm` in the service config")
        t0 = time.perf_counter()
        run = await asyncio.get_running_loop().run_in_executor(None, S.reranker.rerank_run, run, {"q": Query("q", req.query)}, S.ds)
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
    return truncate(run, req.k)["q"], timings


@app.post("/search")
async def search(req: SearchRequest):
    t0 = time.perf_counter()
    hits, timings = await _search(req)
    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return {"query": req.query, "mode": req.mode, "timings": timings,
            "hits": [{"doc_id": d, "score": s, "image_url": f"/page/{d}", "meta": S.ds.corpus[d].meta}
                     for d, s in sorted(hits.items(), key=lambda x: -x[1])]}


@app.post("/rag")
async def rag(req: RAGRequest):
    if S.rag is None:
        raise HTTPException(400, "RAG needs `vlm` in the service config")
    t0 = time.perf_counter()
    hits, timings = await _search(SearchRequest(query=req.query, k=req.k))
    docs = [d for d, _ in sorted(hits.items(), key=lambda x: -x[1])]
    t1 = time.perf_counter()
    answer = await S.rag.aanswer(req.query, [S.ds.corpus[d].image for d in docs])
    timings["generate_ms"] = (time.perf_counter() - t1) * 1000
    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return {"query": req.query, "answer": answer, "sources": [{"doc_id": d, "image_url": f"/page/{d}"} for d in docs], "timings": timings}


@app.get("/page/{doc_id}")
async def page(doc_id: str):
    d = S.ds.corpus.get(doc_id) if S.ds else None
    if d is None or not d.image:
        raise HTTPException(404)
    return FileResponse(d.image)


@app.get("/health")
async def health():
    return {"ok": S.mv is not None, "docs": len(S.ds.corpus) if S.ds else 0, "index": S.mv.build_stats if S.mv else {}}
