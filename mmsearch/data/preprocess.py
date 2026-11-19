"""Convert raw datasets (downloaded ModelScope-first) into the unified retrieval format.

Examples
--------
python -m mmsearch.data.preprocess vidore --subsets all
python -m mmsearch.data.preprocess coco --splits test train
python -m mmsearch.data.preprocess flickr --splits test
python -m mmsearch.data.preprocess docvqa --split validation
python -m mmsearch.data.preprocess infovqa --split validation
python -m mmsearch.data.preprocess chartqa --split test
python -m mmsearch.data.preprocess colpali_train --shards 35      # ~50k pages; 'all' for 118k
python -m mmsearch.data.preprocess colpali_pool --n-pages 20000   # in-domain pool for query-rewrite rewards
python -m mmsearch.data.preprocess pool --name vidore_pool --sources vidore_docvqa vidore_infovqa ... --queries-from vidore_docvqa
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import tarfile
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from ..utils.hub import resolve_dataset
from ..utils.misc import get_logger
from ..utils.paths import processed_dir
from .schema import Doc, Query, RetrievalDataset, load_dataset

log = get_logger(__name__)

# ViDoRe v1 benchmark: short name -> (repo id, domain, query language)
VIDORE_V1: dict[str, tuple[str, str, str]] = {
    "arxivqa": ("vidore/arxivqa_test_subsampled", "scientific_figures", "en"),
    "docvqa": ("vidore/docvqa_test_subsampled", "industry_documents", "en"),
    "infovqa": ("vidore/infovqa_test_subsampled", "infographics", "en"),
    "tabfquad": ("vidore/tabfquad_test_subsampled", "tables_fr", "fr"),
    "tatdqa": ("vidore/tatdqa_test", "financial_reports", "en"),
    "shift": ("vidore/shiftproject_test", "environment_fr", "fr"),
    "synth_ai": ("vidore/syntheticDocQA_artificial_intelligence_test", "artificial_intelligence", "en"),
    "synth_energy": ("vidore/syntheticDocQA_energy_test", "energy", "en"),
    "synth_gov": ("vidore/syntheticDocQA_government_reports_test", "government", "en"),
    "synth_health": ("vidore/syntheticDocQA_healthcare_industry_test", "healthcare", "en"),
}


# ----------------------------------------------------------------------------- helpers
def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:150]


def _img_bytes(cell: Any) -> bytes | None:
    if cell is None:
        return None
    if isinstance(cell, dict):
        return cell.get("bytes")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    return None


def _ext(b: bytes) -> str:
    try:
        fmt = (Image.open(io.BytesIO(b)).format or "png").lower()
    except Exception:  # noqa: BLE001
        fmt = "png"
    return {"jpeg": "jpg", "mpo": "jpg"}.get(fmt, fmt)


def _save_image(b: bytes, images_dir: Path, stem: str) -> str:
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / f"{_safe(stem)}.{_ext(b)}"
    if not path.exists():
        path.write_bytes(b)
    return str(path)


def _iter_parquet(files: list[Path], columns: list[str] | None = None, batch_size: int = 64) -> Iterator[dict]:
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in (columns or pf.schema_arrow.names) if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            yield from batch.to_pylist()


def _parquets(root: str | Path, pattern: str) -> list[Path]:
    files = sorted(Path(root).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no parquet matching {pattern} under {root}")
    return files


# ----------------------------------------------------------------------------- ViDoRe
def build_vidore(subset: str) -> RetrievalDataset:
    repo, domain, lang = VIDORE_V1[subset]
    raw = resolve_dataset(repo, allow_patterns=["data/*", "*.json", "README.md"])
    out = processed_dir() / f"vidore_{subset}"
    images_dir = out / "images"
    corpus: dict[str, Doc] = {}
    queries: dict[str, Query] = {}
    qrels: dict[str, dict[str, int]] = {}
    q_by_text: dict[str, str] = {}
    for row in tqdm(_iter_parquet(_parquets(raw, "data/test-*.parquet")), desc=f"vidore/{subset}"):
        b = _img_bytes(row.get("image"))
        if b is None:
            continue
        doc_id = row.get("image_filename") or hashlib.sha1(b).hexdigest()[:16]
        doc_id = str(doc_id)
        if doc_id not in corpus:
            corpus[doc_id] = Doc(
                doc_id,
                image=_save_image(b, images_dir, doc_id),
                meta={"page": row.get("page"), "source": row.get("source"), "domain": domain},
            )
        qtext = row.get("query")
        if not qtext:
            continue  # distractor page without query (synthetic / shift subsets)
        qid = q_by_text.get(qtext)
        if qid is None:
            qid = f"q{len(queries)}"
            q_by_text[qtext] = qid
            meta = {"domain": domain, "source": row.get("source")}
            if row.get("answer") is not None:
                meta["answers"] = [row["answer"]]
            if row.get("options"):
                meta["options"] = row["options"]
            if row.get("answer_type"):
                meta["answer_type"] = row["answer_type"]
            queries[qid] = Query(qid, text=qtext, lang=lang, meta=meta)
        qrels.setdefault(qid, {})[doc_id] = 1
    ds = RetrievalDataset(f"vidore_{subset}", corpus, queries, qrels, {"split": "test", "domain": domain, "lang": lang, "source_repo": repo})
    ds.save(out)
    log.info("saved %s -> %s", ds, out)
    return ds


# ----------------------------------------------------------------------------- COCO / Flickr30K (Karpathy split, clip-benchmark webdataset)
def _iter_wds(tar_files: list[Path]) -> Iterator[tuple[str, bytes, list[str]]]:
    """Yield (key, image_bytes, captions) from webdataset tars (jpg/png/webp + txt)."""
    for tf in tar_files:
        cur_key, img, caps = None, None, None
        with tarfile.open(tf, "r|*") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                name = Path(m.name).name
                key, _, ext = name.partition(".")
                if key != cur_key:
                    if cur_key is not None and img is not None:
                        yield cur_key, img, caps or []
                    cur_key, img, caps = key, None, None
                data = tar.extractfile(m).read()
                ext = ext.lower()
                if ext in ("jpg", "jpeg", "png", "webp"):
                    img = data
                elif ext == "txt":
                    caps = [c.strip() for c in data.decode("utf-8", errors="ignore").split("\n") if c.strip()]
            if cur_key is not None and img is not None:
                yield cur_key, img, caps or []


def build_image_text(kind: str, split: str, max_images: int | None = None) -> RetrievalDataset:
    repo = {"coco": "clip-benchmark/wds_mscoco_captions", "flickr": "clip-benchmark/wds_flickr30k"}[kind]
    raw = resolve_dataset(repo, allow_patterns=[f"{split}/*", "*.json", "*.txt"], tag=split)
    tars = sorted(Path(raw, split).glob("*.tar"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    out = processed_dir() / f"{kind}_{split}"
    images_dir = out / "images"
    corpus: dict[str, Doc] = {}
    queries: dict[str, Query] = {}
    qrels: dict[str, dict[str, int]] = {}
    by_hash: dict[str, str] = {}
    n_caps: dict[str, int] = {}
    for key, b, caps in tqdm(_iter_wds(tars), desc=f"{kind}/{split}"):
        h = hashlib.sha1(b).hexdigest()
        doc_id = by_hash.get(h)
        if doc_id is None:  # robust to one-caption-per-sample layouts: merge captions of identical images
            if max_images and len(corpus) >= max_images:
                break
            doc_id = f"img{len(corpus)}"
            by_hash[h] = doc_id
            corpus[doc_id] = Doc(doc_id, image=_save_image(b, images_dir, doc_id), meta={"wds_key": key})
        for c in caps:
            i = n_caps.get(doc_id, 0)
            n_caps[doc_id] = i + 1
            qid = f"{doc_id}_c{i}"
            queries[qid] = Query(qid, text=c, lang="en", meta={"image_id": doc_id})
            qrels[qid] = {doc_id: 1}
    ds = RetrievalDataset(f"{kind}_{split}", corpus, queries, qrels, {"split": split, "domain": "natural_images", "source_repo": repo})
    ds.save(out)
    log.info("saved %s -> %s (captions/image=%.2f)", ds, out, len(queries) / max(1, len(corpus)))
    return ds


# ----------------------------------------------------------------------------- DocVQA / InfographicVQA / ChartQA (RAG evaluation with answers)
def _build_vqa(name: str, files: list[Path], qcol: str, acol: str, domain: str, page_key=None, max_queries=None) -> RetrievalDataset:
    out = processed_dir() / name
    images_dir = out / "images"
    corpus: dict[str, Doc] = {}
    queries: dict[str, Query] = {}
    qrels: dict[str, dict[str, int]] = {}
    by_hash: dict[str, str] = {}
    for row in tqdm(_iter_parquet(files), desc=name):
        b = _img_bytes(row.get("image"))
        q = row.get(qcol)
        if b is None or not q:
            continue
        h = hashlib.sha1(b).hexdigest()
        doc_id = by_hash.get(h)
        if doc_id is None:
            doc_id = _safe(page_key(row)) if page_key and page_key(row) else h[:16]
            if doc_id in corpus:  # same key, different bytes -> disambiguate
                doc_id = f"{doc_id}_{h[:6]}"
            by_hash[h] = doc_id
            corpus[doc_id] = Doc(doc_id, image=_save_image(b, images_dir, doc_id), meta={"domain": domain})
        answers = row.get(acol)
        answers = answers if isinstance(answers, list) else [answers]
        qid = str(row.get("questionId") or f"q{len(queries)}")
        if qid in queries:
            qid = f"{qid}_{len(queries)}"
        meta = {"answers": [str(a) for a in answers if a is not None], "domain": domain}
        for extra in ("question_types", "human_or_machine", "answer_type"):
            if row.get(extra) is not None:
                meta[extra] = row[extra]
        queries[qid] = Query(qid, text=str(q), lang="en", meta=meta)
        qrels[qid] = {doc_id: 1}
        if max_queries and len(queries) >= max_queries:
            break
    ds = RetrievalDataset(name, corpus, queries, qrels, {"domain": domain, "task": "vqa"})
    ds.save(out)
    log.info("saved %s -> %s", ds, out)
    return ds


def build_docvqa(split: str = "validation", max_queries: int | None = None) -> RetrievalDataset:
    raw = resolve_dataset("lmms-lab/DocVQA", allow_patterns=[f"DocVQA/{split}-*", "README.md"], tag=f"docvqa_{split}")
    files = _parquets(raw, f"DocVQA/{split}-*.parquet")
    key = lambda r: f"{r.get('ucsf_document_id')}_p{r.get('ucsf_document_page_no')}" if r.get("ucsf_document_id") else None  # noqa: E731
    return _build_vqa(f"docvqa_{split}", files, "question", "answers", "industry_documents", key, max_queries)


def build_infovqa(split: str = "validation", max_queries: int | None = None) -> RetrievalDataset:
    raw = resolve_dataset("lmms-lab/DocVQA", allow_patterns=[f"InfographicVQA/{split}-*", "README.md"], tag=f"infovqa_{split}")
    files = _parquets(raw, f"InfographicVQA/{split}-*.parquet")
    return _build_vqa(f"infovqa_{split}", files, "question", "answers", "infographics", None, max_queries)


def build_chartqa(split: str = "test", max_queries: int | None = None) -> RetrievalDataset:
    raw = resolve_dataset("HuggingFaceM4/ChartQA", allow_patterns=[f"data/{split}-*", "README.md"], tag=split)
    files = _parquets(raw, f"data/{split}-*.parquet")
    return _build_vqa(f"chartqa_{split}", files, "query", "label", "charts", None, max_queries)


# ----------------------------------------------------------------------------- ColPali training set (kept as parquet; read lazily by the trainer)
def download_colpali_train(shards: str | int = 35) -> str:
    if str(shards) == "all":
        patterns = ["data/*", "README.md"]
        tag = "all"
    else:
        n = int(shards)
        patterns = [f"data/train-{i:05d}-of-00082.parquet" for i in range(n)] + ["data/test-*", "README.md"]
        tag = f"{n}shards"
    path = resolve_dataset("vidore/colpali_train_set", allow_patterns=patterns, tag=tag)
    log.info("colpali_train_set (%s) at %s", tag, path)
    return path


# ----------------------------------------------------------------------------- pooled corpus for scale-up latency experiments
def build_colpali_pool(n_pages: int = 20000, shards: str | int = 35, seed: int = 0, name: str = "colpali_train_pool") -> RetrievalDataset:
    """Export a random sample of colpali_train_set (pages + their queries) as a retrieval dataset.
    Used as the in-domain pool for query-rewrite reward computation (stage 4) — never a test set."""
    import random as _random

    root = download_colpali_train(shards)
    files = _parquets(root, "data/train-*.parquet")
    rows_per_file = [pq.ParquetFile(f).metadata.num_rows for f in files]
    total = sum(rows_per_file)
    keep = set(_random.Random(seed).sample(range(total), min(n_pages, total)))
    out = processed_dir() / name
    corpus: dict[str, Doc] = {}
    queries: dict[str, Query] = {}
    qrels: dict[str, dict[str, int]] = {}
    gi = 0
    for f, nrows in zip(files, rows_per_file):
        local = [i for i in range(nrows) if gi + i in keep]
        base = gi
        gi += nrows
        if not local:
            continue
        pf = pq.ParquetFile(f)
        cols = [c for c in ["image", "image_filename", "query", "answer", "source"] if c in pf.schema_arrow.names]
        for li, row in zip(local, pf.read(columns=cols).take(local).to_pylist()):
            if not row.get("query"):
                continue
            b = _img_bytes(row.get("image"))
            if b is None:
                continue
            uid = base + li
            doc_id = f"p{uid}"
            corpus[doc_id] = Doc(doc_id, image=_save_image(b, out / "images", doc_id), meta={"source": row.get("source")})
            qid = f"q{uid}"
            queries[qid] = Query(qid, text=row["query"], lang="en", meta={"answers": [row.get("answer")], "source": row.get("source")})
            qrels[qid] = {doc_id: 1}
    ds = RetrievalDataset(name, corpus, queries, qrels, {"split": "train", "source_repo": "vidore/colpali_train_set"})
    ds.save(out)
    log.info("saved %s", ds)
    return ds


def build_pool(name: str, sources: list[str], queries_from: list[str]) -> RetrievalDataset:
    corpus: dict[str, Doc] = {}
    queries: dict[str, Query] = {}
    qrels: dict[str, dict[str, int]] = {}
    for s in sources:
        ds = load_dataset(s)
        for d in ds.corpus.values():
            nid = f"{s}::{d.doc_id}"
            corpus[nid] = Doc(nid, d.text, d.image, {**d.meta, "pool_source": s})
        if s in queries_from:
            for qid, q in ds.queries.items():
                nq = f"{s}::{qid}"
                queries[nq] = Query(nq, q.text, q.image, q.lang, {**q.meta, "pool_source": s})
                qrels[nq] = {f"{s}::{d}": r for d, r in ds.qrels.get(qid, {}).items()}
    ds = RetrievalDataset(name, corpus, queries, qrels, {"pooled_from": sources, "queries_from": queries_from})
    ds.save(processed_dir() / name)
    log.info("saved pooled %s", ds)
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vidore")
    p.add_argument("--subsets", nargs="+", default=["all"])
    for k in ("coco", "flickr"):
        p = sub.add_parser(k)
        p.add_argument("--splits", nargs="+", default=["test"])
        p.add_argument("--max-images", type=int, default=None)
    for k, default in (("docvqa", "validation"), ("infovqa", "validation"), ("chartqa", "test")):
        p = sub.add_parser(k)
        p.add_argument("--split", default=default)
        p.add_argument("--max-queries", type=int, default=None)
    p = sub.add_parser("colpali_train")
    p.add_argument("--shards", default="35")
    p = sub.add_parser("colpali_pool")
    p.add_argument("--n-pages", type=int, default=20000)
    p.add_argument("--shards", default="35")
    p = sub.add_parser("pool")
    p.add_argument("--name", required=True)
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--queries-from", nargs="+", required=True)
    args = ap.parse_args()

    if args.cmd == "vidore":
        subsets = list(VIDORE_V1) if args.subsets == ["all"] else args.subsets
        for s in subsets:
            build_vidore(s)
    elif args.cmd in ("coco", "flickr"):
        for sp in args.splits:
            build_image_text(args.cmd, sp, args.max_images)
    elif args.cmd == "docvqa":
        build_docvqa(args.split, args.max_queries)
    elif args.cmd == "infovqa":
        build_infovqa(args.split, args.max_queries)
    elif args.cmd == "chartqa":
        build_chartqa(args.split, args.max_queries)
    elif args.cmd == "colpali_train":
        download_colpali_train(args.shards)
    elif args.cmd == "colpali_pool":
        build_colpali_pool(args.n_pages, args.shards)
    elif args.cmd == "pool":
        build_pool(args.name, args.sources, args.queries_from)


if __name__ == "__main__":
    main()
