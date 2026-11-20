"""Train/test leakage guard: do two processed datasets share documents (or queries)?

Any fine-tuning table is worthless if the evaluation corpus overlaps training data, and the failure is easy to
miss (same image re-encoded, same caption reused across a "split"). This compares content fingerprints, not ids:

  documents -> sha1 of the image file bytes (or of the normalised text for text corpora)
  queries   -> sha1 of the normalised query text

python -m mmsearch.data.check_leak --a coco_train --b coco_test
python -m mmsearch.data.check_leak --a vidore_docvqa --b vidore_docvqa__zh     # derived sets must share corpus
Exit code is 1 when doc overlap > 0 or query overlap > 0, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from tqdm import tqdm

from ..utils.misc import get_logger, markdown_table
from .schema import RetrievalDataset, load_dataset

log = get_logger(__name__)


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _text_sha1(text: str) -> str:
    return hashlib.sha1(" ".join((text or "").lower().split()).encode("utf-8")).hexdigest()


def doc_fingerprints(ds: RetrievalDataset, max_docs: int | None = None, desc: str = "hash docs") -> dict[str, str]:
    """fingerprint -> doc_id"""
    out: dict[str, str] = {}
    docs = list(ds.corpus.values())
    if max_docs:
        docs = docs[:max_docs]
    for d in tqdm(docs, desc=f"{ds.name}: {desc}", disable=max_docs is None and len(docs) < 2000):
        fp = _file_sha1(d.image) if d.image and Path(d.image).exists() else _text_sha1(d.text or "")
        out.setdefault(fp, d.doc_id)
    return out


def query_fingerprints(ds: RetrievalDataset, max_queries: int | None = None) -> dict[str, str]:
    qs = list(ds.queries.values())
    if max_queries:
        qs = qs[:max_queries]
    out: dict[str, str] = {}
    for q in qs:
        if q.text:
            out.setdefault(_text_sha1(q.text), q.query_id)
    return out


def compare(a: RetrievalDataset, b: RetrievalDataset, max_docs: int | None = None) -> dict:
    fa, fb = doc_fingerprints(a, max_docs), doc_fingerprints(b, max_docs)
    qa, qb = query_fingerprints(a, max_docs), query_fingerprints(b, max_docs)
    doc_overlap = sorted(set(fa) & set(fb))
    q_overlap = sorted(set(qa) & set(qb))
    return {
        "a": {"name": a.name, "docs": len(a.corpus), "hashed_docs": len(fa), "queries": len(a.queries)},
        "b": {"name": b.name, "docs": len(b.corpus), "hashed_docs": len(fb), "queries": len(b.queries)},
        "doc_overlap": len(doc_overlap),
        "doc_overlap_examples": [(b.corpus[fb[h]].doc_id, a.corpus[fa[h]].doc_id) for h in doc_overlap[:10]],
        "query_overlap": len(q_overlap),
        "query_overlap_examples": [(b.queries[qb[h]].query_id, b.queries[qb[h]].text) for h in q_overlap[:10]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--max-docs", type=int, default=None, help="hash only the first N docs of each side (quick check)")
    ap.add_argument("--allow-overlap", action="store_true", help="exit 0 even if documents overlap (derived datasets)")
    args = ap.parse_args()
    res = compare(load_dataset(args.a), load_dataset(args.b), args.max_docs)
    for side in ("a", "b"):
        s = res[side]
        log.info("%s: %d docs (%d hashed), %d queries", s["name"], s["docs"], s["hashed_docs"], s["queries"])
    log.info("documents shared between %s and %s: %d", res["a"]["name"], res["b"]["name"], res["doc_overlap"])
    log.info("queries   shared between %s and %s: %d", res["a"]["name"], res["b"]["name"], res["query_overlap"])
    rows = [{"check": "documents", "overlap": res["doc_overlap"]}, {"check": "queries", "overlap": res["query_overlap"]}]
    print(markdown_table(rows))
    for ex in res["doc_overlap_examples"]:
        print("  doc in both:", ex)
    for ex in res["query_overlap_examples"]:
        print("  query in both:", ex)
    bad = res["doc_overlap"] > 0 or res["query_overlap"] > 0
    if bad and not args.allow_overlap:
        log.error("LEAKAGE: %s and %s share content — results on this pair are not usable as a test set",
                  res["a"]["name"], res["b"]["name"])
        sys.exit(1)
    log.info("OK: no shared content" if not bad else "OK (overlap allowed by flag)")


if __name__ == "__main__":
    main()
