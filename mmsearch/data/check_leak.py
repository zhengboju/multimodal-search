"""Train/test leakage guard: compare an evaluation set against the training corpus by content, not by id.

A fine-tuning table is worthless if the evaluation corpus overlaps training data, and the split annotation can
lie (same image re-encoded, ids reassigned). Fingerprints: sha1 of image bytes, sha1 of normalised text.

Three different overlaps are reported, because they mean very different things:

* **document overlap** — the eval set's gold image also exists in training. Only this can inflate a score.
* **query text overlap** — the caption string also occurs in training, usually attached to a *different* image.
  COCO captions are short and generic ("A man sitting in front of a laptop computer."), so this is expected;
  it is reported for transparency and can be excluded from the metrics with `--write-clean`.
* **pair overlap** — the same (text, image) pair on both sides. That is the real leak, and it should be zero.

    python -m mmsearch.data.check_leak --a coco_train --b coco_test
    python -m mmsearch.data.check_leak --a coco_train --b coco_test --write-clean coco_test_clean
    python -m mmsearch.data.check_leak --a vidore_docvqa --b vidore_docvqa__zh --allow-overlap
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from tqdm import tqdm

from ..utils.misc import get_logger, markdown_table
from ..utils.paths import processed_dir
from .schema import RetrievalDataset, load_dataset

log = get_logger(__name__)


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_fp(text: str) -> str:
    return hashlib.sha1(" ".join((text or "").lower().split()).encode("utf-8")).hexdigest()


def doc_fingerprints(ds: RetrievalDataset, max_docs: int | None = None) -> dict[str, str]:
    """doc_id -> content fingerprint (image bytes when present, else normalised text)."""
    out: dict[str, str] = {}
    docs = list(ds.corpus.values())
    if max_docs:
        docs = docs[:max_docs]
    for d in tqdm(docs, desc=f"{ds.name}: hash docs", disable=len(docs) < 2000):
        out[d.doc_id] = _file_sha1(d.image) if d.image and Path(d.image).exists() else text_fp(d.text or "")
    return out


def query_fingerprints(ds: RetrievalDataset, max_queries: int | None = None) -> dict[str, str]:
    qs = list(ds.queries.values())
    if max_queries:
        qs = qs[:max_queries]
    return {q.query_id: text_fp(q.text) for q in qs if q.text}


def gold_pairs(ds: RetrievalDataset, doc_fps: dict[str, str]) -> set[tuple[str, str]]:
    """{(query text fp, gold doc fp)} — the exact thing that must never be shared with a training set."""
    out: set[tuple[str, str]] = set()
    for qid, rels in ds.qrels.items():
        q = ds.queries.get(qid)
        if q is None or not q.text:
            continue
        t = text_fp(q.text)
        for d, r in rels.items():
            if r > 0 and d in doc_fps:
                out.add((t, doc_fps[d]))
    return out


def compare(a: RetrievalDataset, b: RetrievalDataset, max_docs: int | None = None) -> dict:
    fa, fb = doc_fingerprints(a, max_docs), doc_fingerprints(b, max_docs)
    qa, qb = query_fingerprints(a, max_docs), query_fingerprints(b, max_docs)
    shared_fps = set(fa.values()) & set(fb.values())
    leaked_docs = sorted(d for d, fp in fb.items() if fp in shared_fps)
    shared_q = set(qa.values()) & set(qb.values())
    leaked_queries = sorted(q for q, fp in qb.items() if fp in shared_q)
    train_pairs = gold_pairs(a, fa)
    b_pairs = gold_pairs(b, fb)
    pair_leak = sorted(b_pairs & train_pairs)
    inv_qb = {fp: q for q, fp in qb.items()}
    inv_fb = {fp: d for d, fp in fb.items()}
    return {
        "a": {"name": a.name, "docs": len(a.corpus), "hashed_docs": len(fa), "queries": len(a.queries)},
        "b": {"name": b.name, "docs": len(b.corpus), "hashed_docs": len(fb), "queries": len(b.queries)},
        "doc_overlap": len(leaked_docs),
        "doc_overlap_pct": 100.0 * len(leaked_docs) / max(1, len(fb)),
        "leaked_doc_ids_b": leaked_docs,
        "query_overlap": len(leaked_queries),
        "query_overlap_pct": 100.0 * len(leaked_queries) / max(1, len(qb)),
        "leaked_query_ids_b": leaked_queries,
        "pair_overlap": len(pair_leak),
        "pair_overlap_examples": [(inv_qb.get(t), inv_fb.get(d)) for t, d in pair_leak[:5]],
        "_fa": fa, "_fb": fb, "_qa": qa, "_qb": qb, "_leaked_docs": leaked_docs, "_leaked_queries": leaked_queries,
    }


def clean_subset(a: RetrievalDataset, b: RetrievalDataset, name: str, drop_dup_queries: bool = True,
                 max_docs: int | None = None) -> tuple[RetrievalDataset, dict]:
    """Write an evaluation set with leaked documents (and optionally duplicated caption texts) removed."""
    res = compare(a, b, max_docs)
    drop_docs = set(res["_leaked_docs"])
    drop_q = set(res["_leaked_queries"]) if drop_dup_queries else set()
    corpus = {d: v for d, v in b.corpus.items() if d not in drop_docs}
    queries = {q: v for q, v in b.queries.items() if q not in drop_q and v.text}
    qrels = {q: {d: r for d, r in b.qrels[q].items() if d in corpus} for q in queries}
    qrels = {q: rels for q, rels in qrels.items() if rels}  # keep only queries that still have a gold page
    out = RetrievalDataset(name, corpus, {q: queries[q] for q in qrels}, qrels,
                           {**b.info, "cleaned_from": b.name, "train_compared": a.name,
                            "dropped_docs": len(drop_docs), "dropped_queries": len(b.queries) - len(qrels)})
    out.save(processed_dir() / name, save_corpus=True)
    stats = {"dropped_docs": len(drop_docs), "dropped_queries": len(b.queries) - len(qrels),
             "kept_docs": len(corpus), "kept_queries": len(qrels), "pair_overlap": res["pair_overlap"]}
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="training set (corpus that the model saw)")
    ap.add_argument("--b", required=True, help="evaluation set")
    ap.add_argument("--max-docs", type=int, default=None, help="hash only the first N docs per side (quick check)")
    ap.add_argument("--max-doc-overlap-pct", type=float, default=0.1,
                    help="tolerated share of eval documents present in training (default 0.1%%)")
    ap.add_argument("--fail-on-query-overlap", action="store_true",
                    help="treat duplicated caption strings as fatal (off by default: generic captions are expected)")
    ap.add_argument("--write-clean", default=None,
                    help="write a leak-free copy of the eval set under storage/processed/<name>")
    ap.add_argument("--keep-duplicated-queries", action="store_true",
                    help="when writing the clean set, keep caption texts that also occur in training")
    ap.add_argument("--allow-overlap", action="store_true", help="always exit 0 (derived datasets share corpora)")
    args = ap.parse_args()
    a, b = load_dataset(args.a), load_dataset(args.b)
    res = compare(a, b, args.max_docs)
    log.info("%s: %d docs, %d queries", a.name, len(a.corpus), len(a.queries))
    log.info("%s: %d docs, %d queries", b.name, len(b.corpus), len(b.queries))
    rows = [
        {"check": "documents (gold image seen in training)", "n": res["doc_overlap"],
         "pct": round(res["doc_overlap_pct"], 3), "verdict": "LEAK" if res["doc_overlap_pct"] > args.max_doc_overlap_pct else "negligible"},
        {"check": "query texts (generic captions, usually other images)", "n": res["query_overlap"],
         "pct": round(res["query_overlap_pct"], 2), "verdict": "expected / benign"},
        {"check": "exact (text, image) pairs", "n": res["pair_overlap"], "pct": 0.0,
         "verdict": "LEAK" if res["pair_overlap"] else "clean"},
    ]
    print(markdown_table(rows, ["check", "n", "pct", "verdict"], digits=3))
    if res["leaked_doc_ids_b"]:
        print("  doc ids in both (first 10):", res["leaked_doc_ids_b"][:10])

    if args.write_clean:
        out, stats = clean_subset(a, b, args.write_clean, drop_dup_queries=not args.keep_duplicated_queries,
                                  max_docs=args.max_docs)
        log.info("wrote %s -> %s", out, processed_dir() / args.write_clean)
        log.info("clean set stats: %s", stats)
        print(markdown_table([stats], digits=2))

    hard_leak = res["doc_overlap_pct"] > args.max_doc_overlap_pct or res["pair_overlap"] > 0
    soft_leak = res["query_overlap"] > 0 and args.fail_on_query_overlap
    if (hard_leak or soft_leak) and not args.allow_overlap:
        log.error("LEAKAGE between %s and %s: %.2f%% of eval documents are in training (%d pairs shared). "
                  "Re-run with --write-clean to evaluate on a leak-free subset.", a.name, b.name,
                  res["doc_overlap_pct"], res["pair_overlap"])
        sys.exit(1)
    log.info("OK: evaluation set is usable as a test set%s",
             "" if not res["doc_overlap"] else f" ({res['doc_overlap']} duplicated document(s) — consider --write-clean)")


if __name__ == "__main__":
    main()
