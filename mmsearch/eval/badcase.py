"""Badcase analysis over benchmark per-query dumps.

python -m mmsearch.eval.badcase --run-dir storage/runs/benchmark/<run> --method colqwen2-v1.0 [--compare bm25-ocr]

Produces badcases.md with: gold-rank failure buckets, metric breakdown by dataset / query type / length /
language, failure clusters (TF-IDF + spherical k-means, no sklearn needed), A-vs-B win/loss analysis,
and concrete examples (query, gold page, top-3 pages) to label by hand.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..data.schema import load_dataset
from ..retrieve.sparse import tokenize
from ..utils.misc import markdown_table, read_jsonl

RANK_BUCKETS = [("rank1", 1, 1), ("rank2-5", 2, 5), ("rank6-10", 6, 10), ("rank11-100", 11, 100)]


def rank_bucket(r: int | None) -> str:
    if r is None:
        return "miss(>k)"
    for name, lo, hi in RANK_BUCKETS:
        if lo <= r <= hi:
            return name
    return "miss(>k)"


def query_type(q: str) -> str:
    s = (q or "").lower()
    if re.search(r"\b(how many|how much|what (?:is|was) the (?:total|number|amount|percentage|value)|percent|%|\d)", s):
        return "numeric"
    if re.search(r"\b(compare|difference|more than|less than|higher|lower|versus|vs\.?)\b", s):
        return "comparison"
    if re.search(r"\b(table|figure|chart|graph|diagram|plot|image)\b", s):
        return "visual_element"
    if re.search(r"^(is|are|does|do|did|can|was|were)\b", s):
        return "yes_no"
    if re.search(r"\b(what is|define|definition|meaning)\b", s):
        return "definition"
    return "other"


def length_bucket(q: str) -> str:
    n = len((q or "").split())
    return "short(<=6)" if n <= 6 else "medium(7-15)" if n <= 15 else "long(>15)"


def breakdown(rows: list[dict], key, metric: str) -> list[dict]:
    g: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        g[key(r)].append(r[metric])
    return sorted(({"group": k, "n": len(v), metric: 100 * float(np.mean(v))} for k, v in g.items()), key=lambda x: -x["n"])


def tfidf_kmeans(texts: list[str], k: int = 6, iters: int = 20, seed: int = 0) -> tuple[np.ndarray, list[list[str]]]:
    docs = [tokenize(t) for t in texts]
    vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
    if not vocab or len(texts) < k:
        return np.zeros(len(texts), dtype=int), [[]]
    X = np.zeros((len(docs), len(vocab)), np.float32)
    for i, d in enumerate(docs):
        for w, c in Counter(d).items():
            X[i, vocab[w]] = c
    df = (X > 0).sum(0)
    X *= np.log((1 + len(docs)) / (1 + df)) + 1
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = (X @ C.T).argmax(1)
        for j in range(k):
            if (a == j).any():
                c = X[a == j].mean(0)
                C[j] = c / (np.linalg.norm(c) + 1e-9)
    a = (X @ C.T).argmax(1)
    inv = {i: w for w, i in vocab.items()}
    terms = [[inv[i] for i in np.argsort(-C[j])[:6]] for j in range(k)]
    return a, terms


def analyse(rows: list[dict], method: str, metric: str, compare: str | None, n_examples: int, n_clusters: int) -> str:
    mine = [r for r in rows if r["method"] == method]
    if not mine:
        raise SystemExit(f"method {method} not found; available: {sorted({r['method'] for r in rows})}")
    md = [f"# Badcase analysis — {method}", "", f"queries: {len(mine)}, mean {metric}: {100 * np.mean([r[metric] for r in mine]):.2f}", ""]

    md += ["## 1. Gold-rank buckets (where does the relevant page land?)", ""]
    b = Counter(rank_bucket(r["gold_rank"]) for r in mine)
    order = [x[0] for x in RANK_BUCKETS] + ["miss(>k)"]
    md.append(markdown_table([{"bucket": k, "n": b.get(k, 0), "share%": 100 * b.get(k, 0) / len(mine)} for k in order], digits=1))

    for title, key in (("dataset", lambda r: r["dataset"]), ("query type (heuristic)", lambda r: query_type(r["query"])),
                       ("query length", lambda r: length_bucket(r["query"])), ("language", lambda r: r.get("lang", "")),
                       ("domain", lambda r: r.get("domain", ""))):
        md += ["", f"## Breakdown by {title}", "", markdown_table(breakdown(mine, key, metric), digits=1)]

    fails = [r for r in mine if (r["gold_rank"] or 10**9) > 1]
    if len(fails) >= n_clusters:
        assign, terms = tfidf_kmeans([r["query"] or "" for r in fails], n_clusters)
        crow = []
        for j in range(n_clusters):
            members = [fails[i] for i in np.where(assign == j)[0]]
            if members:
                crow.append({"cluster": j, "n": len(members), "top_terms": " ".join(terms[j]),
                             "miss%": 100 * np.mean([m["gold_rank"] is None for m in members]),
                             "example": (members[0]["query"] or "")[:80]})
        md += ["", "## Failure clusters (queries with gold rank > 1)", "", markdown_table(crow, digits=1)]

    if compare:
        other = {(r["dataset"], r["query_id"]): r for r in rows if r["method"] == compare}
        pairs = [(r, other[(r["dataset"], r["query_id"])]) for r in mine if (r["dataset"], r["query_id"]) in other]
        wins = [p for p in pairs if p[0][metric] > p[1][metric] + 1e-9]
        losses = [p for p in pairs if p[0][metric] < p[1][metric] - 1e-9]
        md += ["", f"## {method} vs {compare} on {metric}", "",
               f"wins {len(wins)} / losses {len(losses)} / ties {len(pairs) - len(wins) - len(losses)}", ""]
        md.append(markdown_table(breakdown([p[0] for p in losses], lambda r: query_type(r["query"]), metric) or [{"group": "-", "n": 0}], digits=1))
        md += ["", f"### Largest losses vs {compare} (where the other system is right)", ""]
        losses.sort(key=lambda p: p[0][metric] - p[1][metric])
        md.append(markdown_table([{"dataset": a["dataset"], "query": (a["query"] or "")[:90], f"{method}_rank": a["gold_rank"],
                                   f"{compare}_rank": c["gold_rank"]} for a, c in losses[:n_examples]]))

    md += ["", "## Worst examples (label failure mode by hand: OCR-able text / visual layout / ambiguous query / label noise)", ""]
    worst = sorted(fails, key=lambda r: (r["gold_rank"] is not None, -(r["gold_rank"] or 0)))[:n_examples]
    ex_rows = []
    corpora: dict[str, dict] = {}
    for r in worst:
        if r["dataset"] not in corpora:
            try:
                corpora[r["dataset"]] = load_dataset(r["dataset"]).corpus
            except FileNotFoundError:
                corpora[r["dataset"]] = {}
        corpus = corpora[r["dataset"]]
        img = lambda d: Path(corpus[d].image).name if d in corpus and corpus[d].image else d  # noqa: E731
        ex_rows.append({"dataset": r["dataset"], "query": (r["query"] or "")[:100], "gold_rank": r["gold_rank"],
                        "gold": ", ".join(img(d) for d in r["gold"][:2]), "top3": ", ".join(img(d) for d in r["top"][:3]),
                        "failure_mode": ""})
    md.append(markdown_table(ex_rows))
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--metric", default="ndcg@5")
    ap.add_argument("--examples", type=int, default=20)
    ap.add_argument("--clusters", type=int, default=6)
    args = ap.parse_args()
    rows = read_jsonl(Path(args.run_dir) / "per_query.jsonl")
    text = analyse(rows, args.method, args.metric, args.compare, args.examples, args.clusters)
    out = Path(args.run_dir) / f"badcases_{args.method}{'_vs_' + args.compare if args.compare else ''}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
