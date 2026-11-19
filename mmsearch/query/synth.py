"""Stage 4 data construction: retrieval-reward-filtered query-rewrite SFT data (rejection sampling).

python -m mmsearch.query.synth build --config configs/query_rewrite.yaml
python -m mmsearch.query.synth validate --file <out>/human_check.csv      # after manual labelling

Idea: the teacher LLM sees only the query (never the gold page -> no answer leakage) and samples N candidate
rewrites. Each candidate is scored by the downstream retriever (rank / score margin of the gold page in an
in-domain pool). The best candidate is kept only if it does not hurt retrieval -> the student learns rewrites that
are *useful for retrieval*, judged by retrieval itself rather than by human taste. A share of zh/ja translated
queries teaches cross-lingual normalisation to English.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from ..data.schema import Query, load_dataset
from ..eval.metrics import gold_rank, ranked_docs
from ..llm.client import VLMClient
from ..retrieve import build_retriever
from ..utils.config import load_config, save_config
from ..utils.misc import get_logger, write_json, write_jsonl
from ..utils.paths import runs_dir
from .rewriter import QueryRewriter, RewriteResult
from .translate import translate_texts

log = get_logger(__name__)
NOT_FOUND = 1000


def _rank_and_margin(scores: dict[str, float], rels: dict[str, int]) -> tuple[int, float]:
    ranking = ranked_docs(scores)
    r = gold_rank(ranking, rels) or NOT_FOUND
    gold = max((scores[d] for d in rels if d in scores), default=None)
    other = max((s for d, s in scores.items() if rels.get(d, 0) <= 0), default=0.0)
    margin = (gold - other) if gold is not None else -1e9
    return r, margin


def cmd_build(cfg) -> None:
    out = Path(cfg.get("out_dir") or runs_dir() / "query" / cfg.name)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    sc = cfg.synth
    rng = random.Random(int(cfg.get("seed", 0)))
    pool = load_dataset(cfg.pool_dataset).subset_queries(int(sc.n_queries), seed=int(cfg.get("seed", 0)))
    qids = list(pool.queries)

    teacher_cfg = cfg.teacher
    client = VLMClient(teacher_cfg.get("base_url"), teacher_cfg.get("model"), max_concurrency=int(teacher_cfg.get("max_concurrency", 64)))

    # 1) multilingual share: translate a subset of queries to zh / ja (source side only; pages stay as-is)
    ml = sc.get("multilingual") or {}
    queries: dict[str, Query] = {}
    to_tr = set(rng.sample(qids, int(len(qids) * float(ml.get("ratio", 0.0))))) if ml else set()
    by_lang: dict[str, list[str]] = {}
    for q in to_tr:
        by_lang.setdefault(rng.choice(list(ml.get("langs", ["zh"]))), []).append(q)
    for q in qids:
        src = pool.queries[q]
        queries[q] = Query(q, src.text, None, "en", {"orig_text": src.text})
    for lang, ids in by_lang.items():
        for q, t in zip(ids, translate_texts(client, [pool.queries[i].text or "" for i in ids], lang)):
            if t:
                queries[q] = Query(q, t, None, lang, {"orig_text": pool.queries[q].text})

    # 2) reward retriever over the pool
    retr = build_retriever(dict(cfg.reward_retriever))
    retr.index(pool)
    depth = int(sc.get("depth", 100))
    base_run = retr.search(queries, depth)
    base = {q: _rank_and_margin(base_run.get(q, {}), pool.qrels[q]) for q in qids}

    # 3) teacher candidates
    n_cand = int(teacher_cfg.get("n_candidates", 4))
    rw = QueryRewriter(client)
    cands = rw.rewrite_many([queries[q].text or "" for q in qids], n=n_cand, temperature=float(teacher_cfg.get("temperature", 0.8)),
                            desc="teacher rewrites")
    cand_queries: dict[str, Query] = {}
    for q, cs in zip(qids, cands):
        for j, c in enumerate(cs):
            if c is not None:
                cand_queries[f"{q}#{j}"] = Query(f"{q}#{j}", c.rewrite, None, "en")
    cand_run = retr.search(cand_queries, depth)

    # 4) select
    rows, stats = [], {"queries": len(qids), "kept_improved": 0, "kept_easy": 0, "dropped_worse": 0, "no_valid_candidate": 0}
    keep_easy = float(sc.get("keep_easy_ratio", 0.3))
    for q, cs in zip(qids, cands):
        scored = []
        for j, c in enumerate(cs):
            if c is None:
                continue
            r, m = _rank_and_margin(cand_run.get(f"{q}#{j}", {}), pool.qrels[q])
            scored.append((r, -m, j, c))
        if not scored:
            stats["no_valid_candidate"] += 1
            continue
        r, neg_m, j, best = min(scored, key=lambda x: (x[0], x[1]))
        r0, m0 = base[q]
        improved = r < r0 or (r == r0 and -neg_m > m0)
        if r > r0:
            stats["dropped_worse"] += 1
            continue
        if not improved or r0 == 1:
            if rng.random() > keep_easy:
                continue
            stats["kept_easy"] += 1
        else:
            stats["kept_improved"] += 1
        best = RewriteResult(best.rewrite, best.intent, best.entities, queries[q].lang)  # lang label from ground truth
        rows.append({"id": q, "query": queries[q].text, "lang": queries[q].lang, "orig_text": queries[q].meta["orig_text"],
                     "target": best.to_json(), "rewrite": best.rewrite, "intent": best.intent,
                     "orig_rank": r0, "new_rank": r, "orig_margin": m0, "new_margin": -neg_m})
    rng.shuffle(rows)
    n_dev = max(1, int(len(rows) * float(sc.get("dev_ratio", 0.05))))
    write_jsonl(rows[n_dev:], out / "sft_train.jsonl")
    write_jsonl(rows[:n_dev], out / "sft_dev.jsonl")
    stats.update({"kept_total": len(rows), "mrr_orig": float(np.mean([1 / base[q][0] for q in qids])),
                  "mrr_best_kept": float(np.mean([1 / r["new_rank"] for r in rows])) if rows else 0.0,
                  "langs": {l: sum(1 for r in rows if r["lang"] == l) for l in {r["lang"] for r in rows}}})
    write_json(stats, out / "synth_stats.json")
    # human verification sample (plan: 1000 rows, usable-rate threshold 70%)
    sample = rows[: int(sc.get("human_check", 1000))]
    with open(out / "human_check.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lang", "query", "rewrite", "intent", "valid(1/0)", "error_type(meaning_changed/hallucinated_fact/lost_constraint/other)"])
        for r in sample:
            w.writerow([r["id"], r["lang"], r["query"], r["rewrite"], r["intent"], "", ""])
    log.info("synth stats: %s -> %s", stats, out)


def cmd_validate(path: str, threshold: float = 0.7) -> None:
    labels, errs = [], {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("valid(1/0)") or "").strip()
            if v in ("0", "1"):
                labels.append(int(v))
                if v == "0":
                    e = (row.get("error_type(meaning_changed/hallucinated_fact/lost_constraint/other)") or "other").strip() or "other"
                    errs[e] = errs.get(e, 0) + 1
    if not labels:
        raise SystemExit("no labelled rows")
    rate = sum(labels) / len(labels)
    lo = rate - 1.96 * (rate * (1 - rate) / len(labels)) ** 0.5
    print(f"labelled {len(labels)}  usable rate {rate:.1%} (95% CI lower {lo:.1%})  errors {errs}")
    print("PASS" if rate >= threshold else f"FAIL (< {threshold:.0%}): change the prompt or the teacher model")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", required=True)
    b.add_argument("overrides", nargs="*")
    v = sub.add_parser("validate")
    v.add_argument("--file", required=True)
    v.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()
    if args.cmd == "build":
        cmd_build(load_config(args.config, args.overrides))
    else:
        cmd_validate(args.file, args.threshold)


if __name__ == "__main__":
    main()
