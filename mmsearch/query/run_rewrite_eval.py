"""E6: evaluate query rewriting by *downstream retrieval*, with a length-matched random-expansion control.

python -m mmsearch.query.run_rewrite_eval --config configs/query_rewrite.yaml

For each test dataset × language (en + translated zh/ja) and each rewriter (zero-shot teacher, fine-tuned student):
  original | replace (rewrite only) | concat (original + rewrite) | entities | rand_concat (original + random words,
  same length as concat). "Gain comes from semantic completion, not length" <=> concat >> rand_concat.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from ..data.schema import Query, load_dataset
from ..eval.metrics import evaluate_run, paired_permutation_test
from ..eval.run_benchmark import expand_datasets
from ..llm.client import VLMClient
from ..retrieve import build_retriever
from ..utils.config import load_config, save_config
from ..utils.misc import get_logger, markdown_table, read_jsonl, write_json, write_jsonl
from ..utils.paths import runs_dir
from .rewriter import QueryRewriter, RewriteResult, compose, random_expansion
from .translate import translate_dataset

log = get_logger(__name__)


def cached_rewrites(path: Path, rw: QueryRewriter, queries: dict[str, Query]) -> dict[str, RewriteResult | None]:
    cache = {r["query_id"]: r for r in read_jsonl(path)} if path.exists() else {}
    todo = [q for q in queries if q not in cache]
    if todo:
        res = rw.rewrite_many([queries[q].text or "" for q in todo], desc=f"rewrite {path.stem}")
        rows = [{"query_id": q, "query": queries[q].text, **(r[0].__dict__ if r[0] else {"rewrite": None})} for q, r in zip(todo, res)]
        write_jsonl(rows, path, append=True)
        cache.update({r["query_id"]: r for r in rows})
    out = {}
    for q in queries:
        r = cache.get(q)
        out[q] = RewriteResult(r["rewrite"], r.get("intent", "other"), r.get("entities", []), r.get("lang", "en")) if r and r.get("rewrite") else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    ec = cfg.eval
    out = Path(ec.get("out_dir") or runs_dir() / "query" / f"{cfg.name}_eval_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    metrics = list(ec.get("metrics", ["ndcg@5", "recall@10"]))
    retr = build_retriever(dict(ec.get("retriever") or cfg.reward_retriever))
    tcfg = cfg.teacher
    teacher_client = VLMClient(tcfg.get("base_url"), tcfg.get("model"), max_concurrency=64)
    rewriters = {r["name"]: QueryRewriter(VLMClient(r.get("base_url"), r.get("model"), max_concurrency=64)) for r in ec.rewriters}
    modes = list(ec.get("modes", ["replace", "concat"]))
    rng = random.Random(0)

    rows, per_query = [], {}  # per_query[(lang, cond)] -> list of (key, value) for significance tests
    for name in expand_datasets(list(ec.datasets)):
        for lang in ec.get("langs", ["en"]):
            ds = load_dataset(name) if lang == "en" else translate_dataset(name, lang, teacher_client)
            ds = ds.subset_queries(ec.get("max_queries"), seed=0)
            retr.index(ds)
            conds: dict[str, dict[str, Query]] = {"original": ds.queries}
            for rname, rw in rewriters.items():
                rws = cached_rewrites(out / "rewrites" / f"{rname}__{ds.name}.jsonl", rw, ds.queries)
                vocab = [w for r in rws.values() if r for w in r.rewrite.split()]
                for m in modes:
                    conds[f"{rname}:{m}"] = {q: Query(q, compose(v.text or "", rws[q], m), None, v.lang) for q, v in ds.queries.items()}
                if "random_expansion" in ec.get("controls", ["random_expansion"]):
                    conds[f"{rname}:rand_concat"] = {
                        q: Query(q, random_expansion(v.text or "", len(compose(v.text or "", rws[q], "concat").split()), vocab, rng), None, v.lang)
                        for q, v in ds.queries.items()}
            for cond, qs in conds.items():
                run = retr.search(qs, 100)
                agg, pq = evaluate_run(run, ds.qrels, metrics)
                avg_words = float(np.mean([len((q.text or "").split()) for q in qs.values()]))
                rows.append({"dataset": name, "lang": lang, "condition": cond, **{m: agg[m] * 100 for m in metrics}, "avg_words": avg_words})
                for qid, v in pq.items():
                    per_query.setdefault((lang, cond), {})[f"{name}|{qid}"] = v
                log.info("%s/%s %s %s", name, lang, cond, {m: round(agg[m] * 100, 2) for m in metrics})

    write_json(rows, out / "results.json")
    # summary: mean over datasets, delta vs original, paired permutation p-value (pooled queries)
    summ = []
    main_m = metrics[0]
    for lang in dict.fromkeys(r["lang"] for r in rows):
        base_pq = per_query[(lang, "original")]
        base_mean = {m: np.mean([r[m] for r in rows if r["lang"] == lang and r["condition"] == "original"]) for m in metrics}
        for cond in dict.fromkeys(r["condition"] for r in rows if r["lang"] == lang):
            rs = [r for r in rows if r["lang"] == lang and r["condition"] == cond]
            s = {"lang": lang, "condition": cond, **{m: float(np.mean([r[m] for r in rs])) for m in metrics},
                 "avg_words": float(np.mean([r["avg_words"] for r in rs]))}
            for m in metrics:
                s[f"Δ{m}"] = s[m] - base_mean[m]
            if cond != "original":
                pq = per_query[(lang, cond)]
                keys = sorted(set(pq) & set(base_pq))
                s[f"p({main_m})"] = paired_permutation_test([pq[k][main_m] for k in keys], [base_pq[k][main_m] for k in keys], n=5000)
            summ.append(s)
    cols = ["lang", "condition", *metrics, *[f"Δ{m}" for m in metrics], f"p({main_m})", "avg_words"]
    md = "# Query rewriting — downstream retrieval (mean over datasets)\n\n" + markdown_table(summ, cols, 2) + "\n"
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    log.info("-> %s", out)


if __name__ == "__main__":
    main()
