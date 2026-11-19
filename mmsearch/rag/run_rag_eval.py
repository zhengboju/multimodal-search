"""Stage 3: end-to-end multimodal RAG evaluation with failure attribution and top-k sensitivity.

python -m mmsearch.rag.run_rag_eval --config configs/rag_e2e.yaml
(needs a VLM server: bash scripts/serve_vlm.sh Qwen/Qwen2.5-VL-7B-Instruct 8000)

Conditions per query: retrieval@k for every k in `ks`, `oracle` (gold page only), `closed_book` (no image).
Attribution at k* (each query falls in exactly one bucket):
  correct                hit & correct
  correct_without_hit    miss & correct       (answer on another page / prior knowledge / label noise)
  retrieval_miss         miss & wrong & oracle correct   -> fixable by retrieval
  context_distraction    hit  & wrong & oracle correct   -> extra pages hurt the generator (context organisation)
  generation_error       hit  & wrong & oracle wrong     -> VLM can't read/understand the page
  both_fail              miss & wrong & oracle wrong     -> hard question / data issue
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..data.schema import load_dataset
from ..eval.metrics import ranked_docs
from ..eval.plots import plot_attribution, plot_k_sensitivity
from ..llm.client import VLMClient
from ..retrieve import build_retriever
from ..utils.config import load_config, save_config
from ..utils.misc import get_logger, markdown_table, write_json, write_jsonl
from ..utils.paths import runs_dir
from .answer_metrics import metric_for, score
from .pipeline import RAGConfig, RAGGenerator

log = get_logger(__name__)
CORRECT_THRESHOLD = 0.5


def attribute(hit: bool, correct: bool, oracle_correct: bool) -> str:
    if correct:
        return "correct" if hit else "correct_without_hit"
    if hit:
        return "context_distraction" if oracle_correct else "generation_error"
    return "retrieval_miss" if oracle_correct else "both_fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    out = Path(cfg.get("out_dir") or runs_dir() / "rag" / f"{cfg.name}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    ds = load_dataset(cfg.dataset).subset_queries(cfg.get("max_queries"), seed=int(cfg.get("seed", 0)))
    ds.queries = {k: v for k, v in ds.queries.items() if v.answers}
    if not ds.queries:
        raise SystemExit(f"{cfg.dataset} has no reference answers (ViDoRe docvqa/infovqa test answers are hidden; use docvqa_validation)")
    metric = cfg.get("metric", "auto")
    metric = metric_for(ds.name, ds.info.get("domain", "")) if metric == "auto" else metric
    ks = sorted(int(k) for k in cfg.ks)
    kmax = max(ks)
    log.info("%s | metric=%s | ks=%s", ds, metric, ks)

    # 1) retrieval once at depth max(k) (+ optional VLM rerank)
    retr = build_retriever(dict(cfg.retriever))
    retr.index(ds)
    rr_cfg = cfg.get("rerank") or {}
    run = retr.search(ds.queries, max(kmax, int(rr_cfg.get("top_n", 0)), 20))
    if rr_cfg:
        from ..retrieve.rerank import VLMReranker

        run = VLMReranker(top_n=int(rr_cfg["top_n"])).rerank_run(run, ds.queries, ds)
    rankings = {qid: ranked_docs(run.get(qid, {})) for qid in ds.queries}

    # 2) generation for every condition
    gc = cfg.get("generator") or {}
    client = VLMClient(gc.get("base_url"), gc.get("model"), max_concurrency=int(gc.get("max_concurrency", 32)))
    b = cfg.get("budget") or {}
    rag_cfg = RAGConfig(total_tokens=int(b.get("total_tokens", 6144)), strategy=b.get("strategy", "uniform"),
                        min_tokens=int(b.get("min_tokens", 256)), max_tokens=int(b.get("max_tokens", 1280)),
                        max_new_tokens=int(gc.get("max_new_tokens", 64)))
    gen = RAGGenerator(client, rag_cfg)
    qids = list(ds.queries)
    conditions: dict[str, list[list[str]]] = {f"k{k}": [[ds.corpus[d].image for d in rankings[q][:k]] for q in qids] for k in ks}
    if (cfg.get("conditions") or {}).get("oracle", True):
        conditions["oracle"] = [[ds.corpus[d].image for d, r in ds.qrels[q].items() if r > 0][:1] for q in qids]
    if (cfg.get("conditions") or {}).get("closed_book", True):
        conditions["closed_book"] = [[] for _ in qids]

    per_q: list[dict] = []
    results: dict[str, dict[str, float]] = {}
    scores: dict[str, dict[str, float]] = {}
    for cond, imgs in conditions.items():
        preds = gen.answer_many([(ds.queries[q].text or "", im) for q, im in zip(qids, imgs)], desc=f"generate {cond}")
        sc = {}
        for q, p, im in zip(qids, preds, imgs):
            s = score(p, ds.queries[q].answers, metric)
            sc[q] = s
            k = int(cond[1:]) if cond.startswith("k") else None
            hit = any(ds.qrels[q].get(d, 0) > 0 for d in rankings[q][:k]) if k else None
            per_q.append({"condition": cond, "query_id": q, "query": ds.queries[q].text, "answers": ds.queries[q].answers,
                          "pred": p, "score": s, "hit": hit, "n_images": len(im)})
        scores[cond] = sc
        results[cond] = {"accuracy": float(np.mean(list(sc.values()))),
                         "correct_rate": float(np.mean([v >= CORRECT_THRESHOLD for v in sc.values()]))}
        log.info("%s: %s=%.4f", cond, metric, results[cond]["accuracy"])

    # 3) k-sensitivity
    oracle_acc = results.get("oracle", {}).get("accuracy")
    ksens = []
    for k in ks:
        hit = float(np.mean([any(ds.qrels[q].get(d, 0) > 0 for d in rankings[q][:k]) for q in qids]))
        row = {"k": k, "accuracy": results[f"k{k}"]["accuracy"], "hit": hit, "visual_tokens_budget": rag_cfg.total_tokens}
        if oracle_acc is not None:
            row["oracle"] = oracle_acc
        ksens.append(row)
    plot_k_sensitivity(ksens, out / "k_sensitivity.png", f"{ds.name}: {metric} vs k")

    # 4) attribution at k*
    k_star = int(cfg.get("attribution_k", ks[len(ks) // 2]))
    att_rows, counts = [], Counter()
    if "oracle" in scores and f"k{k_star}" in scores:
        for q in qids:
            hit = any(ds.qrels[q].get(d, 0) > 0 for d in rankings[q][:k_star])
            c = scores[f"k{k_star}"][q] >= CORRECT_THRESHOLD
            oc = scores["oracle"][q] >= CORRECT_THRESHOLD
            bucket = attribute(hit, c, oc)
            counts[bucket] += 1
            att_rows.append({"query_id": q, "bucket": bucket})
        plot_attribution(dict(counts), out / f"attribution_k{k_star}.png", f"{ds.name} failure attribution (k={k_star})")
    errors = sum(v for k, v in counts.items() if not k.startswith("correct"))
    att_summary = {k: {"n": v, "share_all%": 100 * v / max(1, len(qids)), "share_errors%": 100 * v / max(1, errors) if not k.startswith("correct") else None}
                   for k, v in counts.items()}

    # 5) report
    write_jsonl(per_q, out / "per_query.jsonl")
    write_jsonl(att_rows, out / "attribution.jsonl")
    write_json({"dataset": ds.name, "metric": metric, "retriever": retr.name, "results": results, "k_sensitivity": ksens,
                "attribution_k": k_star, "attribution": att_summary}, out / "results.json")
    md = [f"# RAG — {ds.name} ({len(qids)} queries, metric {metric}, retriever {retr.name})", "",
          "## Accuracy by condition", "",
          markdown_table([{"condition": c, f"{metric}": r["accuracy"] * 100, "correct%": r["correct_rate"] * 100} for c, r in results.items()], digits=2),
          "", "## k sensitivity", "", markdown_table([{**r, "accuracy": r["accuracy"] * 100, "hit": r["hit"] * 100} for r in ksens], digits=2),
          "", f"## Failure attribution @k={k_star}", "",
          markdown_table([{"bucket": k, **v} for k, v in sorted(att_summary.items(), key=lambda x: -x[1]["n"])], digits=1)]
    if errors:
        retr_share = 100 * counts.get("retrieval_miss", 0) / errors
        gen_share = 100 * (counts.get("generation_error", 0) + counts.get("context_distraction", 0)) / errors
        md += ["", f"retrieval-attributable errors: **{retr_share:.1f}%**, generation-side: **{gen_share:.1f}%** "
                   f"(of which context distraction {100 * counts.get('context_distraction', 0) / errors:.1f}%)"]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    log.info("-> %s", out)


if __name__ == "__main__":
    main()
