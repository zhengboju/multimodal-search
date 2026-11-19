"""IR metrics (self-contained, numerically matching pytrec_eval conventions: linear gain nDCG, ties broken by doc id).

Metric names: ndcg@k, recall@k, hit@k (a.k.a. success / "R@K" in image-text retrieval papers), mrr@k, map@k, precision@k.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

Run = dict[str, dict[str, float]]
Qrels = dict[str, dict[str, int]]

DEFAULT_METRICS = ["ndcg@5", "ndcg@10", "recall@1", "recall@5", "recall@10", "mrr@10"]


def ranked_docs(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def _parse(metric: str) -> tuple[str, int]:
    name, _, k = metric.partition("@")
    return name.lower(), int(k) if k else 1000


def query_metrics(ranking: list[str], rels: dict[str, int], metrics: Iterable[str]) -> dict[str, float]:
    relevant = {d: r for d, r in rels.items() if r > 0}
    out: dict[str, float] = {}
    n_rel = len(relevant)
    for m in metrics:
        name, k = _parse(m)
        top = ranking[:k]
        hits = [1 if d in relevant else 0 for d in top]
        if name == "ndcg":
            dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(top))
            ideal = sorted(relevant.values(), reverse=True)[:k]
            idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
            out[m] = dcg / idcg if idcg > 0 else 0.0
        elif name == "recall":
            out[m] = sum(hits) / n_rel if n_rel else 0.0
        elif name in ("hit", "success"):
            out[m] = 1.0 if any(hits) else 0.0
        elif name in ("mrr", "rr"):
            out[m] = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
        elif name == "precision":
            out[m] = sum(hits) / k
        elif name == "map":
            s, c = 0.0, 0
            for i, h in enumerate(hits):
                if h:
                    c += 1
                    s += c / (i + 1)
            out[m] = s / min(n_rel, k) if n_rel else 0.0
        else:
            raise ValueError(f"unknown metric {m}")
    return out


def gold_rank(ranking: list[str], rels: dict[str, int]) -> int | None:
    """1-based rank of the first relevant doc (None if not retrieved)."""
    for i, d in enumerate(ranking):
        if rels.get(d, 0) > 0:
            return i + 1
    return None


def evaluate_run(run: Run, qrels: Qrels, metrics: Iterable[str] = DEFAULT_METRICS) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Returns (mean over judged queries, per-query metrics). Queries missing from the run score 0."""
    metrics = list(metrics)
    per_q: dict[str, dict[str, float]] = {}
    for qid, rels in qrels.items():
        if not any(r > 0 for r in rels.values()):
            continue
        per_q[qid] = query_metrics(ranked_docs(run.get(qid, {})), rels, metrics)
    agg = {m: float(np.mean([v[m] for v in per_q.values()])) if per_q else 0.0 for m in metrics}
    return agg, per_q


def crosscheck_pytrec(run: Run, qrels: Qrels, k: int = 10) -> dict[str, float] | None:
    """Optional sanity check against pytrec_eval (pip install pytrec-eval-terrier)."""
    try:
        import pytrec_eval
    except ImportError:
        return None
    ev = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}", f"recall.{k}"})
    res = ev.evaluate(run)
    return {
        f"ndcg@{k}": float(np.mean([r[f"ndcg_cut_{k}"] for r in res.values()])),
        f"recall@{k}": float(np.mean([r[f"recall_{k}"] for r in res.values()])),
    }


def bootstrap_ci(values: Iterable[float], n: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n, v.size))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_permutation_test(a: Iterable[float], b: Iterable[float], n: int = 10000, seed: int = 0) -> float:
    """Two-sided paired randomisation test p-value for mean(a) != mean(b) (same queries, same order)."""
    a, b = np.asarray(list(a)), np.asarray(list(b))
    d = a - b
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, d.size))
    return float(((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n + 1))
