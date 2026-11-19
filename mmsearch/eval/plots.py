"""Plots (matplotlib optional): recall–size–latency trade-off with Pareto frontier, RAG k-sensitivity, attribution pie.

python -m mmsearch.eval.plots tradeoff --results <bench>/results.json --metric ndcg@5
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ..utils.misc import get_logger, read_json

log = get_logger(__name__)


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot (pip install matplotlib)")
        return None


def pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Indices of points not dominated (lower x better, higher y better), sorted by x."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best = [], float("-inf")
    for i in order:
        if points[i][1] > best:
            front.append(i)
            best = points[i][1]
    return front


def average_by_method(rows: list[dict], keys: list[str]) -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in keys:
            if isinstance(r.get(k), (int, float)):
                acc[r["method"]][k].append(float(r[k]))
    return {m: {k: sum(v) / len(v) for k, v in d.items()} for m, d in acc.items()}


def plot_tradeoff(results: str | Path, metric: str = "ndcg@5", size_key: str = "total_mb", lat_key: str = "search_p99_ms",
                  out: str | Path | None = None) -> Path | None:
    plt = _plt()
    rows = read_json(results)
    for r in rows:  # exhaustive / hierarchical report index_mb only
        r.setdefault(size_key, r.get("index_mb"))
    avg = average_by_method(rows, [metric, size_key, lat_key])
    methods = [m for m, v in avg.items() if metric in v and size_key in v and lat_key in v]
    if plt is None or not methods:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, xk, xl in ((axes[0], size_key, "index size (MB)"), (axes[1], lat_key, "search P99 latency (ms)")):
        pts = [(avg[m][xk], avg[m][metric] * 100) for m in methods]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=40)
        for m, p in zip(methods, pts):
            ax.annotate(m, p, fontsize=7, xytext=(3, 3), textcoords="offset points")
        f = pareto_front(pts)
        ax.plot([pts[i][0] for i in f], [pts[i][1] for i in f], "r--", lw=1, label="Pareto front")
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(f"{metric} ×100")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Multi-vector compression / index trade-off")
    fig.tight_layout()
    out = Path(out or Path(results).with_name(f"tradeoff_{metric.replace('@', '')}.png"))
    fig.savefig(out, dpi=150)
    log.info("saved %s", out)
    return out


def plot_k_sensitivity(rows: list[dict], out: str | Path, title: str = "RAG accuracy vs top-k") -> Path | None:
    """rows: [{k, accuracy, hit}] (hit = retrieval Hit@k)."""
    plt = _plt()
    if plt is None or not rows:
        return None
    rows = sorted(rows, key=lambda r: r["k"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["k"] for r in rows], [r["accuracy"] * 100 for r in rows], "o-", label="end-to-end accuracy")
    ax.plot([r["k"] for r in rows], [r["hit"] * 100 for r in rows], "s--", label="retrieval Hit@k")
    if any("oracle" in r for r in rows):
        ax.axhline(rows[0]["oracle"] * 100, color="gray", ls=":", label="oracle page")
    ax.set_xlabel("k (pages in prompt)")
    ax.set_ylabel("%")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return Path(out)


def plot_attribution(counts: dict[str, int], out: str | Path, title: str = "Failure attribution") -> Path | None:
    plt = _plt()
    counts = {k: v for k, v in counts.items() if v > 0}
    if plt is None or not counts:
        return None
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(list(counts.values()), labels=[f"{k}\n({v})" for k, v in counts.items()], autopct="%1.1f%%", textprops={"fontsize": 8})
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return Path(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tradeoff")
    t.add_argument("--results", required=True)
    t.add_argument("--metric", default="ndcg@5")
    t.add_argument("--size-key", default="total_mb")
    t.add_argument("--lat-key", default="search_p99_ms")
    args = ap.parse_args()
    if args.cmd == "tradeoff":
        plot_tradeoff(args.results, args.metric, args.size_key, args.lat_key)


if __name__ == "__main__":
    main()
