"""Config-driven retrieval benchmark: retrievers × datasets -> metrics, latency, index size, per-query dumps.

python -m mmsearch.eval.run_benchmark --config configs/vidore_benchmark.yaml
python -m mmsearch.eval.run_benchmark --config configs/benchmark_all.yaml     # suite: one command for everything

Retriever entries (order matters; fusion/rerank reference earlier names):
  {name, type: multivector, encoder: {...}, index: {...}, compress: [...]}
  {name, type: bm25}
  {name, type: dense, model: openai/clip-vit-large-patch14}
  {name, type: fusion, members: [a, b], method: rrf|weighted|zscore, weights: [..]}
  {name, type: rerank, base: a, top_n: 10}
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..data.schema import RetrievalDataset, load_dataset
from ..data.preprocess import VIDORE_V1
from ..retrieve import build_retriever
from ..retrieve.base import Run
from ..retrieve.fusion import fuse, truncate
from ..utils.config import Cfg, load_config, save_config
from ..utils.misc import get_logger, markdown_table, write_json, write_jsonl
from ..utils.paths import repo_root, runs_dir
from .ledger import record_test_eval
from .metrics import DEFAULT_METRICS, evaluate_run, gold_rank, ranked_docs

log = get_logger(__name__)

DATASET_ALIASES = {"vidore_all": [f"vidore_{k}" for k in VIDORE_V1]}


def expand_datasets(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        out.extend(DATASET_ALIASES.get(n, [n]))
    return out


def save_run(run: Run, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(run, f)


def load_run(path: Path) -> Run:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def per_query_rows(ds: RetrievalDataset, method: str, run: Run, per_q: dict[str, dict[str, float]], topn: int = 10) -> list[dict]:
    rows = []
    for qid, m in per_q.items():
        ranking = ranked_docs(run.get(qid, {}))
        q = ds.queries[qid]
        rows.append({
            "dataset": ds.name, "method": method, "query_id": qid, "query": q.text, "lang": q.lang,
            "domain": q.meta.get("domain", ds.info.get("domain", "")),
            "gold": [d for d, r in ds.qrels[qid].items() if r > 0],
            "gold_rank": gold_rank(ranking, ds.qrels[qid]),
            "top": ranking[:topn], **m,
        })
    return rows


class BenchmarkRunner:
    def __init__(self, cfg: Cfg, out_dir: Path):
        self.cfg = cfg
        self.out = out_dir
        self.k = int(cfg.get("k", 100))
        self.metrics = list(cfg.get("metrics", DEFAULT_METRICS))
        self.latency_cfg = cfg.get("latency", {"enabled": True, "n": 100})
        self.rows: list[dict] = []
        self._retrievers: dict[str, Any] = {}

    def _get_retriever(self, spec: dict):
        name = spec["name"]
        if name not in self._retrievers:
            self._retrievers[name] = build_retriever({k: v for k, v in spec.items() if k not in ("skip_datasets", "optional")})
        return self._retrievers[name]

    def run_dataset(self, ds_name: str) -> None:
        ds = load_dataset(ds_name).subset_queries(self.cfg.get("max_queries"), seed=int(self.cfg.get("seed", 0)))
        log.info("=== %s", ds)
        runs: dict[str, Run] = {}
        pq_rows: list[dict] = []
        for spec in self.cfg.retrievers:
            name = spec["name"]
            if ds_name in spec.get("skip_datasets", []):
                continue
            deps = spec.get("members", []) + ([spec["base"]] if "base" in spec else [])
            if any(d not in runs for d in deps):
                log.warning("skip %s: missing member runs %s", name, [d for d in deps if d not in runs])
                continue
            try:
                self._run_one(spec, ds, runs, pq_rows)
            except Exception as e:  # noqa: BLE001
                if not spec.get("optional", False):
                    raise
                log.warning("optional retriever %s failed on %s: %r — skipped", name, ds.name, e)
        write_jsonl(pq_rows, self.out / "per_query.jsonl", append=True)

    def _run_one(self, spec: dict, ds: RetrievalDataset, runs: dict[str, Run], pq_rows: list[dict]) -> None:
        name, typ = spec["name"], spec["type"]
        extra: dict[str, Any] = {}
        t0 = time.perf_counter()
        if typ == "fusion":
            member_runs = [runs[m] for m in spec["members"]]
            kw = {k: v for k, v in spec.items() if k not in ("name", "type", "members", "method", "weights", "optional", "skip_datasets")}
            run = truncate(fuse(member_runs, spec.get("method", "rrf"), spec.get("weights"), **kw), self.k)
        elif typ == "rerank":
            from ..retrieve.rerank import VLMReranker

            rr = VLMReranker(top_n=int(spec.get("top_n", 10)), max_pixels=spec.get("max_pixels", 768 * 28 * 28),
                             interpolate=float(spec.get("interpolate", 0.0)))
            run = truncate(rr.rerank_run(runs[spec["base"]], ds.queries, ds), self.k)
            extra["rerank_s_per_query"] = (time.perf_counter() - t0) / max(1, len(ds.queries))
        else:
            r = self._get_retriever(spec)
            r.index(ds)
            t0 = time.perf_counter()
            run = r.search(ds.queries, self.k)
            extra["batch_search_s"] = time.perf_counter() - t0
            extra.update({k: v for k, v in r.stats().items() if isinstance(v, (int, float, str))})
            if self.latency_cfg.get("enabled", True):
                n = int(self.latency_cfg.get("n", 100))
                if self.latency_cfg.get("e2e", True):  # query encoding + search, batch 1
                    extra.update(r.measure_latency(ds.queries, k=10, n=n))
                if hasattr(r, "measure_search_latency"):  # index-only (precomputed query embeddings)
                    extra.update(r.measure_search_latency(ds.queries, k=10, n=max(n, 200)))
        runs[name] = run
        agg, per_q = evaluate_run(run, ds.qrels, self.metrics)
        row = {"dataset": ds.name, "method": name, "n_queries": len(per_q), "n_docs": len(ds.corpus), **agg, **extra}
        self.rows.append(row)
        pq_rows.extend(per_query_rows(ds, name, run, per_q))
        save_run(run, self.out / "runs" / ds.name / f"{name}.json.gz")
        if ds.info.get("split", "test") == "test":
            record_test_eval(ds.name, name, agg)
        log.info("%s | %s | %s", ds.name, name, {m: round(agg[m], 4) for m in self.metrics})

    def report(self) -> str:
        main = self.cfg.get("main_metric", self.metrics[0])
        write_json(self.rows, self.out / "results.json")
        methods = list(dict.fromkeys(r["method"] for r in self.rows))
        datasets = list(dict.fromkeys(r["dataset"] for r in self.rows))
        by = {(r["method"], r["dataset"]): r for r in self.rows}
        pivot = []
        for m in methods:
            row = {"method": m}
            vals = []
            for d in datasets:
                if (m, d) in by:
                    v = by[(m, d)][main] * 100
                    row[d.replace("vidore_", "")] = v
                    vals.append(v)
            row["avg"] = sum(vals) / len(vals) if vals else 0.0
            pivot.append(row)
        cols = ["method", *[d.replace("vidore_", "") for d in datasets], "avg"]
        md = [f"# {self.cfg.get('name', 'benchmark')}", "", f"## {main} (×100)", "", markdown_table(pivot, cols, 1)]
        # averaged metrics + efficiency per method
        eff_keys = ["latency_p50_ms", "latency_p99_ms", "search_p50_ms", "search_p99_ms", "index_mb", "total_mb", "store_size_mb"]
        summ = []
        for m in methods:
            rs = [r for r in self.rows if r["method"] == m]
            s = {"method": m}
            for k in self.metrics + eff_keys:
                vals = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
                if vals:
                    s[k] = sum(vals) / len(vals) * (100 if k in self.metrics else 1)
            summ.append(s)
        cols2 = ["method"] + [k for k in self.metrics + eff_keys if any(k in s for s in summ)]
        md += ["", "## Mean over datasets (metrics ×100, latency ms, size MB)", "", markdown_table(summ, cols2, 2)]
        text = "\n".join(md) + "\n"
        (self.out / "results.md").write_text(text, encoding="utf-8")
        return text


def run_retrieval_config(cfg: Cfg, out_dir: Path | None = None) -> Path:
    out = Path(out_dir or runs_dir() / "benchmark" / f"{cfg.get('name', 'bench')}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    runner = BenchmarkRunner(cfg, out)
    for d in expand_datasets(list(cfg.datasets)):
        runner.run_dataset(d)
    print(runner.report())
    log.info("results -> %s", out)
    return out


def run_suite(cfg: Cfg) -> None:
    """`suite:` entries: {kind: retrieval|module, config|args}. Modules run as subprocesses (fresh GPU memory)."""
    root = repo_root()
    for item in cfg.suite:
        kind = item.get("kind", "retrieval")
        log.info(">>> suite step: %s", item)
        if kind == "retrieval":
            sub = load_config(root / item["config"], item.get("overrides"))
            run_retrieval_config(sub)
        elif kind == "module":
            cmd = [sys.executable, "-m", item["module"], *[str(a) for a in item.get("args", [])]]
            subprocess.run(cmd, check=True, cwd=root)
        else:
            raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*", help="key=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    if cfg.get("suite"):
        run_suite(cfg)
    else:
        run_retrieval_config(cfg, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
