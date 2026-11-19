"""Closed-loop load test: QPS vs P50/P99 latency at increasing concurrency (plan: 固定硬件下测 QPS-P99 曲线).

python -m mmsearch.service.loadtest --url http://127.0.0.1:8080 --dataset vidore_docvqa --concurrency 1 2 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from pathlib import Path

import httpx
import numpy as np

from ..data.schema import load_dataset
from ..utils.misc import markdown_table, write_json
from ..utils.paths import runs_dir


async def run_level(url: str, endpoint: str, queries: list[str], concurrency: int, duration: float, payload: dict) -> dict:
    lat: list[float] = []
    errors = 0
    stop = time.perf_counter() + duration
    rng = random.Random(concurrency)

    async def worker(client: httpx.AsyncClient):
        nonlocal errors
        while time.perf_counter() < stop:
            t0 = time.perf_counter()
            try:
                r = await client.post(f"{url}{endpoint}", json={"query": rng.choice(queries), **payload})
                r.raise_for_status()
                lat.append((time.perf_counter() - t0) * 1000)
            except Exception:  # noqa: BLE001
                errors += 1

    t_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=120, limits=httpx.Limits(max_connections=concurrency * 2)) as client:
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
    wall = time.perf_counter() - t_start
    a = np.asarray(lat) if lat else np.asarray([float("nan")])
    return {"concurrency": concurrency, "requests": len(lat), "errors": errors, "qps": len(lat) / wall,
            "p50_ms": float(np.percentile(a, 50)), "p90_ms": float(np.percentile(a, 90)), "p99_ms": float(np.percentile(a, 99))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--endpoint", default="/search")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--mode", default="multivector")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    queries = [q.text for q in load_dataset(args.dataset).queries.values() if q.text]
    payload = {"k": args.k, "mode": args.mode} if args.endpoint == "/search" else {"k": min(args.k, 5)}
    rows = []
    for c in args.concurrency:
        r = asyncio.run(run_level(args.url, args.endpoint, queries, c, args.duration, payload))
        rows.append(r)
        print(r)
    out = Path(args.out or runs_dir() / "loadtest" / f"{args.dataset}_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}")
    write_json(rows, out / "loadtest.json")
    md = markdown_table(rows, digits=1)
    (out / "loadtest.md").write_text(md, encoding="utf-8")
    print(md)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["qps"] for r in rows], [r["p99_ms"] for r in rows], "o-", label="P99")
        ax.plot([r["qps"] for r in rows], [r["p50_ms"] for r in rows], "s--", label="P50")
        for r in rows:
            ax.annotate(f"c={r['concurrency']}", (r["qps"], r["p99_ms"]), fontsize=7)
        ax.set_xlabel("QPS")
        ax.set_ylabel("latency (ms)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "qps_latency.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
