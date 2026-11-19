"""LLM-as-judge retrieval relevance + agreement with human labels (Cohen's kappa).

1) judge:  python -m mmsearch.eval.llm_judge judge --run-dir <bench> --method colqwen2-v1.0 --n-queries 200
           -> judge/judgments.jsonl  +  judge/human_annotation.csv (fill the `human_label` column: 0/1/2)
2) agree:  python -m mmsearch.eval.llm_judge agree --run-dir <bench>
           -> Cohen's kappa (binary & quadratic-weighted), confusion matrix, judge-vs-qrels agreement
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np

from ..data.schema import load_dataset
from ..llm.client import VLMClient, user_message
from ..utils.misc import markdown_table, read_jsonl, write_json, write_jsonl

JUDGE_PROMPT = """You are grading a document search engine.
Query: {query}

Grade how relevant the document page shown above is to the query:
2 = the page directly contains the answer / information the query asks for
1 = the page is on the right topic but does not contain the specific answer
0 = the page is unrelated
Respond with JSON only: {{"score": <0|1|2>, "reason": "<one short sentence>"}}"""


def parse_score(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r'"score"\s*:\s*([012])', text) or re.search(r"\b([012])\b", text)
    return int(m.group(1)) if m else None


def cohen_kappa(a: list[int], b: list[int], labels: list[int] | None = None, weights: str | None = None) -> float:
    """Cohen's kappa; weights=None (nominal) | 'linear' | 'quadratic'."""
    labels = labels or sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    O = np.zeros((n, n))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    if O.sum() == 0:
        return 0.0
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    if weights is None:
        W = 1 - np.eye(n)
    else:
        d = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) / max(1, n - 1)
        W = d if weights == "linear" else d**2
    denom = (W * E).sum()
    return float(1 - (W * O).sum() / denom) if denom > 0 else 1.0


def cmd_judge(args) -> None:
    run_dir = Path(args.run_dir)
    rows = [r for r in read_jsonl(run_dir / "per_query.jsonl") if r["method"] == args.method]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n_queries]
    corpora: dict[str, object] = {}
    pairs = []
    for r in rows:
        if r["dataset"] not in corpora:
            corpora[r["dataset"]] = load_dataset(r["dataset"])
        ds = corpora[r["dataset"]]
        docs = list(dict.fromkeys(r["top"][: args.top] + r["gold"][:1]))
        for d in docs:
            pairs.append({"pair_id": f"{r['dataset']}|{r['query_id']}|{d}", "dataset": r["dataset"], "query_id": r["query_id"],
                          "query": r["query"], "doc_id": d, "image": ds.corpus[d].image, "qrel": int(ds.qrels[r["query_id"]].get(d, 0))})
    client = VLMClient(args.base_url, args.model, max_concurrency=args.concurrency)
    msgs = [[user_message(JUDGE_PROMPT.format(query=p["query"]), [p["image"]], args.max_pixels)] for p in pairs]
    outs = client.chat_text(msgs, desc="llm judge", max_tokens=96)
    for p, o in zip(pairs, outs):
        p["judge_raw"] = o
        p["judge"] = parse_score(o)
    out = run_dir / "judge"
    write_jsonl(pairs, out / "judgments.jsonl")
    with open(out / "human_annotation.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "dataset", "query", "image", "human_label"])
        for p in random.Random(args.seed + 1).sample(pairs, len(pairs)):  # shuffled; judge labels hidden to avoid anchoring
            w.writerow([p["pair_id"], p["dataset"], p["query"], p["image"], ""])
    print(f"{len(pairs)} pairs judged -> {out}")


def cmd_agree(args) -> None:
    out = Path(args.run_dir) / "judge"
    judg = {p["pair_id"]: p for p in read_jsonl(out / "judgments.jsonl")}
    human: dict[str, int] = {}
    csv_path = Path(args.human_csv) if args.human_csv else out / "human_annotation.csv"
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("human_label", "").strip() in ("0", "1", "2"):
                human[row["pair_id"]] = int(row["human_label"])
    res: dict[str, object] = {}
    both = [(judg[k]["judge"], h) for k, h in human.items() if k in judg and judg[k]["judge"] is not None]
    if both:
        j, h = [x[0] for x in both], [x[1] for x in both]
        res["n_human"] = len(both)
        res["kappa_3way"] = cohen_kappa(j, h, [0, 1, 2])
        res["kappa_quadratic"] = cohen_kappa(j, h, [0, 1, 2], "quadratic")
        res["kappa_binary(>=2)"] = cohen_kappa([int(x >= 2) for x in j], [int(x >= 2) for x in h], [0, 1])
        res["raw_agreement"] = float(np.mean([a == b for a, b in both]))
        cm = np.zeros((3, 3), int)
        for a, b in both:
            cm[b, a] += 1
        res["confusion_human_rows_judge_cols"] = cm.tolist()
    jq = [(p["judge"], p["qrel"]) for p in judg.values() if p["judge"] is not None]
    if jq:
        res["n_qrels"] = len(jq)
        res["kappa_judge_vs_qrels(binary >=2)"] = cohen_kappa([int(a >= 2) for a, _ in jq], [b for _, b in jq], [0, 1])
        # pages the judge rates 2 although not in qrels = candidate false negatives in the benchmark labels
        res["judge2_not_in_qrels"] = int(sum(1 for a, b in jq if a == 2 and b == 0))
    write_json(res, out / "agreement.json")
    print(json.dumps(res, indent=2))
    print(markdown_table([{k: v for k, v in res.items() if not isinstance(v, list)}], digits=3))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--run-dir", required=True)
    j.add_argument("--method", required=True)
    j.add_argument("--n-queries", type=int, default=200)
    j.add_argument("--top", type=int, default=3)
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--base-url", default=None)
    j.add_argument("--model", default=None)
    j.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    j.add_argument("--concurrency", type=int, default=32)
    a = sub.add_parser("agree")
    a.add_argument("--run-dir", required=True)
    a.add_argument("--human-csv", default=None)
    args = ap.parse_args()
    {"judge": cmd_judge, "agree": cmd_agree}[args.cmd](args)


if __name__ == "__main__":
    main()
