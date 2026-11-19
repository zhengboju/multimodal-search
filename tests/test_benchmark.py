from mmsearch.eval.badcase import analyse
from mmsearch.eval.run_benchmark import BenchmarkRunner
from mmsearch.utils.config import Cfg
from mmsearch.utils.misc import read_jsonl

from .test_pipeline_units import tiny_dataset


def test_benchmark_runner_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MMS_STORAGE", str(tmp_path / "storage"))
    ds = tiny_dataset(tmp_path)
    ds.save(tmp_path / "tiny")
    cfg = Cfg({
        "name": "t", "k": 10, "metrics": ["ndcg@5", "recall@1"], "latency": {"enabled": True, "n": 2},
        "retrievers": [
            {"name": "bm25", "type": "bm25"},
            {"name": "bm25b", "type": "bm25", "k1": 0.9},
            {"name": "fused", "type": "fusion", "members": ["bm25", "bm25b"], "method": "rrf"},
            {"name": "broken", "type": "multivector", "encoder": {"checkpoint": str(tmp_path / "missing")}, "optional": True},
            {"name": "fused2", "type": "fusion", "members": ["bm25", "broken"]},
        ],
    })
    out = tmp_path / "out"
    out.mkdir()
    r = BenchmarkRunner(cfg, out)
    r.run_dataset(str(tmp_path / "tiny"))
    text = r.report()
    methods = {row["method"] for row in r.rows}
    assert methods == {"bm25", "bm25b", "fused"}  # optional failure + dependent fusion skipped
    assert all(row["recall@1"] == 1.0 for row in r.rows)
    assert "latency_p50_ms" in r.rows[0]
    assert "ndcg@5" in text
    pq = read_jsonl(out / "per_query.jsonl")
    assert len(pq) == 9
    md = analyse(pq, "bm25", "ndcg@5", "fused", n_examples=3, n_clusters=2)
    assert "Gold-rank buckets" in md
