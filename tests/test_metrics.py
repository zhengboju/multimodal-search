import math

import pytest

from mmsearch.eval.metrics import evaluate_run, gold_rank, paired_permutation_test, query_metrics, ranked_docs


def test_basic_metrics():
    ranking = ["a", "b", "c", "d"]
    rels = {"b": 1, "d": 1}
    m = query_metrics(ranking, rels, ["recall@1", "recall@2", "hit@2", "mrr@10", "precision@2", "ndcg@4", "map@4"])
    assert m["recall@1"] == 0
    assert m["recall@2"] == 0.5
    assert m["hit@2"] == 1
    assert m["mrr@10"] == 0.5
    assert m["precision@2"] == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert m["ndcg@4"] == pytest.approx(dcg / idcg)
    assert m["map@4"] == pytest.approx((1 / 2 + 2 / 4) / 2)


def test_graded_ndcg_linear_gain():
    m = query_metrics(["x", "y"], {"x": 1, "y": 2}, ["ndcg@2"])
    dcg = 1 + 2 / math.log2(3)
    idcg = 2 + 1 / math.log2(3)
    assert m["ndcg@2"] == pytest.approx(dcg / idcg)


def test_evaluate_run_missing_query_and_ties():
    run = {"q1": {"a": 1.0, "b": 1.0}}
    qrels = {"q1": {"b": 1}, "q2": {"z": 1}}
    agg, per_q = evaluate_run(run, qrels, ["recall@1"])
    assert ranked_docs(run["q1"]) == ["a", "b"]  # ties broken by doc id, deterministic
    assert per_q["q2"]["recall@1"] == 0
    assert agg["recall@1"] == 0
    assert gold_rank(["a", "b"], {"b": 1}) == 2
    assert gold_rank(["a"], {"b": 1}) is None


def test_permutation_test():
    a = [1.0] * 50
    b = [0.0] * 50
    assert paired_permutation_test(a, b, n=2000) < 0.01
    assert paired_permutation_test(a, a, n=500) == 1.0


def test_pytrec_crosscheck_if_available():
    pytrec = pytest.importorskip("pytrec_eval")
    from mmsearch.eval.metrics import crosscheck_pytrec

    run = {"q1": {"a": 3.0, "b": 2.0, "c": 1.0}, "q2": {"x": 1.0, "y": 0.5}}
    qrels = {"q1": {"b": 1, "c": 2}, "q2": {"y": 1}}
    agg, _ = evaluate_run(run, qrels, ["ndcg@10", "recall@10"])
    ref = crosscheck_pytrec(run, qrels, 10)
    assert agg["ndcg@10"] == pytest.approx(ref["ndcg@10"], abs=1e-6)
    assert agg["recall@10"] == pytest.approx(ref["recall@10"], abs=1e-6)
