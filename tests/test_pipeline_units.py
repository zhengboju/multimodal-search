import pytest
import torch
from PIL import Image

from mmsearch.data.schema import Doc, Query, RetrievalDataset, load_dataset
from mmsearch.eval.eval_clip import bidirectional_recall
from mmsearch.eval.llm_judge import cohen_kappa, parse_score
from mmsearch.llm.budget import allocate_tokens, smart_resize, visual_tokens
from mmsearch.query.rewriter import compose, parse_rewrite, random_expansion
from mmsearch.rag.answer_metrics import anls, relaxed_accuracy, token_f1
from mmsearch.rag.run_rag_eval import attribute
from mmsearch.retrieve.fusion import rrf, weighted_sum
from mmsearch.retrieve.sparse import BM25Retriever, tokenize
from mmsearch.utils.config import load_config


def tiny_dataset(tmp_path) -> RetrievalDataset:
    img = tmp_path / "p0.png"
    Image.new("RGB", (40, 30), "white").save(img)
    corpus = {
        "d0": Doc("d0", text="annual revenue grew 12% in 2023 driven by cloud services", image=str(img)),
        "d1": Doc("d1", text="employee headcount and office locations in Europe"),
        "d2": Doc("d2", text="2023年 收入 增长 云服务"),
    }
    queries = {"q0": Query("q0", "revenue growth 2023"), "q1": Query("q1", "office locations"), "q2": Query("q2", "云服务收入", lang="zh")}
    qrels = {"q0": {"d0": 1}, "q1": {"d1": 1}, "q2": {"d2": 1}}
    return RetrievalDataset("tiny", corpus, queries, qrels, {"split": "test"})


def test_schema_roundtrip_and_invert(tmp_path):
    ds = tiny_dataset(tmp_path)
    ds.save(tmp_path / "tiny")
    ds2 = load_dataset(tmp_path / "tiny")
    assert set(ds2.corpus) == set(ds.corpus) and ds2.qrels == ds.qrels
    assert ds2.corpus["d0"].load_image().size == (40, 30)
    inv = ds2.invert()
    assert inv.qrels["d0"] == {"q0": 1}
    assert len(list(ds2.to_samples())) == 3


def test_bm25_multilingual(tmp_path):
    ds = tiny_dataset(tmp_path)
    r = BM25Retriever()
    r.index(ds)
    run = r.search(ds.queries, 3)
    for qid, rels in ds.qrels.items():
        assert max(run[qid], key=run[qid].get) == next(iter(rels))
    assert "云服" in tokenize("云服务")


def test_fusion():
    a = {"q": {"x": 3.0, "y": 2.0, "z": 1.0}}
    b = {"q": {"y": 10.0, "z": 9.0}}
    f = rrf([a, b])
    assert max(f["q"], key=f["q"].get) == "y"
    w = weighted_sum([a, b], [1.0, 1.0])
    assert max(w["q"], key=w["q"].get) == "y"


def test_answer_metrics_and_attribution():
    assert anls("1,234", ["1,234"]) == 1.0
    assert anls("compny", ["company"]) > 0.8
    assert anls("totally different", ["x"]) == 0.0
    assert relaxed_accuracy("10.4", ["10"]) == 1.0
    assert relaxed_accuracy("11", ["10"]) == 0.0
    assert relaxed_accuracy("45%", ["45"]) == 1.0
    assert relaxed_accuracy("Yes", ["yes"]) == 1.0
    assert token_f1("the big red car", ["red car"]) > 0.6
    assert attribute(True, True, True) == "correct"
    assert attribute(False, False, True) == "retrieval_miss"
    assert attribute(True, False, True) == "context_distraction"
    assert attribute(True, False, False) == "generation_error"


def test_budget():
    h, w = smart_resize(1000, 700)
    assert h % 28 == 0 and w % 28 == 0
    assert visual_tokens(2000, 2000, max_pixels=768 * 28 * 28) <= 768
    shares = allocate_tokens(4, 4096, "rank_decay")
    assert shares == sorted(shares, reverse=True) and sum(shares) <= 4096 + 4


def test_bidirectional_recall_perfect():
    img = torch.eye(4)
    txt = torch.cat([torch.eye(4), torch.eye(4)])  # 2 captions per image
    t2i = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    r = bidirectional_recall(img, txt, t2i)
    assert r["i2t_R@1"] == 100 and r["t2i_R@1"] == 100


def test_kappa_and_judge_parse():
    assert cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)
    assert cohen_kappa([0, 1, 0, 1], [1, 0, 1, 0]) < 0
    assert parse_score('{"score": 2, "reason": "x"}') == 2


def test_rewrite_parsing_and_controls():
    import random

    rr = parse_rewrite('noise {"intent": "numeric", "entities": ["revenue"], "rewrite": "total revenue 2023", "lang": "en"} tail')
    assert rr and rr.rewrite == "total revenue 2023"
    assert compose("rev 23", rr, "concat") == "rev 23 total revenue 2023"
    exp = random_expansion("rev 23", 5, ["a", "b"], random.Random(0))
    assert len(exp.split()) == 5
    assert parse_rewrite("not json") is None


def test_config_overrides(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("a: 1\nb: {c: 2, d: 3}\n")
    child = tmp_path / "child.yaml"
    child.write_text("_base_: base.yaml\nb: {c: 5}\n")
    cfg = load_config(child, ["b.d=7", "e=[1, 2]"])
    assert cfg.a == 1 and cfg.b.c == 5 and cfg.b.d == 7 and cfg.e == [1, 2]
