import torch

from mmsearch.eval.modality_gap import (
    CONDITIONS,
    apply_condition,
    build_verdict,
    gap_stats,
    raw_centroid,
    translated,
    unit_centroid,
)
from mmsearch.eval.eval_clip import bidirectional_recall


def fake_embeddings(n_img=40, n_cap_per_img=2, d=16, gap=(0.6, -0.2), seed=0):
    """Image and text clouds sharing content signal but separated by a constant modality-gap vector."""
    g = torch.Generator().manual_seed(seed)
    base = torch.nn.functional.normalize(torch.randn(n_img, d, generator=g), dim=-1)
    img = torch.nn.functional.normalize(base + 0.05 * torch.randn(n_img, d, generator=g), dim=-1)
    shift = torch.tensor(gap[: d // len(gap)] * (d // len(gap)) if d > 2 else [gap[0]])
    shift = torch.zeros(d).scatter(0, torch.arange(len(shift)), shift)
    txt = torch.nn.functional.normalize(base.repeat_interleave(n_cap_per_img, 0) + shift + 0.2 * torch.randn(n_img * n_cap_per_img, d, generator=g), dim=-1)
    t2i = torch.arange(n_img).repeat_interleave(n_cap_per_img)
    return img, txt, t2i


def test_centroids_and_cone_norms():
    img, txt, _ = fake_embeddings()
    assert torch.allclose(unit_centroid(img).norm(), torch.tensor(1.0), atol=1e-5)
    assert 0 < raw_centroid(img).norm() < 1  # unit vectors -> ||mean|| in (0, 1)
    assert torch.allclose((img - raw_centroid(img)).mean(0), torch.zeros(img.shape[1]), atol=1e-5)  # centred before L2


def test_translate_aligns_centroids_and_keeps_scale():
    img, txt, _ = fake_embeddings()
    before = torch.cosine_similarity(unit_centroid(img), unit_centroid(txt), dim=0)
    txt2 = translated(txt, img)
    after = torch.cosine_similarity(unit_centroid(img), unit_centroid(txt2), dim=0)
    assert after > before + 0.5, (before, after)
    assert torch.allclose(txt2.norm(dim=-1), torch.ones(txt2.shape[0]), atol=1e-5)


def test_translate_does_not_reorder_text_candidates():
    """A constant shift of the candidate side must leave i2t ranking (almost) unchanged — this is why
    'gap removal by translation' is the right probe, and mean-centering (which also rescales) is not."""
    img, txt, t2i = fake_embeddings()
    before = bidirectional_recall(img, txt, t2i)
    after = bidirectional_recall(img, translated(txt, img), t2i)
    assert abs(after["i2t_R@1"] - before["i2t_R@1"]) <= 5  # only per-candidate renormalisation can move it
    c = apply_condition("center", img, txt)
    assert not torch.allclose(c[0], img) and not torch.allclose(c[1], txt)


def test_gap_stats_are_consistent():
    img, txt, t2i = fake_embeddings()
    s = gap_stats(img, txt, t2i)
    for k in ("centroid_distance", "centroid_cosine", "img_cone_norm", "txt_cone_norm", "matched_cos_mean",
              "unmatched_cos_mean", "content_margin", "offset_along_gap_mean"):
        assert k in s
    assert s["content_margin"] == s["matched_cos_mean"] - s["unmatched_cos_mean"]
    assert -1.0 <= s["centroid_cosine"] <= 1.0
    assert s["offset_along_gap_mean"] > 0.3  # synthetic pairs are dominated by the injected gap
    for c in CONDITIONS:
        a, b = apply_condition(c, img, txt)
        assert a.shape == img.shape and b.shape == txt.shape

def _table(i2t, t2i, margin):
    mk = lambda a, b, m: {"i2t_R@1": a, "t2i_R@1": b, "matched": m + 0.09, "unmatched": 0.09, "margin": m}  # noqa: E731
    return {
        "original": mk(56.0, 36.0, margin),
        "translate": mk(i2t, t2i, 0.08),
        "center_txt": mk(43.0, 38.0, 0.15),
        "center_img": mk(53.0, 31.0, 0.16),
    }


STATS = dict(offset_along_gap_mean=0.68, img_cone_norm=0.70, txt_cone_norm=0.59, content_margin=0.17, centroid_distance=1.26)


def test_verdict_regimes():
    """The conclusion must follow the measurement, not be hard-coded (an earlier version asserted the wrong one)."""
    destructive = "\\n".join(build_verdict(_table(38.0, 23.0, 0.17), STATS))
    assert "destructive" in destructive and "gap axis" in destructive

    gain = "\\n".join(build_verdict(_table(62.0, 42.0, 0.17), STATS))
    assert "removable nuisance" in gain

    neutral = "\\n".join(build_verdict(_table(56.5, 36.4, 0.17), STATS))
    assert "not what limits retrieval" in neutral

    assert "asymmetry" in neutral.lower() or "Side asymmetry" in neutral
    assert "Content budget" in neutral
    assert build_verdict({}, STATS) == []