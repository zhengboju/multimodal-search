import math

import torch
import torch.nn.functional as F

from mmsearch.models.losses import (
    SigLIPHead,
    colbert_ce_loss,
    colbert_pairwise_ce_loss,
    infonce_loss,
    siglip_loss,
    triplet_loss,
)


def feats(n=8, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_infonce_matches_manual():
    q, d = feats(seed=1), feats(seed=2)
    scale = torch.tensor(math.log(10.0))
    loss = infonce_loss(q, d, q, d, scale)
    logits = 10 * q @ d.t()
    lab = torch.arange(8)
    ref = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    assert torch.allclose(loss, ref, atol=1e-5)


def test_infonce_hard_negatives_and_masking():
    q, d = feats(seed=1), feats(seed=2)
    scale = torch.tensor(math.log(10.0))
    extra = feats(4, seed=3)
    l_mixed = infonce_loss(q, d, q, d, scale, d_extra=extra)
    assert l_mixed > infonce_loss(q, d, q, d, scale)  # more negatives -> higher loss
    # masking an extra column for every row must recover the plain in-batch loss
    dup = d[:1].clone()
    invalid = torch.ones(8, 1, dtype=torch.bool)
    masked = infonce_loss(q, d, q, d, scale, d_extra=dup, extra_invalid=invalid)
    assert torch.allclose(masked, infonce_loss(q, d, q, d, scale), atol=1e-4)
    hard_only = infonce_loss(q, d, q, d, scale, d_extra=extra[:8], hard_only=True)
    assert torch.isfinite(hard_only)


def test_siglip_and_triplet():
    q = feats(seed=4)
    assert torch.isfinite(siglip_loss(q, q, SigLIPHead()))
    # perfectly separated pairs -> zero triplet loss
    eye = torch.eye(8)
    assert triplet_loss(eye, eye, eye, eye, margin=0.2).item() == 0.0
    assert triplet_loss(feats(seed=5), feats(seed=6), feats(seed=5), feats(seed=6)).item() > 0


def test_colbert_losses():
    scores = torch.tensor([[5.0, 1.0, 0.0], [0.0, 4.0, 3.0], [1.0, 1.0, 9.0]])
    pos = torch.arange(3)
    lp = colbert_pairwise_ce_loss(scores, pos)
    ref = torch.stack([F.softplus(torch.tensor(1.0 - 5)), F.softplus(torch.tensor(3.0 - 4)), F.softplus(torch.tensor(1.0 - 9))]).mean()
    assert torch.allclose(lp, ref)
    assert torch.isfinite(colbert_ce_loss(scores, pos, torch.tensor([3.0, 3.0, 3.0])))
