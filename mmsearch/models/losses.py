"""Contrastive losses for dual encoders (E2: InfoNCE vs SigLIP vs triplet) and late-interaction (ColBERT-style) losses.

Conventions: features are L2-normalised. `q` = query/text side, `d` = doc/image side.
In distributed training pass globally-gathered `d_all`/`q_all` and the local row offset so each rank
computes the loss for its own rows against all columns (OpenCLIP "local loss" pattern).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- dual-encoder losses
def _mask_extra(logits: torch.Tensor, n_base: int, invalid: torch.Tensor | None, fill: float = -1e4) -> torch.Tensor:
    """Mask extra (hard-negative) columns that are actually positives of the row (e.g. a mined negative that is
    the paired image of the same caption group). invalid: bool [B, E]."""
    if invalid is None:
        return logits
    pad = torch.zeros(logits.shape[0], n_base, dtype=torch.bool, device=logits.device)
    return logits.masked_fill(torch.cat([pad, invalid], dim=1), fill)


def infonce_loss(q: torch.Tensor, d: torch.Tensor, q_all: torch.Tensor, d_all: torch.Tensor, logit_scale: torch.Tensor,
                 offset: int = 0, d_extra: torch.Tensor | None = None, hard_only: bool = False,
                 extra_invalid: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric InfoNCE.

    q, d        local [B, D] text / image features (row i of q matches row i of d)
    q_all/d_all gathered [B*W, D]; local rows live at [offset, offset+B)
    d_extra     optional hard-negative images [E, D] appended as extra columns for the q->d direction
    hard_only   q->d uses only {positive} ∪ d_extra (ablation "hard negatives only", no in-batch)
    """
    scale = logit_scale.exp().clamp(max=100)
    B = q.shape[0]
    labels = torch.arange(B, device=q.device) + offset
    if hard_only:
        assert d_extra is not None, "hard_only requires hard negatives"
        pos = (q * d).sum(-1, keepdim=True)
        logits_qd = _mask_extra(scale * torch.cat([pos, q @ d_extra.t()], dim=1), 1, extra_invalid)
        loss_qd = F.cross_entropy(logits_qd, torch.zeros(B, dtype=torch.long, device=q.device))
    else:
        cols = d_all if d_extra is None else torch.cat([d_all, d_extra], dim=0)
        loss_qd = F.cross_entropy(_mask_extra(scale * q @ cols.t(), d_all.shape[0], extra_invalid), labels)
    loss_dq = F.cross_entropy(scale * d @ q_all.t(), labels)
    return 0.5 * (loss_qd + loss_dq)


class SigLIPHead(nn.Module):
    """Learnable temperature/bias for the sigmoid loss (init t=log 10, b=-10 as in SigLIP)."""

    def __init__(self, init_scale: float = math.log(10.0), init_bias: float = -10.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(init_scale))
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))


def siglip_loss(q: torch.Tensor, d_all: torch.Tensor, head: SigLIPHead, offset: int = 0,
                d_extra: torch.Tensor | None = None, extra_invalid: torch.Tensor | None = None) -> torch.Tensor:
    """Pairwise sigmoid loss of local queries against all (gathered) docs (+ hard negatives as extra negatives)."""
    cols = d_all if d_extra is None else torch.cat([d_all, d_extra], dim=0)
    logits = head.logit_scale.exp() * q @ cols.t() + head.logit_bias
    B = q.shape[0]
    labels = -torch.ones_like(logits)
    labels[torch.arange(B, device=q.device), torch.arange(B, device=q.device) + offset] = 1.0
    loss = -F.logsigmoid(labels * logits)
    if extra_invalid is not None:
        keep = torch.cat([torch.ones(B, d_all.shape[0], dtype=torch.bool, device=q.device), ~extra_invalid], dim=1)
        loss = loss * keep
    return loss.sum() / B


def triplet_loss(q: torch.Tensor, d: torch.Tensor, q_all: torch.Tensor, d_all: torch.Tensor, margin: float = 0.2,
                 offset: int = 0, d_extra: torch.Tensor | None = None, extra_invalid: torch.Tensor | None = None) -> torch.Tensor:
    """VSE++ max-violation triplet loss (hardest in-batch negative, both directions)."""
    B = q.shape[0]
    idx = torch.arange(B, device=q.device)
    pos = (q * d).sum(-1)
    cols = d_all if d_extra is None else torch.cat([d_all, d_extra], dim=0)
    s_qd = _mask_extra(q @ cols.t(), d_all.shape[0], extra_invalid)
    s_qd[idx, idx + offset] = -1e4  # mask positives
    s_dq = d @ q_all.t()
    s_dq[idx, idx + offset] = -1e4
    cost_q = (margin + s_qd.max(dim=1).values - pos).clamp(min=0)
    cost_d = (margin + s_dq.max(dim=1).values - pos).clamp(min=0)
    return (cost_q + cost_d).mean()


# ----------------------------------------------------------------------------- late interaction (ColBERT / ColPali)
def maxsim_scores(q: torch.Tensor, q_mask: torch.Tensor, d: torch.Tensor, d_mask: torch.Tensor,
                  doc_chunk: int = 256) -> torch.Tensor:
    """Late-interaction scores  s(q, d) = Σ_i max_j <q_i, d_j>.

    q [Bq, Lq, D], q_mask [Bq, Lq];  d [Nd, Ld, D], d_mask [Nd, Ld]  ->  [Bq, Nd] (float32).
    Differentiable; chunked over docs to bound memory (Bq*chunk*Lq*Ld).
    """
    q_mask = q_mask.bool()
    d_mask = d_mask.bool()
    out = []
    for s in range(0, d.shape[0], doc_chunk):
        dc, dm = d[s : s + doc_chunk], d_mask[s : s + doc_chunk]
        sim = torch.einsum("qid,njd->qnij", q, dc)  # [Bq, n, Lq, Ld]
        sim = sim.masked_fill(~dm[None, :, None, :], float("-inf"))
        best = sim.amax(dim=-1)  # [Bq, n, Lq]
        best = best.masked_fill(~q_mask[:, None, :], 0.0)
        out.append(best.float().sum(dim=-1))
    return torch.cat(out, dim=1)


def colbert_pairwise_ce_loss(scores: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
    """ColPali default: softplus(hardest in-batch negative − positive)."""
    B = scores.shape[0]
    rows = torch.arange(B, device=scores.device)
    pos = scores[rows, pos_idx]
    neg = scores.clone()
    neg[rows, pos_idx] = float("-inf")
    return F.softplus(neg.max(dim=1).values - pos).mean()


def colbert_ce_loss(scores: torch.Tensor, pos_idx: torch.Tensor, q_lengths: torch.Tensor | None = None,
                    temperature: float = 0.02) -> torch.Tensor:
    """InfoNCE over in-batch docs; scores optionally normalised by query length (so the temperature is length-agnostic)."""
    if q_lengths is not None:
        scores = scores / q_lengths.clamp(min=1).unsqueeze(1)
    return F.cross_entropy(scores / temperature, pos_idx)
