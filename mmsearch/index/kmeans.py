"""Torch k-means utilities (spherical / cosine), used by PLAID-lite centroids and per-page token pooling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def kmeans(x: torch.Tensor, k: int, iters: int = 10, seed: int = 0, chunk: int = 131072) -> torch.Tensor:
    """Spherical k-means on rows of x [n, D] (assumed L2-normalised). Returns centroids [k, D] (float32)."""
    x = x.float()
    n = x.shape[0]
    k = min(k, n)
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = x[torch.randperm(n, generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        assign = assign_nearest(x, C, chunk)
        sums = torch.zeros_like(C).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k)
        empty = counts == 0
        if empty.any():  # re-seed empty clusters with random points
            ridx = torch.randint(0, n, (int(empty.sum()),), generator=g).to(x.device)
            sums[empty] = x[ridx]
        C = F.normalize(sums, dim=-1)
    return C


@torch.no_grad()
def assign_nearest(x: torch.Tensor, C: torch.Tensor, chunk: int = 131072) -> torch.Tensor:
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    Cf = C.to(x.dtype)
    chunk = max(1024, min(chunk, (1 << 28) // max(1, C.shape[0])))  # bound the [chunk, C] similarity block (~0.5-1 GB)
    for s in range(0, x.shape[0], chunk):
        out[s : s + chunk] = (x[s : s + chunk] @ Cf.t()).argmax(dim=1)
    return out


@torch.no_grad()
def batched_kmeans(X: torch.Tensor, mask: torch.Tensor, k: int, iters: int = 10, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent spherical k-means for each item of a padded batch.

    X [B, L, D], mask [B, L] bool  ->  centroids [B, k, D], valid [B, k] (False where a doc has < k tokens).
    """
    B, L, D = X.shape
    X = X.float()
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.rand(B, L, generator=g).to(X.device).masked_fill(~mask, -1.0)
    kk = min(k, L)
    init = r.topk(kk, dim=1).indices  # random distinct valid tokens first
    valid = torch.gather(mask, 1, init)
    C = torch.gather(X, 1, init.unsqueeze(-1).expand(B, kk, D)).clone()
    for _ in range(iters):
        sim = torch.einsum("bld,bkd->blk", X, C).masked_fill(~valid[:, None, :], float("-inf"))
        assign = sim.argmax(-1)  # [B, L]
        onehot = F.one_hot(assign, kk).to(X.dtype) * mask.unsqueeze(-1).to(X.dtype)
        sums = torch.einsum("blk,bld->bkd", onehot, X)
        counts = onehot.sum(1)  # [B, k]
        C = torch.where((counts > 0).unsqueeze(-1), F.normalize(sums, dim=-1), C)
    return C, valid
