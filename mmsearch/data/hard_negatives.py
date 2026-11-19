"""Hard-negative mining for dual encoders (ANCE-style: mine with the *current* model, refresh periodically).

For every training image we collect images retrieved by its own captions whose rank lies in [lo, hi)
(skipping the very top ranks reduces false negatives — COCO has many near-duplicate scenes).
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def mine_hard_negatives(img_emb: torch.Tensor, txt_emb: torch.Tensor, txt2img: torch.Tensor, rank_range=(3, 50),
                        pool: int = 20, device: str | None = None, chunk: int = 4096) -> list[np.ndarray]:
    """img_emb [N, D], txt_emb [M, D] (normalised), txt2img [M] -> per-image array of hard negative image indices."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32
    I = img_emb.to(dev, dtype)
    lo, hi = rank_range
    N = img_emb.shape[0]
    cands: list[list[int]] = [[] for _ in range(N)]
    for s in range(0, txt_emb.shape[0], chunk):
        T = txt_emb[s : s + chunk].to(dev, dtype)
        sims = T @ I.t()
        own = txt2img[s : s + chunk].to(dev)
        sims[torch.arange(T.shape[0], device=dev), own] = float("-inf")  # drop the positive
        top = sims.topk(hi, dim=1).indices[:, lo:].cpu().numpy()  # ranks [lo, hi) among negatives
        for row, img in zip(top, own.cpu().tolist()):
            cands[img].extend(row.tolist())
    out = []
    for img, c in enumerate(cands):
        # interleave captions' lists by frequency: images retrieved by several captions are the hardest
        uniq, counts = np.unique(np.asarray(c, dtype=np.int64), return_counts=True)
        order = np.argsort(-counts, kind="stable")
        out.append(uniq[order][:pool])
    return out


def random_negatives(n_images: int, pool: int = 20, seed: int = 0) -> list[np.ndarray]:
    """Control condition: same pipeline with random negatives (isolates 'hardness' from 'more negatives')."""
    rng = np.random.default_rng(seed)
    return [rng.choice(n_images, size=pool, replace=False) for _ in range(n_images)]
