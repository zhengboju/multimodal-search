"""Visual-token budget management for Qwen2/2.5-VL style dynamic-resolution VLMs.

Qwen2-VL: an image of H x W (after smart_resize to multiples of 28) costs (H/28)*(W/28) tokens.
Multi-image RAG prompts must share a total visual budget; text RAG intuitions (top-k=10~20) blow up the context.
"""

from __future__ import annotations

import math

PATCH = 28


def smart_resize(height: int, width: int, factor: int = PATCH, min_pixels: int = 4 * PATCH * PATCH,
                 max_pixels: int = 16384 * PATCH * PATCH) -> tuple[int, int]:
    """Replicates Qwen2-VL's resize rule (multiples of `factor`, pixel count clamped)."""
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def visual_tokens(width: int, height: int, max_pixels: int | None = None) -> int:
    h, w = smart_resize(height, width, max_pixels=max_pixels or 16384 * PATCH * PATCH)
    return (h // PATCH) * (w // PATCH)


def tokens_to_pixels(tokens: int) -> int:
    return int(tokens) * PATCH * PATCH


def allocate_tokens(n_images: int, total_tokens: int, strategy: str = "uniform",
                    min_tokens: int = 256, max_tokens: int = 1280, decay: float = 0.5) -> list[int]:
    """Split a total visual-token budget across ranked images.

    uniform    : equal share
    rank_decay : share ∝ 1/(rank+1)^decay — top-ranked pages get more resolution
    Each share is clamped to [min_tokens, max_tokens]; if even min_tokens doesn't fit, fewer images
    should be used (caller decides; we return the clamped shares anyway).
    """
    if n_images <= 0:
        return []
    if strategy == "uniform":
        w = [1.0] * n_images
    elif strategy == "rank_decay":
        w = [1.0 / (r + 1) ** decay for r in range(n_images)]
    else:
        raise ValueError(f"unknown budget strategy {strategy}")
    s = sum(w)
    return [int(min(max_tokens, max(min_tokens, total_tokens * x / s))) for x in w]


def max_images_within(total_tokens: int, min_tokens: int = 256) -> int:
    return max(1, total_tokens // min_tokens)
