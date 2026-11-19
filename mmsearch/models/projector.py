from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Hidden state -> low-dim (128) token embedding, L2-normalised and padding-masked (ColPali head)."""

    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.linear(hidden.to(self.linear.weight.dtype))
        x = F.normalize(x, dim=-1)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x
