"""Minimal torch.distributed helpers (torchrun-compatible, no-ops in single process)."""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main() -> bool:
    return rank() == 0


def init_distributed(timeout_min: int = 60) -> torch.device:
    """Init NCCL process group if launched by torchrun; returns this process' device."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
        device = torch.device("cuda", local_rank())
    else:
        device = torch.device("cpu")
    if int(os.environ.get("WORLD_SIZE", 1)) > 1 and not is_dist():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=datetime.timedelta(minutes=timeout_min),
        )
    return device


def barrier() -> None:
    if is_dist():
        dist.barrier()


def cleanup() -> None:
    if is_dist():
        dist.destroy_process_group()


def all_gather_with_grad(t: torch.Tensor) -> torch.Tensor:
    """Concatenate `t` from all ranks along dim 0, keeping gradients for the local shard."""
    if not is_dist():
        return t
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(t), dim=0)


def all_gather_no_grad(t: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return t
    out = [torch.empty_like(t) for _ in range(world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.cat(out, dim=0)


def gather_padded(emb: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather variable-length multi-vector batches [B, L, D] (+mask [B, L]) by padding L to the global max."""
    if not is_dist():
        return emb, mask
    L = torch.tensor([emb.shape[1]], device=emb.device)
    dist.all_reduce(L, op=dist.ReduceOp.MAX)
    pad = int(L.item()) - emb.shape[1]
    if pad > 0:
        emb = torch.nn.functional.pad(emb, (0, 0, 0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad))
    return all_gather_with_grad(emb), all_gather_no_grad(mask.to(torch.uint8)).bool()  # NCCL: no bool all_gather


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return x
    y = x.detach().clone()
    dist.all_reduce(y)
    return y / world_size()
