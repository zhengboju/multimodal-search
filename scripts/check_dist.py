"""Stage-0 acceptance: torchrun all-reduce across both GPUs + NCCL bus bandwidth (NVLink check).

torchrun --nproc_per_node 2 scripts/check_dist.py
"""

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    r, w = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    x = torch.full((4,), float(r + 1), device="cuda")
    dist.all_reduce(x)
    expect = w * (w + 1) / 2
    assert torch.allclose(x, torch.full_like(x, expect)), x
    if r == 0:
        print(f"all_reduce OK on {w} ranks (sum={x[0].item()})")
    size = 1 << 28  # 256M fp32 = 1 GiB
    buf = torch.ones(size, device="cuda")
    for _ in range(3):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    busbw = buf.numel() * 4 * 2 * (w - 1) / w / dt / 1e9
    if r == 0:
        print(f"all_reduce 1 GiB: {dt * 1000:.1f} ms, bus bandwidth ≈ {busbw:.0f} GB/s (NVLink H20 expect >> 50)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
