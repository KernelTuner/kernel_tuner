#!/usr/bin/env python
"""Tune a Triton matrix multiplication with the autotune decorator, while the application runs.

The kernel is tuned when it is first launched for a new matrix size (the tuning key), after which the best
configuration is launched. The tuning results are stored in wisdom files in the directory "wisdom", so running
this example again launches the best configurations without tuning.
"""

import torch
import triton
import triton.language as tl

from kernel_tuner import autotune


@autotune(
    tune_params={
        "BLOCK_M": [32, 64, 128],
        "BLOCK_N": [32, 64, 128],
        "BLOCK_K": [32, 64],
        "num_warps": [4, 8],
    },
    restrictions=["BLOCK_M * BLOCK_N <= 128 * 64"],
    key=["M", "N", "K"],
    reference=lambda a, b, c, M, N, K: {"c_ptr": (a.float() @ b.float()).half()},
    atol=0.5,  # float16 outputs of large matrices differ by up to a few float16 steps
    wisdom="wisdom",
)
@triton.jit
def matmul(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def grid(meta):
    return (triton.cdiv(meta["M"], meta["BLOCK_M"]), triton.cdiv(meta["N"], meta["BLOCK_N"]))


if __name__ == "__main__":
    for size in (1024, 2048):
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        c = torch.zeros(size, size, device="cuda", dtype=torch.float16)
        matmul[grid](a, b, c, size, size, size)  # tunes on the first launch for this size
        torch.testing.assert_close(c, (a.float() @ b.float()).half(), atol=1e-1, rtol=1e-2)
        print(f"{size}x{size}: best configuration {matmul.best_configs[(size, size, size, *['torch.float16'] * 3)]}")
