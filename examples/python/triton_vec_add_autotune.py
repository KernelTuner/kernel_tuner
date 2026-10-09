#!/usr/bin/env python
"""Autotune a Triton vector add kernel with the Kernel Tuner autotune decorator.

The decorator is placed on top of @triton.jit and the kernel is launched as usual. When the kernel is launched for
a new vector size (the tuning key), Kernel Tuner tunes it on copies of the arguments and launches the best
configuration. Later launches with the same size launch the best configuration directly.

The second kernel shows that the decorator can replace triton.autotune: it accepts the same list of triton.Config
objects, and only benchmarks these configurations.
"""

import torch
import triton
import triton.language as tl

from kernel_tuner import autotune


@autotune(
    tune_params={"BLOCK_SIZE": [128, 256, 512, 1024], "num_warps": [2, 4, 8]},
    key=["n"],  # tune again for every new vector size
)
@triton.jit
def vector_add(c_ptr, a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


# the same kernel, with the configurations of an existing triton.autotune decorator
@autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def vector_add_configs(c_ptr, a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def grid(meta):
    """As in Triton, the grid is a function of the kernel arguments and the configuration."""
    return (triton.cdiv(meta["n"], meta["BLOCK_SIZE"]),)


if __name__ == "__main__":
    for kernel in (vector_add, vector_add_configs):
        for n in (10_000, 10_000_000, 10_000):
            a = torch.randn(n, device="cuda")
            b = torch.randn(n, device="cuda")
            c = torch.empty_like(a)
            kernel[grid](c, a, b, n)  # tunes on the first launch for each n
            torch.testing.assert_close(c, a + b)

        print(f"{kernel.name}: tuned {len(kernel.tuning_results)} times, best configurations:")
        for key, config in kernel.best_configs.items():
            print(f"  n={key[0]}: {config}")
