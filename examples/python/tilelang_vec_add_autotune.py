#!/usr/bin/env python
"""Autotune a TileLang vector add kernel with the Kernel Tuner autotune decorator.

TileLang kernels are created by a kernel factory decorated with @tilelang.jit. The decorator is placed on top of
the factory, the tunable parameters are arguments of the factory. Calling the factory with the other arguments
returns a kernel that is tuned when it is first called. The arguments of the factory are part of the tuning key.
"""

import tilelang
import tilelang.language as T
import torch

from kernel_tuner import autotune


@autotune(tune_params={"block_size": [128, 256, 512, 1024]})
@tilelang.jit
def vector_add(n: int, block_size: int = 128):
    @T.prim_func
    def kernel(
        C: T.Tensor((n,), "float32"),
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(T.ceildiv(n, block_size), threads=block_size) as bx:
            for i in T.Parallel(block_size):
                C[bx * block_size + i] = A[bx * block_size + i] + B[bx * block_size + i]

    return kernel


if __name__ == "__main__":
    for n in (1 << 16, 1 << 24):
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")
        c = torch.empty_like(a)

        vector_add(n)(c, a, b)  # block_size is chosen by the decorator, tuned once for each n

        torch.testing.assert_close(c, a + b)
    print("best configurations:", vector_add.best_configs)
