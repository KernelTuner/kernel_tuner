#!/usr/bin/env python
"""Autotune a Numba CUDA vector add kernel with the Kernel Tuner autotune decorator.

In Numba, the thread block size is part of the launch: kernel[grid, block](*args). Because the block size is the
tunable parameter, the decorator computes the launch dimensions with the grid and threads functions, which
receive the kernel arguments and the tunable parameters of a configuration. These replace the launch dimensions
of the launch, so the kernel can be launched with any placeholder dimensions.

The reference function makes Kernel Tuner verify the output of every configuration while tuning.
"""

import numpy as np
from numba import cuda

from kernel_tuner import autotune


@autotune(
    tune_params={"block_size_x": [32, 64, 128, 256, 512, 1024]},
    key=["n"],
    grid=lambda p: ((p["n"] + p["block_size_x"] - 1) // p["block_size_x"],),
    threads=lambda p: (p["block_size_x"],),
    reference=lambda c, a, b, n: {"c": a.copy_to_host() + b.copy_to_host()},
)
@cuda.jit
def vector_add(c, a, b, n):
    i = cuda.grid(1)
    if i < n:
        c[i] = a[i] + b[i]


if __name__ == "__main__":
    n = 10_000_000
    a = cuda.to_device(np.random.randn(n).astype(np.float32))
    b = cuda.to_device(np.random.randn(n).astype(np.float32))
    c = cuda.device_array_like(a)

    vector_add[1, 1](c, a, b, n)  # the launch dimensions are computed by the decorator

    assert np.allclose(c.copy_to_host(), a.copy_to_host() + b.copy_to_host())
    print("best configuration:", vector_add.best_configs)
