#!/usr/bin/env python
"""Autotune a Warp vector add kernel with the Kernel Tuner autotune decorator.

Two parameters are tuned: the number of elements per thread, a constant in the kernel that Kernel Tuner replaces
by the value of each configuration, and the thread block size, which the decorator passes to wp.launch as
block_dim. The decorated kernel is launched like wp.launch, but with kernel.launch(...). Because the number of
threads depends on the elements per thread, the launch dimensions are a function of the kernel arguments and the
tunable parameters.
"""

import numpy as np
import warp as wp

from kernel_tuner import autotune

wp.config.log_level = wp.LOG_WARNING  # do not print a message for every compiled module
wp.init()

work_per_thread = 1  # replaced by the value of the tunable parameter


@autotune(tune_params={"work_per_thread": [1, 2, 4, 8], "block_dim": [128, 256, 512]}, key=["n"])
@wp.kernel
def vector_add(c: wp.array(dtype=float), a: wp.array(dtype=float), b: wp.array(dtype=float), n: int):
    base = wp.tid() * work_per_thread
    for i in range(work_per_thread):
        if base + i < n:
            c[base + i] = a[base + i] + b[base + i]


def dim(meta):
    return (meta["n"] + meta["work_per_thread"] - 1) // meta["work_per_thread"]


if __name__ == "__main__":
    n = 10_000_000
    a = wp.array(np.random.randn(n).astype(np.float32), dtype=float, device="cuda")
    b = wp.array(np.random.randn(n).astype(np.float32), dtype=float, device="cuda")
    c = wp.zeros(n, dtype=float, device="cuda")

    vector_add.launch(dim, inputs=[c, a, b, n])  # instead of wp.launch(vector_add, dim, inputs=[...])
    # or launch the kernel like a Triton kernel: vector_add[dim](c, a, b, n)

    assert np.allclose(c.numpy(), a.numpy() + b.numpy())
    print("best configuration:", vector_add.best_configs)
