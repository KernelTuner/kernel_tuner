#!/usr/bin/env python
"""This is a minimal example for calling Fortran functions"""

import numpy as np
from kernel_tuner import tune_kernel


def tune():
    size = int(72 * 1024 * 1024)

    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["N"] = [size]
    tune_params["block_size_x"] = [32, 64, 128, 256, 512]

    result, env = tune_kernel(
        "time_vector_add",
        "vector_add_acc.F90",
        size,
        args,
        tune_params,
        lang="C",
        compiler="nvfortran",
        compiler_options=["-fast", "-acc=gpu"],
    )

    return result


if __name__ == "__main__":
    tune()
