#!/usr/bin/env python
"""This is a minimal example for calling Fortran functions"""

from __future__ import print_function
import numpy as np
from kernel_tuner import tune_kernel


def tune():
    size = int(80e6)

    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["N"] = [size]
    tune_params["NTHREADS"] = [16, 8, 4, 2, 1]

    print("compile with gfortran")
    result, _ = tune_kernel(
        "time_vector_add",
        "vector_add.F90",
        size,
        args,
        tune_params,
        lang="C",
        compiler="gfortran",
    )

    return result


if __name__ == "__main__":
    tune()
