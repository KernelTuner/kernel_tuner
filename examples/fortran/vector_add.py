#!/usr/bin/env python
"""This is a minimal example for calling Fortran functions"""

import json
import numpy as np
from kernel_tuner import run_kernel

def tune():

    with open('vector_add.F90', 'r') as f:
        kernel_string = f.read()

    size = 10000000

    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["N"] = size

    answer = run_kernel("vector_add", kernel_string, size, args, tune_params, lang="C", compiler="gfortran")

    assert np.allclose(answer[0], a+b, atol=1e-8)


if __name__ == "__main__":
    tune()
