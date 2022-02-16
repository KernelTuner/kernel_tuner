#!/usr/bin/env python
"""This is a minimal example for calling Fortran functions"""

from __future__ import print_function
import logging
import json
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

    print("compile with ftn using intel on cray")
    result, env = tune_kernel("time_vector_add", "vector_add.F90", size,
                              args, tune_params, lang="C", compiler="ftn")

    print("compile with gfortran")
    result, env = tune_kernel("time_vector_add", "vector_add.F90", size,
                              args, tune_params, lang="C", compiler="gfortran")

    print("compile with pgfortran")
    result, env = tune_kernel("time_vector_add", "vector_add.F90", size,
                              args, tune_params, lang="C", compiler="pgfortran")

    return result

if __name__ == "__main__":
    tune()
