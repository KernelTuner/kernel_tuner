#!/usr/bin/env python

import numpy as np

import kernel_tuner


def tune_vector_add():

    n = np.int32(10000000)

    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros_like(b)

    args = [c, a, b, n]
    tune_params = {}
    tune_params["block_size_x"] = [64, 256, 512]

    #compiler options
    cp = ["-sycl", "-sycl-target", "ptx64"]

    #workaround on my system, probably not neccesary in general
    cp += ["--gcc-toolchain=/cm/shared/package/gcc/6.4.0"]

    results, env = kernel_tuner.tune_kernel("vector_add", "vector_add.cpp", n, args, tune_params,
                            compiler="compute++", compiler_options=cp, lang="SYCL")

    print(results)

    return results, env


if __name__ == "__main__":
    tune_vector_add()

