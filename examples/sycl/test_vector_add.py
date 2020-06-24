#!/usr/bin/env python

import numpy as np

import kernel_tuner


def test_vector_add():

    n = np.int32(1000)

    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros_like(b)

    args = [c, a, b, n]
    params = {}

    #sycl compiler
    compiler = "compute++"

    #sycl options
    compiler_options = ["-sycl", "-sycl-target", "ptx64"]

    #workaround on my system, probably not neccesary in general
    compiler_options += ["--gcc-toolchain=/cm/shared/package/gcc/6.4.0"]

    answer = kernel_tuner.run_kernel("vector_add", "vector_add.cpp", n, args, params,
                            compiler=compiler, compiler_options=compiler_options, lang="SYCL")

    assert np.allclose(answer[0], a+b)


if __name__ == "__main__":
    test_vector_add()

