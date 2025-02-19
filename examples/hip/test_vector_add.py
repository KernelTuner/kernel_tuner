#!/usr/bin/env python
"""Minimal example for a HIP Kernel unit test with the Kernel Tuner"""

import numpy
from kernel_tuner import run_kernel
import pytest

#Check hip is installed and if a HIP capable device is present, if not skip the test
try:
    from hip import hip, hiprtc
except ImportError:
    pytest.skip("HIP Python not installed or PYTHONPATH does not includes HIP Python")
    hip = None
    hiprtc = None

def test_vector_add():

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 10000000
    problem_size = (size, 1)

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]
    params = {"block_size_x": 512}

    answer = run_kernel("vector_add", kernel_string, problem_size, args, params, lang="HIP")

    assert numpy.allclose(answer[0], a+b, atol=1e-8)