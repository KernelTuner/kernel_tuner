#!/usr/bin/env python
"""Minimal example for a parameterized test of a CUDA kernel using Kernel Tuner"""

import numpy
from kernel_tuner import run_kernel
import pytest

@pytest.fixture()
def vector_add():
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 1000
    problem_size = (size, 1)

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]

    return ["vector_add", kernel_string, problem_size, args]

@pytest.mark.parametrize('block_size_x', [2**i for i in range(5,11)])
def test_vector_add(vector_add, block_size_x):

    answer = run_kernel(*vector_add, {"block_size_x": block_size_x})

    a, b = vector_add[-1][1:3]
    assert numpy.allclose(answer[0], a+b, atol=1e-8)
