#!/usr/bin/env python
"""
    This is the vector_add example modified to show
    how to use PythonKernel with the CuPy backend
"""

import cupy as cp
import numpy as np
from kernel_tuner.kernelbuilder import PythonKernel

def kernelbuilder_example():

    # To make this example self-contained we include the kernel as a string
    # here, but you can also just point to a file with the kernel code
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    # Setup the arguments for our vector add kernel
    size = 100000
    a = cp.random.randn(size).astype(np.float32)
    b = cp.random.randn(size).astype(np.float32)
    c = cp.zeros_like(b)
    n = np.int32(size)

    # Note that the type and order should match our GPU code
    # Because the arguments are all CuPy arrays, our PythonKernel does not need to
    # worry about moving data between host and device
    args = [c, a, b, n]

    # We can instantiate a specific kernel configurations
    params = {"block_size_x": 128}

    # Here we construct a Python object that represents the kernel
    # we can use it to conveniently use the GPU kernel in Python
    # applications that want to frequently call the GPU kernel
    vector_add = PythonKernel("vector_add", kernel_string, size, args, params, lang="cupy")

    # We can use the PythonKernel instance as a regular Python function
    vector_add(c, a, b, n)

    # Compare the result in c with a+b computed in Python
    assert np.allclose(c, a+b)


if __name__ == "__main__":
    kernelbuilder_example()
