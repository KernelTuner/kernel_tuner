#!/usr/bin/env python
"""This is the minimal example from the README"""

import json
import numpy as np
from kernel_tuner.kernelbuilder import PythonKernel

def kernelbuilder_example():

    #To make this example self-contained we include the kernel as a string
    #here, but you can also just point to a file with the kernel code
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    #Setup the arguments for our vector add kernel
    size = 10000000
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)
    args = [c, a, b, n] #note that the type and order should match our GPU code

    #The inputs and outputs arrays are used to specify whether
    #the arguments in the argument list are inputs or outputs to the kernel
    inputs = [False, True, True, True]
    outputs=[not i for i in inputs]

    #We can instantiate a specific kernel configurations
    params = {"block_size_x": 128}

    #Here we construct a Python object that represents the kernel
    #we can use it to conveniently use the GPU kernel in Python
    #applications that want to frequently call the GPU kernel
    vector_add = PythonKernel("vector_add", kernel_string, size, args, params,
                              inputs=inputs, outputs=outputs)

    #We can use the PythonKernel instance as a regular Python function
    result = vector_add(c, a, b, n)

    #Here we compare the result with a+b computed in Python
    assert np.allclose(result, a+b)

    return result


if __name__ == "__main__":
    kernelbuilder_example()
