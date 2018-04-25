#!/usr/bin/env
""" Example to show how to use the C++ wrapper

This example shows how to use Kernel Tuner's wrapper
functionality to also call (primitive-typed) C++
functions from Python.
"""

from kernel_tuner import run_kernel
from kernel_tuner import wrappers

import numpy as np

def test_multiply_matrix():

    function_name = "multiply_matrix"

    with open('matrix_multiply.cpp', 'r') as f:
        kernel_string = f.read()

    a = np.random.randn(9).astype(np.float64)
    b = np.random.randn(9).astype(np.float64)
    c = np.zeros_like(a)

    args = [c, a, b, np.int32(3)]
    convert = [True for _ in args]
    convert[-1] = False

    #generate a wrapper function with "extern C" binding that can be called from Python
    kernel_string = wrappers.cpp(function_name, kernel_string, args, convert_to_array=convert)

    answer = run_kernel(function_name + "_wrapper", kernel_string, 1, args, {},
               lang="C")

    #compute expected answer of matrix multiplication with Numpy
    expected = a.reshape(3,3).dot(b.reshape(3,3))

    assert np.allclose(answer[0].reshape(3,3), expected)
