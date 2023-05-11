#!/usr/bin/env python
"""This is the minimal example from the README"""

import numpy
from kernel_tuner import tune_kernel
from kernel_tuner.file_utils import store_output_file, store_metadata_file
import logging
from collections import OrderedDict
import os

def tune():

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 10000000

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    filename = "vector_add.json"
    if os.path.isfile(filename):
        results, env = tune_kernel("vector_add", kernel_string, size, args, tune_params, 
                                strategy="simulated_annealing",
                                lang="HIP", simulation_mode=True, cache="vector_add.json")

        # Store the tuning results in an output file
        store_output_file("vector_add_simulated_annealing.json", results, tune_params)

        return results
    else:
        print(f"{filename} does not exist in the directory, run vector_add.py first.")


if __name__ == "__main__":
    tune()