#!/usr/bin/env python
"""This is the minimal example from the README"""

import numpy
from kernel_tuner import tune_kernel
from kernel_tuner.file_utils import store_output_file, store_metadata_file
import logging
from collections import OrderedDict

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

    results, env = tune_kernel("vector_add", kernel_string, size, args, tune_params, lang="HIP", 
                               cache="vector_add_cache.json", log=logging.DEBUG)

    # Store the metadata of this run
    store_metadata_file("vector_add-metadata.json")

    return results


if __name__ == "__main__":
    tune()