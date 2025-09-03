#!/usr/bin/env python
"""This is the minimal example from the README extended with user-defined metrics"""

from collections import OrderedDict
import json
import numpy
from kernel_tuner import tune_kernel

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

    tune_params = dict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)]

    # This example illustrates how to use metrics
    # metrics can be either specified as functions or using strings

    # metrics need to be OrderedDicts because we can compose
    # earlier defined metrics into new metrics
    metrics = OrderedDict()

    # This metrics is the well-known GFLOP/s performance metric
    # we can define the value of the metric using a function that accepts 1 argument
    # this argument is a dictionary with all the tunable parameters and benchmark results
    # the value of the metric is calculated directly after obtaining the benchmark results
    metrics["GFLOP/s"] = lambda p : (size/1e9) / (p["time"]/1000)

    # Alternatively you can specify the metric using strings
    # in these strings you can use the names of the kernel parameters and benchmark results
    # directly as they will be replaced by the tuner before evaluating this string
    metrics["GB/s"] = f"({size}*4*2/1e9) / (time/1000)"

    result = tune_kernel("vector_add", kernel_string, size, args, tune_params, metrics=metrics)

    with open("vector_add.json", 'w') as fp:
        json.dump(result, fp)

    return result


if __name__ == "__main__":
    tune()
