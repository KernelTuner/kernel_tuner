#!/usr/bin/env python
"""This is the minimal example from the README"""

import json
from collections import OrderedDict

import numpy
from kernel_tuner import tune_kernel
from kernel_tuner.observers.nvml import NVMLObserver

def tune():

    kernel_string = """
__kernel void vector_add(__global float *c, __global const float *a, __global const float *b, int n) {
    int i = get_global_id(0);
    if (i<n) {
    c[i] = a[i] + b[i];
    }
}
"""

    size = 80000000

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    nvmlobserver = NVMLObserver(["nvml_energy", "temperature"])

    metrics = OrderedDict()
    metrics["GFLOPS/W"] = lambda p: (size/1e9) / p["nvml_energy"]

    results, env = tune_kernel("vector_add", kernel_string, size, args, tune_params, observers=[nvmlobserver], metrics=metrics, iterations=32)

    with open("vector_add.json", 'w') as fp:
        json.dump(results, fp)

    return results


if __name__ == "__main__":
    tune()
