#!/usr/bin/env python
"""Minimal example of vector add using jinja2 templates"""

import json
import numpy
from kernel_tuner import tune_kernel

from jinja2 import Template

def tune():

    kernel_template = Template("""
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * {{ block_size_x }} + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """)

    size = 10000000

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    result = tune_kernel("vector_add", kernel_template.render, size, args, tune_params, lang="CUDA")

    with open("vector_add.json", 'w') as fp:
        json.dump(result, fp)

    return result


if __name__ == "__main__":
    tune()
