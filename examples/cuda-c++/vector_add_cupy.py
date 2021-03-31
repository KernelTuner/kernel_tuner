#!/usr/bin/env python
"""This is the minimal example from the README converted to C++11"""

import json
import numpy
from kernel_tuner import tune_kernel

def tune():

    kernel_string = """
template<typename T>__global__ void vector_add(T *c, T *__restrict__ a, T *b, int n) {
    auto i = blockIdx.x * block_size_x + threadIdx.x;
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
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    result, env = tune_kernel("vector_add<float>", kernel_string, size, args, tune_params, lang="cupy")

    with open("vector_add.json", 'w') as fp:
        json.dump(result, fp)

    return result


if __name__ == "__main__":
    tune()
