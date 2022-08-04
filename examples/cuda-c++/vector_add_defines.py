#!/usr/bin/env python
""" This is the example demonstrates how to use Kernel Tuner
    to insert tunable parameters into template arguments
    without using any C preprocessor defines
"""

import numpy as np
import kernel_tuner as kt

def tune():

    kernel_string = """
template<typename T, int blockSize>
__global__ void vector_add(T *c, T *a, T *b, int n) {
    auto i = blockIdx.x * blockSize + threadIdx.x;
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
"""

    size = 10000000

    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    result, env = kt.tune_kernel("vector_add<float, block_size_x>", kernel_string, size, args, tune_params, defines={})

    return result


if __name__ == "__main__":
    tune()
