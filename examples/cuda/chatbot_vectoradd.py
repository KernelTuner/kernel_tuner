#!/usr/bin/env python
"""This is the minimal example from the README"""

import numpy as np
import kernel_tuner
#from kernel_tuner import tune_kernel, run_kernel
#from kernel_tuner.file_utils import store_output_file, store_metadata_file
from chatGPT import ChatGPTuner
#from responses import *
import openai
from chatgpt_queries import *
import key


if __name__ == "__main__":
    # Default array size for testing
    openai.api_key = key.key()

    naive_kernel = """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
    }
    """

    size = 1000017
    problem_size = size

    a = (100 + 10*np.random.randn(size)).astype(np.float32)
    b = (10*np.random.randn(size)).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]
    kernel_name = 'add_vectors'
    tune_params = {'block_size_x': 256}

    verbose = True
    handler = ChatGPTuner(kernel_name,
                          naive_kernel,
                          size,
                          args,
                          tune_params,
                          compiler_options=['-allow-unsupported-compiler'],
                          prompt=None,
                          temperature=0.6,
                          verbose=verbose)

    result_kernel, tune_params = handler.vary_work_per_thread_x()

    print("Final result:")
    print(result_kernel)
    print(f"{tune_params=}")
