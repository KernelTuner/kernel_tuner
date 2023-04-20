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
__global__ void sum_array(float *d_array, float *d_sum, int size) {
    __shared__ float sdata[block_size_x];

    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy data from global memory to shared memory
    if (tid < size) {
        sdata[threadIdx.x] = d_array[tid];
    } else {
        sdata[threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Reduce sum in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write final sum to global memory
    if (threadIdx.x == 0) {
        d_sum[blockIdx.x] = sdata[0];
    }
}
    """

    size = 1000017
    problem_size = size

    input_data = (100 + 10*np.random.randn(size)).astype(np.float32)
    output_data = np.zeros_like(input_data)
    n = np.int32(size)

    args = [input_data, output_data, n]
    kernel_name = 'sum_array'
    tune_params = {'block_size_x': 256}

    reference = [None, np.sum(input_data), None]
    def verify_partial_reduce(cpu_result, gpu_result, atol=None):
        return np.isclose(cpu_result[1], np.sum(gpu_result[1]), atol=atol)

    verbose = True
    handler = ChatGPTuner(kernel_name,
                          naive_kernel,
                          size,
                          args,
                          tune_params,
                          compiler_options=['-allow-unsupported-compiler'],
                          temperature=0.6,
                          answer=reference,
                          verify=verify_partial_reduce,
                          verbose=verbose)

    handler.vary_work_per_thread_x()
