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
from collections import OrderedDict
import key


if __name__ == "__main__":
    # Default array size for testing
    openai.api_key = key.key()

    naive_kernel = """
__global__ void matrix_multiply_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}
    """

    M = np.int32(512)
    N = np.int32(512)
    K = N
    problem_size = (M, N)
    size = np.prod(problem_size)

    A = np.random.randn(*problem_size).astype(np.float32)
    B = np.random.randn(*problem_size).astype(np.float32)
    C = np.zeros_like(A)

    args = [A, B, C, M, N, K]

    kernel_name = 'matrix_multiply_kernel'
    tune_params = {'block_size_x': 32, 'block_size_y': 32}

    verbose = True
    handler = ChatGPTuner(kernel_name,
                          naive_kernel,
                          problem_size,
                          args,
                          tune_params,
                          compiler_options=['-allow-unsupported-compiler'],
                          temperature=0.6,
                          verbose=verbose)

    # Make x dimension tunable
    tunable_kernel, tune_params = handler.vary_work_per_thread_x()

    tune_params = {k:[v] for k,v in tune_params.items()}
    tune_params["tile_size_x"] = [2**i for i in range(4)]
    print(tune_params)
    grid_div_x = ["block_size_x", "tile_size_x"]
    #grid_div_y = ["block_size_y", "tile_size_y"]
    #restrict = ["block_size_x==block_size_y*tile_size_y"]

    # Tune this kernel
    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda p : (2*4096**3/1e9) / (p["time"] / 1e3)

    answer = [A, B, np.dot(A,B), None, None, None]
    res, env = kernel_tuner.tune_kernel(kernel_name,
                                        tunable_kernel,
                                        problem_size,
                                        args,
                                        tune_params,
                                        answer=answer,
                                        #grid_div_y=grid_div_y,
                                        grid_div_x=grid_div_x,
                                        #restrictions=restrict,
                                        verbose=True,
                                        iterations=32,
                                        compiler_options=['-allow-unsupported-compiler'],
                                        metrics=metrics)
