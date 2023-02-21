#!/usr/bin/env python
"""This is the minimal example from the README"""

import numpy
import kernel_tuner
from kernel_tuner import tune_kernel, run_kernel
from kernel_tuner.file_utils import store_output_file, store_metadata_file
from chatGPT import validate_kernel



if __name__ == "__main__":
    # Default array size for testing
    size = 128


    """
    >> Query to ChatGPT:
        "Can you write a CUDA kernel for multypling matrices"

    << Response:
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

    kernel_string = """
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

    kname, kstring, size, args, tune_params, defines = validate_kernel(
                    kernel_string,
                    size,
                    compiler_options=['-allow-unsupported-compiler'])

    print(tune_params)
    """
    >> Query to ChatGPT:
        "Can you make it tunable?"

    << Response:
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

    kernel_string = """
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

    kname, kstring, size, args, tune_params, defines = validate_kernel(
                    kernel_string,
                    size,
                    compiler_options=['-allow-unsupported-compiler'])

    print(tune_params)
    # Our code says that the tune_params are:
    # blockDim.x and blockDim.y

    # Ask chatGPT for good tuning range:
    # >> Can you suggest some tuning values for blockDimy and blockDimx?
    # << It sort of stopped talking at some point (maybe too much traffic)

    # Ask chatGPT for a list of numbers:
    # >> Can you suggest a list of numbers to try for blockDim.y and blockDim.x?
    # << It replied with  16, 32, 64, 128, 256 (for both)

    tune_values = [16, 32, 64, 128, 256]
    for k,v in tune_params.items():
        tune_params[k] = tune_values
    print(tune_params)

    # Run tuning with these parsed values:
    results, env = tune_kernel(kname,
                               kstring,
                               size,
                               args,
                               tune_params,
                               defines=defines,
                               compiler_options=['-allow-unsupported-compiler'])

    # Store the tuning results in an output file
    store_output_file("ChatGPT_matmul.json", results, tune_params)

    # Store the metadata of this run
    store_metadata_file("ChatGPT_matmul-metadata.json")
