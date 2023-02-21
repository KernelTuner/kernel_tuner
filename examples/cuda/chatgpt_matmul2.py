#!/usr/bin/env python
"""This is the minimal example from the README"""

import numpy
import kernel_tuner
from kernel_tuner import tune_kernel, run_kernel
from kernel_tuner.file_utils import store_output_file, store_metadata_file
from chatGPT import validate_kernel



if __name__ == "__main__":
    # Default array size for testing
    size = 512
    #TODO: Why does it die if I put 2048?

    # So KT warned that there was no thread block dimension,
    # I pasted the warning into chatGPT and asked if he could add it to the kernel.

    """
    >> Query to ChatGPT:
        Can you add a threadblock dimension to the kernel code?

    << Response:
    __global__ void matrix_multiply_kernel(float* A, float* B, float* C, int M, int N, int K, int block_size) {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * block_size + ty;
        int col = bx * block_size + tx;

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        float sum = 0.0f;
        for (int k = 0; k < (N + block_size - 1) / block_size; k++) {
            if (k * block_size + tx < N && row < M) {
                As[ty][tx] = A[row * N + k * block_size + tx];
            } else {
                As[ty][tx] = 0.0f;
            }
            if (k * block_size + ty < N && col < K) {
                Bs[ty][tx] = B[(k * block_size + ty) * K + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            __syncthreads();
            for (int i = 0; i < block_size; i++) {
                sum += As[ty][i] * Bs[i][tx];
            }
            __syncthreads();
        }
        if (row < M && col < K) {
            C[row * K + col] = sum;
        }
    }
    """

    # This is a bit strange with block_size as an argument. I said:
    # >> It is a bit strange with block_size as an argument, can you try again?
    # It gave the following kernel_string:

    kernel_string = """
    __global__ void matrix_multiply_kernel(float* A, float* B, float* C, int M, int N, int K) {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * blockDim.y + ty;
        int col = bx * blockDim.x + tx;

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        float sum = 0.0f;
        for (int k = 0; k < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
            if (k * BLOCK_SIZE + tx < N && row < M) {
                As[ty][tx] = A[row * N + k * BLOCK_SIZE + tx];
            } else {
                As[ty][tx] = 0.0f;
            }
            if (k * BLOCK_SIZE + ty < N && col < K) {
                Bs[ty][tx] = B[(k * BLOCK_SIZE + ty) * K + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            __syncthreads();
            for (int i = 0; i < BLOCK_SIZE; i++) {
                sum += As[ty][i] * Bs[i][tx];
            }
            __syncthreads();
        }
        if (row < M && col < K) {
            C[row * K + col] = sum;
        }
    }
    """


    # This is still hard to parse. I decided to ask it:
    # >> Can you write a matrix multiplication CUDA kernel where the tunable
    #     parameters contain 'TUNE' in their variable names?
    # Resulting kernel string:

    kernel_string = """
    __global__ void matrix_multiply_kernel(float* A, float* B, float* C, int M, int N, int K, int TUNE_block_size, int TUNE_num_blocks_x, int TUNE_num_blocks_y) {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * TUNE_block_size + ty;
        int col = bx * TUNE_block_size + tx;

        __shared__ float As[TUNE_block_size][TUNE_block_size];
        __shared__ float Bs[TUNE_block_size][TUNE_block_size];

        float sum = 0.0f;
        for (int k = 0; k < (N + TUNE_block_size - 1) / TUNE_block_size; k++) {
            if (k * TUNE_block_size + tx < N && row < M) {
                As[ty][tx] = A[row * N + k * TUNE_block_size + tx];
            } else {
                As[ty][tx] = 0.0f;
            }
            if (k * TUNE_block_size + ty < N && col < K) {
                Bs[ty][tx] = B[(k * TUNE_block_size + ty) * K + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            __syncthreads();
            for (int i = 0; i < TUNE_block_size; i++) {
                sum += As[ty][i] * Bs[i][tx];
            }
            __syncthreads();
        }
        if (row < M && col < K) {
            C[row * K + col] = sum;
        }
    }
    """

    # This failed to compile, I asked it to try again:
    # >> This fails to compile for me, can you try again?

    kernel_string = """
    __global__ void matrix_multiply_kernel(float* A, float* B, float* C, int M, int N, int K, int TUNE_block_size, int TUNE_num_blocks_x, int TUNE_num_blocks_y) {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * TUNE_block_size + ty;
        int col = bx * TUNE_block_size + tx;

        __shared__ float As[TUNE_block_size][TUNE_block_size];
        __shared__ float Bs[TUNE_block_size][TUNE_block_size];

        float sum = 0.0f;
        for (int k = 0; k < (N + TUNE_block_size - 1) / TUNE_block_size; k++) {
            if (k * TUNE_block_size + tx < N && row < M) {
                As[ty][tx] = A[row * N + k * TUNE_block_size + tx];
            } else {
                As[ty][tx] = 0.0f;
            }
            if (k * TUNE_block_size + ty < N && col < K) {
                Bs[ty][tx] = B[(k * TUNE_block_size + ty) * K + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            __syncthreads();
            for (int i = 0; i < TUNE_block_size; i++) {
                sum += As[ty][i] * Bs[i][tx];
            }
            __syncthreads();
        }
        if (row < M && col < K) {
            C[row * K + col] = sum;
        }
    }
    """

    # Again, did not work, why TUNE_num_blocks_x, TUNE_num_blocks_y are not used?
    # >> TUNE_num_blocks_y is not used here? I think there are unnecessary tunable variables here
    kernel_string = """
    __global__ void matrix_multiply_kernel(float* A, float* B, float* C, int M, int N, int K, int TUNE_block_size) {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * TUNE_block_size + ty;
        int col = bx * TUNE_block_size + tx;

        __shared__ float As[TUNE_block_size][TUNE_block_size];
        __shared__ float Bs[TUNE_block_size][TUNE_block_size];

        float sum = 0.0f;
        for (int k = 0; k < (N + TUNE_block_size - 1) / TUNE_block_size; k++) {
            if (k * TUNE_block_size + tx < N && row < M) {
                As[ty][tx] = A[row * N + k * TUNE_block_size + tx];
            } else {
                As[ty][tx] = 0.0f;
            }
            if (k * TUNE_block_size + ty < N && col < K) {
                Bs[ty][tx] = B[(k * TUNE_block_size + ty) * K + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            __syncthreads();
            for (int i = 0; i < TUNE_block_size; i++) {
                sum += As[ty][i] * Bs[i][tx];
            }
            __syncthreads();
        }
        if (row < M && col < K) {
            C[row * K + col] = sum;
        }
    }
    """

    # Confusion because TUNE_block_size is both in arguments and a tunable parameter.
    # >> This looks good but can you remove TUNE_block_size as function argument? I want it as a tunable parameter that is defined outside the function
    kernel_string = """
    __global__ void matrix_multiply_kernel(float* A, float* B, float* C, int M, int N, int K) {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * TUNE_block_size + ty;
        int col = bx * TUNE_block_size + tx;

        __shared__ float As[TUNE_block_size][TUNE_block_size];
        __shared__ float Bs[TUNE_block_size][TUNE_block_size];

        float sum = 0.0f;
        for (int k = 0; k < (N + TUNE_block_size - 1) / TUNE_block_size; k++) {
            if (k * TUNE_block_size + tx < N && row < M) {
                As[ty][tx] = A[row * N + k * TUNE_block_size + tx];
            } else {
                As[ty][tx] = 0.0f;
            }
            if (k * TUNE_block_size + ty < N && col < K) {
                Bs[ty][tx] = B[(k * TUNE_block_size + ty) * K + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            __syncthreads();
            for (int i = 0; i < TUNE_block_size; i++) {
                sum += As[ty][i] * Bs[i][tx];
            }
            __syncthreads();
        }
        if (row < M && col < K) {
            C[row * K + col] = sum;
        }
    }
    """

    # Again does not really work, maybe has to do with As, Bs etc.
    # >> Can you rewrite it without using shared memory As and Bs?
    kernel_string = """
    __global__ void matrix_multiply_kernel(float* A, float* B, float* C, int M, int N, int K) {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * blockDim.y + ty;
        int col = bx * blockDim.x + tx;

        float sum = 0.0f;
        for (int i = 0; i < N; i += blockDim.x) {
            if (row < M && i + tx < N && col < K) {
                sum += A[row * N + i + tx] * B[(i + tx) * K + col];
            }
        }

        if (row < M && col < K) {
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
    # >> Can you recommend some numbers to try for blockDim.x and blockDim.y?
    # << It recommends (16,16), (32,32), (64,64)

    # I ask:
    # >> Can you recommend tuning ranges for blockDim.x and blockDim.y where they are not always equal?
    # << It recommends:

    tune_values = [4, 8, 16, 32, 64, 128, 256]
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
    store_output_file("ChatGPT_matmul2.json", results, tune_params)

    # Store the metadata of this run
    store_metadata_file("ChatGPT_matmul2-metadata.json")
