#!/usr/bin/env python
"""This is the minimal example from the README"""

import numpy
import kernel_tuner
from kernel_tuner import tune_kernel, run_kernel
from kernel_tuner.file_utils import store_output_file, store_metadata_file
from chatGPT import validate_kernel



if __name__ == "__main__":
    # Default array size for testing
    size = 100000


    """
    >> Query to ChatGPT:
        "Can you write a CUDA kernel for adding vectors"

    << Response:
        __global__ void add_vectors(float* a, float* b, float* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
    """

    kernel_string = """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    kname, kstring, size, args, tune_params, defines = validate_kernel(
                    kernel_string,
                    size,
                    compiler_options=['-allow-unsupported-compiler'])

    """
    >> Query to ChatGPT:
        "Can you make it tunable?"

    << Response:
        __global__ void add_vectors(float* a, float* b, float* c, int n) {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (int i = tid; i < n; i += stride) {
                c[i] = a[i] + b[i];
            }
        }
    """
    #NOTE: ChatGPT also told us that blockDim.x and gridDim.x are
    # tunable here
    kernel_string = """
    __global__ void add_vectors(float* a, float* b, float* c, int n) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;

        for (int i = tid; i < n; i += stride) {
            c[i] = a[i] + b[i];
        }
    }
    """

    kname, kstring, size, args, tune_params, defines = validate_kernel(
                    kernel_string,
                    size,
                    compiler_options=['-allow-unsupported-compiler'])


    # CHAT GPT TIMED OUT, another try:
    kernel_string = """
    __global__ void vectorAdd(float *a, float *b, float *c, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    """

    kname, kstring, size, args, tune_params, defines = validate_kernel(
                    kernel_string,
                    size,
                    compiler_options=['-allow-unsupported-compiler'])


    #Make it tunable: this returned the same thing
    kernel_string = """
    __global__ void vectorAdd(float *a, float *b, float *c, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    """

    kname, kstring, size, args, tune_params, defines = validate_kernel(
                    kernel_string,
                    size,
                    compiler_options=['-allow-unsupported-compiler'])

    # Ask chatGPT for good tuning range:
    # What would be a good range for blockDim.x tuning?
    # << It gave a long story about how to choose it

    # Ask chatGPT for a list of numbers: got this:
    tune_values = [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
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
    store_output_file("ChatGPT_vector_add.json", results, tune_params)

    # Store the metadata of this run
    store_metadata_file("ChatGPT_vector_add-metadata.json")
