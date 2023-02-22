#!/usr/bin/env python
"""This is the minimal example from the README"""

import numpy as np
import kernel_tuner
from kernel_tuner import tune_kernel, run_kernel
from kernel_tuner.file_utils import store_output_file, store_metadata_file
from chatGPT import validate_kernel
from responses import *


chatGPT_queries = {
    "vary_work_per_thread" : {
        "Initial_2elem": lambda kernel_string:
            f"""
            Hi ChatGPT! I would like to ask you to rewrite some CUDA code for me.

            For the remainder of this conversation, please stick to the following:
            * Do not change the argument list of the kernel.

            Our starting point kernel for this conversation is:

            {kernel_string}

            Can you rewrite the kernel such that each thread processes 2 elements.
            """,
        "Tunable_nr_elem": lambda:
            f"""
            Please rewrite the code introducing a for loop to allow the number of elements processed by each thread to vary. This number is specified using a C preprocessor defined constant? Please call this constant 'tile_size_x' in lower case. You can omit the definition of this C preprocessor constant.
            """,
        "Incorrect_kernel": lambda:
            f"""
            This kernel does not produce the correct result, can you try again?
            """,
        "Markdown_response": lambda:
            f"""
            Thank you, the response is in markdown, can you make it a code block?
            """,
        "Fails_to_compile": lambda:
            f"""
            This kernel does not compile for me, can you try again?
            """,
    }
}

#TODO: Remove from response lines before __global__


def validate_kernel(kernel_name, kernel_string, size, args, tune_params,
                    compiler_options=None, answer=None, **kwargs):
    # Does the kernel code compile and run
    try:
        output = run_kernel(kernel_name,
                   kernel_string,
                   size,
                   args,
                   tune_params,
                   compiler_options=compiler_options,
                   **kwargs)

        res = True
        if answer:
            for i,ans in enumerate(answer):
                if not ans is None:
                    res = res and np.allclose(output[i], ans)

        return res
    except Exception as e:
        print(e)
        return e


def vary_work_per_thread_kernel(kernel_name, naive_kernel, size, args, tune_params,
                                compiler_options=None):
    # For testing
    response_list = responses10

    # First obtain a answer on random input to validate if kernels are correct in future
    answer = run_kernel(kernel_name,
                        naive_kernel,
                        size,
                        args,
                        tune_params,
                        compiler_options=['-allow-unsupported-compiler'])

    # Initial query to process 2 elements per thread
    query = chatGPT_queries['vary_work_per_thread']['Initial_2elem'](naive_kernel)
    tune_params['tile_size_x'] = 2

    # Get response from chatGPT
    correct = False
    iter_count = 0
    while not correct:
        iter_count += 1
        response_kernel = response_list[iter_count-1]
        print(f"Testing chatGPT query {iter_count}")

        correct = validate_kernel(kernel_name,
                                response_kernel,
                                size,
                                args,
                                tune_params,
                                answer=answer,
                                grid_div_x=['block_size_x', 'tile_size_x'],
                                compiler_options=['-allow-unsupported-compiler'])
        if isinstance(correct, Exception):
            print("There was an error in compiling and running the kernel.")
            # Decide whether to resubmit with some error message
            query = chatGPT_queries['vary_work_per_thread']['Fails_to_compile']()
            # Append response from chatGPT to responses
        else:
            print("Kernel is correct is", correct)
            query = chatGPT_queries['vary_work_per_thread']['Incorrect_kernel']()
            # Append response from chatGPT to responses

        # Break the while loop if conditions are met
        if iter_count >= 10:
            print("Failed to obtain a valid kernel from chatGPT...")
            break
        if correct:
            break


    # Query to process make elements per thread tunable
    query = chatGPT_queries['vary_work_per_thread']['Tunable_nr_elem']()
    tune_params['tile_size_x'] = 8

    # Get response from chatGPT
    correct = False
    while not correct:
        iter_count += 1
        response_kernel = response_list[iter_count-1]
        print(f"Testing chatGPT query {iter_count}")

        correct = validate_kernel(kernel_name,
                                response_kernel,
                                size,
                                args,
                                tune_params,
                                answer=answer,
                                grid_div_x=['block_size_x', 'tile_size_x'],
                                compiler_options=['-allow-unsupported-compiler'])
        if isinstance(correct, Exception):
            print("There was an error in compiling and running the kernel.")
            # Decide whether to resubmit with some error message
            query = chatGPT_queries['vary_work_per_thread']['Fails_to_compile']()
            # Append response from chatGPT to responses
        else:
            print("Kernel is correct is", correct)
            query = chatGPT_queries['vary_work_per_thread']['Incorrect_kernel']()
            # Append response from chatGPT to responses

        # Break the while loop if conditions are met
        if iter_count >= 10:
            print("Failed to obtain a valid kernel from chatGPT...")
            break
        if correct:
            break



if __name__ == "__main__":
    # Default array size for testing
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

    a = 100 + 10*np.random.randn(size).astype(np.float32)
    b = 10*np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]
    kernel_name = 'add_vectors'
    tune_params = {'block_size_x': 256}

    vary_work_per_thread_kernel(kernel_name,
                        naive_kernel,
                        size,
                        args,
                        tune_params,
                        compiler_options=['-allow-unsupported-compiler'])


    if False:
        answer = run_kernel(kernel_name,
                            naive_kernel,
                            size,
                            args,
                            tune_params,
                            compiler_options=['-allow-unsupported-compiler'])

        kernel_string = """
        __global__ void add_vectors(float* a, float* b, float* c, int n) {
            int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x * 2;
            if (i < n) {
                c[i] = a[i] + b[i];
                if (i + 1 < n) {
                    c[i + 1] = a[i + 1] + b[i + 1];
                }
            }
        }
        """

        correct = validate_kernel(kernel_name,
                                kernel_string,
                                size,
                                args,
                                tune_params,
                                answer=answer,
                                compiler_options=['-allow-unsupported-compiler'])
        print("Kernel is correct is", correct)
