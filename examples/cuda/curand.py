#!/usr/bin/env python
"""Example showing how to tune a CUDA kernel that draws from a curand state
initialized on the host, and updates that state after drawing numbers"""

import numpy
from kernel_tuner import tune_kernel, run_kernel

# curandStateXORWOW's fields (unsigned int d, v[5]; int boxmuller_flag,
# boxmuller_flag_double; float boxmuller_extra; double boxmuller_extra_double)
# sum to 44 bytes, padded to 48 for the trailing double's 8-byte alignment.
CURAND_STATE_SIZE = 48


def tune():
    kernel_string = """
    #define DRAWS_PER_THREAD 16
    #include <curand_kernel.h>

    extern "C" __global__ void setup_kernel(curandState *state, unsigned long long seed, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i < n) {
            curand_init(seed, i, 0, &state[i]);
        }
    }

    __global__ void generate_random(float *output, curandState *state, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i < n) {
            curandState local_state = state[i];

            float sum = 0.0f;
            #pragma unroll unroll_draws
            for (int j = 0; j < DRAWS_PER_THREAD; j++) {
                sum += curand_uniform(&local_state);
            }

            output[i] = sum;
            state[i] = local_state;
        }
    }
    """

    size = 10_000_000
    n = numpy.int32(size)
    seed = numpy.uint64(42)
    compiler_options = ["-O3"]

    # curandState is opaque, host-side content is irrelevant, only its size matters
    state = numpy.zeros(size * CURAND_STATE_SIZE, dtype=numpy.uint8)

    # initialize the curand state from a separate kernel, using a fixed block size
    setup_params = {"block_size_x": 256}
    state = run_kernel("setup_kernel", kernel_string, size, [state, seed, n], setup_params, lang="nvcuda", compiler_options=compiler_options)[0]

    output = numpy.zeros(size).astype(numpy.float32)
    args = [output, state, n]

    tune_params = dict()
    tune_params["block_size_x"] = [32 * i for i in range(33)]
    tune_params["unroll_draws"] = [1, 2, 4, 8, 16]

    # note: each benchmarked launch of generate_random advances the curand state
    # further along its sequence, since the kernel writes the updated state back
    results, env = tune_kernel("generate_random", kernel_string, size, args, tune_params, lang="nvcuda", compiler_options=compiler_options)

    return results


if __name__ == "__main__":
    tune()
