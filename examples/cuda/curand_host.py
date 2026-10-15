#!/usr/bin/env python
"""Example showing how to tune a CUDA kernel that draws numbers generated
with the cuRAND host API"""

import numpy
from kernel_tuner import tune_kernel
from cuda.bindings import driver
from nvmath.bindings import curand

DRAWS_PER_THREAD = 16


def tune():
    kernel_string = """
    __global__ void draw_random(float *output, float *random, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i < n) {
            float sum = 0.0f;
            #pragma unroll unroll_draws
            for (int j = 0; j < DRAWS_PER_THREAD; j++) {
                sum += random[(j * n) + i];
            }

            output[i] = sum;
        }
    }
    """

    size = 10_000_000
    compiler_options = ["--dopt=on"]

    output = numpy.zeros(size).astype(numpy.float32)
    random = numpy.zeros(size * DRAWS_PER_THREAD).astype(numpy.float32)
    n = numpy.int32(size)
    args = [output, random, n]

    # use cuRAND to generate the random numbers on the GPU via the host API
    err, stream = driver.cuStreamCreate(driver.CUstream_flags.CU_STREAM_NON_BLOCKING.value)
    err, device_random = driver.cuMemAlloc(random.nbytes)
    gen = curand.create_generator(curand.RngType.PSEUDO_PHILOX4_32_10)
    curand.set_pseudo_random_generator_seed(gen, 42)
    curand.set_stream(gen, int(stream))
    curand.generate_uniform(gen, int(device_random), size * DRAWS_PER_THREAD)
    driver.cuStreamSynchronize(stream)
    driver.cuMemcpyDtoH(random.ctypes.data, device_random, random.nbytes)
    driver.cuMemFree(device_random)
    curand.destroy_generator(gen)
    driver.cuStreamDestroy(stream)

    tune_params = dict()
    tune_params["DRAWS_PER_THREAD"] = [DRAWS_PER_THREAD]
    tune_params["block_size_x"] = [32 * i for i in range(1, 33)]
    tune_params["unroll_draws"] = [1, 2, 4, 8, 16]

    results, env = tune_kernel(
        "draw_random", kernel_string, size, args, tune_params, lang="nvcuda", compiler_options=compiler_options
    )

    return results


if __name__ == "__main__":
    tune()
