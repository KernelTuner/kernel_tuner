#!/usr/bin/env python
"""Example showing how to tune a HIP kernel that draws numbers generated
with the hipRAND host API"""

import numpy
from kernel_tuner import tune_kernel
from hip import hip, hiprand

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

    output = numpy.zeros(size).astype(numpy.float32)
    random = numpy.zeros(size * DRAWS_PER_THREAD).astype(numpy.float32)
    n = numpy.int32(size)
    args = [output, random, n]

    # use hipRAND to generate the random numbers on the GPU via the host API
    hip.hipInit(0)
    hip.hipSetDevice(0)
    err, stream = hip.hipStreamCreate()
    err, device_random = hip.hipMalloc(random.nbytes)
    err, gen = hiprand.hiprandCreateGenerator(hiprand.hiprandRngType.HIPRAND_RNG_PSEUDO_PHILOX4_32_10)
    hiprand.hiprandSetPseudoRandomGeneratorSeed(gen, 42)
    hiprand.hiprandSetStream(gen, stream)
    hiprand.hiprandGenerateUniform(gen, device_random, random.size)
    hip.hipStreamSynchronize(stream)
    hip.hipMemcpy(random, device_random, random.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost)
    hip.hipFree(device_random)
    hiprand.hiprandDestroyGenerator(gen)
    hip.hipStreamDestroy(stream)

    # Check that something was actually generated
    assert abs(random.mean() - 0.5) < 1e-3

    expected = random.reshape(DRAWS_PER_THREAD, size).sum(axis=0, dtype=numpy.float32)

    tune_params = dict()
    tune_params["DRAWS_PER_THREAD"] = [DRAWS_PER_THREAD]
    tune_params["block_size_x"] = [32 * i for i in range(1, 33)]
    tune_params["unroll_draws"] = [1, 2, 4, 8, 16]

    results, env = tune_kernel(
        "draw_random",
        kernel_string,
        size,
        args,
        tune_params,
        lang="HIP",
        answer=[expected, None, None],
        atol=1e-3,
    )

    return results


if __name__ == "__main__":
    tune()
