#!/usr/bin/env python
"""Example showing how to tune a HIP kernel that draws from a rocRAND state
initialized on the host, and updates that state after drawing numbers"""

import numpy
from kernel_tuner import tune_kernel, run_kernel

# rocrand_state_xorwow wraps a single xorwow_state:
#   unsigned int d, boxmuller_float_state, boxmuller_double_state; (12 bytes)
#   float boxmuller_float;                                         (4 bytes)
#   double boxmuller_double;                                       (8 bytes, needs 8-byte alignment)
#   unsigned int x[5];                                              (20 bytes)
# which sums to 44 bytes, padded to 48 for struct alignment. Unlike CUDA's
# curandState, those Box-Muller fields are dropped from the state entirely
# when rocRAND is built with ROCRAND_DETAIL_BM_NOT_IN_STATE defined, shrinking
# the struct to 24 bytes -- so this layout can vary by build, not just by
# version.
ROCRAND_STATE_SIZE = 48


def tune():
    # The hiprand_kernel.h header pulls in the host API and <type_traits>, which
    # does not compile with hiprtc (used by Kernel Tuner's HIP backend), so we
    # use rocRAND's device headers directly.
    kernel_string = """
    #define DRAWS_PER_THREAD 16
    #include <rocrand/rocrand_xorwow.h>

    // rocrand_uniform.h includes the host-side mtgp32 code, so do the (0, 1] conversion by hand
    __device__ float uniform(rocrand_state_xorwow *state) {
        const float inv = 2.3283064e-10f;  // 2^-32
        return inv + rocrand(state) * inv;
    }

    extern "C" __global__ void setup_kernel(rocrand_state_xorwow *state, unsigned long long seed, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i < n) {
            rocrand_init(seed, i, 0, &state[i]);
        }
    }

    __global__ void generate_random(float *output, rocrand_state_xorwow *state, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i < n) {
            rocrand_state_xorwow local_state = state[i];

            float sum = 0.0f;
            #pragma unroll unroll_draws
            for (int j = 0; j < DRAWS_PER_THREAD; j++) {
                sum += uniform(&local_state);
            }

            output[i] = sum;
            state[i] = local_state;
        }
    }
    """

    size = 10_000_000
    n = numpy.int32(size)
    seed = numpy.uint64(42)
    # rocrand_common.h includes <math.h>, which clashes with hiprtc's built-in
    # types; defining libstdc++'s include guard skips it (gcc's header layout)
    compiler_options = ["-O3", "-D_GLIBCXX_MATH_H"]

    # rocrand_state_xorwow is opaque, host-side content is irrelevant, only its size matters
    state = numpy.zeros(size * ROCRAND_STATE_SIZE, dtype=numpy.uint8)

    # initialize the rocRAND state once from a separate kernel, using a fixed block size
    setup_params = {"block_size_x": 256, "unroll_draws": 1}
    state = run_kernel(
        "setup_kernel",
        kernel_string,
        size,
        [state, seed, n],
        setup_params,
        lang="HIP",
        compiler_options=compiler_options,
    )[0]

    output = numpy.zeros(size).astype(numpy.float32)
    args = [output, state, n]

    tune_params = dict()
    tune_params["block_size_x"] = [32 * i for i in range(1, 33)]
    tune_params["unroll_draws"] = [1, 2, 4, 8, 16]

    # note: each benchmarked launch of generate_random advances the rocRAND state
    # further along its sequence, since the kernel writes the updated state back
    results, env = tune_kernel(
        "generate_random", kernel_string, size, args, tune_params, lang="HIP", compiler_options=compiler_options
    )

    return results


if __name__ == "__main__":
    tune()
