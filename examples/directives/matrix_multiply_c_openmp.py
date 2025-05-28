#!/usr/bin/env python
"""This is an example tuning a naive matrix multiplication using the simplified directives interface"""

from kernel_tuner import tune_kernel, run_kernel
from kernel_tuner.utils.directives import Code, OpenMP, Cxx, process_directives

N = 4096

code = """
#define N 4096

void matrix_multiply(float *A, float *B, float *C) {
    #pragma tuner start mm A(float*:NN) B(float*:NN) C(float*:NN)
    float temp_sum = 0.0f;
    #pragma omp target
    #pragma omp teams distribute collapse(2)
    for ( int i = 0; i < N; i++) {
        for ( int j = 0; j < N; j++ ) {
            temp_sum = 0.0f;
            #pragma omp parallel for num_threads(nthreads) reduction(+:temp_sum)
            for ( int k = 0; k < N; k++ ) {
                temp_sum += A[(i * N) + k] * B[(k * N) + j];
            }
            C[(i * N) + j] = temp_sum;
        }
    }
    #pragma tuner stop
}
"""

# Extract tunable directive
app = Code(OpenMP(), Cxx())
dims = {"NN": N**2}
kernel_string, kernel_args = process_directives(app, code, user_dimensions=dims)

tune_params = dict()
tune_params["nthreads"] = [32 * i for i in range(1, 33)]
metrics = dict()
metrics["time_s"] = lambda x: x["time"] / 10**3
metrics["GB/s"] = lambda x: ((N**3 * 2 * 4) + (N**2 * 4)) / x["time_s"] / 10**9
metrics["GFLOP/s"] = lambda x: (N**3 * 3) / x["time_s"] / 10**9

# compute reference solution from CPU
results = run_kernel(
    "mm", kernel_string["mm"], 0, kernel_args["mm"], {"nthreads": 1}, compiler="nvc++", compiler_options=["-fast"]
)
answer = [None, None, results[2]]

tune_kernel(
    "mm",
    kernel_string["mm"],
    0,
    kernel_args["mm"],
    tune_params,
    metrics=metrics,
    answer=answer,
    compiler="nvc++",
    compiler_options=["-fast", "-mp=gpu"],
)
