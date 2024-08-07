#!/usr/bin/env python
"""This is an example tuning a naive matrix multiplication using the simplified directives interface"""

from kernel_tuner import tune_kernel
from kernel_tuner.utils.directives import (
    Code,
    OpenACC,
    Cxx,
    process_directives
)

code = """
#include <stdlib.h>

#define N 4096

void matrix_multiply(float *A, float *B, float *C) {
    #pragma tuner start mm A(float*:NN) B(float*:NN) C(float*:NN)
    #pragma acc parallel vector_length(nthreads)
    #pragma acc loop
    for ( i = 0; i < N; i++) {
        for ( j = 0; j < N; j++ ) {
            for ( k = 0; k < N; k++ ) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    #pragma tuner stop
}
"""

# Extract tunable directive
app = Code(OpenACC(), Cxx())
dims = {"NN": 4096*4096}
kernel_string, kernel_args = process_directives(app, code, user_dimensions=dims)

tune_params = dict()
tune_params["nthreads"] = [32 * i for i in range(1, 33)]
metrics = dict()
metrics["GB/s"] = lambda x: (4096 * 4096 * 4) / (x["time"] / 10**3) / 10**9

tune_kernel(
    "mm",
    kernel_string["mm"],
    0,
    kernel_args["mm"],
    tune_params,
    metrics=metrics,
    compiler_options=["-fast", "-acc=gpu"],
    compiler="nvc++",
)
