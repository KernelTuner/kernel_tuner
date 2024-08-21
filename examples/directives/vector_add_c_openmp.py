#!/usr/bin/env python
"""This is a simple example for tuning C++ OpenACC code with the kernel tuner"""

from kernel_tuner import tune_kernel
from kernel_tuner.utils.directives import Code, OpenMP, Cxx, process_directives

code = """
#include <stdlib.h>

#define VECTOR_SIZE 1000000

int main(void) {
	int size = VECTOR_SIZE;
	float * a = (float *) malloc(VECTOR_SIZE * sizeof(float));
	float * b = (float *) malloc(VECTOR_SIZE * sizeof(float));
	float * c = (float *) malloc(VECTOR_SIZE * sizeof(float));

	#pragma tuner start vector_add a(float*:VECTOR_SIZE) b(float*:VECTOR_SIZE) c(float*:VECTOR_SIZE) size(int:VECTOR_SIZE)
	#pragma omp target teams num_threads(nthreads)
	#pragma omp distribute parallel for
	for ( int i = 0; i < size; i++ ) {
		c[i] = a[i] + b[i];
	}
	#pragma tuner stop

	free(a);
	free(b);
	free(c);
}
"""

# Extract tunable directive
app = Code(OpenMP(), Cxx())
kernel_string, kernel_args = process_directives(app, code)

tune_params = dict()
tune_params["nthreads"] = [32 * i for i in range(1, 33)]
metrics = dict()
metrics["GB/s"] = (
    lambda x: ((2 * 4 * len(kernel_args["vector_add"][0])) + (4 * len(kernel_args["vector_add"][0])))
    / (x["time"] / 10**3)
    / 10**9
)

answer = [None, None, kernel_args["vector_add"][0] + kernel_args["vector_add"][1], None]

tune_kernel(
    "vector_add",
    kernel_string["vector_add"],
    0,
    kernel_args["vector_add"],
    tune_params,
    metrics=metrics,
    answer=answer,
    compiler_options=["-fast", "-mp=gpu"],
    compiler="nvc++",
)
