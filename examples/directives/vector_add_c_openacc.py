#!/usr/bin/env python
"""This is a simple example for tuning C++ OpenACC code with the kernel tuner"""

from kernel_tuner import tune_kernel
from kernel_tuner.interface import auto_tune_kernel
from kernel_tuner.utils.directives import (
    DirectiveCode,
    OpenACC,
    Cxx,
    preprocess_directive_source,
)

code = """
#include <stdlib.h>

#define VECTOR_SIZE 1000000

int main(void) {
	int size = VECTOR_SIZE;
	float * a = (float *) malloc(VECTOR_SIZE * sizeof(float));
	float * b = (float *) malloc(VECTOR_SIZE * sizeof(float));
	float * c = (float *) malloc(VECTOR_SIZE * sizeof(float));

	#pragma tuner start vector_add a(float*:VECTOR_SIZE) b(float*:VECTOR_SIZE) c(float*:VECTOR_SIZE) size(int:VECTOR_SIZE)
	#pragma acc parallel vector_length(nthreads)
	#pragma acc loop
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
directive = DirectiveCode(OpenACC(), Cxx())
kernel_source, args = preprocess_directive_source("vector_add", code, directive)
print(args)

tune_params = dict()
tune_params["nthreads"] = [32 * i for i in range(1, 3)]
metrics = dict()
metrics["GB/s"] = lambda x: ((2 * 4 * len(args[0])) + (4 * len(args[0]))) / (x["time"] / 10**3) / 10**9

answer = [None, None, args[0] + args[1], None]

auto_tune_kernel(
    "vector_add",
    kernel_source,
    0,    
    tune_params,
    arguments=args,
    metrics=metrics,
    answer=answer,
    compiler_options=["-fast", "-acc=gpu"],
    compiler="nvc++",
)
