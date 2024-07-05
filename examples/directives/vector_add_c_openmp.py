#!/usr/bin/env python
"""This is a simple example for tuning C++ OpenACC code with the kernel tuner"""

from kernel_tuner import tune_kernel
from kernel_tuner.interface import auto_tune_kernel
from kernel_tuner.utils.directives import (
    DirectiveCode,
    OpenMP,
    Cxx,
)

code = """
#include <stdlib.h>
#include <omp.h>

#define VECTOR_SIZE 100000000

void vector_add(float *a, float *b, float *c) {
	#pragma tuner start vector_add a(float*:VECTOR_SIZE) b(float*:VECTOR_SIZE) c(float*:VECTOR_SIZE) size(int:VECTOR_SIZE)
    #pragma omp teams (nteams)
	#pragma omp parallel for num_threads(nthreads)
	for ( int i = 0; i < VECTOR_SIZE; i++ ) {
		c[i] = a[i] + b[i];
	}
	#pragma tuner stop
}
"""

# Extract tunable directive
directive = DirectiveCode(OpenMP(), Cxx())

tune_params = dict()
tune_params["nthreads"] = [16, 32]
tune_params["nteams"] = [2, 4]

auto_tune_kernel(
    "vector_add",
    code,
    0,
    tune_params=tune_params,
    compiler_options=["-fopenmp", "-mp=gpu"],
    compiler="nvc++",
    directive=directive
)
