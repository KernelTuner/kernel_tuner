#!/usr/bin/env python
"""This is a simple example for tuning C++ OpenACC code with the kernel tuner"""

from kernel_tuner import tune_kernel
from kernel_tuner.utils.directives import (
    extract_directive_signature,
    extract_directive_code,
    extract_preprocessor,
    generate_directive_function,
    extract_directive_data,
    allocate_signature_memory,
)
from collections import OrderedDict

code = """
#include <stdlib.h>

#define VECTOR_SIZE 65536

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

# Extract tunable directive and generate kernel_string
preprocessor = extract_preprocessor(code)
signature = extract_directive_signature(code)
body = extract_directive_code(code)
kernel_string = generate_directive_function(
    preprocessor, signature["vector_add"], body["vector_add"]
)

# Allocate memory on the host
data = extract_directive_data(code)
args = allocate_signature_memory(data["vector_add"], preprocessor)

tune_params = OrderedDict()
tune_params["nthreads"] = [32*i for i in range(1, 33)]

answer = [None, None, args[0] + args[1], None]

tune_kernel(
    "vector_add",
    kernel_string,
    0,
    args,
    tune_params,
    answer=answer,
    compiler_options=["-fast", "-acc=gpu"],
    compiler="nvc++",
)
