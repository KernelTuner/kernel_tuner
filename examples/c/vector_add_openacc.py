#!/usr/bin/env python
"""This is a simple example for tuning C++ OpenACC code with the kernel tuner"""

import numpy
from kernel_tuner import tune_kernel
from kernel_tuner.util import (
    extract_directive_signature,
    extract_directive_code,
    extract_preprocessor,
    wrap_cpp_timing,
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
	#pragma acc parallel num_gangs(ngangs) vector_length(nthreads)
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
kernel_string = "\n".join(preprocessor) + "\n#include <chrono>\n#include <ratio>\n"
directive_signatures = extract_directive_signature(code, kernel_name="vector_add")
kernel_string += 'extern "C" ' + directive_signatures["vector_add"] + "{\n"
directive_codes = extract_directive_code(code, kernel_name="vector_add")
kernel_string += wrap_cpp_timing(directive_codes["vector_add"]) + "\n}"

size = 65536

a = numpy.random.randn(size).astype(numpy.float32)
b = numpy.random.randn(size).astype(numpy.float32)
c = numpy.zeros_like(b)
n = numpy.int32(size)

args = [a, b, c, n]

tune_params = OrderedDict()
tune_params["ngangs"] = [2**i for i in range(0, 15)]
tune_params["nthreads"] = [2**i for i in range(0, 11)]

answer = [None, None, a + b, None]

tune_kernel(
    "vector_add",
    kernel_string,
    size,
    args,
    tune_params,
    answer=answer,
    compiler_options=["-fast", "-acc=gpu"],
    compiler="nvc++",
)
