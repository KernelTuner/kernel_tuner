#!/usr/bin/env python
"""This is a simple example for tuning C++ OpenACC code with the kernel tuner"""
import numpy as np

from kernel_tuner import tune_kernel
from kernel_tuner.utils.directives import Code, OpenACC, Cxx, process_directives
from kernel_tuner.observers import BenchmarkObserver


# Naive Python histogram implementation
def histogram(vector, hist):
    for i in range(0, len(vector)):
        hist[vector[i]] += 1
    return hist


# We use this observer to clean output memory in between kernel executions
class MemoryReset(BenchmarkObserver):
    def __init__(self, args):
        self.args = args

    def before_start(self):
        for i, arg in enumerate(self.args):
            if not arg is None:
                self.dev.memcpy_htod(self.dev.allocations[i], arg)

    def get_results(self):
        return {}


code = """
#include <stdlib.h>

#define HIST_SIZE 256
#define VECTOR_SIZE 1000000

int main(void) {
	int * vector = (int *) malloc(VECTOR_SIZE * sizeof(int));
	int * hist = (int *) malloc(HIST_SIZE * sizeof(int));

	#pragma tuner start histogram vector(int*:VECTOR_SIZE) hist(int*:HIST_SIZE)
	#pragma acc parallel num_gangs(ngangs) vector_length(nthreads)
	#pragma acc loop independent
	for ( int i = 0; i < VECTOR_SIZE; i++ ) {
	    #pragma acc atomic update
		hist[vector[i]] += 1;
	}
	#pragma tuner stop

	free(vector);
	free(hist);
}
"""

# Extract tunable directive
app = Code(OpenACC(), Cxx())
kernel_string, kernel_args = process_directives(app, code)

tune_params = dict()
tune_params["ngangs"] = [2**i for i in range(1, 11)]
tune_params["nthreads"] = [32 * i for i in range(1, 33)]
metrics = dict()
metrics["GB/s"] = (
    lambda x: ((2 * 4 * len(kernel_args["histogram"][0])) + (4 * len(kernel_args["histogram"][0])))
    / (x["time"] / 10**3)
    / 10**9
)

kernel_args["histogram"][0] = np.random.randint(0, 256, len(kernel_args["histogram"][0]), dtype=np.int32)
kernel_args["histogram"][1] = np.zeros(len(kernel_args["histogram"][1])).astype(np.int32)
reference_hist = np.zeros_like(kernel_args["histogram"][1]).astype(np.int32)
reference_hist = histogram(kernel_args["histogram"][0], reference_hist)
answer = [None, reference_hist]

tune_kernel(
    "histogram",
    kernel_string["histogram"],
    0,
    kernel_args["histogram"],
    tune_params,
    metrics=metrics,
    answer=answer,
    compiler="nvc++",
    compiler_options=["-fast", "-acc=gpu"],
)
