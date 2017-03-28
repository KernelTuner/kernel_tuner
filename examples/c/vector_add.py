#!/usr/bin/env python
"""This is a simple example for tuning C code with the kernel tuner"""

import numpy
from kernel_tuner import tune_kernel

kernel_string = """ 
#include <omp.h>
#include "timer.h"

typedef float vfloat __attribute__ ((vector_size (vecsize*4)));

float vector_add(vfloat *c, vfloat *a, vfloat *b, int n) {
    unsigned long long start = get_clock();
    int chunk = n/vecsize/nthreads;

    #pragma omp parallel num_threads(nthreads)
    {
        int offset = omp_get_thread_num()*chunk;
        for (int i = offset; i<offset+chunk; i++) {
            c[i] = a[i] + b[i];
        }
    }

    return (get_clock()-start) / get_frequency() / 1000000.0;
}
"""

size = 72*1024*1024

a = numpy.random.randn(size).astype(numpy.float32)
b = numpy.random.randn(size).astype(numpy.float32)
c = numpy.zeros_like(b)
n = numpy.int32(size)

args = [c, a, b, n]

tune_params = dict()
tune_params["vecsize"] = [2**i for i in range(8)]
tune_params["nthreads"] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

answer = [a+b, None, None]

tune_kernel("vector_add", kernel_string, size, args, tune_params, answer=answer)
