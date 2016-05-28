#!/usr/bin/env python
"""This is a simple example for tuning C code with the kernel tuner"""

import numpy
from kernel_tuner import tune_kernel

kernel_string = """ 
#include <omp.h>
#include "timer.h"

#if vectorsize == 1
  #define vfloat float
#else
  typedef float vfloat __attribute__ ((vector_size (vectorsize)));
#endif

float vector_add(vfloat *c, vfloat *a, vfloat *b, int n) {
    unsigned long long start = get_clock();

    #pragma omp parallel num_threads(nthreads)
    {
        int id = omp_get_thread_num();
        int block = n/(vectorsize*nthreads);
        for (int i = id*block; i<(id+1)*block && i<(n/vectorsize); i++) {
            c[i] = a[i] + b[i];
        }
    }

    return (get_clock()-start) / (get_frequency() * 1000000);
}
"""

size = 128*1024*1024
problem_size = (size, 1)

a = numpy.random.randn(size).astype(numpy.float32)
b = numpy.random.randn(size).astype(numpy.float32)
c = numpy.zeros_like(b)
n = numpy.int32(size)

args = [c, a, b, n]

tune_params = dict()
tune_params["vectorsize"] = [1] + [2**i for i in range(2,8)]
tune_params["nthreads"] = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32]

tune_kernel("vector_add", kernel_string, problem_size, args, tune_params)
