#!/usr/bin/env python
"""This is a simple example for tuning C code with the kernel tuner"""

import numpy
from kernel_tuner import tune_kernel
from collections import OrderedDict

kernel_string = """ 
#include <omp.h>
#include <string.h>

typedef float vfloat __attribute__ ((vector_size (vecsize*4)));

// TODO: Supply a C header file which provides this struct and helper functions
// to fill it
// TODO: Add versioning to this strict somehow. (In the name or a first field?)
struct KernelTunerExtendedReturnValue {
    bool success;
    char error[256];
    int count;
    char keys[8][16];
    float values[8];
};

extern "C" void vector_add(vfloat *c, vfloat *a, vfloat *b, int n, KernelTunerExtendedReturnValue *ret) {
    double start = omp_get_wtime();
    int chunk = n/(vecsize*nthreads);

    #pragma omp parallel num_threads(nthreads)
    {
        int offset = omp_get_thread_num()*chunk;
        for (int i = offset; i<offset+chunk && i<n; i++) {
            c[i] = a[i] + b[i];
        }
    }

    ret->success = true;
    ret->count = 1;
    strcpy(ret->keys[0], "time");
    ret->values[0] = (float)((omp_get_wtime() - start)*1e3);
}
"""

size = 72 * 1024 * 1024

a = numpy.random.randn(size).astype(numpy.float32)
b = numpy.random.randn(size).astype(numpy.float32)
c = numpy.zeros_like(b)
n = numpy.int32(size)

args = [c, a, b, n, "KernelTunerExtendedReturnValue"]

tune_params = OrderedDict()
tune_params["nthreads"] = [1, 2, 3, 4, 8, 12, 16, 24, 32]
tune_params["vecsize"] = [1, 2, 4, 8, 16]

answer = [a + b, None, None, None, None]

tune_kernel("vector_add", kernel_string, size, args, tune_params, answer=answer, compiler_options=["-fopenmp", "-O3"])
