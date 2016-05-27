#!/usr/bin/env python
"""This is the minimal example for tuning C code with the kernel tuner"""

import numpy
from kernel_tuner import tune_kernel

kernel_string = """ 
#include <stdint.h>
#include "timer.h"

#if vectorsize == 1
  #define vf float
#else
  typedef float vf __attribute__ ((vector_size (vectorsize)));
#endif

float vector_add(vf *c, vf *a, vf *b, int n) {
    uint64_t start = get_time();

    for (int i = 0; i<n/vectorsize; i++) {
        c[i] = a[i] + b[i];
    }

    return (get_time()-start) / (CPU_MHz * 1000000);
}
"""

size = 1024*1024
problem_size = (size, 1)

a = numpy.random.randn(size).astype(numpy.float32)
b = numpy.random.randn(size).astype(numpy.float32)
c = numpy.zeros_like(b)
n = numpy.int32(size)

args = [c, a, b, n]

tune_params = dict()
tune_params["vectorsize"] = [1] + [2**i for i in range(2,10)]

import subprocess
cpu_speed = subprocess.check_output(["cat /proc/cpuinfo | grep MHz"],shell=True).split()[3]

tune_kernel("vector_add", kernel_string.replace("CPU_MHz", cpu_speed),
            problem_size, args, tune_params, lang="C")
