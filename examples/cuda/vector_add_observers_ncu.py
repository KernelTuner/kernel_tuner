#!/usr/bin/env python
"""This is the minimal example from the README"""
import json

import numpy
from kernel_tuner import tune_kernel
from kernel_tuner.observers.ncu import NCUObserver

def tune():

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 80000000

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    ncu_metrics = ["dram__bytes.sum",                                       # Counter         byte            # of bytes accessed in DRAM
                   "dram__bytes_read.sum",                                  # Counter         byte            # of bytes read from DRAM
                   "dram__bytes_write.sum",                                 # Counter         byte            # of bytes written to DRAM
                   "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",   # Counter         inst            # of FADD thread instructions executed where all predicates were true
                   "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",   # Counter         inst            # of FFMA thread instructions executed where all predicates were true
                   "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",   # Counter         inst            # of FMUL thread instructions executed where all predicates were true
                  ]

    ncuobserver = NCUObserver(metrics=ncu_metrics)

    def total_fp32_flops(p):
        return p["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"] + 2 * p["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"] + p["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"]

    metrics = dict()
    metrics["GFLOP/s"] = lambda p: (total_fp32_flops(p) / 1e9) / (p["time"]/1e3)
    metrics["Expected GFLOP/s"] = lambda p: (size / 1e9) / (p["time"]/1e3)
    metrics["GB/s"] = lambda p: (p["dram__bytes.sum"] / 1e9) / (p["time"]/1e3)
    metrics["Expected GB/s"] = lambda p: (size*4*3 / 1e9) / (p["time"]/1e3)

    results, env = tune_kernel("vector_add", kernel_string, size, args, tune_params, observers=[ncuobserver], metrics=metrics, iterations=7)

    return results


if __name__ == "__main__":
    tune()
