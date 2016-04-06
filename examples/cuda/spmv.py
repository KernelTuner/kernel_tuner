#!/usr/bin/env python

import numpy
from kernel_tuner import tune_kernel

with open('spmv.cu', 'r') as f:
    kernel_string = f.read()

nrows = numpy.int32(128*1024)
ncols = 128*1024
nnz = int(nrows*ncols*0.001)

#generate spare matrix
rows = numpy.asarray([0]+sorted(numpy.random.rand(nrows-1)*nnz)+[nnz]).astype(numpy.int32)
cols = (numpy.random.rand(nnz)*ncols).astype(numpy.int32)
vals = numpy.random.randn(nnz).astype(numpy.float32)

#input and output vector
x = numpy.random.randn(ncols).astype(numpy.float32)
y = numpy.zeros(nrows).astype(numpy.float32)

args = [y, rows, cols, vals, x, nrows]

tune_params = dict()
tune_params["block_size_x"] = [32*i for i in range(1,33)]
tune_params["read_only"] = [0, 1]

problem_size = (nrows, 1)
grid_div_x = ["block_size_x/32"]

tune_kernel("spmv_kernel", kernel_string,
    problem_size, args, tune_params,
    grid_div_x=grid_div_x, verbose=True)


