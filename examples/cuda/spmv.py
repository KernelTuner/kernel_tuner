#!/usr/bin/env python
from __future__ import print_function

import numpy
import time
from scipy.sparse import csr_matrix
from itertools import chain
from collections import OrderedDict

import kernel_tuner

def tune():
    with open('spmv.cu', 'r') as f:
        kernel_string = f.read()

    nrows = numpy.int32(128*1024)
    ncols = 64*1024
    nnz = int(nrows*ncols*0.001)
    #problem_size = (nrows, 1)
    problem_size = nrows

    #generate sparse matrix in CSR
    rows = numpy.asarray([0]+sorted(numpy.random.rand(nrows-1)*nnz)+[nnz]).astype(numpy.int32)
    cols = (numpy.random.rand(nnz)*ncols).astype(numpy.int32)
    vals = numpy.random.randn(nnz).astype(numpy.float32)

    #input and output vector  (y = matrix * x)
    x = numpy.random.randn(ncols).astype(numpy.float32)
    y = numpy.zeros(nrows).astype(numpy.float32)

    args = [y, rows, cols, vals, x, nrows]

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)]
    tune_params["threads_per_row"] = [1, 32]
    tune_params["read_only"] = [0, 1]

    grid_div_x = ["block_size_x/threads_per_row"]

    #compute reference answer using scipy.sparse
    row_ind = list(chain.from_iterable([[i] * (rows[i+1]-rows[i]) for i in range(nrows)]))
    matrix = csr_matrix((vals, (row_ind, cols)), shape=(nrows, ncols))
    start = time.clock()
    expected_y = matrix.dot(x)
    end = time.clock()
    print("computing reference using scipy.sparse took: " + str(start-end / 1000.0) + " ms.")

    answer = [expected_y, None, None, None, None, None]

    return kernel_tuner.tune_kernel("spmv_kernel", kernel_string,
        problem_size, args, tune_params,
        grid_div_x=grid_div_x, verbose=True, answer=answer, atol=1e-4)


if __name__ == "__main__":
    tune()
