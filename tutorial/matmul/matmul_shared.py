#!/usr/bin/env python
import numpy
import kernel_tuner

problem_size = (4096, 4096)
size = numpy.prod(problem_size)

A = numpy.random.randn(*problem_size).astype(numpy.float32)
B = numpy.random.randn(*problem_size).astype(numpy.float32)
C = numpy.zeros_like(A)

args = [C, A, B]

answer = [numpy.dot(A,B), None, None]

params = {"block_size_x": 32, "block_size_y": 32}

results = kernel_tuner.run_kernel("matmul_kernel", "matmul_data_reuse.cu",
                                   problem_size, args, params)  
# answer = run_kernel("matmul_kernel", [get_kernel_path()+"matmul_naive.cu"], problem_size, args, params, lang="C", compiler_options=cp)
