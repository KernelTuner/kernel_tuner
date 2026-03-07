from numba import cuda
import torch
from kernel_tuner import tune_kernel, run_kernel
import numpy as np
from pathlib import Path

FULL_PATH = Path(__file__).resolve()


@cuda.jit
def f(a, b, c):
    tid = cuda.grid(1)
    size = len(c)

    if tid < size:
        c[tid] = a[tid] + b[tid]


def call_numba(kernel_function, args, kwargs, grid, threads):
    numba_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            numba_args.append(cuda.as_cuda_array(arg))
        else:
            numba_args.append(arg)
    kernel_function[grid, threads](*args, **kwargs)


def tune():

    N = 100000

    a = np.random.random(N)
    b = np.random.random(N)
    c = np.zeros(N)
    c_expect = a + b

    args = [a, b, c]
    tune_params = {"block_size_x": [2**i for i in range(10)]}


    results, env = tune_kernel(
        kernel_name="f",
        kernel_source=FULL_PATH,
        problem_size=N,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, c_expect],
        call_function=call_numba,
    )

if __name__ == "__main__":
    tune()