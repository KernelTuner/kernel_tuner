import numpy as np
from numba import cuda
from math import ceil
from kernel_tuner import tune_kernel, run_kernel

#@cuda.jit
def f(a, b, c):
    tid = cuda.grid(1)
    size = len(c)

    if tid < size:
        c[tid] = a[tid] + b[tid]


def call_numba(kernel_function, args, kwargs, grid, threads, params):
    kernel_function[grid[0], threads[0]](*args, **kwargs)


def verify(answer, result_host, atol):
    correct = True
    for i, ans in enumerate(answer):
        if ans is None:
            continue
        res = result_host[i].copy_to_host()
        if not np.allclose(ans, res, atol=atol):
            correct = False 

    return correct


N = 100000000
a = cuda.to_device(np.random.random(N))
b = cuda.to_device(np.random.random(N))
c = cuda.device_array_like(a)
c_expect = a.copy_to_host() + b.copy_to_host()

args = [a, b, c]
tune_params = {"block_size_x": [2**i for i in range(10)]}

results = run_kernel(
    kernel_name="f",
    kernel_source=f,
    problem_size=N,
    arguments=args,
    params={"block_size_x": 32},
    lang="generic_python",
    call_function=call_numba,
    decorator="@cuda.jit"
)


print(np.allclose(results[2], c_expect))

'''
results, env = tune_kernel(
    kernel_name="f",
    kernel_source=f,
    problem_size=N,
    arguments=args,
    tune_params=tune_params,
    lang="generic_python",
    answer=[None, None, c_expect],
    verify=verify,
    call_function=call_numba,
    decorator="@cuda.jit"
)
'''

