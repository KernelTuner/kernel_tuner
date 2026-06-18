from numba import cuda
import numpy as np

from kernel_tuner import tune_kernel
from call_functions import call_numba


@cuda.jit
def f(a, b, c):
    tid = cuda.grid(1)
    size = len(c)

    if tid < size:
        c[tid] = a[tid] + b[tid]


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
        kernel_source=__file__,
        problem_size=N,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, c_expect],
        call_function=call_numba,
    )

if __name__ == "__main__":
    tune()