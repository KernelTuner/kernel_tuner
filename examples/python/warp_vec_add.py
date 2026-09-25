import warp as wp
import numpy as np
import torch

from kernel_tuner import tune_kernel
from call_functions import call_warp


wp.init()


@wp.func
def add_op(x: float, y: float):
    return x + y

@wp.kernel
def vec_add(a: wp.array(dtype=float),
            b: wp.array(dtype=float),
            c: wp.array(dtype=float),
            n: int):

    work_per_thread = 8
    tid = wp.tid()
    base = tid * work_per_thread

    for i in range(work_per_thread):
        idx = base + i
        if idx < n:
            c[idx] = add_op(a[idx], b[idx])


def call_warp(kernel_function, args, kwargs, grid, threads, params):
    warp_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            warp_args.append(wp.from_torch(arg))
        else:
            warp_args.append(arg)
    dim = params['size']
    wp.launch(kernel=kernel_function, dim=dim, inputs=warp_args)
    

def tune():
    n = 1024

    # Create host arrays
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, 0, -1, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    c_expect = a + b

    
    tune_params = dict()
    tune_params["work_per_thread"] = [2**i for i in range(10)]
    tune_params["size"] = [n]
    args = [a, b, c, n]

    results, env = tune_kernel(
        kernel_name="vec_add",
        kernel_source=__file__,
        problem_size=n,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, c_expect, None],
        call_function=call_warp,
    )


if __name__ == "__main__":
    tune()