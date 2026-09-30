import cupy as cp
import torch
from cupyx import jit

kernel_name = "vector_add"


@jit.rawkernel()
def vector_add(c, a, b, n):
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < n:
        c[i] = a[i] + b[i]


def arguments(c, a, b, n):
    return [c, a, b, n]


def tune_params(n):
    return {"block_size_x": [128, 256]}


def call_function(kernel_function, args, kwargs, grid, threads):
    cupy_args = [cp.from_dlpack(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]
    kernel_function(grid, threads, tuple(cupy_args))
