import torch
from numba import cuda

kernel_name = "vector_add"


@cuda.jit
def vector_add(c, a, b, n):
    i = cuda.grid(1)
    if i < n:
        c[i] = a[i] + b[i]


def arguments(c, a, b, n):
    return [c, a, b, n]


def tune_params(n):
    return {"block_size_x": [128, 256]}


def call_function(kernel_function, args, kwargs, grid, threads):
    numba_args = [cuda.as_cuda_array(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]
    kernel_function[grid, threads](*numba_args, **kwargs)
