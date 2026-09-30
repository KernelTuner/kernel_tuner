import torch
import warp as wp

wp.init()

kernel_name = "vector_add"


@wp.kernel
def vector_add(c: wp.array(dtype=float), a: wp.array(dtype=float), b: wp.array(dtype=float), n: int):
    i = wp.tid()
    if i < n:
        c[i] = a[i] + b[i]


def arguments(c, a, b, n):
    return [c, a, b, n]


def tune_params(n):
    return {"block_size_x": [128, 256]}


def call_function(kernel_function, args, kwargs, grid, threads):
    warp_args = [wp.from_torch(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]
    wp.launch(kernel_function, dim=grid[0] * threads[0], inputs=warp_args, block_dim=threads[0])
