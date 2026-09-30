import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

kernel_name = "vector_add"


@cute.kernel
def vector_add_kernel(gC: cute.Tensor, gA: cute.Tensor, gB: cute.Tensor, n: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bdim * bidx + tidx
    if i < n:
        gC[i] = gA[i] + gB[i]


@cute.jit
def vector_add(mC: cute.Tensor, mA: cute.Tensor, mB: cute.Tensor, n: cute.Int32):
    block_size_x = 128
    vector_add_kernel(mC, mA, mB, n).launch(
        grid=(cute.ceil_div(n, block_size_x), 1, 1),
        block=(block_size_x, 1, 1),
    )


def arguments(c, a, b, n):
    return [c, a, b, n]


def tune_params(n):
    return {"block_size_x": [128, 256]}


# kernels compiled with cute.compile, indexed by the kernel function of each configuration
compiled_kernels = {}


def call_function(kernel_function, args, kwargs):
    cute_args = [from_dlpack(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]
    if kernel_function not in compiled_kernels:
        compiled_kernels[kernel_function] = cute.compile(kernel_function, *cute_args)
    compiled_kernels[kernel_function](*cute_args, **kwargs)
