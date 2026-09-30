import triton
import triton.language as tl

kernel_name = "vector_add"


@triton.jit
def vector_add(c_ptr, a_ptr, b_ptr, n, block_size_x: tl.constexpr):
    offsets = tl.program_id(axis=0) * block_size_x + tl.arange(0, block_size_x)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def arguments(c, a, b, n):
    return [c, a, b, n]


def tune_params(n):
    return {"block_size_x": [128, 256]}


def call_function(kernel_function, args, kwargs, grid):
    kernel_function[grid](*args, **kwargs)
