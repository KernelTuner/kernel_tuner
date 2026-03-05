import numpy as np
import torch
import triton
import triton.language as tl
from pathlib import Path

from kernel_tuner import tune_kernel, run_kernel

FULL_PATH = Path(__file__).resolve() 

@triton.jit
def add_op(x, y):
    return x + y

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               block_size_x: tl.constexpr,  # Number of elements each program should process.
               # note: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * block_size_x
    offsets = block_start + tl.arange(0, block_size_x)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = add_op(x, y)
    tl.store(output_ptr + offsets, output, mask=mask)


def call_triton(kernel_function, args, kwargs, grid, threads, params):
    kernel_function[grid](*args, **kwargs)


def tune():
    size = 10000000

    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    c = torch.empty_like(b)
    n = torch.tensor(size, dtype=torch.int32)
    c_expect = a + b

    args = [a, b, c, size] 
    tune_params = dict()
    tune_params["block_size_x"] = [2**i for i in range(11)]

    
    result = run_kernel("add_kernel", FULL_PATH, size, args, {"block_size_x": 256}, 
               lang="generic_python", call_function=call_triton)   
    assert np.allclose(c_expect.cpu(), result[2])
    
    
 
    results, env = tune_kernel(
        kernel_name="add_kernel",
        kernel_source=FULL_PATH,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, c_expect.cpu(), None],
        call_function=call_triton,
    )

    
    
if __name__ == "__main__":
    tune()