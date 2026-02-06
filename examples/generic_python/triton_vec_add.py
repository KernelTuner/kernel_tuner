import numpy as np
import triton.language as tl
import torch
from kernel_tuner import tune_kernel, run_kernel
from kernel_tuner.file_utils import store_output_file, store_metadata_file
import triton
from math import ceil


@triton.jit
def add_op(x, y):
    return x + y
    

#triton.jit
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

# NOTE: can the python file be changed in between? what happens?
# NOTE: tune params in the funcion signature are supported as key word arguments. Do not pass them as args, these 
# will be inserted automatically. You can use them in the call function (kwargs). 

def tune_with_generic():
    size = 10000000

    a = torch.randn(size, device='cuda', dtype=torch.float32)
    b = torch.randn(size, device='cuda', dtype=torch.float32)
    c = torch.empty_like(b)
    n = torch.tensor(size, dtype=torch.int32)
    c_expect = a + b

    args = [a, b, c, size] 
    tune_params = dict()
    tune_params["block_size_x"] = [2**i for i in range(10)]

    '''
    result = run_kernel("add_kernel", add_kernel, size, args, {"block_size_x": 256}, 
               lang="generic_python", call_function=call_triton, decorator="@triton.jit")   
    print(np.allclose(c_expect.cpu(), result[2]))
    '''
    
    results, env = tune_kernel(
        kernel_name="add_kernel",
        kernel_source=add_kernel,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, c_expect.cpu(), None],
        call_function=call_triton,
        decorator="@triton.jit"
    )
    
    

    
    

tune_with_generic()