import torch 
from functools import partial
from typing import List
import time

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from kernel_tuner import tune_kernel

@cute.kernel
def vec_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    size: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_id = bdim * bidx + tidx

    if thread_id < size:
        gC[thread_id] = gA[thread_id] + gB[thread_id]



@cute.jit
def vec_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    size: cute.Int32,
):
    num_threads_per_block = 256

    kernel = vec_add_kernel(mA, mB, mC, size)

    kernel.launch(
        grid=(cute.ceil_div(size, num_threads_per_block), 1, 1),
        block = (num_threads_per_block, 1, 1),
    )




def call_cute(kernel_function, args, kwargs, grid, threads, params):
    cute_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            arg_ = from_dlpack(arg)
            cute_args.append(arg_)
        else:
            cute_args.append(arg)

    kernel_function(*cute_args, **kwargs)




def main():
    size = 16384
    a = torch.randn(size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, device="cuda", dtype=torch.float16)
    c = torch.zeros(size, device="cuda", dtype=torch.float16)

    '''
    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)
    '''

    args = [a, b, c, size]
    tune_params = {"num_threads_per_block": [1, 2, 4, 8, 16, 32, 64, 128, 265, 512, 1024]}
    answer = [None, None, (a+b).cpu(), None]

    from pathlib import Path
    FULL_PATH = Path(__file__).resolve()
    tune_kernel("vec_add", FULL_PATH, size, args, tune_params, answer=answer,
                lang="generic_python", call_function=call_cute, verbose=True)
    #naive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)

    #naive_elementwise_add_(a_, b_, c_)
    #vec_add(a_, b_, c_, size)



    #torch.testing.assert_close(c, a+b)

main()

