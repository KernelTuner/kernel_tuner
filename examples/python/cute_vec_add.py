import torch 

import cutlass
import cutlass.cute as cute

from kernel_tuner import tune_kernel
from call_functions import call_cute

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


def main():
    size = 16384
    a = torch.randn(size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, device="cuda", dtype=torch.float16)
    c = torch.zeros(size, device="cuda", dtype=torch.float16)

    args = [a, b, c, size]
    tune_params = {"num_threads_per_block": [1, 2, 4, 8, 16, 32, 64, 128, 265, 512, 1024]}
    answer = [None, None, (a+b).cpu(), None]

    tune_kernel("vec_add", __file__, size, args, tune_params, answer=answer,
                lang="generic_python", call_function=call_cute, verbose=True)


main()

