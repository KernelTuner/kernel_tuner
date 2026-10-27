import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from kernel_tuner import tune_kernel


# might need export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH to work

## Basic Matmul ================================================================

@cute.kernel
def naive_matmul_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, M: int, K: int, N: int):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    bdx, bdy, _ = cute.arch.block_dim()

    # Global indices
    n = bx * bdx + tx
    m = by * bdy + ty

    if m < M and n < N:
        acc = cutlass.Float32(0.0)

        for k in range(K):
            a = gA[m, k].to(cutlass.Float32)
            b = gB[k, n].to(cutlass.Float32)
            acc += a * b

        gC[m, n] = acc.to(gC.element_type)


@cute.jit
def matmul(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    block_size_x = 16
    block_size_y = 16
    block = (block_size_x, block_size_y, 1)

    M, K = mA.shape
    _, N = mB.shape

    grid = (
        (N + block[0] - 1) // block[0],
        (M + block[1] - 1) // block[1],
        1,
    )

    kernel = naive_matmul_kernel(mA, mB, mC, M, K, N)

    kernel.launch(grid=grid, block=block)


def run_naive_matmul(M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    c_ref = a @ b

    # Convert to CuTe tensors
    a_ = from_dlpack(a, assumed_align=16)  
    b_ = from_dlpack(b, assumed_align=16)  
    c_ = from_dlpack(c, assumed_align=16)  

    compiled_kernel = cute.compile(matmul, a_, b_, c_)
    compiled_kernel(a_, b_, c_)

    assert torch.allclose(c, c_ref, atol=M * 2 **(-11), rtol=1e-2)
    print("Succes")


def tune_naive_matmul(M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    args = [a, b, c]
    size = M * N 
    answer = [None, None, (a @ b).cpu()]
    tune_params = dict()
    tune_params["block_size_x"] = [2**i for i in range(1, 10)]
    tune_params["block_size_y"] = [2**i for i in range(1, 10)]
    restrictions = ["block_size_x * block_size_y >= 32", "block_size_x * block_size_y <= 1024"]

    results, env = tune_kernel("matmul", __file__, size, args, tune_params,
        answer=answer,  atol=M * 2 **(-11), restrictions=restrictions, verbose=False)


if __name__ == "__main__":
    m, n, k = 4096, 4096, 4096
    run_naive_matmul(m, n, k)
    tune_naive_matmul(m, n, k)
