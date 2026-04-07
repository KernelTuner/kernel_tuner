import torch
import cutlass
import cutlass.cute as cute

from cutlass.cute.runtime import from_dlpack
from kernel_tuner import tune_kernel
from pathlib import Path

FULL_PATH = Path(__file__).resolve()

# need export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH to work

# Maybe use this as optimized kernel: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/tensorop_gemm.py


@cute.kernel
def naive_matmul_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    M, K = gA.shape
    _, N = gB.shape

    n = thread_idx % N
    m = thread_idx // N

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

    threads_per_block = 256

    M, _ = mA.shape
    _, N = mB.shape

    total_outputs = M * N

    kernel = naive_matmul_kernel(mA, mB, mC)

    kernel.launch(
        grid=((total_outputs + threads_per_block - 1) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )


# Here we formulate the jit wrapper a bit differently, becuase the grid and block sizes
# are computed by kernel tuner. But we could also use threads per block as a tuning param and 
# keep the same jit wrapper as above
@cute.jit
def matmul_kt(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    grid: cutlass.Constexpr,
    block: cutlass.Constexpr,
):

    kernel = naive_matmul_kernel(mA, mB, mC)

    kernel.launch(
        grid=grid,
        block=block,
    )



def run_example(M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    matmul_ = cute.compile(matmul, a_, b_, c_)
    matmul_(a_, b_, c_)

    torch.testing.assert_close(c, a @ b, atol=1e-2, rtol=1e-2)


def call_cute(kernel_function, args, kwargs, grid, threads, params):
    cute_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            arg_ = from_dlpack(arg)
            cute_args.append(arg_)
        else:
            cute_args.append(arg)

    kernel_function(*cute_args, grid, threads)


def tune(M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    args = [a, b, c]
    size = M * N 
    answer = [None, None, (a @ b).cpu()]
    tune_params = dict()
    tune_params["block_size_x"] = [16, 32, 64, 128]
    tune_params["block_size_y"] = [16, 32, 64, 128]

    results, env = tune_kernel("matmul_kt", FULL_PATH, size, args, tune_params, lang="generic_python", 
        call_function=call_cute, answer=answer, atol=1e-2, verbose=True)






if __name__ == "__main__":
    run_example(128, 128, 128)
    tune(128, 128, 128)
