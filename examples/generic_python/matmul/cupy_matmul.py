import cupy as cp
from cupyx import jit
import numpy as np
import torch

from kernel_tuner import tune_kernel
from pathlib import Path 


@jit.rawkernel()
def gemm_raw_strided(a, b, c, M, N, K): # with outer loop so that we can have more work per thread
    # global thread indices
    row = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    col = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    # grid stride increments
    stride_y = jit.gridDim.y * jit.blockDim.y
    stride_x = jit.gridDim.x * jit.blockDim.x

    i = row
    while i < M:
        j = col
        while j < N:
            acc = 0.0
            for kk in range(K):
                acc += a[i, kk] * b[kk, j]   
            c[i, j] = acc                   
            j += stride_x
        i += stride_y


@jit.rawkernel()
def gemm(a, b, c, M, N, K):
    row = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    col = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    acc = 0.0
    for kk in range(K):
        acc += a[row, kk] * b[kk, col]   
    c[row, col] = acc                   


def run():
    M, K, N = 128, 256, 64

    # random test data
    A = cp.random.random((M, K), dtype=cp.float32)
    B = cp.random.random((K, N), dtype=cp.float32)
    C = cp.zeros((M, N), dtype=cp.float32)

    # launch parameters
    block = (16, 16)
    grid = ((N + block[0] - 1) // block[0],
            (M + block[1] - 1) // block[1])

    # launch kernel
    gemm_raw_strided(grid, block, (A, B, C, M, N, K))

    # validate
    C_ref = A.dot(B)
    print("max error:", float(cp.max(cp.abs(C - C_ref))))


def call_cupyx(kernel_function, args, kwargs, grid, threads):
    cupy_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            cupy_args.append(cp.from_dlpack(arg))
        else:
            cupy_args.append(arg)
    kernel_function(grid, threads, tuple(cupy_args))

def tune():
    M, K, N = 128, 256, 64

    # random test data. Here we had to change cupy to numpy arrays.
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)


    args = [A, B, C, M, N, K]
    size = (N, M)
    tune_params = {"block_size_x": [2**i for i in range(10)], "block_size_y": [2**i for i in range(10)]}
    restrictions = ["block_size_x == block_size_y"]
    source = Path(__file__).resolve()

    results, env = tune_kernel(
        kernel_name="gemm",
        kernel_source=source,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        answer=[None, None, A.dot(B), None, None],
        atol=1e-2,
        call_function=call_cupyx, 
        lang="generic_python",
        verbose=True,  
        restrictions=restrictions,    
    )


if __name__ == "__main__":
    #tune()
    run()