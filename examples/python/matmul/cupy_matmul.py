import cupy as cp
from cupyx import jit
import numpy as np

from kernel_tuner import tune_kernel
from examples.generic_python.call_functions import call_cupyx


@jit.rawkernel()
def gemm(a, b, c, M, N, K):
    row = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    col = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    
    if row < M and col < N:
        acc = 0.0
        for kk in range(K):
            acc += a[row, kk] * b[kk, col]   
        c[row, col] = acc                   


def run(M, N, K):
    # float16 matrices on GPU
    a = cp.random.random((M, K)).astype(cp.float16)
    b = cp.random.random((K, N)).astype(cp.float16)
    c = cp.zeros((M, N), dtype=cp.float16)

    # block / grid configuration
    block = (16, 16)
    grid = ((N + block[0] - 1) // block[0], (M + block[1] - 1) // block[1])

    # launch kernel
    gemm[grid, block](a, b, c, M, N, K)
    cp.cuda.Device().synchronize()

    # Correctness verification
    c_ref = cp.matmul(a, b)
    assert cp.allclose(c, c_ref, rtol=1e-2, atol=1e-1)

    print("Succes")


def tune(M, N, K):
    # random test data. Here we had to use numpy arrays instead of cupy.
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    args = [A, B, C, M, N, K]
    size = (N, M)
    tune_params = {"block_size_x": [2**i for i in range(11)], "block_size_y": [2**i for i in range(11)]}
    restrictions = ["block_size_x * block_size_y <= 1024"]

    results, env = tune_kernel(
        kernel_name="gemm",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        answer=[None, None, A.dot(B), None, None, None],
        atol=1e-1,
        call_function=call_cupyx, 
        lang="generic_python",
        restrictions=restrictions,
        verbose=True,   
    )


if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024
    tune(M, N, K)
    