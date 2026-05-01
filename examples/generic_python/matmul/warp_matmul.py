import torch
import numpy as np
import warp as wp

from kernel_tuner import tune_kernel
from examples.generic_python.call_functions import call_warp

wp.init()
wp.config.enable_backward = False


@wp.kernel()
def gemm(
    A: wp.array2d(dtype=wp.float16), B: wp.array2d(dtype=wp.float16), C: wp.array2d(dtype=wp.float16)
):
    i, j = wp.tid()
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    if i >= M or j >= N:
        return

    # compute dot product
    sum = wp.float32(0.0)
    for k in range(K):
        sum += wp.float32(A[i, k]) * wp.float32(B[k, j])

    # write result
    C[i, j] = wp.float16(sum)


# tile size
TILE_M = 32
TILE_N = 32
TILE_K = 32

# num threads per-tile
TILE_THREADS = 1024

# GEMM example from https://nvidia.github.io/warp/user_guide/tiles.html 
@wp.kernel()
def tile_gemm(
    A: wp.array2d(dtype=wp.float16), 
    B: wp.array2d(dtype=wp.float16), 
    C: wp.array2d(dtype=wp.float16)
):
    # output tile index
    i, j = wp.tid()

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

    K = A.shape[1]
    count = (K + TILE_K - 1) // TILE_K 

    for k in range(0, count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i*TILE_M, k*TILE_K), bounds_check=True)
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k*TILE_K, j*TILE_N), bounds_check=True)

        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, wp.tile_astype(sum, wp.float16), offset=(i*TILE_M, j*TILE_N), bounds_check=True)


def run_gemm(M, N, K):
    rng = np.random.default_rng(42)
    A = rng.random((M, K)).astype(np.float16)
    B = rng.random((K, N)).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    A_wp = wp.array(A)
    B_wp = wp.array(B)
    C_wp = wp.array(C)

    wp.launch(gemm, dim=(M, N), inputs=[A_wp, B_wp, C_wp])

    np.testing.assert_allclose(C_wp.numpy(), A @ B, rtol=1e-2, atol=M * 2**(-11))

    print("Succes")


def run_gemm_tiled(M, N, K):
    rng = np.random.default_rng(42)
    A = rng.random((M, K)).astype(np.float16)
    B = rng.random((K, N)).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    A_wp = wp.array(A)
    B_wp = wp.array(B)
    C_wp = wp.array(C)


    wp.launch_tiled(
        tile_gemm,
        dim=((M + TILE_M - 1) // TILE_M, (N + TILE_N - 1) // TILE_N),
        inputs=[A_wp, B_wp, C_wp],
        block_dim=TILE_THREADS)

    np.testing.assert_allclose(C_wp.numpy(), A @ B, rtol=1e-2, atol=M * 2**(-11))

    print("Succes")


def tune(M, K, N):
    rng = np.random.default_rng(42)
    A = rng.random((M, K)).astype(np.float16)
    B = rng.random((K, N)).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    size = (M, N)

    tune_params = dict()
    tune_params["block_dim"] = [2**i for i in range(5, 11)]
    tune_params["dim"] = [size]

    args = [A, B, C]
    answer = [None, None, A @ B]

    results, env = tune_kernel(
        kernel_name="gemm",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=answer,
        call_function=call_warp,
    )



if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024
    
    #run_gemm(M, N, K)
    #run_gemm_tiled(M, N, K)
    
    tune(M, N, K)

    


    

    

    
