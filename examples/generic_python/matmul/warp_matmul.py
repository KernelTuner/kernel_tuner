import torch
import numpy as np
import warp as wp

from kernel_tuner import tune_kernel
from pathlib import Path 

wp.init()

FULL_PATH = Path(__file__).resolve()

# tile size
TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_THREADS = 64

# GEMM example from https://nvidia.github.io/warp/user_guide/tiles.html 
@wp.kernel
def tile_gemm(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):

    # output tile index
    i, j = wp.tid()

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    count = (K + TILE_K - 1) // TILE_K 

    for k in range(0, count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i*TILE_M, k*TILE_K), bounds_check=True)
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k*TILE_K, j*TILE_N), bounds_check=True)

        # sum += a*b
        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum, offset=(i*TILE_M, j*TILE_N), bounds_check=True)


def run_kernel_direct(M, N, K):
    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    A_wp = wp.array(A)
    B_wp = wp.array(B)
    C_wp = wp.array(C)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_gemm,
            dim=((M + TILE_M - 1) // TILE_M, (N + TILE_N - 1) // TILE_N),
            inputs=[A_wp, B_wp, C_wp],
            block_dim=TILE_THREADS)

    np.testing.assert_allclose(C_wp.numpy(), A @ B, rtol=1e-3)

    print("Example matrix multiplication passed")



def call_warp(kernel_function, args, kwargs, grid, threads, params):
    # Convert Torch tensors to Warp args
    warp_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            warp_args.append(wp.from_torch(arg))
        else:
            warp_args.append(arg)

    # launch kernel
    with wp.Tape() as tape:
        wp.launch_tiled(
            kernel_function,
            dim=grid,
            inputs=warp_args,
            block_dim=params["TILE_THREADS"], # We could directly take threads, but in the given example this is a constant
        )


def tune(M, K, N):
    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    size = (M, N)
    block_size_names = ["TILE_M", "TILE_N"]

    tune_params = dict()
    tune_params["TILE_M"] = [4, 8, 16]
    tune_params["TILE_N"] = [2, 4, 8]
    tune_params["TILE_K"] = [4, 8, 16]
    tune_params["TILE_THREADS"] = [32, 64, 128]

    args = [A, B, C]
    answer = [None, None, A @ B]

    results, env = tune_kernel(
        kernel_name="tile_gemm",
        kernel_source=FULL_PATH,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=answer,
        call_function=call_warp,
        block_size_names=block_size_names,
    )


if __name__ == "__main__":
    #tune(128, 128, 128)

    sizes = [
        (65, 65, 17),
        (67, 71, 19),
        (1, 1, 1),
        (63, 63, 15),
        (129, 130, 33),
    ]

    for size in sizes:
        print(size)
        run_kernel_direct(*size)
