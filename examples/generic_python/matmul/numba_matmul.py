import numpy as np
from numba import cuda, float32

from kernel_tuner import tune_kernel
from examples.generic_python.call_functions import call_numba


# Source: https://nvidia.github.io/numba-cuda/user/examples.html#matrix-multiplication
@cuda.jit(cache=True)
def matmul(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


# Translated to Numba-CUDA from https://github.com/cupy/cupy/blob/main/examples/gemm/sgemm.cu
@cuda.jit(cache=True)
def optimized_matmul(M, N, K, A, B, C):
    DIM_X = 16
    DIM_Y = 16
    BLK_M = 64
    BLK_N = 64
    BLK_K = 16
    THR_M = 4 # Should be equal to BLK_M / DIM_X
    THR_N = 4 # Should be equal to BLK_N / DIM_Y 

    # thread indices
    idx = cuda.threadIdx.x
    idy = cuda.threadIdx.y

    blx = cuda.blockIdx.x
    bly = cuda.blockIdx.y

    # shared memory
    sA = cuda.shared.array((BLK_K, BLK_M), np.float16)
    sB = cuda.shared.array((BLK_N, BLK_K), np.float16)

    # registers
    rC = cuda.local.array((THR_N, THR_M), float32)
    rA = cuda.local.array(THR_M, float32)
    rB = cuda.local.array(THR_N, float32)

    # init accumulator
    for n in range(THR_N):
        for m in range(THR_M):
            rC[n][m] = 0.0

    # global indices
    base_row = blx * BLK_M
    base_col = bly * BLK_N

    # loop over K tiles
    for kk in range(0, K, BLK_K):

        # load A tile into shared memory
        for i in range(idy, BLK_K, DIM_Y):
            for j in range(idx, BLK_M, DIM_X):
                row = base_row + j
                col = kk + i
                if row < M and col < K:
                    sA[i, j] = A[row, col]
                else:
                    sA[i, j] = 0.0

        # load B tile
        for i in range(idy, BLK_N, DIM_Y):
            for j in range(idx, BLK_K, DIM_X):
                row = kk + j
                col = base_col + i
                if row < K and col < N:
                    sB[i, j] = B[row, col]
                else:
                    sB[i, j] = 0.0

        cuda.syncthreads()

        # compute
        for k in range(BLK_K):
            for m in range(THR_M):
                rA[m] = float32(sA[k, m * DIM_X + idx])

            for n in range(THR_N):
                rB[n] = float32(sB[n * DIM_Y + idy, k])

            for n in range(THR_N):
                for m in range(THR_M):
                    rC[n][m] += rA[m] * rB[n]

        cuda.syncthreads()

    # write back
    for n in range(THR_N):
        col = base_col + n * DIM_Y + idy   
        for m in range(THR_M):
            row = base_row + m * DIM_X + idx   
            if row < M and col < N:
                C[row, col] = np.float16(rC[n][m])


def run_basic(M, N, K):
    # create numpy arrays
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    C_ref = A @ B

    # copy to GPU
    A_d = cuda.to_device(A)
    B_d = cuda.to_device(B)
    C_d = cuda.to_device(C)

    # threads per block
    threads = (16, 16)

    # compute grid size (ceil division)
    blocks = (
        (M + threads[0] - 1) // threads[0],
        (N + threads[1] - 1) // threads[1],
    )

    # launch kernel
    matmul[blocks, threads](A_d, B_d, C_d)
    cuda.synchronize()

    # copy result back
    C_result = C_d.copy_to_host()

    # check
    np.testing.assert_allclose(C_result, C_ref, rtol=1e-2, atol=M * 2**(-11))
    print("Succes")


def run_optimized(M, N, K):
    # inputs 
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    C_ref = A @ B

    # move to GPU
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.to_device(C)

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (M + 63) // 64, # BLK_M = 64
        (N + 63) // 64, # BLK_N = 64
    )

    # launch
    optimized_matmul[blocks_per_grid, threads_per_block](M, N, K, dA, dB, dC)
    cuda.synchronize()

    # copy result back
    C_result = dC.copy_to_host()
    
    # check
    np.testing.assert_allclose(C_result, C_ref, rtol=1e-2, atol=M * 2**(-11))
    print("Succes")




def tune_basic(M, N, K):
    # create inputs as normal, but do not copy to device
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    size = (M, N)
    args = [A, B, C]
    tune_params = dict()
    tune_params["block_size_x"] = [2**i for i in range(1, 10)]
    tune_params["block_size_y"] = [2**i for i in range(1, 10)]

    restrictions = ["block_size_x * block_size_y <= 1024"]
    
    answer = [None, None, A @ B]
    atol = M * 2**(-11)

    results, env = tune_kernel(
        kernel_name="matmul",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=answer,
        atol=atol,
        restrictions=restrictions,
        call_function=call_numba,
    )


def tune_optimized(M, N, K):
    # create inputs as normal, but do not copy to device
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    size = (M, N)
    args = [M, N, K, A, B, C]
    tune_params = {
        "DIM_X": [2**i for i in range(1, 10)],
        "DIM_Y": [2**i for i in range(1, 10)],
        "BLK_M": [2**i for i in range(1, 10)],
        "BLK_N": [2**i for i in range(1, 10)],
        "BLK_K": [2**i for i in range(1, 10)],
        "THR_M": [2**i for i in range(1, 9)], # Restricted to BLK_M / DIM_X
        "THR_N": [2**i for i in range(1, 9)], # Restricted to BLK_N / DIM_Y
    }
    
    answer = [None, None, None, None, None, A @ B]
    atol = M * 2**(-11)

    restrictions = [
        "BLK_M % DIM_X == 0",
        "BLK_N % DIM_Y == 0",
        "THR_M == BLK_M / DIM_X",
        "THR_N == BLK_N / DIM_Y",
        "DIM_X * DIM_Y <= 1024",
        "DIM_x * DIM_Y >= 32",
    ]

    results, env = tune_kernel(
        kernel_name="optimized_matmul",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=answer,
        atol=atol,
        restrictions = restrictions,
        block_size_names = ["DIM_X", "DIM_Y"],
        grid_div_x = ["BLK_M"],
        grid_div_y = ["BLK_N"],
        strategy = "bayes_opt",
        strategy_options = {"max_fevals": 100},
        call_function=call_numba,
    )



if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024

    run_basic(M, N, K)
    tune_basic(M, N, K)

    run_optimized(M, N, K)
    tune_optimized(M, N, K)



  