import torch
import numpy as np
from numba import cuda, float32
from kernel_tuner import tune_kernel
from pathlib import Path

FULL_PATH = Path(__file__).resolve()

# Example taken from https://nvidia.github.io/numba-cuda/user/examples.html#matrix-multiplication
@cuda.jit(cache=True)
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


# Example taken from https://nvidia.github.io/numba-cuda/user/examples.html#matrix-multiplication
# Changed data type from float32 to float16
@cuda.jit(cache=True, fastmath=True)
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    TPB = 16 # TEMP voor overhead testing
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=np.float16)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=np.float16)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


# Translated to numba-cuda from https://github.com/cupy/cupy/blob/main/examples/gemm/sgemm.cu
@cuda.jit(cache=True)
def optimized_matmul(M, N, K, A, B, C):
    DIM_X = 16
    DIM_Y = 16
    BLK_M = 64
    BLK_N = 64
    BLK_K = 16
    THR_M = 4
    THR_N = 4

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


def run_matmul(M, N, K):

    # create numpy arrays
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

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
    fast_matmul[blocks, threads](A_d, B_d, C_d)

    # copy result back
    C_result = C_d.copy_to_host()

    # check
    np.testing.assert_allclose(C_result, A @ B, rtol=1e-2)

    print("Correct!")


def call_numba(kernel_function, args, kwargs, grid, threads):
    numba_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            numba_args.append(cuda.as_cuda_array(arg))
        else:
            numba_args.append(arg)
    kernel_function[grid, threads](*args, **kwargs)


def tune(M, N, K):
    # create numpy arrays
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    size = (M, N)
    args = [A, B, C]
    tune_params = dict()
    tune_params["block_size_x"] = [4, 8, 16, 32, 64, 128, 256]
    tune_params["block_size_y"] = [4, 8, 16, 32, 64, 128, 256]
    tune_params["TPB"] = [4, 8, 16, 32, 64, 128, 256]
    restrictions = ["block_size_x == block_size_y", "block_size_x == TPB"]
    
    answer = [None, None, A @ B]
    atol = M * 2**(-11)

    results, env = tune_kernel(
        kernel_name="fast_matmul",
        kernel_source=FULL_PATH,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=answer,
        atol=atol,
        call_function=call_numba,
        restrictions=restrictions
    )


if __name__ == "__main__":
    
    #run_matmul(128, 96, 64)
    #tune(1024, 1024, 1024)


    M, N, K = 1024, 1024, 1024

    # random FP16 inputs
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)

    # output
    C = np.zeros((M, N), dtype=np.float16)

    # copy to device
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.to_device(C)

    # launch config
    threads_per_block = (16, 16)

    blocks_per_grid_x = (M + 64 - 1) // 64
    blocks_per_grid_y = (N + 64 - 1) // 64
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # run kernel
    optimized_matmul[blocks_per_grid, threads_per_block](M, N, K, dA, dB, dC)

    # copy result back
    C_result = dC.copy_to_host()

    # reference (FP32 accumulate like your kernel)
    C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)

    # check error
    max_error = np.max(np.abs(C_result - C_ref))
    print("Max error:", max_error)

    # tolerance (FP16 is noisy)
    if max_error < 4:
        print("✅ Looks correct")
    else:
        print("❌ Something is off")
        print("Expected: ", C_ref, "\nGot: ", C_result)

    '''
    128: 16, 8
    256: 8, 16
    512: 8, 16
    1024: 16, 8
    4096: 16, 8
    8192: 16, 8 
    '''