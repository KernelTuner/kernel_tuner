import torch
import numpy as np
from numba import cuda
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
@cuda.jit(cache=True)
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
    run_matmul(128, 96, 64)
    #tune(1024, 1024, 1024)

    '''
    128: 16, 8
    256: 8, 16
    512: 8, 16
    1024: 16, 8
    4096: 16, 8
    8192: 16, 8 
    '''