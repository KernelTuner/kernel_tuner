import numpy as np
from numba import cuda

from kernel_tuner import tune_kernel


# Source: https://nvidia.github.io/numba-cuda/user/examples.html#matrix-multiplication
@cuda.jit
def matmul(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


def run_basic(M, N, K):
    # create numpy arrays
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)

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
    
    answer = [None, None, (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)]
    atol = M * 2**(-11)

    results, env = tune_kernel(
        kernel_name="matmul",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        answer=answer,
        atol=atol,
        restrictions=restrictions,
    )


if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096

    run_basic(M, N, K)
    tune_basic(M, N, K)
