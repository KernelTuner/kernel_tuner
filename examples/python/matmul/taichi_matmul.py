import numpy as np
import taichi as ti

from kernel_tuner import tune_kernel
from examples.generic_python.call_functions import call_taichi

ti.init(arch=ti.gpu)

BLOCK_DIM = 512
@ti.kernel
def matmul(A: ti.types.ndarray(dtype=ti.f16, ndim=2), B: ti.types.ndarray(dtype=ti.f16, ndim=2), C: ti.types.ndarray(dtype=ti.f16, ndim=2)):
    K_dim = A.shape[1]

    ti.loop_config(block_dim=BLOCK_DIM)  
    for i, j in C:
        sum = 0.0
        for k in range(K_dim):
            sum += A[i, k] * B[k, j]
        C[i, j] = ti.cast(sum, ti.f16)


def run(M, N, K):
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    C_ref = A @ B

    matmul(A, B, C)

    np.testing.assert_allclose(C, C_ref, rtol=1e-2, atol=M * 2**(-11))
    print("Succes")


def tune(M, N, K):
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    size = M * N
    args = [A, B, C]
    tune_params = {"BLOCK_DIM": [2**i for i in range(5, 11)]}

    answer = [None, None, A @ B]

    results, env = tune_kernel(
        kernel_name="matmul",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        answer=answer,
        lang="generic_python",
        call_function=call_taichi,
        atol=M * 2**(-11),
        block_size_names = ["BLOCK_DIM"],
    )

if __name__ == "__main__":
    run(1024, 1024, 1024)
    tune(1024, 1024, 1024)


    
