import torch
import taichi as ti

from kernel_tuner import tune_kernel



ti.init(arch=ti.gpu)


# TODO make sure this is zero-copy
@ti.kernel
def matmul(A: ti.types.ndarray(dtype=ti.f16, ndim=2), B: ti.types.ndarray(dtype=ti.f16, ndim=2), C: ti.types.ndarray(dtype=ti.f16, ndim=2)):
    BLOCK_DIM = 16
    ti.loop_config(block_dim=BLOCK_DIM)  

    K_dim = A.shape[1]

    for i, j in C:
        sum = 0.0
        for k in range(K_dim):
            sum += A[i, k] * B[k, j]
        C[i, j] = ti.cast(sum, ti.f16)


def call_taichi(kernel_function, args, kwargs):
    kernel_function(*args, **kwargs)

def tune(M, N, K):
    torch_A = torch.rand((N, K), device='cuda', dtype=torch.float16)
    torch_B = torch.rand((K, M), device='cuda', dtype=torch.float16)
    torch_C = torch.empty((N, M), device='cuda', dtype=torch.float16)

    size = M * N
    args = [torch_A, torch_B, torch_C]
    tune_params = {"BLOCK_DIM": {4, 8, 16, 32, 64, 128, 256, 512, 1024}}

    answer = [None, None, (torch_A @ torch_B).cpu()]

    results, env = tune_kernel(
        kernel_name="matmul",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        answer=answer,
        lang="generic_python",
        call_function=call_taichi,
        atol=1e-1,
    )

if __name__ == "__main__":
    tune(128, 128, 128)


    
