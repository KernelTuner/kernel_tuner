import torch
import taichi as ti

from kernel_tuner import tune_kernel
from pathlib import Path 

FULL_PATH = Path(__file__).resolve()

ti.init(arch=ti.gpu)

# NOTE tuning works on vibranium. Taichi does not work on DAS6
# TODO make sure this is zero-copy
@ti.kernel
def matmul(A: ti.types.ndarray(), B: ti.types.ndarray(), C: ti.types.ndarray()):
    BLOCK_DIM = 16
    ti.loop_config(block_dim=BLOCK_DIM)  

    K_dim = A.shape[1]

    for i, j in C:
        sum = 0.0
        for k in range(K_dim):
            sum += A[i, k] * B[k, j]
        C[i, j] = sum


def call_taichi(kernel_function, args, kwargs):
    kernel_function(*args, **kwargs)

def tune(M, N, K):
    torch_A = torch.rand((N, K), device='cuda', dtype=torch.float32)
    torch_B = torch.rand((K, M), device='cuda', dtype=torch.float32)
    torch_C = torch.empty((N, M), device='cuda', dtype=torch.float32)

    size = M * N
    args = [torch_A, torch_B, torch_C]
    tune_params = {"BLOCK_DIM": {4, 8, 16, 32, 64, 128, 256, 512, 1024}}

    answer = [None, None, (torch_A @ torch_B).cpu()]

    results, env = tune_kernel(
        kernel_name="matmul",
        kernel_source=FULL_PATH,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        answer=answer,
        lang="generic_python",
        call_function=call_taichi,
    )

if __name__ == "__main__":
    tune(128, 128, 128)


    
