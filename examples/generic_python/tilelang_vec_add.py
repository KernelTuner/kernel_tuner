import tilelang
import tilelang.language as T
import torch
from kernel_tuner import tune_kernel

#@tilelang.jit  # infers target from tensors at first call
def add(N: int, dtype: str = 'float32', block: int = 256,):

    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                # Optional — LegalizeSafeMemoryAccess inserts a guard when an access may be OOB
                C[gi] = A[gi] + B[gi]

    return add_kernel


def run_normal():
    # Host side (PyTorch shown; NumPy/DLPack also supported)
    N = 1 << 20
    A = torch.randn(N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, device='cuda', dtype=torch.float32)
    C = torch.empty(N, device='cuda', dtype=torch.float32)

    kernel = add(N)
    kernel(A, B, C)  # runs on GPU
    torch.testing.assert_close(C, A + B)
    print("done")


def call_tilelang(kernel_function, args, kwargs, grid, threads, params):
    compiled_kernel = kernel_function(**kwargs) # cached, so second time only cache lookup is performed
    compiled_kernel(*args)

def tune():
    N = 1 << 20
    A = torch.randn(N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, device='cuda', dtype=torch.float32)
    C = torch.empty(N, device='cuda', dtype=torch.float32)

    args = [A, B, C]
    tune_params = dict()
    tune_params["block"] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    tune_params["N"] = [N] # Not a tune param, but enables an easier call function

    answer = [None, None, (A + B).cpu()]
    
    res, env = tune_kernel("add", add, N, args, tune_params, lang="generic_python", 
            call_function=call_tilelang, decorator="@tilelang.jit", answer=answer)
    
tune()