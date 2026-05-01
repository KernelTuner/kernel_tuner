import torch
import tilelang
import tilelang.language as T

import itertools

from kernel_tuner import tune_kernel
from examples.generic_python.call_functions import call_tilelang

# https://github.com/tile-ai/tilelang/tree/main/examples/gemm
# num_threads and num_stages added as variables to enable tuning.
@tilelang.jit
def matmul_basic(M:int, N:int, K:int, block_M:int, block_N:int, block_K:int, dtype:str="float16", accum_dtype:str="float32"):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        num_threads = 128
        num_stages = 3
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            # We do use shared memory, even though this is a basic kernel. However, you don't 
            # really get around this because T.gemm can not handle global memory directly.
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            
            # We do use a pipelining optimization here, because this is 'the basic way' 
            # of writing for loops in TileLang. 
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])
            
    return gemm




# https://github.com/tile-ai/tilelang/tree/main/examples/gemm
# num_threads and num_stages were added as metaparameters to make it possible to compare the built-in tuner
# with Kernel Tuner.
# Removed annotated memory layout for tiles A and B
@tilelang.jit
def matmul_opt(M, N, K, block_M, block_N, block_K, num_threads=128, num_stages=3, dummy=0, dtype=T.float16, accum_dtype=T.float):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            # Allocate shared and local fragments
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Enable swizzle-based rasterization for better L2 locality
            T.use_swizzle(panel_size=10, enable=True)

            # Clear the local accumulation buffer
            T.clear(C_local)

            # Pipelined iteration over K dimension
            for idx in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Copy tile of A
                T.copy(A[by * block_M, idx * block_K], A_shared)

                # Parallel copy tile of B
                for ko, j in T.Parallel(block_K, block_N):
                    B_shared[ko, j] = B[idx * block_K + ko, bx * block_N + j]

                # Perform local GEMM on the shared-memory tiles
                T.gemm(A_shared, B_shared, C_local)

            # Copy the result tile back
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main



# TODO
def get_configs(kt=False):
    """
    Generate a list of kernel tuning configuration dictionaries for a tiled matrix-multiply.
    This function is used for tuning experiments with the built-in tuner

    Returns:
        List[dict]: A list of configuration dictionaries 
    """
    
    '''
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    block_K = [32, 64]
    num_stages = [0, 1, 2, 3]
    thread_num = [128, 256]
    enable_rasterization = [True, False]
    '''

    block_M = [64]
    block_N = [64]
    block_K = [32, 64]
    num_stages = [0, 3]
    thread_num = [128]
    enable_rasterization = [True]

    _configs = list(
        itertools.product(
            block_M,
            block_N,
            block_K,
            num_stages,
            thread_num,
            enable_rasterization,
        )
    )

    if kt:
        configs = {
            "block_M": block_M,
            "block_N": block_N,
            "block_K": block_K,
            "num_stages": num_stages,
            "thread_num": thread_num,
            "enable_rasteration": enable_rasterization,  
        }
    else:
        configs = [
            {
                "block_M": c[0],
                "block_N": c[1],
                "block_K": c[2],
                "num_stages": c[3],
                "thread_num": c[4],
                "enable_rasteration": c[5],  # keep param name for backward-compat
            }
            for c in _configs
        ]
    return configs



# TODO
# https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm_autotune.py
# changed gemm_autotune to gemm
# originally, B was transposed. I changed this so the kernel input is the same as in other languages.
@tilelang.autotune(configs=get_configs())
@tilelang.jit
def matmul_opt_autotune(M, N, K, dummy, block_M, block_N, block_K, num_stages, thread_num, enable_rasteration, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    transpose_B=True,
                )
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm



# DEZE mag denk ik weg
def main():
    kernel = matmul_opt_autotune(1024, 1024, 1024)
    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()
    c = torch.empty((1024, 1024), device='cuda', dtype=torch.float16)

    kernel(a, b, c)

    ref_c = a @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")




def run_basic(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    C_ref = A @ B

    kernel = matmul_basic(
        M, N, K,
        block_M=64,
        block_N=64,
        block_K=32,
    )

    kernel(A, B, C)

    assert torch.allclose(C, C_ref, atol=M * 2 **(-11), rtol=1e-2)
    print("Succes")


def run_opt(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    C_ref = A @ B

    kernel = matmul_opt(
        M, N, K,
        dummy=0,
        block_M=64,
        block_N=64,
        block_K=32,
        num_stages=3,
        num_threads=128,
    )

    kernel(A, B, C)

    assert torch.allclose(C, C_ref, atol=M * 2 **(-11), rtol=1e-2)
    print("Succes")


def tune_basic(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    C_ref = A @ B

    size = (M, N)
    

    args = [A, B, C]

    tune_params = dict()
    tune_params["block_M"] = [2**i for i in range(4, 10)]
    tune_params["block_N"] = [2**i for i in range(4, 10)]
    tune_params["block_K"] = [2**i for i in range(4, 10)] 
    tune_params["num_threads"] = [2**i for i in range(5, 11)]
    tune_params["num_stages"] = [1, 2, 3, 4, 5]
    tune_params["M"] = [M]
    tune_params["N"] = [N]
    tune_params["K"] = [K]

    restrictions = [
        "2 * num_stages * block_K * (block_M + block_N) <= 49152"
    ]

    results, env = tune_kernel(
        kernel_name="matmul_basic",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, C_ref.cpu()],
        atol=M * 2**(-11),
        strategy = "bayes_opt",
        strategy_options = {"max_fevals": 100},
        call_function=call_tilelang,
        verbose=True,
        restrictions=restrictions,
    )


def tune_opt(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    C_ref = A @ B

    size = (M, N)
    
    args = [A, B, C]

    tune_params = dict()
    tune_params["block_M"] = [2**i for i in range(4, 10)]
    tune_params["block_N"] = [2**i for i in range(4, 10)]
    tune_params["block_K"] = [2**i for i in range(4, 10)] 
    tune_params["num_threads"] = [2**i for i in range(5, 11)]
    tune_params["num_stages"] = [1, 2, 3, 4, 5]
    tune_params["M"] = [M]
    tune_params["N"] = [N]
    tune_params["K"] = [K]

    restrictions = [
        "2 * num_stages * block_K * (block_M + block_N) <= 49152"
    ]

    results, env = tune_kernel(
        kernel_name="matmul_opt",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, C_ref.cpu()],
        atol=M * 2**(-11),
        strategy = "bayes_opt",
        strategy_options = {"max_fevals": 100},
        call_function=call_tilelang,
        verbose=True,
        restrictions=restrictions,
    )




if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    #run_basic(M, N, K)
    #run_opt(M, N, K)

    #tune_basic(M, N, K)
    tune_opt(M, N, K)


