import torch
import tilelang
import tilelang.language as T

from kernel_tuner import tune_kernel


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


def tune_basic(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    C_ref = A @ B

    size = (M, N)
    

    args = [A, B, C]

    tune_params = dict()
    tune_params["block_M"] = [2**i for i in range(5, 9)]
    tune_params["block_N"] = [2**i for i in range(5, 9)]
    tune_params["block_K"] = [2**i for i in range(4, 8)] 
    tune_params["num_threads"] = [64, 128, 256, 512]
    tune_params["num_stages"] = [0, 1, 2, 3, 4]
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
        answer=[None, None, C_ref.cpu()],
        atol=M * 2**(-11),
        strategy = "bayes_opt",
        strategy_options = {"max_fevals": 100},
        verbose=True,
        restrictions=restrictions,
    )


if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    run_basic(M, N, K)
    tune_basic(M, N, K)
