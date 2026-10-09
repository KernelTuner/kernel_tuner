import torch
import triton
import triton.language as tl

from kernel_tuner import tune_kernel


@triton.jit
def matmul_basic(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Base offsets for this tile
    offs_m = m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Accumulator for C tile
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K in chunks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # Load tiles of A and B
        a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]   # [BLOCK_M, BLOCK_K]
        b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]   # [BLOCK_K, BLOCK_N]

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b)

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def run_basic(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    C_ref = A @ B

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )

    matmul_basic[grid](
        A, B, C,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    assert torch.allclose(C, C_ref, rtol=1e-2, atol= M * 2**(-11))

    print("Passed")


def tune_basic(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    C_ref = A @ B

    size = (M, N)

    args = [A, B, C,
        M, N, K,
    ]

    tune_params = dict()
    tune_params["BLOCK_SIZE_M"] = [2**i for i in range(5, 9)]
    tune_params["BLOCK_SIZE_N"] = [2**i for i in range(5, 9)]
    tune_params["BLOCK_SIZE_K"] = [2**i for i in range(4, 8)] 
    tune_params["num_warps"] = [2, 4, 8, 16]
    tune_params["num_stages"] = [1, 2, 3, 4, 5]

    results, env = tune_kernel(
        kernel_name="matmul_basic",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        answer=[None, None, C_ref.cpu(), None, None, None],
        atol=M * 2**(-11),
        block_size_names = ["BLOCK_SIZE_M", "BLOCK_SIZE_N"],
        strategy = "bayes_opt",
        strategy_options = {"max_fevals": 100},
    )


    


if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    run_basic(M, N, K)
    tune_basic(M, N, K)
