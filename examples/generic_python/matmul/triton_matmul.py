import torch
import triton
import triton.language as tl

from kernel_tuner import tune_kernel
from examples.generic_python.call_functions import call_triton

@triton.jit
def matmul_basic(
        a_ptr, b_ptr, c_ptr, # Pointers
        M, N, K, # Matrix sizes
        stride_am, stride_ak,  # Strides
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, # Tile sizes
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
):
    # Each program computes one BLOCK_SIZE_M x BLOCK_SIZE_N tile of C
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Compute row/column indices for the tile
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create pointers to A and B tiles
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )

        accumulator = tl.dot(a, b, accumulator)

        # advance K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store result
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)




# This kernel is copied from the Triton project:
# https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py
#
# Original example: 03-matrix-multiplication.py
# Copyright (c) the Triton authors
#
# Modifications in file:
# - Removed auto-tuning decorators
# - Removed activation function
@triton.jit
def matmul_opt(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)



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
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    assert torch.allclose(C, C_ref, rtol=1e-2, atol= M * 2**(-11))

    print("Passed")


def run_opt(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    C_ref = A @ B

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    matmul_opt[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
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
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
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
        lang="generic_python",
        answer=[None, None, C_ref.cpu(), None, None, None, None, None, None, None, None, None],
        atol=M * 2**(-11),
        block_size_names = ["BLOCK_SIZE_M", "BLOCK_SIZE_M"],
        strategy = "bayes_opt",
        strategy_options = {"max_fevals": 100},
        call_function=call_triton,
    )


def tune_opt(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    C_ref = A @ B

    size = M * N

    args = [A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    ]

    tune_params = dict()
    tune_params["BLOCK_SIZE_M"] = [2**i for i in range(5, 9)]
    tune_params["BLOCK_SIZE_N"] = [2**i for i in range(5, 9)]
    tune_params["BLOCK_SIZE_K"] = [2**i for i in range(4, 8)] 
    tune_params["GROUP_SIZE_M"] = [4, 6, 8, 10]
    tune_params["num_warps"] = [2, 4, 8, 16]
    tune_params["num_stages"] = [1, 2, 3, 4, 5]
    tune_params["M_dim"] = [M]

    restrictions = ["GROUP_SIZE_M * BLOCK_SIZE_M < M_dim"] # Otherwise grouped ordering has no effect

    results, env = tune_kernel(
        kernel_name="matmul_opt",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, C_ref.cpu(), None, None, None, None, None, None, None, None, None],
        atol=M * 2**(-11),
        block_size_names = ["BLOCK_SIZE_M", "BLOCK_SIZE_M"],
        grid_div_x = ["BLOCK_SIZE_M", "BLOCK_SIZE_N"],
        restrictions = restrictions,
        strategy = "bayes_opt",
        strategy_options = {"max_fevals": 100},
        call_function=call_triton,
    )
    


if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    run_basic(M, N, K)
    run_opt(M, N, K)

    tune_basic(M, N, K)
    tune_opt(M, N, K)
