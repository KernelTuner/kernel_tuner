import torch

import triton
import triton.language as tl

from kernel_tuner import tune_kernel
from kernel_tuner import run_kernel

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

'''
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
'''
#@triton.jit
def matmul_kernel(
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
        GROUP_SIZE_M: tl.constexpr,  num_stages, num_warps#
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
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


def run_matmul(m, n, k):
    a = torch.rand(m, k, dtype=torch.float16).cuda()
    b = torch.rand(k, n, dtype=torch.float16).cuda() 
    c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
    c_expect = a @ b

    grid = lambda META: (triton.cdiv(m, META['BLOCK_SIZE_M']) * triton.cdiv(n, META['BLOCK_SIZE_N']), )

    matmul_kernel[grid](
        a, b, c_actual,  #
        m, n, k,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c_actual.stride(0), c_actual.stride(1),
        128, 256, 64, 8
    )

    torch.testing.assert_close(c_expect, c_actual, atol=1e-2, rtol=1e-2)


def run_matmul_kt(m, n, k):
    a = torch.rand(m, k, dtype=torch.float16).cuda()
    b = torch.rand(k, n, dtype=torch.float16).cuda() 
    c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
    c_expect = a @ b

    size = m * n 

    args = [a, b, c_actual, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), 
        c_actual.stride(0), c_actual.stride(1)]

    params = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M":4, "num_stages":3, "num_warps":4}

    result = run_kernel("matmul_kernel", matmul_kernel, size, args, params=params, grid_div_x=["BLOCK_SIZE_N", "BLOCK_SIZE_M"],
                               lang="generic_python", decorator="@triton.jit", call_function=call_triton, 
                               block_size_names=["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"])
    c_res = result[2]

    assert torch.allclose(c_res, c_expect.cpu(), atol=1e-2, rtol=1e-1)


    



def call_triton(kernel_function, args, kwargs, grid, threads, params):
    #print("using grid: ", grid)
    #print("args: ", args)
    #print("kwargs: ", kwargs)
    kernel_function[grid](*args, **kwargs)




def check_time(m, n, k):
    a = torch.rand(m, k, dtype=torch.float16).cuda()
    b = torch.rand(k, n, dtype=torch.float16).cuda() 
    c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
    c_expect = a @ b

    size = m * n
    args = [a, b, c_actual, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), 
        c_actual.stride(0), c_actual.stride(1)]
    tune_params = dict()
    tune_params["BLOCK_SIZE_M"] = [64, 128]
    tune_params["BLOCK_SIZE_N"] = [64, 128]
    tune_params["BLOCK_SIZE_K"] = [32, 64]
    tune_params["GROUP_SIZE_M"] = [4, 8]
    tune_params["num_stages"] = [3]
    tune_params["num_warps"] = [4]

    restrictions = [
        # tile size budget
        "BLOCK_SIZE_M * BLOCK_SIZE_N <= 16384",

        # aspect ratio <= 4 (no max/min allowed, so expand manually)
        "BLOCK_SIZE_M <= 4 * BLOCK_SIZE_N",
        "BLOCK_SIZE_N <= 4 * BLOCK_SIZE_M",

        # large K only with reasonably large M/N
        "not (BLOCK_SIZE_K == 128 and BLOCK_SIZE_M < 64 and BLOCK_SIZE_N < 64)",

        # 32x32 requires 8 warps
        "not (BLOCK_SIZE_M == 32 and BLOCK_SIZE_N == 32 and num_warps < 8)",
    ]

    grid_div = ["BLOCK_SIZE_N", "BLOCK_SIZE_M"]

    answer = [None] * 12
    answer[2] = c_expect.cpu()
    atol = 1e-2 #m * 2**(-11)


    
    results_ours, _ = tune_kernel("matmul_kernel", matmul_kernel, size, args, tune_params, grid_div_x = grid_div, 
                               restrictions=restrictions, iterations=100, answer=answer, atol=atol,
                               lang="generic_python", decorator="@triton.jit", call_function=call_triton, 
                               block_size_names=["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"])
    

    results_prev, _ = tune_kernel("matmul_kernel", matmul_kernel, size, args, tune_params, grid_div_x = grid_div, 
                              restrictions=restrictions, iterations=100, answer=answer, atol=atol,
                               lang="triton", block_size_names=["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"])
    
    
    
    import time
    num_repeats = 100
    times_direct = []
    for config in results_prev:

        bs_m = config["BLOCK_SIZE_M"]
        bs_n = config["BLOCK_SIZE_N"]
        bs_k = config["BLOCK_SIZE_K"]
        gs_m = config["GROUP_SIZE_M"]
        num_stages = config["num_stages"]
        num_warps = config["num_warps"]

        c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
        
        grid = (triton.cdiv(m, bs_m) * triton.cdiv(n, bs_n, ), )
        jit_function = triton.jit(matmul_kernel)

       
        jit_function[grid](
                a, b, c_actual,  
                m, n, k,  
                a.stride(0), a.stride(1),  
                b.stride(0), b.stride(1),  
                c_actual.stride(0), c_actual.stride(1),
                bs_m, bs_n, bs_k, gs_m, num_stages, num_warps
            )
        

        torch.allclose(c_expect.cpu(), c_actual.cpu(), atol=1e-2)
       

        
        for i in range(num_repeats):
            times = []
            
            #c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
           
            torch.cuda.synchronize()
            start = time.time()
            jit_function[grid](
                a, b, c_actual,  
                m, n, k,  
                a.stride(0), a.stride(1),  
                b.stride(0), b.stride(1),  
                c_actual.stride(0), c_actual.stride(1),
                bs_m, bs_n, bs_k, gs_m, num_stages, num_warps
            )

            torch.cuda.synchronize()
            times.append(time.time() - start)

        avg_time_ms = round((1000 * sum(times) / len(times)), 3)
        times_direct.append(avg_time_ms)
        print(f"BLOCK_SIZE_M={bs_m}, BLOCK_SIZE_N={bs_n}, BLOCK_SIZE_K={bs_k}, GROUP_SIZE_M={gs_m}, num_stages={num_stages}, num_warps={num_warps}, time={avg_time_ms}ms")


    import matplotlib.pyplot as plt

    # Extract times
    times_prev = [cfg['time'] for cfg in results_prev]
    times_ours = [cfg['time'] for cfg in results_ours]

    # x-axis labels
    configs = [f"config{i}" for i in range(len(times_prev))]
    x = range(len(configs))
    
    plt.figure(figsize=(10,6))
    plt.plot(configs, times_prev, marker='o', label='Triton tuned')
    plt.plot(configs, times_ours, marker='s', label='Generic tuned')
    plt.plot(configs, times_direct, marker='x', label='Direct')
    plt.ylabel('Time (ms)')
    plt.xlabel('Configuration')
    plt.title('Kernel execution time per configuration')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ouptut.png")



    
    
    
def tune_matmul(m, n, k):
    a = torch.rand(m, k, dtype=torch.float16).cuda()
    b = torch.rand(k, n, dtype=torch.float16).cuda() 
    c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
    c_expect = a @ b

    size = m * n
    args = [a, b, c_actual, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), 
        c_actual.stride(0), c_actual.stride(1)]
    tune_params = dict()
    tune_params["BLOCK_SIZE_M"] = [64, 128, 256]
    tune_params["BLOCK_SIZE_N"] = [64, 128, 256]
    tune_params["BLOCK_SIZE_K"] = [32, 64, 128]
    tune_params["GROUP_SIZE_M"] = [4, 8]
    tune_params["num_stages"] = [2, 3, 4]
    tune_params["num_warps"] = [4, 8]

    restrictions = [
    # tile size budget
    "BLOCK_SIZE_M * BLOCK_SIZE_N <= 16384",

    # aspect ratio <= 4 (no max/min allowed, so expand manually)
    "BLOCK_SIZE_M <= 4 * BLOCK_SIZE_N",
    "BLOCK_SIZE_N <= 4 * BLOCK_SIZE_M",

    # large K only with reasonably large M/N
    "not (BLOCK_SIZE_K == 128 and BLOCK_SIZE_M < 64 and BLOCK_SIZE_N < 64)",

    # 32x32 requires 8 warps
    "not (BLOCK_SIZE_M == 32 and BLOCK_SIZE_N == 32 and num_warps < 8)",
    ]

    grid_div = ["BLOCK_SIZE_N", "BLOCK_SIZE_M"]

    answer = [None] * 12
    answer[2] = c_expect.cpu()

    results, env = tune_kernel("matmul_kernel", matmul_kernel, size, args, tune_params, grid_div_x = grid_div, 
                               answer = answer, atol=4.0, restrictions=restrictions, 
                               lang="generic_python", decorator="@triton.jit", call_function=call_triton, 
                               block_size_names=["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"], strategy="simulated_annealing")

    



if __name__ == "__main__":
    m, n, k = 8192, 8192, 8192
    #m, n, k = 4096, 4096, 4096
    #tune_matmul(m, n, k)
    #check_time(m, n, k)
    #run_matmul_kt(m, n, k)


    
    


