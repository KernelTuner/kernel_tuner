import tilelang
import tilelang.language as T
import torch
from kernel_tuner import tune_kernel


#@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype: str = 'float16', accum_dtype: str = 'float32'):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Define a grid with enough blocks to cover M×N
        num_threads=128
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):

            # Allocate shared memory for the current tile of A and B
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # Allocate a local (register) fragment for partial accumulations
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Enable swizzle-based rasterization for better L2 locality
            panel_size = 4
            T.use_swizzle(panel_size=panel_size, enable=True)

            # Initialize the local accumulation buffer to zero
            T.clear(C_local)

            num_stages=3

            # Loop over the K dimension in block_K chunks, using a pipeline
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Copy from global memory to shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)

                # Perform a matrix multiply-accumulate on the tile
                T.gemm(A_shared, B_shared, C_local)

            # Copy the accumulated result from local memory (C_local) to global memory (C)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

@tilelang.jit
def matmul_with_decorator(M, N, K, block_M, block_N, block_K, dtype: str = 'float16', accum_dtype: str = 'float32'):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Define a grid with enough blocks to cover M×N
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):

            # Allocate shared memory for the current tile of A and B
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # Allocate a local (register) fragment for partial accumulations
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Initialize the local accumulation buffer to zero
            T.clear(C_local)

            # Loop over the K dimension in block_K chunks, using a 3-stage pipeline
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy from global memory to shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)

                # Perform a matrix multiply-accumulate on the tile
                T.gemm(A_shared, B_shared, C_local)

            # Copy the accumulated result from local memory (C_local) to global memory (C)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main



def run(m, n, k):
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    kernel = matmul(m, n, k, 128, 128, 32)
    kernel(a, b, c)
    ref_c = a @ b
    tol = m * 2**(-11)
    # Validate results
    torch.testing.assert_close(c, ref_c, rtol=tol, atol=tol)


def call_tilelang(kernel_function, args, kwargs, grid, threads, params):
    compiled_kernel = kernel_function(**kwargs)
    compiled_kernel(*args)


def time(m, n, k):
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    c_ans = a @ b

    args = [a, b, c]
    tune_params = dict()
    tune_params["M"] = [m]
    tune_params["K"] = [k]
    tune_params["N"] = [n] 
    tune_params["block_M"] = [64, 128]
    tune_params["block_N"] = [64, 128]
    tune_params["block_K"] = [32, 64]

    results_kt, env = tune_kernel("matmul", matmul, m * n, args, tune_params, lang="generic_python",
            call_function=call_tilelang, decorator="@tilelang.jit", verbose=False, iterations=100)
    
    import time
    num_repeats = 100
    times_direct = []
    for config in results_kt:
        bs_m = config["block_M"]
        bs_n = config["block_N"]
        bs_k = config["block_K"]

        c = torch.empty(m, n, device="cuda", dtype=torch.float16)
        
        kernel = matmul_with_decorator(m, n, k, bs_m, bs_n, bs_k)
        kernel(a, b, c)

        torch.allclose(c.cpu(), c_ans.cpu(), atol=m * 2**(-11))
       
        for i in range(num_repeats):
            times = []
           
            torch.cuda.synchronize()
            start = time.time()
            kernel(a, b, c)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        avg_time_ms = round((1000 * sum(times) / len(times)), 3)
        times_direct.append(avg_time_ms)
        print(f"BLOCK_SIZE_M={bs_m}, BLOCK_SIZE_N={bs_n}, BLOCK_SIZE_K={bs_k}, time={avg_time_ms}ms")


    import matplotlib.pyplot as plt

    # Extract times
    times_kt = [cfg['time'] for cfg in results_kt]

    # x-axis labels
    configs = [f"config{i}" for i in range(len(times_kt))]
    x = range(len(configs))
    
    plt.figure(figsize=(10,6))
    plt.plot(configs, times_kt, marker='s', label='KernelTuner')
    plt.plot(configs, times_direct, marker='x', label='Direct')
    plt.ylabel('Time (ms)')
    plt.xlabel('Configuration')
    plt.title('Kernel execution time per configuration')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tilelang.png")
    print("saved fig")


def tune(m, n, k):
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    c_actual = a @ b

    args = [a, b, c]
    tune_params = dict()
    tune_params["M"] = [m]
    tune_params["K"] = [k]
    tune_params["N"] = [n] 
    tune_params["block_M"] = [64, 128, 256]
    tune_params["block_N"] = [64, 128, 256]
    tune_params["block_K"] = [32, 64, 128]
    tune_params["num_stages"] = [2, 3, 4]
    tune_params["panel_size"] = [4, 8] # equivalent to group size m in Triton
    tune_params["num_threads"] = [64, 128, 256]

    restrictions = [
        # tile size budget
        "block_M * block_N <= 16384",

        # aspect ratio <= 4 (no max/min allowed, so expand manually)
        "block_M <= 4 * block_N",
        "block_N <= 4 * block_M",

        # large K only with reasonably large M/N
        "not (block_K == 128 and block_M < 64 and block_N < 64)",
    ]

    tol = m * 2**(-11)
    answer = [None, None, c_actual.cpu()]

    results, env = tune_kernel("matmul", matmul, m * n, args, tune_params, atol=tol, lang="generic_python",
            call_function=call_tilelang, restrictions=restrictions, answer=answer, decorator="@tilelang.jit", verbose=False)

if __name__ == "__main__":
    #m, n, k = 1024, 1024, 1024
    m, n, k = 8192, 8192, 8192
    time(m, n, k)