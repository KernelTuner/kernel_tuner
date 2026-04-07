import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def matmul_basic(M:int, N:int, K:int, block_M:int, block_N:int, block_K:int, dtype:str="float16", accum_dtype:str="float32"):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # We do use shared memory, even though this is a basic kernel. However, you don't 
            # really get around this because T.gemm can not handle global memory directly.
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            
            # We do use a pipelining optimization here, because this is 'the basic way' 
            # of writing for loops in TileLang. 
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])
            
    return gemm


# https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm_autotune.py
# changed gemm_autotune to gemm
# originally, B was transposed. I changed this so the kernel input is the same as in other languages.
#@tilelang.jit
def matmul_opt(M, N, K, block_M, block_N, block_K, num_stages, thread_num, enable_rasteration, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    transpose_B=False,
                )
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm



def main():

    kernel = matmul_opt(1024, 1024, 1024, 128, 128, 32, 3, 128, False)

    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()
    c = torch.empty((1024, 1024), device='cuda', dtype=torch.float16)

    kernel(a, b, c)

    ref_c = a @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")




if __name__ == "__main__":
    main()
