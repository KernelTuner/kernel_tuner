import tilelang
import tilelang.language as T

kernel_name = "vector_add"


@tilelang.jit
def vector_add(n: int, block_size_x: int = 128):
    @T.prim_func
    def vector_add_kernel(
        C: T.Tensor((n,), "float32"),
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(T.ceildiv(n, block_size_x), threads=block_size_x) as bx:
            for i in T.Parallel(block_size_x):
                C[bx * block_size_x + i] = A[bx * block_size_x + i] + B[bx * block_size_x + i]

    return vector_add_kernel


def arguments(c, a, b, n):
    # n is passed to the kernel factory through the tunable parameters
    return [c, a, b]


def tune_params(n):
    return {"block_size_x": [128, 256], "n": [n]}
