import tilelang
import tilelang.language as T

from kernel_tuner import autotune

AUTOTUNE = dict(tune_params={"block_size_x": [128, 256]}, key=["n"])


@autotune(**AUTOTUNE)
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


def launch(kernel, c, a, b, n):
    kernel(n)(c, a, b)
