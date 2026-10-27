import cutlass.cute as cute

from kernel_tuner import autotune

AUTOTUNE = dict(tune_params={"block_size_x": [128, 256]}, key=["n"])


@cute.kernel
def vector_add_kernel(gC: cute.Tensor, gA: cute.Tensor, gB: cute.Tensor, n: cute.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bdim * bidx + tidx
    if i < n:
        gC[i] = gA[i] + gB[i]


@autotune(**AUTOTUNE)
@cute.jit
def vector_add(mC: cute.Tensor, mA: cute.Tensor, mB: cute.Tensor, n: cute.Int32):
    block_size_x = 128
    vector_add_kernel(mC, mA, mB, n).launch(
        grid=(cute.ceil_div(n, block_size_x), 1, 1),
        block=(block_size_x, 1, 1),
    )


def launch(kernel, c, a, b, n):
    kernel(c, a, b, n)
