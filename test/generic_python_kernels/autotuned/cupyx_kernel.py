import cupy as cp
from cupyx import jit

from kernel_tuner import autotune

AUTOTUNE = dict(
    tune_params={"block_size_x": [128, 256]},
    key=["n"],
    grid=lambda p: ((p["n"] + p["block_size_x"] - 1) // p["block_size_x"],),
    threads=lambda p: (p["block_size_x"],),
)


@autotune(**AUTOTUNE)
@jit.rawkernel()
def vector_add(c, a, b, n):
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < n:
        c[i] = a[i] + b[i]


def launch(kernel, c, a, b, n):
    c, a, b = (cp.from_dlpack(x) for x in (c, a, b))
    kernel((1,), (1,), (c, a, b, n))  # the launch dimensions are replaced by those of the decorator
