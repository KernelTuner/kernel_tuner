from numba import cuda

from kernel_tuner import autotune

# the thread block size is a launch dimension in Numba, so the decorator computes the launch dimensions
AUTOTUNE = dict(
    tune_params={"block_size_x": [128, 256]},
    key=["n"],
    grid=lambda p: ((p["n"] + p["block_size_x"] - 1) // p["block_size_x"],),
    threads=lambda p: (p["block_size_x"],),
)


@autotune(**AUTOTUNE)
@cuda.jit
def vector_add(c, a, b, n):
    i = cuda.grid(1)
    if i < n:
        c[i] = a[i] + b[i]


def launch(kernel, c, a, b, n):
    c, a, b = (cuda.as_cuda_array(x) for x in (c, a, b))
    kernel[1, 1](c, a, b, n)  # the launch dimensions are replaced by those of the decorator
