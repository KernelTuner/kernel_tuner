import warp as wp

from kernel_tuner import autotune

wp.init()

AUTOTUNE = dict(tune_params={"block_dim": [128, 256]}, key=["n"])


@autotune(**AUTOTUNE)
@wp.kernel
def vector_add(c: wp.array(dtype=float), a: wp.array(dtype=float), b: wp.array(dtype=float), n: int):
    i = wp.tid()
    if i < n:
        c[i] = a[i] + b[i]


def launch(kernel, c, a, b, n):
    c, a, b = (wp.from_torch(x) for x in (c, a, b))
    kernel.launch(n, inputs=[c, a, b, n])


def launch_subscript(kernel, c, a, b, n):
    c, a, b = (wp.from_torch(x) for x in (c, a, b))
    kernel[n](c, a, b, n)
