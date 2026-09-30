import warp as wp

kernel_name = "vector_add"


@wp.kernel
def vector_add(c: wp.array(dtype=float), a: wp.array(dtype=float), b: wp.array(dtype=float), n: int):
    i = wp.tid()
    if i < n:
        c[i] = a[i] + b[i]


def arguments(c, a, b, n):
    return [c, a, b, n]


def tune_params(n):
    return {"block_size_x": [128, 256]}
