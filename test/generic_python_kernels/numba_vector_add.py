from numba import cuda

kernel_name = "vector_add"


@cuda.jit
def vector_add(c, a, b, n):
    i = cuda.grid(1)
    if i < n:
        c[i] = a[i] + b[i]


def arguments(c, a, b, n):
    return [c, a, b, n]


def tune_params(n):
    return {"block_size_x": [128, 256]}
