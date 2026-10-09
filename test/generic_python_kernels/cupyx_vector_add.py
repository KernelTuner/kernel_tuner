from cupyx import jit

kernel_name = "vector_add"


@jit.rawkernel()
def vector_add(c, a, b, n):
    i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if i < n:
        c[i] = a[i] + b[i]


def arguments(c, a, b, n):
    return [c, a, b, n]


def tune_params(n):
    return {"block_size_x": [128, 256]}
