from numba import cuda

kernel_name = "vector_add"


# Numba links float16 kernels with other files, which means that Numba cannot store them in its cache
@cuda.jit
def vector_add(c, a, b, n):
    i = cuda.grid(1)
    if i < n:
        c[i] = a[i] + b[i]
