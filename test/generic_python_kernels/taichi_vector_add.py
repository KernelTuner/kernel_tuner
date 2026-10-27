import taichi as ti

ti.init(arch=ti.cuda)

# Taichi silently falls back to the CPU when it cannot use CUDA
requires_skip = None if ti.lang.impl.current_cfg().arch == ti.cuda else "Taichi cannot use CUDA"

kernel_name = "vector_add"

block_size_x = 128  # replaced by the value of the tunable parameter


@ti.kernel
def vector_add(
    c: ti.types.ndarray(dtype=ti.f32, ndim=1),
    a: ti.types.ndarray(dtype=ti.f32, ndim=1),
    b: ti.types.ndarray(dtype=ti.f32, ndim=1),
    n: ti.i32,
):
    ti.loop_config(block_dim=block_size_x)
    for i in range(n):
        c[i] = a[i] + b[i]


def arguments(c, a, b, n):
    return [c, a, b, n]


def tune_params(n):
    return {"block_size_x": [128, 256]}
