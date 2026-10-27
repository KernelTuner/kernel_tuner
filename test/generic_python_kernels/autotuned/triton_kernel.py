import triton
import triton.language as tl

from kernel_tuner import autotune

AUTOTUNE = dict(tune_params={"block_size_x": [128, 256], "num_warps": [2, 4]}, key=["n"])


@autotune(**AUTOTUNE)
@triton.jit
def vector_add(c_ptr, a_ptr, b_ptr, n, block_size_x: tl.constexpr):
    offsets = tl.program_id(axis=0) * block_size_x + tl.arange(0, block_size_x)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def launch(kernel, c, a, b, n):
    kernel[lambda meta: (triton.cdiv(n, meta["block_size_x"]),)](c, a, b, n)
