import cuda.tile as ct
import torch

from kernel_tuner import autotune

AUTOTUNE = dict(tune_params={"block_size_x": [128, 256]}, key=lambda c, a, b: (c.shape[0],))


@autotune(**AUTOTUNE)
@ct.kernel
def vector_add(c, a, b, block_size_x: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(block_size_x,))
    b_tile = ct.load(b, index=(pid,), shape=(block_size_x,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)


def grid(meta):
    return (ct.cdiv(meta["c"].shape[0], meta["block_size_x"]),)


def launch(kernel, c, a, b, n):
    kernel.launch(torch.cuda.current_stream(), grid, (c, a, b))


def launch_subscript(kernel, c, a, b, n):
    kernel[grid](c, a, b)
