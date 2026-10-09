#!/usr/bin/env python
"""Autotune a cuTile vector add kernel with the Kernel Tuner autotune decorator.

The decorated kernel is launched like ct.launch, but with kernel.launch(stream, grid, args). The tuning key is a
function of the kernel arguments, here the length of the vectors.

The tuning results are stored in wisdom files in the directory "wisdom", in the format of Kernel Launcher. When the
example runs again, the kernel is not tuned again, but the best configuration from the wisdom file is launched.
"""

import cuda.tile as ct
import torch

from kernel_tuner import autotune


@autotune(
    tune_params={"tile_size": [128, 256, 512, 1024, 2048]},
    key=lambda c, a, b: c.shape[0],
    wisdom="wisdom",
)
@ct.kernel
def vector_add(c, a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)


def grid(meta):
    """One block per tile, meta contains the kernel arguments and the tunable parameters."""
    return (ct.cdiv(meta["c"].shape[0], meta["tile_size"]),)


if __name__ == "__main__":
    n = 1 << 24
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.empty_like(a)

    # instead of ct.launch(stream, grid, vector_add, (c, a, b, tile_size)), the tile size is chosen by the decorator
    vector_add.launch(torch.cuda.current_stream(), grid, (c, a, b))

    torch.testing.assert_close(c, a + b)
    tuned = "tuned" if vector_add.tuning_results else "read from the wisdom file"
    print(f"best configuration ({tuned}):", vector_add.best_configs)
