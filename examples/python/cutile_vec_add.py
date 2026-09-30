import cuda.tile as ct
import torch

from kernel_tuner import tune_kernel
from call_functions import call_cutile


@ct.kernel
def vec_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)


def tune():
    size = 1 << 24
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.zeros(size, device="cuda", dtype=torch.float32)

    args = [a, b, c]
    # tile sizes must be powers of 2
    tune_params = {"tile_size": [2**i for i in range(5, 13)]}
    answer = [None, None, (a + b).cpu()]

    # Each block processes one tile, so the tile size acts as the block size to compute the grid
    results, env = tune_kernel("vec_add", __file__, size, args, tune_params, answer=answer,
                               block_size_names=["tile_size"], lang="generic_python",
                               call_function=call_cutile)


if __name__ == "__main__":
    tune()
