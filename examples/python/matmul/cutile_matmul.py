import cuda.tile as ct
import torch

from kernel_tuner import tune_kernel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for call_functions.py
from call_functions import call_cutile  # noqa: E402


@ct.kernel
def matmul(A, B, C, tile_m: ct.Constant[int], tile_n: ct.Constant[int], tile_k: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)

    # accumulate in fp32 over all tiles in the K dimension
    acc = ct.full((tile_m, tile_n), 0, dtype=ct.float32)
    for k in range(ct.num_tiles(A, axis=1, shape=(tile_m, tile_k))):
        a = ct.load(A, index=(bidx, k), shape=(tile_m, tile_k), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, bidy), shape=(tile_k, tile_n), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)

    ct.store(C, index=(bidx, bidy), tile=ct.astype(acc, C.dtype))


@ct.kernel
def matmul_swizzled(A, B, C, tile_m: ct.Constant[int], tile_n: ct.Constant[int], tile_k: ct.Constant[int],
                    group_size_m: ct.Constant[int]):
    # map the 1D block index to a tile of C in groups of group_size_m rows of tiles,
    # so consecutive blocks reuse the same tiles of B, which improves L2 cache reuse
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(A.shape[0], tile_m)
    num_bid_n = ct.cdiv(B.shape[1], tile_n)
    num_bid_in_group = group_size_m * num_bid_n
    first_bid_m = (bid // num_bid_in_group) * group_size_m
    group_size = ct.minimum(num_bid_m - first_bid_m, group_size_m)
    bidx = first_bid_m + (bid % num_bid_in_group) % group_size
    bidy = (bid % num_bid_in_group) // group_size

    acc = ct.full((tile_m, tile_n), 0, dtype=ct.float32)
    for k in range(ct.num_tiles(A, axis=1, shape=(tile_m, tile_k))):
        a = ct.load(A, index=(bidx, k), shape=(tile_m, tile_k), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, bidy), shape=(tile_k, tile_n), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)

    ct.store(C, index=(bidx, bidy), tile=ct.astype(acc, C.dtype))


def run(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    tile_m, tile_n, tile_k = 64, 64, 32
    grid = (ct.cdiv(M, tile_m), ct.cdiv(N, tile_n), 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul, (A, B, C, tile_m, tile_n, tile_k))

    assert torch.allclose(C, A @ B, rtol=1e-2, atol=M * 2**(-11))
    print("Passed")


def tune(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    args = [A, B, C]
    answer = [None, None, (A @ B).cpu()]

    tune_params = dict()
    tune_params["tile_m"] = [32, 64, 128, 256]
    tune_params["tile_n"] = [32, 64, 128, 256]
    tune_params["tile_k"] = [16, 32, 64, 128]

    # each block computes one tile of C, so the tile sizes act as block sizes to compute the grid
    results, env = tune_kernel("matmul", __file__, (M, N), args, tune_params, answer=answer,
                               atol=M * 2**(-11), block_size_names=["tile_m", "tile_n"],
                               lang="generic_python", call_function=call_cutile)


def tune_swizzled(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    args = [A, B, C]
    answer = [None, None, (A @ B).cpu()]

    tune_params = dict()
    tune_params["tile_m"] = [32, 64, 128, 256]
    tune_params["tile_n"] = [32, 64, 128, 256]
    tune_params["tile_k"] = [16, 32, 64, 128]
    tune_params["group_size_m"] = [1, 2, 4, 8, 16]

    # 1D grid with one block per tile of C
    results, env = tune_kernel("matmul_swizzled", __file__, M * N, args, tune_params, answer=answer,
                               atol=M * 2**(-11), block_size_names=["tile_m"], grid_div_x=["tile_m", "tile_n"],
                               strategy="bayes_opt", strategy_options={"max_fevals": 100},
                               lang="generic_python", call_function=call_cutile)


if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    run(M, N, K)
    tune(M, N, K)
    tune_swizzled(M, N, K)
