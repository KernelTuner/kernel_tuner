#!/usr/bin/env python
""" This is a minimal example to tune a CUDA Tile vector add kernel """

import numpy
from kernel_tuner import tune_kernel

def tune():

    size = 3*2**24

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [a, b, c, n]

    # TILE_SIZE controls how many elements each tile processes. kernel_tuner injects
    # it as a #define so the kernel can use it as a compile-time constant.
    # grid_x = size / TILE_SIZE (number of tiles), computed automatically by kernel_tuner.
    tune_params = {"TILE_SIZE": [8, 16, 32, 64, 128, 256]}

    answer = [None, None, a+b, None]

    compiler_options = ["-enable-tile", "-std=c++20"]

    results, env = tune_kernel("vector_add_tile<TILE_SIZE>", "vector_add_tile.cu", size, args,
                       tune_params, lang="NVCUDA", compiler_options=compiler_options,
                       answer=answer, verbose=True, grid_div_x=["TILE_SIZE"])

    return results


if __name__ == "__main__":
    tune()
