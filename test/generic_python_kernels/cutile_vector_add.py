import cuda.tile as ct

kernel_name = "vector_add"


@ct.kernel
def vector_add(c, a, b, block_size_x: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(block_size_x,))
    b_tile = ct.load(b, index=(pid,), shape=(block_size_x,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)


def arguments(c, a, b, n):
    # tiles are loaded with bounds checks, so n is not needed
    return [c, a, b]


def tune_params(n):
    # each block processes one tile, so the tile size is the block size used to compute the grid
    return {"block_size_x": [128, 256]}
