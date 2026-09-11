#include "cuda_tile.h"

__tile_global__ void vector_add_tile(float* a, float* b, float* out, int n) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    a   = ct::assume_aligned(a,   16_ic);
    b   = ct::assume_aligned(b,   16_ic);
    out = ct::assume_aligned(out, 16_ic);

    // Step 1: attach a shape to each raw pointer. n is a runtime value (dynamic extent).
    auto aSpan = ct::tensor_span{a,   ct::extents{n}};
    auto bSpan = ct::tensor_span{b,   ct::extents{n}};
    auto oSpan = ct::tensor_span{out, ct::extents{n}};

    // Step 2: partition each span into tiles of TILE_SIZE elements (tuned by kernel_tuner).
    constexpr auto tile = ct::integral_constant<TILE_SIZE>{};
    auto aView = ct::partition_view{aSpan, ct::shape{tile}};
    auto bView = ct::partition_view{bSpan, ct::shape{tile}};
    auto oView = ct::partition_view{oSpan, ct::shape{tile}};

    int  bx    = ct::bid().x;             // this block's tile-space index along .x
    auto aTile = aView.load(bx);          // pick the bx-th tile of a
    auto bTile = bView.load(bx);
    oView.store(aTile + bTile, bx);       // write the tile back at the bx-th position of out
}
