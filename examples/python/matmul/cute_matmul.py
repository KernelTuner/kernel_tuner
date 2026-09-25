import math
import torch
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

from kernel_tuner import tune_kernel
from examples.generic_python.call_functions import call_cute

# might need export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH to work

## Basic Matmul ================================================================

@cute.kernel
def naive_matmul_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, M: int, K: int, N: int):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    bdx, bdy, _ = cute.arch.block_dim()

    # Global indices
    n = bx * bdx + tx
    m = by * bdy + ty

    if m < M and n < N:
        acc = cutlass.Float32(0.0)

        for k in range(K):
            a = gA[m, k].to(cutlass.Float32)
            b = gB[k, n].to(cutlass.Float32)
            acc += a * b

        gC[m, n] = acc.to(gC.element_type)


@cute.jit
def matmul(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
):
    block_size_x = 16
    block_size_y = 16
    block = (block_size_x, block_size_y, 1)

    M, K = mA.shape
    _, N = mB.shape

    grid = (
        (N + block[0] - 1) // block[0],
        (M + block[1] - 1) // block[1],
        1,
    )

    kernel = naive_matmul_kernel(mA, mB, mC, M, K, N)

    kernel.launch(grid=grid, block=block)


def run_naive_matmul(M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    c_ref = a @ b

    # Convert to CuTe tensors
    a_ = from_dlpack(a, assumed_align=16)  
    b_ = from_dlpack(b, assumed_align=16)  
    c_ = from_dlpack(c, assumed_align=16)  

    compiled_kernel = cute.compile(matmul, a_, b_, c_)
    compiled_kernel(a_, b_, c_)

    assert torch.allclose(c, c_ref, atol=M * 2 **(-11), rtol=1e-2)
    print("Succes")


def tune_naive_matmul(M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    args = [a, b, c]
    size = M * N 
    answer = [None, None, (a @ b).cpu()]
    tune_params = dict()
    tune_params["block_size_x"] = [2**i for i in range(1, 10)]
    tune_params["block_size_y"] = [2**i for i in range(1, 10)]
    restrictions = ["block_size_x * block_size_y >= 32, block_size_x * block_size_y <= 1024"]

    results, env = tune_kernel("matmul", __file__, size, args, tune_params, lang="generic_python", 
        call_function=call_cute, answer=answer,  atol=M * 2 **(-11), restrictions=restrictions, verbose=False)



# This kernel is copied from the CuTe DSL project:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/tensorop_gemm.py
#
# Original example: tensorop_gemm.py
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modifications in file:
# - Tuning paramters made more explicit

## Optimized matmul ==========================================================
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



"""
A dense GEMM (C = A * B) example for the NVIDIA Ampere architecture using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Ampere's tensor cores for matrix multiply-accumulate (MMA) operations
    - Threadblock rasterization to improve data re-use
    - Supports multi-stage pipeline to overlap computation and memory access
    - Implements shared memory buffering for epilogue to increase coalesced global memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using asynchronous copies.
2. Perform matrix multiply-accumulate (MMA) operations.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM).

The Ampere tensor core instruction used operates as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

To run this example:

.. code-block:: bash

    python examples/ampere/tensorop_gemm.py                                  \
      --mnkl 8192,8192,8192,1 --atom_layout_mnk 2,2,1                        \
      --ab_dtype Float16                                                     \
      --c_dtype Float16 --acc_dtype Float32                                  \
      --a_major m --b_major n --c_major n

The above example command computes with M=8192, N=8192, K=8192,
batch_count=1. The atom layout's shape is 2x2x1 and the input, mma
accumulator, and output data type are set as fp16, fp32 and fp16,
respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/ampere/tensorop_gemm.py                              \
      --mnkl 8192,8192,8192,1 --atom_layout_mnk 2,2,1                        \
      --ab_dtype Float16                                                     \
      --c_dtype Float16 --acc_dtype Float32                                  \
      --a_major m --b_major n --c_major n                                    \
      --skip_ref_check --iterations 2

Constraints:
* Supported input and output data types: fp16
* Support accumulator data types: f32
* Default tile shape is set to be 128x128x32
* Atom layout's MNK shape is set so that tile shape can be divided by MMA
  instruction shape
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8
"""


class TensorOpGemm:
    def __init__(self):
        self.ab_dtype = cutlass.Float16
        self.c_dtype = cutlass.Float16
        self.acc_dtype = cutlass.Float32
        self.bM = 128 # extracted from cta_tiler for KT support
        self.bN = 128 # extracted from cta_tiler for KT support
        self.bK = 32  # extracted from cta_tiler for KT support
        self.cta_tiler = (self.bM, self.bN, self.bK)
        self.num_stages = 3
        atom_lay_M = 2 # extracted from atom_layout_mnk for KT support
        atom_lay_N = 2 # extracted from atom_layout_mnk for KT support
        atom_lay_K = 1 # extracted from atom_layout_mnk for KT support
        self.atom_layout_mnk = (atom_lay_M, atom_lay_N, atom_lay_K)  # moved from init parameter to here for KT support
        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32

        self.mma_inst_shape = (16, 8, 16)
        mmaM, mmaN, mmaK = self.mma_inst_shape

        assert self.bM % (atom_lay_M * mmaM) == 0, (
            "bM must be divisible by MMA instruction"
        )
        assert self.bN % (atom_lay_N * mmaN) == 0, (
            "bN must be divisible by MMA instruction"
        )
        assert atom_lay_K == 1, "this example does not support atom layout K > 1"
        assert self.bK % mmaK == 0, "bK must be divisible by MMA instruction"
        assert self.num_stages >= 3, "num_stages must be greater than or equal to 3"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # The grid divides the problems's M, N, and L dimensions by the
        # respective modes of the tile shape (bM, bN, 1). The K dimension is
        # handled within a block via a multistage process.

        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout:
        # ///////////////////////////////////////////////////////////////////////////////

        # Creates a layout with the size required for the provided tile
        # size and num stages (stages are used for K dimension) that is also
        # sectioned into 64x8 or 8x32 layout atoms. The swizzle is set so that
        # the atom for the shared memory -> register copy does not encounter
        # bank conflicts

        # assume the input is 16B align
        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type,
            self.a_major_mode,
            ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type,
            self.b_major_mode,
            ab_copy_bits,
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),
        )

        # Creates a similar layout but without num_stages or layout atoms
        sC_layout = self._make_smem_layout_C(
            mC.element_type,
            self.c_major_mode,
            ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[1]),
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled copy:
        # The majorness of tA/tB/tC follows the majorness of gA/gB/gC,
        # enabling merged accesses to global memory for faster data
        # transfer between global and shared memory.
        # ///////////////////////////////////////////////////////////////////////////////

        # Create a copy atom for a global to shared memory asynchronous copy
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=ab_copy_bits,
        )

        # Create thread layouts for tiled copy from the copy atom where the
        # thread layout simply follows the leading dimension of the tensor
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits
        )

        # Creates a synchronous copy atom and thread layouts for the epilogue
        c_copy_bits = 128
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mC.element_type,
            num_bits_per_copy=c_copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy, mC.element_type, self.c_major_mode, c_copy_bits
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled MMA
        # ///////////////////////////////////////////////////////////////////////////////

        # Creates a mma atom with 16x8x16 shape for MNK
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )

        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            # if atom layout's N-mode is 1, to leverage the largest coalesced
            # shared memory -> register copy, set the tiled mma's N mode to 16
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )

        # Created a tiled mma that tiles the atom according to specified layout.
        # For a 2x2x1 atom layout, the mma atom is duplicated 4 times, twice
        # across M and twice across N
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        # grid_dim: ((m + BLK_M - 1) // BLK_M, (n + BLK_N - 1) // BLK_N, l)
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))

        # Add threadblock rasterization to improve re-use of data
        raster_factor = 1
        grid_dim_n = cute.size(grid_dim[1])
        # Thresholds picked so that it doesn't cause too many no-op CTAs
        if grid_dim_n > 5:
            raster_factor = 8
        elif grid_dim_n > 2:
            raster_factor = 4
        elif grid_dim_n > 1:
            raster_factor = 2
        rasterization_remap_grid_dim = (
            cute.size(grid_dim[0]) * raster_factor,
            (cute.size(grid_dim[1]) + raster_factor - 1) // raster_factor,
            cute.size(grid_dim[2]),
        )

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_copy_C,
            tiled_mma,
            raster_factor,
            epilogue_op,
        ).launch(
            grid=rasterization_remap_grid_dim,
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        rasterization_factor: cutlass.Int32,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        offset_tile_x, offset_tile_y = self.raster_tile(
            bidx, bidy, rasterization_factor
        )
        # Early exit if CTA is out of range
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass
        else:
            tiler_coord = (offset_tile_x, offset_tile_y, None)

            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # gA: (BLK_M, BLK_N, k), gB: (BLK_N, BLK_K, k), gC: (BLK_M, BLK_N)
            # ///////////////////////////////////////////////////////////////////////////////
            gA = cute.local_tile(
                mA[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, None, 1),
            )
            gB = cute.local_tile(
                mB[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(None, 1, 1),
            )
            gC = cute.local_tile(
                mC[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, 1, None),
            )

            # By default, if the tensor k mode does not divide into the tile k
            # size, then last tiles in the k dimension are irregular.
            # Instead, make the first tiles irregular when k is irregular.
            # This allows us to handle the irregular tile first to avoid
            # checking for this condition within the mainloop.

            # residual_k is a negative number indicating the amount needed to
            # shift the pointer by in dimension k
            residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(
                gA, mode=[2]
            )

            # move the pointer of gA/gB in the `-k` direction
            gA = cute.domain_offset((0, residual_k, 0), gA)
            gB = cute.domain_offset((0, residual_k, 0), gB)
            # input is 16B aligned
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

            # Construct identity layout for sA and sB (mirrors global tensors,
            # used for predication only)
            mcA = cute.make_identity_tensor(mA.layout.shape)
            mcB = cute.make_identity_tensor(mB.layout.shape)
            cA = cute.local_tile(
                mcA[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, None, 1),
            )
            cB = cute.local_tile(
                mcB[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(None, 1, 1),
            )

            cA = cute.domain_offset((0, residual_k, 0), cA)
            cB = cute.domain_offset((0, residual_k, 0), cB)

            # ///////////////////////////////////////////////////////////////////////////////
            # Create shared memory buffers and get the appropriate fragments for this thread.
            # sA:   (BLK_M, BLK_K, PIPE)       , sB:   (BLK_N, BLK_K, PIPE)
            # tAgA: (CPY, CPY_M, CPY_K, k)     , tBgB: (CPY, CPY_N, CPY_K, k)
            # tAsA: (CPY, CPY_M, CPY_K, PIPE)  , tBsB: (CPY, CPY_N, CPY_K, PIPE)
            # ///////////////////////////////////////////////////////////////////////////////
            @cute.struct
            class SharedStorageAB:
                a: cute.struct.Align[
                    cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)],
                    16,
                ]
                b: cute.struct.Align[
                    cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)],
                    16,
                ]

            @cute.struct
            class SharedStorageC:
                c: cute.struct.Align[
                    cute.struct.MemRange[mC.element_type, cute.cosize(sC_layout)],
                    16,
                ]

            # Shared memory buffer
            smem = cutlass.utils.SmemAllocator()
            # Shared memory allocated for operations with A, B will be
            # overwritten for operations on C. This is to improve performance
            # by reducing the size of shared memory requested by each block
            storage = smem.allocate(
                max(SharedStorageAB.size_in_bytes(), SharedStorageC.size_in_bytes()),
                byte_alignment=16,
            )
            sA = SharedStorageAB(storage).a.get_tensor(sA_layout)
            sB = SharedStorageAB(storage).b.get_tensor(sB_layout)
            sC = SharedStorageC(storage).c.get_tensor(sC_layout)

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            thr_copy_C = tiled_copy_C.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)
            tCsC_epilogue = thr_copy_C.partition_S(sC)
            tCgC_epilogue = thr_copy_C.partition_D(gC)

            # Repeat the partitioning with identity layouts
            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)

            # ///////////////////////////////////////////////////////////////////////////////
            # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
            # of tile_shape
            # ///////////////////////////////////////////////////////////////////////////////

            # For predication over the tensors A (M/K), B (N/K), and (in the
            # epilogue) C (M/N), we will compute it in a fashion similar to an
            # outer product. The predication along one of the dimensions is
            # evaluated and stored in a predication tensor. Then, the
            # predication for the remaining dimension is handled later via an
            # if/else branch at the copy.
            # For A and B, predication booleans along M/N are stored in a
            # predication tensor and along K is handled via a if/else branch.

            # Allocate predicate tensors for M and N. Predication is checked
            # at the granularity of a copy atom, so the predicate tensor does not
            # need separate booleans for individual elements within a copy
            # atom (for example, the elements of tAgA.shape[0][0].)
            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tAgA.shape[0][1],
                        cute.size(tAgA, mode=[1]),
                        cute.size(tAgA, mode=[2]),
                    ),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tBsB.shape[0][1],
                        cute.size(tBsB, mode=[1]),
                        cute.size(tBsB, mode=[2]),
                    ),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            # Set predicates for M/N bounds
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(
                        tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                    )
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cute.elem_less(
                        tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                    )

            # ///////////////////////////////////////////////////////////////////////////////
            # Prefetch Prologue
            # ///////////////////////////////////////////////////////////////////////////////
            # Clear the smem tiles to account for predicated off loads
            tAsA.fill(0)
            tBsB.fill(0)
            cute.arch.sync_threads()
            # Start async loads for the first k-tile. Here we take care of the k residue
            # via if/else check along the k dimension. Because we shifted the identity tensor
            # by the residue_k and because the identity tensor is a coord tensor, the
            # values of any identity tensor element that is poison is less than -1
            num_smem_stages = cute.size(tAsA, mode=[3])
            k_tile_count = cute.size(tAgA, mode=[3])
            k_tile_index = cutlass.Int32(0)

            for k in range(tApA.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, k, k_tile_index],
                        tAsA[None, None, k, 0],
                        pred=tApA[None, None, k],
                    )
            for k in range(tBpB.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, k, k_tile_index],
                        tBsB[None, None, k, 0],
                        pred=tBpB[None, None, k],
                    )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

            # Start async loads for rest of the k-tiles
            for k_tile in range(1, num_smem_stages - 1):
                if k_tile == k_tile_count:
                    tApA.fill(0)
                    tBpB.fill(0)
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, k_tile_index],
                    tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, k_tile_index],
                    tBsB[None, None, None, k_tile],
                    pred=tBpB,
                )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            # ///////////////////////////////////////////////////////////////////////////////
            # Tile MMA compute thread partitions and allocate accumulators
            # ///////////////////////////////////////////////////////////////////////////////
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCsC = thr_mma.partition_C(sC)
            tCgC = thr_mma.partition_C(gC)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            # Clear the accumulator
            tCrC.fill(0.0)

            # ///////////////////////////////////////////////////////////////////////////////
            # Copy Atom A/B retiling
            # ///////////////////////////////////////////////////////////////////////////////

            # Create the copy atoms for the copy from shared memory to register
            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mA.element_type,
            )
            atom_copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mB.element_type,
            )

            # Creates the tiled copy so that it matches the thread-value layout
            # expected by the tiled mma
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

            thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            # Current pipe index in smem to read from / write to
            smem_pipe_read = 0
            smem_pipe_write = num_smem_stages - 1

            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
            tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]

            # ///////////////////////////////////////////////////////////////////////////////
            # PREFETCH register pipeline
            # ///////////////////////////////////////////////////////////////////////////////
            num_k_block = cute.size(tCrA, mode=[2])
            if num_k_block > 1:
                # Wait until our first prefetched tile is loaded in
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                # Prefetch the first k-block rmem from the first k-tile
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

            # ///////////////////////////////////////////////////////////////////////////////
            # Mainloop
            # 1. Shared memory pipeline (gmem -> smem):
            #    The default smem pipeline depth is 3, meaning that for shared
            # memory buffers, we allocate three times the size described by the
            # CTA tiler. We prefetch 2 of these buffers before entering the main
            # loop. Considering only the transfer from global memory to shared
            # memory, the general structure of the mainloop is:
            #   (1) copy k-tile from gmem to smem;
            #   (2) perform gemm computation on k-tile;
            #   (3) wait for the next copy to finish.
            #    The `cute.arch.cp_async_wait_group(num_smem_stages - 2)` command
            # waits for the number of unfinished 'copy' to be <= 1. The advantage
            # of this approach is that it allows for simultaneous production
            # (i.e., step (1)) and consumption (i.e., step (2)) of smem.
            #    A common misconception is to prefetch N buffers and rewrite
            # the pipeline logic to wait on N-1 pending copies. The disadvantage
            # of this approach is that it requires fully consuming a buffer in
            # order to open an empty buffer for the next copy.
            # 2. Register pipeline (smem -> register):
            #    Similarly, the register pipeline produces i+1, consumes i, and
            # produces i+2... Notably, i and i+1 do not use the same register,
            # eliminating dependencies on the same register for better parallelism.
            # 3. Combining the smem and register pipelines results in the mainloop.
            # ///////////////////////////////////////////////////////////////////////////////
            for k_tile in range(k_tile_count):
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    # Load A, B from shared memory to registers for k_block + 1
                    k_block_next = (k_block + 1) % num_k_block  # static
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_p[None, None, k_block_next],
                        tCrA_copy_view[None, None, k_block_next],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_p[None, None, k_block_next],
                        tCrB_copy_view[None, None, k_block_next],
                    )

                    # Fetch next A: To better interleave global memory access and compute
                    # instructions, we intentionally use the sequence: copy A, perform GEMM,
                    # then copy B.
                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(
                                tiled_copy_A,
                                tAgA[None, None, None, k_tile_index],
                                tAsA[None, None, None, smem_pipe_write],
                                pred=tApA,
                            )

                    # Thread-level register gemm for k_block
                    cute.gemm(
                        tiled_mma,
                        tCrC,
                        tCrA[None, None, k_block],
                        tCrB[None, None, k_block],
                        tCrC,
                    )

                    # Fetch next B and update smem pipeline read/write
                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(
                                tiled_copy_B,
                                tBgB[None, None, None, k_tile_index],
                                tBsB[None, None, None, smem_pipe_write],
                                pred=tBpB,
                            )
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

            # Sync before epilogue
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ///////////////////////////////////////////////////////////////////////////////
            # Epilogue with fusion
            # ///////////////////////////////////////////////////////////////////////////////
            tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
            tCrD[None] = epilogue_op(tCrC.load()).to(self.c_dtype)

            # Copy results of D back to shared memory
            cute.autovec_copy(tCrD, tCsC) 

            # Create coord tensor for C
            ceilM, ceilN, _ = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
            mcC = cute.make_identity_tensor(
                (
                    cute.size(ceilM) * self.cta_tiler[0],
                    cute.size(ceilN) * self.cta_tiler[1],
                    1,
                )
            )
            cC = cute.local_tile(
                mcC[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, 1, None),
            )
            tCcC = thr_copy_C.partition_S(cC)

            tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
            # Wait for all writes to shared memory to finish before starting copies
            # using the new layouts
            cute.arch.sync_threads()
            cute.autovec_copy(tCsC_epilogue, tCrC_epilogue) 

            # Create predication tensor for m
            tCpC = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tCgC_epilogue.shape[0][1],
                        cute.size(tCgC_epilogue, mode=[1]),
                        cute.size(tCgC_epilogue, mode=[2]),
                    ),
                    stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in range(tCpC.shape[0]):
                for m in range(tCpC.shape[1]):
                    tCpC[rest_v, m, 0] = cute.elem_less(
                        tCcC[(0, rest_v), m, 0][0], mC.shape[0]
                    )

            # Copy to global memory using better vectorization
            for rest_v in range(tCpC.shape[0]):
                for n in range(tCpC.shape[2]):
                    if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                        cute.copy(
                            tiled_copy_C,
                            tCrC_epilogue[None, None, n],
                            tCgC_epilogue[None, None, n],
                            pred=tCpC[None, None, n],
                        )
        return

    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )
        major_mode_size = 64 if major_mode_size >= 64 else major_mode_size

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            layout_atom_outer,
        )
        layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))
        return layout

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 4),
            0,
            layout_atom_outer,
        )

        # Due to the thread layout of the mma, remove swizzle in C to
        # prevent shared memory fragments owned by an single thread from
        # holding swizzles
        if major_mode == utils.LayoutEnum.COL_MAJOR:
            layout_atom = cute.make_composed_layout(
                cute.make_swizzle(0, 3, 4), 0, layout_atom_outer
            )
        layout = cute.tile_to_shape(
            layout_atom,
            smem_tiler,
            (0, 1),
        )
        return layout

    def _make_gmem_tiled_copy_AB(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bK) // copy_elems
        # thread layout for copy
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        # Value layout for copy
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def _make_gmem_tiled_copy_C(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bN) // copy_elems
        # thread layout for copy
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def raster_tile(self, i, j, f):
        new_i = i // f
        new_j = (i % f) + (j * f)
        return (new_i, new_j)


# Need a custum call function for the optimized kernel because of the tensor layout
def call_cute_custom(kernel_function, args, kwargs, grid, threads, params):
    # Initialize cache if it does not exist
    if not hasattr(call_cute_custom, "custom_cache"):
        call_cute_custom.custom_cache = {}  

    # Convert Torch tensors to CuTe tensors with correct layout
    cute_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            cute_tensor = (
                from_dlpack(arg, assumed_align=16)
                .mark_layout_dynamic(leading_dim=1) 
                .mark_compact_shape_dynamic(
                    mode=1,
                    stride_order=(2, 0, 1),   
                    divisibility=8,        
                )
            )
            cute_args.append(cute_tensor)
        else:
            cute_args.append(arg)

    # Form cache key from tuning parameters
    param_keys = sorted(params.keys())
    cache_str = type(kernel_function).__name__
    for k in param_keys:
        cache_str += "_" + str(params[k]) 
    
    # Check if kernel exists in cache. Otherwise, compile and save
    if cache_str in call_cute_custom.custom_cache:
        compiled_kernel = call_cute_custom.custom_cache[cache_str]
    else: 
        # Wrap in try-except because CuTe gives TypeError for certain block sizes. This 
        # stops the tuning process and we do not want this
        try: 
            compiled_kernel = cute.compile(kernel_function, *cute_args)
            call_cute_custom.custom_cache[cache_str] = compiled_kernel
        except:
            raise RuntimeError("Invalid configuration")
            
        

    compiled_kernel(*cute_args)



def run_optimized(M, N, K, L=1):
    a = torch.randn((L, M, K), device="cuda", dtype=torch.float16).permute(1, 2, 0)
    b = torch.randn((L, N, K), device="cuda", dtype=torch.float16).permute(1, 2, 0) # b is transposed
    c = torch.empty((L, M, N), device="cuda", dtype=torch.float16).permute(1, 2, 0)

    c_ref = torch.matmul(
        a.permute(2, 0, 1),   
        b.permute(2, 1, 0)    
    ).permute(1, 2, 0)  

    def create_cute_tensor(torch_tensor): 
        cute_tensor = (
            from_dlpack(torch_tensor, assumed_align=16)
            .mark_layout_dynamic(leading_dim=1) 
            .mark_compact_shape_dynamic(
                mode=1,
                stride_order=(2, 0, 1),   
                divisibility=8,        
            )
        )
        return cute_tensor

    a_, b_, c_ = create_cute_tensor(a), create_cute_tensor(b), create_cute_tensor(c)

    gemm = TensorOpGemm()
    compiled_kernel = cute.compile(gemm, a_, b_, c_)
    compiled_kernel(a_, b_, c_)

    assert torch.allclose(c_ref, c, atol= M * 2**(-11), rtol=1e-2)
    print("Sucess")


def tune_optimized(M, N, K, L=1):   
    a = torch.randn((L, M, K), device="cuda", dtype=torch.float16).permute(1, 2, 0)
    b = torch.randn((L, N, K), device="cuda", dtype=torch.float16).permute(1, 2, 0) # b is transposed
    c = torch.empty((L, M, N), device="cuda", dtype=torch.float16).permute(1, 2, 0)

    c_ref = torch.matmul(
        a.permute(2, 0, 1),   
        b.permute(2, 1, 0)    
    ).permute(1, 2, 0)        
    

    args = [a, b, c]
    tune_params = {
        "bM":[16, 32, 64, 128, 256], 
        "bN":[32, 64, 128, 256],
        "bK": [16, 32, 64, 128], # must be divisable by 16
        "num_stages": [3, 4, 5], # restricted to >= 3 by the kernel
        "atom_lay_M": [1, 2, 4],
        "atom_lay_N": [1, 2], 
    }

    restrictions = [
        "bM % (atom_lay_M * 16) == 0", "bN % (atom_lay_N * 8) == 0", # layout
        "atom_lay_M * atom_lay_N * 32 <= 1024", # number of threads
        "num_stages * (bK * (bM + bN)) * 2 <= 49152", # SMEM
        "bM >= atom_lay_M * 32", # ensure each atom sees at least 2 MMA tiles per dimension
        "bN >= atom_lay_N * 16", # ensure each atom sees at least 2 MMA tiles per dimension
    ]



    results, env = tune_kernel("TensorOpGemm", __file__, M * N, args, tune_params, verbose=True, restrictions=restrictions, #strategy="bayes_opt",
                               lang="generic_python", call_function=call_cute_custom, answer=[None, None, c_ref.cpu()], atol=M * 2**(-10))



if __name__ == "__main__":
    m, n, k = 4096, 4096, 4096
    run_naive_matmul(m, n, k)
    tune_naive_matmul(m, n, k)

    mnkl = (4096, 4096, 4096, 1)
    tune_optimized(m, n, k)
    run_optimized(m, n, k)
