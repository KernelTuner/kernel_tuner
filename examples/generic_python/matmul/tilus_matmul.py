import tilus
from tilus import float16, float32, int32
from tilus.utils import cdiv


# This kernel is copied from the Tilus project:
# https://github.com/NVIDIA/tilus/blob/main/examples/matmul/matmul_v0.py
#
# Original example: matmul_v0.py
# Copyright (c) the Tilus authors
class MatmulBasic(tilus.Script):
    def __init__(self):
        super().__init__()
        # we define three hyperparameters: ``block_m``, ``block_n``, and ``block_k`` to determine the tile size on
        # m, n, and k dimensions for each `thread block` of the kernel.
        self.block_m = 64
        self.block_n = 64
        self.block_k = 16

    def __call__(
        self,
        m_size: int32,  # the size of the m dimension of the input matrix A and output matrix C
        n_size: int,  # the size of the n dimension of the input matrix B and output matrix C
        k_size: int,  # the size of the k dimension of the input matrix A and B
        a_ptr: ~float16,  # the pointer to the input matrix A, which is a 2D tensor of shape [m_size, k_size]
        b_ptr: ~float16,  # the pointer to the input matrix B, which is a 2D tensor of shape [k_size, n_size]
        c_ptr: ~float16,  # the pointer to the output matrix C, which is a 2D tensor of shape [m_size, n_size]
    ):
        self.attrs.blocks = [
            cdiv(m_size, self.block_m),  # the x dimension size of the grid
            cdiv(n_size, self.block_n),  # the y dimension size of the grid
        ]
        self.attrs.warps = 1  # the number of warps per thread block, must be a compile-time known integer

        # define two int32 variables to store the offsets of the m and n dimensions for the current thread block.
        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        # create two global tensors `ga` and `gb` to represent the input matrices A and B, respectively.
        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])

        # create a register tensor `acc` to accumulate the results of the matrix multiplication.
        acc = self.register_tensor(
            dtype=float32, shape=[self.block_m, self.block_n], init=0.0
        )

        # iterate over the k dimension in blocks of size `block_k`.
        for k in range(cdiv(k_size, self.block_k)):
            # calculate the offset for the current block in the k dimension
            offset_k = k * self.block_k

            # load a block of matrix A and B into register tensors `a` and `b`.
            a = self.load_global(
                ga, offsets=[offset_m, offset_k], shape=[self.block_m, self.block_k]
            )
            b = self.load_global(
                gb, offsets=[offset_k, offset_n], shape=[self.block_k, self.block_n]
            )

            # perform the dot product: acc = a @ b + acc
            self.dot(a, b, acc, out=acc)

        # after the loop, we cast the accumulated result `acc` to float16 type and store it back to the output matrix C.
        acc_f16 = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, acc_f16, offsets=[offset_m, offset_n])




# This kernel is copied from the Tilus project:
# https://github.com/NVIDIA/tilus/blob/main/examples/matmul/matmul_+v5.py
#
# Original example: matmul_v5.py
# Copyright (c) the Tilus authors
#
# Modifications in file:
# - Removed auto-tuning decorators
class MatmulOpt(tilus.Script):
    def __init__(
        self,
        num_warps=None,
        block_m=None,
        block_n=None,
        block_k=None,
        num_stages=None,
        split_k_factor=None,
    ):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.split_k_factor = split_k_factor

    def __call__(
        self,
        m_size: int32,
        n_size: int,
        k_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        self.attrs.blocks = [
            cdiv(m_size, self.block_m),
            cdiv(n_size, self.block_n),
            self.split_k_factor,
        ]
        self.attrs.warps = self.num_warps

        # the k_size for each thread block
        block_k_size = (
            cdiv(cdiv(k_size, self.split_k_factor), self.block_k) * self.block_k
        )
        start_offset_k = self.blockIdx.z * block_k_size
        end_offset_k = min(start_offset_k + block_k_size, k_size)

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_m, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_k, block_n])
        acc = self.register_tensor(dtype=float32, shape=[block_m, block_n], init=0.0)

        for stage in range(self.num_stages - 1):
            offset_k = start_offset_k + stage * self.block_k
            self.copy_async(src=ga, dst=sa[stage], offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb[stage], offsets=[offset_k, offset_n])
            self.copy_async_commit_group()

        self.copy_async_wait_group(n=self.num_stages - 2)
        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.num_stages - 1
        for offset_k in self.range(
            start_offset_k, end_offset_k, block_k, unroll=self.num_stages
        ):
            # computation for current tile
            a = self.load_shared(sa[current_stage])
            b = self.load_shared(sb[current_stage])
            self.dot(a, b, acc, out=acc)

            # preload the next tile of A and B into shared memory
            preload_offset_k = offset_k + (self.num_stages - 1) * block_k
            if preload_offset_k < end_offset_k:
                self.copy_async(
                    src=ga,
                    dst=sa[preload_stage],
                    offsets=[offset_m, preload_offset_k],
                )
                self.copy_async(
                    src=gb,
                    dst=sb[preload_stage],
                    offsets=[preload_offset_k, offset_n],
                )
            self.copy_async_commit_group()

            # update the stage
            current_stage = (current_stage + 1) % self.num_stages
            preload_stage = (preload_stage + 1) % self.num_stages
            self.copy_async_wait_group(n=self.num_stages - 2)
            self.sync()

        # free the shared memory tensors for A and B
        self.free_shared(sa)
        self.free_shared(sb)

        # cast the accumulator to float16 and change the register tensor's layout
        sc = self.shared_tensor(dtype=float16, shape=[block_m, block_n])
        casted_acc = self.cast(acc, dtype=float16)
        self.store_shared(sc, casted_acc)
        self.sync()
        rc = self.load_shared(sc)
        self.free_shared(sc)

        m_blocks, n_blocks = cdiv(m_size, block_m), cdiv(n_size, block_n)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        if self.split_k_factor == 0:
            self.store_global(gc, rc, offsets=[offset_m, offset_n])
        else:
            semaphores = self.global_tensor(
                dtype=int32, shape=[m_blocks, n_blocks], requires_clean=True
            )
            semaphore: ~int32 = ~semaphores[self.blockIdx.x, self.blockIdx.y]

            # load and accumulate the partial result in global memory
            if self.blockIdx.z > 0:
                self.lock_semaphore(semaphore, value=self.blockIdx.z)
                partial_rc = self.load_global(
                    gc, offsets=[offset_m, offset_n], shape=[block_m, block_n]
                )
                self.add(rc, partial_rc, out=rc)

            # store the result to global memory and release the semaphore
            self.store_global(gc, rc, offsets=[offset_m, offset_n])

            # release the semaphore
            self.sync()  # we need to make sure the previous store_global is finished
            self.release_semaphore(
                semaphore, value=(self.blockIdx.z + 1) % self.split_k_factor
            )