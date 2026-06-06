import torch

import tilus
from tilus import float16, float32, int32
from tilus.utils import cdiv

from kernel_tuner import tune_kernel
from examples.generic_python.call_functions import call_tilus


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
        m_size: int32, n_size: int, k_size: int, # Matrix dimensions 
        a_ptr: ~float16, b_ptr: ~float16, c_ptr: ~float16, # Matrix pointers 
    ):
        self.attrs.blocks = [
            cdiv(m_size, self.block_m),  # the x dimension size of the grid
            cdiv(n_size, self.block_n),  # the y dimension size of the grid
        ]
        num_warps = 1 # added for tuning
        self.attrs.warps = num_warps  # the number of warps per thread block, must be a compile-time known integer

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
# https://nvidia.github.io/tilus/stable/tutorials/matmul-ampere/matmul/matmul_v4.html
#
# Original example: matmul_v4.py
# Copyright (c) the Tilus authors
#
# Modifications in file:
# - Removed auto-tuning decorators
# - Added default values (None) for the __init__ parameters
class MatmulOpt(tilus.Script):
    def __init__(self, num_warps=None, block_m=None, block_n=None, block_k=None, num_stages=None):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages

    def __call__(
        self,
        m_size: int32, n_size: int, k_size: int,
        a_ptr: ~float16, b_ptr: ~float16, c_ptr: ~float16,
    ):
        self.attrs.blocks = [cdiv(m_size, self.block_m), cdiv(n_size, self.block_n)]
        self.attrs.warps = self.num_warps

        block_m, block_n, block_k = self.block_m, self.block_n, self.block_k
        offset_m: int32 = block_m * self.blockIdx.x
        offset_n: int32 = block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_m, block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.num_stages, block_k, block_n])
        acc = self.register_tensor(dtype=float32, shape=[block_m, block_n], init=0.0)

        for stage in range(self.num_stages - 1):
            offset_k = stage * self.block_k
            self.copy_async(src=ga, dst=sa[stage], offsets=[offset_m, offset_k])
            self.copy_async(src=gb, dst=sb[stage], offsets=[offset_k, offset_n])
            self.copy_async_commit_group()

        self.copy_async_wait_group(n=self.num_stages - 2)
        self.sync()

        current_stage: int32 = 0
        preload_stage: int32 = self.num_stages - 1
        for offset_k in self.range(0, k_size, block_k, unroll=self.num_stages):
            # computation for current tile
            a = self.load_shared(sa[current_stage])
            b = self.load_shared(sb[current_stage])
            self.dot(a, b, acc, out=acc)

            # preload the next tile of A and B into shared memory
            preload_offset_k = offset_k + (self.num_stages - 1) * block_k
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

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])



def run_basic(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    C_ref = A @ B

    matmul = MatmulBasic()
    torch.cuda.synchronize()
    matmul(M, N, K, A, B, C)
    torch.cuda.synchronize()

    torch.testing.assert_close(C_ref, C, atol=M * 2**(-11), rtol=1e-2)
    print("Succes")


def run_optimized(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    C_ref = A @ B

    matmul = MatmulOpt(num_warps=4, block_m=64, block_n=64, block_k=16, num_stages=3)
    torch.cuda.synchronize()

    matmul(M, N, K, A, B, C)
    torch.cuda.synchronize()

    torch.testing.assert_close(C_ref, C, atol=M * 2**(-11), rtol=1e-2)
    print("Succes")


def tune_basic(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    C_ref = A @ B

    size = (M, N)

    args = [M, N, K, A, B, C]

    tune_params = dict()
    tune_params["block_m"] = [2**i for i in range(5, 9)]
    tune_params["block_n"] = [2**i for i in range(5, 9)]
    tune_params["block_k"] = [2**i for i in range(4, 8)]
    tune_params["num_warps"] = [2, 4, 8, 16]


    results, env = tune_kernel(
        kernel_name="MatmulBasic",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, None, None, None, C_ref.cpu()],
        atol=M * 2**(-11),
        strategy = "bayes_opt",
        strategy_options = {"max_fevals": 200},
        call_function=call_tilus,
    )


def tune_opt(M, N, K):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    C_ref = A @ B

    size = (M, N)

    args = [M, N, K, A, B, C]

    tune_params = dict()
    tune_params["block_m"] = [2**i for i in range(5, 9)]
    tune_params["block_n"] = [2**i for i in range(5, 9)]
    tune_params["block_k"] = [2**i for i in range(4, 8)]
    tune_params["num_warps"] = [2, 4, 8, 16]
    tune_params["num_stages"] = [2, 3, 4, 5]

    # Shared memory restriction
    restrictions = ["2 * num_stages * block_k * (block_m + block_n) <= 49152"]

    results, env = tune_kernel(
        kernel_name="MatmulOpt",
        kernel_source=__file__,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, None, None, None, C_ref.cpu()],
        atol=M * 2**(-11),
        restrictions=restrictions,
        strategy = "bayes_opt",
        strategy_options = {"max_fevals": 200},
        call_function=call_tilus,
    )





if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    #M, N, K = 8192, 8192, 8192
    #run_basic(M, N, K)

    #run_optmized(M, N, K)
  
    tune_basic(M, N, K)

    #tune_opt(M, N, K)