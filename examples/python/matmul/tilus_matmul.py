import torch

import tilus
from tilus import float16, float32, int32
from tilus.utils import cdiv

from kernel_tuner import tune_kernel


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
        answer=[None, None, None, None, None, C_ref.cpu()],
        atol=M * 2**(-11),
        strategy = "bayes_opt",
        strategy_options = {"max_fevals": 200},
    )


if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    run_basic(M, N, K)
    tune_basic(M, N, K)
