from tilus import float16, float32, int32
from tilus.utils import cdiv
import tilus
from kernel_tuner import tune_kernel, run_kernel
import math
import torch


class MatmulV0(tilus.Script):
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


def call_tilus(kernel_function, args, kwargs, grid, threads, params):
    kernel_function(*args, **kwargs) 

def main():
    m, n, k = 4096, 4096, 4096

    # create an instance of the kernel we have just defined
    matmul = MatmulV0()

    a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    c_actual = torch.empty(m, n, dtype=torch.float16).cuda()
    c_expect = a @ b
    
    '''
    torch.cuda.synchronize()
    # launch the kernel by passing required arguments
    matmul(m, n, k, a, b, c_actual)
    torch.cuda.synchronize()

    # check correctness
    torch.testing.assert_close(c_expect, c_actual, atol=1e-2, rtol=1e-2)
    '''

    args = [m, n, k, a, b, c_actual]
    tune_params = dict()
    tune_params["block_m"] = [16, 32, 64, 128, 256]
    tune_params["block_n"] = [16, 32, 64, 128, 256]
    tune_params["block_k"] = [16, 32, 64, 128, 256]


    restrictions = [
        "block_m * block_n <= 4096",
        "block_m * block_k <= 2048",
        "block_k * block_n <= 4096",
        "block_k >= 16",
    ]
    
    results, env = tune_kernel(
        kernel_name="MatmulV0",
        kernel_source=MatmulV0,
        problem_size=[m, n],
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, None, None, None, c_expect.cpu()],
        restrictions=restrictions,
        block_size_names=["block_m", "block_n", "block_k"],
        call_function=call_tilus,
        strategy="simulated_annealing"
    )
    



if __name__ == "__main__":
    main()