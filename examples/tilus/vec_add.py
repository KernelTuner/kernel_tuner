import tilus
from tilus import float32, int32
from tilus.utils import cdiv, benchmark_func
import torch 


class VecAddV(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_size = 256  # number of threads per block

    def __call__(
        self,
        n_size: int32,        # size of the vectors
        a_ptr: ~float32,      # input vector A
        b_ptr: ~float32,      # input vector B
        c_ptr: ~float32       # output vector C
    ):
        # compute the number of blocks needed
        self.attrs.blocks = [cdiv(n_size, self.block_size)]
        self.attrs.warps = 1  # number of warps per block

        # calculate the offset for this block
        offset: int32 = self.block_size * self.blockIdx.x

        # create global views for input/output vectors
        ga = self.global_view(a_ptr, dtype=float32, shape=[n_size])
        gb = self.global_view(b_ptr, dtype=float32, shape=[n_size])
        gc = self.global_view(c_ptr, dtype=float32, shape=[n_size])

        # load a block of A and B into registers
        a = self.load_global(ga, offsets=[offset], shape=[self.block_size])
        b = self.load_global(gb, offsets=[offset], shape=[self.block_size])

        # perform element-wise addition
        c = a + b

        # store the result back to global memory
        self.store_global(gc, c, offsets=[offset])


def main():
    N = 1 << 20  # vector size
    vecadd = VecAddV()

    a = torch.rand(N, dtype=torch.float32).cuda()
    b = torch.rand(N, dtype=torch.float32).cuda()
    c_actual = torch.empty_like(a)
    c_expect = a + b

    torch.cuda.synchronize()
    vecadd(N, a, b, c_actual)
    torch.cuda.synchronize()

    # correctness check
    torch.testing.assert_close(c_expect, c_actual, atol=1e-6, rtol=1e-6)

    # benchmark
    for name, func in [
        ("torch", lambda: a + b),
        ("tilus", lambda: vecadd(N, a, b, c_actual)),
    ]:
        latency = benchmark_func(func, warmup=5, repeat=20)
        print(f"{name} latency: {latency:.3f} ms")

if __name__ == "__main__":
    main()
