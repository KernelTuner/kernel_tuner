import tilus
from tilus import float32, int32
from tilus.utils import cdiv, benchmark_func
import torch 
from kernel_tuner import tune_kernel, run_kernel
from pathlib import Path

FULL_PATH = Path(__file__).resolve()

class VecAddV(tilus.Script):
    def __init__(self, block_size_x=None, num_warps=None):
        super().__init__()
        self.block_size_x = block_size_x  # number of threads per block

    def __call__(
        self,
        n_size: int32,        # size of the vectors
        a_ptr: ~float32,      # input vector A
        b_ptr: ~float32,      # input vector B
        c_ptr: ~float32       # output vector C
    ):  

        # compute the number of blocks needed
        self.attrs.blocks = [cdiv(n_size, self.block_size_x)]
        self.attrs.warps = 4  # number of warps per block

        # calculate the offset for this block
        offset: int32 = self.block_size_x * self.blockIdx.x

        # create global views for input/output vectors
        ga = self.global_view(a_ptr, dtype=float32, shape=[n_size])
        gb = self.global_view(b_ptr, dtype=float32, shape=[n_size])
        gc = self.global_view(c_ptr, dtype=float32, shape=[n_size])

        a = self.load_global(ga, offsets=[offset], shape=[self.block_size_x])
        b = self.load_global(gb, offsets=[offset], shape=[self.block_size_x]) 
        c = a + b
        self.store_global(gc, c, offsets=[offset])


def call_tilus(kernel_function, args, kwargs):
    kernel_function(*args, **kwargs) 


def tune(size):
    a = torch.randn(size, dtype=torch.float32).cuda()
    b = torch.randn(size, dtype=torch.float32).cuda()
    c = torch.empty(size, dtype=torch.float32).cuda()
    c_expect = a + b
    

    args = [size, a, b, c]
    tune_params = dict()
    tune_params["block_size_x"] = [32, 64, 128, 256, 512, 1024]
    tune_params["num_warps"] = [4, 8]


    results, env = tune_kernel(
        kernel_name="VecAddV", 
        kernel_source=FULL_PATH,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, None, c_expect.cpu()],
        call_function=call_tilus,
        verbose=True,
    )


def run(size):
    a = torch.randn(size, dtype=torch.float32).cuda()
    b = torch.randn(size, dtype=torch.float32).cuda()
    c = torch.empty(size, dtype=torch.float32).cuda()
    c_expect = a + b
    

    args = [size, a, b, c]

    results = run_kernel(
        kernel_name="VecAddV", 
        kernel_source=FULL_PATH,
        problem_size=size,
        arguments=args,
        params={"block_size_x": 32},
        lang="generic_python",
        call_function=call_tilus,
    )

    c_expect = c_expect.cpu()

    assert torch.allclose(results[-1], c_expect)


def tune_with_builtin(size):
    TunedVecAdd = tilus.autotune("block_size_x", [32, 64, 128, 256, 512, 1024])(VecAddV)
    vecadd = TunedVecAdd()

    a = torch.randn(size, dtype=torch.float32).cuda()
    b = torch.randn(size, dtype=torch.float32).cuda()
    c = torch.empty(size, dtype=torch.float32).cuda()
    c_expect = a + b

    vecadd(size, a, b, c) # This is where the actual tuning takes place
    torch.cuda.synchronize()

    torch.testing.assert_close(c_expect, c)


if __name__ == "__main__":
    size = 10000000
    #tune(size)
    #run(size)
    tune_with_builtin(size)
