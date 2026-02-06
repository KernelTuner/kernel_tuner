import tilus
from tilus import float32, int32
from tilus.utils import cdiv, benchmark_func
import torch 
from kernel_tuner import tune_kernel, run_kernel



class VecAddV(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_size_x = 32  # number of threads per block

    def __call__(
        self,
        n_size: int32,        # size of the vectors
        a_ptr: ~float32,      # input vector A
        b_ptr: ~float32,      # input vector B
        c_ptr: ~float32       # output vector C
    ):  

        # compute the number of blocks needed
        self.attrs.blocks = [cdiv(n_size, self.block_size_x)]
        self.attrs.warps = 1  # number of warps per block

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


def call_tilus(kernel_function, args, kwargs, grid, threads, params):
    kernel_function(*args, **kwargs) 


def main():

    size = 1024000

    a = torch.randn(size, dtype=torch.float32)
    b = torch.randn(size, dtype=torch.float32)
    c = torch.zeros_like(b)
    c_expect = a + b
    

    args = [size, a, b, c]
    tune_params = dict()
    tune_params["block_size_x"] = [32, 64, 128, 256, 512, 1024]


    
    results, env = tune_kernel(
        kernel_name="VecAddV", 
        kernel_source=VecAddV,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, None, c_expect.cpu()],
        call_function=call_tilus,
        verbose=True,
    )

    

if __name__ == "__main__":
    main()
