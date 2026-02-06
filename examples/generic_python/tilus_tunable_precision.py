import tilus
import hidet
from hidet import float32, float16, int32
#from tilus import float32, int32, float16
from tilus.utils import cdiv, benchmark_func
import torch 
from kernel_tuner import tune_kernel, run_kernel
from kernel_tuner.accuracy import Tunable, AccuracyObserver
import numpy as np

INPUT_TYPE = float32
OUTPUT_TYPE = float32

class VecAddV(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_size_x = 32  # number of threads per block

    def __call__(
        self,
        n_size: int32,        # size of the vectors
        a_ptr: ~INPUT_TYPE,      # input vector A
        b_ptr: ~INPUT_TYPE,      # input vector B
        c_ptr: ~OUTPUT_TYPE       # output vector C
    ):  

        # compute the number of blocks needed
        self.attrs.blocks = [cdiv(n_size, self.block_size_x)]
        self.attrs.warps = 1  # number of warps per block

        # calculate the offset for this block
        offset: int32 = self.block_size_x * self.blockIdx.x

        # create global views for input/output vectors
        ga = self.global_view(a_ptr, dtype=INPUT_TYPE, shape=[n_size])
        gb = self.global_view(b_ptr, dtype=INPUT_TYPE, shape=[n_size])
        gc = self.global_view(c_ptr, dtype=OUTPUT_TYPE, shape=[n_size])

        a = self.load_global(ga, offsets=[offset], shape=[self.block_size_x])
        b = self.load_global(gb, offsets=[offset], shape=[self.block_size_x]) 
        c = a + b
        self.store_global(gc, c, offsets=[offset])


def call_tilus(kernel_function, args, kwargs, grid, threads, params):
    kernel_function(*args, **kwargs) 


def verify(answer, result_host, atol):
    correct = True
    for i, ans in enumerate(answer):
        if ans is None:
            continue
        res = result_host[i].cpu()
        if not torch.allcose(ans, res, atol=atol):
            correct = False 

    return correct



def main():

    size = 1024000

    a_32 = torch.randn(size, dtype=torch.float32)
    b_32 = torch.randn(size, dtype=torch.float32)
    c_32 = torch.zeros_like(b_32)
    c_expect = a_32 + b_32

    a_16 = a_32.to(torch.float16)
    b_16 = b_32.to(torch.float16)
    c_16 = c_32.to(torch.float16)
    

    tune_params = dict()
    tune_params["block_size_x"] = [32, 64, 128, 256, 512, 1024]
    tune_params["INPUT_TYPE"] = [tilus.float16, tilus.float32]
    tune_params["OUTPUT_TYPE"] = [tilus.float16, tilus.float32]

    args = [
        size,
        Tunable("INPUT_TYPE", {
            tilus.float32: a_32,
            tilus.float16: a_16,
        }),
        Tunable("INPUT_TYPE", {
            tilus.float32: b_32,
            tilus.float16: b_16,
        }),
        Tunable("OUTPUT_TYPE", {
            tilus.float32: c_32,
            tilus.float16: c_16,
        }),
    ]

    print(tune_params)

    observers = [AccuracyObserver("RMSE")]


    
    results, env = tune_kernel(
        kernel_name="VecAddV", 
        kernel_source=VecAddV,
        problem_size=size,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, None, c_expect.cpu()],
        observers=observers,
        call_function=call_tilus,
        verify=verify,
        verbose=True,
    )

    

if __name__ == "__main__":
    main()
