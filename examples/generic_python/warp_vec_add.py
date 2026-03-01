import warp as wp
import numpy as np
from kernel_tuner import tune_kernel, run_kernel
import torch

wp.init()



@wp.func
def add_op(x: float, y: float):
    return x + y


#@wp.kernel
def vec_add(a: wp.array(dtype=float),
            b: wp.array(dtype=float),
            c: wp.array(dtype=float),
            n: int):

    work_per_thread = 8
    tid = wp.tid()
    base = tid * work_per_thread

    for i in range(work_per_thread):
        idx = base + i
        if idx < n:
            c[idx] = add_op(a[idx], b[idx])


# TODO do we allways want the call function to have the same parameters
# or do we only require some of them?
def call_warp(kernel_function, args, kwargs, grid, threads, params):
    warp_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            warp_args.append(wp.from_torch(arg))
        else:
            warp_args.append(arg)
    dim = args[3]  
    wp.launch(kernel=kernel_function, dim=dim, inputs=warp_args)
    

# NOTE default verify function only works for numpy/cupy ndarray, torch Tensor or numpy scalar
# That is why we need a costum verify function for warp.
def verify(answer, result_host, atol):
    correct = True
    for i, ans in enumerate(answer):
        if ans is None:
            continue
        print("res: ", type(result_host[i]))
        print("expect: ", type(ans))
        res = result_host[i].numpy()
        if not np.allclose(ans, res, atol=atol):
            correct = False 

    return correct



def tune():
    n = 1024

    # Create host arrays
    a_torch = torch.arange(n, dtype=torch.float32, device="cuda")
    b_torch = torch.arange(n, 0, -1, dtype=torch.float32, device="cuda")
    c_torch = torch.zeros(n, dtype=torch.float32, device="cuda")
    c_expect = a_torch + b_torch


    tune_params = dict()
    tune_params["work_per_thread"] = [2**i for i in range(10)]
    args = [a_torch, b_torch, c_torch, n]


    '''
    results = run_kernel(
        kernel_name="vec_add",
        kernel_source=vec_add,
        problem_size=n,
        arguments=args,
        params={"work_per_thread": 16},
        lang="generic_python",
        call_function=call_warp,
        decorator="@wp.kernel"
    )
    '''

    results, env = tune_kernel(
        kernel_name="vec_add",
        kernel_source=vec_add,
        problem_size=n,
        arguments=args,
        tune_params=tune_params,
        lang="generic_python",
        answer=[None, None, c_expect.cpu(), None],
        call_function=call_warp,
        decorator="@wp.kernel"
    )


tune()