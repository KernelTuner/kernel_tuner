import warp as wp
import numpy as np
from kernel_tuner import tune_kernel, run_kernel



@wp.func
def add_op(x: float, y: float):
    return x + y


#@wp.kernel
def vec_add(a: wp.array(dtype=float),
            b: wp.array(dtype=float),
            c: wp.array(dtype=float),
            n: int,
            work_per_thread: int):

    tid = wp.tid()
    base = tid * work_per_thread

    for i in range(work_per_thread):
        idx = base + i
        if idx < n:
            c[idx] = add_op(a[idx], b[idx])


# TODO do we allways want the call function to have the same parameters
# or do we only require some of them?
def call_warp(kernel_function, args, kwargs, grid, threads, params):
    final_args = list(args)
    final_args.extend(kwargs.values())
    dim = args[3]  
    wp.launch(kernel=kernel_function, dim=dim, inputs=final_args)
    

# NOTE default verify function only works for numpy/cupy ndarray, torch Tensor or numpy scalar
# That is why we need a costum verify function for warp.
def verify(answer, result_host, atol):
    correct = True
    for i, ans in enumerate(answer):
        if ans is None:
            continue
        res = result_host[i].numpy()
        if not np.allclose(ans, res, atol=atol):
            correct = False 

    return correct



def tune():
    n = 1024

    # Create host arrays
    a_np = np.arange(n, dtype=np.float32)
    b_np = np.arange(n, 0, -1, dtype=np.float32)
    c_np = np.zeros(n, dtype=np.float32)
    c_expect = a_np + b_np

    # Create Warp arrays on GPU
    a = wp.array(a_np, dtype=float)
    b = wp.array(b_np, dtype=float)
    c = wp.array(c_np, dtype=float)

    tune_params = dict()
    tune_params["work_per_thread"] = [2**i for i in range(10)]
    args = [a, b, c, n]


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
        answer=[None, None, c_expect, None],
        verify=verify,
        call_function=call_warp,
        decorator="@wp.kernel"
    )


tune()