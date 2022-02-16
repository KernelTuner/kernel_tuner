import numpy as np
from .context import skip_if_no_cuda

import pytest
from kernel_tuner import kernelbuilder
from kernel_tuner import util
from kernel_tuner import integration


@pytest.fixture()
def test_kernel():
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """
    n = np.int32(100)
    a, b, c = np.random.random((3, n)).astype(np.float32)
    params = {"block_size_x": 384}
    return "vector_add", kernel_string, n, [c, a, b, n], params


@skip_if_no_cuda
def test_PythonKernel(test_kernel):
    kernel_name, kernel_string, n, args, params = test_kernel
    kernel_function = kernelbuilder.PythonKernel(*test_kernel)
    reference = kernel_function(*args)
    assert np.allclose(reference[0], args[1]+args[2])


@skip_if_no_cuda
def test_PythonKernel_tuned(test_kernel):
    kernel_name, kernel_string, n, args, params = test_kernel
    c, a, b, n = args
    test_results_file = "test_results_file.json"
    results = params.copy()
    results['time'] = 1.0
    env = {"device_name": "bogus GPU"}
    try:
        #create a fake results file
        integration.store_results(test_results_file, kernel_name, kernel_string, params, n, [results], env)

        #create a kernel using the results
        kernel_function = kernelbuilder.PythonKernel(kernel_name, kernel_string, n, args, results_file=test_results_file)

        #test if params were retrieved correctly
        assert kernel_function.params["block_size_x"] == 384

        #see if it functions properly
        reference = kernel_function(c, a, b, n)
        assert np.allclose(reference[0], a+b)

    finally:
        util.delete_temp_file(test_results_file)
