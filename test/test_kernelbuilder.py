import numpy as np
from .context import skip_if_no_cuda

import pytest
from kernel_tuner import kernelbuilder


@skip_if_no_cuda
def test_PythonKernel():

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

    params = {"block_size_x": 256}

    kernel_function = kernelbuilder.PythonKernel("vector_add", kernel_string, n, [c, a, b, n], params)

    reference = kernel_function(c, a, b, n)


    assert np.allclose(reference[0], a+b)

