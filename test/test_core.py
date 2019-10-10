from __future__ import print_function

import numpy as np
import pytest

import kernel_tuner
from kernel_tuner import core
from kernel_tuner.interface import Options

from .context import skip_if_no_cuda, skip_if_no_noodles

def get_vector_add_args():
    size = int(1e6)
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b).astype(np.float32)
    n = np.int32(size)
    return c, a, b, n

@pytest.fixture
def env():
    kernel_string = """__global__ void vector_add(float *c, float *a, float *b, int n) {
            int i = blockIdx.x * block_size_x + threadIdx.x;
            if (i<n) {
                c[i] = a[i] + b[i];
            }
        } """
    args = get_vector_add_args()
    params = {"block_size_x": 128}

    lang="CUDA"
    kernel_source = core.KernelSource(kernel_string, lang)
    verbose = True
    kernel_options = Options(kernel_name="vector_add", kernel_string=kernel_string, problem_size=args[-1], arguments=args, lang=lang,
                          grid_div_x=None, grid_div_y=None, grid_div_z=None, cmem_args=None, texmem_args=None, block_size_names=None)
    device_options = Options(device=0, platform=0, lang=lang, quiet=False, compiler=None, compiler_options=None)
    dev = core.DeviceInterface(kernel_source, iterations=7, **device_options)
    instance = dev.create_kernel_instance(kernel_source, kernel_options, params, verbose)

    return dev, instance


@skip_if_no_cuda
def test_default_verify_function(env):

    #gpu_args = dev.ready_argument_list(args)
    #func = dev.compile_kernel(instance, verbose)

    dev, instance = env
    args = instance.arguments
    verbose = True

    #1st case, correct answer but not enough items in the list
    answer = [args[1] + args[2]]
    try:
        core._default_verify_function(instance, answer, args, 1e-6, verbose)
        #dev.check_kernel_output(func, gpu_args, instance, answer, 1e-6, None, verbose)
        print("Expected a TypeError to be raised")
        assert False
    except TypeError as expected_error:
        print(str(expected_error))
        assert "The length of argument list and provided results do not match." == str(expected_error)
    except Exception:
        print("Expected a TypeError to be raised")
        assert False

    #2nd case, answer is of wrong type
    answer = [np.ubyte([12]), None, None, None]
    try:
        core._default_verify_function(instance, answer, args, 1e-6, verbose)
        #dev.check_kernel_output(func, gpu_args, instance, answer, 1e-6, None, verbose)
        print("Expected a TypeError to be raised")
        assert False
    except TypeError as expected_error:
        print(str(expected_error))
        assert "Element 0" in str(expected_error)
    except Exception:
        print("Expected a TypeError to be raised")
        assert False


    instance.delete_temp_files()
    assert True
