from __future__ import print_function

import numpy as np

import kernel_tuner
from .context import skip_if_no_cuda, skip_if_no_noodles


def test_random_sample():

    kernel_string = "float test_kernel(float *a) { return 1.0f; }"
    a = np.arange(4, dtype=np.float32)

    tune_params = {"block_size_x": range(1, 25)}
    print(tune_params)

    result, _ = kernel_tuner.tune_kernel(
        "test_kernel", kernel_string, (1, 1), [a], tune_params, sample_fraction=0.1)

    print(result)

    # check that number of benchmarked kernels is 10% (rounded up)
    assert len(result) == 3

    # check all returned results make sense
    for v in result:
        assert v['time'] == 1.0


@skip_if_no_noodles
@skip_if_no_cuda
def test_noodles_runner():

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 100
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]
    tune_params = {"block_size_x": [128+64*i for i in range(15)]}

    result, _ = kernel_tuner.tune_kernel(
        "vector_add", kernel_string, size, args, tune_params,
        use_noodles=True, num_threads=4)

    assert len(result) == len(tune_params["block_size_x"])


def get_vector_add_args():
    size = int(1e6)
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b).astype(np.float32)
    n = np.int32(size)
    return c, a, b, n


@skip_if_no_cuda
def test_diff_evo():

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    } """

    args = get_vector_add_args()
    tune_params = {"block_size_x": [128+64*i for i in range(5)]}

    result, _ = kernel_tuner.tune_kernel(
        "vector_add", kernel_string, args[-1], args, tune_params,
        method="diff_evo", verbose=True)

    print(result)
    assert len(result) > 0


@skip_if_no_cuda
def test_sequential_runner_alt_block_size_names():

    kernel_string = """__global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_dim_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    c, a, b, n = get_vector_add_args()
    args = [c, a, b, n]
    tune_params = {"block_dim_x": [128 + 64 * i for i in range(5)],
                   "block_size_y": [1], "block_size_z": [1]}

    ref = (a+b).astype(np.float32)
    answer = [ref, None, None, None]

    block_size_names = ["block_dim_x"]

    result, _ = kernel_tuner.tune_kernel(
        "vector_add", kernel_string, int(n), args,
        tune_params, grid_div_x=["block_dim_x"], answer=answer,
        block_size_names=block_size_names)

    assert len(result) == len(tune_params["block_dim_x"])


@skip_if_no_cuda
def test_sequential_runner_not_matching_answer1():
    kernel_string = """__global__ void vector_add(float *c, float *a, float *b, int n) {
            int i = blockIdx.x * block_size_x + threadIdx.x;
            if (i<n) {
                c[i] = a[i] + b[i];
            }
        } """
    args = get_vector_add_args()
    answer = [args[1] + args[2]]
    tune_params = {"block_size_x": [128 + 64 * i for i in range(5)]}

    try:
        kernel_tuner.tune_kernel(
            "vector_add", kernel_string, args[-1], args, tune_params,
            method="diff_evo", verbose=True, answer=answer)
        print("Expected a TypeError to be raised")
        assert False
    except TypeError as expected_error:
        print(str(expected_error))
        assert "The length of argument list and provided results do not match." == str(expected_error)
    except Exception:
        print("Expected a TypeError to be raised")
        assert False


@skip_if_no_cuda
def test_sequential_runner_not_matching_answer2():
    kernel_string = """__global__ void vector_add(float *c, float *a, float *b, int n) {
            int i = blockIdx.x * block_size_x + threadIdx.x;
            if (i<n) {
                c[i] = a[i] + b[i];
            }
        } """
    args = get_vector_add_args()
    answer = [np.ubyte([12]), None, None, None]
    tune_params = {"block_size_x": [128 + 64 * i for i in range(5)]}

    try:
        kernel_tuner.tune_kernel(
            "vector_add", kernel_string, args[-1], args, tune_params,
            method="diff_evo", verbose=True, answer=answer)

        print("Expected a TypeError to be raised")
        assert False
    except TypeError as expected_error:
        print(str(expected_error))
        assert "Element 0" in str(expected_error)
    except Exception:
        print("Expected a TypeError to be raised")
        assert False
