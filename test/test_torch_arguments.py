"""Tests for passing PyTorch tensors and other CUDA Array Interface objects as kernel arguments."""

import numpy as np
import pytest

from kernel_tuner import core, run_kernel, tune_kernel
from kernel_tuner.backends.backend import get_device_array, is_host_array

from .context import skip_backend

try:
    import torch

    torch_cuda_present = torch.cuda.is_available()
except ImportError:
    torch = None
    torch_cuda_present = False

skip_if_no_torch = pytest.mark.skipif(torch is None, reason="Torch not installed")
skip_if_no_torch_cuda = pytest.mark.skipif(not torch_cuda_present, reason="Torch not installed or no CUDA device")

# accumulates into c, so verification only passes if outputs are reset between kernel runs
accumulate_kernel = """
__global__ void vector_add(float *c, float *a, float *b, int n) {
    int i = blockIdx.x * block_size_x + threadIdx.x;
    if (i < n) {
        c[i] += a[i] + b[i];
    }
}
"""


class FakeDeviceArray:
    def __init__(self, shape, typestr, strides=None):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (1234, False),
            "strides": strides,
            "version": 3,
        }


def test_get_device_array():
    assert get_device_array(FakeDeviceArray((10, 4), "<f4")) == (1234, 160)
    assert get_device_array(FakeDeviceArray((10, 4), "<f8", strides=(32, 8))) == (1234, 320)
    assert get_device_array(np.zeros(10)) is None
    with pytest.raises(ValueError):
        get_device_array(FakeDeviceArray((10, 4), "<f4", strides=(4, 40)))


def test_is_host_array():
    assert is_host_array(np.zeros(10))
    assert not is_host_array(np.float32(1.0))
    assert not is_host_array(np.int32(1))
    assert not is_host_array(1)
    assert not is_host_array(FakeDeviceArray((10,), "<f4"))


@skip_if_no_torch
def test_default_verify_function_torch():
    answer = [torch.tensor([1.0, 2.0, float("nan")]), None]
    result_host = [torch.tensor([1.0, 2.0, float("nan")]), None]
    instance = core.KernelInstance("name", None, "kernel_string", [], None, None, dict(), [answer[0].clone(), None])
    with pytest.warns(UserWarning, match="NaN"):
        assert core._default_verify_function(instance, answer, result_host, 1e-6, False)


@skip_if_no_torch_cuda
@pytest.mark.parametrize("lang", ["NVCUDA", "PYCUDA"])
@pytest.mark.parametrize("args_on_gpu", [True, False])
@pytest.mark.parametrize("answer_on_gpu", [True, False])
def test_tune_kernel_torch(lang, args_on_gpu, answer_on_gpu):
    skip_backend(lang)
    n = np.int32(10000)
    device = "cuda" if args_on_gpu else "cpu"
    a = torch.randn(n, device=device)
    b = torch.randn(n, device=device)
    c = torch.ones(n, device=device)
    expected = (c + (a + b)).to("cuda" if answer_on_gpu else "cpu")

    results, _ = tune_kernel(
        "vector_add",
        accumulate_kernel,
        n,
        [c, a, b, n],
        {"block_size_x": [128, 256]},
        lang=lang,
        answer=[expected, None, None, None],
        iterations=3,
        quiet=True,
    )
    assert len(results) == 2
    assert all("__error__" not in r for r in results)


@skip_if_no_torch_cuda
@pytest.mark.parametrize("lang", ["NVCUDA", "PYCUDA"])
def test_run_kernel_torch(lang):
    skip_backend(lang)
    n = np.int32(10000)
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.ones(n, device="cuda")
    c_orig = c.clone()

    result = run_kernel("vector_add", accumulate_kernel, n, [c, a, b, n], {"block_size_x": 128}, lang=lang, quiet=True)

    assert torch.allclose(result[0], (c + (a + b)).cpu())
    # the kernel runs on a copy, the user's tensor is not modified
    assert torch.equal(c, c_orig)
