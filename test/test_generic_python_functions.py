from .context import skip_if_no_torch
from .test_kernel_source_fn import call_mock
from kernel_tuner.core import DeviceInterface, KernelInstance
from kernel_tuner.kernel_sources.kernel_source import KernelSource
import numpy as np
from pathlib import Path
import os

try:
    import torch
    torch_present = True
except ImportError:
    pass

KS_FILE = os.path.join(Path(__file__).resolve().parent, "test_kernel_source_fn.py")

# Helper functions ------------------------------

def value_equal(a, b):
    # Torch tensors
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)

    # NumPy arrays
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)

    # Fallback (ints, floats, strings, lists, tuples, dicts)
    return a == b

def get_context():
    params = {'mock_param': 64}
    a = 42
    b = torch.randn(12, device='cuda', dtype=torch.float32)
    args = [a, b]
    ks = KernelSource("mock_kernel", KS_FILE, "generic_python", call_function=call_mock)
    return ks, args, params


# Tests ----------------------------------------------

@skip_if_no_torch
def test_ready_argument_list():
    ks, args, params = get_context()
    dev = DeviceInterface(ks)

    gpu_args = dev.ready_argument_list(args)

    assert len(args) == len(gpu_args)

    # arg 0: python scalar
    assert isinstance(gpu_args[0], int)
    assert gpu_args[0] == args[0]

    # arg 1: torch cuda tensor
    assert isinstance(gpu_args[1], torch.Tensor)
    assert gpu_args[1].is_cuda

    # values equal
    assert torch.allclose(gpu_args[1], args[1])

    # ensure deep copy
    assert gpu_args[1] is not args[1]


@skip_if_no_torch
def test_compile():
    ks, args, params = get_context()

    instance_data = ks.prepare_kernel_instance(
        kernel_options=None,
        params=params,
        grid=None,
        threads=None,
    )
    dev = DeviceInterface(ks)
    instance = KernelInstance(
        name="mock_kernel",
        kernel_source=ks,
        kernel_string=None,
        kernel_fn=instance_data.kernel_fn,
        temp_files=instance_data.temp_files,
        threads=None,
        grid=None,
        params=params,
        arguments=None,
    )
    gpu_args = dev.ready_argument_list(args)

    callable_fn = dev.compile_kernel(instance, verbose=False, gpu_args=gpu_args)

    # The mock function return the mock_param, which should be set to 64.
    assert callable_fn(*args) == 64


@skip_if_no_torch
def test_gpu_kwargs():
    params = {'mock_param': 64}
    a = torch.randn(12, device='cuda', dtype=torch.float32)
    args = [a] # we do not have to specify the kwarg here
    ks = KernelSource("kernel_with_kwarg", KS_FILE, "generic_python", call_function=call_mock)
    dev = DeviceInterface(ks)

    instance_data = ks.prepare_kernel_instance(
        kernel_options=None,
        params=params,
        grid=None,
        threads=None,
    )
    dev = DeviceInterface(ks)
    instance = KernelInstance(
        name="kernel_with_kwarg",
        kernel_source=ks,
        kernel_string=None,
        kernel_fn=instance_data.kernel_fn,
        temp_files=instance_data.temp_files,
        threads=None,
        grid=None,
        params=params,
        arguments=None,
    )
    gpu_args = dev.ready_argument_list(args)
    callable_fn = dev.compile_kernel(instance, verbose=False, gpu_args=gpu_args)

    kwargs = dev.dev.gpu_kwargs
    
    assert kwargs["mock_param"] == 64
    assert callable_fn(*args, **kwargs) == 64

