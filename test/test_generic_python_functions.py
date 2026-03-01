from .context import skip_if_no_torch
from .test_kernel_source_fn import mock_kernel, kernel_with_kwarg, call_mock
from kernel_tuner.core import DeviceInterface, KernelInstance
from kernel_tuner.kernel_sources.kernel_source import KernelSource
import numpy as np

try:
    import torch
    torch_present = True
except ImportError:
    pass


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
    ks = KernelSource("mock_kernel", mock_kernel, "generic_python", call_function=call_mock)
    return ks, args, params


# Tests ----------------------------------------------

@skip_if_no_torch
def test_ready_argument_list():
    ks, args, params = get_context()
    dev = DeviceInterface(ks)
    gpu_args = dev.ready_argument_list(args)

    assert len(args) == len(gpu_args)

    for i, _ in enumerate(gpu_args):
        assert value_equal(args[i], gpu_args[i])
        if type(gpu_args[i]) in (list, dict, torch.Tensor, np.ndarray):
            assert gpu_args[i] is not args[i] # Assure deep copy


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
    ks = KernelSource("kernel_with_kwarg", kernel_with_kwarg, "generic_python", call_function=call_mock)
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

