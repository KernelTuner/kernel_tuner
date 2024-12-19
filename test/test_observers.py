from pytest import raises

import kernel_tuner
from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers.observer import BenchmarkObserver
from kernel_tuner.observers.register import RegisterObserver

from .context import (
    skip_if_no_cuda,
    skip_if_no_cupy,
    skip_if_no_opencl,
    skip_if_no_pycuda,
    skip_if_no_hip,
    skip_if_no_pynvml,
)
from .test_hip_functions import env as env_hip  # noqa: F401
from .test_opencl_functions import env as env_opencl  # noqa: F401
from .test_runners import env  # noqa: F401


@skip_if_no_pycuda
@skip_if_no_pynvml
def test_nvml_observer(env):
    nvmlobserver = NVMLObserver(["nvml_energy", "temperature"])
    env[-1]["block_size_x"] = [128]

    result, _ = kernel_tuner.tune_kernel(*env, observers=[nvmlobserver])

    assert "nvml_energy" in result[0]
    assert "temperature" in result[0]
    assert result[0]["temperature"] > 0

@skip_if_no_pycuda
def test_custom_observer(env):
    env[-1]["block_size_x"] = [128]

    class MyObserver(BenchmarkObserver):
        def get_results(self):
            return {"name": self.dev.name}

    result, _ = kernel_tuner.tune_kernel(*env, observers=[MyObserver()])

    assert "name" in result[0]
    assert len(result[0]["name"]) > 0

@skip_if_no_pycuda
def test_register_observer_pycuda(env):
    result, _ = kernel_tuner.tune_kernel(*env, observers=[RegisterObserver()], lang='CUDA')
    assert "num_regs" in result[0]
    assert result[0]["num_regs"] > 0

@skip_if_no_cupy
def test_register_observer_cupy(env):
    result, _ = kernel_tuner.tune_kernel(*env, observers=[RegisterObserver()], lang='CuPy')
    assert "num_regs" in result[0]
    assert result[0]["num_regs"] > 0

@skip_if_no_cuda
def test_register_observer_nvcuda(env):
    result, _ = kernel_tuner.tune_kernel(*env, observers=[RegisterObserver()], lang='NVCUDA')
    assert "num_regs" in result[0]
    assert result[0]["num_regs"] > 0

@skip_if_no_opencl
def test_register_observer_opencl(env_opencl):
    with raises(NotImplementedError) as err:
        kernel_tuner.tune_kernel(*env_opencl, observers=[RegisterObserver()], lang='OpenCL')
    assert err.errisinstance(NotImplementedError)
    assert "OpenCL" in str(err.value)

@skip_if_no_hip
def test_register_observer_hip(env_hip):
    with raises(NotImplementedError) as err:
        kernel_tuner.tune_kernel(*env_hip, observers=[RegisterObserver()], lang='HIP')
    assert err.errisinstance(NotImplementedError)
    assert "Hip" in str(err.value)
