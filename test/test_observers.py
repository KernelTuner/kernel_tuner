
import pytest

import kernel_tuner
from kernel_tuner.nvml import NVMLObserver
from kernel_tuner.observers import BenchmarkObserver

from .context import skip_if_no_cuda
from .test_runners import env


@skip_if_no_cuda
def test_nvml_observer(env):
    nvmlobserver = NVMLObserver(["nvml_energy", "temperature"])
    env[-1]["block_size_x"] = [128]

    result, _ = kernel_tuner.tune_kernel(*env, observers=[nvmlobserver])

    assert "nvml_energy" in result[0]
    assert "temperature" in result[0]
    assert result[0]["temperature"] > 0


@skip_if_no_cuda
def test_custom_observer(env):
    env[-1]["block_size_x"] = [128]

    class MyObserver(BenchmarkObserver):
        def get_results(self):
            return {"name": self.dev.name}

    result, _ = kernel_tuner.tune_kernel(*env, observers=[MyObserver()])

    assert "name" in result[0]
    assert len(result[0]["name"]) > 0

