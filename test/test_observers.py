
import pytest

import kernel_tuner
from kernel_tuner.nvml import NVMLObserver

from .context import skip_if_no_cuda
from .test_runners import env


@skip_if_no_cuda
def test_nvml_observer(env):
    nvmlobserver = NVMLObserver(["nvml_energy", "temperature"])

    result, _ = kernel_tuner.tune_kernel(*env, observers=[nvmlobserver])

    assert "nvml_energy" in result[0]
    assert "temperature" in result[0]
    assert result[0]["temperature"] > 0
