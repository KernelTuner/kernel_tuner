
import numpy as np

try:
    from unittest.mock import patch, Mock, MagicMock
except ImportError:
    from mock import patch, Mock, MagicMock

import kernel_tuner

from .context import skip_if_no_cupy
from .test_runners import env  # noqa: F401


@skip_if_no_cupy
def test_tune_kernel(env):
    result, _ = kernel_tuner.tune_kernel(*env, lang="cupy", verbose=True)
    assert len(result) > 0


@patch('kernel_tuner.backends.cupy.cupyx')
@patch('kernel_tuner.backends.cupy.cp')
def test_cupy_init_uses_cupyx_not_private_attr(cp_mock, cupyx_mock):
    """Regression test: CupyFunctions should use cupyx.get_runtime_info(), not cp._cupyx."""
    # Setup cp mock
    dev_mock = MagicMock()
    dev_mock.attributes = {"MaxThreadsPerBlock": 1024}
    dev_mock.compute_capability = "75"
    cp_mock.cuda.Device.return_value = dev_mock
    cp_mock.cuda.runtime.driverGetVersion.return_value = 11000

    # Setup cupyx mock to return runtime info string
    runtime_info = "CuPy Version          : 14.0.1\nCUDA Root             : /usr/local/cuda\nDevice 0 Name         : Tesla T4\n"
    cupyx_mock.get_runtime_info.return_value = runtime_info

    from kernel_tuner.backends.cupy import CupyFunctions
    dev = CupyFunctions(device=0)

    # Verify cupyx.get_runtime_info() was called, not cp._cupyx.get_runtime_info()
    cupyx_mock.get_runtime_info.assert_called_once()
    assert dev.name == "Tesla T4"
