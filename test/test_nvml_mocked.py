import numpy as np

try:
    from mock import patch, Mock, MagicMock
except ImportError:
    from unittest.mock import patch, Mock

from kernel_tuner.observers.nvml import get_nvml_pwr_limits, get_nvml_gr_clocks, get_nvml_mem_clocks, get_idle_power



def setup_mock(nvml):
    nvml.return_value.configure_mock(pwr_constraints=(90000, 150000),
                                     supported_mem_clocks=[2100],
                                     supported_gr_clocks={2100: [1000, 2000, 3000]},
                                     pwr_usage=lambda : 5000)

    return nvml


@patch('kernel_tuner.observers.nvml.nvml')
def test_get_nvml_pwr_limits(nvml):
    nvml = setup_mock(nvml)
    result = get_nvml_pwr_limits(0, quiet=True)
    assert result['nvml_pwr_limit'] == [90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]

    result = get_nvml_pwr_limits(0, n=5, quiet=True)
    assert len(result['nvml_pwr_limit']) == 5
    assert result['nvml_pwr_limit'][0] == 90
    assert result['nvml_pwr_limit'][-1] == 150


@patch('kernel_tuner.observers.nvml.nvml')
def test_get_nvml_gr_clocks(nvml):
    nvml = setup_mock(nvml)
    result = get_nvml_gr_clocks(0, quiet=True)
    assert result['nvml_gr_clock'] == [1000, 2000, 3000]

    result = get_nvml_gr_clocks(0, n=2, quiet=True)
    assert result['nvml_gr_clock'] == [1000, 3000]


@patch('kernel_tuner.observers.nvml.nvml')
def test_get_nvml_mem_clocks(nvml):
    nvml = setup_mock(nvml)
    result = get_nvml_mem_clocks(0, quiet=False)
    print(result)
    assert result['nvml_mem_clock'] == [2100]


@patch('kernel_tuner.observers.nvml.nvml')
def test_get_idle_power(nvml):
    nvml = setup_mock(nvml)
    result = get_idle_power(0)
    assert np.isclose(result, 5)

