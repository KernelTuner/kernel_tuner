import os
from .context import skip_if_no_pycuda
from kernel_tuner.energy import energy


cache_filename = os.path.dirname(os.path.realpath(__file__)) + "/synthetic_fp32_cache_NVIDIA_RTX_A4000.json"


def test_create_power_frequency_model():

    ridge_frequency, freqs, nvml_power, fitted_params, scaling = energy.create_power_frequency_model(cache=cache_filename, simulation_mode=True)
    assert ridge_frequency == 1350

