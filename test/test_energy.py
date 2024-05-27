import os

from kernel_tuner.energy import energy

from .context import skip_if_no_pycuda, skip_if_no_pynvml

cache_filename = os.path.dirname(os.path.realpath(__file__)) + "/synthetic_fp32_cache_NVIDIA_RTX_A4000.json"

@skip_if_no_pycuda
@skip_if_no_pynvml
def test_create_power_frequency_model():

    ridge_frequency, freqs, nvml_power, fitted_params, scaling = energy.create_power_frequency_model(cache=cache_filename, simulation_mode=True)
    target_value = 1350
    tolerance = 0.05
    assert target_value * (1-tolerance) <= ridge_frequency <= target_value * (1+tolerance)
