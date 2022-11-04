"""
This module contains a set of helper functions specifically for auto-tuning codes
for energy efficiency.
"""
import numpy as np
import math
from scipy import optimize

from kernel_tuner import tune_kernel
from kernel_tuner.nvml import nvml, NVMLObserver

fp32_kernel_string = """
__device__ void fp32_n_8(
    float2& a, float2& b, float2& c)
{
    // Perform nr_inner * 4 fma
    for (int i = 0; i < nr_inner; i++) {
        a.x += b.x * c.x;
        a.x -= b.y * c.y;
        a.y += b.x * c.y;
        a.y += b.y * c.x;
    }
}

__global__ void fp32_kernel(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < nr_outer; i++) {
        fp32_n_8(a, b, c);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;
}
"""

def get_frequency_power_relation_fp32(device, n_samples=10, use_locked_clocks=False, nvidia_smi_fallback=None):
    """ Use NVML and PyCUDA with a synthetic kernel to obtain samples of frequency-power pairs """

    if drv is None:
        raise ImportError("get_ridge_point_gr_frequency requires PyCUDA")

    # get some numbers about the device
    drv.init()
    dev = drv.Device(device)
    device_name = dev.name().replace(' ', '_')
    multiprocessor_count = dev.get_attribute(
        drv.device_attribute.MULTIPROCESSOR_COUNT)
    max_block_dim_x = dev.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X)

    # kernel arguments
    data_size = (multiprocessor_count, max_block_dim_x)
    data = np.random.random(np.prod(data_size)).astype(float)
    arguments = [data]

    # setup clocks
    nvml_gr_clocks = get_nvml_gr_clocks(device, n=n_samples)

    # idle power
    power_idle = get_idle_power(device)

    # setup tunable parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [max_block_dim_x]
    tune_params["nr_outer"] = [64]
    tune_params["nr_inner"] = [1024]
    tune_params.update(nvml_gr_clocks)

    # metrics
    metrics = OrderedDict()
    metrics["f"] = lambda p: p["core_freq"]

    nvmlobserver = NVMLObserver(
        ["core_freq", "nvml_power"], device=device, nvidia_smi_fallback=nvidia_smi_fallback)

    results, _ = tune_kernel("fp32_kernel", fp32_kernel_string, problem_size=(multiprocessor_count, 64),
                             arguments=arguments, tune_params=tune_params, observers=[nvmlobserver],
                             verbose=False, quiet=True, metrics=metrics, iterations=10,
                             grid_div_x=[], grid_div_y=[])

    freqs = np.array([res["core_freq"] for res in results])
    nvml_power = np.array([res["nvml_power"] for res in results])

    return freqs, nvml_power


def estimated_voltage(X, clock_threshold, voltage_scale):
    clocks = X
    return [1 + ((clock > clock_threshold) * (1e-3 * voltage_scale * (clock-clock_threshold))) for clock in clocks]


def estimated_power(X, clock_threshold, voltage_scale, clock_scale, power_max):
    clocks = X

    n = len(clocks)
    powers = np.zeros(n)

    voltages = estimated_voltage(clocks, clock_threshold, voltage_scale)

    for i in range(n):
        clock = clocks[i]
        voltage = voltages[i]
        power = 1 + clock_scale * clock * voltage**2 * 1e-3
        powers[i] = min(power_max, power)

    return powers


def fit_performance_frequency_model(freqs, nvml_power):
    """ Fit the performance frequency model based on frequency and power measurements """

    nvml_gr_clocks = np.array(freqs)
    nvml_power = np.array(nvml_power)

    clock_min = min(freqs)
    clock_max = max(freqs)

    nvml_gr_clock_normalized = nvml_gr_clocks - clock_min
    nvml_power_normalized = nvml_power / min(nvml_power)

    clock_threshold = np.median(nvml_gr_clock_normalized)
    voltage_scale = 1
    clock_scale = 1
    power_max = max(nvml_power_normalized)

    x = nvml_gr_clock_normalized
    y = nvml_power_normalized

    # fit the model
    p0 = (clock_threshold, voltage_scale, clock_scale, power_max)
    bounds = ([clock_min, 0, 0, 0.9*power_max],
              [clock_max, 1, 1, 1.1*power_max])
    res = optimize.curve_fit(estimated_power, x, y, p0=p0, bounds=bounds)
    clock_threshold, voltage_scale, clock_scale, power_max = np.round(
        res[0], 2)

    fit_parameters = (clock_threshold, voltage_scale, clock_scale, power_max)
    scale_parameters = (clock_min, min(nvml_power))
    return clock_threshold + clock_min, fit_parameters, scale_parameters


def create_performance_frequency_model(device=0, n_samples=10, verbose=False, nvidia_smi_fallback=None, use_locked_clocks=False):
    """ Calculate the most energy-efficient clock frequency of device

    This function uses a performance model to fit the performance-frequency curve
    using a synthethic benchmarking kernel. The method has been described in:

     * Going green: optimizing GPUs for energy efficiency through model-steered auto-tuning
       R. Schoonhoven, B. Veenboer, B. van Werkhoven, K. J. Batenburg
       International Workshop on Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS) at Supercomputing (SC22) 2022

    Requires NVML and PyCUDA.

    :param device: The device ordinal for NVML
    :type device: int

    :param n_samples: Number of frequencies to sample
    :type n_samples: int

    :param verbose: Enable verbose printing of sampled frequencies and power consumption
    :type verbose: bool

    :param nvidia_smi_fallback: Path to nvidia-smi when insufficient permissions to use NVML directly
    :type nvidia_smi_fallback: string

    :param use_locked_clocks: Whether to prefer locked clocks over application clocks
    :type use_locked_clocks: bool

    :returns: The clock frequency closest to the ridge point, fitted parameters, scaling
    :rtype: float

    """
    freqs, nvml_power = get_frequency_power_relation(device, n_samples, nvidia_smi_fallback, use_locked_clocks)

    if verbose:
        print("Clock frequencies:", freqs.tolist())
        print("Power consumption:", nvml_power.tolist())

    ridge_frequency, fitted_params, scaling = fit_model(freqs, nvml_power)

    if verbose:
        print(f"Modelled most energy efficient frequency: {ridge_frequency} MHz")

    all_frequencies = np.array(get_nvml_gr_clocks(device)['nvml_gr_clock'])
    ridge_frequency_final = all_frequencies[np.argmin(abs(all_frequencies - ridge_frequency))]

    if verbose:
        print(f"Closest configurable most energy efficient frequency: {ridge_frequency2} MHz")

    return ridge_frequency_final, fitted_params, scaling


def get_frequency_range_around_ridge(ridge_frequency, all_frequencies, freq_range, number_of_freqs, verbose=False)
    """ Return number_of_freqs frequencies in a freq_range percentage around the ridge_frequency from among all_frequencies """

    min_freq = 1e-2 * (100 - int(freq_range)) * ridge_frequency
    max_freq = 1e-2 * (100 + int(freq_range)) * ridge_frequency
    frequency_selection = np.unique([all_frequencies[np.argmin(abs(
        all_frequencies - f))] for f in np.linspace(min_freq, max_freq, int(number_of_freqs))]).tolist()

    if verbose:
        print(f"Suggested range of frequencies to auto-tune: {frequency_selection} MHz")
        print(f"Search space reduction: {np.round(100 - len(frequency_selection) / len(all_frequencies) * 100, 1)} %%")

    return frequency_selection
