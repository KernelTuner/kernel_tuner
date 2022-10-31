"""
This module contains a set of helper functions specifically for auto-tuning codes
for energy efficiency.
"""
import argparse
from collections import OrderedDict
import numpy as np
import math
from scipy import optimize
import time

try:
    from pycuda import driver as drv
except ImportError as e:
    drv = None
    raise e

from kernel_tuner import tune_kernel
from kernel_tuner.nvml import nvml, NVMLObserver


def get_nvml_pwr_limits(device, n=None):
    """ Get tunable parameter for NVML power limits, n is desired number of values """

    d = nvml(device)
    power_limits = d.pwr_constraints
    power_limit_min = power_limits[0]
    power_limit_max = power_limits[-1]
    power_limit_min *= 1e-3  # Convert to Watt
    power_limit_max *= 1e-3  # Convert to Watt
    power_limit_round = 5
    tune_params = OrderedDict()
    if n == None:
        n = int((power_limit_max - power_limit_min) / power_limit_round)+1

    # Rounded power limit values
    power_limits = power_limit_round * np.round(
        (np.linspace(power_limit_min, power_limit_max, n) / power_limit_round))
    power_limits = sorted(
        list(set([int(power_limit) for power_limit in power_limits])))
    tune_params["nvml_pwr_limit"] = power_limits
    print("Using power limits:", tune_params["nvml_pwr_limit"])
    return tune_params


def get_nvml_gr_clocks(device, n=None):
    """ Get tunable parameter for NVML graphics clock, n is desired number of values """

    d = nvml(device)
    mem_clock = max(d.supported_mem_clocks)
    gr_clocks = d.supported_gr_clocks[mem_clock]

    if n and (len(gr_clocks) > n):
        indices = np.array(
            np.ceil(np.linspace(0, len(gr_clocks)-1, n)), dtype=int)
        gr_clocks = np.array(gr_clocks)[indices]

    tune_params = OrderedDict()
    tune_params["nvml_gr_clock"] = list(gr_clocks)

    return tune_params


def get_nvml_mem_clocks(device, n=None, quiet=False):
    """ Get tunable parameter for NVML memory clock, n is desired number of values """

    d = nvml(device)
    mem_clocks = d.supported_mem_clocks

    if n and len(mem_clocks) > n:
        mem_clocks = mem_clocks[::int(len(mem_clocks)/n)]

    tune_params = OrderedDict()
    tune_params["nvml_mem_clock"] = mem_clocks

    if not quiet:
        print("Using mem frequencies:", tune_params["nvml_mem_clock"])
    return tune_params


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


def get_idle_power(device, n=5, sleep_s=0.1):
    d = nvml(device)
    readings = []
    for _ in range(n):
        time.sleep(sleep_s)
        readings.append(d.pwr_usage())
    return np.mean(readings) * 1e-3  # Watt


def get_measurements_fp32(device, n_samples=10, nvidia_smi_fallback=None, quiet=False):
    """ Calculate the most energy-efficient clock frequency of device

    This function uses a performance model to fit the frequency-voltage curve
    using a synthethic benchmarking kernel. The method has been described in:

     * Going green: optimizing GPUs for energy efficiency through model-steered auto-tuning
       R. Schoonhoven, B. Veenboer, B. van Werkhoven, K. J. Batenburg
       International Workshop on Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS) at Supercomputing (SC22) 2022

    Requires NVML and PyCUDA.

    :params device: The device ordinal for NVML
    :type device: int

    :returns: The clock frequency closest to the ridge point
    :rtype: float

    """

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
    #metrics["v"] = lambda p:p["gr_voltage"]
    metrics["f"] = lambda p: p["core_freq"]

    nvmlobserver = NVMLObserver(
        ["core_freq", "nvml_power"], device=device, nvidia_smi_fallback=nvidia_smi_fallback)

    results, _ = tune_kernel("fp32_kernel", fp32_kernel_string, problem_size=(multiprocessor_count, 64),
                             arguments=arguments, tune_params=tune_params, observers=[
                                 nvmlobserver],
                             verbose=False, quiet=True, metrics=metrics, iterations=10,
                             grid_div_x=[], grid_div_y=[])

    #voltages = np.array([res["gr_voltage"] for res in results])
    freqs = np.array([res["core_freq"] for res in results])
    nvml_power = np.array([res["nvml_power"] for res in results])

    return freqs, nvml_power


def fit_model(freqs, nvml_power):
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

    x = nvml_gr_clock_normalized
    y = nvml_power_normalized

    # fit the model
    p0 = (clock_threshold, voltage_scale, clock_scale, power_max)
    bounds = ([clock_min, 0, 0, 0.9*power_max],
              [clock_max, 1, 1, 1.1*power_max])
    res = optimize.curve_fit(estimated_power, x, y, p0=p0, bounds=bounds)
    clock_threshold, voltage_scale, clock_scale, power_max = np.round(
        res[0], 2)

    return clock_threshold + clock_min


def get_default_parser():
    parser = argparse.ArgumentParser(
        description='Find energy efficient frequencies')
    parser.add_argument("-d", dest="device", nargs="?",
                        default=0, help="GPU ID to use")
    parser.add_argument("-s", dest="samples", nargs="?",
                        default=10, help="Number of frequency samples")
    parser.add_argument("-r", dest="range", nargs="?",
                        default=10, help="Frequency spread (10%% of 'optimum')")
    parser.add_argument("-n", dest="number", nargs="?", default=10,
                        help="Maximum number of suggested frequencies")
    return parser


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    freqs, nvml_power = get_measurements_fp32(
        args.device, n_samples=args.samples)
    if False:
        print("Clock frequencies:", freqs.tolist())
        print("Power consumption:", nvml_power.tolist())
    ridge_frequency = fit_model(freqs, nvml_power)
    print(f"Modelled most energy efficient frequency: {ridge_frequency} MHz")
    all_frequencies = np.array(
        get_nvml_gr_clocks(args.device)['nvml_gr_clock'])
    ridge_frequency2 = all_frequencies[np.argmin(
        abs(all_frequencies - ridge_frequency))]
    print(
        f"Closest configurable most energy efficient frequency: {ridge_frequency2} MHz")
    min_freq = 1e-2 * (100 - int(args.range)) * ridge_frequency
    max_freq = 1e-2 * (100 + int(args.range)) * ridge_frequency
    np.linspace(min_freq, max_freq)
    frequency_selection = np.unique([all_frequencies[np.argmin(abs(
        all_frequencies - f))] for f in np.linspace(min_freq, max_freq, int(args.number))])
    print(
        f"Suggested range of frequencies to auto-tune: {frequency_selection} MHz")
    print(
        f"Search space reduction: {np.round(100 - len(frequency_selection) / len(all_frequencies) * 100, 1)} %%")
