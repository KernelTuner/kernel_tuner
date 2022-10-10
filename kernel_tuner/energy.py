"""
This module contains a set of helper functions specifically for auto-tuning codes
for energy efficiency.
"""
from collections import OrderedDict
import numpy as np

try:
    from pycuda import driver as drv
except ImportError:
    drv = None

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
    power_limits = sorted(list(set([int(power_limit) for power_limit in power_limits])))
    tune_params["nvml_pwr_limit"] = power_limits
    print("Using power limits:", tune_params["nvml_pwr_limit"])
    return tune_params


def get_nvml_gr_clocks(device, n=None):
    """ Get tunable parameter for NVML graphics clock, n is desired number of values """

    d = nvml(device)
    mem_clock = max(d.supported_mem_clocks)
    gr_clocks = d.supported_gr_clocks[mem_clock]

    if n and (len(gr_clocks) > n):
        gr_clocks = gr_clocks[::math.ceil(len(gr_clocks)/n)]

    tune_params = OrderedDict()
    tune_params["nvml_gr_clock"] = gr_clocks[::-1]
    print("Using clock frequencies:", tune_params["nvml_gr_clock"])
    return tune_params


def get_nvml_mem_clocks(device, n=None):
    """ Get tunable parameter for NVML memory clock, n is desired number of values """

    d = nvml(device)
    mem_clocks = d.supported_mem_clocks

    if n and len(mem_clocks) > n:
        mem_clocks = mem_clocks[::int(len(mem_clocks)/n)]

    tune_params = OrderedDict()
    tune_params["nvml_mem_clock"] = mem_clocks
    print("Using mem frequencies:", tune_params["nvml_mem_clock"])
    return tune_params


fp32_kernel_string = """
__device__ void fp32_n_8(
    float2& a, float2& b, float2& c)
{
    // Perform nr_inner * 4 fma
    for (int i = 0; i < nr_inner; i++) {
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(b.x),  "f"(c.x), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(-b.y), "f"(c.y), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.x),  "f"(c.y), "f"(a.y));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.y),  "f"(c.x), "f"(a.y));
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


def get_ridge_point_gr_frequency(device, nvidia_smi_fallback=None):
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
    multiprocessor_count = dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)
    max_block_dim_x = dev.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X)

    # kernel arguments
    data_size = (multiprocessor_count, max_block_dim_x)
    data = np.zeros(np.prod(data_size)).astype(float)
    arguments = [data]

    # setup tunable parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [max_block_dim_x]
    tune_params["nr_outer"] = [64]
    tune_params["nr_inner"] = [1024]
    tune_params.update(get_nvml_gr_clocks(device))

    # metrics
    metrics = OrderedDict()
    metrics["v"] = lambda p:p["gr_voltage"]
    metrics["f"] = lambda p:p["core_freq"]

    nvmlobserver = NVMLObserver(["gr_voltage", "core_freq"], device=device, nvidia_smi_fallback=nvidia_smi_fallback)

    results, _ = tune_kernel("fp32_kernel", fp32_kernel_string, problem_size=(multiprocessor_count),
                    arguments=arguments, tune_params=tune_params, observers=[nvmlobserver],
                    verbose=False, quiet=True, metrics=metrics, iterations=1,
                    grid_div_x=[], grid_div_y=[], cache="detect-ridge-point-synthetic-fp32-cache.json")

    voltages = np.array([res["gr_voltage"] for res in results])
    freqs = np.array([res["core_freq"] for res in results])

    # detect ridge point, using distance to slope line method
    slope = (voltages[-1] - voltages[0]) / len(voltages)
    slope_line = range(len(voltages)) * slope + voltages[0]
    ridge_point_index = (slope_line - voltages).argmax()

    print(f"Ridge point detected at frequency: {freqs[ridge_point_index]}")

    return freqs[ridge_point_index]


def get_gr_clocks_ridge_point_method(gr_clocks, ridge_point, percentage=10):
    """ Get tunable parameter for graphics clock around ridge_point, using all values within percentage """

    clocks = np.array(gr_clocks)

    # get frequency closest to ridge point
    nearest_index = (np.abs(clocks - ridge_point)).argmin()
    nearest_freq = clocks[nearest_index]

    # get clocks within percentage (above and below) of ridge point
    lower_bound = nearest_freq / 100 * (100-percentage)
    upper_bound = nearest_freq / 100 * (100+percentage)
    filtered_clocks = clocks[clocks > lower_bound]
    filtered_clocks = filtered_clocks[filtered_clocks< upper_bound]

    tune_params = OrderedDict()
    tune_params["nvml_gr_clock"] = filtered_clocks
    print("Using clock frequencies:", tune_params["nvml_gr_clock"])
    return tune_params


