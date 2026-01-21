"""Module for kernel tuner utility functions."""
from inspect import signature
from pathlib import Path
import errno
import json
import logging
import numpy as np
import os
import re
import sys
import tempfile
import time
import warnings

from kernel_tuner.accuracy import Tunable

try:
    import cupy as cp
except ImportError:
    cp = np

from kernel_tuner.observers.nvml import NVMLObserver

# number of special values to insert when a configuration cannot be measured


class ErrorConfig(str):
    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__


class InvalidConfig(ErrorConfig):
    pass


class CompilationFailedConfig(ErrorConfig):
    pass


class RuntimeFailedConfig(ErrorConfig):
    pass


class NpEncoder(json.JSONEncoder):
    """Class we use for dumping Numpy objects to JSON."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class TorchPlaceHolder:
    def __init__(self):
        self.Tensor = Exception  # using Exception here as a type that will never be among kernel arguments


class SkippableFailure(Exception):
    """Exception used to raise when compiling or launching a kernel fails for a reason that can be expected."""


class StopCriterionReached(Exception):
    """Exception thrown when a stop criterion has been reached."""


try:
    import torch
except ImportError:
    torch = TorchPlaceHolder()

default_block_size_names = [
    "block_size_x",
    "block_size_y",
    "block_size_z",
]

try:
    from hip._util.types import DeviceArray
except ImportError:
    DeviceArray = Exception  # using Exception here as a type that will never be among kernel arguments


def check_argument_type(dtype, kernel_argument):
    """Check if the numpy.dtype matches the type used in the code."""
    types_map = {
        "bool": ["bool"],
        "uint8": ["uchar", "unsigned char", "uint8_t"],
        "int8": ["char", "int8_t"],
        "uint16": ["ushort", "unsigned short", "uint16_t"],
        "int16": ["short", "int16_t"],
        "uint32": ["uint", "unsigned int", "uint32_t"],
        "int32": ["int", "int32_t"],  # discrepancy between OpenCL and C here, long may be 32bits in C
        "uint64": ["ulong", "unsigned long", "uint64_t"],
        "int64": ["long", "int64_t"],
        "float16": ["half"],
        "float32": ["float"],
        "float64": ["double"],
        "complex64": ["float2"],
        "complex128": ["double2"],
    }
    if dtype in types_map:
        return any([substr in kernel_argument for substr in types_map[dtype]])
    return False  # unknown dtype. do not throw exception to still allow kernel to run.


def check_argument_list(kernel_name, kernel_string, args):
    """Raise an exception if kernel arguments do not match host arguments."""
    kernel_arguments = list()
    collected_errors = list()

    for iterator in re.finditer(kernel_name + "[ \n\t]*" + r"\(", kernel_string):
        kernel_start = iterator.end()
        kernel_end = kernel_string.find(")", kernel_start)
        if kernel_start != 0:
            kernel_arguments.append(kernel_string[kernel_start:kernel_end].split(","))

    for arguments_set, arguments in enumerate(kernel_arguments):
        collected_errors.append(list())
        if len(arguments) != len(args):
            collected_errors[arguments_set].append("Kernel and host argument lists do not match in size.")
            continue

        for i, arg in enumerate(args):
            kernel_argument = arguments[i]

            # Handle tunable arguments
            if isinstance(arg, Tunable):
                continue

            # Handle numpy arrays and other array types
            if not isinstance(arg, (np.ndarray, np.generic, cp.ndarray, torch.Tensor, DeviceArray)):
                raise TypeError(
                    f"Argument at position {i} of type: {type(arg)} should be of type "
                    "np.ndarray, numpy scalar, or HIP Python DeviceArray type"
                )

            correct = True
            if isinstance(arg, np.ndarray):
                if "*" not in kernel_argument:
                    correct = False

            if isinstance(arg, DeviceArray):
                str_dtype = str(np.dtype(arg.typestr))
            else:
                str_dtype = str(arg.dtype)

            if correct and check_argument_type(str_dtype, kernel_argument):
                continue

            collected_errors[arguments_set].append(
                f"Argument at position {i} of dtype: {str_dtype} does not match {kernel_argument}."
            )

        if not collected_errors[arguments_set]:
            # We assume that if there is a possible list of arguments that matches with the provided one
            # it is the right one
            return

    for errors in collected_errors:
        warnings.warn(errors[0], UserWarning)


def check_stop_criterion(to: dict) -> float:
    """Check if the stop criterion is reached.

    Args:
        to (dict): tuning options.

    Raises:
        StopCriterionReached: if the max_fevals is reached or time limit is exceeded.

    Returns:
        float: fraction of budget spent. If both max_fevals and time_limit are set, it returns the fraction of time.
    """
    if "max_fevals" in to:
        if len(to.unique_results) >= to.max_fevals:
            raise StopCriterionReached(f"max_fevals ({to.max_fevals}) reached")
        if not "time_limit" in to:
            return len(to.unique_results) / to.max_fevals
    if "time_limit" in to:
        time_spent = (time.perf_counter() - to.start_time) + (to.simulated_time * 1e-3) + to.startup_time
        if time_spent > to.time_limit:
            raise StopCriterionReached("time limit exceeded")
        return time_spent / to.time_limit
    


def check_tune_params_list(tune_params, observers, simulation_mode=False):
    """Raise an exception if a tune parameter has a forbidden name."""
    forbidden_names = ("grid_size_x", "grid_size_y", "grid_size_z", "time")
    for name, param in tune_params.items():
        if name in forbidden_names:
            raise ValueError("Tune parameter " + name + " with value " + str(param) + " has a forbidden name!")
    if any("nvml_" in param for param in tune_params):
        if not simulation_mode and (not observers or not any(isinstance(obs, NVMLObserver) for obs in observers)):
            raise ValueError("Tune parameters starting with nvml_ require an NVMLObserver!")


def check_block_size_names(block_size_names):
    if block_size_names is not None:
        # do some type checks for the user input
        if not isinstance(block_size_names, list):
            raise ValueError("block_size_names should be a list of strings!")
        if len(block_size_names) > 3:
            raise ValueError("block_size_names should not contain more than 3 names!")
        if not all([isinstance(name, "".__class__) for name in block_size_names]):
            raise ValueError("block_size_names should contain only strings!")


def append_default_block_size_names(block_size_names):
    if block_size_names is None:
        return
    for i, name in enumerate(default_block_size_names):
        if len(block_size_names) < i + 1:
            block_size_names.append(name)


def check_block_size_params_names_list(block_size_names, tune_params):
    if block_size_names is not None:
        for name in block_size_names:
            if name not in tune_params.keys():
                warnings.warn(
                    "Block size name " + name + " is not specified in the tunable parameters list!", UserWarning
                )
    else:  # if default block size names are used
        if not any([k.lower() in default_block_size_names for k in tune_params.keys()]):
            warnings.warn(
                "None of the tunable parameters specify thread block dimensions!",
                UserWarning,
            )
        else:
            # check for alternative case spelling of defaults such as BLOCK_SIZE_X or block_Size_X etc
            result = []
            for k in tune_params.keys():
                if k.lower() in default_block_size_names and k not in default_block_size_names:
                    result.append(k)
            # ensure order of block_size_names is correct regardless of case used
            block_size_names = sorted(result, key=str.casefold)

    return block_size_names


def delete_temp_file(filename):
    """Delete a temporary file, don't complain if no longer exists."""
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise e


def detect_language(kernel_string):
    """Attempt to detect language from the kernel_string."""
    if "__global__" in kernel_string:
        lang = "CUDA"
    elif "__kernel" in kernel_string:
        lang = "OpenCL"
    else:
        lang = "C"
    return lang


def get_best_config(results, objective, objective_higher_is_better=False):
    """Returns the best configuration from a list of results according to some objective."""
    func = max if objective_higher_is_better else min
    ignore_val = sys.float_info.max if not objective_higher_is_better else -sys.float_info.max
    best_config = func(
        results,
        key=lambda x: x[objective] if isinstance(x[objective], float) else ignore_val,
    )
    return best_config


def get_config_string(params, keys=None, units=None):
    """Return a compact string representation of a measurement."""

    def compact_number(v):
        if isinstance(v, float):
            return "{:.3f}".format(round(v, 3))
        else:
            return str(v)

    compact_str_items = []
    if not keys:
        keys = params.keys()
    # first make a list of compact strings for each parameter
    for k, v in params.items():
        if k in keys:
            unit = ""
            # check if units not None not enough, units could be mocked which causes errors
            if isinstance(units, dict) and not isinstance(v, ErrorConfig):
                unit = units.get(k, "")
            compact_str_items.append(k + "=" + compact_number(v) + unit)
    # and finally join them
    compact_str = ", ".join(compact_str_items)
    return compact_str


def get_grid_dimensions(current_problem_size, params, grid_div, block_size_names):
    """Compute grid dims based on problem sizes and listed grid divisors."""

    def get_dimension_divisor(divisor, default, params):
        divisor_num = 1

        if divisor is None:
            divisor_num = params.get(default, 1)
        elif isinstance(divisor, int):
            divisor_num = divisor
        elif callable(divisor):
            divisor_num = divisor(params)
        elif isinstance(divisor, str):
            divisor_num = int(eval(replace_param_occurrences(divisor, params)))
        elif np.iterable(divisor):
            for div in divisor:
                divisor_num *= get_dimension_divisor(div, 1, params)
        else:
            raise ValueError("Error: unrecognized type in grid divisor list, should be any of int, str, callable, or iterable")

        return divisor_num

    divisors = [get_dimension_divisor(d, block_size_names[i], params) for i, d in enumerate(grid_div)]
    return tuple(int(np.ceil(float(current_problem_size[i]) / float(d))) for i, d in enumerate(divisors))


def get_instance_string(params):
    """Combine the parameters to a string mostly used for debug output use of dict is advised."""
    return "_".join([str(i) for i in params.values()])


def get_interval(a: list):
    """Checks if an array can be an interval. Returns (start, end, step) if interval, otherwise None."""
    if len(a) < 3:
        return None
    if not all(isinstance(e, (int, float)) for e in a):
        return None
    a_min = min(a)
    a_max = max(a)
    if len(a) <= 2:
        return (a_min, a_max, a_max - a_min)
    # determine the first step size
    step = a[1] - a_min
    # for each element, the step size should be equal to the first step
    for i, e in enumerate(a):
        if e - a[i - 1] != step:
            return None
    result = (a_min, a_max, step)
    if not all(isinstance(e, (int, float)) for e in result):
        return None
    return result


def get_kernel_string(kernel_source, params=None):
    """Retrieve the kernel source and return as a string.

    This function processes the passed kernel_source argument, which could be
    a function, a string with a filename, or just a string with code already.

    If kernel_source is a function, the function is called with instance
    parameters in 'params' as the only argument.

    If kernel_source looks like filename, the file is read in, but if
    the file does not exist, it is assumed that the string is not a filename
    after all.

    :param kernel_source: One of the sources for the kernel, could be a
        function that generates the kernel code, a string or Path containing a
        filename that points to the kernel source, or just a string that
        contains the code.
    :type kernel_source: string or callable

    :param params: Dictionary containing the tunable parameters for this specific
        kernel instance, only needed when kernel_source is a generator.
    :type param: dict

    :returns: A string containing the kernel code.
    :rtype: string
    """
    # logging.debug('get_kernel_string called with %s', str(kernel_source))
    logging.debug("get_kernel_string called")

    kernel_string = None
    if callable(kernel_source):
        kernel_string = kernel_source(params)
    elif isinstance(kernel_source, Path):
        kernel_string = read_file(kernel_source)
    elif isinstance(kernel_source, str):
        if looks_like_a_filename(kernel_source):
            with open(kernel_source, "r") as f:
                kernel_string = f.read()
        else:
            kernel_string = kernel_source
    else:
        raise TypeError("Error kernel_source is not a string nor a callable function")
    return kernel_string


def get_problem_size(problem_size, params):
    """Compute current problem size."""
    if callable(problem_size):
        problem_size = problem_size(params)
    if isinstance(problem_size, (str, int, np.integer)):
        problem_size = (problem_size,)
    current_problem_size = [1, 1, 1]
    for i, s in enumerate(problem_size):
        if isinstance(s, str):
            current_problem_size[i] = int(eval(replace_param_occurrences(s, params)))
        elif isinstance(s, (int, np.integer)):
            current_problem_size[i] = s
        else:
            raise TypeError("Error: problem_size should only contain strings or integers")
    return current_problem_size


def get_smem_args(smem_args, params):
    """Return a dict with kernel instance specific size."""
    result = smem_args.copy()
    if "size" in result:
        size = result["size"]
        if callable(size):
            size = size(params)
        elif isinstance(size, str):
            size = replace_param_occurrences(size, params)
            size = int(eval(size))
        result["size"] = size
    return result


def get_temp_filename(suffix=None):
    """Return a string in the form of temp_X, where X is a large integer."""
    tmp_file = tempfile.mkstemp(
        suffix=suffix or "", prefix="temp_", dir=os.getcwd()
    )  # or "" for Python 2 compatibility
    os.close(tmp_file[0])
    return tmp_file[1]


def get_thread_block_dimensions(params, block_size_names=None):
    """Thread block size from tuning params, currently using convention."""
    if not block_size_names:
        block_size_names = default_block_size_names

    block_size_x = params.get(block_size_names[0], 256)
    block_size_y = params.get(block_size_names[1], 1)
    block_size_z = params.get(block_size_names[2], 1)
    return (int(block_size_x), int(block_size_y), int(block_size_z))


def get_total_timings(results, env, overhead_time):
    """Sum all timings and put their totals in the env."""
    total_framework_time = 0
    total_strategy_time = 0
    total_compile_time = 0
    total_verification_time = 0
    total_benchmark_time = 0
    if results:
        for result in results:
            if (
                "framework_time" not in result
                or "strategy_time" not in result
                or "compile_time" not in result
                or "verification_time" not in result
            ):
                # warnings.warn("No detailed timings in results")
                return env
            total_framework_time += result["framework_time"]
            total_strategy_time += result["strategy_time"]
            total_compile_time += result["compile_time"]
            total_verification_time += result["verification_time"]
            total_benchmark_time += result["benchmark_time"]

    # add the separate time values to the environment dict
    env["total_framework_time"] = total_framework_time
    env["total_strategy_time"] = total_strategy_time
    env["total_compile_time"] = total_compile_time
    env["total_verification_time"] = total_verification_time
    env["total_benchmark_time"] = total_benchmark_time
    if "simulated_time" in env:
        overhead_time += env["simulated_time"]
    env["overhead_time"] = overhead_time - (
        total_framework_time + total_strategy_time + total_compile_time + total_verification_time + total_benchmark_time
    )
    return env


def print_config(config, tuning_options, runner):
    """Print the configuration string with tunable parameters and benchmark results."""
    print_config_output(tuning_options.tune_params, config, runner.quiet, tuning_options.metrics, runner.units)


def print_config_output(tune_params, params, quiet, metrics, units):
    """Print the configuration string with tunable parameters and benchmark results."""
    print_keys = list(tune_params.keys()) + ["time"]
    if metrics:
        print_keys += metrics.keys()
    output_string = get_config_string(params, print_keys, units)
    if not quiet:
        print(output_string)


def process_metrics(params, metrics):
    """Process user-defined metrics for derived benchmark results.

    Metrics must be a dictionary to support composable metrics. The dictionary keys describe
    the name given to this user-defined metric and will be used as the key in the results dictionaries
    return by Kernel Tuner. The values describe how to calculate the user-defined metric, using either a
    string expression in which the tunable parameters and benchmark results can be used as variables, or
    as a function that accepts a dictionary as argument.

    Example:
    metrics = dict()
    metrics["x"] = "10000 / time"
    metrics["x2"] = "x*x"

    Note that the values in the metric dictionary can also be functions that accept params as argument.


    Example:
    metrics = dict()
    metrics["GFLOP/s"] = lambda p : 10000 / p["time"]

    :param params: A dictionary with tunable parameters and benchmark results.
    :type params: dict

    :param metrics: A dictionary with user-defined metrics that can be used to create derived benchmark results.
    :type metrics: dict

    :returns: An updated params dictionary with the derived metrics inserted along with the benchmark results.
    :rtype: dict

    """
    if not isinstance(metrics, dict):
        raise ValueError("metrics should be a dictionary to preserve order and support composability")
    for k, v in metrics.items():
        if isinstance(v, str):
            value = eval(replace_param_occurrences(v, params))
        elif callable(v):
            value = v(params)
        else:
            raise ValueError("metric dicts values should be strings or callable")
        # We overwrite any existing values for the given key
        params[k] = value
    return params


def looks_like_a_filename(kernel_source):
    """Attempt to detect whether source code or a filename was passed."""
    logging.debug("looks_like_a_filename called")
    result = False
    if isinstance(kernel_source, str):
        result = True
        # test if not too long
        if len(kernel_source) > 250:
            result = False
        # test if not contains special characters
        for c in "();{}\\":
            if c in kernel_source:
                result = False
        # just a safeguard for stuff that looks like code
        for s in ["__global__ ", "__kernel ", "void ", "float "]:
            if s in kernel_source:
                result = False
        # string must contain substring ".c", ".opencl", or ".F"
        result = result and any([s in kernel_source for s in (".c", ".opencl", ".F")])
    logging.debug("kernel_source is a filename: %s" % str(result))
    return result


def prepare_kernel_string(kernel_name, kernel_string, params, grid, threads, block_size_names, lang, defines):
    """Prepare kernel string for compilation.

    Prepends the kernel with a series of C preprocessor defines specific
    to this kernel instance:

     * the thread block dimensions
     * the grid dimensions
     * tunable parameters

    :param kernel_name: Name of the kernel.
    :type kernel_name: string

    :param kernel_string: One of the source files of the kernel as a string containing code.
    :type kernel_string: string

    :param params: A dictionary containing the tunable parameters specific to this instance.
    :type params: dict

    :param grid: A tuple with the grid dimensions for this specific instance.
    :type grid: tuple(x,y,z)

    :param threads: A tuple with the thread block dimensions for this specific instance.
    :type threads: tuple(x,y,z)

    :param block_size_names: A tuple with the names of the thread block dimensions used
        in the code. By default this is ["block_size_x", ...], but the user
        may supply different names if they prefer.
    :type block_size_names: tuple(string)

    :param defines: A dict that describes the variables that should be defined as
        preprocessor macros. Each keys should be the variable names and each value
        is either a string or a function that returns a string. If `None`, each
        tunable parameter is defined as preprocessor macro instead.
    :type defines: dict or None

    :returns: A string containing the source code made specific to this kernel instance.
    :rtype: string

    """
    logging.debug("prepare_kernel_string called for %s", kernel_name)

    kernel_prefix = ""

    # If `defines` is `None`, the default behavior is to define the following variables:
    #  * grid_size_x, grid_size_y, grid_size_z
    #  * block_size_x, block_size_y, block_size_z
    #  * each tunable parameter
    #  * kernel_tuner=1
    if defines is None:
        defines = dict()

        grid_dim_names = ["grid_size_x", "grid_size_y", "grid_size_z"]
        for i, g in enumerate(grid):
            defines[grid_dim_names[i]] = g

        for i, g in enumerate(threads):
            defines[block_size_names[i]] = g

        for k, v in params.items():
            defines[k] = v

        defines["kernel_tuner"] = 1

    for k, v in defines.items():
        if callable(v):
            v = v(params)
        elif isinstance(v, str):
            v = replace_param_occurrences(v, params)

        if not k.isidentifier():
            raise ValueError(f"name is not a valid identifier: {k}")

        # Escape newline characters
        v = str(v)
        v = v.replace("\n", "\\\n")

        if "loop_unroll_factor" in k and lang in ("CUDA", "HIP"):
            # this handles the special case that in CUDA/HIP
            # pragma unroll loop_unroll_factor, loop_unroll_factor should be a constant integer expression
            # in OpenCL this isn't the case and we can just insert "#define loop_unroll_factor N"
            # using 0 to disable specifying a loop unrolling factor for this loop
            if v == "0":
                kernel_string = re.sub(r"\n\s*#pragma\s+unroll\s+" + k, "\n", kernel_string)  # + r"[^\S]*"
            else:
                kernel_prefix += f"constexpr int {k} = {v};\n"
        else:
            kernel_prefix += f"#define {k} {v}\n"

    # since we insert defines above the original kernel code, the line numbers will be incorrect
    # the following preprocessor directive informs the compiler that lines should be counted from 1
    if kernel_prefix:
        kernel_prefix += "#line 1\n"

    # Also replace parameter occurrences inside the kernel name
    name = replace_param_occurrences(kernel_name, params)

    return name, kernel_prefix + kernel_string


def read_file(filename):
    """Return the contents of the file named filename or None if file not found."""
    if isinstance(filename, Path):
        with filename.open() as f:
            return f.read()
    elif os.path.isfile(filename):
        with open(filename, "r") as f:
            return f.read()


def replace_param_occurrences(string: str, params: dict):
    """Replace occurrences of the tuning params with their current value."""
    result = ""

    # Split on tokens and replace a token if it is a key in `params`.
    for part in re.split("([a-zA-Z0-9_]+)", string):
        if part in params:
            result += str(params[part])
        else:
            result += part

    return result


def setup_block_and_grid(problem_size, grid_div, params, block_size_names=None):
    """Compute problem size, thread block and grid dimensions for this kernel."""
    threads = get_thread_block_dimensions(params, block_size_names)
    current_problem_size = get_problem_size(problem_size, params)
    grid = get_grid_dimensions(current_problem_size, params, grid_div, block_size_names)
    return threads, grid


def write_file(filename, string):
    """Dump the contents of string to a file called filename."""
    # ugly fix, hopefully we can find a better one
    if sys.version_info[0] >= 3:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(string)
    else:
        with open(filename, "w") as f:
            f.write(string.encode("utf-8"))


def normalize_verify_function(v):
    """Normalize a user-specified verify function.

    The user-specified function has two required positional arguments (answer, result_host),
    and an optional keyword (or keyword-only) argument atol. We normalize it to always accept
    an atol keyword argument.

    Undefined behaviour if the passed function does not match the required signatures.
    """

    # python 3.3+
    def has_kw_argument(func, name):
        sig = signature(func)
        return name in sig.parameters

    if v is None:
        return None

    if has_kw_argument(v, "atol"):
        return v
    return lambda answer, result_host, atol: v(answer, result_host)



def check_matching_problem_size(cached_problem_size, problem_size):
    """Check the if requested problem size matches the problem size in the cache."""
    cached_problem_size_arr = np.array(cached_problem_size)
    problem_size_arr = np.array(problem_size)
    if cached_problem_size_arr.size != problem_size_arr.size or not (cached_problem_size_arr == problem_size_arr).all():
        raise ValueError(f"Cannot load cache which contains results for different problem_size, cache: {cached_problem_size}, requested: {problem_size}")

def process_cache(cache, kernel_options, tuning_options, runner):
    """Cache file for storing tuned configurations.

    the cache file is stored using JSON and uses the following format:

    .. code-block:: python

        { device_name: "name of device"
          kernel_name: "name of kernel"
          problem_size: (int, int, int)
          tune_params_keys: list
          tune_params:
          cache: {
            "x1,x2,..xN": {"block_size_x": x1, ..., time=0.234342},
            "y1,y2,..yN": {"block_size_x": y1, ..., time=0.134233},
          }
        }


    The last two closing brackets are not required, and everything
    should work as expected if these are missing. This is to allow to continue
    from an earlier (abruptly ended) tuning session.

    """
    # caching only works correctly if tunable_parameters are stored in a dictionary
    if not isinstance(tuning_options.tune_params, dict):
        raise ValueError("Caching only works correctly when tunable parameters are stored in a dictionary")

    # if file does not exist, create new cache
    if not os.path.isfile(cache):
        if tuning_options.simulation_mode:
            raise ValueError(f"Simulation mode requires an existing cachefile: file {cache} does not exist")

        c = dict()
        c["device_name"] = runner.dev.name
        c["kernel_name"] = kernel_options.kernel_name
        c["problem_size"] = kernel_options.problem_size if not callable(kernel_options.problem_size) else "callable"
        c["tune_params_keys"] = list(tuning_options.tune_params.keys())
        c["tune_params"] = tuning_options.tune_params
        c["objective"] = tuning_options.objective
        c["cache"] = {}

        contents = json.dumps(c, cls=NpEncoder, indent="")[:-3]  # except the last "}\n}"

        # write the header to the cachefile
        with open(cache, "w") as cachefile:
            cachefile.write(contents)

        tuning_options.cachefile = cache
        tuning_options.cache = {}

    # if file exists
    else:
        cached_data = read_cache(cache, open_cache=not tuning_options.simulation_mode)

        # if in simulation mode, use the device name from the cache file as the runner device name
        if runner.simulation_mode:
            runner.dev.name = cached_data["device_name"]

        # check if it is safe to continue tuning from this cache
        if cached_data["device_name"] != runner.dev.name:
            raise ValueError(
                f"Cannot load cache which contains results for different device (cache: {cached_data['device_name']}, actual: {runner.dev.name})"
            )
        if cached_data["kernel_name"] != kernel_options.kernel_name:
            raise ValueError(
                f"Cannot load cache which contains results for different kernel (cache: {cached_data['kernel_name']}, actual: {kernel_options.kernel_name})"
            )
        if "problem_size" in cached_data and not callable(kernel_options.problem_size):
            check_matching_problem_size(cached_data["problem_size"], kernel_options.problem_size)
        if cached_data["tune_params_keys"] != list(tuning_options.tune_params.keys()):
            if all(key in tuning_options.tune_params for key in cached_data["tune_params_keys"]):
                raise ValueError(
                    f"All tunable parameters are present, but the order is wrong. \
                    This is not possible because the order must be preserved to lookup the correct configuration in the cache. \
                    Cache has order: {cached_data['tune_params_keys']}, tuning_options has: {list(tuning_options.tune_params.keys())}"
                )
            raise ValueError(
                f"Cannot load cache which contains results obtained with different tunable parameters. \
                Cache at '{cache}' has: {cached_data['tune_params_keys']}, tuning_options has: {list(tuning_options.tune_params.keys())}"
            )

        tuning_options.cachefile = cache
        tuning_options.cache = cached_data["cache"]


def correct_open_cache(cache, open_cache=True):
    """If cache file was not properly closed, pretend it was properly closed."""
    with open(cache, "r") as cachefile:
        filestr = cachefile.read().strip()

    # if file was not properly closed, pretend it was properly closed
    if len(filestr) > 0 and filestr[-3:] not in ["}\n}", "}}}"]:
        # remove the trailing comma if any, and append closing brackets
        if filestr[-1] == ",":
            filestr = filestr[:-1]
        if len(filestr) > 0:
            filestr = filestr + "}\n}"
    else:
        if open_cache:
            # if it was properly closed, open it for appending new entries
            with open(cache, "w") as cachefile:
                cachefile.write(filestr[:-3] + ",")

    return filestr


def read_cache(cache, open_cache=True):
    """Read the cachefile into a dictionary, if open_cache=True prepare the cachefile for appending."""
    filestr = correct_open_cache(cache, open_cache)

    error_configs = {
        "InvalidConfig": InvalidConfig(),
        "CompilationFailedConfig": CompilationFailedConfig(),
        "RuntimeFailedConfig": RuntimeFailedConfig(),
    }

    # replace strings with ErrorConfig instances
    cache_data = json.loads(filestr)
    for element in cache_data["cache"].values():
        for k, v in element.items():
            if isinstance(v, str) and v in error_configs:
                element[k] = error_configs[v]

    return cache_data


def close_cache(cache):
    if not os.path.isfile(cache):
        raise ValueError("close_cache expects cache file to exist")

    with open(cache, "r") as fh:
        contents = fh.read()

    # close to file to make sure it can be read by JSON parsers
    if contents[-1] == ",":
        with open(cache, "w") as fh:
            fh.write(contents[:-1] + "}\n}")


def store_cache(key, params, tuning_options):
    """Stores a new entry (key, params) to the cachefile."""
    # logging.debug('store_cache called, cache=%s, cachefile=%s' % (tuning_options.cache, tuning_options.cachefile))
    if isinstance(tuning_options.cache, dict):
        if key not in tuning_options.cache:
            tuning_options.cache[key] = params

            # Convert ErrorConfig objects to string, wanted to do this inside the JSONconverter but couldn't get it to work
            output_params = params.copy()
            for k, v in output_params.items():
                if isinstance(v, ErrorConfig):
                    output_params[k] = str(v)

            if tuning_options.cachefile:
                with open(tuning_options.cachefile, "a") as cachefile:
                    cachefile.write("\n" + json.dumps({key: output_params}, cls=NpEncoder)[1:-1] + ",")

def dump_cache(obj: str, tuning_options):
    """Dumps a string in the cache, this omits the several checks of store_cache() to speed up the process - with great power comes great responsibility!"""
    if isinstance(tuning_options.cache, dict) and tuning_options.cachefile:
        with open(tuning_options.cachefile, "a") as cachefile:
            cachefile.write(obj)
