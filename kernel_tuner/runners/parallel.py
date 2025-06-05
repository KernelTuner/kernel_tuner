"""A specialized runner that tunes in parallel the parameter space."""
import logging
from time import perf_counter
from datetime import datetime, timezone

from ray import remote, get, put

from kernel_tuner.runners.runner import Runner
from kernel_tuner.core import DeviceInterface
from kernel_tuner.util import ErrorConfig, print_config_output, process_metrics, store_cache


class ParallelRunnerState:
    """This class represents the state of a parallel tuning run."""

    def __init__(self, observers, iterations):
        self.device_options = None
        self.quiet = False
        self.kernel_source = None
        self.warmed_up = False
        self.simulation_mode = False
        self.start_time = None
        self.last_strategy_start_time = None
        self.last_strategy_time = 0
        self.kernel_options = None
        self.observers = observers
        self.iterations = iterations


@remote
def parallel_run(task_id: int, state: ParallelRunnerState, parameter_space, tuning_options):
    dev = DeviceInterface(
        state.kernel_source, iterations=state.iterations, observers=state.observers, **state.device_options
    )
    # move data to the GPU
    gpu_args = dev.ready_argument_list(state.kernel_options.arguments)
    # iterate over parameter space
    results = []
    elements_per_task = len(parameter_space) / tuning_options.parallel_runner
    first_element = task_id * elements_per_task
    last_element = (
        (task_id + 1) * elements_per_task if task_id + 1 < tuning_options.parallel_runner else len(parameter_space)
    )
    for element in parameter_space[first_element:last_element]:
        params = dict(zip(tuning_options.tune_params.keys(), element))

        result = None
        warmup_time = 0

        # check if configuration is in the cache
        x_int = ",".join([str(i) for i in element])
        if tuning_options.cache and x_int in tuning_options.cache:
            params.update(tuning_options.cache[x_int])
            params["compile_time"] = 0
            params["verification_time"] = 0
            params["benchmark_time"] = 0
        else:
            # attempt to warm up the GPU by running the first config in the parameter space and ignoring the result
            if not state.warmed_up:
                warmup_time = perf_counter()
                dev.compile_and_benchmark(state.kernel_source, gpu_args, params, state.kernel_options, tuning_options)
                state.warmed_up = True
                warmup_time = 1e3 * (perf_counter() - warmup_time)

            result = dev.compile_and_benchmark(
                state.kernel_source, gpu_args, params, state.kernel_options, tuning_options
            )

            params.update(result)

            if tuning_options.objective in result and isinstance(result[tuning_options.objective], ErrorConfig):
                logging.debug("kernel configuration was skipped silently due to compile or runtime failure")

        # only compute metrics on configs that have not errored
        if tuning_options.metrics and not isinstance(params.get(tuning_options.objective), ErrorConfig):
            params = process_metrics(params, tuning_options.metrics)

        # get the framework time by estimating based on other times
        total_time = 1000 * ((perf_counter() - state.start_time) - warmup_time)
        params["strategy_time"] = state.last_strategy_time
        params["framework_time"] = max(
            total_time
            - (
                params["compile_time"]
                + params["verification_time"]
                + params["benchmark_time"]
                + params["strategy_time"]
            ),
            0,
        )
        params["timestamp"] = str(datetime.now(timezone.utc))
        state.start_time = perf_counter()

        if result:
            # print configuration to the console
            print_config_output(tuning_options.tune_params, params, state.quiet, tuning_options.metrics, dev.units)

            # add configuration to cache
            store_cache(x_int, params, tuning_options)

        # all visited configurations are added to results to provide a trace for optimization strategies
        results.append(params)

    return results


class ParallelRunner(Runner):
    """ParallelRunner is used to distribute configurations across multiple nodes."""

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers):
        """Instantiate the ParallelRunner.

        :param kernel_source: The kernel source
        :type kernel_source: kernel_tuner.core.KernelSource

        :param kernel_options: A dictionary with all options for the kernel.
        :type kernel_options: kernel_tuner.interface.Options

        :param device_options: A dictionary with all options for the device
            on which the kernel should be tuned.
        :type device_options: kernel_tuner.interface.Options

        :param iterations: The number of iterations used for benchmarking
            each kernel instance.
        :type iterations: int
        """
        self.state = ParallelRunnerState(observers, iterations)
        self.state.quiet = device_options.quiet
        self.state.kernel_source = kernel_source
        self.state.warmed_up = False
        self.state.simulation_mode = False
        self.state.start_time = perf_counter()
        self.state.last_strategy_start_time = self.state.start_time
        self.state.last_strategy_time = 0
        self.state.kernel_options = kernel_options
        # define a dummy device interface
        self.dev = DeviceInterface(kernel_source)

    def get_environment(self, tuning_options):
        # dummy environment
        return self.dev.get_environment()

    def run(self, parameter_space, tuning_options):
        """Iterate through the entire parameter space using a single Python process.

        :param parameter_space: The parameter space as an iterable.
        :type parameter_space: iterable

        :param tuning_options: A dictionary with all options regarding the tuning process.
        :type tuning_options: kernel_tuner.interface.Options

        :returns: A list of dictionaries for executed kernel configurations and their execution times.
        :rtype: dict()
        """
        # given the parameter_space, distribute it over Ray tasks
        logging.debug("parallel runner started for " + self.state.kernel_options.kernel_name)

        results = []
        tasks = []
        parameter_space_ref = put(parameter_space)
        state_ref = put(self.state)
        tuning_options_ref = put(tuning_options)
        for task_id in range(0, tuning_options.parallel_runner):
            tasks.append(parallel_run.remote(task_id, state_ref, parameter_space_ref, tuning_options_ref))
        for task in tasks:
            results.append(get(task))

        return results
