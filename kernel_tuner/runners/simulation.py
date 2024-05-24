"""The simulation runner for sequentially tuning the parameter space based on cached data."""
import logging
from collections import namedtuple
from time import perf_counter

from kernel_tuner import util
from kernel_tuner.runners.runner import Runner

_SimulationDevice = namedtuple("_SimulationDevice", ["max_threads", "env", "quiet"])


class SimulationDevice(_SimulationDevice):
    """Simulated device used by simulation runner."""

    @property
    def name(self):
        return self.env['device_name']

    @name.setter
    def name(self, value):
        self.env['device_name'] = value
        if not self.quiet:
            print("Simulating: " + value)

    def get_environment(self):
        return self.env


class SimulationRunner(Runner):
    """SimulationRunner is used for tuning with a single process/thread."""

    def __init__(self, kernel_source, kernel_options, device_options, iterations, observers):
        """Instantiate the SimulationRunner.

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
        self.quiet = device_options.quiet
        self.dev = SimulationDevice(1024, dict(device_name="Simulation"), self.quiet)

        self.kernel_source = kernel_source
        self.simulation_mode = True
        self.kernel_options = kernel_options

        self.start_time = perf_counter()
        self.last_strategy_start_time = self.start_time
        self.last_strategy_time = 0
        self.units = {}

    def get_environment(self, tuning_options):
        env = self.dev.get_environment()
        env["simulation"] = True
        env["simulated_time"] = tuning_options.simulated_time
        return env

    def run(self, parameter_space, tuning_options):
        """Iterate through the entire parameter space using a single Python process.

        :param parameter_space: The parameter space as an iterable.
        :type parameter_space: iterable

        :param tuning_options: A dictionary with all options regarding the tuning
            process.
        :type tuning_options: kernel_tuner.iterface.Options

        :returns: A list of dictionaries for executed kernel configurations and their
            execution times.
        :rtype: dict()
        """
        logging.debug('simulation runner started for ' + self.kernel_options.kernel_name)

        results = []

        # iterate over parameter space
        for element in parameter_space:

            # check if element is in the cache
            x_int = ",".join([str(i) for i in element])
            if tuning_options.cache and x_int in tuning_options.cache:
                result = tuning_options.cache[x_int].copy()

                # Simulate behavior of sequential runner that when a configuration is
                # served from the cache by the sequential runner, the compile_time,
                # verification_time, and benchmark_time are set to 0.
                # This step is only performed in the simulation runner when a configuration
                # is served from the cache beyond the first timel. That is, when the
                # configuration is already counted towards the unique_results.
                # It is the responsibility of cost_func to add configs to unique_results.
                if x_int in tuning_options.unique_results:

                    result['compile_time'] = 0
                    result['verification_time'] = 0
                    result['benchmark_time'] = 0

                else:
                    # configuration is evaluated for the first time, print to the console
                    util.print_config_output(tuning_options.tune_params, result, self.quiet, tuning_options.metrics, self.units)

                # Everything but the strategy time and framework time are simulated,
                # self.last_strategy_time is set by cost_func
                result['strategy_time'] = self.last_strategy_time

                try:
                    simulated_time = result['compile_time'] + result['verification_time'] + result['benchmark_time']
                    tuning_options.simulated_time += simulated_time
                except KeyError:
                    if "time_limit" in tuning_options:
                        raise RuntimeError(
                            "Cannot use simulation mode with a time limit on a cache file that does not have full compile, verification, and benchmark timings on all configurations"
                        )

                total_time = 1000 * (perf_counter() - self.start_time)
                self.start_time = perf_counter()
                result['framework_time'] = total_time - self.last_strategy_time

                results.append(result)
                continue

            # if the element is not in the cache, raise an error
            check = util.check_restrictions(tuning_options.restrictions, dict(zip(tuning_options['tune_params'].keys(), element)), True)
            err_string = f"kernel configuration {element} not in cache, does {'' if check else 'not '}pass extra restriction check ({check})"
            logging.debug(err_string)
            raise ValueError(f"{err_string} - in simulation mode, all configurations must be present in the cache")

        return results
