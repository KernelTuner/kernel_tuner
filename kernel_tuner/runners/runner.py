"""This module contains the interface for runners."""
from abc import ABC, abstractmethod
from time import perf_counter


class Runner(ABC):
    """Base class for kernel_tuner runners"""

    @abstractmethod
    def __init__(
        self, kernel_source, kernel_options, device_options, iterations, observers
    ):
        """Instantiate a Runner (either sequential or simulation).

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
        self.kernel_source = kernel_source
        self.kernel_options = kernel_options
        self.start_time = perf_counter()
        self.last_strategy_start_time = self.start_time
        self.last_strategy_time = 0

    @abstractmethod
    def get_environment(self):
        pass

    @abstractmethod
    def run(self, parameter_space, tuning_options) -> list[dict]:
        """Iterate through the entire parameter space using a single Python process.

        :param parameter_space: The parameter space as an iterable.
        :type parameter_space: iterable

        :param tuning_options: A dictionary with all options regarding the tuning
            process.
        :type tuning_options: kernel_tuner.iterface.Options

        :returns: A list of dictionaries for executed kernel configurations and their
            execution times.
        :rtype: list(dict())
        """
        pass
