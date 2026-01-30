"""This module contains the interface for runners."""
from __future__ import print_function

from abc import ABC, abstractmethod


class Runner(ABC):
    """Base class for kernel_tuner runners"""

    @abstractmethod
    def __init__(
        self, kernel_source, kernel_options, device_options, iterations, observers
    ):
        pass

    def shutdown(self):
        pass

    def available_parallelism(self):
        """ Gives an indication of how many jobs this runner can execute in parallel. """
        return 1

    @abstractmethod
    def get_device_info(self):
        pass

    @abstractmethod
    def get_environment(self, tuning_options):
        pass

    @abstractmethod
    def run(self, parameter_space, tuning_options):
        pass
