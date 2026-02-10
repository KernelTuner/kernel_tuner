"""This module contains the interface for runners."""
from __future__ import print_function

from abc import ABC, abstractmethod

from kernel_tuner.util import Timer


class Runner(ABC):
    """Base class for kernel_tuner runners"""

    def __init__(self):
        self.timer = Timer()
        self.accumulated_strategy_time = 0

    def add_strategy_time(self, seconds):
        self.accumulated_strategy_time += seconds

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
