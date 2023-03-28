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

    @abstractmethod
    def get_environment(self):
        pass

    @abstractmethod
    def run(self, parameter_space, tuning_options):
        pass
