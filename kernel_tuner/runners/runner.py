"""This module contains the interface for runners."""
from __future__ import print_function

from abc import ABC, abstractmethod
import json


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
        if tuning_options.tracefile_path is not None and tuning_options.tracefile_path != '':
            self.write_to_trace(parameter_space, tuning_options)

    def write_to_trace(self, configurations: list[tuple], tuning_options):
        """Function to write a validated set of configurations to a tracefile if it is defined"""
        tracefile_path: str = tuning_options.tracefile_path
        with open(tracefile_path, 'w', encoding='utf-8') as fp:
            json.dump(configurations, fp)
