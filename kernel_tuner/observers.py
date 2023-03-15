from abc import ABC, abstractmethod

import numpy as np

#check if power_sensor is installed
try:
    import power_sensor
except ImportError:
    power_sensor = None

class BenchmarkObserver(ABC):
    """Base class for Benchmark Observers"""

    def register_device(self, dev):
        """Sets self.dev, for inspection by the observer at various points during benchmarking"""
        self.dev = dev

    def register_configuration(self, params):
        """Called once before benchmarking of a single kernel configuration. The `params` argument is a `dict`
        that stores the configuration parameters."""
        pass

    def before_start(self):
        """before start is called every iteration before the kernel starts"""
        pass

    def after_start(self):
        """after start is called every iteration directly after the kernel was launched"""
        pass

    def during(self):
        """during is called as often as possible while the kernel is running"""
        pass

    def after_finish(self):
        """after finish is called once every iteration after the kernel has finished execution"""
        pass

    @abstractmethod
    def get_results(self):
        """ get_results should return a dict with results that adds to the benchmarking data

            get_results is called only once per benchmarking of a single kernel configuration and
            generally returns averaged values over multiple iterations.
        """
        pass


class AccuracyObserver(BenchmarkObserver):
    """ Observer that can verify or measure the accuracy of the output produced by a kernel. """

    @abstractmethod
    def process_kernel_output(self, answer, output):
        """method will be called once before benchmarking of a single kernel configuration. The arguments
        provided are the `answer` as passed `tune_kernel` and the `output` produced by the kernel"""
        pass


class IterationObserver(BenchmarkObserver):
    pass


class ContinuousObserver(BenchmarkObserver):
    pass


class PowerSensorObserver(BenchmarkObserver):
    """ Observer that an external PowerSensor2 device uses to accurately measure power

    Requires PowerSensor2 hardware and power_sensor Python bindings.

    :param observables: A list of string, containing any of "ps_energy" or "ps_power".
        To measure energy in Joules or power consumption in Watt.
        If not passed "ps_energy" is used to report energy consumption of kernels in Joules.
    :type observables: list

    :param device: A string with the path to the PowerSensor2 device, default "/dev/ttyACM0".
    :type device: string

    """

    def __init__(self, observables=None, device=None):
        if not power_sensor:
            raise ImportError("could not import power_sensor")

        supported = ["ps_energy", "ps_power"]
        for obs in observables:
            if not obs in supported:
                raise ValueError(f"Observable {obs} not in supported: {supported}")
        self.observables = observables or ["ps_energy"]

        device = device or "/dev/ttyACM0"
        self.ps = power_sensor.PowerSensor(device)

        self.begin_state = None
        self.results = {"ps_energy": [], "ps_power": []}

    def after_start(self):
        self.begin_state = self.ps.read()

    def after_finish(self):
        end_state = self.ps.read()
        if "ps_energy" in self.observables:
            ps_measured_e = power_sensor.Joules(self.begin_state, end_state, -1) # Joules
            self.results["ps_energy"].append(ps_measured_e)
        if "ps_power" in self.observables:
            ps_measured_t = end_state.time_at_read - self.begin_state.time_at_read # seconds
            self.results["ps_power"].append(ps_measured_e / ps_measured_t) # Watt

    def get_results(self):
        averages = {key: np.average(values) for key, values in self.results.items()}
        self.results = {"ps_energy": [], "ps_power": []}
        return averages
