from abc import ABC, abstractmethod
import time
import numpy as np

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
        """get_results should return a dict with results that adds to the benchmarking data

        get_results is called only once per benchmarking of a single kernel configuration and
        generally returns averaged values over multiple iterations.
        """
        pass


class IterationObserver(BenchmarkObserver):
    pass


class ContinuousObserver(BenchmarkObserver):
    """Generic observer that measures power while and continuous benchmarking.

        To support continuous benchmarking an Observer should support:
        a .read_power() method, which the ContinuousObserver can call to read power in Watt
    """
    def __init__(self, name, observables, parent, continuous_duration=1):
        self.parent = parent
        self.name = name

        supported = [self.name + "_power", self.name + "_energy", "power_readings"]
        for obs in observables:
            if obs not in supported:
                raise ValueError(f"Observable {obs} not in supported: {supported}")
        self.observables = observables

        # duration in seconds
        self.continuous_duration = continuous_duration

        self.power = 0
        self.energy = 0
        self.power_readings = []
        self.t0 = 0

        # results from the last iteration-based benchmark
        # these are set by the benchmarking function of Kernel Tuner before
        # the continuous observer is called.
        self.results = None

    def before_start(self):
        self.parent.before_start()
        self.power = 0
        self.energy = 0
        self.power_readings = []

    def after_start(self):
        self.parent.after_start()
        self.t0 = time.perf_counter()

    def during(self):
        self.parent.during()
        power_usage = self.parent.read_power()
        timestamp = time.perf_counter() - self.t0
        # only store the result if we get a new measurement from the GPU
        if len(self.power_readings) == 0 or (
            self.power_readings[-1][1] != power_usage
            or timestamp - self.power_readings[-1][0] > 0.01
        ):
            self.power_readings.append([timestamp, power_usage])

    def after_finish(self):
        self.parent.after_finish()
        # safeguard in case we have no measurements, perhaps the kernel was too short to measure anything
        if not self.power_readings:
            return

        # convert to seconds from milliseconds
        execution_time = self.results["time"] / 1e3
        self.power = np.median([d[1] for d in self.power_readings])
        self.energy = self.power * execution_time

    def get_results(self):
        results = self.parent.get_results()
        keys = list(results.keys())
        for key in keys:
            results["pwr_" + key] = results.pop(key)
        if self.name + "_power" in self.observables:
            results[self.name + "_power"] = self.power
        if self.name + "_energy" in self.observables:
            results[self.name + "_energy"] = self.energy
        if "power_readings" in self.observables:
            results["power_readings"] = self.power_readings
        return results

class OutputObserver(BenchmarkObserver):
    """Observer that can verify or measure something about the output produced by a kernel."""

    @abstractmethod
    def process_output(self, answer, output):
        """method will be called once before benchmarking of a single kernel configuration. The arguments
        provided are the `answer` as passed `tune_kernel` and the `output` produced by the kernel
        """
        pass

class PrologueObserver(BenchmarkObserver):
    """Observer that measures something in a seperate kernel invocation prior to the normal benchmark."""

    @abstractmethod
    def before_start(self):
        """prologue start is called before the kernel starts"""
        pass

    @abstractmethod
    def after_finish(self):
        """prologue finish is called after the kernel has finished execution"""
        pass
