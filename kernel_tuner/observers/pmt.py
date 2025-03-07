import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver, ContinuousObserver

# check if pmt is installed
try:
    import pmt
except ImportError:
    pmt = None


class PMTObserver(BenchmarkObserver):
    """Observer that uses the PMT library to measure power

    :param observables:
        One of:

        - A string specifying a single power meter to use
        - A list of string, specifying one or more power meters to use
        - A dictionary, specifying one or more power meters to use,
          including the device identifier. For arduino this should be for
          instance "/dev/ttyACM0". For nvml, it should correspond to the GPU
          id (e.g. '0', or '1'). For some sensors (such as rapl) the device
          id is not used, it should be 'None' in those cases.

        This observer will report "<platform>_energy>" and "<platform>_power" for
        all specified platforms.

    :type observables: string,list/dictionary


    :param use_continuous_observer:
        Boolean to control whether or not to measure power/energy using
        Kernel Tuner's continuous benchmarking mode. This improves measurement
        accuracy when using internal power sensors, such as NVML or ROCM,
        which have limited sampling frequency and might return averages
        instead of instantaneous power readings. Default value: False.

    :type use_continuous_observer: boolean


    :param continuous_duration:
        Number of seconds to measure continuously for.

    :type continuous_duration: scalar

    """

    def __init__(self, observable=None, use_continuous_observer=False, continuous_duration=1):
        if not pmt:
            raise ImportError("could not import pmt")

        # User specifices a dictonary of platforms and corresponding device
        if type(observable) is dict:
            pass
        elif type(observable) is list:
            # user specifies a list of platforms as observable, optionally with an argument
            observable = dict([obs if isinstance(obs, tuple) else (obs, None) for obs in observable])
        else:
            # User specifices a string (single platform) as observable
            observable = {observable: None}
        supported = ["powersensor2", "powersensor3", "nvidia", "likwid", "rapl", "rocm", "xilinx"]
        for obs in observable.keys():
            if not obs in supported:
                raise ValueError(f"Observable {obs} not in supported: {supported}")

        self.pms = [pmt.create(obs[0], obs[1]) for obs in observable.items()]
        self.pm_names = list(observable.keys())

        self.begin_states = [None] * len(self.pms)
        self.initialize_results(self.pm_names)

        if use_continuous_observer:
            self.continuous_observer = PMTContinuousObserver("pmt", [], self, continuous_duration=continuous_duration)

    def initialize_results(self, pm_names):
        self.results = dict()
        for pm_name in pm_names:
            energy_result_name = f"{pm_name}_energy"
            power_result_name = f"{pm_name}_power"
            self.results[energy_result_name] = []
            self.results[power_result_name] = []

    def after_start(self):
        self.begin_states = [pm.read() for pm in self.pms]

    def after_finish(self):
        end_states = [pm.read() for pm in self.pms]
        for i in range(len(self.pms)):
            begin_state = self.begin_states[i]
            end_state = end_states[i]
            measured_energy = pmt.joules(begin_state, end_state)
            measured_power = pmt.watts(begin_state, end_state)
            pm_name = self.pm_names[i]
            energy_result_name = f"{pm_name}_energy"
            power_result_name = f"{pm_name}_power"
            self.results[energy_result_name].append(measured_energy)
            self.results[power_result_name].append(measured_power)

    def get_results(self):
        averages = {key: np.average(values) for key, values in self.results.items()}
        self.initialize_results(self.pm_names)
        return averages


class PMTContinuousObserver(ContinuousObserver):
    """Generic observer that measures power while and continuous benchmarking.

        To support continuous benchmarking an Observer should support:
        a .read_power() method, which the ContinuousObserver can call to read power in Watt
    """
    def before_start(self):
        """ Override default method in ContinuousObserver """
        pass

    def after_start(self):
        self.parent.after_start()

    def during(self):
        """ Override default method in ContinuousObserver """
        pass

    def after_finish(self):
        self.parent.after_finish()

    def get_results(self):
        average_kernel_execution_time_ms = self.results["time"]
        averages = self.parent.get_results()

        # correct energy measurement, because current _energy number is collected over the entire duration
        # we estimate energy as the average power over the continuous duration times the kernel execution time
        for pm_name in self.parent.pm_names:
            energy_result_name = f"{pm_name}_energy"
            power_result_name = f"{pm_name}_power"
            averages[energy_result_name] = averages[power_result_name] * (average_kernel_execution_time_ms / 1e3)

        return averages
