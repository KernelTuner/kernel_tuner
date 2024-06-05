import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver

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

    """

    def __init__(self, observable=None):
        if not pmt:
            raise ImportError("could not import pmt")

        # User specifices a dictonary of platforms and corresponding device
        if type(observable) is dict:
            pass
        elif type(observable) is list:
            # user specifies a list of platforms as observable
            observable = dict([(obs, 0) for obs in observable])
        else:
            # User specifices a string (single platform) as observable
            observable = {observable: None}
        supported = ["arduino", "jetson", "likwid", "nvml", "rapl", "rocm", "xilinx"]
        for obs in observable.keys():
            if not obs in supported:
                raise ValueError(f"Observable {obs} not in supported: {supported}")

        self.pms = [pmt.get_pmt(obs[0], obs[1]) for obs in observable.items()]
        self.pm_names = list(observable.keys())

        self.begin_states = [None] * len(self.pms)
        self.initialize_results(self.pm_names)

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
