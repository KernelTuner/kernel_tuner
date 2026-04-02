import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver

# check if powersensor is installed
try:
    import powersensor
except ImportError:
    powersensor = None


class PowerSensorObserver(BenchmarkObserver):
    """Observer that an external PowerSensor2 device to accurately measure power

    Requires PowerSensor3 hardware and powersensor Python bindings.

    :param observables: A list of string, containing any of "ps_energy" or "ps_power".
        To measure energy in Joules or power consumption in Watt.
        If not passed "ps_energy" is used to report energy consumption of kernels in Joules.
    :type observables: list

    :param device: A string with the path to the PowerSensor2 device, default "/dev/ttyACM0".
    :type device: string

    """

    def __init__(self, observables=None, device=None):
        if not powersensor:
            raise ImportError("could not import powersensor")

        supported = ["ps_energy", "ps_power"]
        for obs in observables:
            if not obs in supported:
                raise ValueError(f"Observable {obs} not in supported: {supported}")
        self.observables = observables or ["ps_energy"]

        device = device or "/dev/ttyACM0"
        self.ps = powersensor.PowerSensor(device)

        self.begin_state = None
        self.results = {key: [] for key in self.observables}

    def after_start(self):
        self.begin_state = self.ps.read()

    def after_finish(self):
        end_state = self.ps.read()
        if "ps_energy" in self.observables:
            ps_measured_e = powersensor.Joules(
                self.begin_state, end_state, -1
            )  # Joules
            self.results["ps_energy"].append(ps_measured_e)
        if "ps_power" in self.observables:
            ps_measured_t = ((end_state.time_at_read - self.begin_state.time_at_read).microseconds / 1e6)    # Seconds

            self.results["ps_power"].append(ps_measured_e / ps_measured_t)  # Watt

    def get_results(self):
        averages = {key: np.average(values) for key, values in self.results.items()}
        self.results = {key: [] for key in self.observables}
        return averages
