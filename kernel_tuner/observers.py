from abc import ABC, abstractmethod

#check if power_sensor is installed
try:
    import power_sensor
except ImportError:
    power_sensor = None

class BenchmarkObserver(ABC):
    """Base class for Benchmark Observers"""

    def before_start(self):
        pass

    def after_start(self):
        pass

    def during(self):
        pass

    def after_finish(self):
        pass

    @abstractmethod
    def get_results(self):
        pass


class PowerSensorObserver():
    """Observer that an external PowerSensor2 device to accurately measure power"""

    def __init__(self, observables=None, device=None):
        if not power_sensor:
            raise ImportError("could not import power_sensor")

        self.observables = observables or ["energy"]
        device = device or "/dev/ttyACM0"
        self.ps = power_sensor.PowerSensor(device)

        self.begin_state = None
        self.results = dict()

    def after_start(self):
        self.begin_state = self.ps.read()

    def after_finish(self):
        end_state = self.ps.read()
        if "energy" in self.observables:
            ps_measured_e = power_sensor.Joules(self.begin_state, end_state, -1) # Joules
            self.results["energy"].append(ps_measured_e)
        if "power" in self.observables:
            ps_measured_t = end_state.time_at_read - self.begin_state.time_at_read # seconds
            self.results["power"].append(ps_measured_e / ps_measured_t) # Watt

    def get_results(self):
        self.averages = {key: np.average(values) for key, values in self.results()}
        self.results = dict()
        return averages
