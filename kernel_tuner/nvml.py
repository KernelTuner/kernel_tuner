import time
import numpy as np

from kernel_tuner.observers import BenchmarkObserver

try:
    import pynvml
except ImportError:
    pynvml = None

class nvml():
    """Class that gathers the NVML functionality for one device"""

    def __init__(self, device_id=0):
        """Create object to control device using NVML"""

        pynvml.nvmlInit()
        self.dev = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        try:
            self._pwr_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.dev)
            self.pwr_constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.dev)
            self.pwr_limit_default = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self.dev)
        except pynvml.NVMLError_NotSupported:
            self._pwr_limit = None
            self.pwr_constraints = [1, 0] # inverted range to make all range checks fail

        try:
            self._persistence_mode = pynvml.nvmlDeviceGetPersistenceMode(self.dev)
        except pynvml.NVMLError_NotSupported:
            self._persistence_mode = None

        try:
            self._auto_boost = pynvml.nvmlDeviceGetAutoBoostedClocksEnabled(self.dev)[0]  # returns [isEnabled, isDefaultEnabled]
        except pynvml.NVMLError_NotSupported:
            self._auto_boost = None

        try:
            self.gr_clock_default = pynvml.nvmlDeviceGetDefaultApplicationsClock(self.dev, pynvml.NVML_CLOCK_GRAPHICS)
            self.sm_clock_default = pynvml.nvmlDeviceGetDefaultApplicationsClock(self.dev, pynvml.NVML_CLOCK_SM)
            self.mem_clock_default = pynvml.nvmlDeviceGetDefaultApplicationsClock(self.dev, pynvml.NVML_CLOCK_MEM)

            self.supported_mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.dev)

            self._gr_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_GRAPHICS)
            self._sm_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_SM)
            self._mem_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_MEM)

            #gather the supported gr clocks for each supported mem clock into a dict
            self.supported_gr_clocks = dict()
            for mem_clock in self.supported_mem_clocks:
                supported_gr_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(self.dev, mem_clock)
                self.supported_gr_clocks[mem_clock] = supported_gr_clocks
        except pynvml.NVMLError_NotSupported:
            self.gr_clock_default = None
            self.sm_clock_default = None
            self.mem_clock_default = None
            self.supported_mem_clocks = []
            self.supported_gr_clocks = dict()

    def __del__(self):
        #try to restore to defaults
        if self._pwr_limit != None and self.pwr_limit != self.pwr_limit_default:
            self.pwr_limit = self.pwr_limit_default
        if self.gr_clock_default != None:
            gr_app_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_GRAPHICS)
            mem_app_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_MEM)
            if gr_app_clock != self.gr_clock_default or mem_app_clock != self.mem_clock_default:
                self.set_clocks(self.mem_clock_default, self.gr_clock_default)

    @property
    def pwr_state(self):
        """Get the Device current Power State"""
        return pynvml.nvmlDeviceGetPowerState(self.dev)

    @property
    def pwr_limit(self):
        """Control the power limit (may require permission), check pwr_constraints for the allowed range"""
        return self._pwr_limit

    @pwr_limit.setter
    def pwr_limit(self, new_limit):
        if not self.pwr_constraints[0] <= new_limit <= self.pwr_constraints[1]:
            raise ValueError(f"Power limit {new_limit} out of range [{self.pwr_constraints[0]}, {self.pwr_constraints[1]}]")
        pynvml.nvmlDeviceSetPowerManagementLimit(self.dev, new_limit)
        self._pwr_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.dev)

    @property
    def persistence_mode(self):
        """Control persistence mode (may require permission), 0 for disabled, 1 for enabled"""
        return self._persistence_mode

    @persistence_mode.setter
    def persistence_mode(self, new_mode):
        if not new_mode in [0, 1]:
            raise ValueError("Illegal value for persistence mode, should be either 0 or 1")
        pynvml.nvmlDeviceSetPersistenceMode(self.dev, new_mode)
        self._persistence_mode = pynvml.nvmlDeviceGetPersistenceMode(self.dev)

    def set_clocks(self, mem_clock, gr_clock):
        """Set the memory and graphics clock for this device (may require permission)"""
        if not mem_clock in self.supported_mem_clocks:
            raise ValueError("Illegal value for memory clock")
        if not gr_clock in self.supported_gr_clocks[mem_clock]:
            raise ValueError("Graphics clock incompatible with memory clock")
        pynvml.nvmlDeviceSetApplicationsClocks(self.dev, mem_clock, gr_clock)
        self._gr_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_GRAPHICS)
        self._sm_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_SM)
        self._mem_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_MEM)

    @property
    def gr_clock(self):
        """Control the graphics clock (may require permission), only values compatible with the memory clock can be set directly"""
        return pynvml.nvmlDeviceGetClockInfo(self.dev, pynvml.NVML_CLOCK_GRAPHICS)

    @gr_clock.setter
    def gr_clock(self, new_clock):
        self.set_clocks(self._mem_clock, new_clock)

    @property
    def mem_clock(self):
        """Control the graphics clock (may require permission), only values compatible with the graphics clock can be set directly"""
        return pynvml.nvmlDeviceGetClockInfo(self.dev, pynvml.NVML_CLOCK_MEM)

    @mem_clock.setter
    def mem_clock(self, new_clock):
        self.set_clocks(new_clock, self._gr_clock)

    @property
    def temperature(self):
        """Get the GPU temperature"""
        return pynvml.nvmlDeviceGetTemperature(self.dev, pynvml.NVML_TEMPERATURE_GPU)

    @property
    def auto_boost(self):
        """Control the auto boost setting (may require permission), 0 for disable, 1 for enabled"""
        return self._auto_boost

    @auto_boost.setter
    def auto_boost(self, setting):
        #might need to use pynvml.NVML_FEATURE_DISABLED or pynvml.NVML_FEATURE_ENABLED instead of 0 or 1
        if not setting in [0, 1]:
            raise ValueError("Illegal value for auto boost enabled, should be either 0 or 1")
        pynvml.nvmlDeviceSetAutoBoostedClocksEnabled(self.dev, setting)
        self._auto_boost = pynvml.nvmlDeviceGetAutoBoostedClocksEnabled(self.dev)[0]

    def pwr_usage(self):
        """Return current power usage in milliwatts"""
        return pynvml.nvmlDeviceGetPowerUsage(self.dev)


class NVMLObserver(BenchmarkObserver):
    """ Observer that measures time using CUDA events during benchmarking """
    def __init__(self, observables, device=0):
        self.nvml = nvml(device)

        supported = ["power_readings", "nvml_power", "nvml_energy", "core_freq", "mem_freq", "temperature"]
        for obs in observables:
            if not obs in supported:
                raise ValueError(f"Observable {obs} not in supported: {supported}")
        self.observables = observables

        self.measure_power = False
        needs_power = ["power_readings", "nvml_power", "nvml_energy"]
        if any([obs in needs_power for obs in observables]):
            self.measure_power = True
            self.power_readings = []

        self.results = dict()
        for obs in observables:
            self.results[obs] = []

    def before_start(self):
        if self.measure_power:
            self.power_readings = []

    def after_start(self):
        self.t0 = time.time()

    def during(self):
        if self.measure_power:
            self.power_readings.append([time.time()-self.t0, self.nvml.pwr_usage()])

    def after_finish(self):
        if self.measure_power:
            execution_time = time.time() - self.t0

            #pre and postfix to start at 0 and end at kernel end
            power_readings = self.power_readings
            if power_readings:
                power_readings = [[0.0, power_readings[0][1]]] + power_readings
                power_readings = power_readings + [[execution_time, power_readings[-1][1]]]

            if "power_readings" in self.observables:
                self.results["power_readings"].append(power_readings) #time in s, power usage in milliwatts

            if "nvml_energy" in self.observables or "nvml_power" in self.observables:
                #compute energy consumption as area under curve
                x = [d[0] for d in power_readings]
                y = [d[1]/1000.0 for d in power_readings] #convert to Watt
                energy = (np.trapz(y,x)) #in Joule
                power = energy / execution_time #in Watt

                if "nvml_energy" in self.observables:
                    self.results["nvml_energy"].append(energy)
                if "nvml_power" in self.observables:
                    self.results["nvml_power"].append(power)

        if "temperature" in self.observables:
            self.results["temperature"].append(self.nvml.temperature)
        if "core_freq" in self.observables:
            self.results["core_freq"].append(self.nvml.gr_clock)
        if "mem_freq" in self.observables:
            self.results["mem_freq"].append(self.nvml.mem_clock)

    def get_results(self):
        averaged_results = dict()

        #return averaged results, except for power_readings
        for obs in self.observables:
            if not obs == "power_readings":
                averaged_results[obs] = np.average(self.results[obs])
        if "power_readings" in self.observables:
            averaged_results["power_readings"] = self.results["power_readings"].copy()

        #clear results for next measurement
        for obs in self.observables:
            self.results[obs] = []

        return averaged_results
