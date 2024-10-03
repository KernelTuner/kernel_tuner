import re
import subprocess
import time

import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver, ContinuousObserver

try:
    import pynvml
except ImportError:
    pynvml = None


class nvml:
    """Class that gathers the NVML functionality for one device."""

    def __init__(self, device_id=0, nvidia_smi_fallback="nvidia-smi", use_locked_clocks=False):
        """Create object to control device using NVML."""
        pynvml.nvmlInit()
        self.dev = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.id = device_id
        self.nvidia_smi = nvidia_smi_fallback

        try:
            self.pwr_limit_default = pynvml.nvmlDeviceGetPowerManagementLimit(self.dev)
            self.pwr_constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.dev)
        except pynvml.NVMLError_NotSupported:
            self.pwr_limit_default = None
            # inverted range to make all range checks fail
            self.pwr_constraints = [1, 0]

        try:
            self._persistence_mode = pynvml.nvmlDeviceGetPersistenceMode(self.dev)
        except pynvml.NVMLError_NotSupported:
            self._persistence_mode = None

        try:
            # returns [isEnabled, isDefaultEnabled]
            self._auto_boost = pynvml.nvmlDeviceGetAutoBoostedClocksEnabled(self.dev)[0]
        except pynvml.NVMLError_NotSupported:
            self._auto_boost = None

        # try to initialize application clocks
        self.modified_clocks = False
        try:
            if not use_locked_clocks:
                self.gr_clock_default = pynvml.nvmlDeviceGetDefaultApplicationsClock(
                    self.dev, pynvml.NVML_CLOCK_GRAPHICS
                )
                self.mem_clock_default = pynvml.nvmlDeviceGetDefaultApplicationsClock(self.dev, pynvml.NVML_CLOCK_MEM)
        except pynvml.NVMLError_NotSupported:
            self.gr_clock_default = None
            self.sm_clock_default = None
            self.mem_clock_default = None
            self.supported_mem_clocks = []
            self.supported_gr_clocks = {}
        self.applications_gr_clock = self.gr_clock_default
        self.applications_mem_clock = self.mem_clock_default

        self.supported_mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self.dev)

        # gather the supported gr clocks for each supported mem clock into a dict
        self.supported_gr_clocks = {}
        for mem_clock in self.supported_mem_clocks:
            supported_gr_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(self.dev, mem_clock)
            self.supported_gr_clocks[mem_clock] = supported_gr_clocks

        # test whether locked gr clocks and mem clocks are supported
        self.use_locked_clocks = use_locked_clocks
        if use_locked_clocks:
            try:
                # try to set highest supported clocks
                mem_clock = self.supported_mem_clocks[0]
                gr_clock = self.supported_gr_clocks[mem_clock][0]
                self.locked_gr_clock = 0
                self.locked_mem_clock = 0
                self.set_clocks(mem_clock, gr_clock)
            except pynvml.NVMLError_NotSupported:
                # switch to using application clocks
                self.use_locked_clocks = False

    def __del__(self):
        # try to restore to defaults
        if self.pwr_limit_default is not None:
            self.pwr_limit = self.pwr_limit_default
        if self.modified_clocks:
            self.reset_clocks()

    @property
    def pwr_state(self):
        """Get the Device current Power State."""
        return pynvml.nvmlDeviceGetPowerState(self.dev)

    @property
    def pwr_limit(self):
        """Control the power limit (may require permission), check pwr_constraints for the allowed range."""
        return pynvml.nvmlDeviceGetPowerManagementLimit(self.dev)

    @pwr_limit.setter
    def pwr_limit(self, new_limit):
        if not self.pwr_constraints[0] <= new_limit <= self.pwr_constraints[1]:
            raise ValueError(
                f"Power limit {new_limit} out of range [{self.pwr_constraints[0]}, {self.pwr_constraints[1]}]"
            )
        if new_limit == self.pwr_limit:
            return
        try:
            pynvml.nvmlDeviceSetPowerManagementLimit(self.dev, new_limit)
        except pynvml.NVMLError_NoPermission:
            if self.nvidia_smi:
                # nvidia-smi expects Watts rather than milliwatts
                new_limit_watt = int(new_limit / 1000.0)
                args = [
                    "sudo",
                    self.nvidia_smi,
                    "-i",
                    str(self.id),
                    "--power-limit=" + str(new_limit_watt),
                ]
                subprocess.run(args, check=True)

    @property
    def persistence_mode(self):
        """Control persistence mode (may require permission), 0 for disabled, 1 for enabled."""
        return self._persistence_mode

    @persistence_mode.setter
    def persistence_mode(self, new_mode):
        if new_mode not in [0, 1]:
            raise ValueError("Illegal value for persistence mode, should be either 0 or 1")
        if self.persistence_mode == new_mode:
            return
        try:
            pynvml.nvmlDeviceSetPersistenceMode(self.dev, new_mode)
            self._persistence_mode = pynvml.nvmlDeviceGetPersistenceMode(self.dev)
        except pynvml.NVMLError_NoPermission:
            args = [
                "sudo",
                self.nvidia_smi,
                "-pm",
                str(new_mode),
            ]
            subprocess.run(args, check=True)
            self._persistence_mode = pynvml.nvmlDeviceGetPersistenceMode(self.dev)

    def set_clocks(self, mem_clock, gr_clock):
        """Set the memory and graphics clock for this device (may require permission)."""

        mem_clock = min(self.supported_mem_clocks, key=lambda x: abs(x - mem_clock))
        if mem_clock not in self.supported_mem_clocks:
            raise ValueError("Illegal value for memory clock")
        if gr_clock not in self.supported_gr_clocks[mem_clock]:
            raise ValueError(
                f"Graphics clock incompatible with memory clock ({mem_clock}), compatible graphics clocks: {self.supported_gr_clocks[mem_clock]}"
            )

        # Check whether persistence mode is set. Without persistence mode, setting the clocks is not meaningful
        # I deliberately removed the try..except clause here, if setting persistence mode fails, setting the clocks should fail
        if self.persistence_mode != 1:
            self.persistence_mode = 1

        if self.use_locked_clocks:
            try:
                pynvml.nvmlDeviceSetGpuLockedClocks(self.dev, gr_clock, gr_clock)
                pynvml.nvmlDeviceSetMemoryLockedClocks(self.dev, mem_clock, mem_clock)
            except pynvml.NVMLError_NoPermission:
                if self.nvidia_smi:
                    args = ["sudo", self.nvidia_smi, "-i", str(self.id)]
                    command_set_mem_clocks = f"--lock-memory-clocks={str(mem_clock)},{str(mem_clock)}"
                    command_set_gpu_clocks = f"--lock-gpu-clocks={str(gr_clock)},{str(gr_clock)}"
                    subprocess.run(args + [command_set_gpu_clocks], check=True)
                    subprocess.run(args + [command_set_mem_clocks], check=True)
            self.locked_gr_clock = gr_clock
            self.locked_mem_clock = mem_clock
        else:
            try:
                pynvml.nvmlDeviceSetApplicationsClocks(self.dev, mem_clock, gr_clock)
            except pynvml.NVMLError_NoPermission:
                if self.nvidia_smi:
                    args = ["sudo", self.nvidia_smi, "-i", str(self.id)]
                    command_set_clocks = f"--applications-clocks={str(mem_clock)},{str(gr_clock)}"
                    subprocess.run(args + [command_set_clocks], check=True)
            self.applications_gr_clock = gr_clock
            self.applications_mem_clock = mem_clock

        # Store the fact that we have modified the clocks
        self.modified_clocks = True

    def reset_clocks(self):
        """Reset the clocks to the default clock if the device uses a non default clock."""
        if self.use_locked_clocks:
            try:
                pynvml.nvmlDeviceResetGpuLockedClocks(self.dev)
                pynvml.nvmlDeviceResetMemoryLockedClocks(self.dev)
            except pynvml.NVMLError_NoPermission:
                if self.nvidia_smi:
                    args = [
                        "sudo",
                        self.nvidia_smi,
                        "-i",
                        str(self.id),
                        "--reset-gpu-clocks",
                    ]
                    subprocess.run(args, check=True)
                    args = [
                        "sudo",
                        self.nvidia_smi,
                        "-i",
                        str(self.id),
                        "--reset-memory-clocks",
                    ]
                    subprocess.run(args, check=True)

        elif self.gr_clock_default is not None:
            gr_app_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_GRAPHICS)
            mem_app_clock = pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_MEM)
            if gr_app_clock != self.gr_clock_default or mem_app_clock != self.mem_clock_default:
                self.set_clocks(self.mem_clock_default, self.gr_clock_default)

    @property
    def gr_clock(self):
        return pynvml.nvmlDeviceGetClockInfo(self.dev, pynvml.NVML_CLOCK_GRAPHICS)

    @gr_clock.setter
    def gr_clock(self, new_clock):
        """Control the graphics clock (may require permission), only values compatible with the memory clock can be set directly."""
        if self.use_locked_clocks:
            if new_clock != self.locked_gr_clock:
                self.set_clocks(self.mem_clock, new_clock)
        else:
            # if using applications clocks
            if new_clock != pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_GRAPHICS):
                self.set_clocks(self.applications_mem_clock, new_clock)

    @property
    def mem_clock(self):
        """Control the memory clock (may require permission), only values compatible with the graphics clock can be set directly."""
        return pynvml.nvmlDeviceGetClockInfo(self.dev, pynvml.NVML_CLOCK_MEM)

    @mem_clock.setter
    def mem_clock(self, new_clock):
        if self.use_locked_clocks:
            if new_clock != self.locked_mem_clock:
                self.set_clocks(new_clock, self.gr_clock)
        # if using applications clocks
        else:
            if new_clock != pynvml.nvmlDeviceGetApplicationsClock(self.dev, pynvml.NVML_CLOCK_MEM):
                self.set_clocks(new_clock, self.applications_gr_clock)

    @property
    def temperature(self):
        """Get the GPU temperature."""
        return pynvml.nvmlDeviceGetTemperature(self.dev, pynvml.NVML_TEMPERATURE_GPU)

    @property
    def auto_boost(self):
        """Control the auto boost setting (may require permission), 0 for disable, 1 for enabled."""
        return self._auto_boost

    @auto_boost.setter
    def auto_boost(self, setting):
        # might need to use pynvml.NVML_FEATURE_DISABLED or pynvml.NVML_FEATURE_ENABLED instead of 0 or 1
        if setting not in [0, 1]:
            raise ValueError("Illegal value for auto boost enabled, should be either 0 or 1")
        pynvml.nvmlDeviceSetAutoBoostedClocksEnabled(self.dev, setting)
        self._auto_boost = pynvml.nvmlDeviceGetAutoBoostedClocksEnabled(self.dev)[0]

    def pwr_usage(self):
        """Return current power usage in milliwatts."""
        NVML_FI_DEV_POWER_INSTANT = 186
        return pynvml.nvmlDeviceGetFieldValues(self.dev, [NVML_FI_DEV_POWER_INSTANT])[0].value.uiVal

    def gr_voltage(self):
        """Return current graphics voltage in millivolts."""
        args = ["nvidia-smi", "-i", str(self.id), "-q", "-d", "VOLTAGE"]
        try:
            result = subprocess.run(args, check=True, capture_output=True)
            m = re.search(r"(\d+\.\d+) mV", result.stdout.decode())
            return float(m.group(1))
        except:
            return np.nan


class NVMLObserver(BenchmarkObserver):
    """Observer that uses NVML to monitor power, energy, clock frequencies, voltages and temperature.

    The NVMLObserver can also be used to tune application-specific clock frequencies or power limits
    in combination with other parameters.

    :param observables: List of quantities that should be observed during tuning, supported are: "power_readings",
        "nvml_power", "nvml_energy", "core_freq", "mem_freq", "temperature", "gr_voltage". If you want to measure the average power
        consumption of a GPU kernel executing on the GPU use "nvml_power". The "power_readings" are the individual power readings
        as reported by NVML and will return a lot of data if you are benchmarking many different kernel configurations.
    :type observables: list of strings

    :param device: Device ordinal used by Nvidia to identify your device, same as reported by nvidia-smi.
    :type device: integer

    :param save_all: If set to True, all data collected by the NVMLObserver for every iteration during benchmarking will be returned.
        If set to False, data will be aggregated over multiple iterations during benchmarking. False by default.
    :type save_all: boolean

    :param nvidia_smi_fallback: String with the location of your nvidia-smi executable to use when Python cannot execute with root privileges, default None.
    :type nvidia_smi_fallback: string

    :param use_locked_clocks: Boolean to opt in to using the locked clocks feature on Ampere or newer GPUs.
        Note, this setting is only relevant when you are tuning with application-specific clocks.
        If set to True, using locked clocks will be preferred over application clocks. If set to False, the Observer
        will set the GPU clocks using the application clocks feature.
        Default is False.
    :type use_locked_clocks: boolean

    :param continuous_duration: Duration to use for energy/power measurements in seconds, default 1 second.
    :type continuous_duration: float

    """

    def __init__(
        self,
        observables,
        device=0,
        save_all=False,
        nvidia_smi_fallback=None,
        use_locked_clocks=False,
        continuous_duration=1,
    ):
        """Create an NVMLObserver."""
        if nvidia_smi_fallback:
            self.nvml = nvml(
                device,
                nvidia_smi_fallback=nvidia_smi_fallback,
                use_locked_clocks=use_locked_clocks,
            )
        else:
            self.nvml = nvml(device, use_locked_clocks=use_locked_clocks)
        self.save_all = save_all

        supported = [
            "power_readings",
            "nvml_power",
            "nvml_energy",
            "core_freq",
            "mem_freq",
            "temperature",
            "gr_voltage",
        ]
        for obs in observables:
            if obs not in supported:
                raise ValueError(f"Observable {obs} not in supported: {supported}")
        self.observables = observables

        self.measure_power = False
        self.needs_power = ["power_readings", "nvml_power", "nvml_energy"]
        if any([obs in self.needs_power for obs in observables]):
            self.measure_power = True
            power_observables = [obs for obs in observables if obs in self.needs_power]
            self.continuous_observer = ContinuousObserver("nvml", power_observables, self, continuous_duration=continuous_duration)

        # remove power observables
        self.observables = [obs for obs in observables if obs not in self.needs_power]

        self.record_gr_voltage = False
        self.t0 = 0
        if "gr_voltage" in observables:
            self.record_gr_voltage = True
            self.gr_voltage_readings = []

        self.results = {}
        for obs in self.observables:
            self.results[obs + "s"] = []

        self.during_obs = [obs for obs in observables if obs in ["core_freq", "mem_freq", "temperature"]]
        self.iteration = {obs: [] for obs in self.during_obs}

    def read_power(self):
        """ Return power in Watt """
        return self.nvml.pwr_usage() / 1e3

    def before_start(self):
        # clear results of the observables for next measurement
        self.iteration = {obs: [] for obs in self.during_obs}
        if self.record_gr_voltage:
            self.gr_voltage_readings = []

    def after_start(self):
        self.t0 = time.perf_counter()
        # ensure during is called at least once
        self.during()

    def during(self):
        if "temperature" in self.observables:
            self.iteration["temperature"].append(self.nvml.temperature)
        if "core_freq" in self.observables:
            self.iteration["core_freq"].append(self.nvml.gr_clock)
        if "mem_freq" in self.observables:
            self.iteration["mem_freq"].append(self.nvml.mem_clock)
        if self.record_gr_voltage:
            self.gr_voltage_readings.append([time.perf_counter() - self.t0, self.nvml.gr_voltage()])

    def after_finish(self):
        if "temperature" in self.observables:
            self.results["temperatures"].append(np.average(self.iteration["temperature"]))
        if "core_freq" in self.observables:
            self.results["core_freqs"].append(np.average(self.iteration["core_freq"]))
        if "mem_freq" in self.observables:
            self.results["mem_freqs"].append(np.average(self.iteration["mem_freq"]))

        if "gr_voltage" in self.observables:
            execution_time = time.time() - self.t0
            gr_voltage_readings = self.gr_voltage_readings
            gr_voltage_readings = [[0.0, gr_voltage_readings[0][1]]] + gr_voltage_readings
            gr_voltage_readings = gr_voltage_readings + [[execution_time, gr_voltage_readings[-1][1]]]
            # time in s, graphics voltage in millivolts
            self.results["gr_voltages"].append(np.average(gr_voltage_readings[:][:][1]))

    def get_results(self):
        averaged_results = {}

        # return averaged results, except when save_all is True
        for obs in self.observables:
            # save all information, if the user requested
            if self.save_all:
                averaged_results[obs + "s"] = self.results[obs + "s"]
            # save averaged results, default
            averaged_results[obs] = np.average(self.results[obs + "s"])

        # clear results for next round
        for obs in self.observables:
            self.results[obs + "s"] = []

        return averaged_results


# High-level Helper functions


def get_nvml_pwr_limits(device, n=None, quiet=False):
    """Get tunable parameter for NVML power limits, n is desired number of values."""
    d = nvml(device)
    power_limits = d.pwr_constraints
    power_limit_min = power_limits[0]
    power_limit_max = power_limits[-1]
    # Min and Max converted to Watt
    power_limit_min *= 1e-3
    power_limit_max *= 1e-3
    power_limit_round = 5
    tune_params = dict()
    if n is None:
        n = int((power_limit_max - power_limit_min) / power_limit_round) + 1

    # Rounded power limit values
    power_limits = power_limit_round * np.round((np.linspace(power_limit_min, power_limit_max, n) / power_limit_round))
    power_limits = sorted(list(set([int(power_limit) for power_limit in power_limits])))
    tune_params["nvml_pwr_limit"] = power_limits

    if not quiet:
        print("Using power limits:", tune_params["nvml_pwr_limit"])
    return tune_params


def get_nvml_gr_clocks(device, n=None, quiet=False):
    """Get tunable parameter for NVML graphics clock, n is desired number of values."""
    d = nvml(device)
    mem_clock = max(d.supported_mem_clocks)
    gr_clocks = d.supported_gr_clocks[mem_clock]

    if n and (len(gr_clocks) > n):
        indices = np.array(np.ceil(np.linspace(0, len(gr_clocks) - 1, n)), dtype=int)
        gr_clocks = np.array(gr_clocks)[indices]

    tune_params = dict()
    tune_params["nvml_gr_clock"] = list(gr_clocks)

    if not quiet:
        print("Using gr frequencies:", tune_params["nvml_gr_clock"])
    return tune_params


def get_nvml_mem_clocks(device, n=None, quiet=False):
    """Get tunable parameter for NVML memory clock, n is desired number of values."""
    d = nvml(device)
    mem_clocks = d.supported_mem_clocks

    if n and len(mem_clocks) > n:
        mem_clocks = mem_clocks[:: int(len(mem_clocks) / n)]

    tune_params = dict()
    tune_params["nvml_mem_clock"] = mem_clocks

    if not quiet:
        print("Using mem frequencies:", tune_params["nvml_mem_clock"])
    return tune_params


def get_idle_power(device, n=5, sleep_s=0.1):
    """Use NVML to measure device idle power consumption."""
    d = nvml(device)
    readings = []
    for _ in range(n):
        time.sleep(sleep_s)
        readings.append(d.pwr_usage())
    return np.mean(readings) * 1e-3  # Watt
