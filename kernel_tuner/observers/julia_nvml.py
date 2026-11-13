"""NVML observer for the Julia backend, using NVML.jl (the NVML wrapper of CUDA.jl) instead of pynvml.

The observables and tunable parameters are the same as those of the :class:`~kernel_tuner.observers.nvml.NVMLObserver`.
The NVML calls are made from the Julia process that runs the kernels, so pynvml is not needed. See ``julia_nvml.jl``
for the Julia side.
"""

import atexit
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from kernel_tuner.observers.nvml import FIELD_TYPE_MAP, SUPPORTED_OBSERVABLES, NVMLObserver  # noqa: F401

NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NO_PERMISSION = 4


class NVMLError(Exception):
    """NVML call failed, ``code`` is the NVML return code (-1 if the error did not come from NVML)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class NVMLError_NotSupported(NVMLError):
    pass


class NVMLError_NoPermission(NVMLError):
    pass


class _JuliaNVMLLibrary:
    """Calls the functions of the ``KernelTunerNVML`` Julia module, and raises NVML errors as Python exceptions."""

    def __init__(self):
        # importing the Julia backend first sets up the signal handling for JuliaCall before it is first imported
        import kernel_tuner.backends.julia  # noqa: F401

        try:
            from juliacall import JuliaError
            from juliacall import Main as jl
        except ImportError as e:
            raise ImportError("JuliaCall not installed. Please run `pip install juliacall`.") from e
        self._error_type = JuliaError
        if not jl.seval("isdefined(Main, :KernelTunerNVML)"):
            try:
                jl.include(str(Path(__file__).parent / "julia_nvml.jl"))
            except JuliaError as e:
                raise ImportError(f"Could not load NVML.jl, are CUDACore.jl and NVML.jl installed? {e}") from e
        self._module = jl.KernelTunerNVML

    def __getattr__(self, name):
        function = getattr(self._module, name)

        def call(*args):
            try:
                return function(*args)
            except self._error_type as e:
                code = self._module.error_code(e.exception)
                error_class = {
                    NVML_ERROR_NOT_SUPPORTED: NVMLError_NotSupported,
                    NVML_ERROR_NO_PERMISSION: NVMLError_NoPermission,
                }.get(code, NVMLError)
                raise error_class(code, f"NVML call {name} failed: {e}") from e

        return call


_library = None


def julia_nvml_library():
    """Load the Julia NVML functions on first use, so that Julia is only started when needed."""
    global _library
    if _library is None:
        _library = _JuliaNVMLLibrary()
    return _library


class JuliaNVML:
    """Class that gathers the NVML functionality for one device, through NVML.jl.

    Has the same interface as :class:`kernel_tuner.observers.nvml.nvml`. The device is selected by exactly one of
    ``cuda_device`` (a ``CUDACore.CuDevice`` from Julia), ``device_id`` (the NVML index, as reported by nvidia-smi),
    ``device_uuid``, or ``device_pci_bus``.
    """

    def __init__(
        self,
        device_id=None,
        device_uuid=None,
        device_pci_bus=None,
        nvidia_smi_fallback=None,
        use_locked_clocks=False,
        cuda_device=None,
    ):
        """Create object to control device using NVML through NVML.jl."""
        # set these first, as __del__ uses them and this __init__ may raise an exception midway
        self.pwr_limit_default = None
        self.modified_clocks = False
        self.modified_pwr_limit = False

        self.lib = julia_nvml_library()
        if cuda_device is not None:
            self.dev = self.lib.device_of(cuda_device)
        elif device_id is not None:
            self.dev = self.lib.device_by_index(int(device_id))
        elif device_uuid is not None:
            self.dev = self.lib.device_by_uuid(str(device_uuid))
        elif device_pci_bus is not None:
            self.dev = self.lib.device_by_pci_bus(str(device_pci_bus))
        else:
            raise ValueError("No device given, pass one of cuda_device, device_id, device_uuid, or device_pci_bus")

        self.id = self.lib.index(self.dev)
        self.uuid = str(self.lib.uuid(self.dev))
        self.pci_bus = str(self.lib.pci_bus(self.dev))
        self.nvidia_smi = nvidia_smi_fallback or "nvidia-smi"

        if device_id is not None and self.id != device_id:
            raise ValueError(f"NVML device ID does not match requested device: {device_id} != {self.id}")

        try:
            self.pwr_limit_default = self.lib.power_limit(self.dev)
            self.pwr_constraints = [int(x) for x in self.lib.power_limit_constraints(self.dev)]
        except NVMLError_NotSupported:
            # inverted range to make all range checks fail
            self.pwr_limit_default = None
            self.pwr_constraints = [1, 0]

        try:
            self._persistence_mode = self.lib.persistence_mode(self.dev)
        except NVMLError_NotSupported:
            self._persistence_mode = None

        try:
            self._auto_boost = self.lib.auto_boost(self.dev)
        except NVMLError_NotSupported:
            self._auto_boost = None

        # gather info on clock defaults
        try:
            self.gr_clock_default = self.lib.default_applications_clock(self.dev, "graphics")
            self.mem_clock_default = self.lib.default_applications_clock(self.dev, "mem")
        except NVMLError_NotSupported:
            self.gr_clock_default = None
            self.mem_clock_default = None
        self.applications_gr_clock = self.gr_clock_default
        self.applications_mem_clock = self.mem_clock_default

        # supported clocks, from high to low, with the supported gr clocks for each supported mem clock
        self.supported_mem_clocks = [int(x) for x in self.lib.supported_memory_clocks(self.dev)]
        self.supported_gr_clocks = {
            mem_clock: [int(x) for x in self.lib.supported_graphics_clocks(self.dev, mem_clock)]
            for mem_clock in self.supported_mem_clocks
        }

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
            except NVMLError_NotSupported:
                # switch to using application clocks
                self.use_locked_clocks = False

    def __del__(self):
        try:
            self.restore_defaults()
        except Exception:
            # Julia or nvidia-smi may no longer be available while Python is shutting down,
            # in which case the defaults were already restored at exit
            pass

    def restore_defaults(self):
        """Restore the power limit and clocks, if they were changed."""
        if self.modified_pwr_limit and self.pwr_limit_default is not None:
            self.pwr_limit = self.pwr_limit_default
        if self.modified_clocks:
            self.reset_clocks()

    def _restore_at_exit(self):
        if not (self.modified_clocks or self.modified_pwr_limit):
            atexit.register(self.restore_defaults)

    def _sudo_nvidia_smi(self, *options):
        subprocess.run(["sudo", self.nvidia_smi, "-i", str(self.id), *options], check=True)

    @property
    def pwr_state(self):
        """Get the Device current Power State."""
        return self.lib.power_state(self.dev)

    @property
    def pwr_limit(self):
        """Control the power limit in milliwatt (may require permission), check pwr_constraints for the allowed range."""
        return self.lib.power_limit(self.dev)

    @pwr_limit.setter
    def pwr_limit(self, new_limit):
        if not self.pwr_constraints[0] <= new_limit <= self.pwr_constraints[1]:
            raise ValueError(
                f"Power limit {new_limit} out of range [{self.pwr_constraints[0]}, {self.pwr_constraints[1]}]"
            )
        if new_limit == self.pwr_limit:
            return
        self._restore_at_exit()
        self.modified_pwr_limit = True
        try:
            self.lib.set_power_limit_b(self.dev, int(new_limit))
        except NVMLError_NoPermission:
            # nvidia-smi expects Watts rather than milliwatts
            self._sudo_nvidia_smi(f"--power-limit={int(new_limit / 1000.0)}")

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
            self.lib.set_persistence_mode_b(self.dev, new_mode)
        except NVMLError_NoPermission:
            self._sudo_nvidia_smi("-pm", str(new_mode))
        self._persistence_mode = self.lib.persistence_mode(self.dev)

    def set_clocks(self, mem_clock, gr_clock):
        """Set the memory and graphics clock for this device (may require permission)."""
        mem_clock = min(self.supported_mem_clocks, key=lambda x: abs(x - mem_clock))
        if gr_clock not in self.supported_gr_clocks[mem_clock]:
            raise ValueError(
                f"Graphics clock incompatible with memory clock ({mem_clock}), compatible graphics clocks: {self.supported_gr_clocks[mem_clock]}"
            )

        # Without persistence mode, setting the clocks is not meaningful, so if setting it fails, setting the clocks fails
        if self.persistence_mode != 1:
            self.persistence_mode = 1

        self._restore_at_exit()
        self.modified_clocks = True
        if self.use_locked_clocks:
            try:
                self.lib.set_locked_clocks_b(self.dev, int(mem_clock), int(gr_clock))
            except NVMLError_NoPermission:
                self._sudo_nvidia_smi(f"--lock-gpu-clocks={gr_clock},{gr_clock}")
                self._sudo_nvidia_smi(f"--lock-memory-clocks={mem_clock},{mem_clock}")
            self.locked_gr_clock = gr_clock
            self.locked_mem_clock = mem_clock
        else:
            try:
                self.lib.set_applications_clocks_b(self.dev, int(mem_clock), int(gr_clock))
            except NVMLError_NoPermission:
                self._sudo_nvidia_smi(f"--applications-clocks={mem_clock},{gr_clock}")
            self.applications_gr_clock = gr_clock
            self.applications_mem_clock = mem_clock

    def reset_clocks(self):
        """Reset the clocks to the default clock if the device uses a non default clock."""
        if self.use_locked_clocks:
            try:
                self.lib.reset_locked_clocks_b(self.dev)
            except NVMLError_NoPermission:
                self._sudo_nvidia_smi("--reset-gpu-clocks")
                self._sudo_nvidia_smi("--reset-memory-clocks")
        elif self.gr_clock_default is not None:
            gr_app_clock = self.lib.applications_clock(self.dev, "graphics")
            mem_app_clock = self.lib.applications_clock(self.dev, "mem")
            if gr_app_clock != self.gr_clock_default or mem_app_clock != self.mem_clock_default:
                self.set_clocks(self.mem_clock_default, self.gr_clock_default)

    @property
    def gr_clock(self):
        """Control the graphics clock (may require permission), only values compatible with the memory clock can be set directly."""
        return self.lib.clock(self.dev, "graphics")

    @gr_clock.setter
    def gr_clock(self, new_clock):
        if self.use_locked_clocks:
            if new_clock != self.locked_gr_clock:
                self.set_clocks(self.mem_clock, new_clock)
        elif new_clock != self.lib.applications_clock(self.dev, "graphics"):
            self.set_clocks(self.applications_mem_clock, new_clock)

    @property
    def mem_clock(self):
        """Control the memory clock (may require permission), only values compatible with the graphics clock can be set directly."""
        return self.lib.clock(self.dev, "mem")

    @mem_clock.setter
    def mem_clock(self, new_clock):
        if self.use_locked_clocks:
            if new_clock != self.locked_mem_clock:
                self.set_clocks(new_clock, self.gr_clock)
        elif new_clock != self.lib.applications_clock(self.dev, "mem"):
            self.set_clocks(new_clock, self.applications_gr_clock)

    @property
    def temperature(self):
        """Get the GPU temperature."""
        return self.lib.temperature(self.dev)

    @property
    def auto_boost(self):
        """Control the auto boost setting (may require permission), 0 for disable, 1 for enabled."""
        return self._auto_boost

    @auto_boost.setter
    def auto_boost(self, setting):
        if setting not in [0, 1]:
            raise ValueError("Illegal value for auto boost enabled, should be either 0 or 1")
        self.lib.set_auto_boost_b(self.dev, setting)
        self._auto_boost = self.lib.auto_boost(self.dev)

    def pwr_usage(self):
        """Return current power usage in milliwatts."""
        return self.lib.power(self.dev)

    def energy_usage(self):
        """Return total energy usage since the driver was loaded in millijoules."""
        return self.lib.energy(self.dev)

    def gr_voltage(self):
        """Return current graphics voltage in millivolts, NVML does not report it, so it is read from nvidia-smi."""
        args = [self.nvidia_smi, "-i", str(self.id), "-q", "-d", "VOLTAGE"]
        try:
            result = subprocess.run(args, check=True, capture_output=True)
            m = re.search(r"(\d+\.\d+) mV", result.stdout.decode())
            return float(m.group(1))
        except Exception:
            return np.nan


class JuliaNVMLObserver(NVMLObserver):
    """Observer for the Julia backend that uses NVML through NVML.jl to monitor power, energy, clock frequencies, voltages and temperature.

    Takes the same arguments and supports the same observables and tunable parameters (``nvml_pwr_limit``,
    ``nvml_gr_clock``, and ``nvml_mem_clock``) as the :class:`~kernel_tuner.observers.nvml.NVMLObserver`, see there.
    By default, the observer uses the GPU that the Julia backend runs on, pass ``device`` (the index as reported by
    nvidia-smi) to observe another one.
    """

    # Minimum time in seconds between two power readings. During continuous benchmarking, the power is polled every
    # few microseconds, while NVML updates it much less often (e.g. every 500 ms on an RTX A6000). Every call into
    # Julia allocates memory, which can trigger a garbage collection pause during the measurement.
    power_read_interval = 1e-3
    _last_power = None
    _last_power_time = 0.0

    def read_power(self):
        """Return power in Watt, read at most once every ``power_read_interval`` seconds."""
        now = time.perf_counter()
        if self._last_power is None or now - self._last_power_time >= self.power_read_interval:
            self._last_power, self._last_power_time = super().read_power(), now
        return self._last_power

    def register_device(self, dev):
        cuda_device = getattr(dev, "backend_device", None) if getattr(dev, "backend_name", None) == "CUDA" else None
        env = getattr(dev, "env", dict())
        uuid = env.get("uuid")
        pci_bus = env.get("pci_bus_id")

        if self.device is not None:
            self.nvml = JuliaNVML(device_id=self.device, **self.nvml_kwargs)
        elif cuda_device is not None:
            self.nvml = JuliaNVML(cuda_device=cuda_device, **self.nvml_kwargs)
        elif uuid is not None:
            self.nvml = JuliaNVML(device_uuid=uuid, **self.nvml_kwargs)
        elif pci_bus is not None:
            self.nvml = JuliaNVML(device_pci_bus=pci_bus, **self.nvml_kwargs)
        else:
            raise ValueError(
                f"failed to detect NVIDIA device: not a Julia CUDA backend, and no UUID or PCI-bus-id in environment: {env}"
            )


# High-level Helper functions, the same as those in kernel_tuner.observers.nvml


def get_nvml_pwr_limits(device, n=None, quiet=False):
    """Get tunable parameter for NVML power limits, n is desired number of values."""
    d = JuliaNVML(device)
    # Min and Max converted to Watt
    power_limit_min = d.pwr_constraints[0] * 1e-3
    power_limit_max = d.pwr_constraints[-1] * 1e-3
    power_limit_round = 5
    if n is None:
        n = int((power_limit_max - power_limit_min) / power_limit_round) + 1

    # Rounded power limit values
    power_limits = power_limit_round * np.round((np.linspace(power_limit_min, power_limit_max, n) / power_limit_round))
    tune_params = {"nvml_pwr_limit": sorted(set(int(power_limit) for power_limit in power_limits))}

    if not quiet:
        print("Using power limits:", tune_params["nvml_pwr_limit"])
    return tune_params


def get_nvml_gr_clocks(device, n=None, quiet=False):
    """Get tunable parameter for NVML graphics clock, n is desired number of values."""
    d = JuliaNVML(device)
    gr_clocks = d.supported_gr_clocks[max(d.supported_mem_clocks)]

    if n and (len(gr_clocks) > n):
        indices = np.array(np.ceil(np.linspace(0, len(gr_clocks) - 1, n)), dtype=int)
        gr_clocks = [gr_clocks[i] for i in indices]

    tune_params = {"nvml_gr_clock": list(gr_clocks)}

    if not quiet:
        print("Using gr frequencies:", tune_params["nvml_gr_clock"])
    return tune_params


def get_nvml_mem_clocks(device, n=None, quiet=False):
    """Get tunable parameter for NVML memory clock, n is desired number of values."""
    d = JuliaNVML(device)
    mem_clocks = d.supported_mem_clocks

    if n and len(mem_clocks) > n:
        mem_clocks = mem_clocks[:: int(len(mem_clocks) / n)]

    tune_params = {"nvml_mem_clock": mem_clocks}

    if not quiet:
        print("Using mem frequencies:", tune_params["nvml_mem_clock"])
    return tune_params


def get_idle_power(device, n=5, sleep_s=0.1):
    """Use NVML to measure device idle power consumption in Watt."""
    d = JuliaNVML(device)
    readings = []
    for _ in range(n):
        time.sleep(sleep_s)
        readings.append(d.pwr_usage())
    return np.mean(readings) * 1e-3
