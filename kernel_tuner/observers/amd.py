import amdsmi
import logging
import numpy as np
import time

# Trapz was renamed to trapezoid in Numpy 2.0
try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

from uuid import UUID
from kernel_tuner.observers import BenchmarkObserver, ContinuousObserver

logger = logging.getLogger(__name__)


def _find_device_by_uuid(devices, hip_uuid):
    result = None

    # Missing input
    if hip_uuid is None:
        return None

    # HIP UUID has a strange encoding: https://github.com/ROCm/ROCm/issues/1642
    try:
        hip_hex = UUID(hex=hip_uuid).bytes.decode("ascii")
    except (UnicodeDecodeError, ValueError):
        hip_hex = str(hip_uuid)

    for index, device in enumerate(devices):
        smi_uuid = str(amdsmi.amdsmi_get_gpu_device_uuid(device))

        # For some reason, only the last 12 bytes match
        if hip_hex[-12:] != smi_uuid[-12:]:
            continue

        # Multiple devices have same UUID?
        if result is not None:
            logger.warning(f"could not find device: multiple have UUID {hip_uuid}")
            return None

        result = index

    return result


def _find_device_by_bdf(devices, pci_domain, pci_bus, pci_device):
    result = None

    # Missing input
    if pci_domain is None or pci_bus is None or pci_device is None:
        return None

    for index, device in enumerate(devices):
        bdf = amdsmi.amdsmi_get_gpu_bdf_id(device)
        x = (bdf >> 32) & 0xFFFFFFFF
        y = (bdf >> 8) & 0xFF
        z = (bdf >> 3) & 0x1F

        if (x, y, z) != (pci_domain, pci_bus, pci_device):
            continue

        if result is not None:
            msg = f"domain {pci_domain}, bus {pci_bus}, device {pci_device}"
            logger.warning(f"could not find device: multiple have PCI {msg}")
            return None

        result = index

    return result


class AMDDevice:
    def __init__(self, device):
        self.device = device

    def total_energy_usage(self):
        """Returns total energy usage since startup."""

        result = amdsmi.amdsmi_get_energy_count(self.device)

        # This field changed name in rocm 6.4
        if "energy_accumulator" not in result:
            if "power" in result:
                result["energy_accumulator"] = result["power"]
            else:
                raise RuntimeError(f"invalid result from amdsmi_get_energy_count: {result}")

        return result

    def current_power_usage(self):
        info = amdsmi.amdsmi_get_power_info(self.device)

        if "current_socket_power" in info:
            # For newer Mi300+ cards
            return info["current_socket_power"]
        elif "average_socket_power" in info:
            # For older cards
            return info["average_socket_power"]
        else:
            raise RuntimeError(f"invalid result from amdsmi_get_power_info: {info}")

    def core_voltage(self):
        """Returns current voltage in Volt."""

        milli_volt = amdsmi.amdsmi_get_gpu_volt_metric(
            self.device,
            amdsmi.AmdSmiVoltageType.VDDGFX,
            amdsmi.AmdSmiVoltageMetric.CURRENT,
        )

        # milli * 1-e3 -> volt
        return milli_volt * 1e-3

    def temperature(self):
        """Returns current temperature in celcius."""
        return amdsmi.amdsmi_get_temp_metric(
            self.device,
            amdsmi.AmdSmiTemperatureType.HOTSPOT,
            amdsmi.AmdSmiTemperatureMetric.CURRENT,
        )

    def mem_temperature(self):
        """Returns current temperature in celcius."""
        return amdsmi.amdsmi_get_temp_metric(
            self.device,
            amdsmi.AmdSmiTemperatureType.VRAM,
            amdsmi.AmdSmiTemperatureMetric.CURRENT,
        )

    def core_freq(self):
        """Returns current core clock frequency in Hz."""
        obj = amdsmi.amdsmi_get_clk_freq(self.device, amdsmi.AmdSmiClkType.GFX)
        freq = obj["frequency"][obj["current"]]
        return freq

    def mem_freq(self):
        """Returns current memory clock frequency in Hz."""
        obj = amdsmi.amdsmi_get_clk_freq(self.device, amdsmi.AmdSmiClkType.MEM)
        freq = obj["frequency"][obj["current"]]
        return freq

    def core_activity(self):
        """Returns core usage as percentage (0-100)."""
        obj = amdsmi.amdsmi_get_gpu_activity(self.device)
        result = obj["gfx_activity"]
        # Result is "N/A" on error, return NaN instead
        return float("nan") if isinstance(result, str) else result

    def mem_activity(self):
        """Returns memory usage as percentage (0-100)."""
        obj = amdsmi.amdsmi_get_gpu_activity(self.device)
        result = obj["umc_activity"]
        # Result is "N/A" on error, return NaN instead
        return float("nan") if isinstance(result, str) else result


SUPPORTED_OBSERVABLES = [
    "energy",
    "power",
    "core_freq",
    "mem_freq",
    "temperature",
    "mem_temperature",
    "core_voltage",
    "core_activity",
    "mem_activity",
]


class AMDSMIContinuousObserver(ContinuousObserver):
    def __init__(self, parent, continuous_duration=1.0):
        self.parent = parent
        self.continuous_duration = continuous_duration
        self.warmup_time = min(0.1, continuous_duration / 2)

        # This assigned by Kernel Tuner's core
        self.results = None

    def before_start(self):
        self.parent.before_start()

    def after_start(self):
        self.warmup_completed = False
        self.start_time = time.perf_counter() + self.warmup_time

    def during(self):
        now = time.perf_counter()

        if not self.warmup_completed:
            if now < self.start_time:
                return

            # Only call `after_start` once warmup time has passed
            self.start_time = now
            self.warmup_completed = True
            self.parent.after_start()

        self.parent.during()

    def after_finish(self):
        if self.warmup_completed:
            self.parent.after_finish()

    def get_results(self):
        if not self.warmup_completed:
            return dict()

        elapsed_sec = time.perf_counter() - self.start_time
        time_sec = self.results["time"] * 1e-3
        ratio = time_sec / elapsed_sec

        # Get results from the parent
        results = self.parent.get_results()

        # The energy field measures the energy over the entire
        # continuous duration. However, we want the average
        # energy usage _per_ kernel. To fix this, we multiply
        # by the ratio of elapsed time to time per kernel
        energy_field = self.parent.field_name("energy")

        if energy_field in results:
            results[energy_field] = results[energy_field] * ratio

        return results


class AMDSMIObserver(BenchmarkObserver):
    """
    BenchmarkObserver that uses amdsmi to monitor AMD GPUs and measure energy usage (`energy`),
    core clock frequency (`core_freq`), memory clock frequency (`mem_freq`), temperature (`temperature`),
    and core voltage (`core_voltage`).
    """

    def __init__(
        self,
        observables=["energy"],
        *,
        device_id=None,
        prefix="amdsmi",
        use_continuous_observer=True,
        continuous_duration=1.0,
    ):
        """
        Initialize the AMDSMIObserver.

        Supported observables are: `energy`, `core_freq`, `mem_freq`, `temperature`, and `core_voltage`.

        :param observables: List of metrics to monitor. Defaults to just energy.
        :param device_id: Specific AMD device index. If None, auto-detection is used.
        :param prefix: Prefix used for name in the metrics. Defaults to "amdsmi".
        """
        for obs in observables:
            if obs not in SUPPORTED_OBSERVABLES:
                raise ValueError(f"Observable {obs} not supported: {SUPPORTED_OBSERVABLES}")

        self.observables = set(observables)
        self.prefix = prefix
        self.device_id = device_id
        self.device = None
        self.use_continuous_observer = use_continuous_observer
        self.continuous_duration = continuous_duration
        self.results_per_iteration = {self.field_name(k): [] for k in self.observables}

    def register_device(self, dev):
        amdsmi.amdsmi_init()
        devices = amdsmi.amdsmi_get_processor_handles()

        env = getattr(dev, "env", dict())

        # Try to find by UUID
        uuid = env.get("uuid")
        uuid_idx = _find_device_by_uuid(devices, uuid)

        # Try to find by PCI information
        pci_domain = env.get("pci_domain_id")
        pci_bus = env.get("pci_bus_id")
        pci_device = env.get("pci_device_id")
        pci_idx = _find_device_by_bdf(devices, pci_domain, pci_bus, pci_device)

        bdf = f"domain {pci_domain}, bus {pci_bus}, device {pci_device}"

        # If no device id is specified by user, get it from the UUID and PCI
        if self.device_id is None:
            if uuid_idx is None:
                raise ValueError(f"failed to detect AMD device: invalid UUID of backend: {uuid}")

            if pci_idx is None:
                raise ValueError(f"failed to detect AMD device: invalid PCI information of backend: {bdf}")

            if uuid_idx != pci_idx:
                raise ValueError("failed to detect AMD device: UUID and PCI information are inconsistent")

            self.device_id = uuid_idx
            logger.info(f"selected AMDSMI device {self.device_id}")

        # Warn if UUID wants a different device
        if uuid_idx is not None and self.device_id != uuid_idx:
            logger.warning(f"specified device has mismatching UUID ({uuid}): {uuid_idx} != {self.device_id}")

        # Warn if PCI wants a different device
        if pci_idx is not None and self.device_id != pci_idx:
            logger.warning(f"specified device has mismatching PCI ({bdf}): {pci_idx} != {self.device_id}")

        if not (0 <= self.device_id < len(devices)):
            raise ValueError(f"invalid AMD SMI device_id {self.device_id}, found {len(devices)} devices")

        self.device = AMDDevice(devices[self.device_id])

        if self.use_continuous_observer:
            self.continuous_observer = AMDSMIContinuousObserver(self, continuous_duration=self.continuous_duration)

    def after_start(self):
        self.energy_after_start = self.device.total_energy_usage()
        self.sample_timestamps = []
        self.sample_values = {k: [] for k in self.results_per_iteration}
        self.sample_metrics()

    def during(self):
        self.sample_metrics()

    def field_name(self, name):
        if self.prefix:
            return f"{self.prefix}_{name}"
        else:
            return name

    def store_sample(self, name, value):
        self.sample_values[self.field_name(name)].append(value)

    def sample_metrics(self):
        self.sample_timestamps.append(time.perf_counter())

        if "core_voltage" in self.observables:
            self.store_sample("core_voltage", self.device.core_voltage())

        if "core_freq" in self.observables:
            self.store_sample("core_freq", self.device.core_freq())

        if "mem_freq" in self.observables:
            self.store_sample("mem_freq", self.device.mem_freq())

        if "temperature" in self.observables:
            self.store_sample("temperature", self.device.temperature())

        if "mem_temperature" in self.observables:
            self.store_sample("mem_temperature", self.device.mem_temperature())

        if "core_activity" in self.observables:
            self.store_sample("core_activity", self.device.core_activity())

        if "mem_activity" in self.observables:
            self.store_sample("mem_activity", self.device.mem_activity())

    def after_finish(self):
        before = self.energy_after_start
        after = self.device.total_energy_usage()
        self.sample_metrics()

        diff = np.uint64(after["energy_accumulator"]) - np.uint64(before["energy_accumulator"])
        elapsed_ns = np.uint64(after["timestamp"]) - np.uint64(before["timestamp"])
        resolution = before["counter_resolution"]
        energy_uj = float(diff) * float(resolution)

        # Energy is an exception as it does not need integration over time
        if "energy" in self.observables:
            # microJ * 1e-6 -> J
            self.results_per_iteration[self.field_name("energy")].append(energy_uj * 1e-6)

        if "power" in self.observables:
            self.results_per_iteration[self.field_name("power")].append(energy_uj / elapsed_ns * 1e3)

        # normalize timestamps to [0, 1] such that integral (trapezoid) is the mean
        xs = np.array(self.sample_timestamps)
        xs = (xs - xs.min()) / (xs.max() - xs.min())

        for key, values in self.sample_values.items():
            # Could not sample, skip field
            if not values:
                continue

            # If all values are the same, take that value directly.
            # This preserve that value bitwise exactly and prevents
            # rounding errors that occur in trapezoid
            if all(v == values[0] for v in values):
                result = values[0]
            else:
                result = trapezoid(values, x=xs)

            self.results_per_iteration[key].append(result)

    def get_results(self):
        results = dict()

        for key in list(self.results_per_iteration):
            # Take average and reset!
            results[key] = np.average(self.results_per_iteration[key])
            self.results_per_iteration[key] = []

        return results
