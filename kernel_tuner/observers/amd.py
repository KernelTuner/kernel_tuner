import amdsmi
import logging
import numpy as np
import time

from uuid import UUID
from kernel_tuner.observers import BenchmarkObserver

logger = logging.getLogger(__name__)


def find_device_by_uuid(devices, hip_uuid):
    result = None

    # Missing input
    if hip_uuid is None:
        return None

    # HIP UUID has a strange encoding: https://github.com/ROCm/ROCm/issues/1642
    try:
        hip_hex = UUID(hex=hip_uuid).bytes.decode("ascii")
    except (UnicodeDecodeError, ValueError):
        return None

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


def find_device_by_bdf(devices, pci_domain, pci_bus, pci_device):
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


SUPPORTED_OBSERVABLES = ["energy", "core_freq", "mem_freq", "temperature", "core_voltage"]


class AMDSMIObserver(BenchmarkObserver):
    """
    BenchmarkObserver that uses amdsmi to monitor AMD GPUs and measure energy usage (`energy`),
    core clock frequency (`core_freq`), memory clock frequency (`mem_freq`), temperature (`temperature`),
    and core voltage (`core_voltage`).
    """

    def __init__(self, observables=["energy"], *, device_id=None, prefix="amdsmi"):
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
        self.iteration_results = {k: [] for k in self.observables}
        self.prefix = prefix
        self.device_id = device_id
        self.device = None

    def register_device(self, dev):
        amdsmi.amdsmi_init()
        devices = amdsmi.amdsmi_get_processor_handles()

        env = getattr(dev, "env", dict())

        # Try to find by UUID
        uuid = env.get("uuid")
        uuid_idx = find_device_by_uuid(devices, uuid)

        # Try to find by PCI information
        pci_domain = env.get("pci_domain_id")
        pci_bus = env.get("pci_bus_id")
        pci_device = env.get("pci_device_id")
        pci_idx = find_device_by_bdf(devices, pci_domain, pci_bus, pci_device)

        bdf = f"domain {pci_domain}, bus {pci_bus}, device {pci_device}"

        # If no device id is specified by user, get it from the UUID and PCI
        if self.device_id is None:
            if uuid_idx is None:
                raise ValueError(f"failed to detect AMD device: invalid UUID of backend: {uuid}")

            if pci_idx is None:
                raise ValueError(
                    f"failed to detect AMD device: invalid PCI information of backend: {bdf}"
                )

            if uuid_idx != pci_idx:
                raise ValueError(
                    "failed to detect AMD device: UUID and PCI information are inconsistent"
                )

            self.device_id = uuid_idx
            logger.info(f"selected AMDSMI device {self.device_id}")

        # Warn if UUID wants a different device
        if self.device_id != uuid_idx:
            logger.warning(
                f"specified device has mismatching UUID ({uuid}): {uuid_idx} != {self.device_id}"
            )

        # Warn if PCI wants a different device
        if self.device_id != pci_idx:
            logger.warning(
                f"specified device has mismatching PCI ({bdf}): {pci_idx} != {self.device_id}"
            )

        if not (0 <= self.device_id < len(devices)):
            raise ValueError(
                f"invalid AMD SMI device_id {self.device_id}, found {len(devices)} devices"
            )

        self.device = devices[self.device_id]

    def after_start(self):
        self.energy_after_start = amdsmi.amdsmi_get_energy_count(self.device)
        self.during_timestamps = []
        self.during_results = {k: [] for k in self.observables if k != "energy"}
        self.during()

    def during(self):
        # Get the current timestamp for measurements
        self.during_timestamps.append(time.perf_counter())

        if "core_voltage" in self.observables:
            milli_volt = amdsmi.amdsmi_get_gpu_volt_metric(
                self.device, amdsmi.AmdSmiVoltageType.VDDGFX, amdsmi.AmdSmiVoltageMetric.CURRENT
            )
            self.during_results["core_voltage"].append(milli_volt * 1e-3)  # milli -> volt

        if "core_freq" in self.observables:
            obj = amdsmi.amdsmi_get_clk_freq(self.device, amdsmi.AmdSmiClkType.GFX)
            freq = obj["frequency"][obj["current"]]
            self.during_results["core_freq"].append(freq)

        if "mem_freq" in self.observables:
            obj = amdsmi.amdsmi_get_clk_freq(self.device, amdsmi.AmdSmiClkType.MEM)
            freq = obj["frequency"][obj["current"]]
            self.during_results["mem_freq"].append(freq)

        if "temperature" in self.observables:
            temp = amdsmi.amdsmi_get_temp_metric(
                self.device,
                amdsmi.AmdSmiTemperatureType.HOTSPOT,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
            )

            self.during_results["temperature"].append(temp)

    def after_finish(self):
        self.during()

        # Energy is special as it does not need integration over time
        if "energy" in self.observables:
            before = self.energy_after_start
            after = amdsmi.amdsmi_get_energy_count(self.device)

            # This field changed names in rocm 6.4
            if "energy_accumulator" in before:
                energy_field = "energy_accumulator"
            elif "power" in before:
                energy_field = "power"
            else:
                raise RuntimeError(f"invalid result from amdsmi_get_energy_count: {before}")

            diff = np.uint64(after[energy_field]) - np.uint64(before[energy_field])
            resolution = before["counter_resolution"]
            energy_mj = float(diff) * float(resolution)
            self.iteration_results["energy"].append(energy_mj * 1e-6)  # microJ -> J

        # For the others, we integrate over time and take the average
        for key, values in self.during_results.items():
            x = self.during_timestamps
            avg = np.trapezoid(values, x) / np.ptp(x)  # np.trapz in older versions of np
            self.iteration_results[key].append(avg)

    def get_results(self):
        results = dict()

        for key in list(self.iteration_results):
            avg = np.average(self.iteration_results[key])  # Average of results at each iteration
            self.iteration_results[key] = []  # Reset to empty

            if self.prefix:
                results[f"{self.prefix}_{key}"] = avg
            else:
                results[key] = avg

        return results
