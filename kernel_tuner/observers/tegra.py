import subprocess
import time
from pathlib import Path
import os

import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver, ContinuousObserver
from kernel_tuner.observers.pmt import PMTObserver
from kernel_tuner.observers.powersensor import PowerSensorObserver


class tegra:
    """Class that gathers the Tegra functionality for one device."""

    def __init__(self, power_path, temp_path):
        """Create object to control GPU core clock on a Tegra device."""
        self.has_changed_clocks = False

        # Get paths
        self.dev_path = self.get_dev_path()
        if temp_path == "":
            self.gpu_temp_path = self.get_temp_path()
        else:
            self.gpu_temp_path = temp_path
        if power_path == "":
            self.gpu_power_path = self.get_power_path()
        else:
            self.gpu_power_path = power_path
        self.gpu_channel = self.get_gpu_channel()

        # Read default clock values
        self.default_min_gr_clock = self._read_clock_file("min_freq")
        self.default_max_gr_clock = self._read_clock_file("max_freq")
        self.supported_gr_clocks = self._read_clock_file("available_frequencies")

        self.default_railgate_status = self._read_railgate_file()

    @staticmethod
    def get_dev_path():
        """Get the path to device core clock control in /sys"""
        # loop to find GPU device name based on jetson_clocks
        for dev in Path("/sys/class/devfreq").iterdir():
            with open(dev / Path("device/of_node/name")) as fp:
                name = fp.read().strip().rstrip("\x00")
            if name in ("gv11b", "gp10b", "ga10b", "gpu"):
                root_path = dev
                break
        else:
            raise FileNotFoundError("No internal tegra GPU found")
        return root_path

    def get_temp_path(self):
        """Find the file which holds the GPU temperature"""
        for zone in Path("/sys/class/thermal").iterdir():
            with open(zone / Path("type")) as fp:
                name = fp.read().strip()
            if name == "GPU-therm":
                gpu_temp_path = str(zone)
                break

        if gpu_temp_path is None:
            raise FileNotFoundError("No GPU sensor for temperature found")

        return gpu_temp_path

    def get_power_path(self, start_path="/sys/bus/i2c/drivers/ina3221"):
        """Search for a file which holds power readings"""
        for entry in os.listdir(start_path):
            path = os.path.join(start_path, entry)
            if os.path.isfile(path) and entry == "curr1_input":
                return start_path
            elif entry in start_path:
                continue
            elif os.path.isdir(path):
                result = self.get_power_path(path)
                if result:
                    return result
        return None

    def get_gpu_channel(self):
        """Get the channel number of the sensor which measures the GPU power"""
        # Iterate over all channels in the of_node dir of the power path to
        # find the channel which holds GPU power information
        for channel_dir in Path(self.gpu_power_path + "/of_node/").iterdir():
            if("channel@" in channel_dir.name):
                with open(channel_dir / Path("label")) as fp:
                    channel_label = fp.read().strip()
                if "GPU" in channel_label:
                    return str(int(channel_dir.name[-1])+1)

        # If this statement is reached, no channel for the GPU was found
        raise FileNotFoundError("No channel found with GPU power readings")

    def _read_railgate_file(self):
        """Read railgate status"""
        with open(self.dev_path / Path("device/railgate_enable")) as fp:
            data = int(fp.read().strip())
        return data

    def _write_railgate_file(self, value):
        """Set railgate status"""
        if value not in (0, 1):
            raise ValueError(f"Illegal governor value {value}, must be 0 or 1")
        full_path = self.dev_path / Path("device/railgate_enable")
        args = [
            "sudo",
            "sh",
            "-c",
            f"echo {value} > {str(full_path)}"
        ]
        subprocess.run(args, check=True)

    def _read_clock_file(self, fname):
        """Read current or available frequency value(s) from a frequency control file"""
        with open(self.dev_path / Path(fname)) as fp:
            raw_data = np.array(fp.read().strip().split())
        if len(raw_data) > 1:
            data = raw_data.astype(int)
        else:
            data = int(raw_data)
        return data

    def _write_clock_file(self, fname, value):
        """Write a frequency value to a core clock control file"""
        self.has_changed_clocks = True
        available_files = ("min_freq", "max_freq")
        if fname not in available_files:
            raise ValueError(f"Illegal filename value: {fname}, must be one of {available_files}")

        if value not in self.supported_gr_clocks:
            raise ValueError(f"Illegal frequency value {value}, must be one of {self.supported_gr_clocks}")

        full_path = self.dev_path / Path(fname)
        args = [
            "sudo",
            "sh",
            "-c",
            f"echo {value} > {str(full_path)}"
        ]
        subprocess.run(args, check=True)

    @property
    def gr_clock(self):
        """Control the core clock frequency"""
        return self._read_clock_file("cur_freq")

    @gr_clock.setter
    def gr_clock(self, new_clock):
        self._write_railgate_file(0)
        cur_clock = self._read_clock_file("cur_freq")
        # system will ignore if we set new min higher than current max, or vice versa
        if new_clock > cur_clock:
            self._write_clock_file("max_freq", new_clock)
            self._write_clock_file("min_freq", new_clock)
        elif new_clock < cur_clock:
            self._write_clock_file("min_freq", new_clock)
            self._write_clock_file("max_freq", new_clock)
        # wait for the new clock to be applied
        while (self._read_clock_file("cur_freq") != new_clock):
            time.sleep(.001)

    def reset_clock(self):
        """Reset the core clock frequency to the original values"""
        self._write_clock_file("min_freq", self.default_min_gr_clock)
        self._write_clock_file("max_freq", self.default_max_gr_clock)
        self._write_railgate_file(self.default_railgate_status)

    def __del__(self):
        # restore original core clocks, if changed
        if self.has_changed_clocks:
            self.reset_clock()

    def read_gpu_temp(self):
        """Read GPU temperature"""
        with open(self.gpu_temp_path + "/temp") as fp:
            temp = int(fp.read())
        return temp / 1000

    def read_gpu_power(self):
        """Read the current and voltage to calculate and return the power int watt"""

        result_cur = subprocess.run(["sudo", "cat", f"{self.gpu_power_path}/curr{self.gpu_channel}_input"], capture_output=True, text=True)
        current = int(result_cur.stdout.strip()) / 1000
        result_vol = subprocess.run(["sudo", "cat", f"{self.gpu_power_path}/in{self.gpu_channel}_input"], capture_output=True, text=True)
        voltage = int(result_vol.stdout.strip()) / 1000

        return current * voltage


class TegraObserver(BenchmarkObserver):
    """Observer that uses /sys/ to monitor and control graphics clock frequencies on a Tegra device.

    :param observables: List of quantities should be observed during tuning, supported is: "core_freq"
    :type observables: list of strings

    :param save_all: If set to True, all data collected by the TegraObserver for every iteration during benchmarking will be returned.
    If set to False, data will be aggregated over multiple iterations during benchmarking. False by default.
    :type save_all: boolean

    """

    def __init__(
        self,
        observables,
        save_all=False,
        power_path="",
        temp_path=""
    ):
        """Create a TegraObserver"""
        self.tegra = tegra(power_path=power_path, temp_path=temp_path)
        self.save_all = save_all
        self._set_units = False

        supported = ["core_freq", "tegra_temp", "tegra_power", "tegra_energy"]
        for obs in observables:
            if obs not in supported:
                raise ValueError(f"Observable {obs} not in supported: {supported}")
        self.observables = observables

        # Observe power measurements with the continuous observer
        self.measure_power = False
        self.needs_power = ["tegra_power", "tegra_energy"]
        if any([obs in self.needs_power for obs in observables]):
            self.measure_power = True
            power_observables = [obs for obs in observables if obs in self.needs_power]
            self.continuous_observer = ContinuousObserver("tegra", power_observables, self, continuous_duration=3)

        # remove power observables
        self.observables = [obs for obs in observables if obs not in self.needs_power]

        self.results = {}
        for obs in self.observables:
            self.results[obs + "s"] = []

        self.during_obs = [
            obs
            for obs in observables
            if obs in ["core_freq", "tegra_temp"]
        ]

        self.iteration = {obs: [] for obs in self.during_obs}


    def read_power(self):
        return self.tegra.read_gpu_power()


    def before_start(self):
        # clear results of the observables for next measurement
        self.iteration = {obs: [] for obs in self.during_obs}
        # Set the power unit to Watts
        if self._set_units == False:
            self.dev.units["power"] = "W"
            self._set_units = True

    def after_start(self):
        self.t0 = time.perf_counter()
        # ensure during is called at least once
        self.during()

    def during(self):
        if "core_freq" in self.observables:
            self.iteration["core_freq"].append(self.tegra.gr_clock)
        if "gpu_temp" in self.observables:
            self.iteration["gpu_temp"].append(self.tegra.read_gpu_temp())

    def after_finish(self):
        if "core_freq" in self.observables:
            self.results["core_freqs"].append(np.average(self.iteration["core_freq"]))
        if "gpu_temp" in self.observables:
            self.results["gpu_temps"].append(np.average(self.iteration["gpu_temp"]))

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


def get_tegra_gr_clocks(n=None, quiet=False):
    """Get tunable parameter for Tegra graphics clock, n is desired number of values."""
    d = tegra()
    gr_clocks = d.supported_gr_clocks

    if n and (len(gr_clocks) > n):
        indices = np.array(np.ceil(np.linspace(0, len(gr_clocks) - 1, n)), dtype=int)
        gr_clocks = np.array(gr_clocks)[indices]

    tune_params = dict()
    tune_params["tegra_gr_clock"] = list(gr_clocks)

    if not quiet:
        print("Using gr frequencies:", tune_params["tegra_gr_clock"])
    return tune_params
