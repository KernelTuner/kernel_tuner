import subprocess
import time
from pathlib import Path

import numpy as np

from kernel_tuner.observers.observer import BenchmarkObserver


class tegra:
    """Class that gathers the Tegra functionality for one device."""

    def __init__(self):
        """Create object to control GPU core clock on a Tegra device."""

        self.dev_path = self.get_dev_path()
        self.default_min_gr_clock = self._read_clock_file("min_freq")
        self.default_max_gr_clock = self._read_clock_file("max_freq")
        self.supported_gr_clocks = self._read_clock_file("available_frequencies")

        self.default_railgate_status = self._read_railgate_file()

    @staticmethod
    def get_dev_path():
        """Get the path to device core clock control in /sys"""
        root_path = Path("/sys/devices/gpu.0")
        gpu_id = root_path.readlink()
        return root_path / Path("devfreq") / gpu_id

    def _read_railgate_file(self):
        """Read railgate status"""
        with open(self.dev_path / Path("device/railgate_enable")) as fp:
            data = int(fp.read().strip())
        return data

    def _write_railgate_file(self, value):
        """Set railgate status"""
        if value not in (0, 1):
            raise ValueError(f"Illegal governor value {value}, must be 0 or 1")
        print(f"Writing {value} to railgate file")
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
        # restore original core clocks
        self.reset_clock()


class TegraObserver(BenchmarkObserver):
    """Observer that uses /sys/ to monitor and control graphics clock frequencies on a Tegra device.

    :param observables: List of quantities should be observed during tuning, supported is: "core_freq"
    :type observables: list of strings

    :param device: Device ordinal used to identify your device, typically 0
    :type device: integer

    :param save_all: If set to True, all data collected by the TegraObserver for every iteration during benchmarking will be returned.
    If set to False, data will be aggregated over multiple iterations during benchmarking. False by default.
    :type save_all: boolean

    """

    def __init__(
        self,
        observables,
        device=0,
        save_all=False
    ):
        """Create a TegraObserver"""
        self.tegra = tegra()
        self.save_all = save_all

        supported = ["core_freq"]
        for obs in observables:
            if obs not in supported:
                raise ValueError(f"Observable {obs} not in supported: {supported}")
        self.observables = observables

        self.results = {}
        for obs in self.observables:
            self.results[obs + "s"] = []

        self.during_obs = [
            obs
            for obs in observables
            if obs in ["core_freq"]
        ]

        self.iteration = {obs: [] for obs in self.during_obs}

    def before_start(self):
        # clear results of the observables for next measurement
        self.iteration = {obs: [] for obs in self.during_obs}

    def after_start(self):
        # ensure during is called at least once
        self.during()

    def during(self):
        if "core_freq" in self.observables:
            self.iteration["core_freq"].append(self.tegra.gr_clock)

    def after_finish(self):
        if "core_freq" in self.observables:
            self.results["core_freqs"].append(np.average(self.iteration["core_freq"]))

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


def get_tegra_gr_clocks(device=0, n=None, quiet=False):
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
