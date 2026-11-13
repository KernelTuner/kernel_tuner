import logging
import os
import plistlib
import select
import signal
import subprocess
import time
from warnings import warn

import numpy as np

from kernel_tuner.observers import BenchmarkObserver, ContinuousObserver

logger = logging.getLogger(__name__)
PLIST_END = b"</plist>"

SUPPORTED_OBSERVABLES = [
    "energy",
    "power",
    "freq_hz",
    "load",
]

FIELD_TYPE_MAP = {
    "energy": float,
    "power": float,
    "freq_hz": int,
    "load": float,
}


class MetalDevice:
    """Wrapper for Metal GPU device metrics via powermetrics."""

    def __init__(self, interval_ms=50):
        self.interval_ms = interval_ms
        self.process = None
        self._samples = []
        self._buffer = b""
        self._window_start = None
        self.cmd_powermetrics_options = [
            "-i",
            str(self.interval_ms),
            "--samplers",
            "gpu_power",
            "-a",
            "0",
            "-f",
            "plist",
        ]
        self.has_noninteractive_permissions = self.check_permissions()

    def check_permissions(self) -> bool:
        """Check if the user has non-interactive sudo permissions to run powermetrics."""
        try:
            # check if we have non-interactive sudo rights on powermetrics
            cmd_check = ["sudo", "-n", "powermetrics", "-n", "1", *self.cmd_powermetrics_options]
            subprocess.run(cmd_check, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            if "password is required" in e.stderr:
                logger.warning(
                    f"Metal observers ideally have non-interactive sudo privileges for powermetrics; {e}, {e.stderr}"
                )
                warn(
                    "Metal observers ideally have non-interactive sudo privileges for powermetrics. Please run `sudo visudo` and add to the bottom: `your_username ALL=(ALL) NOPASSWD: /usr/bin/powermetrics`"
                )
                return False
        return True

    def start_sampling(self):
        """Start powermetrics and wait for its first sample, which marks the start of the measurement window.

        powermetrics emits a sample every ``interval_ms``. Each sample holds the energy used since the previous sample,
        so the samples after the first one together cover the measurement window without gaps.
        The output is read without blocking whenever the observer is called. No background thread is used, as that
        can deadlock when Kernel Tuner runs inside Julia (juliacall). If the pipe fills up in between, powermetrics
        waits, and its next sample covers the longer interval, so no energy is lost.
        """
        if self.process:  # left over from a measurement that was interrupted, e.g. by a failing kernel
            self.stop_sampling()
        cmd = ["sudo", "powermetrics", *self.cmd_powermetrics_options]
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.error(f"Failed to start powermetrics: {e}")
            raise e
        os.set_blocking(self.process.stdout.fileno(), False)
        self._buffer = b""
        self._samples = []

        # the first sample covers the time before the window, it only marks its start
        timeout = max(5.0, 10 * self.interval_ms / 1e3)  # starting sudo and powermetrics can take a while
        if not self._read_until(lambda: len(self._samples) >= 1, timeout):
            logger.error("powermetrics did not produce a sample in time")
        self._window_start = time.perf_counter()

    def read_available(self):
        """Read and parse the samples that powermetrics has written so far, without blocking."""
        if not self.process:
            return
        while True:
            try:
                data = os.read(self.process.stdout.fileno(), 65536)
            except BlockingIOError:
                break
            if not data:  # powermetrics has exited
                break
            self._buffer += data
        while PLIST_END in self._buffer:
            end = self._buffer.index(PLIST_END) + len(PLIST_END)
            document, self._buffer = self._buffer[:end], self._buffer[end:]
            start = document.find(b"<?xml")
            start = document.find(b"<plist") if start < 0 else start
            try:
                self._samples.append(self._parse_sample(plistlib.loads(document[start:])))
            except Exception as e:
                logger.error(f"Failed to parse powermetrics output: {e}")

    def _read_until(self, predicate, timeout):
        """Read samples until ``predicate()`` holds, or until ``timeout`` seconds have passed."""
        deadline = time.perf_counter() + timeout
        while True:
            self.read_available()
            if predicate():
                return True
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or self.process.poll() is not None:
                return False
            select.select([self.process.stdout.fileno()], [], [], min(remaining, 0.01))

    def stop_sampling(self):
        """Request a final sample up to now, stop powermetrics, and return the samples within the measurement window."""
        if not self.process:
            return []
        window_duration = time.perf_counter() - self._window_start

        # SIGIO makes powermetrics emit a sample immediately, so the window ends now instead of at the next interval
        try:
            self.process.send_signal(signal.SIGIO)
        except Exception as e:
            logger.error(f"Failed to send SIGIO to powermetrics: {e}")
        # wait until the samples cover the whole window (a periodic sample may arrive before the requested one)
        covered = lambda: (
            len(self._samples) > 1 and sum(s["elapsed_s"] for s in self._samples[1:]) >= window_duration - 2e-3
        )
        if not self._read_until(covered, timeout=max(1.0, 3 * self.interval_ms / 1e3)):
            logger.warning("powermetrics samples do not cover the full measurement window")

        self.process.terminate()
        try:
            self.process.wait(timeout=(self.interval_ms / 1e3) * 3)
        except subprocess.TimeoutExpired:
            logger.warning("powermetrics process did not terminate in time")
        self.process.stdout.close()
        self.process = None
        return self._samples[1:]

    def _parse_sample(self, data):
        """Parse a single powermetrics plist sample."""
        gpu_data = data.get("gpu", {})
        # units are documented in: https://docs.rs/pumas/latest/src/pumas/modules/powermetrics/plist_parsing.rs.html
        return {
            "energy_j": gpu_data.get("gpu_energy", 0) * 1e-3,  # integer millijoules since the previous sample
            "elapsed_s": data.get("elapsed_ns", self.interval_ms * 1e6) / 1e9,
            "freq_hz": gpu_data.get("freq_hz", np.nan) * 1e6,  # reported in MHz
            "idle_ratio": gpu_data.get("idle_ratio", np.nan),
        }

    @staticmethod
    def summarize(samples, min_load=0.0):
        """Combine the samples of a measurement window into the energy, average power, frequency, and load.

        Samples in which the GPU was mostly idle (load below ``min_load``) are left out, unless that would leave none.
        Energy is reported in whole millijoules per sample, so it is summed over the window before computing the power.
        """
        samples = [s for s in samples if s["elapsed_s"] > 0]
        busy = [s for s in samples if not (1.0 - s["idle_ratio"] < min_load)]
        samples = busy if busy else samples
        if not samples:
            return None
        duration = sum(s["elapsed_s"] for s in samples)
        energy = sum(s["energy_j"] for s in samples)
        weighted = lambda key: sum(s[key] * s["elapsed_s"] for s in samples) / duration
        return {
            "energy": energy,
            "power": energy / duration,
            "freq_hz": weighted("freq_hz"),
            "load": 1.0 - weighted("idle_ratio"),
            "duration": duration,
        }


class MetalContinuousObserver(ContinuousObserver):
    """Continuous observer for Metal that wraps a parent observer.

    Kernel Tuner runs the kernel back-to-back for ``continuous_duration`` seconds. The parent measures the average
    power over that run, and the energy per kernel execution is that power times the execution time of the kernel.
    """

    def __init__(self, parent, continuous_duration=1.0, interval_ms=50):
        self.parent = parent
        self.continuous_duration = continuous_duration
        self.interval_ms = interval_ms

        # This assigned by Kernel Tuner's core
        self.results = None

    def before_start(self):
        self.parent.before_start()

    def after_start(self):
        self.parent.after_start()

    def during(self):
        self.parent.during()

    def after_finish(self):
        self.parent.after_finish()

    def get_results(self):
        results = self.parent.get_results()
        energy_field = self.parent.field_name("energy")
        if energy_field in results:
            # the parent measured the energy of the entire run, scale it to a single kernel execution
            time_sec = self.results["time"] * 1e-3
            results[energy_field] = FIELD_TYPE_MAP["energy"](self.parent.last_power * time_sec)
        return results


class MetalObserver(BenchmarkObserver):
    """BenchmarkObserver that uses powermetrics to monitor Apple Metal GPUs and measure
    energy usage (`energy`), power (`power`), GPU frequency (`freq_hz`), and load (`load`).
    """

    def __init__(
        self,
        observables=SUPPORTED_OBSERVABLES,
        *,
        device_id=None,
        prefix=None,
        use_continuous_observer=True,
        continuous_duration=1.0,
        interval_ms=50,
        min_load=0.1,
    ):
        """Initialize the MetalObserver.

        Supported observables are: `energy`, `power`, `freq_hz`, and `load`.

        :param observables: List of metrics to monitor. Defaults to all as there is no performance penalty for monitoring all metrics.
        :param device_id: Not used for Metal (single GPU), kept for API compatibility.
        :param prefix: Prefix used for name in the metrics. Defaults to "metal".
        :param use_continuous_observer: Whether to use continuous observer. The continuous observer measures the average power while
            the kernel runs back-to-back for `continuous_duration` seconds, and reports the energy per kernel execution as that power
            times the execution time. Without it, each kernel execution is measured on its own, and the energy covers the whole
            measurement window of at least one sampling interval, which is only meaningful for kernels that run much longer than `interval_ms`.
        :param continuous_duration: Duration in seconds for continuous observation.
        :param interval_ms: Sampling interval in milliseconds for powermetrics. Default 50ms as per https://arxiv.org/html/2511.07885v6. Going below 50ms may cause instability issues.
        :param min_load: Minimum threshold for GPU load. Samples below this load are discarded.
        """
        for obs in observables:
            if obs not in SUPPORTED_OBSERVABLES:
                raise ValueError(f"Observable {obs} not supported: {SUPPORTED_OBSERVABLES}")

        self.observables = set(observables)
        self.prefix = prefix
        self.device_id = device_id
        self.device = None
        self.use_continuous_observer = use_continuous_observer
        self.passthrough_observer = (
            use_continuous_observer  # This observer is only used as a passthrough for the continuous observer
        )
        self.continuous_duration = continuous_duration
        self.interval_ms = interval_ms
        self.min_load = min_load
        self.results_per_iteration = {self.field_name(k): [] for k in self.observables}

        if min_load < 0.0 or min_load > 1.0:
            raise ValueError(f"min_load must be between 0.0 and 1.0, got {min_load}")

    def register_device(self, dev):
        """Initialize the Metal device for monitoring."""
        self.device = MetalDevice(interval_ms=self.interval_ms)

        if self.use_continuous_observer:
            self.continuous_observer = MetalContinuousObserver(self, continuous_duration=self.continuous_duration)

    def field_name(self, name):
        if self.prefix:
            return f"{self.prefix}_{name}"
        else:
            return name

    def before_start(self):
        """Start the measurement window before the kernel is launched."""
        self.device.start_sampling()

    def during(self):
        """Read the samples written so far, so that the output pipe of powermetrics does not fill up."""
        self.device.read_available()

    def after_finish(self):
        """End the measurement window once the kernel has finished, and combine its samples."""
        summary = MetalDevice.summarize(self.device.stop_sampling(), self.min_load)
        if summary is None:
            logger.warning("No powermetrics samples were collected for this measurement")
            return
        self.last_power = summary["power"]
        for key in self.observables:
            self.results_per_iteration[self.field_name(key)].append(summary[key])

    def get_results(self):
        """Return averaged results across iterations."""
        results = dict()

        for key in self.results_per_iteration.keys():
            if self.results_per_iteration[key]:
                field_name = key.replace(f"{self.prefix}_", "") if self.prefix else key
                results[key] = FIELD_TYPE_MAP[field_name](np.average(self.results_per_iteration[key]))
                self.results_per_iteration[key] = []

        return results
