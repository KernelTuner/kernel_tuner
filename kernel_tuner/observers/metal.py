import subprocess
from warnings import warn
import plistlib
import logging
import time
import numpy as np
from kernel_tuner.observers import BenchmarkObserver, ContinuousObserver

logger = logging.getLogger(__name__)

SUPPORTED_OBSERVABLES = [
    "energy",
    "power",
    "freq_hz",
    "occupancy",
]


class MetalDevice:
    """Wrapper for Metal GPU device metrics via powermetrics."""

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.process = None
        self._buffer = b""
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
            cmd_check = [
                "sudo",
                "-n",
                "powermetrics",
                "-n",
                "1",
                *self.cmd_powermetrics_options
            ]
            subprocess.run(
                cmd_check, 
                capture_output=True, 
                text=True, 
                check=True
            )
        except subprocess.CalledProcessError as e:
            if "password is required" in e.stderr:
                logger.warning(f"Metal observers ideally have non-interactive sudo privileges for powermetrics; {e}, {e.stderr}")
                warn("Metal observers ideally have non-interactive sudo privileges for powermetrics. Please run `sudo visudo` and add to the bottom: `your_username ALL=(ALL) NOPASSWD: /usr/bin/powermetrics`")
                return False
        return True

    def start_sampling(self):
        """Start the powermetrics process for continuous sampling."""

        # execute the powermetrics command
        cmd = [
            "sudo",
            "powermetrics",
            *self.cmd_powermetrics_options
        ]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            logger.error(f"Failed to start powermetrics: {e}")
            raise e
        self._buffer = b""
        # Give it a moment to start and produce first sample
        time.sleep((self.interval_ms / 1e3) * 2)

    def stop_sampling(self):
        """Stop the powermetrics process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=(self.interval_ms / 1e3) * 3)
            except subprocess.TimeoutExpired:
                logger.warning("powermetrics process did not terminate in time")
            self.process = None

    def read_sample(self):
        """Read and parse a single sample from powermetrics output."""
        if not self.process:
            return None

        # Read available data
        data = self.process.stdout.read1(4096)
        if not data:
            return None

        self._buffer += data

        # Try to extract complete plist documents from buffer
        # plist documents end with </plist>
        while b"</plist>" in self._buffer:
            end_idx = self._buffer.index(b"</plist>") + len(b"</plist>")
            plist_data = self._buffer[:end_idx]
            self._buffer = self._buffer[end_idx:]

            try:
                data = plistlib.loads(plist_data)
                return self._parse_sample(data)
            except (plistlib.InvalidFileException, KeyError) as e:
                logger.error(f"Failed to parse powermetrics output: {e}")
                continue

        return None

    def _parse_sample(self, data):
        """Parse a single powermetrics plist sample."""
        gpu_data = data.get("gpu", {})
        return {
            "gpu_energy_mj": gpu_data.get("gpu_energy"),  # milijoules over interval
            "freq_hz": gpu_data.get("freq_hz") * 1e6,  # Convert MHz to Hz
            "elapsed_ms": data.get("elapsed_ns", self.interval_ms * 1e6) / 1e6,
            "idle_ratio": gpu_data.get("idle_ratio", np.nan),
        }

    def get_power_metrics(self):
        """Get current power metrics from a single sample."""
        sample = self.read_sample()
        if not sample:
            return None

        # gpu_energy is energy in milijoules over the elapsed interval
        # Power (mW) = energy (mJ) / elapsed_ms * 1e3
        power_w = sample["gpu_energy_mj"] / (sample["elapsed_ms"] * 1e3)

        # Energy (J) for this interval
        energy_j = sample["gpu_energy_mj"] * 1e-3

        return {
            "power": power_w,
            "energy": energy_j,
            "freq_hz": sample["freq_hz"],
            "occupancy": 1.0 - sample["idle_ratio"] if not np.isnan(sample["idle_ratio"]) else np.nan,
        }


class MetalContinuousObserver(ContinuousObserver):
    """Continuous observer for Metal that wraps a parent observer."""

    def __init__(self, parent, continuous_duration=1.0, interval_ms=100):
        self.parent = parent
        self.continuous_duration = continuous_duration
        self.interval_ms = interval_ms
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


class MetalObserver(BenchmarkObserver):
    """
    BenchmarkObserver that uses powermetrics to monitor Apple Metal GPUs and measure
    energy usage (`energy`), power (`power`), GPU frequency (`freq_hz`), and occupancy (`occupancy`).
    """

    def __init__(
        self,
        observables=["energy"],
        *,
        device_id=None,
        prefix="metal",
        use_continuous_observer=True,
        continuous_duration=1.0,
        interval_ms=100,
    ):
        """
        Initialize the MetalObserver.

        Supported observables are: `energy`, `power`, `freq_hz`, and `occupancy`.

        :param observables: List of metrics to monitor. Defaults to just energy.
        :param device_id: Not used for Metal (single GPU), kept for API compatibility.
        :param prefix: Prefix used for name in the metrics. Defaults to "metal".
        :param use_continuous_observer: Whether to use continuous observer.
        :param continuous_duration: Duration in seconds for continuous observation.
        :param interval_ms: Sampling interval in milliseconds for powermetrics. Going below 100ms may cause instability issues.
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
        self.interval_ms = interval_ms
        self.results_per_iteration = {self.field_name(k): [] for k in self.observables}

    def register_device(self, dev):
        """Initialize the Metal device for monitoring."""
        self.device = MetalDevice(interval_ms=self.interval_ms)

        if self.use_continuous_observer:
            self.continuous_observer = MetalContinuousObserver(
                self, continuous_duration=self.continuous_duration
            )

    def after_start(self):
        """Start sampling after kernel launch."""
        self.device.start_sampling()
        self.sample_timestamps = []
        self.sample_values = {k: [] for k in self.results_per_iteration}
        self.collect_sample()

    def during(self):
        """Sample metrics during kernel execution."""
        self.collect_sample()

    def field_name(self, name):
        if self.prefix:
            return f"{self.prefix}_{name}"
        else:
            return name

    def after_finish(self):
        """Stop sampling and compute final metrics."""
        self.device.stop_sampling()
        self.sample_metrics()

    def collect_sample(self):
        """Collect a single sample from the Metal device."""
        metrics = self.device.get_power_metrics()
        if metrics:
            self.sample_timestamps.append(time.perf_counter())
            for key, value in metrics.items():
                if key in self.observables:
                    self.sample_values[self.field_name(key)].append(value)

    def sample_metrics(self):
        """Process collected samples and compute integrated metrics."""
        if not self.sample_timestamps:
            return

        # Normalize timestamps to [0, 1] for integration
        xs = np.array(self.sample_timestamps)
        if xs.max() > xs.min():
            xs = (xs - xs.min()) / (xs.max() - xs.min())
        else:
            xs = np.zeros_like(xs)

        for key, values in self.sample_values.items():
            if not values:
                continue

            field_name = key.replace(f"{self.prefix}_", "") if self.prefix else key

            # Energy samples are already energy per interval (J), so sum them
            # Other metrics (power, freq, occupancy) are rates, so integrate (average) them
            if field_name == "energy":
                result = sum(values)
            elif all(v == values[0] for v in values):
                result = values[0]
            else:
                result = np.trapezoid(values, x=xs)

            self.results_per_iteration[key].append(result)

    def get_results(self):
        """Return averaged results across iterations."""
        results = dict()

        for key in list(self.results_per_iteration):
            if self.results_per_iteration[key]:
                results[key] = np.average(self.results_per_iteration[key])
                self.results_per_iteration[key] = []

        return results