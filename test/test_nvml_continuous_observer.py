import types

import pytest

from kernel_tuner.observers.nvml import NVMLContinuousObserver, NVMLObserver
from kernel_tuner.observers.observer import ContinuousObserver


class FakeParent:
    """An NVMLObserver whose device reports a fixed power, and an energy counter that is updated every 100 ms."""

    def __init__(self, clock, power_w=200.0, counter_power_w=250.0, has_energy_counter=True):
        self.clock = clock
        self.power_w = power_w
        self.counter_power_w = counter_power_w
        self.has_energy_counter = has_energy_counter
        self.nvml = types.SimpleNamespace(energy_usage=self.energy_usage)

    def energy_usage(self):
        if not self.has_energy_counter:
            raise RuntimeError("not supported")
        last_update = int(self.clock[0] * 10) / 10  # the counter is updated every 100 ms
        return int(last_update * self.counter_power_w * 1e3)  # millijoule

    def read_power(self):
        return self.power_w

    def before_start(self):
        pass

    def after_start(self):
        pass

    def during(self):
        pass

    def after_finish(self):
        pass

    def get_results(self):
        return {}


def measure(parent, clock, duration, kernel_time_ms=0.5):
    """Run the hooks of the observer as core does, polling every millisecond for `duration` seconds."""
    observer = NVMLContinuousObserver("nvml", ["nvml_power", "nvml_energy"], parent)
    observer.results = {"time": kernel_time_ms}
    observer.before_start()
    observer.after_start()
    end = clock[0] + duration
    while clock[0] < end:
        observer.during()
        clock[0] += 1e-3
    observer.after_finish()
    return observer.get_results()


@pytest.fixture
def clock(monkeypatch):
    clock = [10.05]
    monkeypatch.setattr("kernel_tuner.observers.nvml.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("kernel_tuner.observers.observer.time.perf_counter", lambda: clock[0])
    return clock


def test_power_from_energy_counter(clock):
    results = measure(FakeParent(clock), clock, duration=1.0)
    # the power readings (200 W) are only a fallback, the energy counter (250 W) is used
    assert results["nvml_power"] == pytest.approx(250.0, rel=0.02)
    assert results["nvml_energy"] == pytest.approx(250.0 * 0.5e-3, rel=0.02)


def test_falls_back_to_power_readings_without_energy_counter(clock):
    results = measure(FakeParent(clock, has_energy_counter=False), clock, duration=1.0)
    assert results["nvml_power"] == pytest.approx(200.0)
    assert results["nvml_energy"] == pytest.approx(200.0 * 0.5e-3)


def test_falls_back_to_power_readings_when_the_counter_is_wrong(clock):
    # the counter increases about 15 times too slowly, as on an RTX A6000 with driver 595.91.07
    with pytest.warns(UserWarning, match="energy counter"):
        results = measure(FakeParent(clock, counter_power_w=200.0 / 15), clock, duration=1.0)
    assert results["nvml_power"] == pytest.approx(200.0)


def test_energy_counter_is_opt_in():
    assert type(NVMLObserver(["nvml_power"]).continuous_observer) is ContinuousObserver
    assert type(NVMLObserver(["nvml_power"], use_energy_counter=True).continuous_observer) is NVMLContinuousObserver


def test_falls_back_to_power_readings_when_the_window_is_too_short(clock):
    # one change of the counter is not enough to know how long it took to use that energy
    results = measure(FakeParent(clock), clock, duration=0.15)
    assert results["nvml_power"] == pytest.approx(200.0)
