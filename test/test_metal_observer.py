import types

import pytest

from kernel_tuner.observers.metal import MetalContinuousObserver, MetalDevice, MetalObserver


def sample(energy_j, elapsed_s, idle_ratio=0.1, freq_hz=1e9):
    return {"energy_j": energy_j, "elapsed_s": elapsed_s, "idle_ratio": idle_ratio, "freq_hz": freq_hz}


def test_summarize_sums_energy_over_the_window():
    # powermetrics reports whole millijoules per sample, so the power must come from the totals
    samples = [sample(0.150, 0.050), sample(0.100, 0.050), sample(0.001, 0.0004)]
    summary = MetalDevice.summarize(samples)
    assert summary["energy"] == pytest.approx(0.251)
    assert summary["duration"] == pytest.approx(0.1004)
    assert summary["power"] == pytest.approx(0.251 / 0.1004)


def test_summarize_weights_by_duration():
    samples = [sample(0.1, 0.075, idle_ratio=0.0, freq_hz=1e9), sample(0.1, 0.025, idle_ratio=1.0, freq_hz=2e9)]
    summary = MetalDevice.summarize(samples)
    assert summary["load"] == pytest.approx(0.75)
    assert summary["freq_hz"] == pytest.approx(1.25e9)


def test_summarize_skips_idle_samples_unless_all_are_idle():
    busy, idle = sample(0.3, 0.05, idle_ratio=0.1), sample(0.01, 0.05, idle_ratio=0.99)
    assert MetalDevice.summarize([busy, idle], min_load=0.1)["power"] == pytest.approx(6.0)
    assert MetalDevice.summarize([idle], min_load=0.1)["power"] == pytest.approx(0.2)
    assert MetalDevice.summarize([]) is None


def test_continuous_observer_scales_energy_to_one_kernel():
    parent = MetalObserver.__new__(MetalObserver)  # skip the powermetrics permission check
    parent.prefix = None
    parent.observables = {"energy", "power"}
    parent.min_load = 0.0
    parent.results_per_iteration = {"energy": [], "power": []}
    parent.device = types.SimpleNamespace(stop_sampling=lambda: [sample(1.0, 0.5), sample(1.0, 0.5)])
    parent.after_finish()

    observer = MetalContinuousObserver(parent)
    observer.results = {"time": 0.5}  # milliseconds per kernel execution
    results = observer.get_results()
    assert results["power"] == pytest.approx(2.0)
    assert results["energy"] == pytest.approx(2.0 * 0.5e-3)
