import numpy as np
import pytest

from kernel_tuner import tune_kernel
from kernel_tuner.backends.julia_helper import julia_backend_available_cuda
from kernel_tuner.observers.julia_nvml import (
    JuliaNVML,
    JuliaNVMLObserver,
    NVMLError_NoPermission,
    NVMLError_NotSupported,
)
from kernel_tuner.observers.nvml import NVMLObserver

from .context import skip_if_no_julia

skip_if_no_julia_cuda = pytest.mark.skipif(not julia_backend_available_cuda(), reason="No NVIDIA GPU available")


class FakeLibrary:
    """Stands in for the Julia NVML functions, every setter raises NoPermission as for a user without root."""

    def __init__(self):
        self.app_clocks = {"graphics": 1800, "mem": 8001}

    def power_limit(self, dev):
        return 300000

    def applications_clock(self, dev, kind):
        return self.app_clocks[kind]

    def persistence_mode(self, dev):
        return 1

    def __getattr__(self, name):
        def no_permission(*args):
            raise NVMLError_NoPermission(4, name)

        return no_permission


def fake_device(use_locked_clocks=False):
    """A JuliaNVML for a device with two memory clocks, without starting Julia."""
    d = JuliaNVML.__new__(JuliaNVML)
    d.lib, d.dev, d.id, d.nvidia_smi = FakeLibrary(), None, 2, "nvidia-smi"
    d.pwr_limit_default, d.pwr_constraints = 300000, [100000, 300000]
    d.modified_clocks = d.modified_pwr_limit = False
    d._persistence_mode = 1
    d.gr_clock_default = d.applications_gr_clock = 1800
    d.mem_clock_default = d.applications_mem_clock = 8001
    d.supported_mem_clocks = [8001, 405]
    d.supported_gr_clocks = {8001: [2100, 1800, 1200], 405: [405, 210]}
    d.use_locked_clocks = use_locked_clocks
    d.locked_gr_clock = d.locked_mem_clock = 0
    return d


@pytest.fixture
def sudo_calls(monkeypatch):
    """Record the commands that would be run with sudo, and do not register exit handlers."""
    calls = []
    monkeypatch.setattr("kernel_tuner.observers.julia_nvml.subprocess.run", lambda args, check: calls.append(args))
    monkeypatch.setattr("kernel_tuner.observers.julia_nvml.atexit.register", lambda f: None)
    return calls


def test_is_an_nvml_observer():
    # core and the checks on tunable parameters starting with nvml_ look for an NVMLObserver
    assert issubclass(JuliaNVMLObserver, NVMLObserver)


def test_power_is_read_at_most_once_per_interval(monkeypatch):
    observer = JuliaNVMLObserver(["nvml_power"])
    readings = iter([100_000, 200_000])
    observer.nvml = type("FakeNVML", (), {"pwr_usage": lambda self: next(readings)})()
    clock = [10.0]
    monkeypatch.setattr("kernel_tuner.observers.julia_nvml.time.perf_counter", lambda: clock[0])
    assert observer.read_power() == 100
    clock[0] += observer.power_read_interval / 2
    assert observer.read_power() == 100
    clock[0] += observer.power_read_interval
    assert observer.read_power() == 200


def test_power_limit_falls_back_to_nvidia_smi(sudo_calls):
    d = fake_device()
    d.pwr_limit = 150000
    assert sudo_calls == [["sudo", "nvidia-smi", "-i", "2", "--power-limit=150"]]
    assert d.modified_pwr_limit
    with pytest.raises(ValueError):
        d.pwr_limit = 50000


def test_application_clocks_fall_back_to_nvidia_smi(sudo_calls):
    d = fake_device()
    d.gr_clock = 1200
    assert sudo_calls == [["sudo", "nvidia-smi", "-i", "2", "--applications-clocks=8001,1200"]]
    assert (d.applications_mem_clock, d.applications_gr_clock) == (8001, 1200)
    with pytest.raises(ValueError):
        d.gr_clock = 210  # only supported with the lower memory clock


def test_locked_clocks_fall_back_to_nvidia_smi(sudo_calls):
    d = fake_device(use_locked_clocks=True)
    d.set_clocks(8001, 1200)
    d.reset_clocks()
    assert [args[-1] for args in sudo_calls] == [
        "--lock-gpu-clocks=1200,1200",
        "--lock-memory-clocks=8001,8001",
        "--reset-gpu-clocks",
        "--reset-memory-clocks",
    ]


def test_restore_defaults_only_restores_what_was_changed(sudo_calls):
    d = fake_device()
    d.restore_defaults()
    assert sudo_calls == []
    d.lib.app_clocks["graphics"] = 1200  # as if the clocks were set by an earlier call
    d.modified_clocks = True
    d.restore_defaults()
    assert sudo_calls == [["sudo", "nvidia-smi", "-i", "2", "--applications-clocks=8001,1800"]]


@skip_if_no_julia
@skip_if_no_julia_cuda
def test_device_lookups_agree():
    d = JuliaNVML(device_id=0)
    assert d.uuid.startswith("GPU-")
    for other in [
        JuliaNVML(device_uuid=d.uuid),
        JuliaNVML(device_uuid=d.uuid.removeprefix("GPU-")),
        JuliaNVML(device_pci_bus=d.pci_bus),
    ]:
        assert other.id == d.id


@skip_if_no_julia
@skip_if_no_julia_cuda
def test_readings():
    d = JuliaNVML(device_id=0)
    assert d.pwr_usage() > 0
    assert d.energy_usage() > 0
    assert d.gr_clock > 0 and d.mem_clock > 0
    assert 0 < d.temperature < 120
    assert d.pwr_constraints[0] <= d.pwr_limit <= d.pwr_constraints[1]
    assert d.supported_mem_clocks == sorted(d.supported_mem_clocks, reverse=True)


@skip_if_no_julia
@skip_if_no_julia_cuda
def test_errors_are_translated():
    d = JuliaNVML(device_id=0)
    with pytest.raises((NVMLError_NoPermission, NVMLError_NotSupported)):
        # setting the power limit requires root, and is not supported on all GPUs
        d.lib.set_power_limit_b(d.dev, d.pwr_limit)


@skip_if_no_julia
@skip_if_no_julia_cuda
def test_tune_kernel_with_julia_nvml_observer():
    kernel_string = """
    using KernelAbstractions

    @kernel function vector_add!(c, a, b, n, ::Val{block_size_x} = Val(128)) where {block_size_x}
        i = @index(Global)
        if i <= n
            c[i] = a[i] + b[i]
        end
    end
    """
    size = 50_000_000
    a, b = np.random.randn(size).astype(np.float32), np.random.randn(size).astype(np.float32)
    args = [np.zeros_like(a), a, b, np.int32(size)]
    tune_params = [("block_size_x", [128, 256])]
    observables = ["nvml_power", "nvml_energy", "core_freq", "mem_freq", "temperature"]
    observer = JuliaNVMLObserver(observables, continuous_duration=0.5)

    results, _ = tune_kernel("vector_add!", kernel_string, size, args, tune_params, lang="julia", observers=[observer])

    from juliacall import Main as jl

    # the observer measures the GPU that the Julia backend runs on
    assert observer.nvml.uuid == "GPU-" + str(jl.seval("CUDA.uuid(CUDA.device())"))
    for r in results:
        for observable in observables:
            assert r[observable] > 0, f"{observable} not positive in {r}"
        # the energy of a single run of the kernel
        assert r["nvml_energy"] == pytest.approx(r["nvml_power"] * r["time"] / 1e3)
