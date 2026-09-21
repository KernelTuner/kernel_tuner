from warnings import warn
import numpy as np
import pytest

from kernel_tuner import tune_kernel
from kernel_tuner.backends.julia import JuliaFunctions
from kernel_tuner.core import KernelInstance, KernelSource
from kernel_tuner.observers.metal import MetalObserver, SUPPORTED_OBSERVABLES

from .test_runners import env  # noqa: F401
from .context import skip_if_no_julia


kernel_name = "vector_add!"
kernel_string = r"""
    using KernelAbstractions

    @kernel function vector_add!(
        c, a, b, n, ::Val{block_size_x} = Val(128)
    ) where {block_size_x}
        i = @index(Global)
        if i <= n
            c[i] = a[i] + b[i]
        end
    end
    """


@skip_if_no_julia
def test_ready_argument_list():
    """Ensure Julia backend correctly converts arguments into Julia objects."""
    from juliacall import ValueBase

    size = 1000
    a = np.int32(75)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)

    arguments = [c, a, b]

    dev = JuliaFunctions(0)
    gpu_args = dev.ready_argument_list(arguments)

    # Julia Array maps back through PythonCall as pyjl_pointer-like proxies
    # Scalars remain scalars
    assert isinstance(gpu_args[0], ValueBase)  # Julia GPU Array proxy
    assert isinstance(gpu_args[1], (int, np.int32))  # scalar unchanged
    assert isinstance(gpu_args[2], ValueBase)  # Julia GPU Array proxy


@skip_if_no_julia
def test_compile():
    """Check that Julia kernel code successfully compiles."""
    kernel_sources = KernelSource(kernel_name, kernel_string, "julia")
    kernel_instance = KernelInstance(kernel_name, kernel_sources, kernel_string, [], None, None, dict(), [])

    dev = JuliaFunctions(0)

    try:
        dev.compile(kernel_instance)
    except Exception as e:
        pytest.fail("Did not expect any exception: " + str(e))


@skip_if_no_julia
def test_tune_kernel(env):
    """Run a minimal Julia kernel tuner example."""
    env[0] = kernel_name
    env[1] = kernel_string
    env[4] = list(env[4].items()) # convert from a dict to a list of tuples to preserve order

    result, _ = tune_kernel(*env, lang="julia", verbose=True)

    assert len(result) > 0
    for r in result:
        if '__error__' in r:
            continue  # Skip configurations that failed to compile or run
        assert "time" in r
        assert r["time"] > 0

@skip_if_no_julia
def test_tune_kernel_observers(env):
    """Run a minimal Julia kernel tuner example with observers."""
    env[0] = kernel_name
    env[1] = kernel_string
    env[4] = list(env[4].items()) # convert from a dict to a list of tuples to preserve order

    observers = [MetalObserver(observables=SUPPORTED_OBSERVABLES)]

    result, _ = tune_kernel(*env, observers=observers, lang="julia", verbose=True)

    assert len(result) > 0
    for r in result:
        if '__error__' in r:
            continue  # Skip configurations that failed to compile or run
        assert "time" in r
        assert r["time"] > 0
        assert "metal_power" in r
        assert "metal_freq_hz" in r
        assert "metal_occupancy" in r
        assert "metal_energy" in r
        assert r["metal_power"] > 0
        assert r["metal_freq_hz"] > 0
        assert r["metal_occupancy"] > 0
        assert r["metal_energy"] > 0
