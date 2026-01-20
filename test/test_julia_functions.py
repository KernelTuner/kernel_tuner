import numpy as np
import pytest

from kernel_tuner import tune_kernel
from kernel_tuner.backends.julia import JuliaFunctions
from kernel_tuner.core import KernelInstance, KernelSource

from .test_runners import env  # noqa: F401
from .context import skip_if_no_julia
from juliacall import ValueBase

import subprocess

# try to auto-detect which backend is available
available_backend = None
try:
    subprocess.check_output("nvidia-smi")
    available_backend = "cuda"
except Exception:  # this command not being found can raise quite a few different errors depending on the configuration
    try:
        subprocess.check_output("rocm-smi")
        available_backend = "amd"
    except Exception:
        try:
            subprocess.check_output("intel_gpu_top -J")
            available_backend = "intel"
        except Exception:
            try:
                output = subprocess.check_output('system_profiler SPDisplaysDataType | grep "Metal"')
                if b"Metal Support" in output:
                    available_backend = "metal"
            except Exception:
                pass
if available_backend is None:
    warn("No supported GPU backend detected for Julia tests.")
                

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

    size = 1000
    a = np.int32(75)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)

    arguments = [c, a, b]

    dev = JuliaFunctions(0, compiler_options=[available_backend])
    gpu_args = dev.ready_argument_list(arguments)

    # Julia Array maps back through PythonCall as pyjl_pointer-like proxies
    # Scalars remain scalars
    assert isinstance(gpu_args[0], ValueBase)       # Julia GPU Array proxy
    assert isinstance(gpu_args[1], np.int32)        # scalar unchanged
    assert isinstance(gpu_args[2], ValueBase)       # Julia GPU Array proxy

@skip_if_no_julia
def test_compile():
    """Check that Julia kernel code successfully compiles."""
    kernel_sources = KernelSource(kernel_name, kernel_string, "julia")
    kernel_instance = KernelInstance(kernel_name, kernel_sources, kernel_string, [], None, None, dict(), [])

    dev = JuliaFunctions(0, compiler_options=[available_backend])

    try:
        dev.compile(kernel_instance)
    except Exception as e:
        pytest.fail("Did not expect any exception: " + str(e))

@skip_if_no_julia
def test_tune_kernel(env):
    """Run a minimal Julia kernel tuner example."""
    env[0] = kernel_name
    env[1] = kernel_string

    result, _ = tune_kernel(
        *env,
        lang="julia",
        verbose=True,
        compiler_options=[available_backend]
    )

    assert len(result) > 0
