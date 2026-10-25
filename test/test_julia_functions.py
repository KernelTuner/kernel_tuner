from warnings import warn
import numpy as np
import pytest

from kernel_tuner import run_kernel, tune_kernel
from kernel_tuner.backends.julia import JuliaFunctions
from kernel_tuner.core import KernelInstance, KernelSource

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



axpy_kernel_name = "axpy!"
axpy_kernel_string = r"""
    using KernelAbstractions

    @kernel function axpy!(
        c, a, b, alpha, n, ::Val{block_size_x} = Val(128)
    ) where {block_size_x}
        i = @index(Global)
        if i <= n
            c[i] += alpha * a[i] + b[i]
        end
    end
    """


@pytest.fixture
def axpy_env():
    """Arguments and expected output for a kernel that accumulates into c for the first n elements only."""
    size = 1000
    rng = np.random.default_rng(42)
    a = rng.standard_normal(size).astype(np.float32)
    b = rng.standard_normal(size).astype(np.float32)
    c = rng.standard_normal(size).astype(np.float32)
    alpha = np.float32(2.5)
    n = np.int32(900)

    expected = c.copy()
    expected[:n] += alpha * a[:n] + b[:n]

    return size, [c, a, b, alpha, n], expected


def to_julia_array(arr):
    """Convert a numpy array into a Julia Vector{Float32}."""
    from juliacall import Main as jl

    return jl.seval("x -> Vector{Float32}(x)")(arr)


@skip_if_no_julia
@pytest.mark.parametrize("input_type", ["numpy", "julia"])
def test_run_kernel_arguments_roundtrip(axpy_env, input_type):
    """Check that arrays and scalars are passed to and from a Julia kernel correctly."""
    size, args, expected = axpy_env
    host_args = [np.copy(arg) for arg in args]
    if input_type == "julia":
        args[:3] = [to_julia_array(arg) for arg in args[:3]]

    result = run_kernel(axpy_kernel_name, axpy_kernel_string, size, args, [("block_size_x", 128)], lang="julia")

    assert all(isinstance(r, np.ndarray) for r in result[:3])
    assert np.allclose(result[0], expected, atol=1e-6)
    # elements beyond n are only correct if n was passed correctly
    assert np.array_equal(result[0][args[4]:], host_args[0][args[4]:])
    assert np.array_equal(result[1], host_args[1])
    assert np.array_equal(result[2], host_args[2])
    assert result[3] == host_args[3]
    assert result[4] == host_args[4]
    # the user's arguments are not modified
    for arg, host_arg in zip(args, host_args):
        assert np.array_equal(np.asarray(arg), host_arg)


@skip_if_no_julia
@pytest.mark.parametrize("input_type", ["numpy", "julia"])
@pytest.mark.parametrize("answer_type", ["numpy", "julia"])
def test_tune_kernel_verifies_answer(axpy_env, input_type, answer_type):
    """Check output verification with numpy and Julia arrays, output is reset between configurations."""
    size, args, expected = axpy_env
    if input_type == "julia":
        args[:3] = [to_julia_array(arg) for arg in args[:3]]
    if answer_type == "julia":
        expected = to_julia_array(expected)
    answer = [expected, None, None, None, None]
    tune_params = [("block_size_x", [64, 128, 256])]

    result, _ = tune_kernel(
        axpy_kernel_name, axpy_kernel_string, size, args, tune_params, lang="julia", answer=answer, atol=1e-6
    )

    assert len(result) == 3
    assert all("__error__" not in r for r in result)


@skip_if_no_julia
def test_tune_kernel_detects_wrong_answer(axpy_env, tmp_path, monkeypatch):
    """Check that output verification fails when the kernel output does not match the answer."""
    monkeypatch.chdir(tmp_path)  # a failed verification leaves the kernel source file behind
    size, args, expected = axpy_env
    answer = [expected + 1, None, None, None, None]

    with pytest.raises(RuntimeError, match="verification failed"):
        tune_kernel(
            axpy_kernel_name, axpy_kernel_string, size, args, [("block_size_x", [64])], lang="julia", answer=answer
        )
