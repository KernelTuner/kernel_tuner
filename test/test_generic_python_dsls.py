"""End-to-end tests for tuning kernels written in Python DSLs with lang="generic_python"."""

import importlib
import pickle

import numpy as np
import pytest

from kernel_tuner import tune_kernel
from kernel_tuner.backends.generic_python import GenericPythonFunctions
from kernel_tuner.backends.generic_python_processes import (
    PROCESS_BUILD_DSLS,
    BuildProcessPool,
    _picklable_exception_chain,
    _raise_exception_chain,
)

from .context import (
    skip_if_no_cupy,
    skip_if_no_cute,
    skip_if_no_cutile,
    skip_if_no_numba_cuda,
    skip_if_no_taichi,
    skip_if_no_tilelang,
    skip_if_no_tilus,
    skip_if_no_torch,
    skip_if_no_triton,
    skip_if_no_warp,
)

# every DSL needs Torch with a CUDA device, which the generic_python backend uses for memory and timing
dsls = [
    pytest.param("numba", marks=[skip_if_no_torch, skip_if_no_numba_cuda]),
    pytest.param("cupyx", marks=[skip_if_no_torch, skip_if_no_cupy]),
    pytest.param("warp", marks=[skip_if_no_torch, skip_if_no_warp]),
    pytest.param("taichi", marks=[skip_if_no_torch, skip_if_no_taichi]),
    pytest.param("cute", marks=[skip_if_no_torch, skip_if_no_cute]),
    pytest.param("triton", marks=[skip_if_no_torch, skip_if_no_triton]),
    pytest.param("tilus", marks=[skip_if_no_torch, skip_if_no_tilus]),
    pytest.param("tilelang", marks=[skip_if_no_torch, skip_if_no_tilelang]),
    pytest.param("cutile", marks=[skip_if_no_torch, skip_if_no_cutile]),
]


def tune_vector_add(dsl, parallel_compile):
    """Tune the vector_add kernel of a DSL and check that every configuration is valid."""
    # the kernel modules import their DSL, so only import them when the DSL is installed
    kernel = importlib.import_module(f".generic_python_kernels.{dsl}_vector_add", package=__package__)
    if getattr(kernel, "requires_skip", None):
        pytest.skip(kernel.requires_skip)

    n = 4096
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros_like(a)
    args = kernel.arguments(c, a, b, n)
    answer = [a + b if arg is c else None for arg in args]
    tune_params = kernel.tune_params(n)

    # the language and call function are detected from the kernel source
    results, _ = tune_kernel(
        kernel.kernel_name, kernel.__file__, n, args, tune_params, answer=answer, parallel_compile=parallel_compile
    )

    num_configs = np.prod([len(v) for v in tune_params.values()])
    assert len(results) == num_configs
    # verification raises an exception on wrong results, so every configuration must have a valid time
    failed = [result for result in results if not isinstance(result.get("time"), float)]
    assert not failed, f"configurations failed: {failed}"
    return results


@pytest.mark.parametrize("parallel_compile", [None, 4], ids=["sequential", "parallel_compile"])
@pytest.mark.parametrize("dsl", dsls)
def test_generic_python_dsl(dsl, parallel_compile):
    tune_vector_add(dsl, parallel_compile)


@pytest.mark.parametrize("dsl", [dsl for dsl in dsls if dsl.values[0] in PROCESS_BUILD_DSLS])
def test_generic_python_dsl_process_builds(dsl, monkeypatch):
    """DSLs that cannot compile in parallel threads compile in worker processes, the main process loads them."""
    built = []
    original_build = BuildProcessPool.build

    def recording_build(self, *args, **kwargs):
        built.append(original_build(self, *args, **kwargs))
        return built[-1]

    monkeypatch.setattr(BuildProcessPool, "build", recording_build)
    results = tune_vector_add(dsl, parallel_compile=2)

    # the first configuration is benchmarked twice, for warming up the GPU, but only built once
    assert len(built) == len(results)
    assert all(built), "some kernels were not compiled by a worker process"


@skip_if_no_torch
@skip_if_no_numba_cuda
def test_generic_python_process_builds_fallback(caplog):
    """Kernels are compiled in the main process when the call function cannot be sent to worker processes."""
    from .generic_python_kernels import numba_vector_add as kernel

    n = 4096
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros_like(a)

    def call_numba_local(kernel_function, args, kwargs, grid, threads):
        from numba import cuda

        kernel_function[grid, threads](*[cuda.as_cuda_array(arg) for arg in args[:3]], args[3])

    results, _ = tune_kernel(
        kernel.kernel_name, kernel.__file__, n, [c, a, b, n], {"block_size_x": [128, 256, 512]},
        answer=[a + b, None, None, None], call_function=call_numba_local, parallel_compile=2,
    )
    assert all(isinstance(result.get("time"), float) for result in results)
    assert "Could not compile kernels in worker processes" in caplog.text


@skip_if_no_torch
@skip_if_no_numba_cuda
def test_generic_python_process_builds_not_stored(caplog):
    """Kernels are compiled in the main process when the DSL cannot store the kernels compiled by workers."""
    from .generic_python_kernels import numba_vector_add_fp16 as kernel

    n = 4096
    a = np.random.randn(n).astype(np.float16)
    b = np.random.randn(n).astype(np.float16)
    c = np.zeros_like(a)

    results, _ = tune_kernel(
        kernel.kernel_name, kernel.__file__, n, [c, a, b, n], {"block_size_x": [128, 256, 512]},
        answer=[a + b, None, None, None], atol=1e-2, parallel_compile=2,
    )
    assert all(isinstance(result.get("time"), float) for result in results)
    assert "Could not store kernels compiled by worker processes" in caplog.text


def test_worker_exception_chain():
    """Errors raised in worker processes are raised in the main process with the exceptions they came from."""

    class Unpicklable(Exception):
        def __init__(self, message, handle):
            super().__init__(message)
            self.handle = lambda: handle

    error = wrapped(Unpicklable("too many registers", 1), RuntimeError("compilation failed"))
    chain = pickle.loads(pickle.dumps(_picklable_exception_chain(error)))
    assert [type(e) for e in chain] == [RuntimeError, RuntimeError]
    assert str(chain[1]) == "Unpicklable: too many registers"

    with pytest.raises(RuntimeError) as raised:
        _raise_exception_chain(chain)
    assert GenericPythonFunctions.classify_compile_exception(None, raised.value) == "resource_error"

    # exceptions that lose their message when pickled, like nvJitLinkError, are replaced to keep the message
    error = LosesMessage(4, "\nnvJitLink error log: ptxas error   : Entry function uses too much shared data")
    chain = pickle.loads(pickle.dumps(_picklable_exception_chain(wrapped(error))))
    assert "too much shared data" in str(chain[1])
    with pytest.raises(RuntimeError) as raised:
        _raise_exception_chain(chain)
    assert GenericPythonFunctions.classify_compile_exception(None, raised.value) == "resource_error"

    # exception types are kept when possible, they are used to classify the error
    chain = pickle.loads(pickle.dumps(_picklable_exception_chain(wrapped(NameError("name 'x' is not defined")))))
    with pytest.raises(RuntimeError) as raised:
        _raise_exception_chain(chain)
    assert isinstance(raised.value.__cause__, NameError)
    assert GenericPythonFunctions.classify_compile_exception(None, raised.value) == "user_error"


class LosesMessage(Exception):
    """Like nvJitLinkError, only the status is pickled, not the error log that is part of the message."""

    def __init__(self, status, log=""):
        super().__init__(f"ERROR_PTX_COMPILE ({status}){log}")
        self.status = status

    def __reduce__(self):
        return (type(self), (self.status,))


def wrapped(error, wrapper=None):
    """Raise error and wrap it the way DSLs and call functions do, returning the wrapper."""
    wrapper = wrapper or RuntimeError("wrapper without details")
    try:
        try:
            raise error
        except Exception as e:
            raise wrapper from e
    except Exception as e:
        return e


# messages taken from the errors DSLs raise for configurations that use too many resources
resource_errors = [
    RuntimeError("out of resource: shared memory, Required: 163840, Hardware limit: 101376."),  # Triton
    ValueError("The register usage (256) of given config is too high."),  # Tilus
    RuntimeError("No valid schedule found during building programs:"),  # Tilus
    RuntimeError("Failed to set the allowed dynamic shared memory size to 131072"),  # TileLang
    RuntimeError("No valid warp partition for T.gemm: M=32, N=32 cannot be evenly covered by 16 warps"),  # TileLang
    RuntimeError("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: This indicates that a launch did not occur"),  # Numba
    RuntimeError("ERROR_PTX_COMPILE (4)\nnvJitLink error log: ptxas error   : Entry function uses too much shared data"),
    RuntimeError("error[CUDA_LAUNCH_INVALID_CONFIG]: CUDA launch failed: cudaErrorInvalidValue (1)"),  # CuTe
    RuntimeError("error: cudaErrorInvalidConfiguration (error code: 9)"),  # CuTe
    TypeError("Invalid configuration, CuTe compile failed with TypeError"),  # resource message wins over type
    RuntimeError("Invalid argument \"shape\" of load(): Dimension #0 of shape (96,) is not a power of two"),  # cuTile
    RuntimeError("`tileiras` compiler exceeded timeout 30s. Using a smaller tile size may reduce compilation time."),
    wrapped(RuntimeError("out of resource: shared memory")),  # message only in the chained exception
]

user_errors = [
    NameError("name 'num_warps' is not defined"),  # code errors win over resource keywords
    SyntaxError("invalid syntax"),
    AttributeError("module 'triton.language' has no attribute 'lod'"),
    TypeError("vector_add() takes 4 positional arguments but 5 were given"),
    RuntimeError("Type mismatch: float32 and int32"),
    RuntimeError("Undefined variable undefined_name used"),  # cuTile
    wrapped(NameError("name 'x' is not defined")),
    wrapped(TypeError("unsupported operand")),
]

unknown_errors = [
    # words in file paths or source lines, such as "cuda" and "ast", should not affect classification
    RuntimeError("illegal memory access in /site-packages/numba_cuda/cudadrv/driver.py, last call failed"),
    ValueError("expected a positive value"),
    # a problem with the installation, not with the configuration
    RuntimeError("ERROR_OUTDATED_LIBRARY (14)\nnvJitLink error log: ERROR NVVM_ERROR_INVALID_INPUT (4)"),
    wrapped(ValueError("something went wrong"), RuntimeError("")),
]


@pytest.mark.parametrize("error", resource_errors, ids=repr)
def test_classify_compile_exception_resource(error):
    assert GenericPythonFunctions.classify_compile_exception(None, error) == "resource_error"


@pytest.mark.parametrize("error", user_errors, ids=repr)
def test_classify_compile_exception_user(error):
    assert GenericPythonFunctions.classify_compile_exception(None, error) == "user_error"


@pytest.mark.parametrize("error", unknown_errors, ids=repr)
def test_classify_compile_exception_unknown(error):
    assert GenericPythonFunctions.classify_compile_exception(None, error) == "unknown"
