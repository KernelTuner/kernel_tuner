"""Tests for kernel_tuner.decorator.autotune and the wisdom files in which it stores tuning results."""

import importlib
import json

import pytest

import kernel_tuner
from kernel_tuner import autotune
from kernel_tuner.decorator import AutotunedKernel, _configs_to_search_space
from kernel_tuner.kernel_sources.kernel_source_fn import KernelSourceFn
from kernel_tuner.utils import wisdom

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
    pytest.param("triton", marks=[skip_if_no_torch, skip_if_no_triton]),
    pytest.param("numba", marks=[skip_if_no_torch, skip_if_no_numba_cuda]),
    pytest.param("cupyx", marks=[skip_if_no_torch, skip_if_no_cupy]),
    pytest.param("warp", marks=[skip_if_no_torch, skip_if_no_warp]),
    pytest.param("taichi", marks=[skip_if_no_torch, skip_if_no_taichi]),
    pytest.param("cute", marks=[skip_if_no_torch, skip_if_no_cute]),
    pytest.param("tilus", marks=[skip_if_no_torch, skip_if_no_tilus]),
    pytest.param("tilelang", marks=[skip_if_no_torch, skip_if_no_tilelang]),
    pytest.param("cutile", marks=[skip_if_no_torch, skip_if_no_cutile]),
]


def kernel_module(dsl):
    """Import the module with the autotuned kernel of a DSL, these import their DSL."""
    module = importlib.import_module(f".generic_python_kernels.autotuned.{dsl}_kernel", package=__package__)
    if getattr(module, "requires_skip", None):
        pytest.skip(module.requires_skip)
    return module


def fresh_kernel(module, **options):
    """Decorate the DSL kernel of a module again, so that every test starts without tuning results."""
    return autotune(**{**module.AUTOTUNE, **options})(module.vector_add.kernel)


def vectors(n):
    import torch

    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    c = torch.zeros(n, device="cuda")
    return c, a, b


def launch_and_check(module, kernel, n, launch=None):
    import torch

    c, a, b = vectors(n)
    (launch or module.launch)(kernel, c, a, b, n)
    torch.cuda.synchronize()
    assert torch.allclose(c, a + b)


# tests that do not need a GPU ----------------------------------------------------------------------------------


def test_configs_to_search_space_dicts():
    tune_params, restriction = _configs_to_search_space([{"x": 1, "y": 8}, {"x": 4, "y": 16}, {"x": 4, "y": 8}])
    assert tune_params == {"x": [1, 4], "y": [8, 16]}
    assert restriction({"x": 1, "y": 8})
    assert restriction({"x": 4, "y": 16})
    assert not restriction({"x": 1, "y": 16})


def test_configs_to_search_space_triton_configs():
    class Config:  # the attributes of triton.Config that are used
        def __init__(self, kwargs, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None, pre_hook=None):
            self.kwargs = kwargs
            self.num_warps = num_warps
            self.num_stages = num_stages
            self.num_ctas = num_ctas
            self.maxnreg = maxnreg
            self.pre_hook = pre_hook

    tune_params, restriction = _configs_to_search_space(
        [Config({"BLOCK": 64}, num_warps=2), Config({"BLOCK": 128}, num_stages=4)]
    )
    assert tune_params == {"BLOCK": [64, 128], "num_warps": [2, 4], "num_stages": [3, 4]}
    assert restriction({"BLOCK": 64, "num_warps": 2, "num_stages": 3})
    assert not restriction({"BLOCK": 64, "num_warps": 4, "num_stages": 3})

    # launch options that only some configurations set get the default of Triton for the others
    tune_params, _ = _configs_to_search_space([Config({"BLOCK": 64}, num_ctas=2), Config({"BLOCK": 128})])
    assert tune_params["num_ctas"] == [2, 1]

    with pytest.raises(ValueError):
        _configs_to_search_space([Config({"BLOCK": 64}, pre_hook=lambda args: None)])


def test_configs_to_search_space_errors():
    with pytest.raises(ValueError):
        _configs_to_search_space([{"x": 1}, {"y": 2}])
    with pytest.raises(ValueError):
        _configs_to_search_space([])
    with pytest.raises(TypeError):
        _configs_to_search_space([(1, 2)])


def test_autotune_is_top_level():
    """The decorator is available next to tune_kernel."""
    import kernel_tuner.decorator

    assert kernel_tuner.autotune is kernel_tuner.decorator.autotune
    assert "autotune" in kernel_tuner.__all__


def test_autotune_needs_tune_params_or_configs():
    with pytest.raises(ValueError):
        autotune()
    with pytest.raises(ValueError):
        autotune({"x": [1]}, configs=[{"x": 1}])


def test_wisdom_files(tmp_path):
    filename = wisdom.wisdom_file(tmp_path, "vector_add")
    params = ["block_size_x", "num_warps"]
    results = [{"block_size_x": 128, "num_warps": 4, "time": 2.0}, {"block_size_x": 256, "num_warps": 4, "time": 1.0}]
    key = (4096, "torch.float32")
    wisdom.write_wisdom(filename, "vector_add", params, key, "GPU A", results)
    wisdom.write_wisdom(filename, "vector_add", params, key, "GPU B", results[:1])

    # the file follows the format of Kernel Launcher
    with open(filename) as handle:
        lines = [json.loads(line) for line in handle]
    assert lines[0] == {"tunable_parameters": params, "version": "1.0", "objective": "time", "key": "vector_add"}
    assert lines[1]["config"] == [128, 4]
    assert lines[1]["problem_size"] == [4096]
    assert lines[1]["environment"] == {"device_name": "GPU A"}

    records = wisdom.read_wisdom(filename, params)
    assert len(records) == 3
    assert wisdom.best_wisdom_config(records, key, "GPU A", params) == {"block_size_x": 256, "num_warps": 4}
    assert wisdom.best_wisdom_config(records, key, "GPU B", params) == {"block_size_x": 128, "num_warps": 4}
    assert wisdom.best_wisdom_config(records, key, "GPU C", params) is None
    assert wisdom.best_wisdom_config(records, (8192, "torch.float32"), "GPU A", params) is None

    # results for the same key and GPU are replaced, others are kept
    wisdom.write_wisdom(filename, "vector_add", params, key, "GPU A", [dict(results[0], time=0.5)])
    records = wisdom.read_wisdom(filename, params)
    assert len(records) == 2
    assert wisdom.best_wisdom_config(records, key, "GPU A", params) == {"block_size_x": 128, "num_warps": 4}

    # a file of a kernel with other tunable parameters is neither used nor overwritten
    assert wisdom.read_wisdom(filename, ["block_size_x"]) == []
    wisdom.write_wisdom(filename, "vector_add", ["block_size_x"], key, "GPU A", [{"block_size_x": 64, "time": 1.0}])
    assert len(wisdom.read_wisdom(filename, params)) == 2


@pytest.mark.parametrize(
    "import_statement, decorator",
    [
        ("from kernel_tuner import autotune", "autotune"),
        ("import kernel_tuner", "kernel_tuner.autotune"),
        ("import kernel_tuner as kt", "kt.autotune"),
        ("from kernel_tuner.decorator import autotune", "autotune"),
        ("import kernel_tuner.decorator as ktd", "ktd.autotune"),
    ],
)
def test_kernel_copies_drop_autotune_decorator(tmp_path, import_statement, decorator):
    """The modules created for each configuration contain the kernel without the autotune decorator."""
    source = tmp_path / "kernel.py"
    source.write_text(
        "import functools\n"
        f"{import_statement}\n"
        "\n"
        f"@{decorator}(tune_params={{'x': [1, 2]}})\n"
        "@functools.lru_cache\n"
        "def kernel(a):\n"
        "    return a * x\n"
    )
    kernel_source = KernelSourceFn("kernel", str(source), "generic_python", call_function=lambda f, args, kwargs: None)
    # only the other decorators are kept, x is replaced by the value of the tunable parameter
    assert len(kernel_source.source_tree.decorator_list) == 1
    kernel_function, _ = kernel_source.apply_params_to_source_fn({"x": 3})
    assert not isinstance(kernel_function, AutotunedKernel)
    assert kernel_function(2) == 6


# tests that tune kernels on a GPU ------------------------------------------------------------------------------


@pytest.mark.parametrize("dsl", dsls)
def test_autotune_dsl(dsl):
    """The kernel is tuned when it is launched with a new tuning key, and the best configuration is launched."""
    module = kernel_module(dsl)
    kernel = fresh_kernel(module)
    num_configs = len([config for config in _product(kernel.tune_params)])

    launch_and_check(module, kernel, 5000)
    assert len(kernel.tuning_results) == 1
    assert len(next(iter(kernel.tuning_results.values()))) == num_configs

    # the same key is not tuned again, a new key is
    launch_and_check(module, kernel, 5000)
    assert len(kernel.tuning_results) == 1
    launch_and_check(module, kernel, 7000)
    assert len(kernel.tuning_results) == 2
    for params in kernel.best_configs.values():
        assert all(params[name] in values for name, values in kernel.tune_params.items())


@pytest.mark.parametrize("dsl", [dsl for dsl in dsls if dsl.values[0] in ("warp", "cutile")])
def test_autotune_triton_style_launch(dsl):
    """cuTile and Warp kernels can also be launched like Triton kernels, besides with launch()."""
    module = kernel_module(dsl)
    kernel = fresh_kernel(module)
    launch_and_check(module, kernel, 5000, launch=module.launch_subscript)
    launch_and_check(module, kernel, 5000)
    assert len(kernel.tuning_results) == 1


@pytest.mark.parametrize("dsl", dsls)
def test_autotune_parallel_compile(dsl, monkeypatch):
    """Kernels are compiled in parallel while tuning, in worker processes for the DSLs that need these."""
    from kernel_tuner.backends.generic_python_processes import PROCESS_BUILD_DSLS, BuildProcessPool

    built = []
    original_build = BuildProcessPool.build

    def recording_build(self, *args, **kwargs):
        built.append(original_build(self, *args, **kwargs))
        return built[-1]

    monkeypatch.setattr(BuildProcessPool, "build", recording_build)
    module = kernel_module(dsl)
    kernel = fresh_kernel(module, parallel_compile=2)
    launch_and_check(module, kernel, 5000)
    assert len(next(iter(kernel.tuning_results.values()))) == len(list(_product(kernel.tune_params)))

    # the call function can be sent to worker processes, unless the launch dimensions are lambdas (Numba here)
    if dsl in PROCESS_BUILD_DSLS and dsl != "numba":
        assert built and all(built)


def _product(tune_params):
    import itertools

    return itertools.product(*tune_params.values())


@skip_if_no_torch
@skip_if_no_triton
def test_autotune_wisdom(tmp_path, monkeypatch):
    """Tuning results are stored in a wisdom file, which a new kernel object uses instead of tuning again."""
    module = kernel_module("triton")
    kernel = fresh_kernel(module, wisdom=tmp_path)
    launch_and_check(module, kernel, 5000)
    filename = wisdom.wisdom_file(tmp_path, "vector_add")
    records = wisdom.read_wisdom(filename, list(kernel.tune_params))
    assert len(records) == 4
    assert all(record["problem_size"] == [5000] for record in records)

    def no_tuning(*args, **kwargs):
        raise AssertionError("the kernel is tuned although the wisdom file has results")

    new_kernel = fresh_kernel(module, wisdom=tmp_path)
    monkeypatch.setattr(new_kernel, "_tune", no_tuning)
    launch_and_check(module, new_kernel, 5000)
    assert new_kernel.best_configs == kernel.best_configs


@skip_if_no_torch
@skip_if_no_triton
def test_autotune_triton_configs():
    """A list of triton.Config objects can be used instead of tune_params, only these are benchmarked."""
    import triton

    module = kernel_module("triton")
    configs = [triton.Config({"block_size_x": 128}, num_warps=2), triton.Config({"block_size_x": 256}, num_warps=8)]
    kernel = autotune(configs=configs, key=["n"])(module.vector_add.kernel)
    launch_and_check(module, kernel, 5000)
    results = next(iter(kernel.tuning_results.values()))
    assert sorted((r["block_size_x"], r["num_warps"]) for r in results) == [(128, 2), (256, 8)]


@skip_if_no_torch
@skip_if_no_triton
def test_autotune_reference():
    """The outputs are verified with the reference function, wrong outputs stop tuning with an error."""
    module = kernel_module("triton")
    kernel = fresh_kernel(module, reference=lambda c, a, b, n: {"c_ptr": a + b}, atol=1e-5)
    launch_and_check(module, kernel, 5000)

    wrong = fresh_kernel(module, reference=lambda c, a, b, n: [a - b], atol=1e-5)
    with pytest.raises(RuntimeError, match="verification failed"):
        launch_and_check(module, wrong, 5000)


@skip_if_no_torch
@skip_if_no_numba_cuda
def test_autotune_reference_numpy():
    """Expected outputs computed with NumPy from the arrays of another DSL are compared with the outputs."""
    module = kernel_module("numba")
    kernel = fresh_kernel(module, reference=lambda c, a, b, n: {"c": a.copy_to_host() + b.copy_to_host()})
    launch_and_check(module, kernel, 5000)


@skip_if_no_torch
@skip_if_no_triton
def test_autotune_keyword_arguments():
    """Kernel arguments can be passed by name, also while tuning."""
    import torch
    import triton

    module = kernel_module("triton")
    kernel = fresh_kernel(module)
    c, a, b = vectors(5000)
    kernel[lambda meta: (triton.cdiv(meta["n"], meta["block_size_x"]),)](c, a, b_ptr=b, n=5000)
    torch.cuda.synchronize()
    assert torch.allclose(c, a + b)
