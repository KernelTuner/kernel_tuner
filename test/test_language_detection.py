"""Tests for detecting the language of kernels and the DSL of Python kernels.

The DSL of Python kernels is detected from the source code only, so these tests do not require
any of the DSLs to be installed.
"""

import textwrap

import pytest

from kernel_tuner.kernel_sources.kernel_source import KernelSource
from kernel_tuner.language import Language
from kernel_tuner.utils.call_functions import DEFAULT_CALL_FUNCTIONS, call_triton, get_default_call_function
from kernel_tuner.utils.language_detection import detect_language, detect_python_dsl, is_python_file


def write_source(tmp_path, source, name="kernel.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return path


def test_detect_language1():
    kernel_string = "__global__ void vector_add( ... );"
    lang = detect_language(kernel_string)
    assert lang == "CUDA"


def test_detect_language2():
    kernel_string = "__kernel void vector_add( ... );"
    lang = detect_language(kernel_string)
    assert lang == "OpenCL"


def test_detect_language3():
    kernel_string = "blabla"
    lang = detect_language(kernel_string)
    assert lang == "C"


def test_is_python_file(tmp_path):
    assert is_python_file(write_source(tmp_path, "x = 1"))
    assert is_python_file(str(write_source(tmp_path, "x = 1")))
    assert not is_python_file(write_source(tmp_path, "__global__ void k() {}", name="kernel.cu"))
    assert not is_python_file(str(tmp_path / "does_not_exist.py"))
    assert not is_python_file("__global__ void k() {}")
    assert not is_python_file("x" * 10000)  # too long to be a path
    assert not is_python_file(lambda params: "")


dsl_sources = [
    ("import triton\n@triton.jit\ndef k(): pass", "triton"),
    ("import triton\n@triton.autotune(configs=[])\n@triton.jit\ndef k(): pass", "triton"),
    ("from numba import cuda\n@cuda.jit\ndef k(): pass", "numba"),
    ("import numba.cuda\n@numba.cuda.jit\ndef k(): pass", "numba"),
    ("from numba import jit\n@jit\ndef k(): pass", None),  # Numba for CPUs is not supported
    ("from cupyx import jit\n@jit.rawkernel()\ndef k(): pass", "cupyx"),
    ("import warp as wp\n@wp.kernel()\ndef k(): pass", "warp"),
    ("import taichi as ti\n@ti.kernel\ndef k(): pass", "taichi"),
    ("import cutlass.cute as cute\n@cute.jit\ndef k(): pass", "cute"),
    ("import cutlass\n@cutlass.cute.kernel\ndef k(): pass", "cute"),
    ("import cutlass.cute as cute\nclass k:\n    @cute.jit\n    def __call__(self): pass", "cute"),
    ("import tilus\nclass k(tilus.Script):\n    def __call__(self): pass", "tilus"),
    ("from tilus import Script\nclass k(Script):\n    def __call__(self): pass", "tilus"),
    ("import tilelang\n@tilelang.jit\ndef k(): pass", "tilelang"),
    ("import cuda.tile as ct\n@ct.kernel\ndef k(): pass", "cutile"),
    ("from cuda import tile\n@tile.kernel\ndef k(): pass", "cutile"),
    ("import cuda.tile\n@cuda.tile.kernel\ndef k(): pass", "cutile"),
    ("def k(): pass", None),
    ("import functools\n@functools.cache\ndef k(): pass", None),
    ("import triton\ndef helper(): pass\n@triton.jit\ndef other(): pass\ndef k(): pass", None),
]


@pytest.mark.parametrize("source, dsl", dsl_sources, ids=[s for s, _ in dsl_sources])
def test_detect_python_dsl(tmp_path, source, dsl):
    assert detect_python_dsl("k", write_source(tmp_path, source)) == dsl


def test_every_dsl_has_a_default_call_function():
    detected_dsls = {dsl for _, dsl in dsl_sources if dsl is not None}
    assert detected_dsls == set(DEFAULT_CALL_FUNCTIONS)
    with pytest.raises(ValueError):
        get_default_call_function("unknown_dsl")


def test_kernel_source_detects_generic_python_and_call_function(tmp_path):
    path = write_source(tmp_path, "import triton\n@triton.jit\ndef k(): pass")
    ks = KernelSource("k", str(path), None)
    assert ks.lang == Language.GENERIC_PYTHON
    # call_triton accepts all optional arguments, so normalizing it returns the function itself
    assert ks.call_function is call_triton


def test_kernel_source_user_call_function_takes_precedence(tmp_path):
    path = write_source(tmp_path, "import triton\n@triton.jit\ndef k(): pass")

    def my_call_function(kernel_function, args, kwargs, grid, threads, params):
        pass

    ks = KernelSource("k", path, None, call_function=my_call_function)
    assert ks.lang == Language.GENERIC_PYTHON
    assert ks.call_function is my_call_function

    # a call function also makes undecorated kernels usable
    path = write_source(tmp_path, "def k(): pass", name="plain.py")
    ks = KernelSource("k", path, "generic_python", call_function=my_call_function)
    assert ks.call_function is my_call_function


def test_kernel_source_unknown_dsl_requires_call_function(tmp_path):
    path = write_source(tmp_path, "def k(): pass")
    with pytest.raises(ValueError, match="Could not detect the Python DSL"):
        KernelSource("k", path, None)


def test_kernel_source_non_python_file_is_not_generic_python(tmp_path):
    path = write_source(tmp_path, "__global__ void k(float *a) {}", name="kernel.cu")
    ks = KernelSource("k", str(path), None)
    assert ks.lang == Language.CUDA
