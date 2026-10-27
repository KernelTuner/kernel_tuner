"""Tests for compiling kernels in parallel threads with the ParallelCompileRunner (parallel_compile option)."""

import threading

import numpy as np
import pytest

from kernel_tuner import tune_kernel
from kernel_tuner.core import DeviceInterface

from .context import skip_backend, skip_if_no_gcc

CUDA_KERNEL = """
extern "C" __global__ void vector_add(float *c, float *a, float *b, int n) {
    __shared__ float buffer[block_size_x * shared_factor];
    int i = blockIdx.x * block_size_x + threadIdx.x;
    buffer[threadIdx.x * shared_factor] = 1.0f;
    if (i < n) {
        c[i] = a[i] + b[i] * buffer[threadIdx.x * shared_factor];
    }
}
"""

C_KERNEL = """
#include <stdlib.h>
float vector_add(float *c, float *a, float *b, int n) {
    for (int i = 0; i < n; i += block_size_x) {
        for (int j = i; j < i + block_size_x && j < n; j++) {
            c[j] = a[j] + b[j];
        }
    }
    return 0.0f;
}
"""


def vector_add_args(size=10000):
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    return [c, a, b, np.int32(size)], [a + b, None, None, None]


def valid_configs(results):
    return sorted(
        tuple(sorted((k, v) for k, v in r.items() if k in ("block_size_x", "shared_factor")))
        for r in results
        if isinstance(r.get("time"), float)
    )


@pytest.mark.parametrize("lang", ["NVCUDA", "PYCUDA", "CUPY"])
def test_parallel_compile_cuda(lang, monkeypatch):
    skip_backend(lang)
    args, answer = vector_add_args()
    # 64 * 512 floats is 128 KB of shared memory, too much for any GPU, so those configurations are skipped
    tune_params = {"block_size_x": [32, 64, 128, 256, 512], "shared_factor": [1, 64]}

    # record which threads the kernels are built in
    build_threads = []
    original_build_kernel = DeviceInterface.build_kernel

    def recording_build_kernel(self, instance, gpu_args=None):
        build_threads.append(threading.current_thread())
        return original_build_kernel(self, instance, gpu_args)

    monkeypatch.setattr(DeviceInterface, "build_kernel", recording_build_kernel)

    sequential, _ = tune_kernel("vector_add", CUDA_KERNEL, 10000, args, tune_params, lang=lang, answer=answer)
    assert all(t is threading.main_thread() for t in build_threads)
    build_threads.clear()

    parallel, _ = tune_kernel(
        "vector_add", CUDA_KERNEL, 10000, args, tune_params, lang=lang, answer=answer, parallel_compile=4
    )
    assert len(build_threads) == 10
    assert all(t is not threading.main_thread() for t in build_threads)

    # the same configurations are valid, and the ones that use too much shared memory are skipped
    assert len(parallel) == len(sequential) == 10
    assert valid_configs(parallel) == valid_configs(sequential)
    assert 0 < len(valid_configs(parallel)) < 10
    assert all(r["compile_time"] > 0 for r in parallel if isinstance(r.get("time"), float))


@skip_if_no_gcc
def test_parallel_compile_backend_without_build():
    """The C backend does not implement build(), so kernels are compiled when they are loaded."""
    args, answer = vector_add_args()
    tune_params = {"block_size_x": [1, 2, 4, 8, 16, 32]}
    results, _ = tune_kernel(
        "vector_add", C_KERNEL, 10000, args, tune_params, lang="C", answer=answer, parallel_compile=3
    )
    assert len(valid_configs(results)) == 6


@skip_if_no_gcc
def test_parallel_compile_chunks_and_budget(monkeypatch):
    args, answer = vector_add_args()
    tune_params = {"block_size_x": list(range(1, 41))}

    built = []
    original_build_kernel = DeviceInterface.build_kernel

    def counting_build_kernel(self, instance, gpu_args=None):
        built.append(instance.params["block_size_x"])
        return original_build_kernel(self, instance, gpu_args)

    monkeypatch.setattr(DeviceInterface, "build_kernel", counting_build_kernel)

    # with 2 threads, the 40 configurations are processed in chunks of 8
    results, _ = tune_kernel(
        "vector_add", C_KERNEL, 10000, args, tune_params, lang="C", answer=answer, parallel_compile=2
    )
    assert len(valid_configs(results)) == 40
    assert sorted(built) == list(range(1, 41))

    # only configurations within the budget are built
    built.clear()
    results, _ = tune_kernel(
        "vector_add", C_KERNEL, 10000, args, tune_params, lang="C", answer=answer, parallel_compile=2,
        strategy="random_sample", strategy_options={"max_fevals": 11},
    )
    assert len(results) == 11
    assert len(built) == 11
    assert sorted(built) == sorted(r["block_size_x"] for r in results)


@pytest.mark.parametrize("lang", ["NVCUDA", "PYCUDA", "CUPY"])
def test_parallel_compile_error_is_raised(lang, tmp_path, monkeypatch):
    skip_backend(lang)
    # the kernel source is written to the current directory when compilation fails
    monkeypatch.chdir(tmp_path)
    args, _ = vector_add_args()
    kernel = CUDA_KERNEL.replace("buffer[threadIdx.x * shared_factor] = 1.0f;", "undefined_variable = 1.0f;")
    tune_params = {"block_size_x": [32, 64], "shared_factor": [1]}
    with pytest.raises(Exception):
        tune_kernel("vector_add", kernel, 10000, args, tune_params, lang=lang, parallel_compile=2, quiet=True)


def test_parallel_compile_option_combinations():
    args, _ = vector_add_args()
    tune_params = {"block_size_x": [32]}
    with pytest.raises(ValueError, match="parallel_compile"):
        tune_kernel("vector_add", C_KERNEL, 10000, args, tune_params, lang="C", parallel_compile=2, parallel=2)
    with pytest.raises(ValueError, match="parallel_compile"):
        tune_kernel(
            "vector_add", C_KERNEL, 10000, args, tune_params, lang="C", parallel_compile=2, simulation_mode=True
        )
