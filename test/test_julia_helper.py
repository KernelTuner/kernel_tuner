"""Tests for the detection of a local CUDA toolkit for the Julia backend, these do not need Julia or a GPU."""

import os
import stat

import pytest

from kernel_tuner.backends import julia_helper


def make_toolkit(base, version="12.4", complete=True):
    """Create a fake CUDA toolkit directory with a ptxas that reports the given version."""
    bin_dir = base / "bin"
    bin_dir.mkdir(parents=True)
    ptxas = bin_dir / "ptxas"
    ptxas.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {version}, V{version}.0'\n")
    ptxas.chmod(ptxas.stat().st_mode | stat.S_IEXEC)
    if complete:
        (base / "lib64").mkdir()
        (base / "lib64" / "libcudart.so.12").write_text("")
        (base / "nvvm" / "libdevice").mkdir(parents=True)
        (base / "nvvm" / "libdevice" / "libdevice.10.bc").write_text("")
    return base


@pytest.fixture
def toolkit_env(monkeypatch, tmp_path):
    """Isolate the toolkit discovery from the system, with a driver that supports CUDA 12.8."""
    for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", julia_helper.LOCAL_CUDA_TOOLKIT_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(julia_helper.shutil, "which", lambda name: None)
    monkeypatch.setattr(julia_helper.os.path, "isdir", lambda path: str(path).startswith(str(tmp_path)))
    monkeypatch.setattr(julia_helper, "cuda_driver_version", lambda: (12, 8))
    return tmp_path


skip_on_windows = pytest.mark.skipif(os.name == "nt", reason="fake ptxas is a shell script")


@skip_on_windows
def test_find_local_cuda_toolkit_from_env(toolkit_env, monkeypatch):
    """A toolkit given by the environment variables is found."""
    toolkit = make_toolkit(toolkit_env / "cuda")
    monkeypatch.setenv("CUDA_HOME", str(toolkit))
    assert julia_helper.find_local_cuda_toolkit() == (str(toolkit), (12, 4))


@skip_on_windows
def test_find_local_cuda_toolkit_incompatible(toolkit_env, monkeypatch):
    """Toolkits that CUDA.jl can not use with the driver are rejected."""
    # newer major version than the driver supports
    monkeypatch.setenv("CUDA_HOME", str(make_toolkit(toolkit_env / "cuda13", version="13.0")))
    assert julia_helper.find_local_cuda_toolkit() is None
    # too old
    monkeypatch.setenv("CUDA_HOME", str(make_toolkit(toolkit_env / "cuda11", version="11.8")))
    assert julia_helper.find_local_cuda_toolkit() is None
    # incomplete, e.g. only the compiler
    monkeypatch.setenv("CUDA_HOME", str(make_toolkit(toolkit_env / "partial", complete=False)))
    assert julia_helper.find_local_cuda_toolkit() is None


@skip_on_windows
def test_find_local_cuda_toolkit_no_driver(toolkit_env, monkeypatch):
    """Without an NVIDIA driver, no toolkit is used."""
    monkeypatch.setenv("CUDA_HOME", str(make_toolkit(toolkit_env / "cuda")))
    monkeypatch.setattr(julia_helper, "cuda_driver_version", lambda: None)
    assert julia_helper.find_local_cuda_toolkit() is None


@skip_on_windows
def test_local_cuda_toolkit_preferences(toolkit_env, monkeypatch):
    """The preferences follow the detection and the environment variable."""
    local = {name: {"local": "true"} for name in julia_helper.CUDA_TOOLKIT_JLLS}
    # no local toolkit: download
    assert julia_helper.local_cuda_toolkit_preferences() is None
    # local toolkit found
    monkeypatch.setenv("CUDA_HOME", str(make_toolkit(toolkit_env / "cuda")))
    with pytest.warns(UserWarning, match="local CUDA 12.4 toolkit"):
        assert julia_helper.local_cuda_toolkit_preferences() == local
    # opt-out
    monkeypatch.setenv(julia_helper.LOCAL_CUDA_TOOLKIT_ENV, "no")
    assert julia_helper.local_cuda_toolkit_preferences() is None
    # forced, even without a detected toolkit
    monkeypatch.delenv("CUDA_HOME")
    monkeypatch.setenv(julia_helper.LOCAL_CUDA_TOOLKIT_ENV, "yes")
    assert julia_helper.local_cuda_toolkit_preferences() == local
