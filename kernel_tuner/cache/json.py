"""Provides types for cache in JSON format."""

from __future__ import annotations
from typing import TypedDict, Any


_CacheLineOptionalJSON = TypedDict("_CacheLineOptionalJSON", {"GFLOP/s": float})


class CacheLineOptionalJSON(TypedDict, _CacheLineOptionalJSON, total=False):
    """TypedDict for optional data in a cache line."""

    times: list[float]
    # "GFLOP/s": float


class CacheLineJSON(TypedDict, CacheLineOptionalJSON):
    """TypedDict for data required in a cache line."""

    time: Any
    compile_time: float
    verification_time: int
    benchmark_time: float
    strategy_time: int
    framework_time: float
    timestamp: str


class CacheFileJSON(TypedDict):
    """TypedDict for the contents of a cache file."""

    version: str
    device_name: str
    kernel_name: str
    problem_size: str
    tune_params_keys: list[str]
    tune_params: dict[str, list]  # is every param a number?
    objective: str
    cache: dict[str, CacheLineJSON]
