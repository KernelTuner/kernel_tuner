"""Provides types for cache in JSON format."""

from __future__ import annotations

from typing import Any, TypedDict

_CacheLineOptionalJSON = TypedDict("_CacheLineOptionalJSON", {})


class CacheLineOptionalJSON(TypedDict, _CacheLineOptionalJSON, total=False):
    """TypedDict for optional data in a cache line."""

    times: list[float]


class CacheLineJSON(TypedDict, CacheLineOptionalJSON):
    """TypedDict for data required in a cache line."""

    time: Any
    compile_time: float
    verification_time: float
    benchmark_time: float
    strategy_time: float
    framework_time: float
    timestamp: str


class CacheFileJSON(TypedDict):
    """TypedDict for the contents of a cache file."""

    schema_version: str
    device_name: str
    kernel_name: str
    problem_size: str
    tune_params_keys: list[str]
    tune_params: dict[str, list]
    objective: str
    cache: dict[str, CacheLineJSON]

class T4ResultMeasurementJSON(TypedDict):
    """TypedDict for the measurements of a T4 result line."""

    name: str
    value: float
    unit: str

class T4ResultTimesJSON(TypedDict):
    """TypedDict for the times of a T4 result line."""

    compilation_time: float
    framework: float
    search_algorithm: float
    validation: float
    runtimes: list[float]

class T4ResultLineJSON(TypedDict):
    """TypedDict for the contents of a T4 result line."""

    timestamp: str
    configuration: dict[str, Any]
    times: T4ResultTimesJSON
    invalidity: str
    correctness: int
    measurements: list[T4ResultMeasurementJSON]
    objectives: list[str]

class T4FileJSON(TypedDict):
    """TypedDict for the contents of a T4 file."""

    results: list[T4ResultLineJSON]
    schema_version: str
