from __future__ import annotations
from typing import TypedDict, Any


_non_identifier_keys = TypedDict("CacheEntryJSON", {
    "GFLOP/s": float
})


class CacheLineOptionalJSON(TypedDict, _non_identifier_keys, total=False):
    times: list[float]


class CacheLineJSON(TypedDict, CacheLineOptionalJSON):
    time: Any
    compile_time: float
    verification_time: int
    benchmark_time: float
    # "GFLOP/s": float
    strategy_time: int
    framework_time: float
    timestamp: str


class CacheFileJSON(TypedDict):
    device_name: str
    kernel_name: str
    problem_size: str
    tune_params_keys: list[str]
    tune_params: dict[str, list]  # is every param a number?
    objective: str
    cache: dict[str, CacheLineJSON]
