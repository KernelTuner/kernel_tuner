"""Provides utilities for reading and writing cache files."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Union, Optional, Dict, Iterable
from collections.abc import Mapping, Sequence
from functools import cached_property
from datetime import datetime

from semver import Version
import json

import kernel_tuner
import kernel_tuner.util as util
from .json import CacheFileJSON, CacheLineJSON
from .file import write_cache, append_cache_line


frozendict = MappingProxyType


PROJECT_DIR = Path(kernel_tuner.__file__).parent
SCHEMA_DIR = PROJECT_DIR / "schema"
CACHE_SCHEMAS_DIR = SCHEMA_DIR / "cache"

SORTED_VERSIONS: list[Version] = sorted(Version.parse(p.name) for p in CACHE_SCHEMAS_DIR.iterdir())
VERSIONS: list[Version] = SORTED_VERSIONS
LATEST_VERSION: Version = VERSIONS[-1]


def get_schema_path(version: Version):
    """Returns the path to the schema of the cache of a specific version."""
    return CACHE_SCHEMAS_DIR / str(version)


class Cache:
    """Interface for writing and reading cache files."""

    RESERVED_PARAM_KEYS: set = {
        "time",
        "compile_time",
        "verification_time",
        "benchmark_time",
        "strategy_time",
        "framework_time",
        "timestamp",
        "times",
        "GFLOP/s",
    }

    time: Any
    compile_time: float
    verification_time: int
    benchmark_time: float
    strategy_time: int
    framework_time: float
    timestamp: str

    @classmethod
    def create(
        cls,
        filename: PathLike,
        *,
        device_name: str,
        kernel_name: str,
        problem_size: Any,
        tune_params_keys: Sequence[str],
        tune_params: dict[str, Sequence],
        objective: str,
    ) -> "Cache":
        """Creates a new cache file.

        For parameters of type Sequence, a list or tuple should be given as argument.
        """
        if not isinstance(device_name, str):
            raise ValueError("Argument device_name should be a string")
        if not isinstance(kernel_name, str):
            raise ValueError("Argument kernel_name should be a string")
        if not isinstance(tune_params_keys, Sequence) and not all(isinstance(key, str) for key in tune_params_keys):
            raise ValueError("Argument tune_params_keys should be a list of strings")
        if not isinstance(tune_params, Mapping) or not all(
            isinstance(key, str) and isinstance(value, Sequence) for key, value in tune_params.items()
        ):
            raise ValueError(
                "Argument tune_params should be a dict with:\n"
                "- keys being parameter keys (and all parameter keys should be used)\n"
                "- values being the parameter's list of possible values"
            )
        if not isinstance(objective, str):
            raise ValueError("Expected objective to be a string")
        if set(tune_params_keys) != set(tune_params.keys()):
            raise ValueError("Expected tune_params to have exactly the same keys as in the list tune_params_keys")
        if len(cls.RESERVED_PARAM_KEYS & set(tune_params_keys)) > 0:
            raise ValueError("Found a reserved key in tune_params_keys")

        cache_json = {
            "version": str(LATEST_VERSION),
            "device_name": device_name,
            "kernel_name": kernel_name,
            "problem_size": problem_size,
            "tune_params_keys": tune_params_keys,
            "tune_params": tune_params,
            "objective": objective,
            "cache": {},
        }

        write_cache(cache_json, filename)
        return cls(filename, cache_json)  # type: ignore

    @classmethod
    def read(cls, filename: PathLike):
        """Reads an existing cache file."""
        with open(filename, "r") as file:
            cache_json = json.load(file)
            # TODO: Validate and convert cache file
        return cls(filename, cache_json)

    def __init__(self, filename: PathLike, cache_json: CacheFileJSON):
        """Inits a cache file instance, given that the file referred to by ``filename`` contains data ``cache_json``."""
        self._filename = Path(filename)
        self._cache_json = cache_json

    @cached_property
    def filepath(self) -> Path:
        """Returns the path to the cache file."""
        return self._filename

    @cached_property
    def version(self) -> Version:
        """Version of the cache file."""
        return Version.parse(self._cache_json["version"])

    @cached_property
    def device_name(self) -> str:
        """Name of the device."""
        return self._cache_json["device_name"]

    @cached_property
    def kernel_name(self) -> str:
        """Name of the kernel."""
        return self._cache_json["kernel_name"]

    @cached_property
    def problem_size(self) -> Any:
        """Problem size of the kernel being tuned."""
        return self._cache_json["problem_size"]

    @cached_property
    def tune_params_keys(self) -> tuple[str, ...]:
        """List of names (keys) of the tunable parameters."""
        return tuple(self._cache_json["tune_params_keys"])

    @cached_property
    def tune_params(self) -> frozendict[str, tuple[Any, ...]]:
        """Dictionary containing per tunable parameter a tuple of its possible values."""
        return frozendict({key: tuple(value) for key, value in self._cache_json["tune_params"].items()})

    @cached_property
    def objective(self) -> str:
        """Objective of tuning the kernel."""
        return self._cache_json["objective"]

    @cached_property
    def lines(self) -> Lines:
        """List of cache lines."""
        return self.Lines(self, self._filename, self._cache_json)

    class Lines(Mapping):
        """Cache lines in a cache file."""

        def __init__(self, cache: Cache, filename: PathLike, cache_json: CacheFileJSON):
            """Inits a new CacheLines instance."""
            self._cache = cache
            self._filename = filename
            self._lines = cache_json["cache"]

        def __getitem__(self, line_id: str):
            """Returns a cache line given the parameters (in order)."""
            return self._lines[line_id]

        def __iter__(self):
            """Returns an iterator over the keys of the cache lines."""
            return iter(self._lines)

        def __len__(self):
            """Returns the number of cache lines."""
            return len(self._lines)

        def __contains__(self, line_id):
            """Returns whether there exists a cache line with id ``line_id``."""
            return line_id in self._lines

        def append(
            self,
            *,
            time: Union[float, util.ErrorConfig],
            compile_time: float,
            verification_time: int,
            benchmark_time: float,
            strategy_time: int,
            framework_time: float,
            timestamp: datetime,
            times: Optional[list[float]] = None,
            GFLOP_per_s: Optional[float] = None,
            **params,
        ):
            """Appends a cache line to the cache lines."""
            param_list = []
            for key in self._cache.tune_params_keys:
                if key in params:
                    param_list.append(params[key])
                else:
                    raise ValueError(f"Expected tune param key {key} to be present in parameters")
            line_id = self.__get_line_id(param_list)
            line: dict = {
                "time": time,
                "compile_time": compile_time,
                "verification_time": verification_time,
                "benchmark_time": benchmark_time,
                "strategy_time": strategy_time,
                "framework_time": framework_time,
                "timestamp": str(timestamp),
            }
            if times is not None:
                line["times"] = times
            if GFLOP_per_s is not None:
                line["GFLOP/s"] = GFLOP_per_s
            line.update(params)
            self._lines[line_id] = line  # type: ignore
            append_cache_line(line_id, line, self._filename)

        def get(self, line_id: Optional[str] = None, default=None, **params):
            """Returns a cache line corresponding with ``line_id``.

            If the line_id is given and not None, the line corresponding to ``line_id`` is returned. Otherwise the
            keyword parameters are checked. If ``params`` contains all of the keys present in ``tune_params_keys``,
            then these parameters are used to filter the element we want.

            If all parameters from ``tune_params_keys`` are specified, a single line is returned, and ``default`` if it
            does not exist. Otherwise a list containing all lines that match the given parameters are returned, and in
            this case, default is disregarded.

            It should be noted that partially matching the lines in the above manner is slow, as this implementation
            generates the fill line ids for all of its matches. A future implementation might want to improve upon this
            process with a custom datastructure.

            If ``line_id`` is none and no parameters are defined, a list of all the lines is returned.
            """
            if line_id is None and len(params) == 0:
                return list(self._lines.values())
            if not all(key in self._cache.tune_params_keys for key in params):
                raise ValueError("The keys in the parameters should be in `tune_params_keys`")

            line_ids: Iterable[str]
            multiple = False
            if line_id is not None:
                line_ids = (line_id,)
            else:
                # Match all line ids that correspond to params
                param_lists: list[list[Any]] = [[]]
                for key in self._cache.tune_params_keys:
                    # If a tunable key is found, only match the value of the parameter
                    if key in params:
                        value = params[key]
                        for par_list in param_lists:
                            par_list.append(value)
                    # If the tunable key is not present, generate all possible matchin keys
                    else:
                        multiple = True
                        prev_lists = param_lists
                        param_lists = []
                        for value in self._cache.tune_params[key]:
                            param_lists.extend(it + [value] for it in prev_lists)
                line_ids = list(map(self.__get_line_id, param_lists))

            if multiple:
                lines_json_iter = (self._lines[k] for k in line_ids if k in self._lines)
                return list(Cache.Line(self._cache, line) for line in lines_json_iter)

            line_id = next(iter(line_ids))
            line_json = self._lines.get(line_id)
            if line_json is None:
                return default
            return Cache.Line(self._cache, line_json)

        def __get_line_id(self, param_list: list[Any]):
            return ",".join(map(str, param_list))

    class Line(Mapping):
        """Cache line in a cache file."""

        compile_time: float
        verification_time: int
        benchmark_time: float
        strategy_time: int
        framework_time: float

        @property
        def time(self) -> Union[float, util.ErrorConfig]:
            """The time of a cache line."""
            time_or_error = self["time"]
            if isinstance(time_or_error, str):
                return util.ErrorConfig.from_str(time_or_error)
            return time_or_error

        @property
        def times(self) -> Optional[list[float]]:
            """The times attribute."""
            return self.get("times")

        @property
        def timestamp(self) -> datetime:
            """The timestamp as a datetime object."""
            return datetime.fromisoformat(self["timestamp"])

        @property
        def GFLOP_per_s(self) -> Optional[float]:
            """The number of GFLOPs per second."""
            return self.get("GFLOP/s")

        def __init__(self, cache: Cache, line_json: CacheLineJSON):
            """Inits a new CacheLines instance."""
            self._cache = cache
            self._line: Dict = line_json  # type: ignore

        def __getitem__(self, key: str):
            """Returns an item in a line."""
            return self._line[key]

        def __len__(self) -> int:
            """Returns the number of attributes in a line."""
            return len(self._line)

        def __iter__(self):
            """Returns an iterator over the line's keys."""
            return iter(self._line)

        def __contains__(self, key: object) -> bool:
            """Returns whether a line contains a key."""
            return key in self._line

        def __getattr__(self, name: str):
            """Accesses members of the dict as if they were attributes."""
            return self[name]
