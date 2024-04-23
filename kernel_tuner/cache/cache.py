"""Provides utilities for reading and writing cache files."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any, Union, Optional, Dict, Iterable
from collections.abc import Mapping
from functools import cached_property
from datetime import datetime

import jsonschema
from semver import Version

import kernel_tuner.util as util
from .json import CacheFileJSON, CacheLineJSON
from .file import read_cache, write_cache, append_cache_line
from .convert import convert_cache_file
from .versions import LATEST_VERSION
from .paths import get_schema_path


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
        tune_params_keys: list[str],
        tune_params: dict[str, list],
        objective: str,
    ) -> "Cache":
        """Creates a new cache file.

        For parameters of type Sequence, a list or tuple should be given as argument.
        """
        if not isinstance(device_name, str):
            raise ValueError("Argument device_name should be a string")
        if not isinstance(kernel_name, str):
            raise ValueError("Argument kernel_name should be a string")
        if not isinstance(tune_params_keys, list) and not all(isinstance(key, str) for key in tune_params_keys):
            raise ValueError("Argument tune_params_keys should be a list of strings")
        if not isinstance(tune_params, Mapping) or not all(
            isinstance(key, str) and isinstance(value, list) for key, value in tune_params.items()
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
            "schema_version": str(LATEST_VERSION),
            "device_name": device_name,
            "kernel_name": kernel_name,
            "problem_size": problem_size,
            "tune_params_keys": tune_params_keys,
            "tune_params": tune_params,
            "objective": objective,
            "cache": {},
        }
        cls.validate_json(cache_json)  # NOTE: Validate the cache just to be sure
        write_cache(cache_json, filename)
        return cls(filename, cache_json)  # type: ignore

    @classmethod
    def read(
        cls,
        filename: PathLike,
        # , *, force_update=False
    ):
        """Reads an existing cache file.

        When ``force_update`` is set, the cache file is updated to the latest version when this is not the case.
        Otherwise the file will be loaded in readonly mode.
        """
        cache_json = read_cache(filename)
        cls.validate_json(cache_json)

        # version = Version.parse(cache_json["schema_version"])
        # if force_update and version < LATEST_VERSION:
        #     convert_cache_file(filename)
        #     cache_json = read_cache(filename)
        #     cls.validate_json(cache_json)  # NOTE: Validate the case a second time, just to be sure

        return cls(filename, cache_json)

    @classmethod
    def validate_json(cls, cache_json: Any):
        """Validates cache json."""
        schema_path = get_schema_path(cache_json["schema_version"])
        with open(schema_path, "r") as file:
            schema = json.load(file)
        jsonschema.validate(instance=cache_json, schema=schema)

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
        return Version.parse(self._cache_json["schema_version"])

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
    def tune_params_keys(self) -> list[str]:
        """List of names (keys) of the tunable parameters."""
        return self._cache_json["tune_params_keys"].copy()

    @cached_property
    def tune_params(self) -> dict[str, list[Any]]:
        """Dictionary containing per tunable parameter a tuple of its possible values."""
        return {key: value.copy() for key, value in self._cache_json["tune_params"].items()}

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
            **tune_params,
        ):
            """Appends a cache line to the cache lines."""
            if not (isinstance(time, float) or isinstance(time, util.ErrorConfig)):
                raise ValueError("Argument time should be a float or an ErrorConfig")
            if not isinstance(compile_time, float):
                raise ValueError("Argument compile_time should be a float")
            if not isinstance(verification_time, int):
                raise ValueError("Argument verification_time should be an int")
            if not isinstance(benchmark_time, float):
                raise ValueError("Argument benchmark_time should be a float")
            if not isinstance(strategy_time, int):
                raise ValueError("Argument strategy_time should be an int")
            if not isinstance(framework_time, float):
                raise ValueError("Argument framework_time should be a float")
            if not isinstance(timestamp, datetime):
                raise ValueError("Argument timestamp should be a Python datetime")
            if times is not None and not (isinstance(times, list) and all(isinstance(time, float) for time in times)):
                raise ValueError("Argument times should be a list of floats or None")
            if GFLOP_per_s is not None and not isinstance(GFLOP_per_s, float):
                raise ValueError("Argument GFLOP_per_s should be a float or None")

            line_id = self.__get_line_id_from_tune_params_dict(tune_params)
            # TODO: Decide whether to keep the data pure JSON or still allow Python objects.
            # If the latter is the case, then we should program Cache.Line accordingly.
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
            line.update(tune_params)
            self._lines[line_id] = line  # type: ignore
            append_cache_line(line_id, line, self._filename)

        def __get_line_id_from_tune_params_dict(self, tune_params: dict) -> str:
            param_list = []
            for key in self._cache.tune_params_keys:
                if key in tune_params:
                    value = tune_params[key]
                    if value not in self._cache.tune_params[key]:
                        raise ValueError(f"Invalid value {value} for tunable parameter {key}")
                    param_list.append(value)
                else:
                    raise ValueError(f"Expected tune param key {key} to be present in parameters")
            return self.__get_line_id(param_list)

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
            if line_id is not None:
                line_ids = (line_id,)
                multiple = False
            else:
                line_ids = self.__get_matching_line_ids(params)
                multiple = not all(key in params for key in self._cache.tune_params_keys)

            if multiple:
                lines_json_iter = (self._lines[k] for k in line_ids)
                return list(Cache.Line(self._cache, line) for line in lines_json_iter)
            line_id = next(iter(line_ids), None)
            if line_id is None:
                return default
            line_json = self._lines.get(line_id)
            if line_json is None:
                return default
            return Cache.Line(self._cache, line_json)

        def __get_line_id(self, param_list: list[Any]):
            return ",".join(map(str, param_list))

        def __get_matching_line_ids(self, params: dict[str, Any]):
            param_lists: list[list[Any]] = [[]]
            for key in self._cache.tune_params_keys:
                # If a tunable key is found, only match the value of the parameter
                if key in params:
                    value = params[key]
                    for par_list in param_lists:
                        par_list.append(value)
                # If the tunable key is not present, generate all possible matchin keys
                else:
                    prev_lists = param_lists
                    param_lists = []
                    for value in self._cache.tune_params[key]:
                        param_lists.extend(it + [value] for it in prev_lists)
            return [line_id for line_id in map(self.__get_line_id, param_lists) if line_id in self._lines]

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
