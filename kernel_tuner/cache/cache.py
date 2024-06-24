"""Provides utilities for reading and writing cache files.

In order to modify and read cache files, the Cache class should be used, see its docstring.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from functools import cached_property
from functools import lru_cache as cache
from os import PathLike
from pathlib import Path
from typing import cast, Any, Union, Optional, Dict, Iterable, Iterator

import jsonschema
from semver import Version

import kernel_tuner.util as util
from .convert import convert_cache
from .json import CacheFileJSON, CacheLineJSON
from .json_encoder import CacheEncoder
from .file import read_cache, write_cache, append_cache_line
from .versions import LATEST_VERSION, VERSIONS
from .paths import get_schema_path


class Cache:
    """Writes and reads cache files.

    Cache files can be opened using ``Cache.open()`` and created using ``Cache.create()``. In both cases, a ``Cache``
    instance is returned. This object simultaneously keeps track of the file, as well as its json contents in an
    efficient manner. To read cache files of any old version, use `Cache.read()`, which returns a ``Cache`` instance
    which which won't be able to be mutated. Note that the cache file should not be changed during the Cache instance's
    lifetime, as the instance's state would in that case not correspond to the file's JSON content. To automatically
    detect changes in the Cache instance, one could use os.path.getmtime() in order to detect whenever the cache file
    changes.

    The Cache class furthermore contains an easily usable interface for reading cache file properties, e.g.
    `Cache.kernel_name` or `Cache.version`, and an easily usable interface for matching cache lines from their
    parameters `Cache.lines.get()`, and an easily usable interface for appending cache lines `Cache.lines.append()`
    to cache files.

    Properties:
        filepath: filepath to the cache.
        version: schema version of the cache.
        lines: cache lines of the json file.

        device_name
        kernel_name
        problem_size
        tune_params_keys
        tune_params
        objective

    See Also:
        Docstring from `Cache.Lines` explaining how to read and append cache lines
        Docstring from `Cache.Line` explaining how to read properties from cache lines
    """

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

        Returns a Cache instance of which the lines are modifiable.
        """
        if not isinstance(device_name, str):
            raise ValueError("Argument device_name should be a string")
        if not isinstance(kernel_name, str):
            raise ValueError("Argument kernel_name should be a string")
        if not isinstance(tune_params_keys, list) and not all(isinstance(key, str) for key in tune_params_keys):
            raise ValueError("Argument tune_params_keys should be a list of strings")
        if not isinstance(tune_params, dict) or not all(
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

        # main dictionary for new cache, note it is very important that 'cache' is the last key in the dict
        cache_json: CacheFileJSON = {
            "schema_version": str(LATEST_VERSION),
            "device_name": device_name,
            "kernel_name": kernel_name,
            "problem_size": problem_size,
            "tune_params_keys": tune_params_keys,
            "tune_params": tune_params,
            "objective": objective,
            "cache": {},
        }

        cls.validate_json(cache_json)
        write_cache(cast(dict, cache_json), filename)
        return cls(filename, cache_json, read_only=False)

    @classmethod
    def open(cls, filename: PathLike):
        """Opens an existing cache file. Returns a Cache instance with modifiable lines.

        This cache file should have the latest version
        """
        cache_json = read_cache(filename)
        assert Version.parse(cache_json["schema_version"]) == LATEST_VERSION, "Cache file is not of the latest version."
        cls.validate_json(cache_json)
        return cls(filename, cache_json, read_only=False)

    @classmethod
    def read(cls, filename: PathLike, read_only=False):
        """Loads an existing cache. Returns a Cache instance.

        If the cache file does not have the latest version, then it will be read after virtually converting it to the
        latest version. The file in this case is kept the same.
        """
        cache_json = read_cache(filename)

        # convert cache to latest schema if needed, then validate
        if "schema_version" not in cache_json or cache_json["schema_version"] != LATEST_VERSION:
            cache_json = convert_cache(cache_json)
            # if not read-only mode, update the file
            if not read_only:
                write_cache(cast(dict, cache_json), filename)

        cls.validate_json(cache_json)

        return cls(filename, cache_json, read_only=read_only)

    @classmethod
    def validate(cls, filename: PathLike):
        """Validates a cache file and raises an error if invalid."""
        cache_json = read_cache(filename)
        cls.validate_json(cache_json)

    @classmethod
    def validate_json(cls, cache_json: Any):
        """Validates cache json."""
        if "schema_version" not in cache_json:
            raise jsonschema.ValidationError("Key 'schema_version' is not present in cache data")
        schema_version = cache_json["schema_version"]
        cls.__validate_json_schema_version(schema_version)
        schema = cls.__get_schema_for_version(schema_version)
        format_checker = _get_format_checker()
        jsonschema.validate(instance=cache_json, schema=schema, format_checker=format_checker)

    @classmethod
    def __validate_json_schema_version(cls, version: str):
        try:
            if Version.parse(version) in VERSIONS:
                return
        except (ValueError, TypeError):
            pass
        raise jsonschema.ValidationError(f"Invalid version {repr(version)} found.")

    @classmethod
    def __get_schema_for_version(cls, version: str):
        schema_path = get_schema_path(version)
        with open(schema_path, "r") as file:
            return json.load(file)

    def __init__(self, filename: PathLike, cache_json: CacheFileJSON, *, read_only: bool):
        """Inits a cache file instance, given that the file referred to by ``filename`` contains data ``cache_json``.

        Argument ``cache_json`` is a cache dictionary expected to have the latest cache version.
        """
        self._filename = Path(filename)
        self._cache_json = cache_json
        self._read_only = read_only

    @cached_property
    def filepath(self) -> Path:
        """Returns the path to the cache file."""
        return self._filename

    @cached_property
    def version(self) -> Version:
        """Version of the cache file."""
        return Version.parse(self._cache_json["schema_version"])

    @cached_property
    def lines(self) -> Union[Lines, ReadableLines]:
        """List of cache lines."""
        if self._read_only:
            return self.ReadableLines(self, self._filename, self._cache_json)
        else:
            return self.Lines(self, self._filename, self._cache_json)

    class Lines(Mapping):
        """Cache lines in a cache file.

        Behaves exactly like an only readable dict, except with an `append` method for appending lines.

        Usage Example:
            cache: Cache = ...

            print("Line with id 0,0,0 is ", cache.lines["0,0,0"])
            print(f"There are {len(cache.lines)} lines")

            cache.lines.append(..., tune_param_a=1, tune_param_b=2, tune_param_c=3)

            print(f"There are {len(cache.lines)} lines")
            for line_id, line in cache.lines.items():
                print(f"Line {line_id} has value {line}.")

            # If there are more tunable parameter keys than just "a",
            # then cache.lines.get(a=...) returns a list.
            for line in cache.lines.get(a=1):
                print(f"Line {line} is one of the lines with `a=1`")

        See Also:
            collections.abc.Mapping: https://docs.python.org/3/library/collections.abc.html
        """

        def __init__(self, cache: Cache, filename: PathLike, cache_json: CacheFileJSON):
            """Inits a new CacheLines instance."""
            self._cache = cache
            self._filename = filename
            self._lines = cache_json["cache"]

        def __getitem__(self, line_id: str) -> Cache.Line:
            """Returns a cache line given the parameters (in order)."""
            return Cache.Line(self._cache, self._lines[line_id])

        def __iter__(self) -> Iterator[str]:
            """Returns an iterator over the keys of the cache lines."""
            return iter(self._lines)

        def __len__(self) -> int:
            """Returns the number of cache lines."""
            return len(self._lines)

        def __contains__(self, line_id) -> bool:
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
        ) -> None:
            """Appends a cache line to the cache lines."""
            if not (isinstance(time, float) or isinstance(time, util.ErrorConfig)):
                raise ValueError("Argument time should be a float or an ErrorConfig")
            if not isinstance(compile_time, float):
                raise ValueError("Argument compile_time should be a float")
            if not isinstance(verification_time, (int, float)):
                raise ValueError("Argument verification_time should be an int or float")
            # It is possible that verification_time is a bool which is also of instance int. Check and cast to be sure.
            elif isinstance(verification_time, bool):
                verification_time = int(verification_time)
            if not isinstance(benchmark_time, float):
                raise ValueError("Argument benchmark_time should be a float")
            if not isinstance(strategy_time, (int, float)):
                raise ValueError("Argument strategy_time should be an int or float")
            # It is possible that strategy_time is a bool which is also of instance int. Check and cast to be sure.
            elif isinstance(strategy_time, bool):
                strategy_time = int(strategy_time)
            if not isinstance(framework_time, float):
                raise ValueError(f"Argument framework_time should be a float, received: {framework_time} ({type(framework_time)})")
            if not isinstance(timestamp, datetime):
                raise ValueError("Argument timestamp should be a Python datetime")
            if times is not None and not (isinstance(times, list) and all(isinstance(time, float) for time in times)):
                raise ValueError("Argument times should be a list of floats or None")
            if GFLOP_per_s is not None and not isinstance(GFLOP_per_s, float):
                raise ValueError("Argument GFLOP_per_s should be a float or None")

            line_id = self.__get_line_id_from_tune_params_dict(tune_params)
            if line_id in self._lines:
                raise KeyError("Line with given tunable parameters already exists")

            line = self.__get_line_json_object(
                time,
                compile_time,
                verification_time,
                benchmark_time,
                strategy_time,
                framework_time,
                timestamp,
                times,
                GFLOP_per_s,
                tune_params,
            )
            self._lines[line_id] = line
            append_cache_line(line_id, line, self._filename)

        def get(self, line_id: Optional[str] = None, default=None, **params) -> Union[Cache.Line, list[Cache.Line]]:
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
                return list(Cache.Line(self._cache, line) for line in self._lines.values())
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

        def __get_line_id(self, param_list: list[Any]) -> str:
            return json.dumps(param_list, separators=(",", ":"))[1:-1]

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

        def __get_line_json_object(
            self,
            time: Union[float, util.ErrorConfig],
            compile_time: float,
            verification_time: int,
            benchmark_time: float,
            strategy_time: int,
            framework_time: float,
            timestamp: datetime,
            times: Optional[list[float]],
            GFLOP_per_s: Optional[float],
            tune_params,
        ):
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
            line = {**line, **tune_params}

            return line

    class ReadableLines(Lines):
        """Cache lines in a read_only cache file."""

        def append(*args, **kwargs):
            """ Method to append lines to cache file, should not happen with read-only cache """
            raise ValueError(f"Attempting to write to read-only cache")

    class Line(Mapping):
        """Cache line in a cache file.

        Every instance of this class behaves in principle as if it were a readable dict. Items can be accessed via the
        instance's attributes, or via __getitem__ using the traditional brackets (`line[...]`). In addition, the aliased
        properties automatically convert json data to python objects or can reference some dict item that does not have
        a key that can be used as attribute. Items accessed using __getitem__ will always return a json serializable
        object.

        Alias Properties:
            time: error `util.ErrorConfig` or a number `float`
            times: a list of floats (`float`) or `None`
            timestamp: a `datetime` object of the timestamp
            GFLOP_per_s: alias of "GFLOP/s"

        Usage Example:
            from datetime import datetime

            cache: Cache = ...
            line = cache.lines[...]

            # Useful alias for GFLOP/s
            assert line.GFLOP_per_s == line["GFLOP/s"]

            # The timestamp attribute is automatically converted to a `datetime` object
            assert isinstance(line.timestamp, datetime)
            assert isinstance(line["timestamp"], str)

        See Also:
            collections.abc.Mapping: https://docs.python.org/3/library/collections.abc.html
        """

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
            item = self._line[key]
            if not (item is None or isinstance(item, (bool, int, float, str, list, dict))):
                # FIX: This will convert the root object of any not json serializable object to a json serializable
                # object, but it will not convert any items from the object returned to be json serializable.
                encoder = _get_cache_line_json_encoder()
                item = encoder.default(item)
            return item

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

        def todict(self):
            """Returns the cache line as a dictionary."""
            return self._line.copy()

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


@cache
def _get_cache_line_json_encoder():
    return CacheEncoder(indent="")


@cache
def _get_format_checker():
    """Returns a JSON format checker instance."""
    format_checker = jsonschema.FormatChecker()

    @format_checker.checks("date-time")
    def _check_iso_datetime(instance):
        try:
            datetime.fromisoformat(instance)
            return True
        except (ValueError, TypeError):
            return False

    return format_checker
