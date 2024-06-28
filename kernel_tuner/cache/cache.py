"""Provides utilities for reading and writing cache files.

In order to modify and read cache files, the Cache class should be used, see its docstring.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime
from functools import cached_property
from functools import lru_cache as cache
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union, cast

import jsonschema
import kernel_tuner.util as util
import numpy as np
from semver import Version

from .convert import convert_cache
from .file import append_cache_line
from .json import CacheFileJSON, CacheLineJSON
from .paths import get_schema_path
from .versions import LATEST_VERSION, VERSIONS

INFINITY = float("inf")


def __get_schema_for_version(version: str):
    schema_path = get_schema_path(version)
    with open(schema_path, "r") as file:
        return json.load(file)


LATEST_SCHEMA = __get_schema_for_version(LATEST_VERSION)
SCHEMA_REQUIRED_KEYS = LATEST_SCHEMA["required"]
LINE_SCHEMA = LATEST_SCHEMA["properties"]["cache"]["additionalProperties"]
RESERVED_PARAM_KEYS = set(LATEST_SCHEMA["properties"]["cache"]["additionalProperties"]["properties"].keys())


class InvalidCacheError(Exception):
    """Cache file reading or writing failed."""

    def __init__(self, filename: PathLike, message: str, error: Optional[Exception] = None):
        """Constructor for the InvalidCacheError class."""
        super().__init__(str(filename), message, error)
        self.filename = str(filename)
        self.message = message
        self.error = error


class Cache(OrderedDict):
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
        if not isinstance(tune_params, dict) or not all(
            isinstance(key, str) and isinstance(value, list) for key, value in tune_params.items()
        ):
            raise ValueError(
                "Argument tune_params should be a dict with:\n"
                "- keys being parameter keys (and all parameter keys should be used)\n"
                "- values being the parameter's list of possible values"
            )
        if set(tune_params_keys) != set(tune_params.keys()):
            raise ValueError("Expected tune_params to have exactly the same keys as in the list tune_params_keys")
        if len(RESERVED_PARAM_KEYS & set(tune_params_keys)) > 0:
            raise ValueError("Found a reserved key in tune_params_keys")

        # main dictionary for new cache, note it is very important that 'cache' is the last key in the dict
        schema_version = str(LATEST_VERSION)
        inputs = {key: value for key, value in locals().items() if key in SCHEMA_REQUIRED_KEYS}
        cache = Cache(filename=filename, read_only=False, **inputs)

        write_cache_file(cache, filename)
        return cache

    @classmethod
    def read(cls, filename: PathLike, read_only=False):
        """Loads an existing cache. Returns a Cache instance.

        If the cache file does not have the latest version, then it will be read after virtually converting it to the
        latest version. The file in this case is kept the same.
        """
        cache_json = read_cache_file(filename)

        # convert cache to latest schema if needed
        if "schema_version" not in cache_json or cache_json["schema_version"] != LATEST_VERSION:
            cache_json = convert_cache(cache_json)
            # if not read-only mode, update the file
            if not read_only:
                write_cache_file(cast(dict, cache_json), filename)

        cache = Cache(filename=filename, read_only=read_only, **cache_json)
        return cache

    def __init__(self, filename=None, read_only=False, **kwargs):
        """Creates an instance of the Cache"""
        self._filename = Path(filename)
        self._read_only = read_only
        super().__init__(**kwargs)
        self.update(kwargs)

        # ensure 'cache' key is present and is last in the dictionary
        if "cache" not in self:
            self["cache"] = {}
        self.move_to_end("cache")

        jsonschema.validate(instance=self, schema=LATEST_SCHEMA)

        if read_only:
            self.lines = Cache.ReadableLines(self, filename)
        else:
            self.lines = Cache.Lines(self, filename)

    @cached_property
    def filepath(self) -> Path:
        """Returns the path to the cache file."""
        return self._filename

    @cached_property
    def version(self) -> Version:
        """Version of the cache file."""
        return Version.parse(self["schema_version"])

    def __getattr__(self, name):
        if not name.startswith("_"):
            return self[name]
        return super(Cache, self).__getattr__(name)

    class Lines(dict):
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

        def __init__(self, cache: Cache, filename: PathLike):
            """Inits a new CacheLines instance."""
            self._cache = cache
            self._filename = filename
            super().__init__()

            for key, value in cache["cache"].items():
                if not isinstance(value, Cache.Line):
                    self[key] = Cache.Line(value)
                else:
                    self[key] = value

        def append(self, **kwargs) -> None:
            """Appends a cache line to the cache lines."""
            if isinstance(kwargs[self._cache.objective], util.ErrorConfig):
                kwargs[self._cache.objective] = str(kwargs[self._cache.objective])
            line = Cache.Line(**kwargs)

            tune_params = {key: value for key, value in kwargs.items() if key in self._cache.tune_params_keys}
            line_id = self.__get_line_id_from_tune_params_dict(tune_params)
            if line_id in self:
                raise KeyError("Line with given tunable parameters already exists")

            self[line_id] = line
            line_str = _encode_cache_line(line_id, line)
            append_cache_line(line_id, line_str, self._filename)

        def get_from_params(self, default=None, **params) -> Union[Cache.Line, list[Cache.Line]]:
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
            if not default:
                default = []

            if len(params) == 0:
                return list(Cache.Line(line) for line in self.values())
            if not all(key in self._cache.tune_params_keys for key in params):
                raise ValueError("The keys in the parameters should be in `tune_params_keys`")

            line_ids = self.__get_matching_line_ids(params)
            results = [self[k] for k in line_ids]
            if not results:
                return default
            elif len(results) == 1:
                results = results[0]
            return results

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
            return [line_id for line_id in map(self.__get_line_id, param_lists) if line_id in self]

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

    class ReadableLines(Lines):
        """Cache lines in a read_only cache file."""

        def append(*args, **kwargs):
            """Method to append lines to cache file, should not happen with read-only cache"""
            raise ValueError("Attempting to write to read-only cache")

    class Line(dict):
        """Cache line in a cache file"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # check if line has all the required fields and of correct types
            jsonschema.validate(instance=self, schema=LINE_SCHEMA, format_checker=_get_format_checker())

        def __getattr__(self, name):
            if not name.startswith("_"):
                return self[name]
            return super(Line, self).__getattr__(name)


def _encode_cache_line(line_id, line):
    return json.dumps({line_id: line}, cls=CacheEncoder, indent=None).strip()[1:-1]


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


class CacheEncoder(json.JSONEncoder):
    """JSON encoder for Kernel Tuner cache lines.

    Extend the default JSONEncoder with support for the following objects and types:

    +-------------------+----------------------------------+
    | Python            | JSON                             |
    +===================+==================================+
    | util.ErrorConfig  | str                              |
    | np.integer        | int                              |
    | np.floating       | float                            |
    | np.ndarray        | list                             |
    | dict              | object, with 'cache' as last key |
    +-------------------+----------------------------------+

    """

    def __init__(self, *, indent=None, separators=None, **kwargs):
        """Constructor for CacheJSONEncoder, with sensible defaults."""
        if indent is None:
            separators = (", ", ": ")

        super().__init__(indent=indent, separators=separators, **kwargs)

    def default(self, o):
        """Converts non-jsonifiable objects to jsonifiable objects."""

        if isinstance(o, util.ErrorConfig):
            return str(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        super().default(o)

    def iterencode(self, obj, *args, **kwargs):
        """encode an iterator, ensuring 'cache' is the last entry for encoded dicts"""

        # ensure key 'cache' is last in any encoded dictionary
        if isinstance(obj, dict):
            obj = OrderedDict(obj)
            if "cache" in obj:
                obj.move_to_end("cache")

        yield from super().iterencode(obj, *args, **kwargs)


def read_cache_file(filename: PathLike):
    """Reads a cache file and returns its content as a dictionary.

    Parameters:
        filename (PathLike): The path to the cache file.

    Returns:
        dict: The content of the cache file.
    """
    with open(filename, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise InvalidCacheError(filename, "Cache file is not parsable", e)
    return data


def write_cache_file(cache_json: dict, filename: PathLike):
    """Writes a cache file with the given content.

    Parameters:
        cache_file (dict): The content to be written to the cache file.
        filename (PathLike): The path to write the cache file.
    """

    # extract entries from cache
    cache_entries = cache_json["cache"]
    cache_json["cache"] = {}

    # first write cache without entries
    with open(filename, "w") as file:
        json.dump(cache_json, file, cls=CacheEncoder, indent="  ")

    # restore internal state
    cache_json["cache"] = cache_entries

    # add entries line by line
    for key, line in cache_entries.items():
        line_str = _encode_cache_line(key, line)
        append_cache_line(key, line_str, filename)


def convert_cache_file(filestr: PathLike, conversion_functions=None, versions=None, target_version=None):
    """Convert a cache file to the latest/later version.

    Parameters:
        ``filestr`` is the name of the cachefile.

        ``conversion_functions`` is a ``dict[str, Callable[[dict], dict]]``
        mapping a version to a corresponding conversion function.

        ``versions`` is a sorted ``list`` of ``str``s containing the versions.

        ``target`` is the version that the cache should be converted to. By
        default it is the latest version in ``versions``.

    Raises:
        ``ValueError`` if:

            given cachefile has no "schema_version" field and can not be converted
            to version 1.0.0,

            the cachefile's version is higher than the newest version,

            the cachefile's version is not a real version.
    """
    # Load cache from file
    cache = read_cache_file(filestr)

    # Convert cache
    cache = convert_cache(cache, conversion_functions, versions, target_version)

    # Update cache file
    write_cache_file(cache, filestr)


def validate(filename: PathLike):
    """Validates a cache file and raises an error if invalid."""
    cache_json = read_cache_file(filename)
    validate_json(cache_json)


def validate_json(cache_json: Any):
    """Validates cache json."""
    if "schema_version" not in cache_json:
        raise jsonschema.ValidationError("Key 'schema_version' is not present in cache data")
    schema_version = cache_json["schema_version"]
    __validate_json_schema_version(schema_version)
    schema = __get_schema_for_version(schema_version)
    jsonschema.validate(instance=cache_json, schema=schema, format_checker=_get_format_checker())


def __validate_json_schema_version(version: str):
    try:
        if Version.parse(version) in VERSIONS:
            return
    except (ValueError, TypeError):
        pass
    raise jsonschema.ValidationError(f"Invalid version {repr(version)} found.")
