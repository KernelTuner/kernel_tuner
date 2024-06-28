from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import semver
from kernel_tuner.cache.json import (
    CacheFileJSON,
    T4FileJSON,
    T4ResultLineJSON,
    T4ResultMeasurementJSON,
    T4ResultTimesJSON,
)
from kernel_tuner.cache.paths import CACHE_SCHEMAS_DIR
from kernel_tuner.cache.versions import VERSIONS

CONVERSION_FUNCTIONS: dict[str, Callable[[dict], dict]]

DEFAULT_VALUES = {
    "object": {},
    "array": [],
    "string": "default string",
    "integer": 0,
    "number": 0,
    "true": True,
    "false": False,
    "null": None,
}


def convert_cache(cache: dict, conversion_functions=None, versions=None, target_version=None) -> dict:
    """Convert a cache dict to the latest/later version.

    Parameters:
        ``cache`` is the cache dictionary

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
    if conversion_functions is None:
        conversion_functions = CONVERSION_FUNCTIONS

    if versions is None:
        versions = list(map(str, VERSIONS))

    if target_version is None:
        target_version = versions[-1]

    if "schema_version" not in cache:
        cache = unversioned_convert(cache, CACHE_SCHEMAS_DIR)

    version = cache["schema_version"]

    if semver.VersionInfo.parse(version).compare(target_version) > 0:
        raise ValueError(
            f"Target version ({target_version}) should not be " f"smaller than the cache's version ({version})"
        )

    if version not in versions:
        raise ValueError(f"Version ({version}) should be a real " f"existing version")

    if target_version not in versions:
        raise ValueError(f"Target version ({target_version}) should be " f"a real existing version")

    # Main convert loop
    while version != target_version:
        if version in conversion_functions:
            cache = conversion_functions[version](cache)
        else:
            cache = default_convert(cache, version, versions, CACHE_SCHEMAS_DIR)

        version = cache["schema_version"]

    return cache


def default_convert(cache: dict, oldver: str, versions: list, schema_path: Path) -> dict:
    """Attempts a default conversion of ``cache`` to the next highest version.

    Parameters:
        ``cache`` is a ``dict`` representing the cachefile.

        ``oldver`` is the version of the cachefile.

        ``versions`` is a sorted ``list`` of ``str``s containing the versions.

        ``schema_path`` is a ``pathlib`` ``Path``  to the directory containing
        the schema versions.

    Returns:
        a ``dict`` representing the converted cache file.
    """
    # Get the next version
    parts = ["patch", "minor", "major"]
    for part in parts:
        newver = str(semver.VersionInfo.parse(oldver).next_version(part))
        if newver in versions:
            break

    old_schema_path = schema_path / oldver / "schema.json"
    new_schema_path = schema_path / newver / "schema.json"

    with open(old_schema_path) as o, open(new_schema_path) as n:
        old_schema = json.load(o)
        new_schema = json.load(n)

    new_cache = {}
    for key in new_schema["properties"]:
        # It may be the case that the cache does't have a key because it is not
        # required, so check if the key is in the cache
        if key in old_schema["properties"] and key in cache:
            new_cache[key] = cache[key]
        else:
            new_cache[key] = DEFAULT_VALUES[(new_schema["properties"][key]["type"])]

    new_cache["schema_version"] = newver

    return new_cache


def unversioned_convert(cache: dict, schema_path: Path) -> dict:
    """Attempts a conversion of an unversioned cache file to version 1.0.0.

    Parameters:
        ``cache`` is a ``dict`` representing the cachefile.

    Returns:
        a ``dict`` representing the converted cache file.

    Raises:
        ``ValueError`` if given cache file is too old and no suitable
        conversion exists.
    """
    cache["schema_version"] = "1.0.0"

    if "objective" not in cache:
        cache["objective"] = "time"

    for key, entry in cache["cache"].items():
        if "timestamp" not in entry:
            cache["cache"][key]["timestamp"] = "2024-06-18 11:36:56.137831+00:00"
        for missing_key in ["compile_time", "benchmark_time", "framework_time", "strategy_time"]:
            if missing_key not in entry:
                cache["cache"][key][missing_key] = 0

    path = schema_path / "1.0.0/schema.json"

    with open(path) as s:
        versioned_schema = json.load(s)

    missing_keys = [key for key in versioned_schema["properties"] if key not in cache]
    if missing_keys:
        raise ValueError(
            f"Cache file too old, missing key{'s' if len(missing_keys) > 1 else ''} "
            + f"{', '.join(missing_keys)}, no suitable conversion to version 1.0.0 exists."
        )

    return cache


def convert_cache_to_t4(cache: CacheFileJSON) -> T4FileJSON:
    """Converts a cache file to the T4 auto-tuning format.

    ``cache`` is a ``CacheFileJSON`` representing the cache file to convert.

    Returns a ``T4FileJSON`` representing the converted cache file.
    """
    t4 = T4FileJSON(results=[], schema_version="1.0.0")

    for cache_line in cache["cache"].values():
        times = T4ResultTimesJSON(
            compilation_time=cache_line["compile_time"],
            framework=cache_line["framework_time"],
            search_algorithm=cache_line["strategy_time"],
            validation=cache_line["verification_time"],
            runtimes=cache_line["times"],
        )

        measurement = T4ResultMeasurementJSON(name=cache["objective"], value=cache_line[cache["objective"]], unit="")

        result = T4ResultLineJSON(
            timestamp=cache_line["timestamp"],
            configuration={tune_param_key: cache_line[tune_param_key] for tune_param_key in cache["tune_params_keys"]},
            times=times,
            # We assume that the supplied cache file is correct
            invalidity="correct",
            correctness=1,
            measurements=[measurement],
            objectives=[cache["objective"]],
        )
        t4["results"].append(result)

    return t4


########################################################################
# Add conversion functions here which:                                 #
#                                                                      #
# have "_c<old version>_to_<new version>" as name,                     #
# have a single argument 'cache',                                      #
# return 'cache'.                                                      #
#                                                                      #
# The conversion functions are expected to change the "schema_version" #
# field to <new version> themselves.                                   #
#                                                                      #
# For example:                                                         #
# def _c_1_0_0_to_1_1_0(cache):                                        #
#     ...                                                              #
#     cache["schema_version"] = "1.1.0"                                #
#     return cache                                                     #
#                                                                      #
# the list of conversion functions then has to be updated, like this:  #
# CONVERSION_FUNCTIONS = {                                             #
#    ...                                                               #
#    "1.0.0": _c_1_0_0_to_1_1_0,                                       #
#    ...                                                               #
# }                                                                    #
########################################################################

CONVERSION_FUNCTIONS = {}
