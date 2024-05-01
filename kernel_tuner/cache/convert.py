from __future__ import annotations

import json
from os import PathLike
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

PROJECT_DIR = Path(__file__).parents[0]

SCHEMA_VERSIONS_PATH = PROJECT_DIR / "../schema/cache"

VERSIONS = list(sorted((p.name for p in SCHEMA_VERSIONS_PATH.iterdir()), key=semver.Version.parse))

CONVERSION_FUNCTIONS: dict[str, Callable[[dict], dict]]

DEFAULT_VALUES = {
    "object":   dict(),
    "array":    list(),
    "string":   "default string",
    "integer":  0,
    "number":   0,
    "true":     True,
    "false" :   False,
    "null":     None
}



def convert_cache_file(filestr : PathLike, 
                       conversion_functions=None,
                       versions=None):
    """Convert a cache file to the newest version.

    ``filestr`` is the name of the cachefile.

    ``conversion_functions`` is a ``dict[str, Callable[[dict], dict]]``
    mapping a version to a corresponding conversion function.

    ``versions`` is a sorted ``list`` of ``str``s containing the versions.

    
    raises ``ValueError`` if:

    given cachefile has no "schema_version" field,

    the cachefile's version is higher than the newest version,

    the cachefile's version is not a real version.
    """
    if conversion_functions is None:
        conversion_functions = CONVERSION_FUNCTIONS

    if versions is None:
        versions = VERSIONS

    # Load cache
    with open(filestr, 'r') as cachefile:
        cache = json.load(cachefile)

    if "schema_version" not in cache:
        raise ValueError("Cache file has no \"schema_version\" field, "
                         "unversioned conversion not yet implemented.")
    
    version = cache["schema_version"]
    target_version = versions[-1]

    if semver.VersionInfo.parse(version).compare(target_version) > 0:
        raise ValueError(f"Target version ({target_version}) should not be "
                         f"smaller than the cache's version ({version})")

    if version not in versions:
        raise ValueError(f"Version ({version}) should be a real "
                         f"existing version")
    
    # Main convert loop
    while version != target_version:
        if version in conversion_functions:
            cache = conversion_functions[version](cache)
        else:
            cache = default_convert(cache, version, versions,
                                    SCHEMA_VERSIONS_PATH)

        version = cache["schema_version"]

    with open(filestr, 'w') as cachefile: 
        cachefile.write(json.dumps(cache, indent=4))

    return


def default_convert(cache       : dict,
                    oldver      : str,
                    versions    : list,
                    schema_path : Path) -> dict:
    """Attempts a default conversion of ``cache`` to the next highest version.

    ``cache`` is a ``dict`` representing the cachefile.

    ``oldver`` is the version of the cachefile.

    ``versions`` is a sorted ``list`` of ``str``s containing the versions.

    ``schema_path`` is a ``pathlib`` ``Path``  to the directory containing
    the schema versions.

    Returns a ``dict`` representing the converted cache file.
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
        
    new_cache = dict()
    for key in new_schema["properties"]:
        # It may be the case that the cache does't have a key because it is not
        # required, so check if the key is in the cache
        if key in old_schema["properties"] and key in cache:
            new_cache[key] = cache[key]
        else:
            new_cache[key] = DEFAULT_VALUES[(new_schema["properties"][key]["type"])]

    new_cache["schema_version"] = newver

    return new_cache


def convert_cache_to_t4(cache: CacheFileJSON) -> T4FileJSON:
    """Converts a cache file to the T4 auto-tuning format.
    
    ``cache`` is a ``CacheFileJSON`` representing the cache file to convert.

    Returns a ``T4FileJSON`` representing the converted cache file.
    """
    t4 = T4FileJSON(results = [], schema_version = "1.0.0")

    for cache_line in cache["cache"].values():
        times = T4ResultTimesJSON(
            compilation_time = cache_line["compile_time"],
            framework = cache_line["framework_time"],
            search_algorithm = cache_line["strategy_time"],
            validation = cache_line["verification_time"],
            runtimes = cache_line["times"]
        )

        measurement = T4ResultMeasurementJSON(
            name = cache["objective"],
            value = cache_line[cache["objective"]],
            unit = ""
        )

        result = T4ResultLineJSON(
            timestamp = cache_line["timestamp"],
            configuration = {
                tune_param_key: cache_line[tune_param_key] for tune_param_key in cache["tune_params_keys"]
            },
            times = times,
            # We assume that the supplied cache file is correct
            invalidity = "correct",
            correctness = 1,
            measurements = [ measurement ],
            objectives = [ cache["objective"] ]
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

CONVERSION_FUNCTIONS = {

}
