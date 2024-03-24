from __future__ import annotations

import json
import semver
from pathlib import Path
from typing import Callable


PROJECT_DIR = Path(__file__).parents[0]

SCHEMA_VERSIONS_PATH = PROJECT_DIR / "../schema/cache"

VERSIONS = list(map (lambda v:v.name, sorted(SCHEMA_VERSIONS_PATH.iterdir())))

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



def convert_cache_file(filestr : str, 
                       conversion_functions=None,
                       versions=None):
    """Convert a cache file to the newest version.

    ``filestr`` is the name of the cachefile.

    ``conversion_functions`` is a ``dict[str, Callable[[dict], dict]]``
    mapping a version to a corresponding conversion function.

    ``versions`` is a ``list`` of ``str``s containing the versions.

    
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
        raise ValueError(f"Cache file has no \"schema_version\" field, "
                         f"unversioned conversion not yet implemented.")
    
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
            cache = default_convert(cache, version, versions)

        version = cache["schema_version"]

    with open(filestr, 'w') as cachefile: 
        cachefile.write(json.dumps(cache, indent=4))

    return


def default_convert(cache    : dict,
                    oldver   : str,
                    versions : list) -> dict:
    """Attempts a default conversion of a cachefile to the next highest version.

    ``cache`` is a ``dict`` representing the cachefile.

    ``oldver`` is the version of the cachefile.

    ``versions`` is a ``list`` of ``str``s containing the versions.
    """
    newver = versions[versions.index(oldver) + 1]

    old_schema_path = SCHEMA_VERSIONS_PATH / oldver / "schema.json"
    new_schema_path = SCHEMA_VERSIONS_PATH / newver / "schema.json"

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

    return new_cache



#######################################################################
# Add conversion functions here that:                                 #
#                                                                     #
# has "_c<old version>_to_<new version>"" as name,                    #
# has a single argument 'cache',                                      #
# returns 'cache'.                                                    #
#                                                                     #
# For example:                                                        #
# def _c_1_0_0_to_1_1_0(cache):                                       #
#     ...                                                             #
#     return cache                                                    #
#                                                                     #
# the list of conversion functions then has to be updated, like this: #
# CONVERSION_FUNCTIONS = {                                            #
#    ...                                                              #
#    "1.0.0": _c_1_0_0_to_1_1_0,                                      #
#    ...                                                              #
# }                                                                   #
#######################################################################

CONVERSION_FUNCTIONS = {

}
