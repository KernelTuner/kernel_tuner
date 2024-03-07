import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parents[0]

SCHEMA_VERSIONS_PATH = PROJECT_DIR / 'cache'

CURRENT_VERSION = "1.2.0"

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


def convert_cache_file(filestr):
    # Hardcoded for now
    version = "1.0.0"

    # Obtain a sorted list of all versions
    versions = list(map (lambda v:v.name, sorted(SCHEMA_VERSIONS_PATH.iterdir())))

    func_list = globals()

    for v in range(len(versions)-1):
        oldver = versions[v]
        newver = versions[v+1]

        if oldver == version:
            # Convert cachefile from oldver to newver

            with open(filestr, 'r') as cachefile:
                cache = json.load(cachefile)

            # Look first for an implemented conversion function
            # If no such function exists, attempt default conversion
            func = ("c" + oldver + "_to_" + newver).replace(".", "_")


            if func in func_list:
                cache = func_list[func](cache)
            else:
                cache = default_convert(cache, oldver, newver)


            with open(filestr, 'w') as cachefile: 
                cachefile.write(json.dumps(cache, indent=4))

            version = newver

    print("Cache file converted from version {} to version {}".format("1.0.0", version))
    return


def default_convert(cache, oldver, newver):
    old_schema_path = SCHEMA_VERSIONS_PATH / oldver / "schema.json"
    new_schema_path = SCHEMA_VERSIONS_PATH / newver / "schema.json"

    with open(old_schema_path) as o, open(new_schema_path) as n:
        old_schema = json.load(o)
        new_schema = json.load(n)
        
    new_cache = dict()
    for key in new_schema["properties"]:
        if key in old_schema["properties"]:
            new_cache[key] = cache[key]
        else:
            new_cache[key] = DEFAULT_VALUES[(new_schema["properties"][key]["type"])]

    return new_cache

# Add conversion functions here with naming scheme
# c<old version>_to_<new version>    


def c1_1_0_to_1_1_1(cache):
    # No action needed

    return cache


def c1_1_1_to_1_2_0(cache):
    # Could potentially grab the type from the corresponding schema file
    cache["field1"] = DEFAULT_VALUES["object"]

    return cache