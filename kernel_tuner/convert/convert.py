import json
import jsonschema
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
    version = "1.0.0"

    # Obtain a sorted list of all versions
    versions = list(map (lambda v:v.name, sorted(SCHEMA_VERSIONS_PATH.iterdir())))

    func_list = globals()

    for v in range(len(versions)-1):
        oldver = versions[v]
        newver = versions[v+1]

        if oldver == version:
            func = ("c" + oldver + "_to_" + newver).replace(".", "_")

            if func in func_list:
                print("Calling {}".format(func))
                func_list[func](filestr)
            else:
                print("Calling default conversion")
                default_convert(filestr, oldver, newver)

            version = versions[v+1]

    print("Cache file converted to version {}".format(version))
    return


# Add conversion functions here with naming scheme
# c<old version>_to_<new version>    

"""
def c1_0_0_to_1_1_0(filestr):
    # Convert
    print("Function c1_0_0_to_1_1_0 called")
    return
"""

def c1_1_0_to_1_1_1(filestr):
    # Convert
    print("Function c1_0_0_to_1_1_1 called")
    return


def c1_1_1_to_1_2_0(filestr):
    # Convert
    print("Function c1_1_0_to_1_2_0 called")
    return


def default_convert(filestr, oldver, newver):
    old_schema_path = SCHEMA_VERSIONS_PATH / oldver / "schema.json"
    new_schema_path = SCHEMA_VERSIONS_PATH / newver / "schema.json"

    with open(old_schema_path) as o, open(new_schema_path) as n:
        old_schema = json.load(o)
        new_schema = json.load(n)
        
    cachefile = open(filestr)
    old_cache = json.load(cachefile)
    new_cache = dict()
        
    for key in new_schema["properties"]:
        if key in old_schema["properties"]:
            new_cache[key] = old_cache[key]
        else:
            default_value = DEFAULT_VALUES[(new_schema["properties"][key]["type"])]
            new_cache[key] = default_value

    # Write to new file instead for testing purposes
    with open("new_cache.json", 'w') as f:
        f.write(json.dumps(new_cache, sort_keys=True, indent=4))

    cachefile.close()

    return

convert_cache_file("old_cache.json")