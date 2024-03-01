import json
from pathlib import Path

PROJECT_DIR = Path('.')

SCHEMA_VERSIONS_PATH = PROJECT_DIR / 'cache'

CURRENT_VERSION = "1.2.0"

def convert_cache_file(filestr):
    #cache_data = json.loads(filestr)
    #version = cache_data["version"]
    version = "1.0.0"

    version = version.replace(".", "_")

    versions = sorted(SCHEMA_VERSIONS_PATH.iterdir())
    versions = list(map (lambda v:v.name,              versions))
    versions = list(map (lambda v:v.replace(".", "_"), versions))



    for v in range(len(versions)-1):
        if versions[v] == version:
            func = "c" + versions[v] + "_to_" + versions[v+1]
            print("Calling {}".format(func))
            globals()[func](filestr)
            version = versions[v+1]

    print("Cache file converted to version {}".format(version.replace("_", ".")))
    return


# Add conversion functions here with naming scheme
# c<old version>_to_<new version>    

def c1_0_0_to_1_1_0(filestr):
    # Convert
    print("Function c1_0_0_to_1_1_0 called")
    return


def c1_1_0_to_1_1_1(filestr):
    # Convert
    print("Function c1_0_0_to_1_1_1 called")
    return


def c1_1_1_to_1_2_0(filestr):
    # Convert
    print("Function c1_1_0_to_1_2_0 called")
    return



convert_cache_file("abc")