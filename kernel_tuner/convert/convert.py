import json
from pathlib import Path

PROJECT_DIR = Path('.')

SCHEMA_VERSIONS_PATH = PROJECT_DIR / 'cache'

CURRENT_VERSION = "1.2.0"

def convert_cache_file(filestr):
    #cache_data = json.loads(filestr)
    #version = cache_data["version"]
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

def c1_0_0_to_1_1_0(filestr):
    # Convert
    print("Function c1_0_0_to_1_1_0 called")
    return

"""
def c1_1_0_to_1_1_1(filestr):
    # Convert
    print("Function c1_0_0_to_1_1_1 called")
    return
"""

def c1_1_1_to_1_2_0(filestr):
    # Convert
    print("Function c1_1_0_to_1_2_0 called")
    return


def default_convert(filestr, oldver, newver):
    # Convert
    print("Attempting default convert")
    return


convert_cache_file("abc")