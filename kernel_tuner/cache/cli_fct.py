"""
This file cli_fct.py contains the helper functions used in cli.py.
This way, we can split the files correctly and not obtain messy code.
Merging now works, inspecting and conversion still needs to be completed.
"""

from pathlib import Path
from os import PathLike
from typing import *
from .paths import *
from .cache import *
import json
import jsonschema


def fileExists(fileName: PathLike) -> bool:
    """Validates if the file specified by fileName even exists."""
    return Path(fileName).is_file()

def checkEquivalence(listOfFiles: list[PathLike]):
    """Checks equivalence of set parameters for files in `listOfFiles`.
    Assumes that validateFiles() has been passed.
    We use the first file (listOfFiles[0]) as our base file, and compare everything with that."""
    baseFile = Cache.read(listOfFiles[0])

    for i in range(1, len(listOfFiles)):
        tempFile = Cache.read(listOfFiles[i])

        # Now the equivalence logic

        if (baseFile.version != tempFile.version):
            print("Error in merging; files '{}' and '{}' are not of the same schema version.".format(str(listOfFiles[0]), str(listOfFiles[i])))
            exit()

        if (baseFile.device_name != tempFile.device_name):
            print("Error in merging; key 'device_name' is not equivalent for files '{}' and '{}'.".format(str(listOfFiles[0]), str(listOfFiles[i])))
            exit()

        if (baseFile.kernel_name != tempFile.kernel_name):
            print("Error in merging; key 'kernel_name' is not equivalent for files '{}' and '{}'.".format(str(listOfFiles[0]), str(listOfFiles[i])))
            exit()

        # Q: should this be equivalent?
        if (baseFile.problem_size != tempFile.problem_size):
            print("Error in merging; key 'problem_size' is not equivalent for files '{}' and '{}'.".format(str(listOfFiles[0]), str(listOfFiles[i])))
            exit()

        if (baseFile.objective != tempFile.objective):
            print("Error in merging; key 'objective' is not equivalent for files '{}' and '{}'.".format(str(listOfFiles[0]), str(listOfFiles[i])))
            exit()

        if (baseFile.tune_params_keys != tempFile.tune_params_keys):
            print("Error in merging; key 'tune_params_keys' is not equivalent for files '{}' and '{}'.".format(str(listOfFiles[0]), str(listOfFiles[i])))
            exit()


def validateFiles(listOfFiles: list[PathLike]):
    """Validates cachefiles. Validates existence and correctness of the files in `listOfFiles`.
    Correctness means being a valid JSON file adhering to a JSON schema."""

    for i in listOfFiles:
        if not fileExists(i):
            print("Error in reading file '{}'. File does not exist.".format(str(i)))
            exit()

        try:
            Cache.read(i)

        except KeyError:
            print("Error in reading file '{}'. Key {} does not exist.".format(str(i), str(e)))
            exit()

        except FileNotFoundError:
            # We now know that the filenotfounderror is from the json schema lookup.
            print("Error. Either your value for 'schema_version' is invalid or the jsonschema does not exist in the current repository.")
            exit()

        except jsonschema.exceptions.ValidationError:
            print("Error. File '{}' is not adhering to the JSON schema.".format(str(i)))
            exit()

def mergeFiles(listOfFiles: list[PathLike], ofile: PathLike):
    """Merges the actual files and writes to the file `ofile`."""
    """Assumes that checks in validateFiles() have been performed."""

    resultingOutput = Cache.read(listOfFiles[0])
    resultingOutput.create(ofile, device_name=resultingOutput.device_name, \
    kernel_name=resultingOutput.kernel_name, problem_size=resultingOutput.problem_size, \
    tune_params_keys=resultingOutput.tune_params_keys, tune_params=resultingOutput.tune_params, \
    objective=resultingOutput.objective)

    resultingOutput._filename = ofile

    # Now for each file add the cache content.
    # Does not check for duplicates
    for i in range(0, len(listOfFiles)):

        tempFile = Cache.read(listOfFiles[i])

        for line in tempFile.lines:
            tune_params = {key: tempFile.lines[line][key] for key in tempFile.tune_params_keys}
            resultingOutput.lines.append(time=tempFile.lines[line]["time"],
                             compile_time=tempFile.lines[line]["compile_time"],
                             verification_time=tempFile.lines[line]["verification_time"],
                             benchmark_time=tempFile.lines[line]["benchmark_time"],
                             strategy_time=tempFile.lines[line]["strategy_time"],
                             framework_time=tempFile.lines[line]["framework_time"],
                             timestamp=tempFile.lines.get(line).timestamp,
                             times=tempFile.lines[line]["times"],
                             GFLOP_per_s=tempFile.lines[line]["GFLOP/s"],
                             **tune_params)

def cliAppend(inFile: PathLike, appendFile: PathLike):
    """The function handling the appending to file `inFile`, reading content
    from `appendFile` (cacheline entries).
    Not completed yet.
    """
    print("[*] Appending not completed.")

def cliCheck(inFile: PathLike, checkEntry: str):
    """Checks if entry (string) `checkEntry` is inside file `inFile`, by using
    the `cache.py` library.
    Does not perform syntax checking on `checkEntry`."""

    validateFiles([inFile])

    iFile = Cache.read(inFile)

    if (iFile.lines.get(checkEntry) != None):
        print("Cacheline entry '{}' is contained in cachefile '{}'.".format(str(checkEntry), str(inFile)))
        exit(0)

    else:
        print("Cacheline entry '{}' is not contained in cachefile '{}'.".format(str(checkEntry), str(inFile)))
        exit(0)



def cliRemove(inFile: PathLike, removeEntry: str):
    """Tries to remove entry `removeEntry` from file `inFile`, by using the
    `cache.py` library. Note that the `cache.py` has no removing functionality
    yet, hence we remove by appending everything but the entry. Note that this
    is very inefficient, but for now there is no better (safe) way yet.
    First we check if the entry actually exists in the cachefile. If not, there
    is nothing to do.
    Not completed yet.
    """

    validateFiles([inFile])

    cacheFile = Cache.read(inFile)

    if (cacheFile.lines.get(removeEntry) == None):
        print("Error. Entry '{}' is not contained in cachefile '{}'.".format(str(removeEntry), str(inFile)))
        exit(0)




def cli_convert(apRes):
    """The main function for handling the conversion of a cachefile.
    Not completed yet."""
    print("[*] In cli_convert()")

def cli_inspect(apRes):
    """The main function for handling the inspection of a cachefile.
    Not completed yet."""

    # we only allow either add, remove, or check
    # append
    if (apRes.append and not (apRes.remove != None or apRes.check != None)):
        cliAppend(apRes.inFile, apRes.file)

    elif (not apRes.append and apRes.remove != None and apRes.check == None):
        cliRemove(apRes.infile, apRes.remove)

    elif (not apRes.append and apRes.remove == None and apRes.check != None):
        cliCheck(apRes.infile, apRes.check)

    else:
        print("Error. You must select one of the options appending, removing or checking.")
        exit(1)

    exit()

def cli_merge(apRes):
    """The main function for handling the merging of two or more cachefiles.
    First, we must validate the existence and validity of cachefiles, then we merge."""
    fileList = apRes.files

    # Perform validation, equivalence and after merge.
    validateFiles(fileList)

    checkEquivalence(fileList)

    mergeFiles(fileList, apRes.output)

    print("[*] Merging finished. Output file: '{}'.".format(str(apRes.output)))
    exit()
