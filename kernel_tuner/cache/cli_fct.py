"""
This file cli_fct.py contains the helper functions used in cli.py.
This way, we can split the files correctly and not obtain messy code.
Merging now works, inspecting and conversion still needs to be completed.
"""

from .cache import Cache
from .file import read_cache, write_cache 
from .convert import convert_cache_file

from pathlib import Path
from os import PathLike
from typing import List

from shutil import copyfile

import argparse
import json
import jsonschema


def fileExists(fileName: PathLike) -> bool:
    """Validates if the file specified by fileName even exists."""
    return Path(fileName).is_file()

def checkEquivalence(listOfFiles: List[PathLike]):
    """Checks equivalence of set parameters for files in `listOfFiles`.
    Assumes that all files have been validated.
    We use the first file (listOfFiles[0]) as our base file, and compare everything with that."""
    baseFile = Cache.read(listOfFiles[0])

    for i in range(1, len(listOfFiles)):
        tempFile = Cache.read(listOfFiles[i])

        # Now the equivalence logic

        # Merging is yet to be updated to work with different schema versions.
        checkProperties(        )
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


def mergeFiles(listOfFiles: List[PathLike], ofile: PathLike):
    """Merges the actual files and writes to the file `ofile`."""
    """Assumes that all files have been validated."""
    # FIXME: Cannot be guaranteed that the order of the cachelines in the files is also kept when merging
    # From cache.py (json.load).

    resultingOutput = Cache.read(listOfFiles[0])
    resultingOutput.create(ofile, device_name=resultingOutput.device_name, \
    kernel_name=resultingOutput.kernel_name, problem_size=resultingOutput.problem_size, \
    tune_params_keys=resultingOutput.tune_params_keys, tune_params=resultingOutput.tune_params, \
    objective=resultingOutput.objective)

    # We read so the ._filename changes for append
    resultingOutput = Cache.read(ofile)

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



def cli_get(apRes: argparse.Namespace):
    """Checks if entry (string) `checkEntry` is inside file `inFile`, by using
    the `cache.py` library.
    Does not perform syntax checking on `checkEntry`."""

    iFile = Cache.read(apRes.infile[0])

    cacheLine = iFile.lines.get(apRes.key) # apres-ski?

    if cacheLine == None:
        raise ValueError(f"Cacheline entry '{apRes.key}' is not contained in cachefile '{apRes.infile[0]}'.")

    else:
        print("[*] Cacheline entry '{}' content [*]\n\n************************".format(str(apRes.key)))
        print(dict(cacheLine.items()))
        print("************************")



def cli_delete(apRes: argparse.Namespace):
    """
    Tries to remove entry `removeEntry` from file `inFile`, by using the
    `file.py` functions read_cache, write_cache. 
    We delete the json entry ["cache"][`removeEntry`] from the returned JSON object
    from read_cache, then use write_cache() to write the result to the desired output file
    `outFile`.
    First we check if the entry actually exists in the cachefile using the library. If not, there
    is nothing to do.
    """

    inFile = outFile = apRes.infile[0] 

    if (apRes.output != None):
        outFile = apRes.output 

    cacheFile = Cache.read(inFile)

    
    if (cacheFile.lines.get(apRes.key) == None):
        raise ValueError(f"Entry '{apRes.key}' is not contained in cachefile '{inFile}'.")


    # FIXME: want to use the "safe" library version instead of these functions.
    # At time of commit library still needs to be updated.
    jsonData = read_cache(inFile)

    del jsonData["cache"][apRes.key]


    write_cache(jsonData, outFile)

    print("\n[*] Writing to output file '{}' after removing entry '{}' completed.".format(str(outFile), str(apRes.key)))






def cli_convert(apRes: argparse.Namespace):
    """The main function for handling the conversion of a cachefile.
    Not completed yet."""

    read_file  = apRes.infile
    write_file = apRes.output

    if not fileExists(read_file):
        raise ValueError(f"Can not find file \"{read_file}\"")
    
    if write_file is not None and write_file[-5:] != ".json":
        raise ValueError(f"Please specify a .json file for the output file")
    
    if write_file is None:
        write_file = read_file
    else:
        copyfile(read_file, write_file)
    
    convert_cache_file(filestr=write_file,
                       target_version=apRes.target)





def cli_merge(apRes: argparse.Namespace):
    """The main function for handling the merging of two or more cachefiles.
    First, we must validate the existence and validity of cachefiles, then we merge."""
    fileList = apRes.files

    if (len(fileList) < 2):
        raise ValueError(f"Not enough (< 2) files provided to merge.")
    # Perform validation, equivalence and after merge.

    for i in fileList:
        Cache.read(i)

    # Tobias: You would need to add convert to equivalent schema version function call here

    checkEquivalence(fileList)

    mergeFiles(fileList, apRes.output)

    print("[*] Merging finished. Output file: '{}'.".format(str(apRes.output)))
