"""This file cli_fct.py contains several functions used to perform several operations on cachefiles.

This is one of:
   - `convert`: one may convert a cachefile to a specified (higher) version; for this we have function convert().
   - `convert-t4`: one may convert a cachefile to T4 format; for this we have function convert_t4().
   - `delete-line`: one may delete a certain cacheline from a cachefile; required that the cachefile is of the newest
       JSON schema version. For this we have the delete_line() function.
   - `get-line`: One may obtain the cacheline data of a certain cacheline inside a cachefile. For this, we have the 
       get_line() function.
   - `merge`: one may merge several (with possible non-equivalent version) cachefiles into one cachefile;
      for this we have the merge_files() function. The resulting merged cachefile is always using the **newest** JSON
      schema.

   For these main operations to work, there are also helper functions such as convert_new_schema().
"""  

from os import PathLike
from pathlib import Path
from shutil import copyfile
from typing import List

from packaging import version

from .cache import Cache
from .convert import convert_cache_file, convert_cache_to_t4
from .file import read_cache, write_cache
from .versions import LATEST_VERSION


def convert_new_schema(file_list: List[PathLike]):
    """Converts all non-new versioned files in file_list to the newest."""
    max_version = get_highest_schema_version(file_list)

    for file in file_list:
        cachefile = Cache.read(file)

        if str(cachefile.version) != max_version:
            # We write to the input filename.
            convert_cache_file(filestr=file, target_version=max_version)
        

def get_highest_schema_version(file_list: List[PathLike]) -> str:
    """Returns the highest `schema_version` of all cachefiles in `file_list`."""
    version_list = []

    for file in file_list:
        cachefile = Cache.read(file)
        version_list.append(str(cachefile.version))

    newest_version = max(version_list, key=version.parse)

    return newest_version

    

def file_exists(file: PathLike) -> bool:
    """Validates if the file specified by fileName even exists."""
    return Path(file).is_file()

def check_equivalence(file_list: List[PathLike]):
    """Checks equivalence of set parameters for files in `file_list`.

    Assumes that all files have been validated.
    We use the first file (file_list[0]) as our base file, and compare everything with that.
    """
    base_file = Cache.read(file_list[0])


    for file in file_list[1:]:
        temp_file = Cache.read(file)

        # Now the equivalence logic

        # Merging is yet to be updated to work with different schema versions.
        
        if base_file.version != temp_file.version:
            raise KeyError(f"Merge error; files '{file_list[0]}' and '{file}' do not have the same schema version.")

        elif base_file.device_name != temp_file.device_name:
            raise KeyError(f"Merge error; key 'device_name' not equivalent for '{file_list[0]}' and '{file}'.")

        elif base_file.kernel_name != temp_file.kernel_name:
            raise KeyError(f"Merge error; key 'kernel_name' not equivalent for '{file_list[0]}' and '{file}'.")

        elif base_file.problem_size != temp_file.problem_size:
            raise KeyError(f"Merge error; key 'problem_size' not equivalent for '{file_list[0]}' and '{file}'.")

        elif base_file.objective != temp_file.objective:
            raise KeyError(f"Merge error; key 'objective' not equivalent for '{file_list[0]}' and '{file}'.")

        elif base_file.tune_params_keys != temp_file.tune_params_keys:
            raise KeyError(f"Merge error; key 'tune_params_keys' not equivalent for '{file_list[0]}' and '{file}'.")


def merge_files(file_list: List[PathLike], ofile: PathLike):
    """Merges the actual files and writes to the file `ofile`.

    Assumes that all files have been validated.
    """
    # FIXME: Cannot be guaranteed that the order of the cachelines in the files is also kept when merging
    # From cache.py (json.load).

    resulting_output = Cache.read(file_list[0])
    resulting_output.create(ofile, \
        device_name=resulting_output.device_name, \
        kernel_name=resulting_output.kernel_name, \
        problem_size=resulting_output.problem_size, \
        tune_params_keys=resulting_output.tune_params_keys, \
        tune_params=resulting_output.tune_params, \
        objective=resulting_output.objective)

    # We read so the ._filename changes for append
    resulting_output = Cache.read(ofile)

    # Now for each file add the cache content.
    for file in file_list:

        temp_file = Cache.read(file)

        for line in temp_file.lines:
            if resulting_output.lines.get(line) is not None:
                raise KeyError(f"Merge error; overlap for key '{line}' in several files.")
            
            temp_line = temp_file.lines[line]
            tune_params = {key: temp_line[key] for key in temp_file.tune_params_keys}
            resulting_output.lines.append(time=temp_line["time"],
                         compile_time=temp_line["compile_time"],
                         verification_time=temp_line["verification_time"],
                         benchmark_time=temp_line["benchmark_time"],
                         strategy_time=temp_line["strategy_time"],
                         framework_time=temp_line["framework_time"],
                         timestamp=temp_file.lines.get(line).timestamp,
                         times=temp_line["times"],
                         GFLOP_per_s=temp_line["GFLOP/s"],
                         **tune_params)


def get_line(infile: PathLike, key: any):
    """Checks if entry (string) `apRes.key` is inside file `in_file`, by using the `cache.py` library."""
    cache_infile = Cache.read(infile)

    cache_line = cache_infile.lines[key]

    print(f"[*] Cacheline entry '{key}' content [*]\n\n************************")
    print(dict(cache_line.items()))
    print("************************")



def delete_line(infile: PathLike,  delete_key: any, outfile=None):
    """Tries to remove entry `delete_key` from file `infile`, then write to `outfile` by using the `cache.py` functions.

    First we check if the entry actually exists in the cachefile using the library. If not, there
    is nothing to do.
    If it exists, we delete by inverse-appending.
    """
    if outfile is None:
        outfile = infile


    cache_infile = Cache.read(infile)

    # We require the file to be of the latest version.
    if cache_infile.version != LATEST_VERSION:
        raise ValueError(f"Cachefile '{infile}' is of version {str(cache_infile.version)} but should be of version "\
                         f"{str(LATEST_VERSION)} (latest).")

    if cache_infile.lines.get(delete_key) is None:
        raise KeyError(f"Entry '{delete_key}' is not contained in cachefile '{infile}'.")

    cache_infile.create(outfile, \
        device_name=cache_infile.device_name, \
        kernel_name=cache_infile.kernel_name, \
        problem_size=cache_infile.problem_size, \
        tune_params_keys=cache_infile.tune_params_keys, \
        tune_params=cache_infile.tune_params, \
        objective=cache_infile.objective)


    # We read so the ._filename changes for append
    resulting_output = Cache.read(outfile)

    for line in cache_infile.lines:
        if line != delete_key:
            temp_line = cache_infile.lines[line]
            tune_params = {key: temp_line[key] for key in cache_infile.tune_params_keys}
            resulting_output.lines.append(time=temp_line["time"],
                         compile_time=temp_line["compile_time"],
                         verification_time=temp_line["verification_time"],
                         benchmark_time=temp_line["benchmark_time"],
                         strategy_time=temp_line["strategy_time"],
                         framework_time=temp_line["framework_time"],
                         timestamp=cache_infile.lines.get(line).timestamp,
                         times=temp_line["times"],
                         GFLOP_per_s=temp_line["GFLOP/s"],
                         **tune_params)

    print(f"Writing to output file '{outfile}' after removing entry '{delete_key}' completed.")



def convert(read_file: PathLike, write_file=None, target=None):
    """The main function for handling the version conversion of a cachefile."""
    if not file_exists(read_file):
        raise ValueError(f"Can not find file \"{read_file}\"")
    
    if write_file is not None and write_file[-5:] != ".json":
        raise ValueError("Please specify a .json file for the output file")
    
    # If no output file is specified, let the conversion overwrite the input file
    if write_file is None:
        write_file = read_file
        
    else:
        copyfile(read_file, write_file)
    
    convert_cache_file(filestr=write_file,
                       target_version=target)



def convert_t4(read_file: PathLike, write_file=None):
    """The main function for handling the T4 conversion of a cachefile."""
    if not file_exists(read_file):
        raise ValueError(f"Can not find file \"{read_file}\"")
    
    if write_file is not None and write_file[-5:] != ".json":
        raise ValueError("Please specify a .json file for the output file")
    
    cache = read_cache(read_file)

    t4_cache = convert_cache_to_t4(cache)

    write_cache(t4_cache, write_file)



def merge(file_list: List[PathLike], outfile: PathLike):
    """The main function for handling the merging of two or more cachefiles.

    First, we must validate the existence and validity of cachefiles, then we merge.
    """
    if (len(file_list) < 2):
        raise ValueError("Not enough (< 2) files provided to merge.")

    # Perform validation, conversion, equivalence check and after merge.

    for i in file_list:
        Cache.read(i)

    # Convert all files in `file_list` that are not matching the newest `schema_version` to the newest schema version.
    convert_new_schema(file_list)

    check_equivalence(file_list)

    merge_files(file_list, outfile)

    print(f"[*] Merging finished. Output file: '{outfile}'.")

