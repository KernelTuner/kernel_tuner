import json
from pathlib import Path

from kernel_tuner.cache.KTLibrary.CacheJSONEncoder import CacheJSONEncoder



def read_cache_file(file_path: Path) -> dict:
    """
    Reads a cache file and returns its content as a dictionary.

    Parameters:
        file_path (Path): The path to the cache file.

    Returns:
        dict: The content of the cache file.
    """
    with open(file_path, "r") as file:
        json_data = json.load(file)
        print("JSON data loaded successfully")
        return json_data


def write_cache_file(cache_file: dict, file_path: Path):
    """
    Writes a cache file with the given content.

    Parameters:
        cache_file (dict): The content to be written to the cache file.
        file_path (Path): The path to write the cache file.
    """
    with open(file_path, "w+") as file:
        json.dump(cache_file, file, cls=CacheJSONEncoder, indent=0)
