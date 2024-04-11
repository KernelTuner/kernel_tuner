"""Provides utilities for reading and writing cachefiles."""

import json
import re
from os import PathLike

from kernel_tuner.cache.json_encoder import CacheJSONEncoder


TERMINATE_REGEX = re.compile(r"}\s*}$")


def read_cache_file(file_path: PathLike) -> dict:
    """Reads a cache file and returns its content as a dictionary.

    Parameters:
        file_path (Path): The path to the cache file.

    Returns:
        dict: The content of the cache file.
    """
    with open(file_path, "r") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            pass  # Loading failed

        try:  # Try to load the file with appended terminator
            return json.loads(file.read() + "}}")
        except json.JSONDecodeError:
            raise ValueError(f"Cache file {file} is corrupted")


def write_cache_file(cache_json: dict, file_path: PathLike, *, terminate=True):
    """Writes a cache file with the given content.

    Parameters:
        cache_file (dict): The content to be written to the cache file.
        file_path (Path): The path to write the cache file.
        terminated (bool): Whether to add the final two accolades ('}')
    """
    with open(file_path, "w") as file:
        text = json.dumps(cache_json, cls=CacheJSONEncoder, indent=0)
        if not terminate:
            # TODO: Write test
            text = TERMINATE_REGEX.sub("", text)
        file.write(text)
