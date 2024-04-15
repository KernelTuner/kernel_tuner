"""Provides utilities for reading and writing cachefiles."""

import json
import re
from os import PathLike

from kernel_tuner.cache.json_encoder import CacheJSONEncoder


OPTIONAL_COMMA_END_REGEX = re.compile(r",?$")
CLOSING_BRACES_REGEX = re.compile(r"}\s*}$")


def read_cache_file(file_path: PathLike):
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

    with open(file_path, "r") as file:
        try:  # Try to load the file with closing braces appended to it
            text = file.read()
            text = OPTIONAL_COMMA_END_REGEX.sub("}}", text)
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Cache file {file_path} is corrupted")


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
            text = CLOSING_BRACES_REGEX.sub("", text)
        file.write(text)
