"""Provides utilities for reading and writing cachefiles."""

import json
import re
import os
from os import PathLike

from kernel_tuner.cache.json_encoder import CacheJSONEncoder


OPTIONAL_COMMA_END_REGEX = re.compile(r",?$")
CLOSING_BRACES_REGEX = re.compile(r"}\s*}\s*$")


def read_cache_file(filename: PathLike):
    """Reads a cache file and returns its content as a dictionary.

    Parameters:
        filename (PathLike): The path to the cache file.

    Returns:
        dict: The content of the cache file.
    """
    with open(filename, "r") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            pass  # Loading failed

        file.seek(0)
        try:  # Try to load the file with closing braces appended to it
            text = file.read()
            text = OPTIONAL_COMMA_END_REGEX.sub("}}", text, 1)
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Cache file {filename} is corrupted")


def write_cache_file(cache_json: dict, filename: PathLike, *, keep_open=False):
    r"""Writes a cache file with the given content.

    Parameters:
        cache_file (dict): The content to be written to the cache file.
        filename (PathLike): The path to write the cache file.
        keep_open (bool): If true, add a comma instead of the final two braces ('}\n}')
    """
    with open(filename, "w") as file:
        text = json.dumps(cache_json, cls=CacheJSONEncoder, indent=0)
        if keep_open:
            text = CLOSING_BRACES_REGEX.sub(",", text)
        file.write(text)


def close_cache_file(filename: PathLike):
    with open(filename, "rb+") as file:
        file.seek(-1, os.SEEK_END)
        ch = file.peek(1)
        while ch.isspace() and ch != b",":
            if file.tell() == 0:
                raise ValueError(f"Cache file {filename} is corrupted")
            file.seek(-1, os.SEEK_CUR)
            ch = file.peek(1)
        file.seek(1, os.SEEK_CUR)
        file.write(b"}\n}")
