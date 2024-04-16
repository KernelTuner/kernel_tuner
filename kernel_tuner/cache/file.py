"""Provides utilities for reading and writing cachefiles.

Regarding being opened or closed, there are three states cache files can be in:
1. Closed: the cache file contains valid cache json
2. Properly open: the cache file contains a cache json object of which the root and its last `cache` property are not
   closed, meaning the last two braces (`}}`) are missing. In addition, these braces have been replaced with a comma.
3. Improperly open: the cache file contains a cache json object of which the root and its last `cache` property are
   not closed, meaning the last two braces (`}}`) are missing. No comma is present.
"""

from __future__ import annotations

import json
import re
import os
import io
from os import PathLike
from typing import Callable

from kernel_tuner.cache.json_encoder import CacheJSONEncoder


OPTIONAL_COMMA_END_REGEX = re.compile(r",?$")
CLOSING_BRACES_REGEX = re.compile(r"\s*}\s*}\s*$")


def read_cache(filename: PathLike, *, ensure_open=False, ensure_closed=False):
    """Reads a cache file and returns its content as a dictionary.

    It might seem unreasonable to put cache opening and closing functionality here. However, since the only way to
    determine whether the cache is open or closed is to parse it, this is the only place to reliably ensure that the
    cache is open or closed. This functionality can be invoked via the options in the keyword arguments.

    Parameters:
        filename (PathLike): The path to the cache file.
        ensure_open (bool): If true, the cache will be ensured to be properly opened
        ensure_closed (bool): If true, the cache will be ensured to be closed

    Returns:
        dict: The content of the cache file.
    """
    if ensure_open and ensure_closed:
        raise ValueError("Cache files cannot be simultaneously opened and closed")

    data = None
    is_open = False
    is_properly_open = False

    # Try to load the cache as both open and closed.
    # Store whether the cache was open and properly opened
    with open(filename, "r") as file:
        try:  # Try load the cache as closed
            data = json.load(file)
        except json.JSONDecodeError:
            # The cache is not closed, so we will try to read it as being open
            file.seek(0)
            try:  # Try load the cache as being opened
                text = file.read()
                is_open = True
                is_properly_open = text.endswith(",")
                text = OPTIONAL_COMMA_END_REGEX.sub("}}", text, 1)
                data = json.loads(text)
            except json.JSONDecodeError:
                raise ValueError(f"Cache file {filename} is corrupted")

    # Now ensure the cache is open or closed
    if ensure_open:
        if not is_open:
            open_closed_cache(filename)
        elif not is_properly_open:
            fix_improperly_opened_cache(filename)
    elif ensure_closed and is_open:
        close_opened_cache(filename)
    return data


def write_cache(cache_json: dict, filename: PathLike, *, keep_open=False):
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


def append_cache_line(key: str, cache_line: dict, cache_filename: PathLike):
    """Appends a cache line to an open cache file."""
    text = json.dumps({key: cache_line})
    text = "\n" + text.strip()[1:-1] + ","
    with open(cache_filename, "a") as file:
        file.write(text)


def fix_improperly_opened_cache(filename: PathLike):
    """Properly opens a improperly opened cache file."""
    with open(filename, "rb+") as file:
        # Seek the last `}` ("cache" property closing brace)
        file.seek(0, os.SEEK_END)
        _seek_back_while(lambda ch: ch != b"}", file)

        # Skip the closing brace of the cache line suffix the cache file with a comma
        file.seek(1, os.SEEK_CUR)
        file.write(b",")
        file.truncate()


def close_opened_cache(filename: PathLike):
    """Closes an open cache file."""
    with open(filename, "rb+") as file:
        # Seek the last `}` ("cache" property closing brace)
        file.seek(0, os.SEEK_END)
        _seek_back_while(lambda ch: ch != b"}", file)

        # Skip the closing brace of the cache line and close the cache file
        file.seek(1, os.SEEK_CUR)
        file.write(b"}\n}")
        file.truncate()


def open_closed_cache(filename: PathLike):
    """Opens a closed cache file."""
    with open(filename, "rb+") as file:
        # Seek the last `}` (root closing brace)
        file.seek(0, os.SEEK_END)
        _seek_back_while(lambda ch: ch != b"}", file)

        # Seek the second last `}` ("cache" property closing brace)
        file.seek(-1, os.SEEK_CUR)
        _seek_back_while(lambda ch: ch != b"}", file)

        # Replace it with a comma and truncate the file.
        file.write(b",")
        file.truncate()


def open_cache(filename: PathLike):
    """Closes any cache file. Needs to read the whole file in order to guarantee this."""
    read_cache(filename, ensure_open=True)


def close_cache(filename: PathLike):
    """Opens any cache file. Needs to read the whole file in order to guarantee this."""
    read_cache(filename, ensure_closed=True)


def _seek_back_while(predicate: Callable[[bytes], bool], buf: io.BufferedRandom):
    while predicate(_read_back(buf)):
        pass


def _read_back(buf: io.BufferedRandom, size=1):
    if buf.tell() < size:
        raise EOFError("Cannot read backwards any further")
    buf.seek(-size, os.SEEK_CUR)
    ch = buf.peek(size)[:size]
    return ch
