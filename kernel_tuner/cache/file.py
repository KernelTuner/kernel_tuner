"""Provides utilities for reading and writing cachefiles."""

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
    """Closes a cache file by appending the last braces."""
    with open(filename, "rb+") as file:
        _seek_end_of_cache_lines(file, filename=filename)
        file.write(b"}\n}")
        file.truncate()


def open_cache_file(filename: PathLike):
    """Opens a cache file by replacing the last braces by a comma."""
    with open(filename, "rb+") as file:
        _seek_end_of_cache_lines(file, filename=filename)
        file.write(b",")
        file.truncate()


def _seek_end_of_cache_lines(file: io.BufferedRandom, *, filename=""):
    try:
        # Go to the last non-space, non-comma character in the file
        file.seek(0, os.SEEK_END)
        _seek_back_while(_char_is_space_or_comma, file)

        # This character should be a closing brace
        if not file.peek(1).startswith(b"}"):
            raise ValueError(f"Cache file {filename} is corrupted")

        # Go to the previous non-space character in the file
        # If this is a closing brace, this must be
        _seek_back_while(_char_is_space, file)
        if not file.peek(1).startswith(b"}"):
            # If this is not a brace, it should be the last character within a cache line object
            # Skip two bytes of which first being from the content of the last cache line.
            # The second byte is the closing tag of the last cache line
            file.seek(2, os.SEEK_CUR)
    except EOFError:
        raise ValueError(f"Cache file {filename} is corrupted")


def _seek_back_while(predicate: Callable[[bytes], bool], buf: io.BufferedRandom):
    while predicate(_read_back(buf)):
        pass


def _read_back(buf: io.BufferedRandom, size=1):
    if buf.tell() < size:
        raise EOFError()
    buf.seek(-size, os.SEEK_CUR)
    ch = buf.peek(size)[:size]
    return ch


def _char_is_space(ch: bytes) -> bool:
    return ch.isspace()


def _char_is_space_or_comma(ch: bytes) -> bool:
    return _char_is_space(ch) or ch == b","
