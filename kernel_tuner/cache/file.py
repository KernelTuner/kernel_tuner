"""Provides utilities for reading and writing cachefiles."""

from __future__ import annotations

import json
import os
import io
from os import PathLike
from typing import Callable, Optional

from kernel_tuner.cache.json_encoder import CacheEncoder, CacheLineEncoder


class InvalidCacheError(Exception):
    """Error raised when reading a cache file fails."""

    def __init__(self, filename: PathLike, message: str, error: Optional[Exception] = None):
        """Constructor for the InvalidCacheError class."""
        super().__init__(str(filename), message, error)
        self.filename = str(filename)
        self.message = message
        self.error = error


class CacheLinePosition:
    """Dataclass for reme."""

    is_initialized: bool = False
    is_first_line: bool = False
    file_position: int = 0


def read_cache(filename: PathLike):
    """Reads a cache file and returns its content as a dictionary.

    Parameters:
        filename (PathLike): The path to the cache file.

    Returns:
        dict: The content of the cache file.
    """
    # Try to load the cache as both open and closed.
    # Store whether the cache was open and properly opened
    with open(filename, "r") as file:
        try:  # Try load the cache as closed
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise InvalidCacheError(filename, "Cache file is not parsable", e)
    return data


def write_cache(cache_json: dict, filename: PathLike):
    r"""Writes a cache file with the given content.

    Parameters:
        cache_file (dict): The content to be written to the cache file.
        filename (PathLike): The path to write the cache file.
    """
    with open(filename, "w") as file:
        json.dump(cache_json, file, cls=CacheEncoder, indent=0)


def append_cache_line(
    key: str, cache_line: dict, filename: PathLike, position: Optional[CacheLinePosition] = None
) -> CacheLinePosition:
    """Appends a cache line to an open cache file.

    If ``position`` is unset, it will assume the "cache" property comes last in the root object when determining the
    position to insert the cache line at.

    Returns the position of the next cache line.
    """
    p = position or CacheLinePosition()
    if not p.is_initialized:
        _unsafe_get_next_cache_line_position(filename, p)
    return _append_cache_line_at(key, cache_line, filename, p)


def _append_cache_line_at(
    key: str, cache_line: dict, filename: PathLike, position: CacheLinePosition
) -> CacheLinePosition:
    with open(filename, "r+") as file:
        # Save cache closing braces properties coming after "cache" as text in suffix
        file.seek(position.file_position)
        after_text = file.read()

        # Append the cache line at the right position in the file
        file.seek(position.file_position)
        text = ""
        if not position.is_first_line:
            text += ","
        text += "\n"
        text += json.dumps({key: cache_line}, cls=CacheLineEncoder).strip()[1:-1]
        file.write(text)

        # Update the position
        next_pos = CacheLinePosition()
        next_pos.is_initialized = True
        next_pos.is_first_line = False
        next_pos.file_position = file.tell()

        # Close off the cache
        file.write(after_text)
        file.truncate()

    return next_pos


# FIXME: This function is unsafe: it only works when "cache" property is stored last
def _unsafe_get_next_cache_line_position(filename: PathLike, state: CacheLinePosition):
    with open(filename, "rb+") as file:
        # Seek the last `}` (root closing brace)
        file.seek(0, os.SEEK_END)
        _seek_back_while(lambda ch: ch != b"}", file)

        # Seek the second last `}` ("cache" property closing brace)
        file.seek(-1, os.SEEK_CUR)
        _seek_back_while(lambda ch: ch != b"}", file)

        # Test if the cache object is empty
        _seek_back_while(lambda ch: ch.isspace(), file)
        state.is_first_line = file.peek(1).startswith(b"{")
        file.seek(1, os.SEEK_CUR)

        # Find the current position in the cache file
        state.file_position = file.tell()

    # Mark that the state has been initialized
    state.is_initialized = True


def _seek_back_while(predicate: Callable[[bytes], bool], buf: io.BufferedRandom):
    while predicate(_read_back(buf)):
        pass


def _read_back(buf: io.BufferedRandom, size=1):
    buf.seek(-size, os.SEEK_CUR)
    ch = buf.peek(size)[:size]
    return ch
