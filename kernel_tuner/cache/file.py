"""Provides utilities for reading and writing cachefiles."""

from __future__ import annotations

import io
import json
import os
from os import PathLike
from typing import Callable, Optional



class CachedLinePosition:
    """Position of a cache line in a cache file."""

    is_initialized: bool = False
    is_first_line: bool = False
    file_position: int = 0


def append_cache_line(
    cache_line: str, filename: PathLike, position: Optional[CachedLinePosition] = None
) -> CachedLinePosition:
    """Appends a cache line to an open cache file.

    If ``position`` is unset, it will assume the "cache" property comes last in the root object when determining the
    position to insert the cache line at.

    Returns the position of the next cache line.
    """
    p = position or CachedLinePosition()
    if not p.is_initialized:
        _unsafe_get_next_cache_line_position(filename, p)
    return _append_cache_line_at(cache_line, filename, p)


def _append_cache_line_at(
    cache_line: str, filename: PathLike, position: CachedLinePosition
) -> CachedLinePosition:
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
        text += cache_line
        file.write(text)

        # Update the position
        next_pos = CachedLinePosition()
        next_pos.is_initialized = True
        next_pos.is_first_line = False
        next_pos.file_position = file.tell()

        # Close off the cache
        file.write(after_text)
        file.truncate()

    return next_pos


# This function only works when "cache" property is stored last
def _unsafe_get_next_cache_line_position(filename: PathLike, state: CachedLinePosition):
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
