"""Provides an instance of JSONEncoder in order to encode JSON to our own readable format.

Extends the default Python JSONEncoder to format Kernel Tuner caches.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from typing import Tuple

import numpy as np

import kernel_tuner.util as util

INFINITY = float("inf")


class CacheEncoder(json.JSONEncoder):
    """JSON encoder for Kernel Tuner cache lines.

    Supports the following objects and types by default:

    +-------------------+---------------+
    | Python            | JSON          |
    +===================+===============+
    | util.ErrorConfig  | str           |
    +-------------------+---------------+
    | np.integer        | int           |
    +-------------------+---------------+
    | np.floating       | float         |
    +-------------------+---------------+
    | np.ndarray        | list          |
    +-------------------+---------------+
    | dict              | object        |
    +-------------------+---------------+
    | list, tuple       | array         |
    +-------------------+---------------+
    | str               | string        |
    +-------------------+---------------+
    | int, float        | number        |
    +-------------------+---------------+
    | True              | true          |
    +-------------------+---------------+
    | False             | false         |
    +-------------------+---------------+
    | None              | null          |
    +-------------------+---------------+

    To extend this to recognize other objects, subclass and implement a
    ``.default()`` method with another method that returns a serializable
    object for ``o`` if possible, otherwise it should call the superclass
    implementation (to raise ``TypeError``).

    """

    def __init__(self, *, indent=None, separators=None, **kwargs):
        """Constructor for CacheJSONEncoder, with sensible defaults."""
        if indent is None:
            separators = (", ", ": ")

        super().__init__(indent=indent, separators=separators, **kwargs)

    def default(self, o):
        """Converts non-jsonifiable objects to jsonifiable objects."""

        if isinstance(o, util.ErrorConfig):
            return str(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        super().default(o)

    def iterencode(self, obj, *args, **kwargs):
        """encode an iterator, ensuring 'cache' is the last entry for encoded dicts"""

        # ensure key 'cache' is last in any encoded dictionary
        if isinstance(obj, dict):
            obj = OrderedDict(obj)
            if "cache" in obj:
                obj.move_to_end("cache")

        yield from super().iterencode(obj, *args, **kwargs)
