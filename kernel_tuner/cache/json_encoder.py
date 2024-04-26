"""Provides an instance of JSONEncoder in order to encode JSON to our own readable format.

Modified from https://github.com/python/cpython/blob/3.8/Lib/json/encoder.py in order to format cache in Kernel Tuner
cache format.
"""

from __future__ import annotations

import json
from typing import Tuple

import numpy as np

import kernel_tuner.util as util


INFINITY = float("inf")


class CacheEncoder(json.JSONEncoder):
    """JSON encoder for Kernel Tuner cache.

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

    item_separator = ", "
    key_separator = ": "

    def __init__(
        self,
        *,
        skipkeys=False,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=True,
        sort_keys=False,
        indent=None,
        separators=None,
        default=None,
    ):
        """Constructor for CacheJSONEncoder, with sensible defaults."""
        super().__init__(
            skipkeys=skipkeys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
            default=default,
        )

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

    def encode(self, o):
        """Return a JSON string representation of a Python data structure.

        >>> from json.encoder import JSONEncoder
        >>> JSONEncoder().encode({"foo": ["bar", "baz"]})
        '{"foo": ["bar", "baz"]}'

        """
        # This is for extremely simple cases and benchmarks.
        if isinstance(o, str):
            ### MODIFICATION: encode strings using super().encode in order to access native c interface for converting
            ###   strings to json strings in cases this c interface exists
            super().encode(o)
            ### END MODIFICATION
        # This doesn't pass the iterator directly to ''.join() because the
        # exceptions aren't as detailed.  The list call should be roughly
        # equivalent to the PySequence_Fast that ''.join() would do.
        chunks = self.iterencode(o, _one_shot=True)
        if not isinstance(chunks, (list, tuple)):
            chunks = list(chunks)
        return "".join(chunks)

    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string representation as available.

        For example::

            for chunk in JSONEncoder().iterencode(bigobject):
                mysocket.write(chunk)

        """
        if self.check_circular:
            markers = {}
        else:
            markers = None
        ### MODIFICATION: encode strings using super().encode in order to access native c interface for converting
        ###   strings to json strings in cases this c interface exists
        _encoder = super().encode
        ### END MODIFICATION

        def floatstr(o, allow_nan=self.allow_nan, _repr=float.__repr__, _inf=INFINITY, _neginf=-INFINITY):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            if o != o:
                text = "NaN"
            elif o == _inf:
                text = "Infinity"
            elif o == _neginf:
                text = "-Infinity"
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError("Out of range float values are not JSON compliant: " + repr(o))

            return text

        _iterencode = _make_iterencode(
            markers,
            self.default,
            _encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return _iterencode(o, ())


def _make_iterencode(
    markers,
    _default,
    _encoder,
    _indent,
    _floatstr,
    _key_separator,
    _item_separator,
    _sort_keys,
    _skipkeys,
    _one_shot,
    ## HACK: hand-optimized bytecode; turn globals into locals
    ValueError=ValueError,
    dict=dict,
    float=float,
    id=id,
    int=int,
    isinstance=isinstance,
    list=list,
    str=str,
    tuple=tuple,
    _intstr=int.__repr__,
):
    if _indent is not None and not isinstance(_indent, str):
        _indent = " " * _indent

    def _get_indent_level(_current_path: Tuple):
        if _indent is None:
            return None
        if len(_current_path) > 0 and _current_path[0] == "cache":
            if len(_current_path) > 1:
                return None
            return 0
        return len(_current_path)

    ### SINGLE LINE MODIFICATION: Use the current path to `o` as a parameter instead of the indentation level
    def _iterencode_list(lst, _current_path: Tuple):
        if not lst:
            yield "[]"
            return
        if markers is not None:
            markerid = id(lst)
            if markerid in markers:
                raise ValueError("Circular reference detected")
            markers[markerid] = lst
        buf = "["
        ### MODIFICATION: Let _current_indent_level vary depending on the path
        _current_indent_level = _get_indent_level(_current_path)
        if _current_indent_level is not None:
            newline_indent = "\n" + _indent * _current_indent_level
            separator = _item_separator + newline_indent
            buf += newline_indent
        ### END MODIFICATION
        else:
            newline_indent = None
            ### MODIFICATION: Use spaces after the comma in compact representation
            separator = ", "
            ### END MODIFICATION
        first = True
        for index, value in enumerate(lst):
            if first:
                first = False
            else:
                buf = separator
            if isinstance(value, str):
                yield buf + _encoder(value)
            elif value is None:
                yield buf + "null"
            elif value is True:
                yield buf + "true"
            elif value is False:
                yield buf + "false"
            elif isinstance(value, int):
                # Subclasses of int/float may override __repr__, but we still
                # want to encode them as integers/floats in JSON. One example
                # within the standard library is IntEnum.
                yield buf + _intstr(value)
            elif isinstance(value, float):
                # see comment above for int
                yield buf + _floatstr(value)
            else:
                yield buf
                ### MODIFICATION: pass item_path instead of _current_indent_level
                item_path = _current_path + (index,)
                if isinstance(value, (list, tuple)):
                    chunks = _iterencode_list(value, item_path)
                elif isinstance(value, dict):
                    chunks = _iterencode_dict(value, item_path)
                else:
                    chunks = _iterencode(value, item_path)
                ### END MODIFICATION
                yield from chunks
        if newline_indent is not None:
            ### SINGLE LINE MODIFICATION: determine the previous indent level from the path
            _current_indent_level = _get_indent_level(_current_path[:-1])
            yield "\n" + _indent * _current_indent_level
        yield "]"
        if markers is not None:
            del markers[markerid]

    ### SINGLE LINE MODIFICATION: Use the current path to `o` as a parameter instead of the indentation level
    def _iterencode_dict(dct, _current_path: Tuple):
        if not dct:
            yield "{}"
            return
        if markers is not None:
            markerid = id(dct)
            if markerid in markers:
                raise ValueError("Circular reference detected")
            markers[markerid] = dct
        yield "{"
        ### MODIFICATION: Let _current_indent_level vary depending on the path
        _current_indent_level = _get_indent_level(_current_path)
        if _current_indent_level is not None:
            newline_indent = "\n" + _indent * _current_indent_level
            item_separator = _item_separator + newline_indent
            yield newline_indent
        ### END MODIFICATION
        else:
            newline_indent = None
            ### MODIFICATION: Use spaces after the comma in compact representation
            item_separator = ", "
            ### END MODIFICATION
        first = True
        if _sort_keys:
            items = sorted(dct.items())
        ### MODIFICATION: Order the cache attribute in the root object to be last
        elif len(_current_path) == 0:
            items = list((key, value) for key, value in dct.items() if key != "cache")
            if "cache" in dct:
                items.append(("cache", dct["cache"]))
        ### END MODIFICATION:
        else:
            items = dct.items()
        for key, value in items:
            if isinstance(key, str):
                pass
            # JavaScript is weakly typed for these, so it makes sense to
            # also allow them.  Many encoders seem to do something like this.
            elif isinstance(key, float):
                # see comment for int/float in _make_iterencode
                key = _floatstr(key)
            elif key is True:
                key = "true"
            elif key is False:
                key = "false"
            elif key is None:
                key = "null"
            elif isinstance(key, int):
                # see comment for int/float in _make_iterencode
                key = _intstr(key)
            elif _skipkeys:
                continue
            else:
                raise TypeError(f"keys must be str, int, float, bool or None, " f"not {key.__class__.__name__}")
            if first:
                first = False
            else:
                yield item_separator
            yield _encoder(key)
            yield _key_separator
            if isinstance(value, str):
                yield _encoder(value)
            elif value is None:
                yield "null"
            elif value is True:
                yield "true"
            elif value is False:
                yield "false"
            elif isinstance(value, int):
                # see comment for int/float in _make_iterencode
                yield _intstr(value)
            elif isinstance(value, float):
                # see comment for int/float in _make_iterencode
                yield _floatstr(value)
            else:
                ### MODIFICATION: pass item_path instead of _current_indent_level
                item_path = _current_path + (key,)
                if isinstance(value, (list, tuple)):
                    chunks = _iterencode_list(value, item_path)
                elif isinstance(value, dict):
                    chunks = _iterencode_dict(value, item_path)
                else:
                    chunks = _iterencode(value, item_path)
                yield from chunks
                ### END MODIFICATION
        if newline_indent is not None:
            ### SINGLE LINE MODIFICATION: determine the previous indent level from the path
            _current_indent_level = _get_indent_level(_current_path[:-1])
            yield "\n" + _indent * _current_indent_level
        yield "}"
        if markers is not None:
            del markers[markerid]

    ### SINGLE LINE MODIFICATION: Use the current path to `o` as a parameter instead of the indentation level
    def _iterencode(o, _current_path: Tuple):
        if isinstance(o, str):
            yield _encoder(o)
        elif o is None:
            yield "null"
        elif o is True:
            yield "true"
        elif o is False:
            yield "false"
        elif isinstance(o, int):
            # see comment for int/float in _make_iterencode
            yield _intstr(o)
        elif isinstance(o, float):
            # see comment for int/float in _make_iterencode
            yield _floatstr(o)
        elif isinstance(o, (list, tuple)):
            yield from _iterencode_list(o, _current_path)
        elif isinstance(o, dict):
            yield from _iterencode_dict(o, _current_path)
        else:
            if markers is not None:
                markerid = id(o)
                if markerid in markers:
                    raise ValueError("Circular reference detected")
                markers[markerid] = o
            o = _default(o)
            yield from _iterencode(o, _current_path)
            if markers is not None:
                del markers[markerid]

    return _iterencode


class CacheLineEncoder(json.JSONEncoder):
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

    def __init__(
        self,
        *,
        skipkeys=False,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=True,
        sort_keys=False,
        indent=None,
        separators=None,
        default=None,
    ):
        """Constructor for CacheJSONEncoder, with sensible defaults."""
        if indent is None:
            separators = (", ", ": ")

        super().__init__(
            skipkeys=skipkeys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
            default=default,
        )

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
