"""Provides an instance of JSONEncoder in order to encode JSON to our own readable format."""

from __future__ import annotations

import json
import numpy as np

from kernel_tuner.util import ErrorConfig


# Custom encoder class that inherits from json.JSONEncoder
class CacheJSONEncoder(json.JSONEncoder):
    """Subclass of JSONEncoder used in Kernel Tuner.

    This encoder ensures that each cache entry gets its own separate line, in order to make cache files more readable
    to the human eye.
    """

    def __init__(self, *args, **kwargs):
        """Initializes a new CacheJSONEncoder object."""
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def default(self, o):
        """Method for converting non-standard python objects to JSON."""
        print(type(o))
        if isinstance(o, ErrorConfig):
            return str(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        super().default(o)

    def encode(self, o, force=False):
        """Encodes any JSON object.

        Overwrites the encode function of the JSONEncoder by a custom one. It acts as the usual
        encoder unless o is of type list, tuple or dict. We then call _encode_list and _encode_object
        to handle our custom encoding.

        Parameters:
            o (any): The object we are encoding.
            force (bool): Whether we should force the json string to be on a single line.

        Returns:
            str: The json we have encoded as string.
        """
        if isinstance(o, (list, tuple)):
            return self._encode_list(o, force)
        if isinstance(o, dict):
            return self._encode_object(o, force)
        return json.dumps(
            o,
            skipkeys=self.skipkeys,
            ensure_ascii=self.ensure_ascii,
            check_circular=self.check_circular,
            allow_nan=self.allow_nan,
            sort_keys=self.sort_keys,
            indent=self.indent,
            separators=(self.item_separator, self.key_separator),
            default=self.default if hasattr(self, "default") else None,
        )

    def _encode_list(self, o, force=False):
        """Encodes a list or tuple like normal unless we force it to print on a single line.

        Parameters:
            o (list or tuple): The object we are encoding.
            force (bool): Whether we should force the json string to be on a single line.

        Returns:
            str: The json we have encoded as string.
        """
        if force:
            return "[" + ", ".join(self.encode(el) for el in o) + "]"
        if not o:
            return "[]"
        self.indentation_level += 1
        output = [self._indent_str + self.encode(el) for el in o]
        self.indentation_level -= 1
        return "[\n" + ",\n".join(output) + "\n" + self._indent_str + "]"

    # For encode object we go through what the object contains and write it to the file.
    def _encode_object(self, o, force=False):
        """Encodes a dict like normal unless we force it to print on a single line.

        Parameters:
            o (dict): The object we are encoding.
            force (bool): Whether we should force the json string to be on a single line.

        Returns:
            str: The json we have encoded as string.
        """
        if not o:
            return "{}"

        if force:
            return "{ " + ", ".join(f"{json.dumps(k)}: {self.encode(el)}" for k, el in o.items()) + "}"

        has_cache = False
        cache_values = ""

        # If we detect the key entry "cache", we remove this key and write it using our custom
        # function to convert it to a single line. This is done later in the code where we check for the boolean
        # "has_cache". Otherwise, we continue like normal (so the normal expected format gets used)
        if "cache" in o.keys():
            cache_values = o.get("cache")
            o.pop("cache", None)
            has_cache = True

        o = {str(k) if k is not None else "null": v for k, v in o.items()}

        if self.sort_keys:
            o = dict(sorted(o.items(), key=lambda x: x[0]))

        self.indentation_level += 1
        output = [f"{self._indent_str}{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]

        # We call _encode_cache if we detect the key "cache"
        if has_cache:
            output.append(self._encode_cache(cache_values))

        self.indentation_level -= 1
        return "{\n" + ",\n".join(output) + "\n" + self._indent_str + "}"

    def _encode_cache(self, o: dict) -> str:
        """Encodes the value of the cache key in the cache file.

        Encodes a dict in the way we want for our cache files where every item in the key "cache"
        is written line by line improving readability.

        Parameters:
            o (dict): The object we are encoding.

        Returns:
            str: The json we have encoded as string.
        """
        cache_values = self._indent_str + '"cache": {\n'
        self.indentation_level += 1
        for key, item in o.items():
            cache_values += self._indent_str + '"' + key + '": {'
            cache_values += ", ".join(f"{json.dumps(k)}: {self.encode(el, True)}" for k, el in item.items())
            cache_values += "},\n"
        self.indentation_level -= 1
        cache_values = cache_values.strip(",\n")
        cache_values += "}"
        return cache_values

    @property
    def _indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " " * (self.indentation_level * self.indent)
        elif isinstance(self.indent, str):
            return self.indentation_level * self.indent
        else:
            raise ValueError(f"indent must either be of type int or str (is: {type(self.indent)})")
