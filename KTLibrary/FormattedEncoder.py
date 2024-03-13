#!/usr/bin/env python3
from __future__ import annotations

import json


class FormattedEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o, force=False):
        """Encode JSON object *o* with respect to single line lists."""
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
        if force:
            return "[" + ", ".join(self.encode(el) for el in o) + "]"
        self.indentation_level += 1
        output = [self.indent_str + self.encode(el) for el in o]
        self.indentation_level -= 1
        return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"

    def _encode_cache(self, o: dict) -> str:
        cache_values = self.indent_str + '"cache": {\n'
        self.indentation_level += 1
        for key, item in o.items():
            cache_values += self.indent_str + "\"" + key + "\": {"
            cache_values += ", ".join(f"{json.dumps(k)}: {self.encode(el, True)}" for k, el in item.items())
            cache_values += "},\n"
        self.indentation_level -= 1
        cache_values = cache_values.strip(",\n")
        cache_values += "}"
        return cache_values

    def _encode_object(self, o, force=False):
        if not o:
            return "{}"

        if force:
            return (
                    "{ "
                    + ", ".join(f"{json.dumps(k)}: {self.encode(el)}" for k, el in o.items())
                    + "}"
            )

        has_cache = False
        cache_values = ""
        if "cache" in o.keys():
            cache_values = o.get("cache")
            o.pop("cache", None)
            has_cache = True

        o = {str(k) if k is not None else "null": v for k, v in o.items()}

        if self.sort_keys:
            o = dict(sorted(o.items(), key=lambda x: x[0]))

        self.indentation_level += 1
        output = [
            f"{self.indent_str}{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()
        ]
        if has_cache:
            output.append(self._encode_cache(cache_values))
        self.indentation_level -= 1

        return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"

    def iterencode(self, o, **kwargs):
        return self.encode(o)

    @property
    def indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " " * (self.indentation_level * self.indent)
        elif isinstance(self.indent, str):
            return self.indentation_level * self.indent
        else:
            raise ValueError(
                f"indent must either be of type int or str (is: {type(self.indent)})"
            )
