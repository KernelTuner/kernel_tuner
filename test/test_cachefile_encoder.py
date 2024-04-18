import json
import numpy as np

from kernel_tuner.cache.json_encoder import CacheEncoder
from kernel_tuner.util import ErrorConfig, InvalidConfig, CompilationFailedConfig, RuntimeFailedConfig


def dumps(data):
    return json.dumps(data, cls=CacheEncoder, indent=0)


class TestCacheJSONEncoder:
    def test_encode_null(self):
        assert dumps(None) == "null"

    def test_encode_bool(self):
        assert dumps(True) == "true"
        assert dumps(False) == "false"

    def test_encode_number(self):
        assert dumps(3) == "3"
        assert dumps(4.5) == "4.5"

    def test_encode_string(self):
        assert dumps("Hello world") == '"Hello world"'

    def test_encode_array(self):
        assert dumps([]) == "[]"
        assert dumps([1, 2, 3, 4]) == "[\n1,\n2,\n3,\n4\n]"

    def test_encode_dict(self):
        assert dumps({}) == "{}"
        assert dumps({"a": 1}) == '{\n"a": 1\n}'
        assert (
            dumps(
                {
                    "a": 1,
                    "b": 2,
                    "c": 3,
                }
            )
            == '{\n"a": 1,\n"b": 2,\n"c": 3\n}'
        )

    def test_encode_cache(self):
        assert dumps({"cache": {"1": {"a": 3}, "2": {"b": 4}}}) == '{\n"cache": {\n"1": {"a": 3},\n"2": {"b": 4}\n}\n}'

    def test_encode_cache_last(self):
        assert dumps({"b": 2, "cache": {}, "d": 4}) == '{\n"b": 2,\n"d": 4,\n"cache": {}\n}'

    def test_encode_error_config(self):
        assert dumps(ErrorConfig()) == '"ErrorConfig"'
        assert dumps(InvalidConfig()) == '"InvalidConfig"'
        assert dumps(CompilationFailedConfig()) == '"CompilationFailedConfig"'
        assert dumps(RuntimeFailedConfig()) == '"RuntimeFailedConfig"'

    def test_encode_np_int(self):
        assert dumps(np.int16(1234)) == "1234"

    def test_encode_np_float(self):
        assert dumps(np.float16(1.0)) == "1.0"

    def test_encode_np_array(self):
        assert dumps(np.array([3, 1, 4, 1, 5])) == "[\n3,\n1,\n4,\n1,\n5\n]"
