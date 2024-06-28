import json
import numpy as np

from kernel_tuner.cache.cache import CacheEncoder
from kernel_tuner.util import ErrorConfig, InvalidConfig, CompilationFailedConfig, RuntimeFailedConfig


def dumps(data):
    return json.dumps(data, cls=CacheEncoder, indent="")


class TestCacheJSONEncoder:

    def test_encode_cache_last(self):
        test_obj = {"b": 2, "cache": {}, "d": 4}
        dump_str = dumps(test_obj)
        expected = '{\n"b": 2,\n"d": 4,\n"cache": {}\n}'
        print("received:")
        print(dump_str)
        print("expected:")
        print(expected)
        assert dump_str == expected

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
