import json
from pathlib import Path
from copy import deepcopy

from kernel_tuner.cache.json import CacheFileJSON, CacheLineJSON

import pytest
import jsonschema
import kernel_tuner

KERNEL_TUNER_PATH = Path(kernel_tuner.__file__).parent
SCHEMA_PATH = KERNEL_TUNER_PATH / "schema/cache/1.0.0/schema.json"

TEST_PATH = Path(__file__).parent
TEST_CACHE_PATH = TEST_PATH / "test_cache_files"
SMALL_CACHE_PATH = TEST_CACHE_PATH / "small_cache.json"
LARGE_CACHE_PATH = TEST_CACHE_PATH / "large_cache.json"

with open(SMALL_CACHE_PATH) as f:
    SMALL_CACHE = json.load(f)
with open(LARGE_CACHE_PATH) as f:
    LARGE_CACHE = json.load(f)


@pytest.fixture(scope="session")
def validator() -> jsonschema.Draft202012Validator:
    with open(SCHEMA_PATH) as f:
        schema_json = json.load(f)
        jsonschema.Draft202012Validator.check_schema(schema_json)
        return jsonschema.Draft202012Validator(schema_json)


@pytest.fixture()
def cache() -> CacheFileJSON:
    return deepcopy(SMALL_CACHE)


@pytest.fixture()
def large(cache):
    cache.clear()
    cache.update(deepcopy(LARGE_CACHE))


@pytest.fixture(params=[0, -1], ids=["first_cache_line", "last_cache_line"])
def cache_line(cache, request) -> CacheLineJSON:
    '''
    Important: when using fixture large, don't put `cache_line` before fixture
    `large` in the parameter list of a test or fixture'''
    index = request.param
    cache_keys = list(cache['cache'].keys())
    cache_key = cache_keys[index]
    return cache['cache'][cache_key]


@pytest.fixture()
def is_valid(validator, cache):
    yield  # let the test apply some modifications
    validator.validate(cache)  # assert the cache is valid


@pytest.fixture()
def is_invalid(validator, cache):
    yield  # let the test apply some modifications
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validator.validate(cache)  # assert the cache is invalid


class TestCacheFileSchema:
    def test_small_cache_is_valid(self, cache, is_valid): pass

    def test_large_cache_is_valid(self, large, cache, is_valid): pass

    def test_schema_version_is_valid(self, cache, is_invalid):
        cache["schema_version"] = "0.1.0"

    @pytest.mark.parametrize("key", [
        "schema_version",
        "device_name",
        "kernel_name",
        "problem_size",
        "tune_params_keys",
        "tune_params",
        "objective",
        "cache",
    ])
    def test_required_keys__in_root(self, cache, is_invalid, key):
        del cache[key]

    @pytest.mark.parametrize("key,value", [
        ("schema_version", 1234),
        ("device_name", 2312),
        ("kernel_name", True),
        ("problem_size", 2.5),
        ("tune_params_keys", {}),
        ("tune_params", []),
        ("objective", 4.5),
        ("cache", []),
    ])
    def test_property_types__in_root(self, cache, is_invalid, key, value):
        cache[key] = value

    @pytest.mark.parametrize("key", [
        "time",
        "compile_time",
        "verification_time",
        "benchmark_time",
        "strategy_time",
        "framework_time",
        "timestamp",
    ])
    def test_required_keys__in_cache_data(self, cache_line, is_invalid, key):
        del cache_line[key]

    @pytest.mark.parametrize("key,value", [
        ("time", True),
        ("times", {}),
        ("times", ["x"]),
        ("compile_time", None),
        ("verification_time", 2.5),
        ("benchmark_time", "Invalid"),
        ("GFLOP/s", False),
        ("strategy_time", 100.001),
        ("framework_time", "15"),
        ("timestamp", 42),
    ])
    def test_property_types__in_cache_data(self, cache_line, is_invalid,
                                           key, value):
        cache_line[key] = value
