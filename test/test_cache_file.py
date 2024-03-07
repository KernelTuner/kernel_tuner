import json
from pathlib import Path
from copy import deepcopy

import sys
print(sys.path)  # noqa

from kernel_tuner.cache.json import CacheFileJSON

import pytest
import jsonschema


PROJECT_DIR = Path(__file__).parents[1]
SCHEMA_PATH = PROJECT_DIR / "kernel_tuner/schema/cache/1.0.0/schema.json"
TEST_CACHE_PATH = PROJECT_DIR / "test/test_cache_files"
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
def large(cache: CacheFileJSON):
    cache.clear()
    cache.update(deepcopy(LARGE_CACHE))


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

    @pytest.mark.parametrize("key", [
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

    @pytest.mark.parametrize("index", [0, 1])
    @pytest.mark.parametrize("key", [
        "time",
        "compile_time",
        "verification_time",
        "benchmark_time",
        "strategy_time",
        "framework_time",
        "timestamp",
    ])
    def test_required_keys__in_cache_data(self, cache, is_invalid, index, key):
        cache_keys = list(cache['cache'].keys())
        cache_key = cache_keys[index]
        del cache['cache'][cache_key][key]

    @pytest.mark.parametrize("index", [0, 1])
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
    def test_property_types__in_cache_data(self, cache, is_invalid,
                                           index, key, value):
        cache_keys = list(cache['cache'].keys())
        cache_key = cache_keys[index]
        cache['cache'][cache_key][key] = value
