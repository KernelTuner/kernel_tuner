import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

import kernel_tuner
from kernel_tuner.cache.json import CacheFileJSON, CacheLineJSON
from kernel_tuner.cache.cache import Cache
import jsonschema
import pytest
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


@pytest.fixture()
def cache() -> CacheFileJSON:
    return deepcopy(SMALL_CACHE)


@pytest.fixture()
def large(cache):
    cache.clear()
    cache.update(deepcopy(LARGE_CACHE))


@pytest.fixture(params=[0, -1], ids=["first_cache_line", "last_cache_line"])
def cache_line(cache, request) -> CacheLineJSON:
    """Fixture for obtaining a reference to an arbitrary cache line.

    Note: When using fixture large, don't put `cache_line` before
    fixture `large` in the parameter list of a test or fixture
    """
    index = request.param
    cache_keys = list(cache["cache"].keys())
    cache_key = cache_keys[index]
    return cache["cache"][cache_key]


@pytest.fixture()
def is_valid(cache):
    yield  # let the test apply some modifications

    Cache.validate_json(cache)  # assert the cache is valid


@pytest.fixture()
def is_invalid(cache):
    yield  # let the test apply some modifications
    with pytest.raises(jsonschema.exceptions.ValidationError):
        Cache.validate_json(cache)  # assert the cache is invalid


@pytest.fixture()
def invalid_timestamp_cache() -> dict:
    return {
        "schema_version": "1.0.0",
        "device_name": "Testing",
        "kernel_name": "convolution_kernel",
        "problem_size": [4096, 4096],
        "tune_params_keys": ["block_size_x"],
        "tune_params": {"block_size_x": [16, 32]},
        "objective": "time",
        "cache": {
            "16,1,1,1,0,0,0,1,15,15": {
                "block_size_x": 16,
                "time": 3.875,
                "compile_time": 918.6,
                "verification_time": 0,
                "benchmark_time": 126.3,
                "strategy_time": 0,
                "framework_time": 105.2,
                "timestamp": "2023",
            }
        },
    }


@pytest.fixture()
def invalid_schemaversion_cache() -> dict:
    return {
        "schema_version": "20.9.8",
        "device_name": "Testing",
        "kernel_name": "convolution_kernel",
        "problem_size": [4096, 4096],
        "tune_params_keys": ["block_size_x"],
        "tune_params": {"block_size_x": [16, 32]},
        "objective": "time",
        "cache": {
            "16,1,1,1,0,0,0,1,15,15": {
                "block_size_x": 16,
                "time": 3.875,
                "compile_time": 918.6,
                "verification_time": 0,
                "benchmark_time": 126.3,
                "strategy_time": 0,
                "framework_time": 105.2,
                "timestamp": "2023-12-22T10:33:29.006875+00:00",
            }
        },
    }


class TestCacheFileSchema:
    def test_small_cache_is_valid(self, cache, is_valid):
        pass

    def test_large_cache_is_valid(self, large, cache, is_valid):
        pass

    def test_schema_version_is_valid(self, cache, is_invalid):
        cache["schema_version"] = "0.1.0"

    @pytest.mark.parametrize(
        "key",
        [
            "schema_version",
            "device_name",
            "kernel_name",
            "problem_size",
            "tune_params_keys",
            "tune_params",
            "objective",
            "cache",
        ],
    )
    def test_required_keys__in_root(self, cache, is_invalid, key):
        del cache[key]

    @pytest.mark.parametrize(
        "key,value",
        [
            ("schema_version", 1234),
            ("device_name", 2312),
            ("kernel_name", True),
            ("problem_size", 2.5),
            ("tune_params_keys", {}),
            ("tune_params", []),
            ("objective", 4.5),
            ("cache", []),
        ],
    )
    def test_property_types_invalid__in_root(self, cache, is_invalid, key, value):
        cache[key] = value

    @pytest.mark.parametrize(
        "key,value",
        [
            ("schema_version", "1.0.0"),
            ("device_name", "test_device"),
            ("kernel_name", "test_kernel"),
            ("problem_size", [100, 100]),
            ("tune_params_keys", ["block_size_x"]),
            ("tune_params", { "block_size_x": [128, 256, 512, 1024] }),
            ("objective", "time"),
            ("cache", {})
        ],
    )
    def test_property_types_valid__in_root(self, cache, is_valid, key, value):
        cache[key] = value

    @pytest.mark.parametrize(
        "key,value",
        {
            ("my_very_uncommon_key", 1),
            ("fewajfewaijfewa", 2),
        },
    )
    def test_additional_props_allowed__in_root(self, cache, is_valid, key, value):
        cache[key] = value

    @pytest.mark.parametrize(
        "key",
        [
            "time",
            "compile_time",
            "verification_time",
            "benchmark_time",
            "strategy_time",
            "framework_time",
            "timestamp",
        ],
    )
    def test_required_keys__in_cache_line(self, cache_line, is_invalid, key):
        del cache_line[key]

    @pytest.mark.parametrize(
        "key,value",
        [
            ("time", True),
            ("times", {}),
            ("times", ["x"]),
            ("compile_time", None),
            ("verification_time", True),
            ("benchmark_time", "Invalid"),
            ("strategy_time", "123"),
            ("framework_time", "15"),
            ("timestamp", 42),
        ],
    )
    def test_property_types__in_cache_line(self, cache_line, is_invalid, key, value):
        cache_line[key] = value

    @pytest.mark.parametrize(
        "key,value",
        [
            ("anyParameter", 45),
            ("xyz", [2, 3, 4]),
        ],
    )
    def test_additional_props_allowed__in_cache_line(self, cache_line, is_valid, key, value):
        cache_line[key] = value

    def test_invalid_timestamp_cache(self, cache, invalid_timestamp_cache, is_invalid):
        cache.update(invalid_timestamp_cache)

    def test_invalid_schema_version(self, cache, invalid_schemaversion_cache, is_invalid):
        cache.update(invalid_schemaversion_cache)
