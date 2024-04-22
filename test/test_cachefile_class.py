import pytest
import json

from kernel_tuner.cache.cache import CacheHeader, Cache


class TestCache:
    @pytest.fixture
    def cache_path(self, tmp_path):
        return tmp_path / "cache.json"

    @pytest.fixture
    def header(self):
        return CacheHeader(
            device_name="Test device",
            kernel_name="Test kernel",
            problem_size=[256, 256],
            tune_params_keys=["a", "b", "c"],
            tune_params={"a": [1, 2], "b": [3, 4], "c": [5, 6]},
            objective="maximize",
        )

    @pytest.fixture
    def assert_create__raises_ValueError(cache_path, header):
        yield
        with pytest.raises(ValueError):
            Cache.create(cache_path, **vars(header))

    def test_create(self, cache_path, header):
        Cache.create(cache_path, **vars(header))
        with open(cache_path) as file:
            data = json.load(file)
        assert "version" in data
        assert {**data, "version": "*"} == {"version": "*", **vars(header), "cache": {}}

    def test_create__invalid_device_name(self, header, assert_create__raises_ValueError):
        header.device_name = 3

    def test_create__invalid_kernel_name(self, header, assert_create__raises_ValueError):
        header.kernel_name = True

    def test_create__invalid_tune_params__types(self, header, assert_create__raises_ValueError):
        header.tune_params_keys = [1, True, False]
        header.tune_params = {1: [1], True: [True], False: [False]}

    def test_create__invalid_tune_params__mismatch(self, header, assert_create__raises_ValueError):
        header.tune_params_keys = ["a", "b"]

    def test_create__invalid_objective(self, header, assert_create__raises_ValueError):
        header.objective = 3.141592
