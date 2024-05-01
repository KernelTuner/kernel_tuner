import os
import pytest
import json
import semver
from datetime import datetime
from types import SimpleNamespace

import kernel_tuner.util as util
from kernel_tuner.cache.file import write_cache
from kernel_tuner.cache.cache import Cache


class TestCache:
    @pytest.fixture
    def cache_path(self, tmp_path):
        return tmp_path / "cache.json"

    @pytest.fixture
    def header(self):
        return SimpleNamespace(
            device_name="Test device",
            kernel_name="Test kernel",
            problem_size=[256, 256],
            tune_params_keys=["a", "b", "c"],
            tune_params={"a": [0, 1], "b": [0, 1], "c": [0, 1]},
            objective="maximize",
        )

    @pytest.fixture(scope="class")
    def now(self):
        return datetime.now()

    @pytest.fixture
    def cache_lines(self, now):
        LINE_JSON_TEMPLATE = {
            "time": 0.0,
            "times": [1.0],
            "compile_time": 2.0,
            "verification_time": 3,
            "benchmark_time": 4.0,
            "strategy_time": 6,
            "framework_time": 7.0,
            "timestamp": str(now),
        }

        def param_obj(a, b, c):
            return {"a": a, "b": b, "c": c}

        return {
            "0,0,0": {**param_obj(0, 0, 0), **LINE_JSON_TEMPLATE},
            "0,0,1": {**param_obj(0, 0, 1), **LINE_JSON_TEMPLATE},
            "0,1,0": {**param_obj(0, 1, 0), **LINE_JSON_TEMPLATE},
            "1,1,0": {**param_obj(1, 1, 0), **LINE_JSON_TEMPLATE},
        }

    @pytest.fixture
    def cache_json(self, header, cache_lines):
        return {"schema_version": "1.0.0", **vars(header), "cache": cache_lines}

    @pytest.fixture
    def assert_create__raises_ValueError(self, cache_path, header):
        yield
        with pytest.raises(ValueError):
            Cache.create(cache_path, **vars(header))

    def test_create(self, cache_path, header):
        Cache.create(cache_path, **vars(header))
        with open(cache_path) as file:
            data = json.load(file)
        assert "schema_version" in data
        assert {**data, "schema_version": "*"} == {"schema_version": "*", **vars(header), "cache": {}}

    def test_create__returns_object(self, cache_path, header):
        cache = Cache.create(cache_path, **vars(header))
        assert os.path.normpath(cache.filepath) == os.path.normpath(cache_path)

    def test_create__invalid_device_name(self, header, assert_create__raises_ValueError):
        header.device_name = 3

    def test_create__invalid_kernel_name(self, header, assert_create__raises_ValueError):
        header.kernel_name = True

    def test_create__invalid_tune_params__types(self, header, assert_create__raises_ValueError):
        header.tune_params_keys = [1, True, False]
        header.tune_params = {1: [1], True: [True], False: [False]}

    def test_create__invalid_tune_params__mismatch(self, header, assert_create__raises_ValueError):
        header.tune_params_keys = ["a", "b"]

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
            "times",
            "GFLOP/s",
        ],
    )
    def test_create__invalid_tune_params__reserved_keys(self, header, key, assert_create__raises_ValueError):
        header.tune_params_keys.append(key)
        header.tune_params[key] = [1, 2]

    def test_create__invalid_objective(self, header, assert_create__raises_ValueError):
        header.objective = 3.141592

    @pytest.fixture
    def cache_file(self, cache_path, cache_json):
        write_cache(cache_json, cache_path)
        yield cache_path

    @pytest.fixture
    def cache(self, cache_file):
        return Cache.read(cache_file)

    @pytest.fixture
    def cache_line_read(self, cache) -> Cache.Line:
        return cache.lines.get(a=0, b=0, c=0)

    def test_read(self, cache, header, cache_lines):
        pass

    def test_version(self, cache):
        assert isinstance(cache.version, semver.Version)

    def test_header(self, cache, header):
        assert cache.device_name == header.device_name
        assert cache.kernel_name == header.kernel_name
        assert list(cache.tune_params_keys) == header.tune_params_keys
        assert list(cache.tune_params["a"]) == header.tune_params["a"]
        assert list(cache.problem_size) == header.problem_size
        assert cache.objective == header.objective

    def test_lines_get(self, cache, cache_lines):
        assert cache.lines["0,0,0"] == cache_lines["0,0,0"]
        assert cache.lines.get(a=0, b=0, c=0) == cache_lines["0,0,0"]

    def test_lines_get__default(self, cache):
        assert cache.lines.get("1,0,0", "Hi!") == "Hi!"
        assert cache.lines.get(a=1, b=0, c=0, default="DEF") == "DEF"

    def test_lines_get__multiple(self, cache):
        assert len(cache.lines.get(b=1, c=1)) == 0
        assert len(cache.lines.get(b=2)) == 0
        assert len(cache.lines.get(a=0)) == 3

    def test_lines_get__no_KeyError(self, cache):
        cache.lines.get("gibberish")

    def test_lines_get__with_non_existing_param_key(self, cache):
        with pytest.raises(ValueError):
            cache.lines.get(d=0)

    def test_line_attributes(self, cache_line_read, now):
        assert cache_line_read.time == 0
        assert cache_line_read.times == [1]
        assert cache_line_read.compile_time == 2
        assert cache_line_read.verification_time == 3
        assert cache_line_read.benchmark_time == 4
        assert cache_line_read.GFLOP_per_s is None
        assert cache_line_read.strategy_time == 6
        assert cache_line_read.framework_time == 7
        assert cache_line_read.timestamp == now

    def test_line_dict(self, cache_line_read, cache_json, now):
        assert "GFLOP/s" not in cache_line_read
        assert dict(cache_line_read) == {
            "a": 0,
            "b": 0,
            "c": 0,
            "time": 0,
            "times": [1],
            "compile_time": 2,
            "verification_time": 3,
            "benchmark_time": 4,
            "strategy_time": 6,
            "framework_time": 7,
            "timestamp": str(now),
        }

    def test_line_dict__json_and_non_json(self, cache: Cache, now):
        cache.lines.append(
            time=util.RuntimeFailedConfig(),
            compile_time=1.0,
            verification_time=2,
            benchmark_time=3.0,
            strategy_time=4,
            framework_time=5.0,
            timestamp=now,
            times=[6.0],
            a=1,
            b=1,
            c=1,
        )
        line = cache.lines.get(a=1, b=1, c=1)

        assert line["time"] == util.RuntimeFailedConfig.__name__
        assert isinstance(line.time, util.RuntimeFailedConfig)

        assert line["timestamp"] == str(now)
        assert isinstance(line.timestamp, datetime)

    @pytest.fixture
    def cache_line(self):
        return SimpleNamespace(
            time=99.9,
            compile_time=1.0,
            verification_time=2,
            benchmark_time=3.0,
            strategy_time=4,
            framework_time=5.0,
            timestamp=datetime.now(),
        )

    @pytest.fixture
    def full_cache_line(self, cache_line):
        cache_line.a = 1
        cache_line.b = 1
        cache_line.c = 1
        return cache_line

    @pytest.fixture
    def assert_can_append_line(self, cache, cache_line):
        yield
        prev_len = len(cache.lines)
        cache.lines.append(**vars(cache_line))
        assert len(cache.lines) == prev_len + 1
        cache = Cache.read(cache.filepath)
        assert len(cache.lines) == prev_len + 1

    @pytest.fixture
    def assert_append_line__raises_ValueError(self, cache, cache_line):
        yield
        with pytest.raises(ValueError):
            cache.lines.append(**vars(cache_line))

    def test_line_append(self, full_cache_line, assert_can_append_line):
        pass

    def test_line_append__with_ErrorConfig(self, full_cache_line, assert_can_append_line):
        full_cache_line.time = util.InvalidConfig()

    def test_line_append__with_partial_params(self, cache_line, assert_append_line__raises_ValueError):
        cache_line.a = 1

    def test_line_append__with_invalid_tune_params(self, full_cache_line, assert_append_line__raises_ValueError):
        full_cache_line.a = 2

    def test_line_append__with_invalid_time(self, full_cache_line, assert_append_line__raises_ValueError):
        full_cache_line.time = "999"

    def test_line_append__with_invalid_compile_time(self, full_cache_line, assert_append_line__raises_ValueError):
        full_cache_line.compile_time = True

    def test_line_append__with_invalid_verificat_time(self, full_cache_line, assert_append_line__raises_ValueError):
        full_cache_line.verification_time = None

    def test_line_append__with_invalid_benchmark_time(self, full_cache_line, assert_append_line__raises_ValueError):
        full_cache_line.benchmark_time = []

    def test_line_append__with_invalid_strategy_time(self, full_cache_line, assert_append_line__raises_ValueError):
        full_cache_line.strategy_time = {}

    def test_line_append__with_invalid_framework_time(self, full_cache_line, assert_append_line__raises_ValueError):
        full_cache_line.framework_time = False

    def test_line_append__with_invalid_timestamp(self, full_cache_line, assert_append_line__raises_ValueError):
        full_cache_line.timestamp = 88

    def test_line_append__with_invalid_times(self, full_cache_line, assert_append_line__raises_ValueError):
        full_cache_line.times = ["Hello"]
