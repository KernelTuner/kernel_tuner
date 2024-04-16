import pytest
import shutil
from copy import deepcopy
from pathlib import Path

from kernel_tuner.cache.file import (
    read_cache_file,
    write_cache_file,
    append_cache_line,
    close_cache_file,
    open_cache_file,
)


TEST_PATH = Path(__file__).parent
TEST_CACHE_PATH = TEST_PATH / "test_cache_files"
CLOSED_CACHE_PATH = TEST_CACHE_PATH / "small_cache.json"
OPEN_CACHE_NO_COMMA_PATH = TEST_CACHE_PATH / "open_cache_no_comma.json"
OPEN_CACHE_COMMA_PATH = TEST_CACHE_PATH / "open_cache_with_comma.json"


@pytest.fixture(
    params=[CLOSED_CACHE_PATH, OPEN_CACHE_NO_COMMA_PATH, OPEN_CACHE_COMMA_PATH],
    ids=["closed cache", "open cache without comma", "open cache with comma"],
)
def cache_1_path(request):
    return request.param


@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "output.json"


def test_read_cache_file(cache_1_path):
    # Read device name of the given file
    file_content = read_cache_file(cache_1_path)
    device_name = file_content.get("device_name")

    # Check if the expected value of device name is in the file
    assert device_name == "Testing"


def test_read_cache_file_ensure_open(cache_1_path, output_path):
    shutil.copy(cache_1_path, output_path)
    _ = read_cache_file(output_path, ensure_open=True)

    with open(output_path) as output, open(OPEN_CACHE_COMMA_PATH) as expected:
        assert output.read() == expected.read()


def test_read_cache_file_ensure_closed(cache_1_path, output_path, request):
    shutil.copy(cache_1_path, output_path)
    _ = read_cache_file(output_path, ensure_closed=True)

    with open(output_path) as output, open(CLOSED_CACHE_PATH) as expected:
        assert output.read() == expected.read()


def test_write_cache_file(output_path):
    sample_cache = read_cache_file(CLOSED_CACHE_PATH)

    write_cache_file(sample_cache, output_path)
    with open(output_path, "r") as output, open(CLOSED_CACHE_PATH, "r") as input:
        assert output.read() == input.read()


def test_write_open_cache_file(output_path):
    sample_cache = read_cache_file(OPEN_CACHE_COMMA_PATH)

    write_cache_file(sample_cache, output_path, keep_open=True)
    with open(output_path, "r") as output, open(OPEN_CACHE_COMMA_PATH, "r") as input:
        assert output.read() == input.read()


def test_append_cache_line(output_path):
    sample_cache = read_cache_file(CLOSED_CACHE_PATH)

    smaller_cache = deepcopy(sample_cache)
    key = next(iter(smaller_cache["cache"].keys()))
    line = smaller_cache["cache"].pop(key)

    write_cache_file(smaller_cache, output_path, keep_open=True)
    append_cache_line(key, line, output_path)

    print(sample_cache)
    assert read_cache_file(output_path) == sample_cache


def test_close_cache_file(cache_1_path, output_path):
    shutil.copy(cache_1_path, output_path)
    close_cache_file(output_path)
    with open(output_path, "r") as output, open(CLOSED_CACHE_PATH, "r") as input:
        assert output.read() == input.read()


def test_open_cache_file(cache_1_path, output_path):
    shutil.copy(cache_1_path, output_path)
    open_cache_file(output_path)
    with open(output_path, "r") as output, open(OPEN_CACHE_COMMA_PATH, "r") as input:
        assert output.read() == input.read()
