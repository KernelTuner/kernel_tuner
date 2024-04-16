import pytest
import shutil
from copy import deepcopy
from pathlib import Path

from kernel_tuner.cache.file import (
    read_cache,
    write_cache,
    append_cache_line,
    close_opened_cache,
    open_closed_cache,
    open_cache,
    close_cache,
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


def test_read_cache(cache_1_path, output_path):
    shutil.copy(cache_1_path, output_path)

    # Read device name of the given file
    file_content = read_cache(output_path)
    device_name = file_content.get("device_name")

    # Check if the expected value of device name is in the file
    assert device_name == "Testing"

    # Check if the file has remained unchanged
    with open(output_path) as output, open(cache_1_path) as expected:
        assert output.read() == expected.read()


def test_read_cache_ensuring_open_and_closed(cache_1_path):
    with pytest.raises(ValueError):
        read_cache(cache_1_path, ensure_open=True, ensure_closed=True)


def test_open_cache(cache_1_path, output_path):
    shutil.copy(cache_1_path, output_path)
    open_cache(output_path)

    with open(output_path) as output, open(OPEN_CACHE_COMMA_PATH) as expected:
        assert output.read() == expected.read()


def test_close_cache(cache_1_path, output_path, request):
    shutil.copy(cache_1_path, output_path)
    close_cache(output_path)

    with open(output_path) as output, open(CLOSED_CACHE_PATH) as expected:
        assert output.read() == expected.read()


def test_write_cache(output_path):
    sample_cache = read_cache(CLOSED_CACHE_PATH)

    write_cache(sample_cache, output_path)
    with open(output_path, "r") as output, open(CLOSED_CACHE_PATH, "r") as input:
        assert output.read() == input.read()


def test_write_open_cache(output_path):
    sample_cache = read_cache(OPEN_CACHE_COMMA_PATH)

    write_cache(sample_cache, output_path, keep_open=True)
    with open(output_path, "r") as output, open(OPEN_CACHE_COMMA_PATH, "r") as input:
        assert output.read() == input.read()


def test_append_cache_line(output_path):
    sample_cache = read_cache(CLOSED_CACHE_PATH)

    smaller_cache = deepcopy(sample_cache)
    key = next(iter(smaller_cache["cache"].keys()))
    line = smaller_cache["cache"].pop(key)

    write_cache(smaller_cache, output_path, keep_open=True)
    append_cache_line(key, line, output_path)

    print(sample_cache)
    assert read_cache(output_path) == sample_cache


def test_close_opened_cache(cache_1_path, output_path):
    shutil.copy(cache_1_path, output_path)
    close_opened_cache(output_path)
    with open(output_path, "r") as output, open(CLOSED_CACHE_PATH, "r") as input:
        assert output.read() == input.read()


def test_open_closed_cache(cache_1_path, output_path):
    shutil.copy(cache_1_path, output_path)
    open_closed_cache(output_path)
    with open(output_path, "r") as output, open(OPEN_CACHE_COMMA_PATH, "r") as input:
        assert output.read() == input.read()
