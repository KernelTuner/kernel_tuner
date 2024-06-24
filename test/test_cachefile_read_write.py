import re

import pytest
import shutil
from copy import deepcopy
from pathlib import Path

from kernel_tuner.cache.file import (
    InvalidCacheError,
    CachedLinePosition,
    read_cache,
    write_cache,
    append_cache_line,
)


TEST_DIR = Path(__file__).parent
TEST_CACHE_DIR = TEST_DIR / "test_cache_files"
SMALL_CACHE_PATH = TEST_CACHE_DIR / "small_cache.json"
LARGE_CACHE_PATH = TEST_CACHE_DIR / "large_cache.json"


@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "output.json"


@pytest.fixture(params=[SMALL_CACHE_PATH, LARGE_CACHE_PATH], ids=["small", "large"])
def cache_path(request):
    return request.param


def test_read_cache(cache_path, output_path):
    shutil.copy(cache_path, output_path)

    # Read device name of the given file
    file_content = read_cache(output_path)
    device_name = file_content.get("device_name")

    # Check if the expected value of device name is in the file
    assert isinstance(device_name, str)

    # Check if the file has remained unchanged
    with open(output_path) as output, open(cache_path) as expected:
        assert output.read().rstrip() == expected.read().rstrip()


def test_read_cache__which_is_unparsable(output_path):
    with open(output_path, "w") as file:
        file.write("INVALID")

    with pytest.raises(InvalidCacheError):
        read_cache(output_path)


def test_write_cache(cache_path, output_path):
    sample_cache = read_cache(cache_path)

    write_cache(sample_cache, output_path)
    regex = re.compile(r'[\s]+')

    # do a whitespace insensitive compare
    with open(output_path, "r") as output, open(cache_path, "r") as input:
        assert regex.sub("", output.read().rstrip()) == regex.sub("", input.read().rstrip())


def test_append_cache_line(cache_path, output_path):
    sample_cache = read_cache(cache_path)

    smaller_cache = deepcopy(sample_cache)
    key = next(iter(smaller_cache["cache"].keys()))
    line = smaller_cache["cache"].pop(key)

    write_cache(smaller_cache, output_path)
    append_cache_line(key, line, output_path)

    assert read_cache(output_path) == sample_cache


def test_append_cache_line__with_position(cache_path, output_path):
    sample_cache = read_cache(cache_path)

    empty_cache = deepcopy(sample_cache)
    cache_lines = deepcopy(empty_cache["cache"])
    empty_cache["cache"].clear()
    write_cache(empty_cache, output_path)

    pos = CachedLinePosition()
    for key, line in cache_lines.items():
        pos = append_cache_line(key, line, output_path, pos)

    assert read_cache(output_path) == sample_cache


def test_append_cache_line__to_empty_cache(cache_path, output_path):
    sample_cache = read_cache(cache_path)

    empty_cache = deepcopy(sample_cache)
    cache_lines = deepcopy(empty_cache["cache"])
    empty_cache["cache"].clear()

    write_cache(empty_cache, output_path)
    for key, line in cache_lines.items():
        append_cache_line(key, line, output_path)

    assert read_cache(output_path) == sample_cache
