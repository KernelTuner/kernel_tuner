import pytest
import filecmp
from pathlib import Path

from kernel_tuner.cache.file import read_cache_file, write_cache_file


TEST_PATH = Path(__file__).parent
TEST_CACHE_PATH = TEST_PATH / "test_cache_files"
CACHE_SAMPLE_PATH = TEST_CACHE_PATH / "convolution_A100.json"
OPEN_CACHE_PATH_1 = TEST_CACHE_PATH / "open_cache_no_comma.json"
OPEN_CACHE_PATH_2 = TEST_CACHE_PATH / "open_cache_with_comma.json"


@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "output.json"


@pytest.mark.parametrize(
    "filename",
    [CACHE_SAMPLE_PATH, OPEN_CACHE_PATH_1, OPEN_CACHE_PATH_2],
    ids=["closed cache", "open cache without comma", "open cache with comma"],
)
def test_read_cache_file(filename):
    # Read device name of the given file
    file_content = read_cache_file(CACHE_SAMPLE_PATH)
    device_name = file_content.get("device_name")

    # Check if the expected value of device name is in the file
    assert device_name == "NVIDIA A100-PCIE-40GB"


def test_write_cache_file(output_path):
    sample_cache = read_cache_file(CACHE_SAMPLE_PATH)

    write_cache_file(sample_cache, output_path)

    assert filecmp.cmp(output_path, CACHE_SAMPLE_PATH)
