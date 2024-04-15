import pytest
import filecmp
from pathlib import Path

from kernel_tuner.cache.file import read_cache_file, write_cache_file


TEST_PATH = Path(__file__).parent
TEST_CACHE_PATH = TEST_PATH / "test_cache_files"
CACHE_SAMPLE_PATH = TEST_CACHE_PATH / "convolution_A100.json"


# Function that caculate the hash of a given file
def test_read_cache_file():
    # Read device name of the given file
    file_content = read_cache_file(CACHE_SAMPLE_PATH)
    device_name = file_content.get("device_name")

    # Check if the expected value of device name is in the file
    assert device_name == "NVIDIA A100-PCIE-40GB"


@pytest.fixture
def tmp_output_file(tmp_path):
    return tmp_path / "test.json"


def test_write_cache_file(tmp_output_file):
    sample_cache = read_cache_file(CACHE_SAMPLE_PATH)

    write_cache_file(sample_cache, tmp_output_file)

    assert filecmp.cmp(tmp_output_file, CACHE_SAMPLE_PATH)
