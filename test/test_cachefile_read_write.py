import os
import pytest
from pathlib import Path
import hashlib
from kernel_tuner.cache.KTLibrary.cachefile import read_cache_file, write_cache_file


# Function that caculate the hash of a given file
def calculate_file_hash(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_read_cache_file():
    current_file_dir = Path(__file__).resolve().parent.parent

    input_file = current_file_dir / 'test' / 'SampleCacheFiles' / 'convolution_A100.json'

    # Read device name of the given file
    file_content = read_cache_file(input_file)
    device_name = file_content.get('device_name')
   
    # Check if the expected value of device name is in the file
    assert device_name == 'NVIDIA A100-PCIE-40GB'


@pytest.fixture
def tmp_output_file(tmp_path):
    return tmp_path / 'test.json'


def test_write_cache_file(tmp_output_file):

    current_file_dir = Path(__file__).resolve().parent.parent

    input_file = current_file_dir / 'test' / 'SampleCacheFiles' / 'convolution_A100.json'

    p = tmp_output_file
    
    # cache_file = read_cache_file(file_path)
    write_cache_file(read_cache_file(input_file), p)

    # Calculate the hashes of the original file and the written file
    written_hash = calculate_file_hash(p)
    original_hash = calculate_file_hash(input_file)
    
    # Check if both hashes are the same
    assert written_hash == original_hash
