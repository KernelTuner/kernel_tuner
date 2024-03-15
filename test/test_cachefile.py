import os
from pathlib import Path
import hashlib
from KTLibrary.KTLibrary import KTLibrary

# Function that caculate the hash of a given file
def calculate_file_hash(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def test_read_cache_file():
    testLibrary = KTLibrary()
    file_path = 'SampleCacheFiles/convolution_A100.json'

    testLibrary.read_file(file_path)

    # Read device name of the given file
    file = testLibrary.cache_file
    device_name = file.get('device_name')
   
    # Check if the expected value of device name is in the file
    assert device_name == 'NVIDIA A100-PCIE-40GB'

def test_write_cache_file():
    testLibrary = KTLibrary()
    file_path = 'SampleCacheFiles/convolution_A100.json'

    testLibrary.read_file(file_path)
    p = Path.joinpath(Path(os.getcwd()), 'KTLibrary', 'test.json')
    testLibrary.write_cache_file(testLibrary.cache_file, p)

    # Calculate the hashes of the original file and the written file
    written_hash = calculate_file_hash(p)
    original_hash = calculate_file_hash(file_path)
    
    # Check if both hashes are the same
    assert written_hash == original_hash
