import os
from pathlib import Path

from KTLibrary.KTLibrary import KTLibrary


if __name__ == '__main__':

    testLibrary = KTLibrary()

    testLibrary.read_file('SampleCacheFiles/convolution_A100.json')

    p = Path.joinpath(
        Path(os.getcwd()), 'KTLibrary', 'test.csv'
    )

    testLibrary.write_cache_file(testLibrary.cache_file, p)