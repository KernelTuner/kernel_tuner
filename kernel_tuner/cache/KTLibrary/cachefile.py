import json
from pathlib import Path

from kernel_tuner.cache.KTLibrary.CacheJSONEncoder import CacheJSONEncoder


def read_cache_file(file_path: Path) -> dict:
    with open(file_path, "r") as file:
        json_data = json.load(file)
        print("JSON data loaded successfully")
        return json_data


def write_cache_file(cache_file: dict, file_path: Path):
    with open(file_path, "w+") as file:
        json.dump(cache_file, file, cls=CacheJSONEncoder, indent=0)
