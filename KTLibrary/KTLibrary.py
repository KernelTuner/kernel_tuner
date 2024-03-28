import json
from pathlib import Path
from typing import Optional

from KTLibrary.CacheJSONEncoder import CacheJSONEncoder


class KTLibrary:
    def __init__(self):
        self.cache_file: Optional[dict] = None

    def read_file(self, file_path: str):
        self.cache_file = self.read_cache_file(file_path)

    @staticmethod
    def read_cache_file(file_path: str) -> dict:
        with open(file_path, "r") as file:
            json_data = json.load(file)
            print("JSON data loaded successfully")
            return json_data

    @staticmethod
    def write_cache_file(cache_file: dict, file_path: Path):
        with open(file_path, "w+") as file:
            json.dump(cache_file, file, cls=CacheJSONEncoder, indent=0)
