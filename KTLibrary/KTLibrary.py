import json
from pathlib import Path
from typing import Optional

from KTLibrary.FormattedEncoder import FormattedEncoder


class KTLibrary:
    def __init__(self):
        self.cache_file: Optional[dict] = None

    def read_file(self, file_path: str):
        self.cache_file = self.read_cache_file(file_path)

    @staticmethod
    def read_cache_file(file_path: str) -> dict:
        # Read the cache file where we catch any exceptions if thrown and printed for the user to see.
        try:
            with open(file_path, "r") as file:
                json_data = json.load(file)
                print("JSON data loaded successfully")
        except FileNotFoundError as e:
            print("File not found: ", e)
        except json.JSONDecodeError as e:
            print("Invalid JSON format: ", e)
        except PermissionError as e:
            print("Permission denied", e)
        return json_data

    @staticmethod
    def write_cache_file(cache_file: dict, file_path: Path):
        # Write the cache to a file using our custom FormattedEncoder.
        # Any exception caught while doing so will be printed.
        try:
            with open(file_path, "w+") as file:
                json.dump(cache_file, file, cls=FormattedEncoder, indent=0)
        except PermissionError as e:
            print("Permission denied: ", e)
