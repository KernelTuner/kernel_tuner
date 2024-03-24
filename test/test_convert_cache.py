import json
from pathlib import Path
from shutil import copyfile

import pytest
import jsonschema
import kernel_tuner

from kernel_tuner.cache.convert import convert_cache_file
from kernel_tuner.cache.convert import VERSIONS

KERNEL_TUNER_PATH = Path(kernel_tuner.__file__).parent
TEST_PATH         = Path(__file__).parent

# Mock schema files
MOCK_CONVERT_PATH = TEST_PATH / "test_convert_files"
MOCK_CONVERT_FILE = MOCK_CONVERT_PATH / "mock_cache.json"

MOCK_SCHEMAS_PATH = MOCK_CONVERT_PATH / "mock_schemas"
MOCK_SCHEMA_OLD   = MOCK_SCHEMAS_PATH / "1.0.0/schema.json"
MOCK_SCHEMA_NEW   = MOCK_SCHEMAS_PATH / "1.2.0/schema.json"

# Actual schema files
CONVERT_PATH      = TEST_PATH / "test_cache_files"
CONVERT_FILE      = CONVERT_PATH / "small_cache.json"

SCHEMAS_PATH      = KERNEL_TUNER_PATH / "schema/cache"
SCHEMA_OLD        = SCHEMAS_PATH / VERSIONS[ 0] / "schema.json"
SCHEMA_NEW        = SCHEMAS_PATH / VERSIONS[-1] / "schema.json"



class TestConvertCache:
    # Test conversion system using mock schema and cache files
    def test_convert_mock(self, tmp_path):
        # The conversion function converts the file it is given, so make a copy
        MOCK_CONVERT_COPY = tmp_path / "temp_cache.json"
        copyfile(MOCK_CONVERT_FILE, MOCK_CONVERT_COPY)

        with open(MOCK_CONVERT_COPY) as c, open(MOCK_SCHEMA_OLD) as s:
            mock_cache  = json.load(c)
            mock_schema = json.load(s)
            jsonschema.validate(mock_cache, mock_schema)
        
        convert_cache_file(MOCK_CONVERT_COPY, 
                           self._CONVERT_FUNCTIONS,
                           self._VERSIONS)

        with open(MOCK_CONVERT_COPY) as c, open(MOCK_SCHEMA_NEW) as s:
            mock_cache  = json.load(c)
            mock_schema = json.load(s)
            jsonschema.validate(mock_cache, mock_schema)

        return
    
    # Test the implemented conversion functions
    def test_convert_real(self, tmp_path):
        # The conversion function converts the file it is given, so make a copy
        CONVERT_COPY = tmp_path / "temp_cache.json"
        copyfile(CONVERT_FILE, CONVERT_COPY)

        with open(CONVERT_COPY) as c, open(SCHEMA_OLD) as s:
            mock_cache  = json.load(c)
            mock_schema = json.load(s)
            jsonschema.validate(mock_cache, mock_schema)
        
        convert_cache_file(CONVERT_COPY)

        with open(CONVERT_COPY) as c, open(SCHEMA_NEW) as s:
            mock_cache  = json.load(c)
            mock_schema = json.load(s)
            jsonschema.validate(mock_cache, mock_schema)

        return

    # Mock convert functions
    def _c1_0_0_to_1_1_0(cache):
        cache["field2"] = dict()
        cache["schema_version"] = "1.1.0"
        return cache

    def _c1_1_0_to_1_1_1(cache):
        cache["schema_version"] = "1.1.1"
        return cache
    
    def _c1_1_1_to_1_2_0(cache):
        cache["field1"] = dict()
        cache["schema_version"] = "1.2.0"
        return cache

    _CONVERT_FUNCTIONS = {
        "1.0.0": _c1_0_0_to_1_1_0,
        "1.1.0": _c1_1_0_to_1_1_1,
        "1.1.1": _c1_1_1_to_1_2_0, 
    }

    _VERSIONS = [
        "1.0.0",
        "1.1.0",
        "1.1.1",
        "1.2.0"
    ]