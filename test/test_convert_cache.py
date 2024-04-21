import json
from pathlib import Path
from shutil import copyfile

import pytest
import jsonschema
import kernel_tuner

from kernel_tuner.cache.convert import convert_cache_file
from kernel_tuner.cache.convert import unversioned_convert
from kernel_tuner.cache.convert import default_convert
from kernel_tuner.cache.convert import VERSIONS

KERNEL_TUNER_PATH = Path(kernel_tuner.__file__).parent
TEST_PATH         = Path(__file__).parent
TEST_CONVERT_PATH = TEST_PATH / "test_convert_files"

# Mock schema files
MOCK_CACHE_FILE = TEST_CONVERT_PATH / "mock_cache.json"

MOCK_SCHEMAS_PATH = TEST_CONVERT_PATH / "mock_schemas"
MOCK_SCHEMA_OLD   = MOCK_SCHEMAS_PATH / "1.0.0/schema.json"
MOCK_SCHEMA_NEW   = MOCK_SCHEMAS_PATH / "1.2.0/schema.json"
UPGRADED_SCHEMA   = MOCK_SCHEMAS_PATH / "1.1.0/schema.json"

# Actual schema files
REAL_CACHE_FILE      = TEST_CONVERT_PATH / "real_cache.json"

SCHEMAS_PATH      = KERNEL_TUNER_PATH / "schema/cache"
SCHEMA_OLD        = SCHEMAS_PATH / VERSIONS[ 0] / "schema.json"
SCHEMA_NEW        = SCHEMAS_PATH / VERSIONS[-1] / "schema.json"

# Test files
NO_VERSION_FIELD  = TEST_CONVERT_PATH / "no_version_field.json"
TOO_HIGH_VERSION  = TEST_CONVERT_PATH / "too_high_version.json"
NOT_REAL_VERSION  = TEST_CONVERT_PATH / "not_real_version.json"




class TestConvertCache:
    # Test using mock schema/cache files and conversion functions
    def test_conversion_system(self, tmp_path):
        TEST_COPY = tmp_path / "temp_cache.json"
        copyfile(MOCK_CACHE_FILE, TEST_COPY)

        with open(TEST_COPY) as c, open(MOCK_SCHEMA_OLD) as s:
            mock_cache  = json.load(c)
            mock_schema = json.load(s)
            jsonschema.validate(mock_cache, mock_schema)
        
        convert_cache_file(TEST_COPY, 
                           self._CONVERT_FUNCTIONS,
                           self._VERSIONS)

        with open(TEST_COPY) as c, open(MOCK_SCHEMA_NEW) as s:
            mock_cache  = json.load(c)
            mock_schema = json.load(s)
            jsonschema.validate(mock_cache, mock_schema)

        return
    
    # Test using implemented schema/cache files and conversion functions
    def test_convert_real(self, tmp_path):
        TEST_COPY = tmp_path / "temp_cache.json"
        copyfile(REAL_CACHE_FILE, TEST_COPY)

        with open(TEST_COPY) as c, open(SCHEMA_OLD) as s:
            real_cache  = json.load(c)
            real_schema = json.load(s)
            jsonschema.validate(real_cache, real_schema)
        
        convert_cache_file(TEST_COPY)

        with open(TEST_COPY) as c, open(SCHEMA_NEW) as s:
            real_cache  = json.load(c)
            real_schema = json.load(s)
            jsonschema.validate(real_cache, real_schema)

        return
    
    def test_no_version_field(self, tmp_path):
        TEST_COPY = tmp_path / "temp_cache.json"
        copyfile(NO_VERSION_FIELD, TEST_COPY)

        with open(TEST_COPY) as c:
            cache = json.load(c)

        cache = unversioned_convert(cache, MOCK_SCHEMAS_PATH)
        
        with open(MOCK_SCHEMA_OLD) as s:
            schema = json.load(s)
            jsonschema.validate(cache, schema)

        return

    def test_too_high_version(self, tmp_path):
        TEST_COPY = tmp_path / "temp_cache.json"
        copyfile(TOO_HIGH_VERSION, TEST_COPY)

        with pytest.raises(ValueError):
            convert_cache_file(TEST_COPY,
                               self._CONVERT_FUNCTIONS,
                               self._VERSIONS)
            
    def test_not_real_version(self, tmp_path):
        TEST_COPY = tmp_path / "temp_cache.json"
        copyfile(NOT_REAL_VERSION, TEST_COPY)

        with pytest.raises(ValueError):
            convert_cache_file(TEST_COPY,
                               self._CONVERT_FUNCTIONS,
                               self._VERSIONS)
            
    def test_default_convert(self):
        with open(MOCK_CACHE_FILE) as c:
            cache = json.load(c)
    
        cache = default_convert(cache,
                                "1.0.0",
                                self._VERSIONS,
                                MOCK_SCHEMAS_PATH)

        with open(UPGRADED_SCHEMA) as s:
            upgraded_schema = json.load(s)
            jsonschema.validate(cache, upgraded_schema)
    
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