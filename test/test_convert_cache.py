import json
from pathlib import Path
from shutil import copyfile

import pytest
import jsonschema
import kernel_tuner

from kernel_tuner.convert.convert import convert_cache_file
from kernel_tuner.util import delete_temp_file


KERNEL_TUNER_PATH = Path(kernel_tuner.__file__).parent
SCHEMAS_PATH = KERNEL_TUNER_PATH / "schema/cache/convert_temp"
SCHEMA_OLD = SCHEMAS_PATH / "1.0.0/schema.json"
SCHEMA_NEW = SCHEMAS_PATH / "1.2.0/schema.json"

TEST_PATH = Path(__file__).parent
TEST_CONVERT_PATH = TEST_PATH / "test_convert_files"
TEST_CONVERT_FILE = TEST_CONVERT_PATH / "old_cache.json"
TEST_CONVERT_COPY = TEST_CONVERT_PATH / "temp_cache.json"


class TestConvertCache:
    def test_convert(self):
        # The conversion function converts the file it is given, so make a copy
        copyfile(TEST_CONVERT_FILE, TEST_CONVERT_COPY)

        with open(TEST_CONVERT_COPY) as c, open(SCHEMA_OLD) as s:
            cache      = json.load(c)
            old_schema = json.load(s)
            jsonschema.validate(cache, old_schema)
            
        convert_cache_file(TEST_CONVERT_COPY)

        with open(TEST_CONVERT_COPY) as c, open(SCHEMA_NEW) as s:
            cache      = json.load(c)
            new_schema = json.load(s)
            jsonschema.validate(cache, new_schema)

        delete_temp_file(TEST_CONVERT_COPY)

        return