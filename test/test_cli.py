import json
import jsonschema

from pathlib import Path
from shutil import copyfile

import pytest

from kernel_tuner.cache.paths import CACHE_SCHEMAS_DIR
from kernel_tuner.cache.versions import VERSIONS

from kernel_tuner.cache.cli import parse_args

TEST_PATH         = Path(__file__).parent
TEST_CONVERT_PATH = TEST_PATH / "test_convert_files"

TEST_CONVERT_PATH = TEST_PATH / "test_convert_files"
REAL_CACHE_FILE   = TEST_CONVERT_PATH / "real_cache.json"

SCHEMA_NEW        = CACHE_SCHEMAS_DIR / str(VERSIONS[-1]) / "schema.json"



class TestCli:
    def test_convert(self, tmp_path):
        TEST_COPY_SRC = tmp_path / "temp_cache_src.json"
        TEST_COPY_DST = tmp_path / "temp_cache_dst.json"

        copyfile(REAL_CACHE_FILE, TEST_COPY_SRC)

        parser = parse_args(['convert',
                             '-i', f'{TEST_COPY_SRC}',
                             '-o', f'{TEST_COPY_DST}'])
        
        parser.func(parser)
        
        with open(TEST_COPY_DST) as c, open(SCHEMA_NEW) as s:
            real_cache  = json.load(c)
            real_schema = json.load(s)
            jsonschema.validate(real_cache, real_schema)

    def test_convert_no_file(self, tmp_path):
        parser = parse_args(['convert', '-i', 'bogus.json'])

        with pytest.raises(ValueError):
            parser.func(parser)