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


    def test_merge_one_file(self):
        parser = parse_args(['merge', '1.json', '-o', 'test.json'])

        with pytest.raises(ValueError):
            parser.func(parser)

    def test_merge_invalid_file(self, tmp_path):
        invalid_file = tmp_path / "nonexistent.json"
        invalid_file_two = tmp_path / "nonexistent2.json"

        parser = parse_args(["merge", str(invalid_file), str(invalid_file_two), "-o", "test.json"])

        with pytest.raises(FileNotFoundError):
            parser.func(parser)


    def test_delete_invalid_file(self, tmp_path):
        delete_file = tmp_path / "nonexistent.json"

        parser = parse_args(["delete-line", str(delete_file), "--key", "1"])

        with pytest.raises(FileNotFoundError):
            parser.func(parser)


