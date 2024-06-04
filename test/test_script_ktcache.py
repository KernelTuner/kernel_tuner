import json
from pathlib import Path
from shutil import copyfile

import jsonschema
import pytest

from kernel_tuner.scripts.ktcache import parse_args
from kernel_tuner.cache.file import read_cache
from kernel_tuner.cache.paths import CACHE_SCHEMAS_DIR
from kernel_tuner.cache.versions import VERSIONS
from kernel_tuner.cache.convert import convert_cache_file

TEST_PATH = Path(__file__).parent
TEST_CACHE_PATH = TEST_PATH / "test_cache_files"
TEST_CONVERT_PATH = TEST_PATH / "test_convert_files"

REAL_CACHE_FILE = TEST_CONVERT_PATH / "real_cache.json"

T4_CACHE = TEST_CONVERT_PATH / "t4_cache.json"
T4_TARGET = TEST_CONVERT_PATH / "t4_target.json"

SCHEMA_NEW = CACHE_SCHEMAS_DIR / str(VERSIONS[-1]) / "schema.json"


class TestCli:
    def test_convert(self, tmp_path):
        TEST_COPY_SRC = tmp_path / "temp_cache_src.json"
        TEST_COPY_DST = tmp_path / "temp_cache_dst.json"

        copyfile(REAL_CACHE_FILE, TEST_COPY_SRC)

        parser = parse_args(["convert", "--in", f"{TEST_COPY_SRC}", "--out", f"{TEST_COPY_DST}"])

        parser.func(parser)

        with open(TEST_COPY_DST) as c, open(SCHEMA_NEW) as s:
            real_cache = json.load(c)
            real_schema = json.load(s)
            jsonschema.validate(real_cache, real_schema)

    def test_convert_no_file(self, tmp_path):
        parser = parse_args(["convert", "--in", "bogus.json"])

        with pytest.raises(FileNotFoundError):
            parser.func(parser)

    def test_convert_unversioned(self, tmp_path):

        TEST_COPY_UNVERSIONED_SRC = TEST_CACHE_PATH / "small_cache_unversioned.json"
        TEST_COPY_UNVERSIONED_DST = tmp_path / "small_cache_unversioned.json"

        TEST_COPY_VERSIONED_SRC = TEST_CACHE_PATH / "small_cache.json"
        TEST_COPY_VERSIONED_DST = tmp_path / "small_cache.json"

        UNVERSIONED_CONVERT_OUT = tmp_path / "small_cache_unversioned_out.json"

        copyfile(TEST_COPY_UNVERSIONED_SRC, TEST_COPY_UNVERSIONED_DST)
        copyfile(TEST_COPY_VERSIONED_SRC, TEST_COPY_VERSIONED_DST)

        parser = parse_args(["convert", "--in", f"{TEST_COPY_UNVERSIONED_DST}", "--allow-version-absence", \
                            "-T", "1.0.0", "--out", f"{UNVERSIONED_CONVERT_OUT}"])
        
        parser.func(parser)

        convert_result = read_cache(UNVERSIONED_CONVERT_OUT)
        small_content = read_cache(TEST_COPY_VERSIONED_DST)

        assert convert_result == small_content 


    def test_t4(self, tmp_path):
        TEST_COPY_DST = tmp_path / "temp_cache_dst.json"

        parser = parse_args(["t4", "--in", f"{T4_CACHE}", "--out", f"{TEST_COPY_DST}"])

        parser.func(parser)

        with open(TEST_COPY_DST) as t4_file, open(T4_TARGET) as t4_target_file:
            t4 = json.load(t4_file)
            t4_target = json.load(t4_target_file)

        if t4 != t4_target:
            raise ValueError("Converted T4 does not match target T4")

    def test_deleteline_invalid_file(self, tmp_path):
        delete_file = tmp_path / "nonexistent.json"

        parser = parse_args(["delete-line", str(delete_file), "--key", "1"])

        with pytest.raises(FileNotFoundError):
            parser.func(parser)

    def test_deleteline_invalid_key(self, tmp_path):
        TEST_SMALL_CACHEFILE_SRC = TEST_CACHE_PATH / "small_cache.json"
        TEST_SMALL_CACHEFILE_DST = tmp_path / "small_cache.json"

        copyfile(TEST_SMALL_CACHEFILE_SRC, TEST_SMALL_CACHEFILE_DST)

        convert_cache_file(TEST_SMALL_CACHEFILE_DST)

        parser = parse_args(["delete-line", str(TEST_SMALL_CACHEFILE_DST), "--key", "1,1"])

        with pytest.raises(KeyError):
            parser.func(parser)

    def test_deleteline_valid_key(self, tmp_path):
        TEST_SMALL_CACHEFILE_THREE_ENTRIES_SRC = TEST_CACHE_PATH / "small_cache_three_entries.json"
        TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST = tmp_path / "small_cache_three_entries.json"

        TEST_SMALL_CACHEFILE_SRC = TEST_CACHE_PATH / "small_cache.json"
        TEST_SMALL_CACHEFILE_DST = tmp_path / "small_cache.json"

        copyfile(TEST_SMALL_CACHEFILE_THREE_ENTRIES_SRC, TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST)
        copyfile(TEST_SMALL_CACHEFILE_SRC, TEST_SMALL_CACHEFILE_DST)

        convert_cache_file(TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST)
        convert_cache_file(TEST_SMALL_CACHEFILE_DST)

        # Removing key 32,1 from small_cache_three_entries.json should result in small_cache.json

        parser = parse_args(["delete-line", str(TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST), "--key", "32,1"])

        parser.func(parser)

        delete_result = read_cache(TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST)
        small_content = read_cache(TEST_SMALL_CACHEFILE_DST)

        assert delete_result == small_content

    def test_getline_invalid_file(self, tmp_path):
        INPUT_FILE = tmp_path / "nonexistent.json"

        parser = parse_args(["get-line", str(INPUT_FILE), "--key", "1"])

        with pytest.raises(FileNotFoundError):
            parser.func(parser)

    def test_getline_invalid_key(self, tmp_path):
        TEST_SMALL_CACHEFILE_SRC = TEST_CACHE_PATH / "large_cache.json"
        TEST_SMALL_CACHEFILE_DST = tmp_path / "large_cache.json"

        copyfile(TEST_SMALL_CACHEFILE_SRC, TEST_SMALL_CACHEFILE_DST)

        # We know cacheline key 1 is not contained in test_cache_files/large_cache.json

        parser = parse_args(["get-line", str(TEST_SMALL_CACHEFILE_DST), "--key", "1"])

        with pytest.raises(KeyError):
            parser.func(parser)

    def test_getline_valid_key(self, tmp_path):
        TEST_SMALL_CACHEFILE_SRC = TEST_CACHE_PATH / "large_cache.json"
        TEST_SMALL_CACHEFILE_DST = tmp_path / "large_cache.json"

        copyfile(TEST_SMALL_CACHEFILE_SRC, TEST_SMALL_CACHEFILE_DST)

        # We know cacheline key 16,1,1,2,0,0,1,1,15,15 is contained in test_cache_files/large_cache.json
        parser = parse_args(["get-line", str(TEST_SMALL_CACHEFILE_DST), "--key", "16,1,1,2,0,0,1,1,15,15"])

        parser.func(parser)

    def test_merge_invalid_file(self, tmp_path):
        INVALID_FILE = tmp_path / "nonexistent.json"
        INVALID_FILE_TWO = tmp_path / "nonexistent2.json"

        parser = parse_args(["merge", str(INVALID_FILE), str(INVALID_FILE_TWO), "-o", "test.json"])

        with pytest.raises(FileNotFoundError):
            parser.func(parser)

    def test_merge_nonequiv_key(self, tmp_path):
        # These files have nonequivalent `device_name`
        TEST_SMALL_CACHEFILE_SRC = TEST_CACHE_PATH / "small_cache.json"
        TEST_SMALL_CACHEFILE_DST = tmp_path / "small_cache.json"
        TEST_LARGE_CACHEFILE_SRC = TEST_CACHE_PATH / "large_cache.json"
        TEST_LARGE_CACHEFILE_DST = tmp_path / "large_cache.json"

        copyfile(TEST_SMALL_CACHEFILE_SRC, TEST_SMALL_CACHEFILE_DST)
        copyfile(TEST_LARGE_CACHEFILE_SRC, TEST_LARGE_CACHEFILE_DST)

        parser = parse_args(["merge", str(TEST_SMALL_CACHEFILE_DST), str(TEST_LARGE_CACHEFILE_DST), "-o", "test.json"])

        with pytest.raises(ValueError):
            parser.func(parser)

    def test_merge_one_file(self):
        parser = parse_args(["merge", "1.json", "-o", "test.json"])

        with pytest.raises(ValueError):
            parser.func(parser)

    def test_merge_correct_two_files(self, tmp_path):
        TEST_SMALL_CACHEFILE_ONE_ENTRY_SRC = TEST_CACHE_PATH / "small_cache_one_entry.json"
        TEST_SMALL_CACHEFILE_ONE_ENTRY_DST = tmp_path / "small_cache_one_entry.json"
        TEST_SMALL_CACHEFILE_THREE_ENTRIES_SRC = TEST_CACHE_PATH / "small_cache_three_entries.json"
        TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST = tmp_path / "small_cache_three_entries.json"

        TEST_SMALL_CACHEFILE_SRC = TEST_CACHE_PATH / "small_cache.json"
        TEST_SMALL_CACHEFILE_DST = tmp_path / "small_cache.json"

        TEST_MERGE_OUTPUT = tmp_path / "merge_out.json"

        copyfile(TEST_SMALL_CACHEFILE_ONE_ENTRY_SRC, TEST_SMALL_CACHEFILE_ONE_ENTRY_DST)
        copyfile(TEST_SMALL_CACHEFILE_THREE_ENTRIES_SRC, TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST)
        copyfile(TEST_SMALL_CACHEFILE_SRC, TEST_SMALL_CACHEFILE_DST)

        # The newly created merge result will use the latest version, so convert the desired result
        convert_cache_file(TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST)

        # Merging small_cache_one_entry.json and small_cache.json should result in small_cache_three_entries.json

        parser = parse_args(
            [
                "merge",
                str(TEST_SMALL_CACHEFILE_DST),
                str(TEST_SMALL_CACHEFILE_ONE_ENTRY_DST),
                "--out",
                str(TEST_MERGE_OUTPUT),
            ]
        )

        parser.func(parser)

        merge_result = read_cache(TEST_MERGE_OUTPUT)

        dest_output = read_cache(TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST)

        assert merge_result == dest_output

    def test_merge_when_keys_overlap(self, tmp_path):
        TEST_SMALL_CACHEFILE_ONE_ENTRY_SRC = TEST_CACHE_PATH / "small_cache_one_entry.json"
        TEST_SMALL_CACHEFILE_ONE_ENTRY_DST = tmp_path / "small_cache_one_entry.json"
        TEST_SMALL_CACHEFILE_THREE_ENTRIES_SRC = TEST_CACHE_PATH / "small_cache_three_entries.json"
        TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST = tmp_path / "small_cache_three_entries.json"

        OUT_FILE = tmp_path / "out.json"

        copyfile(TEST_SMALL_CACHEFILE_ONE_ENTRY_SRC, TEST_SMALL_CACHEFILE_ONE_ENTRY_DST)
        copyfile(TEST_SMALL_CACHEFILE_THREE_ENTRIES_SRC, TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST)

        # We know that small_cache_one_entry.json and small_cache_three_entries.json have overlap for key 32,1
        parser = parse_args(
            [
                "merge",
                str(TEST_SMALL_CACHEFILE_ONE_ENTRY_DST),
                str(TEST_SMALL_CACHEFILE_THREE_ENTRIES_DST),
                "--out",
                str(OUT_FILE),
            ]
        )

        with pytest.raises(KeyError):
            parser.func(parser)
