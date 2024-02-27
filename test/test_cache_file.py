import json
from pathlib import Path, PurePath
import glob
import os

import pytest
import jsonschema


def _get_cache_name(path: os.PathLike):
    path = PurePath(path)
    return str(path.relative_to(TEST_CACHE_PATH))


def _recursive_glob(path: os.PathLike):
    return glob.glob(path.__fspath__(), recursive=True)


PROJECT_DIR = Path(__file__).parents[1]

SCHEMA_PATH = PROJECT_DIR / "kernel_tuner/schema/cache/1.0.0/schema.json"
TEST_CACHE_PATH = PROJECT_DIR / "test/test_cache_files"

VALID_CACHE_PATHS = _recursive_glob(TEST_CACHE_PATH / "**/valid_*.json")
VALID_CACHE_NAMES = [_get_cache_name(p) for p in VALID_CACHE_PATHS]

INVALID_CACHE_PATHS = _recursive_glob(TEST_CACHE_PATH / "**/invalid_*.json")
INVALID_CACHE_NAMES = [_get_cache_name(p) for p in INVALID_CACHE_PATHS]


@pytest.fixture(scope="session")
def validator():
    with open(SCHEMA_PATH) as f:
        schema_json = json.load(f)
        jsonschema.Draft202012Validator.check_schema(schema_json)
        return jsonschema.Draft202012Validator(schema_json)


@pytest.fixture(
    scope="session",
    params=VALID_CACHE_PATHS,
    ids=VALID_CACHE_NAMES)
def valid_cache(request):
    cache_path = request.param
    with open(cache_path) as f:
        return json.load(f)


@pytest.fixture(
    scope="session",
    params=INVALID_CACHE_PATHS,
    ids=INVALID_CACHE_NAMES)
def invalid_cache(request):
    cache_path = request.param
    with open(cache_path) as f:
        return json.load(f)


def test_valid_cache(validator, valid_cache):
    validator.validate(valid_cache)


def test_invalid_cache(validator, invalid_cache):
    with pytest.raises(jsonschema.exceptions.ValidationError):
        validator.validate(invalid_cache)
