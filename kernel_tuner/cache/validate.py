"""Validator of the cache files."""

import json
import jsonschema
from datetime import datetime
from .paths import get_schema_path
from .versions import VERSIONS, Version

from functools import lru_cache


def _get_format_checker():
    """Returns a JSON format checker instance."""
    format_checker = jsonschema.FormatChecker()

    @format_checker.checks("date-time")
    def _check_iso_datetime(instance):
        try:
            datetime.fromisoformat(instance)
            return True
        except ValueError:
            return False

    return format_checker


@lru_cache
def get_format_checker():
    """Returns a cached JSON format checker instance."""
    return _get_format_checker()


def validate_data(data_file):
    """Validate the input data in a cache file against its corresponding schema.

    Parameters:
    data_file (str): The path to the cache file that needs to be validated.

    Raises:
    - ValueError:
      If the cache file does not have a "schema_version" field, or if the version does not match a valid version.
    - FileNotFoundError: If the schema file cannot be found at the specified path.
    """
    with open(data_file, "r") as f:
        data = json.load(f)

    try:
        schema_version = Version.parse(data["schema_version"])
        if schema_version not in VERSIONS:
            raise ValueError(f"Schema version '{data['schema_version']}' is not a valid version.")

        schema_path = get_schema_path(data["schema_version"])
    except KeyError:
        raise ValueError(f"Schema version '{data['schema_version']}' not found in the data.")

    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Schema file not found at path '{schema_path}': {e}")

    jsonschema.validate(instance=data, schema=schema, format_checker=get_format_checker())
