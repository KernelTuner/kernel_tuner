"""Module containing paths within Kernel Tuner."""

from pathlib import Path
from typing import Union

from semver import Version

import kernel_tuner

PROJECT_DIR = Path(kernel_tuner.__file__).parent
SCHEMA_DIR = PROJECT_DIR / "schema"
CACHE_SCHEMAS_DIR = SCHEMA_DIR / "cache"


def get_schema_path(version: Union[Version, str]):
    """Returns the path to the schema of the cache of a specific version."""
    return CACHE_SCHEMAS_DIR / str(version) / "schema.json"
