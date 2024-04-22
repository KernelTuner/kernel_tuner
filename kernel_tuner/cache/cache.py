"""Provides utilities for reading and writing cache files."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from semver import Version
import json

import kernel_tuner
from .json import CacheFileJSON


PROJECT_DIR = Path(kernel_tuner.__file__).parent
SCHEMA_DIR = PROJECT_DIR / "schema"
CACHE_SCHEMAS_DIR = SCHEMA_DIR / "cache"

SORTED_VERSIONS: list[Version] = sorted(Version.parse(p.name) for p in CACHE_SCHEMAS_DIR.iterdir())
VERSIONS: list[Version] = SORTED_VERSIONS
LATEST_VERSION: Version = VERSIONS[-1]


def get_schema_path(version: Version):
    """Returns the path to the schema of the cache of a specific version."""
    return CACHE_SCHEMAS_DIR / str(version)


@dataclass
class CacheHeader:
    """Header of the cache file."""

    device_name: str
    kernel_name: str
    problem_size: list[int]
    tune_params_keys: list[str]
    tune_params: dict[str, list]
    objective: str


class Cache:
    """Interface for writing and reading cache files."""

    @classmethod
    def create(
        cls,
        filename: PathLike,
        *,
        device_name: str,
        kernel_name: str,
        problem_size: Any,
        tune_params_keys: list[str],
        tune_params: dict[str, list],
        objective: str,
    ):
        """Creates a new cache file."""
        if not isinstance(device_name, str):
            raise ValueError("Argument device_name should be a string")
        if not isinstance(kernel_name, str):
            raise ValueError("Argument kernel_name should be a string")
        if not isinstance(tune_params_keys, list) and not all(isinstance(key, str) for key in tune_params_keys):
            raise ValueError("Argument tune_params_keys should be a list of strings")
        if not isinstance(tune_params, dict) or not all(
            isinstance(key, str) and isinstance(value, list) for key, value in tune_params.items()
        ):
            raise ValueError(
                "Argument tune_params should be a dict with:\n"
                "- keys being parameter keys (and all parameter keys should be used)\n"
                "- values being the parameter's list of possible values"
            )
        if not isinstance(objective, str):
            raise ValueError("Expected objective to be a string")
        if set(tune_params_keys) != set(tune_params.keys()):
            raise ValueError("Expected tune_params to have exactly the same keys as in the list tune_params_keys")

        cache_json = {
            "version": str(LATEST_VERSION),
            "device_name": device_name,
            "kernel_name": kernel_name,
            "problem_size": problem_size,
            "tune_params_keys": tune_params_keys,
            "tune_params": tune_params,
            "objective": objective,
            "cache": {},
        }

        with open(filename, "w") as file:
            json.dump(cache_json, file)
