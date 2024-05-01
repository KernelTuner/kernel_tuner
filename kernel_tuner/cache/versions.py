"""Versions of the cache files."""

from __future__ import annotations

from semver import Version

from .paths import CACHE_SCHEMAS_DIR

SORTED_VERSIONS: list[Version] = sorted(Version.parse(p.name) for p in CACHE_SCHEMAS_DIR.iterdir())
VERSIONS: list[Version] = SORTED_VERSIONS
LATEST_VERSION: Version = VERSIONS[-1]
