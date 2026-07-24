"""Shared state values for cache freshness and processing outcomes."""

from enum import StrEnum


class CacheState(StrEnum):
    """Validity of a persisted artifact against its current inputs."""

    MISSING = "missing"
    CURRENT = "current"
    STALE = "stale"
    INVALID = "invalid"


class SyncState(StrEnum):
    """Outcome of synchronising or generating a derived artifact."""

    MISSING = "missing"
    CURRENT = "current"
    INDEXED = "indexed"
    GENERATED = "generated"
    WARNING = "warning"
