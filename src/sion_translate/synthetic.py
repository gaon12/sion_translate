# Synthetic markers are read from arbitrary JSON record mappings.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_SYNTHETIC_PREFIXES: tuple[str, ...] = (
    "bt_",
    "concat_",
    "revise_",
    "synthetic_",
)
DEFAULT_SYNTHETIC_SAMPLING_WEIGHT = 0.5


def normalize_synthetic_prefixes(
    prefixes: Iterable[str] = (),
    *,
    legacy_prefix: str | None = None,
) -> tuple[str, ...]:
    """Return the built-in and configured synthetic prefixes without duplicates."""

    values: list[str] = list(DEFAULT_SYNTHETIC_PREFIXES)
    if legacy_prefix is not None:
        values.append(legacy_prefix)
    values.extend(prefixes)
    normalized: list[str] = []
    seen: set[str] = set()
    for prefix in values:
        value = str(prefix)
        if not value:
            raise ValueError("synthetic prefixes must be non-empty")
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return tuple(normalized)


def synthetic_path(path: str | Path, prefixes: Iterable[str]) -> bool:
    return Path(path).name.startswith(tuple(prefixes))


def synthetic_record(row: Any) -> bool:
    """Recognize an explicit JSON boolean, including common metadata nesting."""

    if not isinstance(row, Mapping):
        return False
    if row.get("synthetic") is True:
        return True
    metadata = row.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("synthetic") is True
