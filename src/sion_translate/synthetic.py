# Synthetic markers are read from arbitrary JSON record mappings.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_SYNTHETIC_PREFIXES: tuple[str, ...] = (
    "bt_",
    "queue_bt_",
    "concat_",
    "revise_",
    "synthetic_",
)
DEFAULT_SYNTHETIC_SAMPLING_WEIGHT = 0.5
_UNSAFE_PREFIX_CHARACTERS = frozenset('/\\<>:"|?*')


def _validate_synthetic_prefix(prefix: object) -> str:
    value = str(prefix)
    if not value:
        raise ValueError("synthetic prefixes must be non-empty")
    if (
        value in {".", ".."}
        or value != value.strip()
        or value.endswith(".")
        or any(character in _UNSAFE_PREFIX_CHARACTERS for character in value)
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(
            "synthetic prefixes must be safe filename prefixes without path "
            f"separators or reserved characters; got {value!r}"
        )
    return value


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
        value = _validate_synthetic_prefix(prefix)
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
