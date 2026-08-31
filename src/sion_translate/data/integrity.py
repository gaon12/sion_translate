"""Authenticated inventories for generated indexed-dataset payloads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

from sion_translate.fingerprint import file_sha256

DATASET_ARTIFACT_INVENTORY_SCHEMA = "sion-indexed-artifact-inventory-v1"
_PAYLOAD_ROOTS = frozenset({"train", "validation", "test", "refinement_evidence"})


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_digest(value: object, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"dataset artifact {path} has an invalid SHA-256 digest")
    return value


def _payload_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for payload_root_name in sorted(_PAYLOAD_ROOTS):
        payload_root = root / payload_root_name
        if not payload_root.exists():
            continue
        if payload_root.is_symlink() or not payload_root.is_dir():
            raise ValueError(f"dataset payload root is not a regular directory: {payload_root}")
        for candidate in sorted(payload_root.rglob("*")):
            if candidate.is_symlink():
                raise ValueError(f"dataset payload cannot be a symbolic link: {candidate}")
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise ValueError(f"dataset payload is not a regular file: {candidate}")
            relative = candidate.relative_to(root).as_posix()
            files[relative] = candidate
    if not files:
        raise ValueError(f"dataset has no indexed payload files: {root}")
    return files


def _stat_snapshot(files: Mapping[str, Path]) -> tuple[tuple[str, int, int, int], ...]:
    snapshot: list[tuple[str, int, int, int]] = []
    for relative, path in sorted(files.items()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"dataset payload is no longer a regular file: {path}")
        stat = path.stat()
        snapshot.append(
            (
                relative,
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(getattr(stat, "st_ino", 0)),
            )
        )
    return tuple(snapshot)


def build_dataset_artifact_inventory(root: str | Path) -> dict[str, Any]:
    """Hash every generated shard and sidecar below the indexed split roots."""

    dataset_root = Path(root)
    files = _payload_files(dataset_root)
    before = _stat_snapshot(files)
    entries = [
        {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for relative, path in sorted(files.items())
    ]
    if _stat_snapshot(files) != before:
        raise RuntimeError("dataset payload changed while its artifact inventory was built")
    return {
        "schema": DATASET_ARTIFACT_INVENTORY_SCHEMA,
        "files": entries,
    }


def _validated_inventory(raw_inventory: object) -> dict[str, tuple[int, str]]:
    if not isinstance(raw_inventory, Mapping):
        raise ValueError("dataset manifest has no artifact inventory")
    inventory = cast(Mapping[object, object], raw_inventory)
    if inventory.get("schema") != DATASET_ARTIFACT_INVENTORY_SCHEMA:
        raise ValueError("dataset artifact inventory has an unsupported schema")
    if set(inventory) != {"schema", "files"}:
        raise ValueError("dataset artifact inventory has unexpected fields")
    raw_files = inventory.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("dataset artifact inventory has no files")
    files: dict[str, tuple[int, str]] = {}
    for raw_entry in cast(list[object], raw_files):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("dataset artifact inventory contains a non-object entry")
        entry = cast(Mapping[object, object], raw_entry)
        if set(entry) != {"path", "size", "sha256"}:
            raise ValueError("dataset artifact inventory entry has unexpected fields")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            raise ValueError("dataset artifact inventory path must be a string")
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] not in _PAYLOAD_ROOTS
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.as_posix() != raw_path
        ):
            raise ValueError(f"dataset artifact inventory path is unsafe: {raw_path!r}")
        raw_size = entry.get("size")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
            raise ValueError(f"dataset artifact {raw_path} has an invalid size")
        if raw_path in files:
            raise ValueError(f"dataset artifact inventory path is duplicated: {raw_path}")
        files[raw_path] = (
            raw_size,
            _validate_digest(entry.get("sha256"), path=raw_path),
        )
    return files


def validate_dataset_artifact_inventory(
    root: str | Path,
    manifest: Mapping[str, Any] | None = None,
    *,
    require_manifest: bool = True,
) -> str | None:
    """Verify payload bytes and return the canonical inventory digest.

    The payload is hashed on every call. File metadata is not an authentication
    token: on Windows and some network filesystems an in-place same-length write
    can retain every observable stat field. Callers that already performed a
    coordinated verification may explicitly skip a redundant constructor check.
    """

    dataset_root = Path(root)
    if manifest is None:
        manifest_path = dataset_root / "manifest.json"
        if not manifest_path.is_file():
            if require_manifest:
                raise FileNotFoundError(f"dataset manifest is missing: {manifest_path}")
            return None
        try:
            raw_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"dataset manifest cannot be read: {manifest_path}") from error
        if not isinstance(raw_manifest, Mapping):
            raise ValueError("dataset manifest must be a JSON object")
        manifest = cast(Mapping[str, Any], raw_manifest)
    if manifest.get("artifact_inventory") is None:
        authenticated_format = manifest.get("format") in {
            "sion-indexed-parallel-v6",
            "sion-foundation-indexed-v2",
            "sion-foundation-indexed-v3",
        }
        if not require_manifest and not authenticated_format:
            return None
    stored = _validated_inventory(manifest.get("artifact_inventory"))
    inventory_digest = hashlib.sha256(
        _canonical_json(manifest.get("artifact_inventory"))
    ).hexdigest()
    actual_files = _payload_files(dataset_root)
    if set(actual_files) != set(stored):
        missing = sorted(set(stored) - set(actual_files))
        unexpected = sorted(set(actual_files) - set(stored))
        raise ValueError(
            "dataset artifact file set differs from its manifest "
            f"(missing={missing[:8]}, unexpected={unexpected[:8]})"
        )
    snapshot = _stat_snapshot(actual_files)
    for relative, path in sorted(actual_files.items()):
        expected_size, expected_sha256 = stored[relative]
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"dataset artifact size mismatch for {relative}: {actual_size} != {expected_size}"
            )
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"dataset artifact SHA-256 mismatch for {relative}")
    verified_snapshot = _stat_snapshot(actual_files)
    if verified_snapshot != snapshot:
        raise RuntimeError("dataset payload changed while its artifact inventory was verified")
    return inventory_digest


def dataset_artifact_problem(root: str | Path) -> str | None:
    """Return a rebuild reason for a missing, legacy, or corrupted payload."""

    try:
        validate_dataset_artifact_inventory(root)
    except (OSError, RuntimeError, ValueError) as error:
        return str(error)
    return None


__all__ = [
    "DATASET_ARTIFACT_INVENTORY_SCHEMA",
    "build_dataset_artifact_inventory",
    "dataset_artifact_problem",
    "validate_dataset_artifact_inventory",
]
