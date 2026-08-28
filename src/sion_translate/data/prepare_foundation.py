"""Convert monolingual and structured-reasoning corpora into foundation shards.

The output exposes the same indexed-shard interface as a parallel dataset. For a
denoising example, the corrupted sentence is the encoder input and the original
sentence is the decoder target. The collator creates fresh corruption for every batch,
which varies masked spans across epochs instead of fixing one corrupted copy on disk.
Foundation format v3 authenticates that a denoising target equals its source and stores
those token bytes only once; the indexed reader reconstructs the logical target. A
structured-reasoning example has an independent target and stores both token streams.

Two fields distinguish denoising records from ordinary parallel records:

- ``forward_only=True`` prevents bidirectional expansion. Both logical directions are
  identical here, so expansion would train every sentence twice.
- ``src_language == tgt_language`` tells the collator to select the corresponding
  ``<denoise_xx>`` task tag.

A file named ``reasoning_*.jsonl`` is not interpreted as ordinary ``text`` for
reconstruction. Its ``prompt`` becomes the encoder input, while delimited ``think`` and
``answer`` fields become the decoder target. The first source token stores
``<reason_xx>``; the collator moves it into the task prefix so reasoning rows survive
even when denoising is configured for every ordinary monolingual row.
"""

# Foundation preparation aggregates dynamic worker result payloads.
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import shutil
import sqlite3
import stat
import tempfile
import unicodedata
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, cast

import numpy as np

from sion_translate.artifacts import FOUNDATION_RELEASE_NAME
from sion_translate.data.monolingual import (
    DEFAULT_LANGUAGE_SAMPLING_ALPHA,
    MonolingualDiscovery,
    MonolingualSource,
    assess_language_balance,
    discover_monolingual_sources,
    segment_text,
)
from sion_translate.data.prepare import SHARED_TARGET_INDEX_DTYPE, ShardWriter, infer_register
from sion_translate.data.integrity import (
    build_dataset_artifact_inventory,
    dataset_artifact_problem,
)
from sion_translate.data.reasoning import (
    ReasoningDataError,
    ReasoningRecord,
    is_reasoning_jsonl,
    parse_reasoning_row,
    serialize_reasoning_record,
)
from sion_translate.fingerprint import file_sha256
from sion_translate.locking import _exclusive_lock  # pyright: ignore[reportPrivateUsage]
from sion_translate.splitting import choose_split_for_key
from sion_translate.tokenizer import SionTokenizer, normalize_text

FOUNDATION_INDEX_FORMAT = "sion-foundation-indexed-v3"
FOUNDATION_PREPROCESSING_SCHEMA = "foundation-mixed-objectives-v6"
FOUNDATION_SOURCE_IDENTITY_SCHEMA = "corpus-relative-posix-sha256-v1"
FOUNDATION_TOKENIZER_IDENTITY_SCHEMA = "content-sha256-v1"
FOUNDATION_DEDUPLICATION_BACKEND = "sqlite-blake2b-128-v1"
_DEDUPLICATION_CACHE_KIB = 8 * 1024
_FOUNDATION_RESUME_DATABASE = ".foundation-resume.sqlite3"
_FOUNDATION_RESUME_SCHEMA = "sion-foundation-resume-v1"
_FOUNDATION_RESUME_USER_VERSION = 1
_FOUNDATION_OUTPUT_LOCK_SCHEMA = "sion-foundation-output-lock-v1"
# Checkpointing is based on physical input lines, including rejected rows. The
# interval is deliberately independent of record acceptance, segmentation, and
# language so the same inputs always produce the same shard boundaries.
_FOUNDATION_CHECKPOINT_INTERVAL = 100_000

# Capacity planning samples physical lines at deterministic byte offsets and
# measures their actual token/index/dedup footprint.  The margin covers source
# regions missed by the sample, SQLite page variance, manifests, and filesystem
# allocation.  A streaming reserve check still protects against estimator drift.
_FOUNDATION_SAMPLE_LINES_PER_SOURCE = 64
_FOUNDATION_SAMPLE_MAX_LINE_BYTES = 256 * 1024
_FOUNDATION_ESTIMATE_MARGIN_NUMERATOR = 3
_FOUNDATION_ESTIMATE_MARGIN_DENOMINATOR = 2
_FOUNDATION_MIN_ESTIMATED_BYTES_PER_SOURCE_BYTE = 1
_FOUNDATION_DEDUP_BYTES_PER_RECORD = 32
_FOUNDATION_SPACE_RESERVE_BYTES = 512 * 1024 * 1024
_FOUNDATION_SPACE_RESERVE_DIVISOR = 50
_FOUNDATION_EMPTY_DEDUP_SHA256 = hashlib.sha256(b"").hexdigest()


@dataclass
class LanguageStats:
    lines_read: int = 0
    accepted: int = 0
    too_short: int = 0
    # Rows discarded for exceeding the limit. Segmentation should keep this at zero.
    too_long: int = 0
    # Documents split into multiple segments and the resulting segment count.
    segmented_documents: int = 0
    segments: int = 0
    duplicate: int = 0
    empty_after_tokenization: int = 0
    reasoning_records: int = 0
    reasoning_rejected: int = 0
    reasoning_prompt_truncated: int = 0
    reasoning_think_truncated: int = 0
    reasoning_answer_truncated: int = 0
    read_rejects: dict[str, int] = field(default_factory=dict)


@dataclass
class FoundationPrepareStats:
    languages: dict[str, LanguageStats] = field(default_factory=dict)
    train_records: int = 0
    validation_records: int = 0

    @property
    def total_records(self) -> int:
        return self.train_records + self.validation_records

    def accepted_per_language(self) -> dict[str, int]:
        return {language: stats.accepted for language, stats in self.languages.items()}


@dataclass(frozen=True)
class _FileSnapshot:
    """Identity of one input before it is allowed to influence an artifact."""

    resolved_path: str
    size_bytes: int
    sha256: str
    modified_ns: int
    changed_ns: int
    device: int
    inode: int
    file_attributes: int


@dataclass(frozen=True)
class _FoundationCursor:
    """The next source position after one durably committed physical line."""

    source_id: int
    physical_lines: int
    total_physical_lines: int


@dataclass(frozen=True)
class _FoundationResumeState:
    """Authenticated state committed in the same transaction as deduplication."""

    checkpoint_sequence: int
    cursor: _FoundationCursor
    next_shard_indices: dict[str, int]
    stats: FoundationPrepareStats
    source_record_counts: tuple[int, ...]
    artifact_inventory: tuple[dict[str, object], ...]
    dedup_digest_count: int
    dedup_sequence_sha256: str


@dataclass(frozen=True)
class _PartialFoundationGeneration:
    path: Path
    state: _FoundationResumeState
    journal: _FoundationResumeJournal


@dataclass(frozen=True)
class _StagingRecovery:
    complete_stats: FoundationPrepareStats | None = None
    partial: _PartialFoundationGeneration | None = None


def _canonical_json(value: object) -> str:
    """Serialize resume authority without platform- or locale-dependent details."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stats_payload(stats: FoundationPrepareStats) -> dict[str, object]:
    return {
        "train_records": stats.train_records,
        "validation_records": stats.validation_records,
        "languages": {
            language: asdict(language_stats) for language, language_stats in stats.languages.items()
        },
    }


def _resume_state_payload(
    state: _FoundationResumeState,
    *,
    contract_sha256: str,
) -> dict[str, object]:
    return {
        "schema": _FOUNDATION_RESUME_SCHEMA,
        "contract_sha256": contract_sha256,
        "checkpoint_sequence": state.checkpoint_sequence,
        "cursor": {
            "source_id": state.cursor.source_id,
            "physical_lines": state.cursor.physical_lines,
            "total_physical_lines": state.cursor.total_physical_lines,
        },
        "next_shard_indices": dict(state.next_shard_indices),
        "stats": _stats_payload(state.stats),
        "source_record_counts": list(state.source_record_counts),
        "artifact_inventory": list(state.artifact_inventory),
        "dedup_digest_count": state.dedup_digest_count,
        "dedup_sequence_sha256": state.dedup_sequence_sha256,
    }


def _nonnegative_resume_integer(payload: Mapping[object, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"foundation resume {name} must be a non-negative integer")
    return value


def _validated_resume_inventory(raw_inventory: object) -> tuple[dict[str, object], ...]:
    if not isinstance(raw_inventory, list):
        raise ValueError("foundation resume artifact_inventory must be a list")
    inventory: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_entry in cast(list[object], raw_inventory):
        if not isinstance(raw_entry, dict):
            raise ValueError("foundation resume inventory entry must be an object")
        entry = cast(dict[object, object], raw_entry)
        if set(entry) != {"path", "size", "sha256"}:
            raise ValueError("foundation resume inventory entry has unexpected fields")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            raise ValueError("foundation resume inventory path must be a string")
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] not in {"train", "validation"}
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.as_posix() != raw_path
        ):
            raise ValueError(f"foundation resume inventory path is unsafe: {raw_path!r}")
        if raw_path in seen:
            raise ValueError(f"foundation resume inventory path is duplicated: {raw_path}")
        size = entry.get("size")
        digest = entry.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"foundation resume inventory size is invalid: {raw_path}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"foundation resume inventory digest is invalid: {raw_path}")
        seen.add(raw_path)
        inventory.append({"path": raw_path, "size": size, "sha256": digest})
    if [cast(str, entry["path"]) for entry in inventory] != sorted(seen):
        raise ValueError("foundation resume inventory must be sorted by path")
    return tuple(inventory)


def _parse_resume_state(
    raw_payload: object,
    *,
    contract_sha256: str,
    languages: tuple[str, ...],
    source_count: int,
) -> _FoundationResumeState:
    if not isinstance(raw_payload, dict):
        raise ValueError("foundation resume state must be an object")
    payload = cast(dict[object, object], raw_payload)
    if set(payload) != {
        "schema",
        "contract_sha256",
        "checkpoint_sequence",
        "cursor",
        "next_shard_indices",
        "stats",
        "source_record_counts",
        "artifact_inventory",
        "dedup_digest_count",
        "dedup_sequence_sha256",
    }:
        raise ValueError("foundation resume state has unexpected fields")
    if payload.get("schema") != _FOUNDATION_RESUME_SCHEMA:
        raise ValueError("foundation resume state schema changed")
    if payload.get("contract_sha256") != contract_sha256:
        raise ValueError("foundation resume state belongs to a different preparation contract")
    checkpoint_sequence = _nonnegative_resume_integer(payload, "checkpoint_sequence")

    raw_cursor = payload.get("cursor")
    if not isinstance(raw_cursor, dict):
        raise ValueError("foundation resume cursor is invalid")
    cursor_payload = cast(dict[object, object], raw_cursor)
    if set(cursor_payload) != {
        "source_id",
        "physical_lines",
        "total_physical_lines",
    }:
        raise ValueError("foundation resume cursor is invalid")
    cursor = _FoundationCursor(
        source_id=_nonnegative_resume_integer(cursor_payload, "source_id"),
        physical_lines=_nonnegative_resume_integer(cursor_payload, "physical_lines"),
        total_physical_lines=_nonnegative_resume_integer(
            cursor_payload,
            "total_physical_lines",
        ),
    )
    if cursor.source_id > source_count or (
        cursor.source_id == source_count and cursor.physical_lines != 0
    ):
        raise ValueError("foundation resume cursor is outside the ordered source list")
    if cursor.physical_lines > cursor.total_physical_lines:
        raise ValueError("foundation resume cursor line counts are inconsistent")
    if cursor.source_id < source_count:
        if checkpoint_sequence == 0:
            if cursor != _FoundationCursor(0, 0, 0):
                raise ValueError("foundation initial resume cursor is invalid")
        elif (
            cursor.physical_lines == 0
            or cursor.total_physical_lines % _FOUNDATION_CHECKPOINT_INTERVAL != 0
            or checkpoint_sequence != cursor.total_physical_lines // _FOUNDATION_CHECKPOINT_INTERVAL
        ):
            raise ValueError("foundation resume cursor is not a deterministic checkpoint")
    elif checkpoint_sequence != (
        cursor.total_physical_lines // _FOUNDATION_CHECKPOINT_INTERVAL + 1
    ):
        raise ValueError("foundation final resume checkpoint sequence is invalid")

    raw_indices = payload.get("next_shard_indices")
    if not isinstance(raw_indices, dict):
        raise ValueError("foundation resume shard cursor is invalid")
    indices_payload = cast(dict[object, object], raw_indices)
    if set(indices_payload) != {"train", "validation"}:
        raise ValueError("foundation resume shard cursor is invalid")
    next_shard_indices = {
        split: _nonnegative_resume_integer(indices_payload, split)
        for split in ("train", "validation")
    }

    raw_stats = payload.get("stats")
    stats = _stats_from_manifest(
        {
            "languages": list(languages),
            "stats": raw_stats,
        }
    )
    if tuple(stats.languages) != languages:
        raise ValueError("foundation resume language order changed")

    raw_source_counts = payload.get("source_record_counts")
    if not isinstance(raw_source_counts, list):
        raise ValueError("foundation resume source_record_counts is invalid")
    source_count_values = cast(list[object], raw_source_counts)
    if len(source_count_values) != source_count:
        raise ValueError("foundation resume source_record_counts is invalid")
    source_record_counts = tuple(
        _nonnegative_resume_integer({"count": value}, "count") for value in source_count_values
    )
    if sum(source_record_counts) != stats.total_records:
        raise ValueError("foundation resume source counts differ from its statistics")
    artifact_inventory = _validated_resume_inventory(payload.get("artifact_inventory"))
    dedup_digest_count = _nonnegative_resume_integer(payload, "dedup_digest_count")
    raw_dedup_sha256 = payload.get("dedup_sequence_sha256")
    if (
        not isinstance(raw_dedup_sha256, str)
        or len(raw_dedup_sha256) != 64
        or raw_dedup_sha256 != raw_dedup_sha256.lower()
        or any(character not in "0123456789abcdef" for character in raw_dedup_sha256)
    ):
        raise ValueError("foundation resume deduplication digest is invalid")
    if dedup_digest_count == 0 and raw_dedup_sha256 != _FOUNDATION_EMPTY_DEDUP_SHA256:
        raise ValueError("foundation empty deduplication digest is invalid")
    if checkpoint_sequence == 0:
        empty_stats = FoundationPrepareStats(
            languages={language: LanguageStats() for language in languages}
        )
        if (
            next_shard_indices != {"train": 0, "validation": 0}
            or stats != empty_stats
            or any(source_record_counts)
            or artifact_inventory
            or dedup_digest_count
            or raw_dedup_sha256 != _FOUNDATION_EMPTY_DEDUP_SHA256
        ):
            raise ValueError("foundation initial resume state is not empty")

    return _FoundationResumeState(
        checkpoint_sequence=checkpoint_sequence,
        cursor=cursor,
        next_shard_indices=next_shard_indices,
        stats=stats,
        source_record_counts=source_record_counts,
        artifact_inventory=artifact_inventory,
        dedup_digest_count=dedup_digest_count,
        dedup_sequence_sha256=raw_dedup_sha256,
    )


def _logical_relative_path(root: Path, path: Path, *, role: str) -> str:
    """Return a portable POSIX path that is confined below ``root``."""

    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        relative = resolved_path.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise ValueError(f"foundation {role} must stay below the corpus root: {path}") from error
    if not relative.parts:
        return "."
    logical_path = PurePosixPath(*relative.parts).as_posix()
    if logical_path.startswith("/") or ".." in PurePosixPath(logical_path).parts:
        raise ValueError(f"foundation {role} has an unsafe logical path: {path}")
    return logical_path


def _source_logical_path(discovery: MonolingualDiscovery, path: Path) -> str:
    return _logical_relative_path(discovery.root, path, role="source")


def _source_identity_digest(records: list[dict[str, object]]) -> str:
    """Hash the ordered, path-portable source identity records."""

    identity_records = [
        {
            "language": record["language"],
            "logical_path": record["logical_path"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "task": record["task"],
        }
        for record in records
    ]
    payload = json.dumps(
        identity_records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_digest(language: str, text: str) -> bytes:
    return hashlib.blake2b(f"{language}\0{text}".encode("utf-8"), digest_size=16).digest()


def _reasoning_digest(language: str, prompt: str, think: str, answer: str) -> bytes:
    payload = f"reasoning\0{language}\0{prompt}\0{think}\0{answer}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()


def _extend_dedup_sequence_sha256(previous_sha256: str, digest: bytes) -> str:
    """Bind one unique digest to its deterministic insertion position."""

    if len(digest) != 16:
        raise ValueError("foundation deduplication digest must contain 16 bytes")
    value = hashlib.sha256()
    value.update(bytes.fromhex(previous_sha256))
    value.update(digest)
    return value.hexdigest()


def _is_usable(text: str) -> bool:
    """Reject a line that contains only control or otherwise invisible characters."""

    return any(not unicodedata.category(char).startswith("C") for char in text)


def _source_sha256(path: Path) -> str:
    """Hash one raw source while preserving the path in read failures."""

    try:
        return file_sha256(path)
    except OSError as error:
        raise OSError(f"Cannot read the foundation source hash: {path}: {error}") from error


def _is_reparse_stat(value: os.stat_result) -> bool:
    """Return whether an lstat result identifies a link or Windows reparse point."""

    return bool(
        stat.S_ISLNK(value.st_mode)
        or getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _assert_no_reparse_components(path: Path, *, role: str) -> None:
    """Reject links and junctions in every existing component of ``path``."""

    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        current /= part
        if not _path_exists(current):
            continue
        try:
            current_stat = os.lstat(current)
        except OSError as error:
            raise OSError(f"Cannot inspect foundation {role} path: {current}") from error
        if _is_reparse_stat(current_stat):
            raise ValueError(
                f"Foundation {role} cannot traverse a symlink or reparse point: {current}"
            )


def _regular_file_stat(path: Path, *, role: str) -> os.stat_result:
    _assert_no_reparse_components(path, role=role)
    try:
        value = os.lstat(path)
    except OSError as error:
        raise OSError(f"Cannot inspect foundation {role}: {path}") from error
    if _is_reparse_stat(value) or not stat.S_ISREG(value.st_mode):
        raise ValueError(f"Foundation {role} must be a regular file without reparse points: {path}")
    return value


def _assert_regular_directory(path: Path, *, role: str) -> None:
    _assert_no_reparse_components(path, role=role)
    try:
        value = os.lstat(path)
    except OSError as error:
        raise OSError(f"Cannot inspect foundation {role}: {path}") from error
    if _is_reparse_stat(value) or not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"Foundation {role} must be a regular directory: {path}")


def _file_snapshot(path: Path, *, role: str) -> _FileSnapshot:
    """Capture a stable path/size/hash snapshot, rejecting a hash-time mutation."""

    try:
        stat_before = _regular_file_stat(path, role=role)
        resolved_before = path.resolve(strict=True)
        content_hash = file_sha256(path)
        resolved_after = path.resolve(strict=True)
        stat_after = _regular_file_stat(path, role=role)
    except (OSError, ValueError) as error:
        raise OSError(f"Cannot read the foundation {role} snapshot: {path}: {error}") from error

    identity_before = (
        str(resolved_before),
        stat_before.st_size,
        stat_before.st_mtime_ns,
        stat_before.st_ctime_ns,
        stat_before.st_dev,
        stat_before.st_ino,
        getattr(stat_before, "st_file_attributes", 0),
    )
    identity_after = (
        str(resolved_after),
        stat_after.st_size,
        stat_after.st_mtime_ns,
        stat_after.st_ctime_ns,
        stat_after.st_dev,
        stat_after.st_ino,
        getattr(stat_after, "st_file_attributes", 0),
    )
    if identity_before != identity_after:
        raise RuntimeError(f"Foundation {role} changed while its snapshot was created: {path}")
    return _FileSnapshot(
        resolved_path=str(resolved_after),
        size_bytes=stat_after.st_size,
        sha256=content_hash,
        modified_ns=stat_after.st_mtime_ns,
        changed_ns=stat_after.st_ctime_ns,
        device=stat_after.st_dev,
        inode=stat_after.st_ino,
        file_attributes=getattr(stat_after, "st_file_attributes", 0),
    )


def _source_snapshot(path: Path) -> _FileSnapshot:
    return _file_snapshot(path, role="source file")


def _tokenizer_snapshot(path: Path) -> _FileSnapshot:
    return _file_snapshot(path, role="tokenizer")


def _capture_source_snapshots(
    discovery: MonolingualDiscovery,
) -> tuple[_FileSnapshot, ...]:
    snapshots: list[_FileSnapshot] = []
    for source in discovery.sources:
        snapshot = _source_snapshot(source.path)
        if snapshot.size_bytes != source.size_bytes:
            raise RuntimeError(
                "Foundation source size changed after discovery: "
                f"{source.path} ({source.size_bytes} -> {snapshot.size_bytes} bytes)"
            )
        snapshots.append(snapshot)
    return tuple(snapshots)


def _build_resume_contract(
    discovery: MonolingualDiscovery,
    source_snapshots: tuple[_FileSnapshot, ...],
    tokenizer_snapshot: _FileSnapshot,
    *,
    minimum_characters: int,
    maximum_characters: int,
    max_tokens: int,
    max_target_tokens: int,
    deduplicate: bool,
    shard_size: int,
    validation_fraction: float,
    language_sampling_alpha: float,
    minimum_language_share: float,
    reasoning_sample_share: float,
    release_name: str,
) -> dict[str, object]:
    """Bind resumable bytes to every input that can change deterministic output."""

    if len(source_snapshots) != len(discovery.sources):
        raise ValueError("foundation resume source snapshot count is invalid")
    ordered_sources: list[dict[str, object]] = [
        {
            "id": source_id,
            "language": source.language,
            "logical_path": _source_logical_path(discovery, source.path),
            "size_bytes": snapshot.size_bytes,
            "sha256": snapshot.sha256,
            "task": "reasoning" if is_reasoning_jsonl(source.path) else "denoising",
        }
        for source_id, (source, snapshot) in enumerate(
            zip(discovery.sources, source_snapshots, strict=True)
        )
    ]
    skipped_entries = [
        {
            "logical_path": _logical_relative_path(
                discovery.root,
                entry.path,
                role="skipped entry",
            ),
            "reason": entry.reason,
        }
        for entry in discovery.skipped
    ]
    return {
        "schema": _FOUNDATION_RESUME_SCHEMA,
        "format": FOUNDATION_INDEX_FORMAT,
        "preprocessing_schema": FOUNDATION_PREPROCESSING_SCHEMA,
        "source_identity_schema": FOUNDATION_SOURCE_IDENTITY_SCHEMA,
        "tokenizer_identity_schema": FOUNDATION_TOKENIZER_IDENTITY_SCHEMA,
        "release_name": release_name,
        "languages": list(discovery.languages),
        "languages_without_data": list(discovery.languages_without_data),
        "unconfigured_languages": list(discovery.unconfigured_languages),
        "sources": ordered_sources,
        "sources_sha256": _source_identity_digest(ordered_sources),
        "skipped": skipped_entries,
        "tokenizer_identity": {
            "schema": FOUNDATION_TOKENIZER_IDENTITY_SCHEMA,
            "size_bytes": tokenizer_snapshot.size_bytes,
            "sha256": tokenizer_snapshot.sha256,
        },
        "storage": {
            "index_dtype": json.loads(json.dumps(SHARED_TARGET_INDEX_DTYPE.descr)),
            "target_storage": "row-shared-source-v1",
        },
        "options": {
            "checkpoint_interval_physical_lines": _FOUNDATION_CHECKPOINT_INTERVAL,
            "deduplicate": deduplicate,
            "deduplication_backend": (
                FOUNDATION_DEDUPLICATION_BACKEND if deduplicate else "disabled"
            ),
            "language_sampling_alpha": language_sampling_alpha,
            "maximum_characters": maximum_characters,
            "max_tokens": max_tokens,
            "max_target_tokens": max_target_tokens,
            "minimum_characters": minimum_characters,
            "minimum_language_share": minimum_language_share,
            "reasoning_sample_share": reasoning_sample_share,
            "shard_size": shard_size,
            "validation_fraction": validation_fraction,
        },
    }


def _verify_source_metadata(
    discovery: MonolingualDiscovery,
    expected: tuple[_FileSnapshot, ...],
) -> None:
    configured_languages = tuple(
        dict.fromkeys((*discovery.languages, *discovery.languages_without_data))
    )
    rediscovered = discover_monolingual_sources(discovery.root, configured_languages)
    if len(expected) != len(rediscovered.sources):
        raise RuntimeError(
            "Foundation source list changed during preparation: "
            f"{len(expected)} -> {len(rediscovered.sources)} files"
        )
    if len(expected) != len(discovery.sources):
        raise RuntimeError("Foundation source snapshot count does not match discovery")
    expected_by_path: dict[str, tuple[str, _FileSnapshot]] = {}
    for source, snapshot in zip(discovery.sources, expected, strict=True):
        if snapshot.resolved_path in expected_by_path:
            raise RuntimeError("Foundation source paths contain a duplicate")
        expected_by_path[snapshot.resolved_path] = (source.language, snapshot)
    for source in rediscovered.sources:
        try:
            source_stat = _regular_file_stat(source.path, role="source file")
            resolved_path = source.path.resolve(strict=True)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Foundation source changed during preparation: {source.path}"
            ) from error
        expected_source = expected_by_path.get(str(resolved_path))
        if expected_source is None or expected_source[0] != source.language:
            raise RuntimeError(
                "Foundation source path or language mapping changed during preparation"
            )
        expected_snapshot = expected_source[1]
        actual_metadata = (
            str(resolved_path),
            source_stat.st_size,
            source_stat.st_mtime_ns,
            source_stat.st_ctime_ns,
            source_stat.st_dev,
            source_stat.st_ino,
            getattr(source_stat, "st_file_attributes", 0),
        )
        expected_metadata = (
            expected_snapshot.resolved_path,
            expected_snapshot.size_bytes,
            expected_snapshot.modified_ns,
            expected_snapshot.changed_ns,
            expected_snapshot.device,
            expected_snapshot.inode,
            expected_snapshot.file_attributes,
        )
        if actual_metadata != expected_metadata:
            raise RuntimeError(f"Foundation source changed during preparation: {source.path}")


def _verify_source_snapshots(
    discovery: MonolingualDiscovery,
    expected: tuple[_FileSnapshot, ...],
) -> None:
    _verify_source_metadata(discovery, expected)
    for source, expected_snapshot in zip(discovery.sources, expected, strict=True):
        try:
            actual_snapshot = _source_snapshot(source.path)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"Foundation source changed during preparation: {source.path}"
            ) from error
        if actual_snapshot != expected_snapshot:
            raise RuntimeError(f"Foundation source changed during preparation: {source.path}")


def _verify_tokenizer_snapshot(path: Path, expected: _FileSnapshot) -> None:
    try:
        actual = _tokenizer_snapshot(path)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"Foundation tokenizer changed during preparation: {path}") from error
    if actual != expected:
        raise RuntimeError(f"Foundation tokenizer changed during preparation: {path}")


def _path_exists(path: Path) -> bool:
    """Like ``Path.exists``, but also protect a dangling destination symlink."""

    return os.path.lexists(path)


def _refuse_existing_output(output_dir: Path) -> None:
    if _path_exists(output_dir):
        raise FileExistsError(
            "Output directory must not exist (an existing path is treated as not empty): "
            f"{output_dir}. Move it aside or choose another foundation.dataset_dir."
        )


def _remove_staging_path(path: Path) -> None:
    if not _path_exists(path):
        return
    _regular_staging_tree(path)
    quarantine = path.with_name(f".{path.name}.rejected-{uuid.uuid4().hex}")
    os.rename(path, quarantine)
    _fsync_directory(path.parent)
    # Revalidate after the namespace move so cleanup never follows an entry
    # that was exchanged between the first inspection and the rename.
    _regular_staging_tree(quarantine)
    shutil.rmtree(quarantine)
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    # Windows delegates fsync to _commit, which requires a writable handle.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories, files = _regular_staging_tree(root)
    for artifact in files:
        _fsync_file(artifact)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _regular_staging_tree(root: Path) -> tuple[list[Path], list[Path]]:
    """Inspect a private tree without following symlinks or Windows junctions."""

    _assert_regular_directory(root, role="staging directory")
    directories = [root]
    files: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise RuntimeError(f"Cannot inspect foundation staging: {directory}") from error
        for entry in entries:
            artifact = Path(entry.path)
            try:
                artifact_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise RuntimeError(
                    f"Cannot inspect foundation staging artifact: {artifact}"
                ) from error
            if _is_reparse_stat(artifact_stat):
                raise RuntimeError(
                    f"Foundation staging cannot contain a symlink or reparse point: {artifact}"
                )
            if stat.S_ISDIR(artifact_stat.st_mode):
                directories.append(artifact)
                pending.append(artifact)
            elif stat.S_ISREG(artifact_stat.st_mode):
                files.append(artifact)
            else:
                raise RuntimeError(
                    f"Foundation staging contains a non-regular artifact: {artifact}"
                )
    return directories, files


def _foundation_staging_size(root: Path | None) -> int:
    """Return bytes already charged to an authenticated private generation."""

    if root is None:
        return 0
    _directories, files = _regular_staging_tree(root)
    return sum(_regular_file_stat(path, role="staging artifact").st_size for path in files)


def _read_bounded_sample_line(handle: BinaryIO) -> tuple[bytes, bool]:
    """Read one physical line without allocating an unbounded document."""

    first = handle.readline(_FOUNDATION_SAMPLE_MAX_LINE_BYTES + 1)
    if not first:
        return b"", False
    truncated = len(first) > _FOUNDATION_SAMPLE_MAX_LINE_BYTES or not first.endswith(b"\n")
    if len(first) > _FOUNDATION_SAMPLE_MAX_LINE_BYTES:
        first = first[:_FOUNDATION_SAMPLE_MAX_LINE_BYTES]
    if truncated and not first.endswith(b"\n"):
        while True:
            remainder = handle.readline(_FOUNDATION_SAMPLE_MAX_LINE_BYTES)
            if not remainder or remainder.endswith(b"\n"):
                break
    return first, truncated


def _sample_source_physical_lines(path: Path, size_bytes: int) -> Iterator[bytes]:
    """Sample deterministic byte-stratified lines from one immutable source."""

    sample_count = min(_FOUNDATION_SAMPLE_LINES_PER_SOURCE, max(1, size_bytes))
    denominator = max(1, sample_count - 1)
    offsets = tuple(
        dict.fromkeys(
            (max(0, size_bytes - 1) * index) // denominator for index in range(sample_count)
        )
    )
    seen_starts: set[int] = set()
    with path.open("rb") as handle:
        for offset in offsets:
            if offset:
                handle.seek(offset - 1)
                previous = handle.read(1)
                if previous != b"\n":
                    _discarded, truncated = _read_bounded_sample_line(handle)
                    if truncated:
                        # One physical line spans this and later strata. The
                        # bounded prefix below is sufficient for density
                        # estimation without repeatedly scanning the same tail.
                        handle.seek(offset)
            else:
                handle.seek(0)
            start = handle.tell()
            if start in seen_starts:
                continue
            seen_starts.add(start)
            raw_line, truncated = _read_bounded_sample_line(handle)
            if not raw_line:
                continue
            yield raw_line
            if truncated:
                return


def _sampled_foundation_storage_bytes(
    path: Path,
    raw_line: bytes,
    *,
    language: str,
    tokenizer: SionTokenizer,
    minimum_characters: int,
    maximum_characters: int,
    max_tokens: int,
    max_target_tokens: int,
    deduplicate: bool,
) -> int:
    """Measure the shard footprint represented by one sampled physical line."""

    reasoning_source = is_reasoning_jsonl(path)
    try:
        line = raw_line.decode(
            "utf-8-sig",
            errors="strict" if reasoning_source else "replace",
        )
    except UnicodeDecodeError:
        return 0
    scratch = LanguageStats()
    per_record_overhead = SHARED_TARGET_INDEX_DTYPE.itemsize + (
        _FOUNDATION_DEDUP_BYTES_PER_RECORD if deduplicate else 0
    )
    if reasoning_source:
        record = _parse_reasoning_physical_line(
            line,
            language=language,
            language_stats=scratch,
        )
        if record is None:
            return 0
        encoded = serialize_reasoning_record(
            record,
            tokenizer,
            max_source_tokens=max_tokens + 1,
            max_target_tokens=max_target_tokens,
        )
        return 4 * (len(encoded.source_ids) + len(encoded.target_ids)) + per_record_overhead

    raw_text = _parse_monolingual_physical_line(path, line, scratch)
    if raw_text is None:
        return 0
    document = normalize_text(raw_text)
    if not _is_usable(document):
        return 0
    segments = segment_text(
        document,
        maximum_characters=maximum_characters,
        minimum_characters=minimum_characters,
    )
    measured = 0
    for segment in segments:
        token_ids = tokenizer.encode(segment)[:max_tokens]
        # Denoising aliases the target to the source, so only one uint32 token
        # stream is materialized. An empty tokenization still occupies dedup
        # state but does not create an index row.
        measured += 4 * len(token_ids)
        if token_ids:
            measured += SHARED_TARGET_INDEX_DTYPE.itemsize
        if deduplicate:
            measured += _FOUNDATION_DEDUP_BYTES_PER_RECORD
    return measured


def _estimate_foundation_generation_bytes(
    discovery: MonolingualDiscovery,
    tokenizer_model: str | Path,
    *,
    minimum_characters: int,
    maximum_characters: int,
    max_tokens: int,
    max_target_tokens: int,
    deduplicate: bool,
) -> int:
    """Estimate complete storage from deterministic, tokenizer-aware samples."""

    tokenizer = SionTokenizer(tokenizer_model)
    estimated_total = 0
    for source in discovery.sources:
        sampled_source_bytes = 0
        sampled_storage_bytes = 0
        for raw_line in _sample_source_physical_lines(source.path, source.size_bytes):
            sampled_source_bytes += len(raw_line)
            sampled_storage_bytes += _sampled_foundation_storage_bytes(
                source.path,
                raw_line,
                language=source.language,
                tokenizer=tokenizer,
                minimum_characters=minimum_characters,
                maximum_characters=maximum_characters,
                max_tokens=max_tokens,
                max_target_tokens=max_target_tokens,
                deduplicate=deduplicate,
            )
        floor = source.size_bytes * _FOUNDATION_MIN_ESTIMATED_BYTES_PER_SOURCE_BYTE
        if sampled_source_bytes:
            sampled_projection = (
                sampled_storage_bytes * source.size_bytes + sampled_source_bytes - 1
            ) // sampled_source_bytes
        else:
            sampled_projection = 0
        unbuffered = max(floor, sampled_projection)
        estimated_total += (
            unbuffered * _FOUNDATION_ESTIMATE_MARGIN_NUMERATOR
            + _FOUNDATION_ESTIMATE_MARGIN_DENOMINATOR
            - 1
        ) // _FOUNDATION_ESTIMATE_MARGIN_DENOMINATOR
    return estimated_total


def _preflight_foundation_disk_space(
    discovery: MonolingualDiscovery,
    tokenizer_model: str | Path,
    output_parent: Path,
    *,
    existing_staging: Path | None,
    minimum_characters: int,
    maximum_characters: int,
    max_tokens: int,
    max_target_tokens: int,
    deduplicate: bool,
) -> int:
    """Refuse a generation that cannot retain its estimated output and reserve.

    Existing resumable bytes are subtracted because ``disk_usage().free`` already
    excludes them. The estimate is tokenizer- and source-aware, while a 50% margin
    and an independent streaming reserve guard against distribution drift.
    """

    estimated_generation_bytes = _estimate_foundation_generation_bytes(
        discovery,
        tokenizer_model,
        minimum_characters=minimum_characters,
        maximum_characters=maximum_characters,
        max_tokens=max_tokens,
        max_target_tokens=max_target_tokens,
        deduplicate=deduplicate,
    )
    reserve_bytes = max(
        _FOUNDATION_SPACE_RESERVE_BYTES,
        estimated_generation_bytes // _FOUNDATION_SPACE_RESERVE_DIVISOR,
    )
    existing_bytes = _foundation_staging_size(existing_staging)
    additional_bytes = max(0, estimated_generation_bytes - existing_bytes)
    required_free_bytes = additional_bytes + reserve_bytes
    try:
        free_bytes = shutil.disk_usage(output_parent).free
    except OSError as error:
        raise OSError(
            error.errno,
            f"Cannot inspect free disk space for foundation preparation: {output_parent}",
            str(output_parent),
        ) from error
    if free_bytes < required_free_bytes:
        raise OSError(
            errno.ENOSPC,
            "Insufficient free disk space for foundation preparation: "
            f"need at least {required_free_bytes:,} free bytes, found {free_bytes:,}; "
            f"estimated generation {estimated_generation_bytes:,}, "
            f"existing resumable bytes {existing_bytes:,}, reserve {reserve_bytes:,}",
            str(output_parent),
        )
    return reserve_bytes


def _preflight_foundation_checkpoint_space(
    output_dir: Path,
    minimum_free_bytes: int,
) -> None:
    """Keep enough free space to close and roll back one in-flight checkpoint."""

    if minimum_free_bytes < _FOUNDATION_SPACE_RESERVE_BYTES:
        raise ValueError("foundation checkpoint reserve is below the absolute safety floor")

    try:
        free_bytes = shutil.disk_usage(output_dir).free
    except OSError as error:
        raise OSError(
            error.errno,
            f"Cannot inspect free disk space at a foundation checkpoint: {output_dir}",
            str(output_dir),
        ) from error
    if free_bytes < minimum_free_bytes:
        raise OSError(
            errno.ENOSPC,
            "Foundation preparation reached its checkpoint disk reserve: "
            f"need {minimum_free_bytes:,} free bytes, found {free_bytes:,}",
            str(output_dir),
        )


def _checkpoint_payload_files(root: Path) -> dict[str, Path]:
    """Return regular direct-child shard files without accepting extra trees."""

    files: dict[str, Path] = {}
    for split in ("train", "validation"):
        split_root = root / split
        _assert_regular_directory(split_root, role=f"{split} resume payload")
        for entry in os.scandir(split_root):
            path = Path(entry.path)
            value = entry.stat(follow_symlinks=False)
            if _is_reparse_stat(value) or not stat.S_ISREG(value.st_mode):
                raise ValueError(
                    f"Foundation resume payload must contain only regular shard files: {path}"
                )
            files[f"{split}/{entry.name}"] = path
    return files


def _build_checkpoint_inventory(
    root: Path,
    previous: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    """Extend an immutable shard inventory without rehashing its full prefix."""

    files = _checkpoint_payload_files(root)
    previous_by_path = {cast(str, entry["path"]): entry for entry in previous}
    missing = set(previous_by_path) - set(files)
    if missing:
        raise RuntimeError(
            f"Foundation committed shards disappeared while checkpointing: {sorted(missing)}"
        )
    before = {
        relative: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            getattr(path.stat(), "st_ino", 0),
        )
        for relative, path in files.items()
        if relative not in previous_by_path
    }
    inventory: list[dict[str, object]] = []
    for relative, path in sorted(files.items()):
        prior = previous_by_path.get(relative)
        if prior is not None:
            if path.stat().st_size != prior["size"]:
                raise RuntimeError(
                    f"Foundation committed shard size changed while checkpointing: {relative}"
                )
            inventory.append(dict(prior))
            continue
        inventory.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    after = {
        relative: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            getattr(path.stat(), "st_ino", 0),
        )
        for relative, path in files.items()
        if relative not in previous_by_path
    }
    if after != before:
        raise RuntimeError("Foundation shard bytes changed while checkpointing")
    return tuple(inventory)


def _inventory_shard_groups(
    inventory: tuple[dict[str, object], ...],
) -> dict[str, dict[int, set[str]]]:
    groups: dict[str, dict[int, set[str]]] = {"train": {}, "validation": {}}
    for entry in inventory:
        relative = cast(str, entry["path"])
        split, name = relative.split("/", maxsplit=1)
        prefix, separator, suffix = name.partition(".")
        if (
            not separator
            or len(prefix) != 5
            or not prefix.isdecimal()
            or suffix not in {"src.bin", "tgt.bin", "idx.npy"}
        ):
            raise ValueError(f"foundation resume shard name is invalid: {relative}")
        groups[split].setdefault(int(prefix), set()).add(suffix)
    return groups


def _validate_checkpoint_payload(
    root: Path,
    state: _FoundationResumeState,
    discovery: MonolingualDiscovery,
    *,
    deduplicate: bool,
) -> None:
    """Authenticate committed shard bytes and their semantic cursor boundaries."""

    files = _checkpoint_payload_files(root)
    expected = {cast(str, entry["path"]): entry for entry in state.artifact_inventory}
    before: dict[str, tuple[int, int, int]] = {}
    for relative, entry in expected.items():
        path = files.get(relative)
        if path is None:
            raise ValueError(f"foundation resume shard is missing: {relative}")
        value = _regular_file_stat(path, role="resume shard")
        before[relative] = (value.st_size, value.st_mtime_ns, value.st_ino)
        if value.st_size != entry["size"]:
            raise ValueError(f"foundation resume shard size changed: {relative}")
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"foundation resume shard digest changed: {relative}")

    groups = _inventory_shard_groups(state.artifact_inventory)
    indexed_source_counts = np.zeros(len(discovery.sources), dtype=np.int64)
    split_totals = {"train": 0, "validation": 0}
    language_to_id = {
        language: language_id for language_id, language in enumerate(discovery.languages)
    }
    source_language_ids = np.asarray(
        [language_to_id[source.language] for source in discovery.sources],
        dtype=np.int64,
    )
    source_is_denoising = np.asarray(
        [not is_reasoning_jsonl(source.path) for source in discovery.sources],
        dtype=np.bool_,
    )
    for split in ("train", "validation"):
        split_groups = groups[split]
        next_index = state.next_shard_indices[split]
        if set(split_groups) != set(range(next_index)):
            raise ValueError(f"foundation resume {split} shard sequence is not contiguous")
        for shard_index in range(next_index):
            suffixes = split_groups[shard_index]
            if suffixes != {"src.bin", "tgt.bin", "idx.npy"}:
                raise ValueError(f"foundation resume {split} shard {shard_index:05d} is incomplete")
            prefix = f"{shard_index:05d}"
            index_path = root / split / f"{prefix}.idx.npy"
            try:
                # Load one bounded shard into memory. A NumPy memmap keeps a file
                # handle alive through the exception traceback on Windows, which
                # would prevent quarantining a semantically corrupt generation.
                index = np.load(index_path, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ValueError(f"foundation resume index cannot be read: {index_path}") from error
            if index.dtype != SHARED_TARGET_INDEX_DTYPE:
                raise ValueError(f"foundation resume index dtype changed: {index_path}")
            source_ids = np.asarray(index["source_id"], dtype=np.int64)
            if source_ids.size and (
                int(source_ids.min()) < 0 or int(source_ids.max()) >= len(discovery.sources)
            ):
                raise ValueError(f"foundation resume source id is invalid: {index_path}")
            expected_language_ids = source_language_ids[source_ids]
            if not np.array_equal(
                index["src_language_id"], expected_language_ids
            ) or not np.array_equal(index["tgt_language_id"], expected_language_ids):
                raise ValueError(f"foundation resume language mapping changed: {index_path}")
            if not bool((np.asarray(index["forward_only"], dtype=np.uint8) == 1).all()):
                raise ValueError(f"foundation resume forward-only contract changed: {index_path}")
            shared = np.asarray(index["target_shared"], dtype=np.uint8)
            if not bool(np.isin(shared, (0, 1)).all()):
                raise ValueError(f"foundation resume target-sharing flag changed: {index_path}")
            if not np.array_equal(shared.astype(np.bool_), source_is_denoising[source_ids]):
                raise ValueError(f"foundation resume target sharing changed: {index_path}")
            src_lengths = np.asarray(index["src_length"], dtype=np.uint64)
            tgt_lengths = np.asarray(index["tgt_length"], dtype=np.uint64)
            if not bool((src_lengths > 0).all()) or not bool((tgt_lengths > 0).all()):
                raise ValueError(f"foundation resume contains an empty token row: {index_path}")
            shared_mask = shared.astype(np.bool_)
            if bool(shared_mask.any()) and (
                not np.array_equal(src_lengths[shared_mask], tgt_lengths[shared_mask])
                or not np.array_equal(
                    np.asarray(index["src_register"])[shared_mask],
                    np.asarray(index["tgt_register"])[shared_mask],
                )
            ):
                raise ValueError(
                    f"foundation resume shared targets contradict their source rows: {index_path}"
                )
            if not bool((np.asarray(index["quality_score"], dtype=np.uint8) == 100).all()):
                raise ValueError(f"foundation resume quality contract changed: {index_path}")
            if not bool((np.asarray(index["synthetic"], dtype=np.uint8) == 0).all()):
                raise ValueError(f"foundation resume synthetic-data contract changed: {index_path}")
            src_offsets = np.asarray(index["src_offset"], dtype=np.uint64)
            tgt_offsets = np.asarray(index["tgt_offset"], dtype=np.uint64)
            expected_src_offsets = np.concatenate(
                (np.zeros(1, dtype=np.uint64), np.cumsum(src_lengths[:-1], dtype=np.uint64))
            )
            stored_tgt_lengths = np.where(shared.astype(np.bool_), 0, tgt_lengths)
            expected_tgt_offsets = np.concatenate(
                (
                    np.zeros(1, dtype=np.uint64),
                    np.cumsum(stored_tgt_lengths[:-1], dtype=np.uint64),
                )
            )
            if not np.array_equal(src_offsets, expected_src_offsets) or not np.array_equal(
                tgt_offsets,
                expected_tgt_offsets,
            ):
                raise ValueError(f"foundation resume token offsets changed: {index_path}")
            if (root / split / f"{prefix}.src.bin").stat().st_size != int(
                src_lengths.sum(dtype=np.uint64)
            ) * 4:
                raise ValueError(f"foundation resume source token length changed: {index_path}")
            if (root / split / f"{prefix}.tgt.bin").stat().st_size != int(
                stored_tgt_lengths.sum(dtype=np.uint64)
            ) * 4:
                raise ValueError(f"foundation resume target token length changed: {index_path}")
            indexed_source_counts += np.bincount(
                source_ids,
                minlength=len(discovery.sources),
            )[: len(discovery.sources)]
            split_totals[split] += len(index)

    if tuple(indexed_source_counts.tolist()) != state.source_record_counts:
        raise ValueError("foundation resume source counts differ from committed shards")
    if (
        split_totals["train"] != state.stats.train_records
        or split_totals["validation"] != state.stats.validation_records
    ):
        raise ValueError("foundation resume split counts differ from committed shards")
    indexed_language_counts = {
        language: int(
            indexed_source_counts[
                [
                    source_id
                    for source_id, source in enumerate(discovery.sources)
                    if source.language == language
                ]
            ].sum(dtype=np.int64)
        )
        for language in discovery.languages
    }
    if indexed_language_counts != state.stats.accepted_per_language():
        raise ValueError("foundation resume language counts differ from committed shards")
    indexed_reasoning_counts = {
        language: int(
            indexed_source_counts[
                [
                    source_id
                    for source_id, source in enumerate(discovery.sources)
                    if source.language == language and is_reasoning_jsonl(source.path)
                ]
            ].sum(dtype=np.int64)
        )
        for language in discovery.languages
    }
    if any(
        indexed_reasoning_counts[language] != state.stats.languages[language].reasoning_records
        for language in discovery.languages
    ):
        raise ValueError("foundation resume reasoning counts differ from committed shards")
    if deduplicate:
        expected_digest_count = state.stats.total_records + sum(
            language.empty_after_tokenization for language in state.stats.languages.values()
        )
        if state.dedup_digest_count != expected_digest_count:
            raise ValueError("foundation resume deduplication state is incomplete")
        if (
            state.dedup_digest_count > 0
            and state.dedup_sequence_sha256 == _FOUNDATION_EMPTY_DEDUP_SHA256
        ):
            raise ValueError("foundation resume deduplication content is incomplete")
    elif (
        state.dedup_digest_count != 0
        or any(language.duplicate for language in state.stats.languages.values())
        or state.dedup_sequence_sha256 != _FOUNDATION_EMPTY_DEDUP_SHA256
    ):
        raise ValueError("foundation resume has deduplication state while it is disabled")

    after = {
        relative: (
            files[relative].stat().st_size,
            files[relative].stat().st_mtime_ns,
            getattr(files[relative].stat(), "st_ino", 0),
        )
        for relative in expected
    }
    if after != before:
        raise RuntimeError("Foundation shard bytes changed during resume authentication")


def _remove_uncommitted_foundation_tail(
    root: Path,
    inventory: tuple[dict[str, object], ...],
) -> None:
    """Remove only shard files written after the last committed authority record."""

    allowed_root_names = {
        "train",
        "validation",
        "manifest.json",
        _FOUNDATION_RESUME_DATABASE,
        f"{_FOUNDATION_RESUME_DATABASE}-journal",
    }
    for entry in os.scandir(root):
        if entry.name not in allowed_root_names:
            raise ValueError(f"foundation resume contains an unexpected artifact: {entry.path}")
    expected = {cast(str, entry["path"]) for entry in inventory}
    files = _checkpoint_payload_files(root)
    removed = False
    for relative, path in files.items():
        if relative not in expected:
            path.unlink()
            removed = True
    manifest_path = root / "manifest.json"
    if _path_exists(manifest_path):
        _regular_file_stat(manifest_path, role="uncommitted resume manifest")
        manifest_path.unlink()
        removed = True
    if removed:
        for split in ("train", "validation"):
            _fsync_directory(root / split)
        _fsync_directory(root)


def _fsync_foundation_payload(
    root: Path,
    committed_inventory: tuple[dict[str, object], ...],
) -> None:
    """Durably close only the files added after the prior checkpoint."""

    committed_paths = {cast(str, entry["path"]) for entry in committed_inventory}
    files = _checkpoint_payload_files(root)
    missing = committed_paths - set(files)
    if missing:
        raise RuntimeError(
            f"Foundation committed shards disappeared before checkpoint sync: {sorted(missing)}"
        )
    for relative, path in files.items():
        if relative not in committed_paths:
            _fsync_file(path)
    for split in ("train", "validation"):
        _fsync_directory(root / split)
    _fsync_directory(root)


class _FoundationResumeBusy(RuntimeError):
    """Another process still owns a private foundation generation."""


class _FoundationResumeJournal:
    """Durable transaction authority for cursor, shards, stats, and deduplication."""

    def __init__(
        self,
        path: Path,
        connection: sqlite3.Connection,
        contract: dict[str, object],
    ) -> None:
        self.path = path
        self._connection = connection
        self.contract = contract
        self.contract_sha256 = _json_sha256(contract)
        self._digest_count = 0
        self._digest_sequence_sha256 = _FOUNDATION_EMPTY_DEDUP_SHA256
        self._closed = False

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(path, timeout=0.0, isolation_level=None)
        try:
            connection.execute("PRAGMA busy_timeout=0")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(f"PRAGMA cache_size=-{_DEDUPLICATION_CACHE_KIB}")
            connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        except BaseException:
            connection.close()
            raise
        return connection

    @classmethod
    def create(
        cls,
        path: Path,
        contract: dict[str, object],
        initial_state: _FoundationResumeState,
    ) -> _FoundationResumeJournal:
        if _path_exists(path):
            raise FileExistsError(f"foundation resume database already exists: {path}")
        if initial_state.dedup_digest_count != 0:
            raise ValueError("a new foundation resume database must start without digests")
        if initial_state.dedup_sequence_sha256 != _FOUNDATION_EMPTY_DEDUP_SHA256:
            raise ValueError("a new foundation resume database has an invalid digest authority")
        connection = cls._connect(path)
        journal = cls(path, connection, contract)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE state ("
                "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                "payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE digests ("
                "digest BLOB PRIMARY KEY, sequence INTEGER NOT NULL UNIQUE) WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?), (?, ?)",
                (
                    "contract",
                    _canonical_json(contract),
                    "contract_sha256",
                    journal.contract_sha256,
                ),
            )
            connection.execute(f"PRAGMA user_version={_FOUNDATION_RESUME_USER_VERSION}")
            journal._write_state(initial_state)
            connection.commit()
            _fsync_file(path)
            _fsync_directory(path.parent)
            connection.execute("BEGIN IMMEDIATE")
        except BaseException:
            connection.close()
            for suffix in ("", "-journal", "-shm", "-wal"):
                candidate = Path(f"{path}{suffix}")
                if _path_exists(candidate):
                    candidate.unlink()
            raise
        return journal

    @classmethod
    def open(
        cls,
        path: Path,
        contract: dict[str, object],
        *,
        languages: tuple[str, ...],
        source_count: int,
    ) -> tuple[_FoundationResumeJournal, _FoundationResumeState]:
        _regular_file_stat(path, role="resume database")
        try:
            connection = cls._connect(path)
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as error:
            if "locked" in str(error).lower():
                raise _FoundationResumeBusy(
                    f"Foundation staging generation is still active: {path.parent}"
                ) from error
            raise
        journal = cls(path, connection, contract)
        try:
            journal._validate_schema()
            metadata_rows = connection.execute(
                "SELECT key, value FROM metadata ORDER BY key"
            ).fetchall()
            metadata = {str(key): str(value) for key, value in metadata_rows}
            if set(metadata) != {"contract", "contract_sha256"}:
                raise ValueError("foundation resume metadata schema is invalid")
            if metadata["contract_sha256"] != journal.contract_sha256:
                raise ValueError("foundation resume contract digest changed")
            if metadata["contract"] != _canonical_json(contract):
                raise ValueError("foundation resume contract changed")
            state_row = connection.execute(
                "SELECT payload, payload_sha256 FROM state WHERE singleton = 1"
            ).fetchone()
            if state_row is None:
                raise ValueError("foundation resume state is missing")
            raw_state_text, stored_state_sha256 = state_row
            if not isinstance(raw_state_text, str) or not isinstance(stored_state_sha256, str):
                raise ValueError("foundation resume state encoding is invalid")
            if hashlib.sha256(raw_state_text.encode("utf-8")).hexdigest() != stored_state_sha256:
                raise ValueError("foundation resume state digest changed")
            try:
                raw_state: object = json.loads(raw_state_text)
            except json.JSONDecodeError as error:
                raise ValueError("foundation resume state JSON is invalid") from error
            state = _parse_resume_state(
                raw_state,
                contract_sha256=journal.contract_sha256,
                languages=languages,
                source_count=source_count,
            )
            digest_count, digest_sequence_sha256 = journal._authenticate_digests()
            if digest_count != state.dedup_digest_count:
                raise ValueError("foundation resume deduplication count changed")
            if digest_sequence_sha256 != state.dedup_sequence_sha256:
                raise ValueError("foundation resume deduplication content changed")
            journal._digest_count = digest_count
            journal._digest_sequence_sha256 = digest_sequence_sha256
        except BaseException:
            connection.rollback()
            connection.close()
            raise
        return journal, state

    def _validate_schema(self) -> None:
        integrity_rows = self._connection.execute("PRAGMA integrity_check").fetchall()
        if integrity_rows != [("ok",)]:
            raise ValueError("foundation resume database integrity check failed")
        user_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if user_version != _FOUNDATION_RESUME_USER_VERSION:
            raise ValueError("foundation resume database version changed")
        tables = {
            str(name)
            for (name,) in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if tables != {"metadata", "state", "digests"}:
            raise ValueError("foundation resume database tables are invalid")
        expected_sql = {
            "metadata": (
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
            ),
            "state": (
                "CREATE TABLE state (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                "payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL)"
            ),
            "digests": (
                "CREATE TABLE digests (digest BLOB PRIMARY KEY, "
                "sequence INTEGER NOT NULL UNIQUE) WITHOUT ROWID"
            ),
        }
        for table, expected in expected_sql.items():
            schema_row = self._connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if schema_row != (expected,):
                raise ValueError(f"foundation resume database table changed: {table}")

    def _authenticate_digests(self) -> tuple[int, str]:
        """Recompute the state-bound digest chain in deterministic insertion order."""

        count = 0
        sequence_sha256 = _FOUNDATION_EMPTY_DEDUP_SHA256
        rows = self._connection.execute("SELECT sequence, digest FROM digests ORDER BY sequence")
        for raw_sequence, raw_digest in rows:
            count += 1
            if (
                isinstance(raw_sequence, bool)
                or not isinstance(raw_sequence, int)
                or raw_sequence != count
                or not isinstance(raw_digest, bytes)
                or len(raw_digest) != 16
            ):
                raise ValueError("foundation resume database contains an invalid digest row")
            sequence_sha256 = _extend_dedup_sequence_sha256(
                sequence_sha256,
                raw_digest,
            )
        return count, sequence_sha256

    def _write_state(self, state: _FoundationResumeState) -> None:
        payload_text = _canonical_json(
            _resume_state_payload(state, contract_sha256=self.contract_sha256)
        )
        payload_sha256 = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        self._connection.execute(
            "INSERT INTO state (singleton, payload, payload_sha256) VALUES (1, ?, ?) "
            "ON CONFLICT(singleton) DO UPDATE SET "
            "payload = excluded.payload, payload_sha256 = excluded.payload_sha256",
            (payload_text, payload_sha256),
        )

    def add_digest(self, digest: bytes) -> bool:
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO digests (digest, sequence) VALUES (?, ?)",
            (sqlite3.Binary(digest), self._digest_count + 1),
        )
        inserted = cursor.rowcount == 1
        if inserted:
            self._digest_count += 1
            self._digest_sequence_sha256 = _extend_dedup_sequence_sha256(
                self._digest_sequence_sha256,
                digest,
            )
        return inserted

    @property
    def digest_sequence_sha256(self) -> str:
        """Return the insertion-ordered digest authority for the open transaction."""

        return self._digest_sequence_sha256

    def commit_checkpoint(self, state: _FoundationResumeState) -> None:
        # ``COUNT(*)`` over a tens-of-millions-row B-tree at every checkpoint
        # makes total preparation quadratic. The connection owns an exclusive
        # transaction, so this insertion counter is exact between the full
        # count performed once when a journal is opened and the next commit.
        if self._digest_count != state.dedup_digest_count:
            raise ValueError("foundation checkpoint omitted deduplication state")
        if self._digest_sequence_sha256 != state.dedup_sequence_sha256:
            raise ValueError("foundation checkpoint omitted deduplication content")
        self._write_state(state)
        self._connection.commit()
        _fsync_file(self.path)
        _fsync_directory(self.path.parent)
        self._connection.execute("BEGIN IMMEDIATE")

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._connection.rollback()
        finally:
            self._connection.close()
            self._closed = True

    def remove(self) -> None:
        self.close()
        # Delete sidecars before the authoritative database. If cleanup is
        # interrupted, the remaining main file still makes the generation
        # recognizable as resumable instead of stranding an unexplained sidecar.
        for suffix in ("-wal", "-shm", "-journal", ""):
            candidate = Path(f"{self.path}{suffix}")
            if _path_exists(candidate):
                _regular_file_stat(candidate, role="resume database artifact")
                candidate.unlink()
        _fsync_directory(self.path.parent)


class _ResumableFoundationShardWriter(ShardWriter):
    """Start a regular writer after the last authenticated closed shard."""

    def __init__(
        self,
        root: Path,
        split: str,
        shard_size: int,
        language_to_id: dict[str, int],
        *,
        start_shard_index: int,
    ) -> None:
        self._resume_start_shard_index = start_shard_index
        self._resume_first_open = True
        super().__init__(
            root,
            split,
            shard_size,
            language_to_id,
            shared_targets=True,
        )

    def _open_shard(self) -> None:
        if self._resume_first_open:
            self.shard_index = self._resume_start_shard_index
            self._resume_first_open = False
        prefix = f"{self.shard_index:05d}."
        if any(path.name.startswith(prefix) for path in self.root.iterdir()):
            raise FileExistsError(
                f"foundation resume would overwrite shard {self.split}/{self.shard_index:05d}"
            )
        super()._open_shard()


def _open_foundation_writers(
    output_dir: Path,
    shard_size: int,
    language_to_id: dict[str, int],
    next_shard_indices: Mapping[str, int],
) -> dict[str, ShardWriter]:
    writers: dict[str, ShardWriter] = {}
    try:
        for split in ("train", "validation"):
            writers[split] = _ResumableFoundationShardWriter(
                output_dir,
                split,
                shard_size,
                language_to_id,
                start_shard_index=next_shard_indices[split],
            )
    except BaseException:
        _close_shard_writers(writers, suppress_errors=True)
        raise
    return writers


def _next_shard_indices(writers: Mapping[str, ShardWriter]) -> dict[str, int]:
    return {
        split: writer.shard_index + int(bool(writer.records)) for split, writer in writers.items()
    }


def _close_shard_writers(
    writers: dict[str, ShardWriter],
    *,
    suppress_errors: bool,
) -> None:
    first_error: BaseException | None = None
    for writer in writers.values():
        try:
            writer.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
    if first_error is not None and not suppress_errors:
        raise first_error


def _iter_source_physical_lines(
    path: Path,
    *,
    skip_lines: int,
    strict_utf8: bool,
) -> Iterator[tuple[int, str]]:
    """Yield whole physical lines after a validated, content-bound cursor."""

    errors = "strict" if strict_utf8 else "replace"
    observed = 0
    with path.open("r", encoding="utf-8-sig", errors=errors) as handle:
        for line_number, line in enumerate(handle, start=1):
            observed = line_number
            if line_number <= skip_lines:
                continue
            yield line_number, line
    if observed < skip_lines:
        raise ValueError(f"foundation resume cursor exceeds physical lines in source: {path}")


def _add_read_reject(language_stats: LanguageStats, reason: str) -> None:
    language_stats.read_rejects[reason] = language_stats.read_rejects.get(reason, 0) + 1


def _parse_monolingual_physical_line(
    path: Path,
    line: str,
    language_stats: LanguageStats,
) -> str | None:
    raw = line.strip()
    if not raw:
        _add_read_reject(language_stats, "blank")
        return None
    if path.suffix.lower() == ".txt":
        language_stats.lines_read += 1
        return raw
    try:
        row: object = json.loads(raw)
    except json.JSONDecodeError:
        _add_read_reject(language_stats, "malformed_json")
        return None
    if not isinstance(row, dict) or "text" not in row:
        _add_read_reject(language_stats, "missing_text_key")
        return None
    value = cast(dict[object, object], row)["text"]
    if not isinstance(value, str):
        _add_read_reject(language_stats, "non_string_text")
        return None
    text = value.strip()
    if not text:
        _add_read_reject(language_stats, "blank")
        return None
    language_stats.lines_read += 1
    return text


def _parse_reasoning_physical_line(
    line: str,
    *,
    language: str,
    language_stats: LanguageStats,
) -> ReasoningRecord | None:
    language_stats.lines_read += 1
    raw = line.strip()
    if not raw:
        language_stats.reasoning_rejected += 1
        _add_read_reject(language_stats, "reasoning_blank")
        return None
    try:
        row: object = json.loads(raw)
    except json.JSONDecodeError:
        language_stats.reasoning_rejected += 1
        _add_read_reject(language_stats, "reasoning_malformed_json")
        return None
    if not isinstance(row, dict):
        language_stats.reasoning_rejected += 1
        _add_read_reject(language_stats, "reasoning_non_object")
        return None
    try:
        return parse_reasoning_row(
            cast(dict[str, object], row),
            expected_language=language,
        )
    except ReasoningDataError:
        language_stats.reasoning_rejected += 1
        _add_read_reject(language_stats, "reasoning_invalid_record")
        return None


def _foundation_checkpoint_committed(
    _output_dir: Path,
    _state: _FoundationResumeState,
) -> None:
    """Test seam invoked only after a checkpoint has become authoritative."""


def _commit_foundation_checkpoint(
    output_dir: Path,
    writers: dict[str, ShardWriter],
    journal: _FoundationResumeJournal,
    previous: _FoundationResumeState,
    *,
    cursor: _FoundationCursor,
    stats: FoundationPrepareStats,
    source_record_counts: list[int],
    dedup_digest_count: int,
) -> _FoundationResumeState:
    next_indices = _next_shard_indices(writers)
    _close_shard_writers(writers, suppress_errors=False)
    _fsync_foundation_payload(output_dir, previous.artifact_inventory)
    state = _FoundationResumeState(
        checkpoint_sequence=previous.checkpoint_sequence + 1,
        cursor=cursor,
        next_shard_indices=next_indices,
        stats=stats,
        source_record_counts=tuple(source_record_counts),
        artifact_inventory=_build_checkpoint_inventory(
            output_dir,
            previous.artifact_inventory,
        ),
        dedup_digest_count=dedup_digest_count,
        dedup_sequence_sha256=journal.digest_sequence_sha256,
    )
    journal.commit_checkpoint(state)
    _foundation_checkpoint_committed(output_dir, state)
    return state


def _publish_staged_directory(staging_dir: Path, output_dir: Path) -> None:
    """Atomically publish on the same filesystem without replacing a destination."""

    _refuse_existing_output(output_dir)
    _fsync_directory_tree(staging_dir)
    _fsync_directory(output_dir.parent)
    renamed = False
    try:
        # ``os.rename`` is atomic within this sibling directory and, unlike
        # ``os.replace``, refuses an existing destination directory on Windows.
        os.rename(staging_dir, output_dir)
        renamed = True
        _fsync_directory(output_dir.parent)
    except OSError as error:
        if renamed:
            try:
                os.rename(output_dir, staging_dir)
                _fsync_directory(output_dir.parent)
            except OSError as rollback_error:
                raise RuntimeError(
                    "Foundation publication durability failed, and the staging "
                    "rollback also failed: "
                    f"{output_dir}"
                ) from rollback_error
            raise
        _fsync_directory(output_dir.parent)
        if not renamed and _path_exists(output_dir):
            raise FileExistsError(
                f"Foundation output appeared while publishing: {output_dir}"
            ) from error
        raise


def _publication_failure_is_resumable(
    error: BaseException,
    staging_dir: Path,
    output_dir: Path,
    *,
    generation_complete: bool,
) -> bool:
    """Keep a completed private generation after an ordinary publication failure."""

    if (
        not generation_complete
        or not isinstance(error, OSError)
        or isinstance(error, FileExistsError)
        or not _path_exists(staging_dir)
        or _path_exists(output_dir)
    ):
        return False
    try:
        _assert_regular_directory(staging_dir, role="staging directory")
        _regular_file_stat(staging_dir / "manifest.json", role="staging manifest")
    except (OSError, ValueError):
        return False
    return True


def _partial_failure_is_resumable(
    staging_dir: Path,
    output_dir: Path,
    discovery: MonolingualDiscovery,
    tokenizer_model: str | Path,
    *,
    source_snapshots: tuple[_FileSnapshot, ...],
    tokenizer_snapshot: _FileSnapshot,
    resume_contract: dict[str, object],
    deduplicate: bool,
) -> bool:
    """Keep an authenticated checkpoint after an interruption before publication."""

    if not _path_exists(staging_dir) or _path_exists(output_dir):
        return False
    journal: _FoundationResumeJournal | None = None
    try:
        _verify_source_snapshots(discovery, source_snapshots)
        _verify_tokenizer_snapshot(Path(tokenizer_model), tokenizer_snapshot)
        journal, state = _FoundationResumeJournal.open(
            staging_dir / _FOUNDATION_RESUME_DATABASE,
            resume_contract,
            languages=discovery.languages,
            source_count=len(discovery.sources),
        )
        _validate_checkpoint_payload(
            staging_dir,
            state,
            discovery,
            deduplicate=deduplicate,
        )
        # A complete zero-row cursor represents a permanent input/configuration
        # rejection, not useful work that should be retried forever.
        return state.stats.total_records > 0 or state.cursor.source_id < len(discovery.sources)
    except _FoundationResumeBusy:
        raise
    except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError):
        return False
    finally:
        if journal is not None:
            journal.close()


def foundation_dataset_problem(
    output_dir: str | Path,
    discovery: MonolingualDiscovery,
    tokenizer_model: str | Path,
    *,
    minimum_characters: int,
    maximum_characters: int,
    max_tokens: int,
    max_target_tokens: int,
    deduplicate: bool,
    shard_size: int,
    validation_fraction: float,
    language_sampling_alpha: float,
    minimum_language_share: float,
    reasoning_sample_share: float,
    release_name: str,
    allow_offline_sources: bool = False,
) -> str | None:
    """Return why a prepared foundation dataset must be rebuilt, if anything."""

    manifest_path = Path(output_dir) / "manifest.json"
    try:
        raw_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return f"Cannot read the manifest: {error}"
    if not isinstance(raw_manifest, dict):
        return "Manifest is not a JSON object"
    manifest = cast(dict[str, Any], raw_manifest)
    if manifest.get("format") != FOUNDATION_INDEX_FORMAT:
        return "Foundation indexed format changed"
    if manifest.get("stage") != "foundation":
        return "Foundation stage identity is invalid"
    if manifest.get("source_identity_schema") != FOUNDATION_SOURCE_IDENTITY_SCHEMA:
        return (
            "foundation source identity is obsolete; rebuild the dataset to replace "
            "machine-specific absolute paths with portable corpus-relative identities"
        )
    if manifest.get("preprocessing_schema") != FOUNDATION_PREPROCESSING_SCHEMA:
        return "Foundation preprocessing schema changed"
    if manifest.get("release_name") != release_name:
        return "Foundation release_name changed"
    try:
        current_tokenizer = _tokenizer_snapshot(Path(tokenizer_model))
    except (OSError, RuntimeError) as error:
        return f"Cannot read the tokenizer hash: {error}"
    tokenizer_hash = current_tokenizer.sha256
    if manifest.get("tokenizer_sha256") != tokenizer_hash:
        return "Foundation tokenizer changed"
    if manifest.get("fingerprint") != {"tokenizer_sha256": tokenizer_hash}:
        return "Foundation tokenizer fingerprint is invalid"
    raw_tokenizer_identity: object = manifest.get("tokenizer_identity")
    if not isinstance(raw_tokenizer_identity, dict):
        return "Foundation tokenizer identity is missing"
    tokenizer_identity = cast(dict[str, Any], raw_tokenizer_identity)
    if tokenizer_identity != {
        "schema": FOUNDATION_TOKENIZER_IDENTITY_SCHEMA,
        "size_bytes": current_tokenizer.size_bytes,
        "sha256": tokenizer_hash,
    }:
        return "Foundation tokenizer identity changed"

    expected_options = {
        "deduplicate": deduplicate,
        "deduplication_backend": (FOUNDATION_DEDUPLICATION_BACKEND if deduplicate else "disabled"),
        "maximum_characters": maximum_characters,
        "max_tokens": max_tokens,
        "max_target_tokens": max_target_tokens,
        "minimum_characters": minimum_characters,
        "reasoning_sample_share": reasoning_sample_share,
        "shard_size": shard_size,
        "validation_fraction": validation_fraction,
    }
    raw_options: object = manifest.get("preprocessing_options")
    options = cast(dict[str, Any], raw_options) if isinstance(raw_options, dict) else {}
    if any(options.get(name) != value for name, value in expected_options.items()):
        return "Foundation preprocessing options changed"

    raw_sampling: object = manifest.get("language_sampling")
    if not isinstance(raw_sampling, dict):
        return "Foundation language-sampling contract is missing"
    sampling = cast(dict[str, Any], raw_sampling)
    if sampling.get("alpha") != language_sampling_alpha:
        return "Foundation language-sampling alpha changed"
    if sampling.get("minimum_share") != minimum_language_share:
        return "Foundation minimum language share changed"

    raw_sources: object = manifest.get("sources")
    if not isinstance(raw_sources, list):
        return "Foundation source list is missing"
    source_values = cast(list[object], raw_sources)
    actual_sources: list[tuple[str, str, int, str, str]] = []
    for raw_source in source_values:
        if not isinstance(raw_source, dict):
            return "Foundation source entry is invalid"
        source = cast(dict[str, Any], raw_source)
        try:
            source_hash = source.get("sha256")
            if (
                not isinstance(source_hash, str)
                or len(source_hash) != 64
                or any(character not in "0123456789abcdef" for character in source_hash)
            ):
                return "Foundation source entry has an invalid SHA-256"
            logical_path = source.get("logical_path")
            if (
                not isinstance(logical_path, str)
                or not logical_path
                or logical_path.startswith("/")
                or ".." in PurePosixPath(logical_path).parts
                or PurePosixPath(logical_path).as_posix() != logical_path
            ):
                return "Foundation source entry has an invalid logical_path"
            actual_sources.append(
                (
                    str(source.get("language", "")),
                    logical_path,
                    int(source.get("size_bytes", -1)),
                    str(source.get("task", "")),
                    source_hash,
                )
            )
        except (TypeError, ValueError):
            return "Foundation source entry is invalid"
    identity_payload: list[dict[str, object]] = [
        {
            "language": language,
            "logical_path": logical_path,
            "size_bytes": size_bytes,
            "task": task,
            "sha256": source_hash,
        }
        for language, logical_path, size_bytes, task, source_hash in actual_sources
    ]
    if manifest.get("sources_sha256") != _source_identity_digest(identity_payload):
        return "Foundation aggregate source fingerprint is invalid"
    semantic_discovery = discovery
    if allow_offline_sources and not discovery.sources:
        # A prepared-only GPU bundle intentionally omits source corpora. Build a
        # manifest-backed discovery view for semantic/index validation while
        # retaining the aggregate source digest as the provenance authority.
        semantic_discovery = MonolingualDiscovery(
            root=discovery.root,
            sources=tuple(
                MonolingualSource(
                    language=language,
                    path=discovery.root.joinpath(*PurePosixPath(logical_path).parts),
                    size_bytes=size_bytes,
                )
                for language, logical_path, size_bytes, _task, _digest in actual_sources
            ),
        )
    else:
        # Preserve a path-specific diagnostic when a source from the caller's
        # discovery vanished; rediscovery alone would only report a set mismatch.
        for source in discovery.sources:
            if not source.path.is_file():
                try:
                    _source_sha256(source.path)
                except OSError as error:
                    return str(error)
        configured_languages = tuple(
            dict.fromkeys((*discovery.languages, *discovery.languages_without_data))
        )
        rediscovered = discover_monolingual_sources(discovery.root, configured_languages)
        rediscovered_sources: set[tuple[str, str, int, str, str]] = set()
        for source in rediscovered.sources:
            try:
                source_hash = _source_sha256(source.path)
                logical_path = _source_logical_path(rediscovered, source.path)
            except (OSError, ValueError) as error:
                return str(error)
            rediscovered_sources.add(
                (
                    source.language,
                    logical_path,
                    source.size_bytes,
                    "reasoning" if is_reasoning_jsonl(source.path) else "denoising",
                    source_hash,
                )
            )
        expected_sources: list[tuple[str, str, int, str, str]] = []
        for source in discovery.sources:
            try:
                source_hash = _source_sha256(source.path)
                logical_path = _source_logical_path(discovery, source.path)
            except (OSError, ValueError) as error:
                return str(error)
            expected_sources.append(
                (
                    source.language,
                    logical_path,
                    source.size_bytes,
                    "reasoning" if is_reasoning_jsonl(source.path) else "denoising",
                    source_hash,
                )
            )
        if (
            len(rediscovered_sources) != len(rediscovered.sources)
            or rediscovered_sources != set(expected_sources)
            or actual_sources != expected_sources
            or len(actual_sources) != len(source_values)
        ):
            return "Foundation source list, size, or content changed"
    artifact_problem = dataset_artifact_problem(output_dir)
    if artifact_problem is not None:
        return f"Foundation indexed payload is corrupt: {artifact_problem}"
    semantic_problem = _foundation_manifest_semantic_problem(
        Path(output_dir),
        manifest,
        semantic_discovery,
        language_sampling_alpha=language_sampling_alpha,
        minimum_language_share=minimum_language_share,
        reasoning_sample_share=reasoning_sample_share,
        sources_offline=allow_offline_sources and not discovery.sources,
    )
    if semantic_problem is not None:
        return semantic_problem
    return None


_LANGUAGE_STAT_INTEGER_FIELDS = (
    "lines_read",
    "accepted",
    "too_short",
    "too_long",
    "segmented_documents",
    "segments",
    "duplicate",
    "empty_after_tokenization",
    "reasoning_records",
    "reasoning_rejected",
    "reasoning_prompt_truncated",
    "reasoning_think_truncated",
    "reasoning_answer_truncated",
)


def _nonnegative_manifest_integer(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"foundation staging stats.{name} must be a non-negative integer")
    return value


def _stats_from_manifest(manifest: dict[str, Any]) -> FoundationPrepareStats:
    raw_stats: object = manifest.get("stats")
    if not isinstance(raw_stats, dict):
        raise ValueError("Foundation staging statistics are missing")
    stats_payload = cast(dict[str, Any], raw_stats)
    if set(stats_payload) != {"train_records", "validation_records", "languages"}:
        raise ValueError("Foundation staging statistics schema is invalid")
    raw_languages: object = stats_payload.get("languages")
    if not isinstance(raw_languages, dict):
        raise ValueError("Foundation staging language statistics are missing")

    language_stats: dict[str, LanguageStats] = {}
    for raw_language, raw_payload in cast(dict[object, object], raw_languages).items():
        if not isinstance(raw_language, str) or not isinstance(raw_payload, dict):
            raise ValueError("Foundation staging language-statistics entry is invalid")
        payload = cast(dict[str, Any], raw_payload)
        required_fields = {*_LANGUAGE_STAT_INTEGER_FIELDS, "read_rejects"}
        if set(payload) != required_fields:
            raise ValueError("Foundation staging language-statistics schema is invalid")
        integers = {
            name: _nonnegative_manifest_integer(payload, name)
            for name in _LANGUAGE_STAT_INTEGER_FIELDS
        }
        raw_rejects: object = payload.get("read_rejects")
        if not isinstance(raw_rejects, dict):
            raise ValueError("Foundation staging read_rejects mapping is invalid")
        rejects: dict[str, int] = {}
        for reason, count in cast(dict[object, object], raw_rejects).items():
            if (
                not isinstance(reason, str)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("Foundation staging read_rejects entry is invalid")
            rejects[reason] = count
        language_stats[raw_language] = LanguageStats(
            lines_read=integers["lines_read"],
            accepted=integers["accepted"],
            too_short=integers["too_short"],
            too_long=integers["too_long"],
            segmented_documents=integers["segmented_documents"],
            segments=integers["segments"],
            duplicate=integers["duplicate"],
            empty_after_tokenization=integers["empty_after_tokenization"],
            reasoning_records=integers["reasoning_records"],
            reasoning_rejected=integers["reasoning_rejected"],
            reasoning_prompt_truncated=integers["reasoning_prompt_truncated"],
            reasoning_think_truncated=integers["reasoning_think_truncated"],
            reasoning_answer_truncated=integers["reasoning_answer_truncated"],
            read_rejects=rejects,
        )

    train_records = _nonnegative_manifest_integer(stats_payload, "train_records")
    validation_records = _nonnegative_manifest_integer(
        stats_payload,
        "validation_records",
    )
    if sum(item.accepted for item in language_stats.values()) != (
        train_records + validation_records
    ):
        raise ValueError("Foundation staging statistics totals do not match")
    raw_manifest_languages: object = manifest.get("languages")
    if not isinstance(raw_manifest_languages, list) or not all(
        isinstance(language, str) for language in raw_manifest_languages
    ):
        raise ValueError("Foundation staging language identity does not match")
    manifest_languages = cast(list[str], raw_manifest_languages)
    if len(manifest_languages) != len(set(manifest_languages)) or set(manifest_languages) != set(
        language_stats
    ):
        raise ValueError("Foundation staging language identity does not match")
    ordered_language_stats = {language: language_stats[language] for language in manifest_languages}
    return FoundationPrepareStats(
        languages=ordered_language_stats,
        train_records=train_records,
        validation_records=validation_records,
    )


def _read_staging_stats(staging_dir: Path) -> FoundationPrepareStats:
    try:
        raw_manifest: object = json.loads(
            (staging_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read the foundation staging manifest") from error
    if not isinstance(raw_manifest, dict):
        raise ValueError("foundation staging manifest must be an object")
    return _stats_from_manifest(cast(dict[str, Any], raw_manifest))


def _foundation_manifest_semantic_problem(
    output_dir: Path,
    manifest: dict[str, Any],
    discovery: MonolingualDiscovery,
    *,
    language_sampling_alpha: float,
    minimum_language_share: float,
    reasoning_sample_share: float,
    sources_offline: bool = False,
) -> str | None:
    """Validate every manifest field consumed after dataset preparation."""

    expected_top_level = {"train", "validation", "manifest.json"}
    try:
        actual_top_level = {path.name for path in output_dir.iterdir()}
    except OSError as error:
        return f"Cannot inspect foundation dataset tree: {error}"
    if actual_top_level != expected_top_level:
        return (
            "foundation dataset top-level artifacts are incomplete or unexpected: "
            f"missing={sorted(expected_top_level - actual_top_level)}, "
            f"unexpected={sorted(actual_top_level - expected_top_level)}"
        )

    try:
        stats = _stats_from_manifest(manifest)
    except (TypeError, ValueError) as error:
        return f"Foundation manifest statistics are invalid: {error}"

    expected_languages = list(discovery.languages)
    if list(stats.languages) != expected_languages:
        return "Foundation manifest language order or list differs from the sources"
    expected_language_to_id = {language: index for index, language in enumerate(expected_languages)}
    raw_language_to_id: object = manifest.get("language_to_id")
    if not isinstance(raw_language_to_id, dict):
        return "Foundation manifest language_to_id is invalid"
    language_to_id = cast(dict[object, object], raw_language_to_id)
    if set(language_to_id) != set(expected_language_to_id) or any(
        isinstance(language_to_id[language], bool)
        or not isinstance(language_to_id[language], int)
        or language_to_id[language] != expected_id
        for language, expected_id in expected_language_to_id.items()
    ):
        return "Foundation manifest language_to_id is invalid"
    if manifest.get("language_pairs") != [[language, language] for language in expected_languages]:
        return "Foundation manifest language_pairs is invalid"
    if manifest.get("source_only_languages") != []:
        return "Foundation manifest source_only_languages is invalid"
    if manifest.get("storage_sides") != ["src", "tgt"]:
        return "Foundation manifest storage_sides is invalid"
    if manifest.get("target_storage") != "row-shared-source-v1":
        return "foundation manifest target_storage contract is invalid"
    expected_index_dtype = json.loads(json.dumps(SHARED_TARGET_INDEX_DTYPE.descr))
    if manifest.get("index_dtype") != expected_index_dtype:
        return "Foundation manifest index_dtype is invalid"

    raw_sampling: object = manifest.get("language_sampling")
    if not isinstance(raw_sampling, dict):
        return "Foundation manifest language_sampling is invalid"
    sampling = cast(dict[str, Any], raw_sampling)
    for name, expected_value in (
        ("alpha", language_sampling_alpha),
        ("minimum_share", minimum_language_share),
    ):
        value = sampling.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != expected_value
        ):
            return f"Foundation manifest sampling field is invalid: {name}"
    raw_counts: object = sampling.get("counts")
    raw_weights: object = sampling.get("weights")
    if not isinstance(raw_counts, dict) or not isinstance(raw_weights, dict):
        return "Foundation manifest sampling counts or weights are invalid"
    counts = cast(dict[object, object], raw_counts)
    weights = cast(dict[object, object], raw_weights)
    expected_language_keys = set(expected_languages)
    if set(counts) != expected_language_keys or set(weights) != expected_language_keys:
        return "Foundation manifest sampling language keys are invalid"

    normalized_counts: dict[str, int] = {}
    normalized_weights: dict[str, float] = {}
    for language in expected_languages:
        count = counts[language]
        weight = weights[language]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return f"Foundation manifest sampling count is invalid: {language}"
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            return f"Foundation manifest sampling weight is invalid: {language}"
        if count != stats.languages[language].accepted:
            return f"Foundation manifest sampling count differs from statistics: {language}"
        normalized_counts[language] = count
        normalized_weights[language] = float(weight)

    expected_balance = assess_language_balance(
        normalized_counts,
        alpha=language_sampling_alpha,
        minimum_share=minimum_language_share,
    )
    for language in expected_languages:
        if not math.isclose(
            normalized_weights[language],
            expected_balance.weights[language],
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            return f"Foundation manifest sampling weight differs from calculation: {language}"
    if sampling.get("warnings") != list(expected_balance.warnings):
        return "Foundation manifest sampling warnings differ from calculation"

    raw_reasoning: object = manifest.get("reasoning")
    if not isinstance(raw_reasoning, dict):
        return "Foundation manifest reasoning contract is invalid"
    reasoning = cast(dict[str, Any], raw_reasoning)
    if reasoning.get("sample_share") != reasoning_sample_share:
        return "Foundation manifest reasoning sample_share is invalid"

    raw_sources: object = manifest.get("sources")
    if not isinstance(raw_sources, list):
        return "Foundation manifest source count is invalid"
    source_values = cast(list[object], raw_sources)
    if len(source_values) != len(discovery.sources):
        return "Foundation manifest source count is invalid"
    source_records: list[int] = []
    source_language_ids: list[int] = []
    source_tasks: list[str] = []
    for source_id, (raw_source, expected_source) in enumerate(
        zip(source_values, discovery.sources, strict=True)
    ):
        if not isinstance(raw_source, dict):
            return f"Foundation manifest source is not an object: {source_id}"
        source = cast(dict[str, Any], raw_source)
        raw_source_id = source.get("id")
        if (
            isinstance(raw_source_id, bool)
            or not isinstance(raw_source_id, int)
            or raw_source_id != source_id
        ):
            return f"Foundation manifest source id is not contiguous: {source_id}"
        if set(source) != {
            "id",
            "language",
            "logical_path",
            "name",
            "records",
            "sha256",
            "size_bytes",
            "task",
        }:
            return f"Foundation manifest source fields are invalid: {source_id}"
        name = source.get("name")
        if not isinstance(name, str) or not name or name != expected_source.path.name:
            return f"Foundation manifest source name is invalid: {source_id}"
        if source.get("language") != expected_source.language:
            return f"Foundation manifest source language is invalid: {source_id}"
        if sources_offline:
            expected_logical_path = expected_source.path.relative_to(discovery.root).as_posix()
        else:
            try:
                expected_logical_path = _source_logical_path(discovery, expected_source.path)
            except ValueError as error:
                return str(error)
        if source.get("logical_path") != expected_logical_path:
            return f"Foundation manifest source logical_path is invalid: {source_id}"
        size_bytes = source.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes != expected_source.size_bytes
        ):
            return f"Foundation manifest source size_bytes is invalid: {source_id}"
        source_hash = source.get("sha256")
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            return f"Foundation manifest source sha256 is invalid: {source_id}"
        expected_task = "reasoning" if is_reasoning_jsonl(expected_source.path) else "denoising"
        if source.get("task") != expected_task:
            return f"Foundation manifest source task is invalid: {source_id}"
        source_tasks.append(expected_task)
        records = source.get("records")
        if isinstance(records, bool) or not isinstance(records, int) or records < 0:
            return f"Foundation manifest source record count is invalid: {source_id}"
        source_records.append(records)
        source_language_ids.append(expected_language_to_id[expected_source.language])

    indexed_counts = np.zeros(len(source_records), dtype=np.int64)
    split_totals: dict[str, int] = {}
    for split in ("train", "validation"):
        split_total = 0
        expected_artifacts: set[str] = set()
        for index_path in sorted((output_dir / split).glob("*.idx.npy")):
            try:
                index = np.load(index_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                return f"Cannot read foundation index metadata: {index_path}: {error}"
            names = index.dtype.names
            required_fields = {
                "source_id",
                "src_offset",
                "src_length",
                "tgt_offset",
                "tgt_length",
                "src_language_id",
                "tgt_language_id",
                "src_register",
                "tgt_register",
                "forward_only",
                "target_shared",
            }
            if names is None or not required_fields.issubset(names):
                return f"Foundation index metadata schema is invalid: {index_path}"
            if index.dtype != SHARED_TARGET_INDEX_DTYPE:
                return f"Foundation index dtype is invalid: {index_path}"
            source_ids = np.asarray(index["source_id"], dtype=np.int64)
            if source_ids.size and (
                int(source_ids.min()) < 0 or int(source_ids.max()) >= len(source_records)
            ):
                return f"Foundation index source_id is out of range: {index_path}"
            indexed_counts += np.bincount(
                source_ids,
                minlength=len(source_records),
            )[: len(source_records)]
            expected_ids = np.asarray(source_language_ids, dtype=np.int64)[source_ids]
            if not np.array_equal(
                np.asarray(index["src_language_id"], dtype=np.int64),
                expected_ids,
            ) or not np.array_equal(
                np.asarray(index["tgt_language_id"], dtype=np.int64),
                expected_ids,
            ):
                return f"Foundation index language/source mapping is invalid: {index_path}"
            if not bool((np.asarray(index["forward_only"], dtype=np.uint8) == 1).all()):
                return f"Foundation index forward_only contract is invalid: {index_path}"
            shared = np.asarray(index["target_shared"], dtype=np.uint8)
            if not bool(np.isin(shared, (0, 1)).all()):
                return f"foundation index target_shared flag is invalid: {index_path}"
            expected_shared = np.asarray(
                [source_tasks[source_id] == "denoising" for source_id in source_ids],
                dtype=np.bool_,
            )
            if not np.array_equal(shared.astype(np.bool_), expected_shared):
                return f"foundation index target storage disagrees with source tasks: {index_path}"

            src_offsets = np.asarray(index["src_offset"], dtype=np.uint64)
            src_lengths = np.asarray(index["src_length"], dtype=np.uint64)
            tgt_offsets = np.asarray(index["tgt_offset"], dtype=np.uint64)
            tgt_lengths = np.asarray(index["tgt_length"], dtype=np.uint64)
            expected_src_offsets = np.concatenate(
                (np.zeros(1, dtype=np.uint64), np.cumsum(src_lengths[:-1], dtype=np.uint64))
            )
            stored_tgt_lengths = np.where(shared.astype(np.bool_), 0, tgt_lengths)
            expected_tgt_offsets = np.concatenate(
                (
                    np.zeros(1, dtype=np.uint64),
                    np.cumsum(stored_tgt_lengths[:-1], dtype=np.uint64),
                )
            )
            if not np.array_equal(src_offsets, expected_src_offsets) or not np.array_equal(
                tgt_offsets,
                expected_tgt_offsets,
            ):
                return f"foundation token offsets are not contiguous: {index_path}"
            shared_mask = shared.astype(np.bool_)
            if bool(shared_mask.any()) and (
                not np.array_equal(src_lengths[shared_mask], tgt_lengths[shared_mask])
                or not np.array_equal(
                    np.asarray(index["src_register"])[shared_mask],
                    np.asarray(index["tgt_register"])[shared_mask],
                )
            ):
                return f"foundation shared targets contradict their source rows: {index_path}"
            prefix = index_path.name.removesuffix(".idx.npy")
            src_path = index_path.with_name(f"{prefix}.src.bin")
            tgt_path = index_path.with_name(f"{prefix}.tgt.bin")
            expected_artifacts.update({index_path.name, src_path.name, tgt_path.name})
            if src_path.stat().st_size != int(src_lengths.sum(dtype=np.uint64)) * 4:
                return f"foundation source token payload length is invalid: {src_path}"
            if tgt_path.stat().st_size != int(stored_tgt_lengths.sum(dtype=np.uint64)) * 4:
                return f"foundation target token payload length is invalid: {tgt_path}"
            split_total += len(index)
        actual_artifacts = {path.name for path in (output_dir / split).iterdir()}
        if actual_artifacts != expected_artifacts:
            return (
                f"foundation {split} artifacts are incomplete or unexpected: "
                f"missing={sorted(expected_artifacts - actual_artifacts)}, "
                f"unexpected={sorted(actual_artifacts - expected_artifacts)}"
            )
        split_totals[split] = split_total

    if indexed_counts.tolist() != source_records:
        return "Foundation manifest source records differ from index metadata"
    if split_totals.get("train") != stats.train_records:
        return "Foundation manifest train_records differs from index metadata"
    if split_totals.get("validation") != stats.validation_records:
        return "Foundation manifest validation_records differs from index metadata"
    return None


def _staging_candidates(output_dir: Path) -> list[Path]:
    parent = output_dir.parent
    if not parent.is_dir():
        return []
    return sorted(parent.glob(f".{output_dir.name}.staging-*"))


def _foundation_output_lock_filename(output_dir: Path) -> str:
    """Return a portable lock identity for one exact foundation output path."""

    identity = os.path.normcase(str(_absolute_path(output_dir))).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return f".{_FOUNDATION_OUTPUT_LOCK_SCHEMA}-{digest}.lock"


def _foundation_output_conflict_message(
    output_dir: Path,
) -> Callable[[Path, str], str]:
    def describe(_root: Path, holder: str) -> str:
        return (
            f"foundation dataset output is locked by another process: {output_dir}\n"
            f"  current holder: {holder}\n"
            "  Wait for that preparation to finish or choose a different output directory."
        )

    return describe


@contextmanager  # pyright: ignore[reportDeprecated]
def _foundation_output_lock(output_dir: Path) -> Iterator[Path]:
    """Serialize discovery recovery, generation, and publication for one output."""

    with _exclusive_lock(
        output_dir.parent,
        filename=_foundation_output_lock_filename(output_dir),
        conflict_message=_foundation_output_conflict_message(output_dir),
    ) as lock_path:
        yield lock_path


def _clean_orphan_staging(output_dir: Path) -> None:
    for candidate in _staging_candidates(output_dir):
        _remove_staging_path(candidate)


def _recover_or_clean_staging(
    output_dir: Path,
    discovery: MonolingualDiscovery,
    tokenizer_model: str | Path,
    *,
    source_snapshots: tuple[_FileSnapshot, ...],
    tokenizer_snapshot: _FileSnapshot,
    minimum_characters: int,
    maximum_characters: int,
    max_tokens: int,
    max_target_tokens: int,
    deduplicate: bool,
    shard_size: int,
    validation_fraction: float,
    language_sampling_alpha: float,
    minimum_language_share: float,
    reasoning_sample_share: float,
    release_name: str,
    resume_contract: dict[str, object],
) -> _StagingRecovery:
    complete: list[tuple[Path, FoundationPrepareStats]] = []
    partial: list[_PartialFoundationGeneration] = []
    for candidate in _staging_candidates(output_dir):
        try:
            _regular_staging_tree(candidate)
        except (OSError, RuntimeError, ValueError) as error:
            for generation in partial:
                generation.journal.close()
            raise RuntimeError(f"Refusing unsafe foundation staging path: {candidate}") from error
        try:
            manifest_before = _file_snapshot(
                candidate / "manifest.json",
                role="staging manifest",
            )
            problem = foundation_dataset_problem(
                candidate,
                discovery,
                tokenizer_model,
                minimum_characters=minimum_characters,
                maximum_characters=maximum_characters,
                max_tokens=max_tokens,
                max_target_tokens=max_target_tokens,
                deduplicate=deduplicate,
                shard_size=shard_size,
                validation_fraction=validation_fraction,
                language_sampling_alpha=language_sampling_alpha,
                minimum_language_share=minimum_language_share,
                reasoning_sample_share=reasoning_sample_share,
                release_name=release_name,
            )
            if problem is not None:
                raise ValueError(problem)
            raw_recovery_manifest: object = json.loads(
                (candidate / "manifest.json").read_text(encoding="utf-8")
            )
            if not isinstance(raw_recovery_manifest, dict):
                raise ValueError("foundation staging manifest must be an object")
            manifest_payload = cast(dict[str, Any], raw_recovery_manifest)
            raw_sampling: object = manifest_payload.get("language_sampling")
            if not isinstance(raw_sampling, dict):
                raise ValueError("Foundation staging language_sampling is missing")
            sampling = cast(dict[str, Any], raw_sampling)
            if sampling.get("alpha") != language_sampling_alpha:
                raise ValueError("Foundation staging language-sampling alpha differs")
            if sampling.get("minimum_share") != minimum_language_share:
                raise ValueError("Foundation staging minimum language share differs")
            recovered_stats = _read_staging_stats(candidate)
            manifest_after = _file_snapshot(
                candidate / "manifest.json",
                role="staging manifest",
            )
            if manifest_after != manifest_before:
                raise ValueError("Foundation staging manifest changed during authentication")
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, TypeError):
            journal: _FoundationResumeJournal | None = None
            try:
                journal, state = _FoundationResumeJournal.open(
                    candidate / _FOUNDATION_RESUME_DATABASE,
                    resume_contract,
                    languages=discovery.languages,
                    source_count=len(discovery.sources),
                )
                _validate_checkpoint_payload(
                    candidate,
                    state,
                    discovery,
                    deduplicate=deduplicate,
                )
                _remove_uncommitted_foundation_tail(candidate, state.artifact_inventory)
                # Close the validation/cleanup race and prove that cleanup retained
                # exactly the authenticated prefix before any writer can append.
                _validate_checkpoint_payload(
                    candidate,
                    state,
                    discovery,
                    deduplicate=deduplicate,
                )
            except _FoundationResumeBusy:
                if journal is not None:
                    journal.close()
                for generation in partial:
                    generation.journal.close()
                raise
            except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError):
                if journal is not None:
                    journal.close()
                _remove_staging_path(candidate)
                continue
            partial.append(
                _PartialFoundationGeneration(
                    path=candidate,
                    state=state,
                    journal=journal,
                )
            )
            continue
        complete.append((candidate, recovered_stats))

    if complete:
        for generation in partial:
            generation.journal.close()
            _remove_staging_path(generation.path)
        selected, recovered_stats = max(
            complete,
            key=lambda item: (item[0].stat().st_mtime_ns, item[0].name),
        )
        # Leave only the selected complete generation before the atomic rename;
        # a crash immediately after publication must not strand other large trees.
        for candidate, _stats in complete:
            if candidate != selected:
                _remove_staging_path(candidate)
        try:
            _verify_source_snapshots(discovery, source_snapshots)
            _verify_tokenizer_snapshot(Path(tokenizer_model), tokenizer_snapshot)
            _publish_staged_directory(selected, output_dir)
        except BaseException as error:
            if not _publication_failure_is_resumable(
                error,
                selected,
                output_dir,
                generation_complete=True,
            ):
                _remove_staging_path(selected)
            raise
        return _StagingRecovery(complete_stats=recovered_stats)

    if not partial:
        return _StagingRecovery()
    selected_partial = max(
        partial,
        key=lambda item: (
            item.state.checkpoint_sequence,
            item.path.stat().st_mtime_ns,
            item.path.name,
        ),
    )
    for generation in partial:
        if generation != selected_partial:
            generation.journal.close()
            _remove_staging_path(generation.path)
    return _StagingRecovery(partial=selected_partial)


def _prepare_foundation_dataset_in_staging(
    discovery: MonolingualDiscovery,
    tokenizer_model: str | Path,
    output_dir: str | Path,
    *,
    minimum_characters: int = 8,
    maximum_characters: int = 4000,
    max_tokens: int = 510,
    max_target_tokens: int | None = None,
    deduplicate: bool = True,
    shard_size: int = 200_000,
    validation_fraction: float = 0.002,
    language_sampling_alpha: float = DEFAULT_LANGUAGE_SAMPLING_ALPHA,
    minimum_language_share: float = 0.05,
    reasoning_sample_share: float = 0.05,
    release_name: str = FOUNDATION_RELEASE_NAME,
    source_snapshots: tuple[_FileSnapshot, ...],
    tokenizer_snapshot: _FileSnapshot,
    checkpoint_reserve_bytes: int,
    resume_state: _FoundationResumeState | None = None,
    resume_journal: _FoundationResumeJournal | None = None,
) -> FoundationPrepareStats:
    """Build a complete dataset inside a private, unpublished directory."""

    if not discovery.sources:
        raise ValueError(
            "The monolingual corpus has no usable training files. "
            f"root={discovery.root}. Place .txt or .jsonl files inside language-code "
            "directories."
        )
    if minimum_characters < 1:
        raise ValueError("minimum_characters must be positive")
    if maximum_characters <= minimum_characters:
        raise ValueError("maximum_characters must be greater than minimum_characters")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if max_target_tokens is None:
        max_target_tokens = max_tokens
    if max_target_tokens < 6:
        raise ValueError(
            "max_target_tokens must leave room for reasoning trace markers and content"
        )
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    if not 0.0 <= reasoning_sample_share <= 0.10:
        raise ValueError("reasoning_sample_share must be in [0, 0.10]")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SionTokenizer(tokenizer_model)
    languages = discovery.languages
    missing_tags = sorted(set(languages) - set(tokenizer.denoise_tags))
    if missing_tags:
        raise ValueError(
            "Tokenizer is missing denoise tags for the monolingual languages: "
            f"{missing_tags}; retrain it with these languages configured"
        )
    reasoning_languages = tuple(
        dict.fromkeys(
            source.language for source in discovery.sources if is_reasoning_jsonl(source.path)
        )
    )
    missing_reasoning_tags = sorted(set(reasoning_languages) - set(tokenizer.reasoning_tags))
    if missing_reasoning_tags:
        raise ValueError(
            "Tokenizer is missing reasoning task tags for structured corpora: "
            f"{missing_reasoning_tags}; retrain it after adding the reasoning files"
        )
    language_to_id = {language: index for index, language in enumerate(languages)}
    resume_contract = _build_resume_contract(
        discovery,
        source_snapshots,
        tokenizer_snapshot,
        minimum_characters=minimum_characters,
        maximum_characters=maximum_characters,
        max_tokens=max_tokens,
        max_target_tokens=max_target_tokens,
        deduplicate=deduplicate,
        shard_size=shard_size,
        validation_fraction=validation_fraction,
        language_sampling_alpha=language_sampling_alpha,
        minimum_language_share=minimum_language_share,
        reasoning_sample_share=reasoning_sample_share,
        release_name=release_name,
    )
    if (resume_state is None) != (resume_journal is None):
        raise ValueError("foundation resume state and journal must be supplied together")
    if resume_state is None:
        for split in ("train", "validation"):
            (output_dir / split).mkdir(parents=True, exist_ok=False)
        stats = FoundationPrepareStats(
            languages={language: LanguageStats() for language in languages}
        )
        source_record_counts = [0 for _source in discovery.sources]
        resume_state = _FoundationResumeState(
            checkpoint_sequence=0,
            cursor=_FoundationCursor(
                source_id=0,
                physical_lines=0,
                total_physical_lines=0,
            ),
            next_shard_indices={"train": 0, "validation": 0},
            stats=stats,
            source_record_counts=tuple(source_record_counts),
            artifact_inventory=(),
            dedup_digest_count=0,
            dedup_sequence_sha256=_FOUNDATION_EMPTY_DEDUP_SHA256,
        )
        resume_journal = _FoundationResumeJournal.create(
            output_dir / _FOUNDATION_RESUME_DATABASE,
            resume_contract,
            resume_state,
        )
    else:
        assert resume_journal is not None
        if resume_journal.path.parent != output_dir:
            raise ValueError("foundation resume journal belongs to a different staging directory")
        if resume_journal.contract != resume_contract:
            raise ValueError("foundation resume journal contract changed")
        stats = resume_state.stats
        source_record_counts = list(resume_state.source_record_counts)
    assert resume_journal is not None
    current_state = resume_state
    cursor = current_state.cursor
    dedup_digest_count = current_state.dedup_digest_count
    try:
        writers = _open_foundation_writers(
            output_dir,
            shard_size,
            language_to_id,
            current_state.next_shard_indices,
        )
    except BaseException:
        resume_journal.close()
        raise
    source_ids = {source.path: index for index, source in enumerate(discovery.sources)}

    def record_segment(
        text: str,
        *,
        language: str,
        language_stats: LanguageStats,
        source_id: int,
    ) -> None:
        """Write one segment, rejecting duplicates and empty token sequences here."""

        nonlocal dedup_digest_count
        if deduplicate:
            digest = _text_digest(language, text)
            if not resume_journal.add_digest(digest):
                language_stats.duplicate += 1
                return
            dedup_digest_count += 1
        token_ids = tokenizer.encode(text)[:max_tokens]
        if not token_ids:
            language_stats.empty_after_tokenization += 1
            return
        # Denoising has no test split. Its checkpoint-selection signal is reconstruction
        # loss; final quality is evaluated on translation holdouts in the next stage.
        split = choose_split_for_key(f"{language}\0{text}", validation_fraction, 0.0)
        if split == "test":
            split = "train"
        register = infer_register(text, language)
        writers[split].add(
            src_ids=token_ids,
            tgt_ids=token_ids,
            src_register=register,
            tgt_register=register,
            src_language=language,
            tgt_language=language,
            source_id=source_id,
            quality_score=100,
            synthetic=False,
            # Disable bidirectional expansion. Denoising has identical logical
            # directions, so expansion would train every sentence exactly twice.
            forward_only=True,
            # Denoising reconstructs the exact source token sequence. The v3
            # foundation format authenticates that invariant per row and stores
            # those bytes once. Reasoning rows keep an independent target.
            shared_target=True,
        )
        language_stats.accepted += 1
        if split == "train":
            stats.train_records += 1
        else:
            stats.validation_records += 1
        source_record_counts[source_id] += 1

    def record_reasoning(
        record: ReasoningRecord,
        *,
        language: str,
        language_stats: LanguageStats,
        source_id: int,
    ) -> None:
        """Write one structured prompt-to-trace example without denoising it."""

        nonlocal dedup_digest_count
        digest = _reasoning_digest(language, record.prompt, record.think, record.answer)
        if deduplicate:
            if not resume_journal.add_digest(digest):
                language_stats.duplicate += 1
                return
            dedup_digest_count += 1
        encoded = serialize_reasoning_record(
            record,
            tokenizer,
            # max_tokens historically limits source *content*.  The reasoning
            # source additionally stores one task token that the collator pops.
            max_source_tokens=max_tokens + 1,
            max_target_tokens=max_target_tokens,
        )
        split = choose_split_for_key(
            f"reasoning\0{language}\0{record.prompt}\0{record.answer}",
            validation_fraction,
            0.0,
        )
        if split == "test":
            split = "train"
        writers[split].add(
            src_ids=encoded.source_ids,
            tgt_ids=encoded.target_ids,
            src_register=infer_register(record.prompt, language),
            tgt_register=infer_register(record.answer, language),
            src_language=language,
            tgt_language=language,
            source_id=source_id,
            quality_score=100,
            synthetic=False,
            forward_only=True,
        )
        language_stats.accepted += 1
        language_stats.reasoning_records += 1
        language_stats.reasoning_prompt_truncated += int(encoded.prompt_truncated)
        language_stats.reasoning_think_truncated += int(encoded.think_truncated)
        language_stats.reasoning_answer_truncated += int(encoded.answer_truncated)
        if split == "train":
            stats.train_records += 1
        else:
            stats.validation_records += 1
        source_record_counts[source_id] += 1

    try:
        for source_id, source in enumerate(discovery.sources):
            if source_id < cursor.source_id:
                continue
            language = source.language
            language_stats = stats.languages[language]
            skip_lines = cursor.physical_lines if source_id == cursor.source_id else 0
            reasoning_source = is_reasoning_jsonl(source.path)
            for physical_line, raw_line in _iter_source_physical_lines(
                source.path,
                skip_lines=skip_lines,
                strict_utf8=reasoning_source,
            ):
                if reasoning_source:
                    reasoning_record = _parse_reasoning_physical_line(
                        raw_line,
                        language=language,
                        language_stats=language_stats,
                    )
                    if reasoning_record is not None:
                        record_reasoning(
                            reasoning_record,
                            language=language,
                            language_stats=language_stats,
                            source_id=source_id,
                        )
                else:
                    raw_text = _parse_monolingual_physical_line(
                        source.path,
                        raw_line,
                        language_stats,
                    )
                    if raw_text is not None:
                        document = normalize_text(raw_text)
                        if not _is_usable(document):
                            language_stats.too_short += 1
                        else:
                            # Segment long documents instead of discarding or truncating
                            # them. Historical audits found severe character loss when
                            # document-sized lines were treated as sentence-sized rows.
                            segments = segment_text(
                                document,
                                maximum_characters=maximum_characters,
                                minimum_characters=minimum_characters,
                            )
                            if not segments:
                                language_stats.too_short += 1
                            else:
                                if len(segments) > 1:
                                    language_stats.segmented_documents += 1
                                language_stats.segments += len(segments)
                                for text in segments:
                                    record_segment(
                                        text,
                                        language=language,
                                        language_stats=language_stats,
                                        source_id=source_id,
                                    )
                cursor = _FoundationCursor(
                    source_id=source_id,
                    physical_lines=physical_line,
                    total_physical_lines=cursor.total_physical_lines + 1,
                )
                if cursor.total_physical_lines % _FOUNDATION_CHECKPOINT_INTERVAL == 0:
                    _preflight_foundation_checkpoint_space(
                        output_dir,
                        checkpoint_reserve_bytes,
                    )
                    current_state = _commit_foundation_checkpoint(
                        output_dir,
                        writers,
                        resume_journal,
                        current_state,
                        cursor=cursor,
                        stats=stats,
                        source_record_counts=source_record_counts,
                        dedup_digest_count=dedup_digest_count,
                    )
                    writers = _open_foundation_writers(
                        output_dir,
                        shard_size,
                        language_to_id,
                        current_state.next_shard_indices,
                    )
            cursor = _FoundationCursor(
                source_id=source_id + 1,
                physical_lines=0,
                total_physical_lines=cursor.total_physical_lines,
            )

        if current_state.cursor.source_id < len(discovery.sources):
            _preflight_foundation_checkpoint_space(
                output_dir,
                checkpoint_reserve_bytes,
            )
            cursor = _FoundationCursor(
                source_id=len(discovery.sources),
                physical_lines=0,
                total_physical_lines=cursor.total_physical_lines,
            )
            current_state = _commit_foundation_checkpoint(
                output_dir,
                writers,
                resume_journal,
                current_state,
                cursor=cursor,
                stats=stats,
                source_record_counts=source_record_counts,
                dedup_digest_count=dedup_digest_count,
            )
        else:
            # A final authenticated cursor can be left behind by a manifest or
            # publication failure.  Closing the empty writers is sufficient;
            # committing another synthetic "final" checkpoint would violate
            # the deterministic sequence contract after a second interruption.
            _close_shard_writers(writers, suppress_errors=False)
        # Incremental checkpoints reuse hashes for the immutable shard prefix so
        # checkpoint cost stays linear. Authenticate the entire final prefix once
        # before any manifest can endorse it.
        _validate_checkpoint_payload(
            output_dir,
            current_state,
            discovery,
            deduplicate=deduplicate,
        )
    except BaseException:
        _close_shard_writers(writers, suppress_errors=True)
        resume_journal.close()
        raise
    resume_journal.close()

    if stats.total_records == 0:
        raise ValueError(
            "The monolingual corpus produced no usable training sentences. Check "
            "minimum_characters, maximum_characters, and the source file formats."
        )

    # Do not derive an inventory or manifest from bytes read across two source
    # generations.  The final snapshot includes the resolved path, size, hash,
    # timestamp, and file identity captured before source iteration began.
    _verify_source_snapshots(discovery, source_snapshots)
    _verify_tokenizer_snapshot(Path(tokenizer_model), tokenizer_snapshot)

    balance = assess_language_balance(
        stats.accepted_per_language(),
        alpha=language_sampling_alpha,
        minimum_share=minimum_language_share,
    )
    source_manifest_records: list[dict[str, object]] = [
        {
            "id": source_ids[source.path],
            "language": source.language,
            "name": source.path.name,
            "logical_path": _source_logical_path(discovery, source.path),
            "size_bytes": source_snapshots[index].size_bytes,
            "sha256": source_snapshots[index].sha256,
            "task": "reasoning" if is_reasoning_jsonl(source.path) else "denoising",
            "records": source_record_counts[source_ids[source.path]],
        }
        for index, source in enumerate(discovery.sources)
    ]
    manifest = {
        "format": FOUNDATION_INDEX_FORMAT,
        "stage": "foundation",
        "release_name": release_name,
        "objective": (
            "span-corruption-denoising+structured-reasoning"
            if any(item.reasoning_records for item in stats.languages.values())
            else "span-corruption-denoising"
        ),
        "languages": list(languages),
        "language_to_id": language_to_id,
        # Denoising is a single-language task, not a translation pair. Recording the
        # same language on both logical sides lets the indexed reader retain its normal
        # direction semantics, while forward_only prevents reverse duplication.
        "language_pairs": [[language, language] for language in languages],
        "source_only_languages": [],
        "storage_sides": ["src", "tgt"],
        "index_dtype": SHARED_TARGET_INDEX_DTYPE.descr,
        "target_storage": "row-shared-source-v1",
        # This is a display label, not a filesystem identity. The adjacent
        # content-addressed object authenticates a tokenizer after relocation.
        "tokenizer_model": Path(tokenizer_snapshot.resolved_path).name,
        "tokenizer_sha256": tokenizer_snapshot.sha256,
        "fingerprint": {"tokenizer_sha256": tokenizer_snapshot.sha256},
        "tokenizer_identity": {
            "schema": FOUNDATION_TOKENIZER_IDENTITY_SCHEMA,
            "size_bytes": tokenizer_snapshot.size_bytes,
            "sha256": tokenizer_snapshot.sha256,
        },
        "preprocessing_schema": FOUNDATION_PREPROCESSING_SCHEMA,
        "preprocessing_options": {
            "deduplicate": deduplicate,
            "deduplication_backend": (
                FOUNDATION_DEDUPLICATION_BACKEND if deduplicate else "disabled"
            ),
            "maximum_characters": maximum_characters,
            "max_tokens": max_tokens,
            "max_target_tokens": max_target_tokens,
            "minimum_characters": minimum_characters,
            "reasoning_sample_share": reasoning_sample_share,
            "shard_size": shard_size,
            "validation_fraction": validation_fraction,
        },
        "language_sampling": {
            "alpha": language_sampling_alpha,
            "minimum_share": minimum_language_share,
            "weights": balance.weights,
            "counts": balance.counts,
            "warnings": list(balance.warnings),
        },
        "source_identity_schema": FOUNDATION_SOURCE_IDENTITY_SCHEMA,
        "sources_sha256": _source_identity_digest(source_manifest_records),
        "sources": source_manifest_records,
        "skipped": [
            {
                "logical_path": _logical_relative_path(
                    discovery.root,
                    entry.path,
                    role="skipped entry",
                ),
                "reason": entry.reason,
            }
            for entry in discovery.skipped
        ],
        "stats": {
            "train_records": stats.train_records,
            "validation_records": stats.validation_records,
            "languages": {
                language: asdict(language_stats)
                for language, language_stats in stats.languages.items()
            },
        },
        "reasoning": {
            "contract": "prompt-to-delimited-trace-v1",
            "languages": list(reasoning_languages),
            "records": sum(item.reasoning_records for item in stats.languages.values()),
            "sample_share": reasoning_sample_share,
            "trace_symbols": ["<think>", "</think>", "<answer>", "</answer>"],
        },
        "artifact_inventory": build_dataset_artifact_inventory(output_dir),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _fsync_file(manifest_path)
    _fsync_directory(output_dir)
    resume_journal.remove()
    return stats


def _prepare_foundation_dataset_locked(
    discovery: MonolingualDiscovery,
    tokenizer_model: str | Path,
    output_dir: str | Path,
    *,
    minimum_characters: int = 8,
    maximum_characters: int = 4000,
    max_tokens: int = 510,
    max_target_tokens: int | None = None,
    deduplicate: bool = True,
    shard_size: int = 200_000,
    validation_fraction: float = 0.002,
    language_sampling_alpha: float = DEFAULT_LANGUAGE_SAMPLING_ALPHA,
    minimum_language_share: float = 0.05,
    reasoning_sample_share: float = 0.05,
    release_name: str = FOUNDATION_RELEASE_NAME,
) -> FoundationPrepareStats:
    """Prepare one generation while the caller owns the exact output lease."""

    if not discovery.sources:
        raise ValueError(
            "The monolingual corpus has no usable training files. "
            f"root={discovery.root}. Place .txt or .jsonl files inside language-code "
            "directories."
        )
    final_output = Path(output_dir)
    if _path_exists(final_output):
        _clean_orphan_staging(final_output)
        _refuse_existing_output(final_output)
    if max_target_tokens is None:
        max_target_tokens = max_tokens
    final_output.parent.mkdir(parents=True, exist_ok=True)
    _assert_regular_directory(final_output.parent, role="output parent")
    source_snapshots = _capture_source_snapshots(discovery)
    tokenizer_snapshot = _tokenizer_snapshot(Path(tokenizer_model))
    # This immediate rediscovery also rejects a plan whose source list changed
    # before preparation acquired its artifact lock.
    _verify_source_metadata(discovery, source_snapshots)
    resume_contract = _build_resume_contract(
        discovery,
        source_snapshots,
        tokenizer_snapshot,
        minimum_characters=minimum_characters,
        maximum_characters=maximum_characters,
        max_tokens=max_tokens,
        max_target_tokens=max_target_tokens,
        deduplicate=deduplicate,
        shard_size=shard_size,
        validation_fraction=validation_fraction,
        language_sampling_alpha=language_sampling_alpha,
        minimum_language_share=minimum_language_share,
        reasoning_sample_share=reasoning_sample_share,
        release_name=release_name,
    )
    recovered = _recover_or_clean_staging(
        final_output,
        discovery,
        tokenizer_model,
        source_snapshots=source_snapshots,
        tokenizer_snapshot=tokenizer_snapshot,
        minimum_characters=minimum_characters,
        maximum_characters=maximum_characters,
        max_tokens=max_tokens,
        max_target_tokens=max_target_tokens,
        deduplicate=deduplicate,
        shard_size=shard_size,
        validation_fraction=validation_fraction,
        language_sampling_alpha=language_sampling_alpha,
        minimum_language_share=minimum_language_share,
        reasoning_sample_share=reasoning_sample_share,
        release_name=release_name,
        resume_contract=resume_contract,
    )
    if recovered.complete_stats is not None:
        return recovered.complete_stats
    _refuse_existing_output(final_output)
    partial = recovered.partial
    try:
        checkpoint_reserve_bytes = _preflight_foundation_disk_space(
            discovery,
            tokenizer_model,
            final_output.parent,
            existing_staging=partial.path if partial is not None else None,
            minimum_characters=minimum_characters,
            maximum_characters=maximum_characters,
            max_tokens=max_tokens,
            max_target_tokens=max_target_tokens,
            deduplicate=deduplicate,
        )
    except BaseException:
        # Recovery returns an exclusively locked journal.  A read-only capacity
        # refusal must release that ownership while preserving the authenticated
        # checkpoint for a later attempt on a roomier volume.
        if partial is not None:
            partial.journal.close()
        raise
    if partial is None:
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{final_output.name}.staging-",
                dir=final_output.parent,
            )
        )
        resume_state = None
        resume_journal = None
    else:
        staging_dir = partial.path
        resume_state = partial.state
        resume_journal = partial.journal
    generation_complete = False
    try:
        stats = _prepare_foundation_dataset_in_staging(
            discovery,
            tokenizer_model,
            staging_dir,
            minimum_characters=minimum_characters,
            maximum_characters=maximum_characters,
            max_tokens=max_tokens,
            max_target_tokens=max_target_tokens,
            deduplicate=deduplicate,
            shard_size=shard_size,
            validation_fraction=validation_fraction,
            language_sampling_alpha=language_sampling_alpha,
            minimum_language_share=minimum_language_share,
            reasoning_sample_share=reasoning_sample_share,
            release_name=release_name,
            source_snapshots=source_snapshots,
            tokenizer_snapshot=tokenizer_snapshot,
            checkpoint_reserve_bytes=checkpoint_reserve_bytes,
            resume_state=resume_state,
            resume_journal=resume_journal,
        )
        # Close the small post-verification window occupied by inventory and
        # manifest construction before the durable rename.
        _verify_source_snapshots(discovery, source_snapshots)
        _verify_tokenizer_snapshot(Path(tokenizer_model), tokenizer_snapshot)
        generation_complete = True
        _publish_staged_directory(staging_dir, final_output)
        return stats
    except BaseException as error:
        if resume_journal is not None:
            resume_journal.close()
        complete_resumable = _publication_failure_is_resumable(
            error,
            staging_dir,
            final_output,
            generation_complete=generation_complete,
        )
        partial_resumable = False
        if not complete_resumable:
            partial_resumable = _partial_failure_is_resumable(
                staging_dir,
                final_output,
                discovery,
                tokenizer_model,
                source_snapshots=source_snapshots,
                tokenizer_snapshot=tokenizer_snapshot,
                resume_contract=resume_contract,
                deduplicate=deduplicate,
            )
        if not complete_resumable and not partial_resumable:
            _remove_staging_path(staging_dir)
        raise


def prepare_foundation_dataset(
    discovery: MonolingualDiscovery,
    tokenizer_model: str | Path,
    output_dir: str | Path,
    *,
    minimum_characters: int = 8,
    maximum_characters: int = 4000,
    max_tokens: int = 510,
    max_target_tokens: int | None = None,
    deduplicate: bool = True,
    shard_size: int = 200_000,
    validation_fraction: float = 0.002,
    language_sampling_alpha: float = DEFAULT_LANGUAGE_SAMPLING_ALPHA,
    minimum_language_share: float = 0.05,
    reasoning_sample_share: float = 0.05,
    release_name: str = FOUNDATION_RELEASE_NAME,
) -> FoundationPrepareStats:
    """Transactionally prepare and atomically publish one foundation generation."""

    if not discovery.sources:
        raise ValueError(
            "The monolingual corpus has no usable training files. "
            f"root={discovery.root}. Place .txt or .jsonl files inside language-code "
            "directories."
        )
    final_output = Path(output_dir)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    _assert_regular_directory(final_output.parent, role="output parent")
    with _foundation_output_lock(final_output):
        return _prepare_foundation_dataset_locked(
            discovery,
            tokenizer_model,
            final_output,
            minimum_characters=minimum_characters,
            maximum_characters=maximum_characters,
            max_tokens=max_tokens,
            max_target_tokens=max_target_tokens,
            deduplicate=deduplicate,
            shard_size=shard_size,
            validation_fraction=validation_fraction,
            language_sampling_alpha=language_sampling_alpha,
            minimum_language_share=minimum_language_share,
            reasoning_sample_share=reasoning_sample_share,
            release_name=release_name,
        )


def render_prepare_report(stats: FoundationPrepareStats) -> list[str]:
    """Summarize preparation and expose discarded lines by reason."""

    lines = [
        f"Foundation dataset: train {stats.train_records:,} / "
        f"validation {stats.validation_records:,}"
    ]
    for language in sorted(stats.languages):
        language_stats = stats.languages[language]
        dropped = {
            "too_short": language_stats.too_short,
            "too_long": language_stats.too_long,
            "duplicate": language_stats.duplicate,
            "empty_after_tokenization": language_stats.empty_after_tokenization,
            **language_stats.read_rejects,
        }
        rendered = ", ".join(f"{name} {count:,}" for name, count in dropped.items() if count)
        lines.append(
            f"  {language}: read {language_stats.lines_read:,} -> "
            f"accepted {language_stats.accepted:,}"
            + (
                f" (reasoning {language_stats.reasoning_records:,})"
                if language_stats.reasoning_records
                else ""
            )
            + (f" (rejected: {rendered})" if rendered else "")
        )
    return lines


__all__ = [
    "FOUNDATION_DEDUPLICATION_BACKEND",
    "FOUNDATION_INDEX_FORMAT",
    "FOUNDATION_PREPROCESSING_SCHEMA",
    "FOUNDATION_SOURCE_IDENTITY_SCHEMA",
    "FOUNDATION_TOKENIZER_IDENTITY_SCHEMA",
    "FoundationPrepareStats",
    "LanguageStats",
    "foundation_dataset_problem",
    "prepare_foundation_dataset",
    "render_prepare_report",
]
