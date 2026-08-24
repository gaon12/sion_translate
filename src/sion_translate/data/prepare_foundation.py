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
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from sion_translate.artifacts import FOUNDATION_RELEASE_NAME
from sion_translate.data.monolingual import (
    DEFAULT_LANGUAGE_SAMPLING_ALPHA,
    MonolingualDiscovery,
    ReadStats,
    assess_language_balance,
    discover_monolingual_sources,
    iter_monolingual_lines,
    segment_text,
)
from sion_translate.data.prepare import SHARED_TARGET_INDEX_DTYPE, ShardWriter, infer_register
from sion_translate.data.integrity import (
    build_dataset_artifact_inventory,
    dataset_artifact_problem,
)
from sion_translate.data.reasoning import (
    ReasoningReadStats,
    ReasoningRecord,
    is_reasoning_jsonl,
    iter_reasoning_records,
    serialize_reasoning_record,
)
from sion_translate.fingerprint import file_sha256
from sion_translate.splitting import choose_split_for_key
from sion_translate.tokenizer import SionTokenizer, normalize_text

FOUNDATION_INDEX_FORMAT = "sion-foundation-indexed-v3"
FOUNDATION_PREPROCESSING_SCHEMA = "foundation-mixed-objectives-v6"
FOUNDATION_SOURCE_IDENTITY_SCHEMA = "corpus-relative-posix-sha256-v1"
FOUNDATION_TOKENIZER_IDENTITY_SCHEMA = "content-sha256-v1"
FOUNDATION_DEDUPLICATION_BACKEND = "sqlite-blake2b-128-v1"
_DEDUPLICATION_DATABASE = ".foundation-dedup.sqlite3"
_DEDUPLICATION_CACHE_KIB = 8 * 1024
_DEDUPLICATION_COMMIT_INTERVAL = 100_000


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

    def merge_read(self, stats: ReadStats) -> None:
        for reason, count in stats.reasons().items():
            self.read_rejects[reason] = self.read_rejects.get(reason, 0) + count

    def merge_reasoning_read(self, stats: ReasoningReadStats) -> None:
        self.reasoning_rejected += stats.rejected
        for reason, count in (
            ("reasoning_blank", stats.blank),
            ("reasoning_malformed_json", stats.malformed_json),
            ("reasoning_non_object", stats.non_object),
            ("reasoning_invalid_record", stats.invalid_record),
        ):
            if count:
                self.read_rejects[reason] = self.read_rejects.get(reason, 0) + count


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


class _DiskDigestIndex:
    """Bound deduplication memory with a disk-backed unique digest index.

    The index is scratch state inside the unpublished staging directory. SQLite
    enforces uniqueness deterministically while its negative ``cache_size``
    value caps the page cache in KiB. Durability is deliberately disabled: a
    failed preparation discards the complete staging generation, so recovering
    this private index would provide no value.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection = sqlite3.connect(path)
        self._pending = 0
        try:
            self._connection.execute("PRAGMA journal_mode=OFF")
            self._connection.execute("PRAGMA synchronous=OFF")
            self._connection.execute("PRAGMA temp_store=FILE")
            self._connection.execute(f"PRAGMA cache_size=-{_DEDUPLICATION_CACHE_KIB}")
            self._connection.execute("PRAGMA locking_mode=EXCLUSIVE")
            self._connection.execute("CREATE TABLE digests (digest BLOB PRIMARY KEY) WITHOUT ROWID")
            self._connection.execute("BEGIN IMMEDIATE")
        except BaseException:
            self._connection.close()
            raise

    def add(self, digest: bytes) -> bool:
        """Insert a digest and return ``True`` only for its first occurrence."""

        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO digests (digest) VALUES (?)",
            (sqlite3.Binary(digest),),
        )
        inserted = cursor.rowcount == 1
        self._pending += 1
        if self._pending >= _DEDUPLICATION_COMMIT_INTERVAL:
            self._connection.commit()
            self._connection.execute("BEGIN IMMEDIATE")
            self._pending = 0
        return inserted

    def close(self) -> None:
        """Close and remove every private SQLite artifact before publication."""

        try:
            self._connection.commit()
        finally:
            self._connection.close()
        for suffix in ("", "-journal", "-shm", "-wal"):
            candidate = Path(f"{self.path}{suffix}")
            if _path_exists(candidate):
                candidate.unlink()


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
        discovery,
        language_sampling_alpha=language_sampling_alpha,
        minimum_language_share=minimum_language_share,
        reasoning_sample_share=reasoning_sample_share,
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
) -> FoundationPrepareStats | None:
    valid: list[tuple[Path, FoundationPrepareStats]] = []
    for candidate in _staging_candidates(output_dir):
        try:
            _regular_staging_tree(candidate)
        except (OSError, RuntimeError, ValueError) as error:
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
            _remove_staging_path(candidate)
            continue
        valid.append((candidate, recovered_stats))

    if not valid:
        return None
    selected, recovered_stats = max(
        valid,
        key=lambda item: (item[0].stat().st_mtime_ns, item[0].name),
    )
    # Leave only the selected recoverable generation before the atomic rename;
    # a crash immediately after publication must not strand other large trees.
    for candidate, _stats in valid:
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
    return recovered_stats


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

    deduplication_index = (
        _DiskDigestIndex(output_dir / _DEDUPLICATION_DATABASE) if deduplicate else None
    )
    writers: dict[str, ShardWriter] = {}
    try:
        for split in ("train", "validation"):
            writers[split] = ShardWriter(
                output_dir,
                split,
                shard_size,
                language_to_id,
                shared_targets=True,
            )
    except BaseException:
        _close_shard_writers(writers, suppress_errors=True)
        if deduplication_index is not None:
            deduplication_index.close()
        raise
    stats = FoundationPrepareStats(languages={language: LanguageStats() for language in languages})
    source_ids = {source.path: index for index, source in enumerate(discovery.sources)}
    source_record_counts = {source_id: 0 for source_id in source_ids.values()}

    def record_segment(
        text: str,
        *,
        language: str,
        language_stats: LanguageStats,
        source_id: int,
    ) -> None:
        """Write one segment, rejecting duplicates and empty token sequences here."""

        if deduplication_index is not None:
            digest = _text_digest(language, text)
            if not deduplication_index.add(digest):
                language_stats.duplicate += 1
                return
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

        digest = _reasoning_digest(language, record.prompt, record.think, record.answer)
        if deduplication_index is not None and not deduplication_index.add(digest):
            language_stats.duplicate += 1
            return
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
        for source in discovery.sources:
            language = source.language
            language_stats = stats.languages[language]
            if is_reasoning_jsonl(source.path):
                reasoning_read_stats = ReasoningReadStats()
                for record in iter_reasoning_records(
                    source.path,
                    expected_language=language,
                    stats=reasoning_read_stats,
                ):
                    record_reasoning(
                        record,
                        language=language,
                        language_stats=language_stats,
                        source_id=source_ids[source.path],
                    )
                language_stats.lines_read += reasoning_read_stats.physical_lines
                language_stats.merge_reasoning_read(reasoning_read_stats)
                continue
            read_stats = ReadStats()
            for raw_text in iter_monolingual_lines(source.path, stats=read_stats):
                language_stats.lines_read += 1
                document = normalize_text(raw_text)
                if not _is_usable(document):
                    language_stats.too_short += 1
                    continue
                # Segment long documents instead of discarding or truncating them. A
                # historical audit found that whole-line rejection lost 97.3% of
                # e_gov, 92.8% of aozora, and 68.0% of kowiki characters; combined with
                # truncation, the audited pipeline lost 25.8% of all characters.
                segments = segment_text(
                    document,
                    maximum_characters=maximum_characters,
                    minimum_characters=minimum_characters,
                )
                if not segments:
                    language_stats.too_short += 1
                    continue
                if len(segments) > 1:
                    language_stats.segmented_documents += 1
                language_stats.segments += len(segments)
                for text in segments:
                    record_segment(
                        text,
                        language=language,
                        language_stats=language_stats,
                        source_id=source_ids[source.path],
                    )
            language_stats.merge_read(read_stats)
    except BaseException:
        _close_shard_writers(writers, suppress_errors=True)
        if deduplication_index is not None:
            deduplication_index.close()
        raise
    else:
        try:
            _close_shard_writers(writers, suppress_errors=False)
        finally:
            if deduplication_index is not None:
                deduplication_index.close()

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
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stats


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
    """Transactionally prepare foundation shards and publish one complete generation."""

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
    )
    if recovered is not None:
        return recovered
    _refuse_existing_output(final_output)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output.name}.staging-",
            dir=final_output.parent,
        )
    )
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
        )
        # Close the small post-verification window occupied by inventory and
        # manifest construction before the durable rename.
        _verify_source_snapshots(discovery, source_snapshots)
        _verify_tokenizer_snapshot(Path(tokenizer_model), tokenizer_snapshot)
        generation_complete = True
        _publish_staged_directory(staging_dir, final_output)
        return stats
    except BaseException as error:
        if not _publication_failure_is_resumable(
            error,
            staging_dir,
            final_output,
            generation_complete=generation_complete,
        ):
            _remove_staging_path(staging_dir)
        raise


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
