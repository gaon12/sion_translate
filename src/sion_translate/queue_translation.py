"""Resumable, auditable translation of monolingual JSONL queues.

Queue files are immutable inputs.  Each source row receives a result record,
while only rows that pass forward and round-trip checks are copied into
manifest-gated private training parts. Progress is committed after atomic shard
writes so a stopped multi-day run can safely resume from its byte offset.
Accepted shards first live in a private run namespace. Only fully published
parts from an approved teacher policy are copied into the top-level training
namespace. A two-phase pending publish prevents partial data from becoming
training-visible.
"""

# Queue manifests are persisted JSON objects and SacreBLEU exposes incomplete
# annotations. Validate their shape at runtime and contain Unknown types here.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
from typing import Any, BinaryIO, Protocol, cast

from sacrebleu.metrics.chrf import CHRF

from sion_translate.data.quality import (
    assess_pair,
    canonical_text,
    language_fraction,
)
from sion_translate.evaluation import multiset_f1, numeric_tokens
from sion_translate.language_tags import canonicalize_language_pair, canonicalize_language_tag
from sion_translate.scripts_registry import (
    canonicalize_script_policy_name,
    script_letter_count,
    script_of,
    scripts_for_language,
)
from sion_translate.structured import structured_similarity


MANIFEST_SCHEMA = "sion-translation-queue-v1"
RESULT_SCHEMA = "sion-translation-result-v1"
ACCEPTED_OWNER_SCHEMA = "sion-accepted-namespace-owner-v1"
ACCEPTED_SHARD_PREFIX = "queue_bt_"
LEGACY_ACCEPTED_SHARD_PREFIX = "bt_"
PRIVATE_ACCEPTED_DIRNAME = ".queue-runs"
SOURCE_SNAPSHOT_FILENAME = ".queue-source.snapshot.jsonl"
SOURCE_INDEX_FILENAME = ".queue-source.index.sqlite3"
LEGACY_PUBLIC_MARKER = "verified-legacy-public-parts-v1"
# Keep a single adversarial identifier from dominating the disk-backed index or
# being copied into every diagnostic and result artifact.
MAX_QUEUE_ID_UTF8_BYTES = 4 * 1024
PIPELINE_VERSION = 3
SIGNATURE_VERSION = 2
RUN_LOCK_FILENAME = ".queue-translation.lock"
ACCEPTED_LOCK_FILENAME = RUN_LOCK_FILENAME
_PROVENANCE_PLACEHOLDERS = frozenset({"n/a", "na", "none", "tbd", "unknown", "unset"})
_CONTENT_IDENTITY_FIELDS = ("path", "size", "sha256")
_RUNTIME_IDENTITY_FIELDS = (*_CONTENT_IDENTITY_FIELDS, "device", "inode", "mtime_ns")
_ARTIFACT_RUNTIME_FIELDS = (
    "device",
    "inode",
    "mtime_ns",
    "nlink",
    "mode",
    "file_attributes",
)
_LOCAL_INTEGRITY_DESCRIPTOR = {
    "algorithm": "sha256",
    "scope": "configuration_only",
    "authentication": False,
}
_CHRF = CHRF(word_order=0)


class RetryableQueueTranslationError(RuntimeError):
    """Abort only the current uncommitted shard after a transient runtime fault."""


class PermanentQueueRowError(ValueError):
    """Identify an input-specific translation failure that is safe to persist."""


def _manifest_run_id(manifest: Mapping[str, Any]) -> str:
    """Return a path-safe queue run identifier."""

    value = manifest.get("run_id")
    if (
        not isinstance(value, str)
        or len(value) != 16
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("queue manifest run_id must be 16 lowercase hexadecimal characters")
    return value


def _exact_non_negative_integer(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _validate_artifact_shape(value: object, *, field: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an artifact object")
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{field}.path must be a non-empty string")
    _exact_non_negative_integer(value.get("size"), field=f"{field}.size")
    _exact_non_negative_integer(value.get("rows"), field=f"{field}.rows")
    digest = value.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{field}.sha256 must be a lowercase SHA-256")
    recorded_runtime_fields = [name for name in _ARTIFACT_RUNTIME_FIELDS if name in value]
    if recorded_runtime_fields and len(recorded_runtime_fields) != len(_ARTIFACT_RUNTIME_FIELDS):
        raise ValueError(f"{field} must record every runtime identity field or none")
    for name in recorded_runtime_fields:
        _exact_non_negative_integer(value.get(name), field=f"{field}.{name}")


def _expected_teacher_review_required(manifest: Mapping[str, Any]) -> bool:
    review = manifest.get("teacher_review")
    if not isinstance(review, Mapping):
        return False
    stats = manifest.get("stats")
    progress = manifest.get("progress")
    if not isinstance(stats, Mapping) or not isinstance(progress, Mapping):
        return False
    generated = stats.get("generated")
    if type(generated) is not int:
        accepted = stats.get("accepted")
        rejected = stats.get("rejected")
        if type(accepted) is not int or type(rejected) is not int:
            return False
        generated = accepted + rejected
    pilot_rows = review.get("pilot_rows")
    if type(pilot_rows) is not int:
        return False
    return generated >= pilot_rows or (progress.get("complete") is True and generated > 0)


def _validate_manifest_control_state(manifest: Mapping[str, Any]) -> None:
    """Validate mutable control fields before any resume or approval decision."""

    progress = manifest.get("progress")
    if not isinstance(progress, Mapping):
        raise ValueError("queue manifest progress must be an object")
    for field in ("completed_rows", "source_byte_offset", "next_part"):
        _exact_non_negative_integer(
            progress.get(field),
            field=f"queue manifest progress.{field}",
        )
    if type(progress.get("complete")) is not bool:
        raise ValueError("queue manifest progress.complete must be a boolean")

    stats = manifest.get("stats")
    if not isinstance(stats, Mapping):
        raise ValueError("queue manifest stats must be an object")
    required_stats = ("processed", "accepted", "rejected", "errors", "skipped_existing")
    stat_fields = required_stats + (("generated",) if "generated" in stats else ())
    for field in stat_fields:
        _exact_non_negative_integer(
            stats.get(field),
            field=f"queue manifest stats.{field}",
        )

    parts = manifest.get("parts")
    if parts is not None:
        if not isinstance(parts, list):
            raise ValueError("queue manifest parts must be a list")
        for index, part in enumerate(parts):
            if not isinstance(part, Mapping):
                raise ValueError(f"queue manifest parts[{index}] must be an object")
            for field in ("part", "source_start_index", "source_rows"):
                _exact_non_negative_integer(
                    part.get(field),
                    field=f"queue manifest parts[{index}].{field}",
                )
            if "source_end_byte_offset" in part:
                _exact_non_negative_integer(
                    part.get("source_end_byte_offset"),
                    field=f"queue manifest parts[{index}].source_end_byte_offset",
                )
            if "generated_rows" in part:
                _exact_non_negative_integer(
                    part.get("generated_rows"),
                    field=f"queue manifest parts[{index}].generated_rows",
                )
            if type(part.get("published")) is not bool:
                raise ValueError(f"queue manifest parts[{index}].published must be a boolean")
            _validate_artifact_shape(
                part.get("result"), field=f"queue manifest parts[{index}].result"
            )
            _validate_artifact_shape(
                part.get("accepted"),
                field=f"queue manifest parts[{index}].accepted",
            )
            training = part.get("training")
            if training is not None:
                _validate_artifact_shape(
                    training,
                    field=f"queue manifest parts[{index}].training",
                )
            status_counts = part.get("status_counts")
            if status_counts is not None:
                if not isinstance(status_counts, Mapping):
                    raise ValueError(
                        f"queue manifest parts[{index}].status_counts must be an object"
                    )
                for status in ("accepted", "rejected", "error", "skipped_existing"):
                    _exact_non_negative_integer(
                        status_counts.get(status),
                        field=f"queue manifest parts[{index}].status_counts.{status}",
                    )

    training_set = manifest.get("training_set")
    if training_set is not None:
        _validate_artifact_shape(training_set, field="queue manifest training_set")

    review = manifest.get("teacher_review")
    if review is None:
        return
    if not isinstance(review, Mapping):
        raise ValueError("queue manifest teacher_review must be an object or null")
    pilot_rows = review.get("pilot_rows")
    if type(pilot_rows) is not int or pilot_rows <= 0:
        raise ValueError("queue manifest teacher_review.pilot_rows must be a positive integer")
    for field in ("review_required", "approved"):
        if type(review.get(field)) is not bool:
            raise ValueError(f"queue manifest teacher_review.{field} must be a boolean")
    expected_required = _expected_teacher_review_required(manifest)
    if review.get("review_required") is not expected_required:
        raise ValueError(
            "queue manifest teacher_review.review_required contradicts the verified pilot progress"
        )
    approved_at = review.get("approved_at")
    approved_by = review.get("approved_by")
    if review["approved"] is True:
        if not isinstance(approved_at, str) or not approved_at.strip():
            raise ValueError("approved teacher review requires a non-empty approved_at")
        if not isinstance(approved_by, str) or not canonical_text(approved_by):
            raise ValueError("approved teacher review requires a non-empty approved_by")
    elif approved_at is not None or approved_by is not None:
        raise ValueError("unapproved teacher review cannot record approval metadata")


def _assert_plain_file(path: Path, *, label: str) -> None:
    """Reject links and Windows reparse points used in private queue state."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {path}") from exc
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(metadata.st_mode) or file_attributes & reparse_point:
        raise ValueError(f"{label} must not be a symbolic link or reparse point: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    if metadata.st_nlink != 1:
        raise ValueError(f"{label} must not have hard-link aliases: {path}")


def _ensure_private_run_directory(accepted_dir: Path, run_id: str) -> Path:
    """Create and validate the manifest-gated run directory below accepted_dir."""

    accepted_root = accepted_dir.resolve(strict=True)
    private_root = accepted_dir / PRIVATE_ACCEPTED_DIRNAME
    run_root = private_root / run_id
    for directory, label in (
        (private_root, "private accepted root"),
        (run_root, "private accepted run directory"),
    ):
        directory.mkdir(exist_ok=True)
        metadata = os.lstat(directory)
        file_attributes = getattr(metadata, "st_file_attributes", 0)
        reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(metadata.st_mode) or file_attributes & reparse_point:
            raise ValueError(f"{label} must not be a symbolic link or reparse point: {directory}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} is not a directory: {directory}")
        try:
            directory.resolve(strict=True).relative_to(accepted_root)
        except ValueError as exc:
            raise ValueError(f"{label} escapes accepted_dir: {directory}") from exc
    return run_root


class TranslatorLike(Protocol):
    """The subset of :class:`Translator` used by the queue runner."""

    tokenizer: Any

    @property
    def translation_model_path(self) -> str: ...

    @property
    def tokenizer_model_path(self) -> str: ...

    @property
    def tokenizer_metadata_path(self) -> str | None: ...

    @property
    def token_features_path(self) -> str | None: ...

    @property
    def translation_directions(self) -> Sequence[Sequence[str]]: ...

    @property
    def export_metadata(self) -> Mapping[str, Any]: ...

    @property
    def tokenizer_metadata(self) -> Mapping[str, Any] | None: ...

    def translate(
        self,
        texts: Sequence[str],
        *,
        source_language: str,
        target_language: str,
        num_beams: int,
        max_new_tokens: int,
        batch_size: int,
        max_output_length_ratio: float,
        max_output_length_margin: int,
    ) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class QueueTranslationOptions:
    batch_size: int = 16
    shard_size: int = 1_000
    num_beams: int = 1
    max_new_tokens: int = 128
    max_output_length_ratio: float = 2.0
    max_output_length_margin: int = 12
    roundtrip_enabled: bool = True
    roundtrip_num_beams: int = 1
    roundtrip_max_new_tokens: int = 128
    roundtrip_max_output_length_ratio: float = 2.0
    roundtrip_max_output_length_margin: int = 12
    min_roundtrip_score: float = 0.65
    min_pair_score: int = 80
    min_target_language_fraction: float = 0.50
    required_target_scripts: tuple[tuple[str, str, int], ...] = ()
    min_structured_similarity: float = 1.0

    def validate(self) -> None:
        for field, value in (
            ("batch_size", self.batch_size),
            ("shard_size", self.shard_size),
            ("num_beams", self.num_beams),
            ("roundtrip_num_beams", self.roundtrip_num_beams),
            ("max_new_tokens", self.max_new_tokens),
            ("roundtrip_max_new_tokens", self.roundtrip_max_new_tokens),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field, value in (
            ("max_output_length_margin", self.max_output_length_margin),
            ("roundtrip_max_output_length_margin", self.roundtrip_max_output_length_margin),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        for field, value in (
            ("max_output_length_ratio", self.max_output_length_ratio),
            ("roundtrip_max_output_length_ratio", self.roundtrip_max_output_length_ratio),
            ("min_roundtrip_score", self.min_roundtrip_score),
            ("min_target_language_fraction", self.min_target_language_fraction),
            ("min_structured_similarity", self.min_structured_similarity),
        ):
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{field} must be a finite number")
        if self.max_output_length_ratio <= 0 or self.roundtrip_max_output_length_ratio <= 0:
            raise ValueError("output length ratios must be positive")
        if not 0.0 <= self.min_roundtrip_score <= 1.0:
            raise ValueError("min_roundtrip_score must be in [0, 1]")
        if type(self.min_pair_score) is not int or not 0 <= self.min_pair_score <= 100:
            raise ValueError("min_pair_score must be an integer in [0, 100]")
        if not 0.0 <= self.min_target_language_fraction <= 1.0:
            raise ValueError("min_target_language_fraction must be in [0, 1]")
        seen_script_requirements: set[tuple[str, str]] = set()
        for index, requirement in enumerate(self.required_target_scripts):
            raw_requirement: Any = requirement
            if (
                not isinstance(raw_requirement, Sequence)
                or isinstance(raw_requirement, (str, bytes))
                or len(raw_requirement) != 3
            ):
                raise ValueError(
                    "required_target_scripts entries must be (language, script, minimum) triples"
                )
            raw_language, raw_script, minimum = cast(Sequence[object], raw_requirement)
            language = canonicalize_language_tag(
                raw_language,
                field=f"required_target_scripts[{index}].language",
            )
            try:
                script = canonicalize_script_policy_name(raw_script)
            except ValueError as exc:
                raise ValueError(f"required_target_scripts[{index}].script is invalid") from exc
            if type(minimum) is not int or minimum <= 0:
                raise ValueError(
                    f"required_target_scripts[{index}].minimum must be a positive integer"
                )
            key = (language, script)
            if key in seen_script_requirements:
                raise ValueError(
                    "required_target_scripts contains a duplicate canonical language/script rule"
                )
            seen_script_requirements.add(key)
        if not 0.0 <= self.min_structured_similarity <= 1.0:
            raise ValueError("min_structured_similarity must be in [0, 1]")

    def target_script_requirements(self, target_language: str) -> dict[str, int]:
        """Return explicit writing-system requirements for one canonical target tag."""

        canonical_target = canonicalize_language_tag(
            target_language,
            field="target_language",
        )
        return {
            canonicalize_script_policy_name(script): minimum
            for language, script, minimum in self.required_target_scripts
            if canonicalize_language_tag(language, field="required_target_scripts.language")
            == canonical_target
        }


def sha256_file(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _stable_file_artifact(path: Path, *, label: str) -> dict[str, Any]:
    """Hash one plain file and reject path or inode changes during the read."""

    _assert_plain_file(path, label=label)
    before = os.lstat(path)
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened_before = os.fstat(descriptor)
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    identities = (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink),
        (
            opened_before.st_dev,
            opened_before.st_ino,
            opened_before.st_size,
            opened_before.st_mtime_ns,
            opened_before.st_nlink,
        ),
        (
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_size,
            opened_after.st_mtime_ns,
            opened_after.st_nlink,
        ),
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink),
    )
    if len(set(identities)) != 1 or size != after.st_size:
        raise RuntimeError(f"{label} changed while its content identity was captured: {path}")
    return {
        "path": str(path.resolve(strict=True)),
        "size": size,
        "sha256": digest.hexdigest(),
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
        "nlink": after.st_nlink,
        "mode": stat.S_IMODE(after.st_mode),
        "file_attributes": getattr(after, "st_file_attributes", 0),
    }


def _stable_digest(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _signature_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude volatile file metadata while retaining content identity."""

    source = configuration.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("queue configuration has no source identity")
    stable_source = {key: source.get(key) for key in ("path", "size", "sha256")}
    stable = {
        **dict(configuration),
        "source": stable_source,
    }
    source_snapshot = configuration.get("source_snapshot")
    if isinstance(source_snapshot, Mapping):
        stable["source_snapshot"] = {
            key: source_snapshot.get(key) for key in ("path", "size", "sha256")
        }
    return stable


def _validate_run_signature_binding(manifest: Mapping[str, Any]) -> None:
    """Bind current run IDs to the signed configuration and preserve audited legacy IDs."""

    run_id = _manifest_run_id(manifest)
    signature = manifest.get("run_signature")
    if (
        not isinstance(signature, str)
        or len(signature) != 64
        or signature != signature.lower()
        or any(character not in "0123456789abcdef" for character in signature)
    ):
        raise ValueError("queue manifest run_signature must be a lowercase SHA-256")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("queue manifest has no valid configuration")
    prefix = configuration.get("accepted_shard_prefix")
    if prefix == ACCEPTED_SHARD_PREFIX:
        if configuration.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError("current queue prefix requires the current pipeline version")
        if run_id != signature[:16]:
            raise ValueError("current queue manifest run_id does not match run_signature")
        if configuration.get("legacy_public_marker") is not None:
            raise ValueError("current queue manifest cannot carry a legacy public marker")
        return
    if prefix == LEGACY_ACCEPTED_SHARD_PREFIX:
        if configuration.get("legacy_public_marker") != LEGACY_PUBLIC_MARKER:
            raise ValueError("legacy queue prefix requires an explicit verified migration marker")
        legacy_signature = configuration.get("legacy_run_signature")
        legacy_run_id = configuration.get("legacy_run_id")
        if (
            not isinstance(legacy_signature, str)
            or len(legacy_signature) != 64
            or legacy_signature != legacy_signature.lower()
            or any(character not in "0123456789abcdef" for character in legacy_signature)
            or legacy_run_id != legacy_signature[:16]
            or run_id != legacy_run_id
        ):
            raise ValueError("legacy queue run_id is not bound to its original signature")
        return
    if prefix is None and _is_exact_legacy_manifest(manifest):
        if run_id != signature[:16]:
            raise ValueError("legacy queue manifest run_id does not match run_signature")
        return
    raise ValueError("queue manifest accepted_shard_prefix is not an allowed policy value")


def _content_identity(value: object, *, field: str) -> dict[str, Any]:
    """Verify one recorded file identity against its current bytes and stat."""

    if not isinstance(value, Mapping):
        raise ValueError(f"queue run_metadata.{field} must be a content identity object")
    path = value.get("path")
    size = value.get("size")
    digest = value.get("sha256")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"queue run_metadata.{field}.path must be a non-empty string")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"queue run_metadata.{field}.size must be a non-negative integer")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"queue run_metadata.{field}.sha256 must be a lowercase SHA-256")
    artifact_path = Path(path)
    try:
        resolved = artifact_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"queue run_metadata.{field}.path does not exist: {artifact_path}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"queue run_metadata.{field}.path is not a file: {resolved}")
    before = resolved.stat()
    observed_digest = sha256_file(resolved)
    after = resolved.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise ValueError(f"queue runtime artifact changed while hashing: {resolved}")
    if size != after.st_size or digest != observed_digest:
        raise ValueError(
            f"queue run_metadata.{field} does not match the current artifact bytes: {resolved}"
        )
    identity: dict[str, Any] = {"path": str(resolved), "size": size, "sha256": digest}
    runtime_stat = {
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }
    recorded_stat_fields = [name for name in runtime_stat if name in value]
    if recorded_stat_fields and len(recorded_stat_fields) != len(runtime_stat):
        raise ValueError(f"queue run_metadata.{field} must record all runtime stat fields or none")
    for name, observed in runtime_stat.items():
        if name not in value:
            continue
        recorded = value.get(name)
        if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded != observed:
            raise ValueError(
                f"queue run_metadata.{field}.{name} does not match the current artifact"
            )
        identity[name] = recorded
    return identity


_TRANSLATOR_ARTIFACT_PATH_ATTRIBUTES = {
    "translation_model": "translation_model_path",
    "tokenizer": "tokenizer_model_path",
    "tokenizer_metadata": "tokenizer_metadata_path",
    "token_features": "token_features_path",
}
_TRANSLATOR_ARTIFACT_IDENTITY_ATTRIBUTES = {
    "translation_model": "translation_model_identity",
    "tokenizer": "tokenizer_model_identity",
    "tokenizer_metadata": "tokenizer_metadata_identity",
    "token_features": "token_features_identity",
}


def _bind_translator_artifact(
    translator: TranslatorLike,
    *,
    field: str,
    identity: Mapping[str, Any] | None,
    verify_load_identity: bool,
) -> None:
    """Bind recorded bytes to the path and load-time identity used by Translator."""

    attribute = _TRANSLATOR_ARTIFACT_PATH_ATTRIBUTES[field]
    loaded_path = getattr(translator, attribute, None)
    if identity is None:
        if loaded_path is not None:
            raise ValueError(
                f"translator loaded {field} but queue run_metadata records it as absent"
            )
        if (
            verify_load_identity
            and getattr(
                translator,
                _TRANSLATOR_ARTIFACT_IDENTITY_ATTRIBUTES[field],
                None,
            )
            is not None
        ):
            raise ValueError(
                f"Translator load identity records {field} but queue run_metadata records it as absent"
            )
        return
    if not isinstance(loaded_path, (str, Path)):
        raise ValueError(f"translator does not expose its loaded {field} path")
    try:
        resolved_loaded = Path(loaded_path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"translator loaded {field} path is missing: {loaded_path}"
        ) from exc
    if resolved_loaded != Path(str(identity["path"])).resolve(strict=True):
        raise ValueError(
            f"queue run_metadata.{field} differs from the artifact loaded by the translator"
        )
    if not verify_load_identity:
        return
    loaded_identity = getattr(
        translator,
        _TRANSLATOR_ARTIFACT_IDENTITY_ATTRIBUTES[field],
        None,
    )
    if not isinstance(loaded_identity, Mapping):
        raise ValueError(f"Translator does not expose a verified load identity for {field}")
    fields = (
        _RUNTIME_IDENTITY_FIELDS
        if all(name in identity for name in _RUNTIME_IDENTITY_FIELDS)
        else _CONTENT_IDENTITY_FIELDS
    )
    if any(loaded_identity.get(name) != identity.get(name) for name in fields):
        raise ValueError(
            f"queue run_metadata.{field} differs from the artifact identity loaded by Translator"
        )


def _bind_translator_tokenizer_object(
    translator: TranslatorLike,
    tokenizer_identity: Mapping[str, Any],
) -> None:
    tokenizer = getattr(translator, "tokenizer", None)
    loaded_path = getattr(tokenizer, "model_path", None)
    if not isinstance(loaded_path, (str, Path)):
        raise ValueError("translator tokenizer does not expose its loaded model path")
    if Path(loaded_path).resolve(strict=True) != Path(str(tokenizer_identity["path"])).resolve(
        strict=True
    ):
        raise ValueError(
            "queue run_metadata.tokenizer differs from the tokenizer used by the translator"
        )


def _verify_tokenizer_metadata_payload(
    identity: Mapping[str, Any],
    translator_metadata: Mapping[str, Any] | None,
) -> None:
    path = Path(str(identity["path"]))
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid tokenizer metadata artifact: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"tokenizer metadata artifact must be an object: {path}")
    if translator_metadata is None or dict(payload) != dict(translator_metadata):
        raise ValueError("translator tokenizer metadata differs from its recorded sidecar bytes")


def _canonical_direction_graph(value: object, *, field: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{field} must be a non-empty ordered direction sequence")
    directions: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_direction in enumerate(value):
        direction = canonicalize_language_pair(
            raw_direction,
            field=f"{field}[{index}]",
        )
        if direction in seen:
            raise ValueError(f"{field} contains a duplicate canonical direction: {direction!r}")
        seen.add(direction)
        directions.append(direction)
    return tuple(directions)


def _metadata_direction_graph(value: object, *, field: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        return ()
    raw_directions = value.get("translation_directions")
    if raw_directions is None:
        return ()
    return _canonical_direction_graph(raw_directions, field=f"{field}.translation_directions")


def _validated_run_lineage(
    run_metadata: Mapping[str, Any],
    translator: TranslatorLike,
    artifact_directions: Sequence[tuple[str, str]],
    *,
    legacy_manifest: bool,
    allow_unverified_translator: bool,
) -> dict[str, Any]:
    """Validate provenance and bind it to Translator's captured load identities."""

    from sion_translate.inference import Translator

    # Subclasses can bypass ``Translator.__init__`` and fabricate the captured
    # identity attributes, so only the concrete loader owns the verified path.
    verified_runtime = type(translator) is Translator
    if not verified_runtime and not allow_unverified_translator:
        raise TypeError(
            "queue lineage requires sion_translate.inference.Translator with captured "
            "load-time identities; pass allow_unverified_translator=True only for an "
            "explicitly unverified custom runtime"
        )

    lineage: dict[str, Any] = {
        "runtime_verification": (
            "translator_load_identity_v1" if verified_runtime else "unverified_custom_translator"
        )
    }
    for field in ("source_dataset", "source_revision", "source_license"):
        value = run_metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"queue run_metadata.{field} must be an explicit non-empty string")
        normalized = value.strip()
        if normalized.casefold() in _PROVENANCE_PLACEHOLDERS:
            raise ValueError(f"queue run_metadata.{field} cannot use an unknown placeholder")
        lineage[field] = value

    lineage["translation_model"] = _content_identity(
        run_metadata.get("translation_model"),
        field="translation_model",
    )
    lineage["tokenizer"] = _content_identity(
        run_metadata.get("tokenizer"),
        field="tokenizer",
    )
    if "token_features" not in run_metadata:
        raise ValueError("queue run_metadata must explicitly record token_features")
    raw_features = run_metadata.get("token_features")
    lineage["token_features"] = (
        None if raw_features is None else _content_identity(raw_features, field="token_features")
    )
    raw_tokenizer_metadata = run_metadata.get("tokenizer_metadata")
    if not legacy_manifest and "tokenizer_metadata" not in run_metadata:
        raise ValueError("queue run_metadata must explicitly record tokenizer_metadata")
    lineage["tokenizer_metadata"] = (
        None
        if raw_tokenizer_metadata is None
        else _content_identity(raw_tokenizer_metadata, field="tokenizer_metadata")
    )
    _bind_translator_artifact(
        translator,
        field="translation_model",
        identity=lineage["translation_model"],
        verify_load_identity=verified_runtime,
    )
    _bind_translator_artifact(
        translator,
        field="tokenizer",
        identity=lineage["tokenizer"],
        verify_load_identity=verified_runtime,
    )
    _bind_translator_artifact(
        translator,
        field="token_features",
        identity=lineage["token_features"],
        verify_load_identity=verified_runtime,
    )
    _bind_translator_artifact(
        translator,
        field="tokenizer_metadata",
        identity=lineage["tokenizer_metadata"],
        verify_load_identity=verified_runtime,
    )
    _bind_translator_tokenizer_object(translator, lineage["tokenizer"])
    translator_token_features = getattr(translator, "token_features", None)
    if lineage["token_features"] is None and translator_token_features is not None:
        raise ValueError(
            "translator loaded token_features but queue run_metadata records them as absent"
        )
    translator_tokenizer_metadata = getattr(translator, "tokenizer_metadata", None)
    if lineage["tokenizer_metadata"] is None and translator_tokenizer_metadata is not None:
        raise ValueError(
            "translator loaded tokenizer metadata but queue run_metadata records it as absent"
        )
    if lineage["tokenizer_metadata"] is not None:
        _verify_tokenizer_metadata_payload(
            lineage["tokenizer_metadata"],
            translator_tokenizer_metadata,
        )

    canonical_directions = tuple(artifact_directions)
    raw_recorded_directions = run_metadata.get("translation_directions")
    if raw_recorded_directions is None:
        if not legacy_manifest:
            raise ValueError("new queue runs require run_metadata.translation_directions")
    else:
        recorded_directions = _canonical_direction_graph(
            raw_recorded_directions,
            field="queue run_metadata.translation_directions",
        )
        if recorded_directions != canonical_directions:
            raise ValueError(
                "queue run_metadata.translation_directions differs from the translator graph"
            )

    export_graph = _metadata_direction_graph(
        getattr(translator, "export_metadata", None),
        field="translator export metadata",
    )
    tokenizer_graph = _metadata_direction_graph(
        getattr(translator, "tokenizer_metadata", None),
        field="translator tokenizer metadata",
    )
    raw_graph_source = run_metadata.get("translation_graph_source")
    if raw_graph_source is None and legacy_manifest:
        if export_graph == canonical_directions:
            graph_source = "translation_model"
        elif tokenizer_graph == canonical_directions and lineage["tokenizer_metadata"] is not None:
            graph_source = "tokenizer_metadata"
        else:
            raise ValueError(
                "legacy queue run cannot verify its translation graph from the recorded "
                "model or tokenizer-metadata identities"
            )
    else:
        if raw_graph_source not in {"translation_model", "tokenizer_metadata"}:
            raise ValueError(
                "queue run_metadata.translation_graph_source must be translation_model or "
                "tokenizer_metadata"
            )
        graph_source = cast(str, raw_graph_source)

    if graph_source == "translation_model":
        if export_graph != canonical_directions:
            raise ValueError(
                "translator export metadata does not verify the advertised direction graph"
            )
    elif lineage["tokenizer_metadata"] is None or tokenizer_graph != canonical_directions:
        raise ValueError(
            "translator tokenizer metadata does not verify the advertised direction graph"
        )

    lineage["translation_directions"] = [list(direction) for direction in canonical_directions]
    lineage["translation_graph_source"] = graph_source
    return lineage


def load_queue_run_metadata(output_dir: str | Path) -> dict[str, Any] | None:
    """Return resume metadata after checking the local configuration digest.

    This digest detects ordinary corruption. It is not authentication and does
    not defend against a writer that can replace both the JSON and its digest.
    """

    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        loaded: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid queue manifest: {manifest_path}") from exc
    if not isinstance(loaded, Mapping) or loaded.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported queue manifest: {manifest_path}")
    manifest = cast(Mapping[str, Any], loaded)
    _manifest_run_id(manifest)
    _validate_manifest_control_state(manifest)
    if manifest.get("parts") is None and not _is_exact_legacy_manifest(manifest):
        raise ValueError(
            "current queue manifest is missing its committed part ledger; refusing legacy downgrade"
        )
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("queue manifest has no valid configuration")
    typed_configuration = cast(Mapping[str, Any], configuration)
    recorded_signature = manifest.get("run_signature")
    if not isinstance(recorded_signature, str) or recorded_signature not in {
        _stable_digest(_signature_configuration(typed_configuration)),
        _stable_digest(typed_configuration),
    }:
        raise ValueError("queue manifest configuration digest is invalid")
    _validate_run_signature_binding(manifest)
    run_metadata = typed_configuration.get("run_metadata")
    if not isinstance(run_metadata, Mapping):
        raise ValueError("queue manifest has no valid run_metadata")
    return dict(cast(Mapping[str, Any], run_metadata))


def load_signed_queue_run_metadata(output_dir: str | Path) -> dict[str, Any] | None:
    """Compatibility alias; the local digest is not an authenticity signature."""

    return load_queue_run_metadata(output_dir)


def _is_exact_legacy_manifest(manifest: Mapping[str, Any]) -> bool:
    """Recognize the exact pre-part-ledger manifest shape eligible for migration."""

    legacy_fields = {
        "schema",
        "run_id",
        "run_signature",
        "created_at",
        "updated_at",
        "configuration",
        "progress",
        "stats",
        "teacher_review",
    }
    if set(manifest) != legacy_fields:
        return False
    configuration = manifest.get("configuration")
    if (
        not isinstance(configuration, Mapping)
        or set(configuration)
        != {"pipeline_version", "source", "options", "run_metadata", "accepted_dir"}
        or configuration.get("pipeline_version") != 1
    ):
        return False
    recorded_signature = manifest.get("run_signature")
    if not isinstance(recorded_signature, str):
        return False
    if recorded_signature != _stable_digest(configuration):
        return False
    if manifest.get("run_id") != recorded_signature[:16]:
        return False
    progress = manifest.get("progress")
    return (
        isinstance(progress, Mapping)
        and all(
            type(progress.get(field)) is int
            for field in ("completed_rows", "source_byte_offset", "next_part")
        )
        and type(progress.get("complete")) is bool
    )


def _secure_temporary_path(path: Path) -> tuple[int, Path]:
    """Create an unpredictable, exclusive sibling used for atomic publication."""

    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    return descriptor, Path(name)


def _fsync_directory(directory: Path) -> None:
    """Persist directory-entry changes on POSIX and Windows."""

    resolved = directory.resolve(strict=True)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # pyright: ignore[reportAttributeAccessIssue]
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(resolved),
            0x40000000,  # GENERIC_WRITE is required to flush a directory handle.
            0x00000007,  # Share reads, writes, and deletes with cooperating workers.
            None,
            3,  # OPEN_EXISTING
            0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS permits directory handles.
            None,
        )
        invalid_handle = wintypes.HANDLE(-1).value
        if handle == invalid_handle:
            raise OSError(ctypes.get_last_error(), f"cannot open directory for flush: {resolved}")
        try:
            if not kernel32.FlushFileBuffers(handle):
                raise OSError(
                    ctypes.get_last_error(),
                    f"cannot flush queue directory: {resolved}",
                )
        finally:
            kernel32.CloseHandle(handle)
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(resolved, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_unlink(path: Path, *, missing_ok: bool = False) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        if missing_ok:
            return
        raise
    _fsync_directory(path.parent)


def _write_atomic_text(
    path: Path,
    writer: Callable[[Any], None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _secure_temporary_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        _assert_plain_file(temporary, label="atomic queue temporary file")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _durable_unlink(temporary, missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    def write(handle: Any) -> None:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    _write_atomic_text(path, write)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    def write(handle: Any) -> None:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    _write_atomic_text(path, write)


def _cleanup_orphaned_initialization_temps(output_dir: Path) -> None:
    """Remove only queue-owned temp files left before the first manifest commit."""

    patterns = (
        f".{SOURCE_SNAPSHOT_FILENAME}.*.tmp",
        f".{SOURCE_INDEX_FILENAME}.*.tmp*",
        ".manifest.json.*.tmp",
    )
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                _durable_unlink(path)


def _acquire_file_lock(handle: BinaryIO) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_file_lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager  # pyright: ignore[reportDeprecated]
def _directory_run_lock(
    directory: Path,
    *,
    lock_filename: str,
    label: str,
) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / lock_filename
    with lock_path.open("a+b") as handle:
        try:
            _acquire_file_lock(handle)
        except OSError as exc:
            raise RuntimeError(
                f"{label} is already being translated: {directory.resolve()}"
            ) from exc
        try:
            yield
        finally:
            _release_file_lock(handle)


@contextmanager  # pyright: ignore[reportDeprecated]
def _queue_run_lock(output_dir: Path) -> Iterator[None]:
    """Hold an OS-released single-writer lock for one queue output directory."""

    with _directory_run_lock(
        output_dir,
        lock_filename=RUN_LOCK_FILENAME,
        label="queue output",
    ):
        yield


@contextmanager  # pyright: ignore[reportDeprecated]
def _accepted_run_lock(accepted_dir: Path) -> Iterator[None]:
    """Serialize publishers sharing an accepted-shard namespace."""

    with _directory_run_lock(
        accepted_dir,
        lock_filename=ACCEPTED_LOCK_FILENAME,
        label="accepted queue namespace",
    ):
        yield


def _manifest_accepted_shard_prefix(manifest: Mapping[str, Any]) -> str:
    configuration = manifest.get("configuration")
    raw_prefix = (
        configuration.get("accepted_shard_prefix") if isinstance(configuration, Mapping) else None
    )
    if raw_prefix is None and _is_exact_legacy_manifest(manifest):
        return LEGACY_ACCEPTED_SHARD_PREFIX
    if raw_prefix == ACCEPTED_SHARD_PREFIX:
        if (
            not isinstance(configuration, Mapping)
            or configuration.get("pipeline_version") != PIPELINE_VERSION
        ):
            raise ValueError("current accepted shard prefix requires the current pipeline version")
        return ACCEPTED_SHARD_PREFIX
    if raw_prefix == LEGACY_ACCEPTED_SHARD_PREFIX:
        if (
            not isinstance(configuration, Mapping)
            or configuration.get("legacy_public_marker") != LEGACY_PUBLIC_MARKER
        ):
            raise ValueError("legacy accepted shard prefix lacks a verified migration marker")
        return LEGACY_ACCEPTED_SHARD_PREFIX
    raise ValueError("queue manifest accepted_shard_prefix is not allowed")


def _accepted_shard_path(
    manifest: Mapping[str, Any],
    *,
    accepted_dir: Path,
    input_stem: str,
    part_index: int,
) -> Path:
    prefix = _manifest_accepted_shard_prefix(manifest)
    configuration = manifest.get("configuration")
    migrated_legacy = (
        prefix == LEGACY_ACCEPTED_SHARD_PREFIX
        and isinstance(configuration, Mapping)
        and configuration.get("legacy_public_marker") == LEGACY_PUBLIC_MARKER
    )
    if prefix == ACCEPTED_SHARD_PREFIX or migrated_legacy:
        # New queue outputs intentionally do not carry a public ``.jsonl``
        # suffix. Migrated legacy shards are quarantined here as well, so only
        # a complete consolidated set can enter top-level discovery.
        private_filename = f"part-{part_index:06d}.accepted.jsonl.private"
        run_root = _ensure_private_run_directory(accepted_dir, _manifest_run_id(manifest))
        return run_root / private_filename
    filename = f"{prefix}{input_stem}_{_manifest_run_id(manifest)}_{part_index:06d}.jsonl"
    return accepted_dir / filename


def _legacy_public_accepted_path(
    manifest: Mapping[str, Any],
    *,
    accepted_dir: Path,
    input_stem: str,
    part_index: int,
) -> Path:
    """Return the old public shard path consumed only by verified migration."""

    return accepted_dir / (
        f"{LEGACY_ACCEPTED_SHARD_PREFIX}{input_stem}_{_manifest_run_id(manifest)}_"
        f"{part_index:06d}.jsonl"
    )


def _training_stage_path(
    manifest: Mapping[str, Any],
    *,
    accepted_dir: Path,
    input_stem: str,
    part_index: int,
) -> Path:
    """Return the private staging path for one policy-verified training part."""

    if _manifest_accepted_shard_prefix(manifest) not in {
        ACCEPTED_SHARD_PREFIX,
        LEGACY_ACCEPTED_SHARD_PREFIX,
    }:
        raise ValueError("queue manifest has no supported training materialization policy")
    del input_stem
    run_root = _ensure_private_run_directory(accepted_dir, _manifest_run_id(manifest))
    return run_root / f"part-{part_index:06d}.training.jsonl.private"


def _training_set_path(
    manifest: Mapping[str, Any],
    *,
    accepted_dir: Path,
    input_stem: str,
) -> Path:
    """Return the single public path exposed only after the complete run is ready."""

    if _manifest_accepted_shard_prefix(manifest) not in {
        ACCEPTED_SHARD_PREFIX,
        LEGACY_ACCEPTED_SHARD_PREFIX,
    }:
        raise ValueError("queue manifest has no supported consolidated training policy")
    return accepted_dir / f"{ACCEPTED_SHARD_PREFIX}{input_stem}_{_manifest_run_id(manifest)}.jsonl"


def _pending_accepted_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.pending")


def _publish_no_replace(pending_path: Path, accepted_path: Path) -> None:
    """Atomically publish without overwriting a target created by another host."""

    pending_before = os.lstat(pending_path)
    if not stat.S_ISREG(pending_before.st_mode) or pending_before.st_nlink != 1:
        raise ValueError(f"publication source is not an unaliased plain file: {pending_path}")
    source_identity = (pending_before.st_dev, pending_before.st_ino)
    try:
        os.link(pending_path, accepted_path)
    except FileExistsError:
        raise
    except OSError as exc:
        raise OSError(
            f"filesystem does not support atomic no-clobber publication: {accepted_path}"
        ) from exc
    try:
        _fsync_directory(accepted_path.parent)
        pending_linked = os.lstat(pending_path)
        accepted_linked = os.lstat(accepted_path)
        if (
            not stat.S_ISREG(accepted_linked.st_mode)
            or (pending_linked.st_dev, pending_linked.st_ino) != source_identity
            or (accepted_linked.st_dev, accepted_linked.st_ino) != source_identity
            or pending_linked.st_nlink != 2
            or accepted_linked.st_nlink != 2
        ):
            raise ValueError(f"publication source changed or gained an alias: {pending_path}")
        if os.name == "nt":
            # Windows will not unlink a read-only file. Both names still refer
            # to the verified queue-owned inode, and callers restore the final
            # read-only mode after their post-publication checks.
            os.chmod(pending_path, 0o600)
        _durable_unlink(pending_path)
        accepted_after = os.lstat(accepted_path)
        if (
            not stat.S_ISREG(accepted_after.st_mode)
            or (accepted_after.st_dev, accepted_after.st_ino) != source_identity
            or accepted_after.st_nlink != 1
        ):
            raise ValueError(f"published file has an unsafe filesystem identity: {accepted_path}")
    except Exception:
        try:
            accepted_metadata = os.lstat(accepted_path)
            if (accepted_metadata.st_dev, accepted_metadata.st_ino) == source_identity:
                if os.name == "nt":
                    os.chmod(accepted_path, 0o600)
                _durable_unlink(accepted_path)
        except FileNotFoundError:
            pass
        raise


def _claim_accepted_namespace(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    accepted_dir: Path,
    input_stem: str,
) -> None:
    """Persistently bind one accepted run ID to its owning output manifest."""

    accepted_dir.mkdir(parents=True, exist_ok=True)
    run_id = _manifest_run_id(manifest)
    owner_path = accepted_dir / f".{input_stem}_{run_id}.owner.json"
    owner = {
        "schema": ACCEPTED_OWNER_SCHEMA,
        "run_id": run_id,
        "output_dir": str(output_dir.resolve()),
        "manifest": str((output_dir / "manifest.json").resolve()),
    }
    if owner_path.is_file():
        try:
            existing = json.loads(owner_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid accepted namespace owner: {owner_path}") from exc
        if existing != owner:
            raise FileExistsError(
                f"accepted queue namespace is already owned by another output: {owner_path}"
            )
        return
    descriptor, temporary = _secure_temporary_path(owner_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            json.dump(owner, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _assert_plain_file(temporary, label="accepted namespace owner temporary file")
        _publish_no_replace(temporary, owner_path)
    except FileExistsError as exc:
        try:
            existing = json.loads(owner_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as read_error:
            raise ValueError(f"invalid accepted namespace owner: {owner_path}") from read_error
        if existing != owner:
            raise FileExistsError(
                f"accepted queue namespace is already owned by another output: {owner_path}"
            ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _durable_unlink(temporary, missing_ok=True)


def _jsonl_artifact(path: Path) -> dict[str, Any]:
    _assert_plain_file(path, label="queue JSONL artifact")
    path_before = os.lstat(path)
    digest = hashlib.sha256()
    rows = 0
    final_byte = b""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened_before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
                rows += block.count(b"\n")
                final_byte = block[-1:]
            opened_after = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    path_after = os.lstat(path)
    identities = {
        (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_nlink,
        )
        for metadata in (path_before, opened_before, opened_after, path_after)
    }
    if len(identities) != 1:
        raise RuntimeError(f"queue JSONL artifact changed while hashing: {path}")
    size = path_after.st_size
    if size and final_byte != b"\n":
        rows += 1
    return {
        "path": str(path.resolve()),
        "size": size,
        "rows": rows,
        "sha256": digest.hexdigest(),
        "device": path_after.st_dev,
        "inode": path_after.st_ino,
        "mtime_ns": path_after.st_mtime_ns,
        "nlink": path_after.st_nlink,
        "mode": stat.S_IMODE(path_after.st_mode),
        "file_attributes": getattr(path_after, "st_file_attributes", 0),
    }


def _refresh_artifact_runtime(path: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Refresh cheap stat fields after making an already-hashed artifact immutable."""

    _assert_plain_file(path, label="queue artifact runtime refresh")
    metadata = os.lstat(path)
    if metadata.st_size != artifact.get("size"):
        raise RuntimeError(f"queue artifact size changed before runtime identity refresh: {path}")
    return {
        **dict(artifact),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mtime_ns": metadata.st_mtime_ns,
        "nlink": metadata.st_nlink,
        "mode": stat.S_IMODE(metadata.st_mode),
        "file_attributes": getattr(metadata, "st_file_attributes", 0),
    }


def _artifact_runtime_unchanged(path: Path, artifact: Mapping[str, Any]) -> bool:
    """Check a persisted inode identity without re-reading a potentially huge shard."""

    if type(artifact.get("size")) is not int or any(
        type(artifact.get(field)) is not int for field in _ARTIFACT_RUNTIME_FIELDS
    ):
        return False
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    recorded_mode = artifact["mode"]
    return (
        metadata.st_nlink == 1
        and metadata.st_size == artifact["size"]
        and recorded_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
        and stat.S_ISREG(metadata.st_mode)
        and all(
            artifact[field] == observed
            for field, observed in (
                ("device", metadata.st_dev),
                ("inode", metadata.st_ino),
                ("mtime_ns", metadata.st_mtime_ns),
                ("nlink", metadata.st_nlink),
                ("mode", stat.S_IMODE(metadata.st_mode)),
                ("file_attributes", getattr(metadata, "st_file_attributes", 0)),
            )
        )
    )


def _artifact_content_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    if _artifact_runtime_unchanged(path, expected):
        return True
    observed = _jsonl_artifact(path)
    return all(observed[field] == expected.get(field) for field in ("size", "rows", "sha256"))


def _copy_plain_file_no_replace(
    source: Path,
    target: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy stable source bytes and verify the no-clobber final before returning."""

    _validate_artifact_shape(expected, field="expected private training artifact")
    _assert_plain_file(source, label="private accepted training source")
    source_path_before = os.lstat(source)
    source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source, source_flags)
    descriptor, temporary = _secure_temporary_path(target)
    published_identity: tuple[int, int] | None = None
    try:
        source_opened_before = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_opened_before.st_mode) or source_opened_before.st_nlink != 1:
            raise ValueError(f"private accepted training source is not an unaliased file: {source}")
        digest = hashlib.sha256()
        rows = 0
        final_byte = b""
        copied_size = 0
        with (
            os.fdopen(source_descriptor, "rb") as source_handle,
            os.fdopen(descriptor, "wb") as target_handle,
        ):
            source_descriptor = -1
            descriptor = -1
            while block := source_handle.read(8 * 1024 * 1024):
                target_handle.write(block)
                digest.update(block)
                copied_size += len(block)
                rows += block.count(b"\n")
                final_byte = block[-1:]
            source_opened_after = os.fstat(source_handle.fileno())
            target_handle.flush()
            os.fsync(target_handle.fileno())
        source_path_after = os.lstat(source)
        source_identities = {
            (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_nlink,
            )
            for metadata in (
                source_path_before,
                source_opened_before,
                source_opened_after,
                source_path_after,
            )
        }
        if len(source_identities) != 1:
            raise RuntimeError(f"private accepted training source changed while copying: {source}")
        if copied_size and final_byte != b"\n":
            rows += 1
        copied = {
            "size": copied_size,
            "rows": rows,
            "sha256": digest.hexdigest(),
        }
        if any(copied[field] != expected.get(field) for field in copied):
            raise ValueError("private accepted training source changed before publication")
        _assert_plain_file(temporary, label="training materialization temporary file")
        temporary_metadata = os.lstat(temporary)
        published_identity = (temporary_metadata.st_dev, temporary_metadata.st_ino)
        _publish_no_replace(temporary, target)
        final_metadata = os.lstat(target)
        if (
            not stat.S_ISREG(final_metadata.st_mode)
            or final_metadata.st_nlink != 1
            or (final_metadata.st_dev, final_metadata.st_ino) != published_identity
        ):
            raise ValueError(f"published training file has an unsafe filesystem identity: {target}")
        observed = _jsonl_artifact(target)
        if any(observed[field] != expected.get(field) for field in ("size", "rows", "sha256")):
            raise ValueError(f"published training file differs from its verified source: {target}")
        os.chmod(target, 0o444)
        return _refresh_artifact_runtime(target, observed)
    except Exception:
        if target.exists() and published_identity is not None:
            try:
                target_metadata = os.lstat(target)
                if (target_metadata.st_dev, target_metadata.st_ino) == published_identity:
                    os.chmod(target, 0o600)
                    _durable_unlink(target)
            except (FileNotFoundError, OSError):
                pass
        raise
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if descriptor >= 0:
            os.close(descriptor)
        _durable_unlink(temporary, missing_ok=True)


def _teacher_review_allows_training(manifest: Mapping[str, Any]) -> bool:
    review = manifest.get("teacher_review")
    return review is None or (
        isinstance(review, Mapping)
        and review.get("review_required") is True
        and review.get("approved") is True
    )


def _stage_training_parts(
    manifest: dict[str, Any],
    *,
    accepted_dir: Path,
    input_stem: str,
    part_indices: Sequence[int] | None = None,
    verify_content: bool = False,
) -> bool:
    """Stage policy-verified parts privately; no incomplete set becomes discoverable."""

    _manifest_accepted_shard_prefix(manifest)
    parts = manifest.get("parts")
    if not isinstance(parts, list):
        raise ValueError("current queue manifest parts must be a list")
    allowed = _teacher_review_allows_training(manifest)
    changed = False
    indices = range(len(parts)) if part_indices is None else part_indices
    public_path = _training_set_path(
        manifest,
        accepted_dir=accepted_dir,
        input_stem=input_stem,
    )
    if not allowed and (manifest.get("training_set") is not None or public_path.exists()):
        raise ValueError(
            f"unapproved teacher output is present in the public training namespace: {public_path}"
        )
    for part_index in indices:
        if type(part_index) is not int or not 0 <= part_index < len(parts):
            raise ValueError(f"invalid queue training part index: {part_index!r}")
        part = parts[part_index]
        if not isinstance(part, dict):
            raise ValueError(f"queue part {part_index:06d} must be an object")
        training_path = _training_stage_path(
            manifest,
            accepted_dir=accepted_dir,
            input_stem=input_stem,
            part_index=part_index,
        )
        recorded_training = part.get("training")
        if not allowed:
            if recorded_training is not None or training_path.exists():
                raise ValueError(
                    "unapproved teacher output is present in the private training stage: "
                    f"{training_path}"
                )
            continue
        if part.get("published") is not True:
            raise ValueError(f"queue part {part_index:06d} is not privately published")
        accepted = part.get("accepted")
        if not isinstance(accepted, Mapping):
            raise ValueError(f"queue part {part_index:06d} has no accepted artifact")
        if accepted.get("rows") == 0:
            if recorded_training is not None or training_path.exists():
                raise ValueError(f"empty accepted part {part_index:06d} has a training stage")
            continue
        private_path = _accepted_shard_path(
            manifest,
            accepted_dir=accepted_dir,
            input_stem=input_stem,
            part_index=part_index,
        )
        _assert_plain_file(private_path, label=f"private accepted part {part_index:06d}")
        if not _artifact_content_matches(private_path, accepted):
            raise ValueError(f"private accepted part {part_index:06d} failed verification")
        if training_path.exists():
            if not isinstance(recorded_training, Mapping):
                raise FileExistsError(
                    f"unowned private training stage cannot be overwritten: {training_path}"
                )
            recorded_path = Path(str(recorded_training.get("path", "")))
            if recorded_path.resolve() != training_path.resolve():
                raise ValueError(
                    f"queue part {part_index:06d} training path contradicts its manifest"
                )
            if (
                not verify_content
                and _artifact_runtime_unchanged(training_path, recorded_training)
                and all(
                    recorded_training.get(field) == accepted.get(field)
                    for field in ("size", "rows", "sha256")
                )
            ):
                continue
            _assert_plain_file(training_path, label=f"training stage {part_index:06d}")
            if not _artifact_content_matches(training_path, accepted):
                raise ValueError(f"private training stage {part_index:06d} failed verification")
        else:
            training_artifact = _copy_plain_file_no_replace(
                private_path,
                training_path,
                expected=accepted,
            )
            part["training"] = training_artifact
            changed = True
            continue
        training_artifact = _jsonl_artifact(training_path)
        recorded_path = Path(str(recorded_training.get("path", "")))
        if recorded_path.resolve() != training_path.resolve() or any(
            recorded_training.get(field) != training_artifact[field]
            for field in ("size", "rows", "sha256")
        ):
            raise ValueError(f"queue part {part_index:06d} training metadata contradicts its bytes")
        if recorded_training != training_artifact:
            part["training"] = training_artifact
            changed = True
    return changed


def _publish_complete_training_set(
    manifest: dict[str, Any],
    *,
    accepted_dir: Path,
    input_stem: str,
) -> bool:
    """Atomically expose one complete accepted set after all private stages exist."""

    _manifest_accepted_shard_prefix(manifest)
    public_path = _training_set_path(
        manifest,
        accepted_dir=accepted_dir,
        input_stem=input_stem,
    )
    if not _teacher_review_allows_training(manifest):
        if manifest.get("training_set") is not None or public_path.exists():
            raise ValueError(f"unapproved public training set exists: {public_path}")
        return False
    if manifest["progress"].get("complete") is not True:
        if manifest.get("training_set") is not None or public_path.exists():
            raise ValueError(f"incomplete queue has a public training set: {public_path}")
        return False
    parts = manifest.get("parts")
    if not isinstance(parts, list):
        raise ValueError("current queue manifest parts must be a list")

    descriptor, temporary = _secure_temporary_path(public_path)
    expected_digest = hashlib.sha256()
    expected_size = 0
    expected_rows = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            for part_index, part in enumerate(parts):
                accepted = part.get("accepted") if isinstance(part, Mapping) else None
                if not isinstance(accepted, Mapping):
                    raise ValueError(f"queue part {part_index:06d} has no accepted artifact")
                rows = _exact_non_negative_integer(
                    accepted.get("rows"),
                    field=f"queue part {part_index:06d} accepted rows",
                )
                if rows == 0:
                    continue
                training = part.get("training") if isinstance(part, Mapping) else None
                if not isinstance(training, Mapping):
                    raise ValueError(f"queue part {part_index:06d} has no verified training stage")
                stage_path = _training_stage_path(
                    manifest,
                    accepted_dir=accepted_dir,
                    input_stem=input_stem,
                    part_index=part_index,
                )
                _validate_artifact(
                    training,
                    expected_path=stage_path,
                    label=f"training stage {part_index:06d}",
                )
                with stage_path.open("rb") as stage:
                    while block := stage.read(8 * 1024 * 1024):
                        output.write(block)
                        expected_digest.update(block)
                        expected_size += len(block)
                expected_rows += rows
            output.flush()
            os.fsync(output.fileno())
        expected = {
            "path": str(public_path.resolve()),
            "size": expected_size,
            "rows": expected_rows,
            "sha256": expected_digest.hexdigest(),
        }
        temporary_artifact = _jsonl_artifact(temporary)
        if any(
            temporary_artifact[field] != expected[field] for field in ("size", "rows", "sha256")
        ):
            raise RuntimeError("consolidated queue training temporary failed verification")
        published_identity: tuple[int, int] | None = None
        if public_path.exists():
            _assert_plain_file(public_path, label="public queue training set")
            if not _artifact_content_matches(public_path, expected):
                raise FileExistsError(
                    f"public training set collision cannot be overwritten: {public_path}"
                )
        else:
            temporary_metadata = os.lstat(temporary)
            published_identity = (temporary_metadata.st_dev, temporary_metadata.st_ino)
            try:
                _publish_no_replace(temporary, public_path)
                public_metadata = os.lstat(public_path)
                if (
                    not stat.S_ISREG(public_metadata.st_mode)
                    or public_metadata.st_nlink != 1
                    or (public_metadata.st_dev, public_metadata.st_ino) != published_identity
                ):
                    raise ValueError(
                        f"public queue training set has an unsafe filesystem identity: {public_path}"
                    )
            except Exception:
                if public_path.exists():
                    try:
                        public_metadata = os.lstat(public_path)
                        if (public_metadata.st_dev, public_metadata.st_ino) == published_identity:
                            os.chmod(public_path, 0o600)
                            _durable_unlink(public_path)
                    except (FileNotFoundError, OSError):
                        pass
                raise
        try:
            _assert_plain_file(public_path, label="public queue training set")
            artifact = _jsonl_artifact(public_path)
            if any(artifact[field] != expected[field] for field in ("size", "rows", "sha256")):
                raise ValueError("published complete queue training set failed verification")
        except Exception:
            if public_path.exists() and published_identity is not None:
                try:
                    public_metadata = os.lstat(public_path)
                    if (public_metadata.st_dev, public_metadata.st_ino) == published_identity:
                        os.chmod(public_path, 0o600)
                        _durable_unlink(public_path)
                except (FileNotFoundError, OSError):
                    pass
            raise
        os.chmod(public_path, 0o444)
        artifact = _refresh_artifact_runtime(public_path, artifact)
        recorded = manifest.get("training_set")
        if recorded is not None:
            if not isinstance(recorded, Mapping) or recorded != artifact:
                raise ValueError("queue manifest training_set contradicts its public bytes")
            return False
        manifest["training_set"] = artifact
        return True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _durable_unlink(temporary, missing_ok=True)


def _recover_pending_accepted_parts(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    accepted_dir: Path,
    input_stem: str,
) -> bool:
    """Publish a committed accepted shard if a crash preceded its final rename."""

    next_part = int(manifest["progress"]["next_part"])
    parts = manifest.get("parts")
    changed = False
    if parts is not None and not isinstance(parts, list):
        raise ValueError("queue manifest parts must be a list")
    if isinstance(parts, list) and len(parts) != next_part:
        raise ValueError("queue manifest part count does not match progress.next_part")
    missing_published_parts: list[int] = []
    corrupt_pending_parts: list[int] = []
    validation_only_parts: set[int] = set()
    if isinstance(parts, list):
        # A local manifest digest is not attestation. Atomically gate every
        # committed private part as unpublished before semantic validation, so
        # manifest-aware consumers never observe bytes during verification.
        for part_index, part in enumerate(parts[:next_part]):
            expected = part.get("accepted") if isinstance(part, Mapping) else None
            if not isinstance(part, dict) or not isinstance(expected, Mapping):
                raise ValueError(f"accepted part {part_index:06d} has no valid manifest metadata")
            if part.get("published") is not True:
                continue
            accepted_path = _accepted_shard_path(
                manifest,
                accepted_dir=accepted_dir,
                input_stem=input_stem,
                part_index=part_index,
            )
            pending_path = _pending_accepted_path(accepted_path)
            part["published"] = False
            changed = True
            if _artifact_content_matches(accepted_path, expected):
                continue
            if accepted_path.exists():
                validation_only_parts.add(part_index)
            elif not pending_path.exists():
                missing_published_parts.append(part_index)
            elif not _artifact_content_matches(pending_path, expected):
                corrupt_pending_parts.append(part_index)
        if changed:
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            _atomic_write_json(manifest_path, manifest)
            # Existing finals now pass through full semantic validation while
            # the persisted manifest remains unpublished.
            for part_index in validation_only_parts:
                parts[part_index]["published"] = True
        if missing_published_parts:
            labels = ", ".join(f"{index:06d}" for index in missing_published_parts)
            raise FileNotFoundError(f"accepted parts are missing after quarantine: {labels}")
        if corrupt_pending_parts:
            labels = ", ".join(f"{index:06d}" for index in corrupt_pending_parts)
            raise ValueError(
                "published accepted parts were quarantined because neither final nor pending "
                f"bytes are recoverable: {labels}"
            )
    for part_index in range(next_part):
        accepted_path = _accepted_shard_path(
            manifest,
            accepted_dir=accepted_dir,
            input_stem=input_stem,
            part_index=part_index,
        )
        pending_path = _pending_accepted_path(accepted_path)
        if part_index in validation_only_parts:
            continue
        if isinstance(parts, list) and part_index < len(parts):
            part = parts[part_index]
            expected = part.get("accepted") if isinstance(part, Mapping) else None
            if not isinstance(part, dict) or not isinstance(expected, Mapping):
                raise ValueError(f"accepted part {part_index:06d} has no valid manifest metadata")
            if part.get("published") is True and not pending_path.exists():
                continue
            final_matches = _artifact_content_matches(accepted_path, expected)
            pending_matches = _artifact_content_matches(pending_path, expected)
            if final_matches:
                if pending_path.exists():
                    _durable_unlink(pending_path)
                if part.get("published") is not True:
                    part["published"] = True
                    changed = True
                continue
            if pending_matches:
                try:
                    _publish_no_replace(pending_path, accepted_path)
                except FileExistsError as exc:
                    if _artifact_content_matches(accepted_path, expected):
                        _durable_unlink(pending_path)
                    else:
                        raise ValueError(
                            f"accepted part {part_index:06d} collided during recovery"
                        ) from exc
                if part.get("published") is not True:
                    part["published"] = True
                    changed = True
                continue
            if not accepted_path.exists() and not pending_path.exists():
                raise FileNotFoundError(
                    f"accepted part {part_index:06d} and its pending recovery are missing"
                )
            raise ValueError(f"accepted part {part_index:06d} does not match its manifest")
        if pending_path.is_file() and not accepted_path.exists():
            _publish_no_replace(pending_path, accepted_path)
        elif pending_path.exists():
            raise ValueError(f"ambiguous legacy accepted part recovery for {part_index:06d}")
    return changed


def _validate_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_path: Path,
    label: str,
) -> dict[str, Any]:
    recorded_path = Path(str(artifact.get("path", "")))
    if recorded_path.resolve() != expected_path.resolve():
        raise ValueError(f"{label} path does not match its queue manifest")
    if not expected_path.is_file():
        raise FileNotFoundError(f"{label} is missing: {expected_path}")
    _assert_plain_file(expected_path, label=label)
    observed = _jsonl_artifact(expected_path)
    for field in ("size", "rows", "sha256"):
        if artifact.get(field) != observed[field]:
            raise ValueError(
                f"{label} integrity mismatch for {field}: "
                f"expected {artifact.get(field)!r}, observed {observed[field]!r}"
            )
    return observed


def _accepted_provenance(
    result: Mapping[str, Any],
    *,
    run_id: str,
    run_lineage: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    quality = result.get("quality")
    roundtrip = quality.get("roundtrip") if isinstance(quality, Mapping) else None
    cycle_score = roundtrip.get("score") if isinstance(roundtrip, Mapping) else None
    provenance: dict[str, Any] = {
        "type": "machine_translation",
        "queue_id": result["id"],
        "run_id": run_id,
        "source_index": result["source_index"],
        "roundtrip_score": cycle_score,
        "source_queue": {field: source_identity.get(field) for field in _CONTENT_IDENTITY_FIELDS},
    }
    for field in (
        "source_dataset",
        "source_revision",
        "source_license",
        "translation_model",
        "tokenizer",
        "tokenizer_metadata",
        "token_features",
        "translation_directions",
        "translation_graph_source",
        "runtime_verification",
    ):
        provenance[field] = run_lineage[field]
    return provenance


def _canonical_accepted_row(
    result: Mapping[str, Any],
    *,
    run_id: str,
    run_lineage: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    row_id = result.get("id")
    source = result.get("source")
    translation = result.get("translation")
    source_index = result.get("source_index")
    if (
        not isinstance(row_id, str)
        or not row_id
        or not isinstance(source, str)
        or not isinstance(translation, str)
        or type(source_index) is not int
    ):
        raise ValueError("accepted result is missing its verified text identity")
    direction = canonicalize_language_pair(
        (result.get("source_lang"), result.get("target_lang")),
        field="accepted result direction",
    )
    return {
        "source_language": direction[0],
        "target_language": direction[1],
        "source": source,
        "translation": translation,
        "id": row_id,
        "synthetic": True,
        "training_direction": list(direction),
        "provenance": _accepted_provenance(
            result,
            run_id=run_id,
            run_lineage=run_lineage,
            source_identity=source_identity,
        ),
    }


def _normalize_legacy_accepted_row(
    accepted: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Reconcile an old accepted row with its corresponding reproduced result row."""

    if accepted == expected:
        return dict(expected)
    if accepted.get("id") != expected["id"] or accepted.get("synthetic") is not True:
        raise ValueError(f"accepted identity mismatch in {label}")

    expected_direction = cast(list[str], expected["training_direction"])
    explicit = "source_language" in accepted or "target_language" in accepted
    language_keys: set[str] = set()
    if explicit:
        try:
            observed_direction = canonicalize_language_pair(
                (accepted.get("source_language"), accepted.get("target_language")),
                field=f"accepted direction in {label}",
            )
        except ValueError as exc:
            raise ValueError(f"accepted direction mismatch in {label}") from exc
        if list(observed_direction) != expected_direction:
            raise ValueError(f"accepted direction mismatch in {label}")
        if (
            accepted.get("source") != expected["source"]
            or accepted.get("translation") != expected["translation"]
        ):
            raise ValueError(f"accepted source/translation mismatch in {label}")
        allowed_keys = set(expected)
    else:
        values_by_language: dict[str, list[tuple[str, Any]]] = {
            expected_direction[0]: [],
            expected_direction[1]: [],
        }
        for raw_key, value in accepted.items():
            try:
                language = canonicalize_language_tag(raw_key, field=f"accepted key in {label}")
            except ValueError:
                continue
            if language in values_by_language:
                values_by_language[language].append((raw_key, value))
        if any(len(values) != 1 for values in values_by_language.values()):
            raise ValueError(f"accepted language keys are ambiguous in {label}")
        source_key, source_value = values_by_language[expected_direction[0]][0]
        target_key, target_value = values_by_language[expected_direction[1]][0]
        if source_value != expected["source"] or target_value != expected["translation"]:
            raise ValueError(f"accepted source/translation mismatch in {label}")
        language_keys.update((source_key, target_key))
        allowed_keys = {
            *language_keys,
            "id",
            "synthetic",
            "training_direction",
            "provenance",
        }
    unexpected_keys = set(accepted) - allowed_keys
    if unexpected_keys:
        raise ValueError(f"accepted record has unverified fields in {label}: {unexpected_keys}")

    raw_training_direction = accepted.get("training_direction")
    if raw_training_direction is not None:
        try:
            observed_training_direction = canonicalize_language_pair(
                raw_training_direction,
                field=f"accepted training_direction in {label}",
            )
        except ValueError as exc:
            raise ValueError(f"accepted training_direction mismatch in {label}") from exc
        if list(observed_training_direction) != expected_direction:
            raise ValueError(f"accepted training_direction mismatch in {label}")

    provenance = accepted.get("provenance")
    expected_provenance = cast(Mapping[str, Any], expected["provenance"])
    if not isinstance(provenance, Mapping):
        raise ValueError(f"accepted provenance mismatch in {label}")
    required_provenance = {"type", "queue_id", "run_id", "roundtrip_score"}
    if not required_provenance.issubset(provenance):
        raise ValueError(f"accepted provenance mismatch in {label}")
    unexpected_provenance = set(provenance) - set(expected_provenance)
    if unexpected_provenance or any(
        type(provenance[key]) is not type(expected_provenance.get(key))
        or provenance[key] != expected_provenance.get(key)
        for key in provenance
    ):
        raise ValueError(f"accepted provenance mismatch in {label}")
    return dict(expected)


def _validate_result_source_identity(
    result: Mapping[str, Any],
    source_raw: bytes,
    *,
    source_index: int,
    label: str,
) -> None:
    """Reconcile one immutable queue row with its persisted result record."""

    expected, needs_translation = _parse_queue_line(source_raw, source_index)
    if result.get("source_index") != source_index:
        raise ValueError(f"source_index mismatch against the source queue in {label}")
    row_id = result.get("id")
    if row_id != expected.get("id"):
        raise ValueError(f"queue id mismatch against the source queue in {label}")

    for field in ("source_lang", "target_lang", "source"):
        if result.get(field) != expected.get(field):
            raise ValueError(f"{field} mismatch against the source queue in {label}")

    status = result.get("status")
    if needs_translation:
        if status == "skipped_existing":
            raise ValueError(f"pending source row became skipped_existing in {label}")
        return

    expected_status = expected["status"]
    if status != expected_status:
        raise ValueError(
            f"source queue {expected_status} semantics mismatch in {label}: {status!r}"
        )
    if result.get("translation") != expected.get("translation"):
        raise ValueError(f"source queue translation mismatch in {label}")
    if expected_status == "error" and result.get("error") != expected.get("error"):
        raise ValueError(f"source queue parse error mismatch in {label}")


def _part_semantics(
    result_path: Path,
    accepted_path: Path,
    *,
    source_start_index: int,
    run_id: str,
    run_lineage: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    source_handle: BinaryIO,
) -> tuple[dict[str, int], int, list[dict[str, Any]], bool, list[bytes]]:
    """Validate result/accepted equivalence and normalize verified legacy rows."""

    status_counts: Counter[str] = Counter()
    accepted_results: list[dict[str, Any]] = []
    source_rows: list[bytes] = []
    generated_rows = 0
    with result_path.open("r", encoding="utf-8") as handle:
        for offset, line in enumerate(handle):
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {result_path}:{offset + 1}") from exc
            if not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA:
                raise ValueError(f"invalid result record in {result_path}:{offset + 1}")
            if (
                type(result.get("source_index")) is not int
                or result.get("source_index") != source_start_index + offset
            ):
                raise ValueError(f"non-contiguous source_index in {result_path}:{offset + 1}")
            if result.get("run_id") != run_id:
                raise ValueError(f"run_id mismatch in {result_path}:{offset + 1}")
            source_raw = source_handle.readline()
            if not source_raw:
                raise ValueError(
                    f"result row has no corresponding immutable source queue row: "
                    f"{result_path}:{offset + 1}"
                )
            source_rows.append(source_raw)
            _validate_result_source_identity(
                result,
                source_raw,
                source_index=source_start_index + offset,
                label=f"{result_path}:{offset + 1}",
            )
            row_id = result.get("id")
            status = result.get("status")
            if status not in {"accepted", "rejected", "error", "skipped_existing"}:
                raise ValueError(f"invalid result status in {result_path}:{offset + 1}")
            status_counts[status] += 1
            if result.get("translation") is not None and status != "skipped_existing":
                generated_rows += 1
            if status == "accepted":
                if not isinstance(row_id, str) or not row_id:
                    raise ValueError(f"accepted result has no id in {result_path}:{offset + 1}")
                accepted_results.append(result)

    accepted_rows: list[dict[str, Any]] = []
    with accepted_path.open("r", encoding="utf-8") as handle:
        for offset, line in enumerate(handle):
            try:
                accepted = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {accepted_path}:{offset + 1}") from exc
            if not isinstance(accepted, dict):
                raise ValueError(f"invalid accepted record in {accepted_path}:{offset + 1}")
            accepted_rows.append(accepted)
    if len(accepted_rows) != len(accepted_results):
        raise ValueError("accepted shard is not the exact accepted subset of its result shard")

    normalized_rows: list[dict[str, Any]] = []
    changed = False
    for offset, (accepted, result) in enumerate(
        zip(accepted_rows, accepted_results, strict=True),
        start=1,
    ):
        expected = _canonical_accepted_row(
            result,
            run_id=run_id,
            run_lineage=run_lineage,
            source_identity=source_identity,
        )
        normalized = _normalize_legacy_accepted_row(
            accepted,
            expected,
            label=f"{accepted_path}:{offset}",
        )
        normalized_rows.append(normalized)
        changed = changed or accepted != normalized
    return (
        {
            status: status_counts[status]
            for status in ("accepted", "rejected", "error", "skipped_existing")
        },
        generated_rows,
        normalized_rows,
        changed,
        source_rows,
    )


def _translate_resilient(
    translator: TranslatorLike,
    texts: Sequence[str],
    *,
    source_language: str,
    target_language: str,
    num_beams: int,
    max_new_tokens: int,
    batch_size: int,
    max_output_length_ratio: float,
    max_output_length_margin: int,
) -> tuple[list[str | None], list[str | None]]:
    """Translate a batch and bisect failures so one row cannot stop a long run."""

    outputs: list[str | None] = [None] * len(texts)
    errors: list[str | None] = [None] * len(texts)

    def run(indices: list[int]) -> None:
        if not indices:
            return
        selected = [texts[index] for index in indices]
        try:
            translated = translator.translate(
                selected,
                source_language=source_language,
                target_language=target_language,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                batch_size=min(batch_size, len(selected)),
                max_output_length_ratio=max_output_length_ratio,
                max_output_length_margin=max_output_length_margin,
            )
            if len(translated) != len(selected):
                raise RuntimeError(
                    f"translator returned {len(translated)} rows for {len(selected)} inputs"
                )
        except Exception as exc:
            if len(indices) > 1:
                midpoint = len(indices) // 2
                run(indices[:midpoint])
                run(indices[midpoint:])
                return
            if isinstance(exc, PermanentQueueRowError):
                errors[indices[0]] = f"{type(exc).__name__}: {exc}"
                return
            raise RetryableQueueTranslationError(
                "translator runtime failed for one row; the current shard was not "
                "committed and can be retried safely"
            ) from exc
        for index, translation in zip(indices, translated, strict=True):
            outputs[index] = canonical_text(str(translation))

    ordered = sorted(
        range(len(texts)),
        key=lambda index: len(_content_tokens(translator, texts[index])),
    )
    for start in range(0, len(ordered), batch_size):
        run(ordered[start : start + batch_size])
    return outputs, errors


def _content_tokens(translator: TranslatorLike, text: str) -> list[object]:
    tokenizer = getattr(translator, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        typed_encode = cast(Callable[[str], Sequence[object]], encode)
        return list(typed_encode(text))
    return [character for character in canonical_text(text) if not character.isspace()]


def roundtrip_quality(
    source: str,
    roundtrip: str,
    *,
    tokenize: Callable[[str], Sequence[object]],
) -> dict[str, float | bool]:
    """Score whether a forward candidate can reconstruct its source."""

    cycle_chrf = _CHRF.sentence_score(roundtrip, [source]).score / 100.0
    token_f1 = multiset_f1(tokenize(source), tokenize(roundtrip))
    number_f1 = multiset_f1(numeric_tokens(source), numeric_tokens(roundtrip))
    structured_score, critical_mismatch = structured_similarity(source, roundtrip)
    source_length = sum(not character.isspace() for character in source)
    roundtrip_length = sum(not character.isspace() for character in roundtrip)
    length_score = math.exp(-abs(math.log((roundtrip_length + 1.0) / (source_length + 1.0))))
    score = (
        0.50 * cycle_chrf
        + 0.20 * token_f1
        + 0.15 * number_f1
        + 0.10 * structured_score
        + 0.05 * length_score
    )
    if critical_mismatch:
        score *= 0.5
    return {
        "score": score,
        "chrf": cycle_chrf,
        "token_f1": token_f1,
        "number_f1": number_f1,
        "structured": structured_score,
        "length": length_score,
        "critical_structured_mismatch": critical_mismatch,
    }


def _error_result(source_index: int, reason: str, *, raw_id: object = None) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "source_index": source_index,
        "id": raw_id if isinstance(raw_id, str) else None,
        "status": "error",
        "translation": None,
        "error": reason,
    }


def _parse_queue_line(raw: bytes, source_index: int) -> tuple[dict[str, Any], bool]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _error_result(source_index, f"invalid_utf8: {exc}"), False
    try:
        row = json.loads(text)
    except json.JSONDecodeError as exc:
        return _error_result(source_index, f"invalid_json: {exc}"), False
    if not isinstance(row, dict):
        return _error_result(source_index, "invalid_record: expected object"), False
    row_id = row.get("id")
    raw_source_language = row.get("source_lang")
    raw_target_language = row.get("target_lang")
    source = row.get("source")
    if not isinstance(row_id, str) or not row_id:
        return _error_result(source_index, "invalid_record: missing id"), False
    try:
        row_id_size = len(row_id.encode("utf-8"))
    except UnicodeEncodeError:
        return _error_result(
            source_index,
            "invalid_record: id contains an invalid Unicode scalar value",
        ), False
    if row_id_size > MAX_QUEUE_ID_UTF8_BYTES:
        return _error_result(
            source_index,
            "invalid_record: id exceeds the 4096-byte UTF-8 limit",
        ), False
    try:
        source_language = canonicalize_language_tag(
            raw_source_language,
            field="queue source_lang",
        )
        target_language = canonicalize_language_tag(
            raw_target_language,
            field="queue target_lang",
        )
    except ValueError as exc:
        return _error_result(
            source_index,
            f"invalid_record: invalid source_lang/target_lang: {exc}",
            raw_id=row_id,
        ), False
    if source_language == target_language:
        return _error_result(
            source_index,
            "invalid_record: source_lang and target_lang identify the same language",
            raw_id=row_id,
        ), False
    if not isinstance(source, str) or not canonical_text(source):
        return _error_result(
            source_index,
            "invalid_record: missing source",
            raw_id=row_id,
        ), False
    result = {
        "schema": RESULT_SCHEMA,
        "source_index": source_index,
        "id": row_id,
        "source_lang": source_language,
        "target_lang": target_language,
        "source": canonical_text(source),
        "translation": row.get("translation"),
        "status": row.get("status", "pending"),
    }
    if result["translation"] is not None or result["status"] != "pending":
        result["status"] = "skipped_existing"
        return result, False
    return result, True


def _artifact_translation_directions(
    translator: TranslatorLike,
) -> tuple[tuple[str, str], ...]:
    raw_directions = getattr(translator, "translation_directions", None)
    if (
        not isinstance(raw_directions, Sequence)
        or isinstance(raw_directions, (str, bytes))
        or not raw_directions
    ):
        raise ValueError(
            "translator has no artifact-bound translation_directions; "
            "load an export with an explicit language graph"
        )
    return _canonical_direction_graph(
        raw_directions,
        field="translator translation_directions",
    )


def _source_index_artifact(path: Path) -> dict[str, Any]:
    artifact = _stable_file_artifact(path, label="queue source SQLite index")
    return {
        **artifact,
        "rows": 1,
    }


def _read_source_index(
    index_path: Path,
    *,
    source_snapshot: Mapping[str, Any],
    verify_integrity: bool = True,
) -> set[tuple[str, str]]:
    """Read the bounded on-disk queue index after checking its source binding."""

    _assert_plain_file(index_path, label="queue source SQLite index")
    uri = f"{index_path.resolve(strict=True).as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as database:
        if verify_integrity:
            integrity = database.execute("PRAGMA quick_check").fetchone()
            if integrity != ("ok",):
                raise ValueError(f"queue source SQLite index failed integrity check: {index_path}")
        metadata = dict(database.execute("SELECT key, value FROM metadata"))
        if metadata.get("source_size") != str(source_snapshot.get("size")) or metadata.get(
            "source_sha256"
        ) != source_snapshot.get("sha256"):
            raise ValueError("queue source SQLite index is bound to different source bytes")
        rows = database.execute(
            "SELECT source_language, target_language FROM directions "
            "ORDER BY source_language, target_language"
        ).fetchall()
    return {canonicalize_language_pair(row, field="queue source index direction") for row in rows}


def _ensure_source_index(
    source_snapshot_path: Path,
    output_dir: Path,
    source_snapshot: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None,
) -> tuple[Path, dict[str, Any], set[tuple[str, str]]]:
    """Build or validate a disk-backed duplicate-ID and direction index."""

    index_path = output_dir / SOURCE_INDEX_FILENAME
    if index_path.exists():
        trusted_runtime_identity = expected is not None and _artifact_runtime_unchanged(
            index_path,
            expected,
        )
        if trusted_runtime_identity:
            typed_expected = cast(Mapping[str, Any], expected)
            _validate_artifact_shape(typed_expected, field="queue source SQLite index")
            recorded_path = Path(str(typed_expected.get("path", "")))
            if recorded_path.resolve() != index_path.resolve():
                raise ValueError("queue source SQLite index path differs from its manifest")
            observed: dict[str, Any] = dict(typed_expected)
        else:
            observed = _source_index_artifact(index_path)
            if expected is not None and any(
                observed.get(field) != expected.get(field) for field in ("path", "size", "sha256")
            ):
                raise ValueError("queue source SQLite index differs from its manifest")
        directions = _read_source_index(
            index_path,
            source_snapshot=source_snapshot,
            verify_integrity=not trusted_runtime_identity,
        )
        os.chmod(index_path, 0o444)
        return index_path, observed, directions
    if expected is not None:
        raise FileNotFoundError(f"queue source SQLite index is missing: {index_path}")

    descriptor, temporary = _secure_temporary_path(index_path)
    os.close(descriptor)
    descriptor = -1
    database: sqlite3.Connection | None = None
    try:
        database = sqlite3.connect(temporary)
        database.execute("PRAGMA journal_mode=DELETE")
        database.execute("PRAGMA synchronous=FULL")
        database.execute("PRAGMA temp_store=FILE")
        database.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        database.execute(
            "CREATE TABLE queue_ids (id TEXT PRIMARY KEY, source_index INTEGER NOT NULL)"
        )
        database.execute(
            "CREATE TABLE directions ("
            "source_language TEXT NOT NULL, target_language TEXT NOT NULL, "
            "PRIMARY KEY (source_language, target_language))"
        )
        database.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("schema", "sion-queue-source-index-v1"),
                ("source_size", str(source_snapshot.get("size"))),
                ("source_sha256", str(source_snapshot.get("sha256"))),
            ),
        )
        with source_snapshot_path.open("rb") as source_handle:
            for source_index, raw in enumerate(source_handle):
                result, needs_translation = _parse_queue_line(raw, source_index)
                row_id = result.get("id")
                if isinstance(row_id, str) and row_id:
                    try:
                        database.execute(
                            "INSERT INTO queue_ids(id, source_index) VALUES (?, ?)",
                            (row_id, source_index),
                        )
                    except sqlite3.IntegrityError as exc:
                        previous = database.execute(
                            "SELECT source_index FROM queue_ids WHERE id = ?",
                            (row_id,),
                        ).fetchone()
                        previous_index = previous[0] if previous is not None else "unknown"
                        raise ValueError(
                            f"duplicate queue id {row_id!r} at source rows "
                            f"{previous_index} and {source_index}"
                        ) from exc
                if needs_translation:
                    database.execute(
                        "INSERT OR IGNORE INTO directions(source_language, target_language) "
                        "VALUES (?, ?)",
                        (result["source_lang"], result["target_lang"]),
                    )
        database.commit()
        database.close()
        database = None
        # Windows rejects fsync on a read-only descriptor. The temporary index is
        # still private here, so opening it read/write is safe and makes the
        # durability barrier portable.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        _assert_plain_file(temporary, label="queue source SQLite index temporary file")
        os.replace(temporary, index_path)
        _fsync_directory(index_path.parent)
        _assert_plain_file(index_path, label="queue source SQLite index")
        os.chmod(index_path, 0o444)
    finally:
        if database is not None:
            database.close()
        if descriptor >= 0:
            os.close(descriptor)
        _durable_unlink(temporary, missing_ok=True)
    artifact = _source_index_artifact(index_path)
    directions = _read_source_index(index_path, source_snapshot=source_snapshot)
    return index_path, artifact, directions


def _validate_requested_directions(
    requested: set[tuple[str, str]],
    artifact_directions: set[tuple[str, str]],
    *,
    roundtrip_enabled: bool,
) -> None:
    missing_forward = sorted(requested - artifact_directions)
    if missing_forward:
        unsupported = ", ".join(f"{source}→{target}" for source, target in missing_forward)
        supported = ", ".join(
            f"{source}→{target}" for source, target in sorted(artifact_directions)
        )
        raise ValueError(
            f"queue requests untrained translation directions: {unsupported} "
            f"(artifact directions: {supported})"
        )
    if not roundtrip_enabled:
        return
    required_reverse = {(target, source) for source, target in requested}
    missing_reverse = sorted(required_reverse - artifact_directions)
    if missing_reverse:
        unsupported = ", ".join(f"{source}→{target}" for source, target in missing_reverse)
        raise ValueError(
            "round-trip filtering requires reverse directions absent from the artifact graph: "
            f"{unsupported}; use --no-roundtrip only if forward-only filtering is intended"
        )


def _validate_target_script_policies(
    requested: set[tuple[str, str]],
    options: QueueTranslationOptions,
) -> None:
    """Require explicit policy for unknown or multi-script target identities."""

    missing: list[str] = []
    for target_language in sorted({target for _source, target in requested}):
        if options.target_script_requirements(target_language):
            continue
        scripts = scripts_for_language(target_language)
        if scripts is None or len(scripts) > 1:
            missing.append(target_language)
    if missing:
        targets = ", ".join(missing)
        raise ValueError(
            "queue targets with unknown or multiple writing systems require an explicit "
            f"required_target_scripts policy: {targets}"
        )


def _forward_quality(
    source: str,
    translation: str,
    *,
    source_language: str,
    target_language: str,
) -> tuple[dict[str, Any], list[str]]:
    assessment = assess_pair(
        source,
        translation,
        languages=(source_language, target_language),
    )
    number_exact = Counter(numeric_tokens(source)) == Counter(numeric_tokens(translation))
    structured_score, critical_mismatch = structured_similarity(source, translation)
    target_fraction = language_fraction(translation, target_language)
    target_script_characters = Counter(
        script for character in translation if (script := script_of(character)) is not None
    )
    quality = {
        "pair_score": assessment.score,
        "pair_warnings": list(assessment.warning_reasons),
        "number_exact": number_exact,
        "structured": structured_score,
        "critical_structured_mismatch": critical_mismatch,
        "target_language_fraction": target_fraction,
        "target_script_characters": dict(sorted(target_script_characters.items())),
    }
    reasons = list(assessment.rejection_reasons)
    return quality, reasons


def _process_raw_rows(
    raw_rows: Sequence[bytes],
    *,
    start_index: int,
    translator: TranslatorLike,
    options: QueueTranslationOptions,
    seen_ids: set[str] | None = None,
    source_index_database: sqlite3.Connection | None = None,
    expected_directions: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Run the deterministic queue policy for source rows or reproduce a committed part."""

    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for offset, raw in enumerate(raw_rows):
        result, needs_translation = _parse_queue_line(raw, start_index + offset)
        row_id = result.get("id")
        if source_index_database is not None and isinstance(row_id, str) and row_id:
            indexed = source_index_database.execute(
                "SELECT source_index FROM queue_ids WHERE id = ?",
                (row_id,),
            ).fetchone()
            expected_source_index = start_index + offset
            if indexed != (expected_source_index,):
                raise ValueError(
                    f"queue source index does not bind {row_id!r} to source row "
                    f"{expected_source_index}"
                )
        if seen_ids is not None and isinstance(row_id, str) and row_id:
            if row_id in seen_ids:
                raise ValueError(f"duplicate queue id in processing stream: {row_id!r}")
            seen_ids.add(row_id)
        results.append(result)
        if needs_translation:
            pending.append(result)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in pending:
        key = (result["source_lang"], result["target_lang"])
        groups.setdefault(key, []).append(result)
    current_directions = _artifact_translation_directions(translator)
    if expected_directions is not None and current_directions != tuple(expected_directions):
        raise ValueError("translator direction graph changed after queue lineage validation")
    artifact_directions = set(current_directions)
    _validate_requested_directions(
        set(groups),
        artifact_directions,
        roundtrip_enabled=options.roundtrip_enabled,
    )
    for (source_language, target_language), jobs in groups.items():
        forward, errors = _translate_resilient(
            translator,
            [job["source"] for job in jobs],
            source_language=source_language,
            target_language=target_language,
            num_beams=options.num_beams,
            max_new_tokens=options.max_new_tokens,
            batch_size=options.batch_size,
            max_output_length_ratio=options.max_output_length_ratio,
            max_output_length_margin=options.max_output_length_margin,
        )
        for job, translation, error in zip(jobs, forward, errors, strict=True):
            if error is not None or not translation:
                job["status"] = "error"
                job["error"] = error or "empty_translation"
                job["translation"] = None
                continue
            job["translation"] = translation
            quality, reasons = _forward_quality(
                job["source"],
                translation,
                source_language=source_language,
                target_language=target_language,
            )
            job["quality"] = {"forward": quality}
            if quality["pair_score"] < options.min_pair_score:
                reasons.append("pair_score")
            if not quality["number_exact"]:
                reasons.append("number_mismatch")
            if (
                quality["critical_structured_mismatch"]
                or quality["structured"] < options.min_structured_similarity
            ):
                reasons.append("structured_mismatch")
            target_fraction = quality["target_language_fraction"]
            if (
                isinstance(target_fraction, (int, float))
                and target_fraction < options.min_target_language_fraction
            ):
                reasons.append("target_language")
            script_requirements = options.target_script_requirements(target_language)
            script_letter_counts = {
                script: script_letter_count(translation, script) for script in script_requirements
            }
            quality["target_script_letters"] = script_letter_counts
            for script, minimum in script_requirements.items():
                if script_letter_counts.get(script, 0) < minimum:
                    reasons.append(f"target_script:{script}")
            if reasons:
                job["status"] = "rejected"
                job["rejection_reasons"] = list(dict.fromkeys(reasons))
            else:
                job["status"] = "forward_pass"

    if options.roundtrip_enabled:
        cycle_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for job in pending:
            if job["status"] == "forward_pass":
                key = (job["target_lang"], job["source_lang"])
                cycle_groups.setdefault(key, []).append(job)
        for (source_language, target_language), jobs in cycle_groups.items():
            cycles, errors = _translate_resilient(
                translator,
                [str(job["translation"]) for job in jobs],
                source_language=source_language,
                target_language=target_language,
                num_beams=options.roundtrip_num_beams,
                max_new_tokens=options.roundtrip_max_new_tokens,
                batch_size=options.batch_size,
                max_output_length_ratio=options.roundtrip_max_output_length_ratio,
                max_output_length_margin=options.roundtrip_max_output_length_margin,
            )
            for job, cycle, error in zip(jobs, cycles, errors, strict=True):
                if error is not None or not cycle:
                    job["status"] = "error"
                    job["error"] = error or "empty_roundtrip"
                    continue
                cycle_quality = roundtrip_quality(
                    job["source"],
                    cycle,
                    tokenize=lambda text: _content_tokens(translator, text),
                )
                job["roundtrip"] = cycle
                job["quality"]["roundtrip"] = cycle_quality
                reasons = []
                if cycle_quality["score"] < options.min_roundtrip_score:
                    reasons.append("roundtrip_score")
                if cycle_quality["number_f1"] < 1.0:
                    reasons.append("roundtrip_number_mismatch")
                if (
                    cycle_quality["critical_structured_mismatch"]
                    or cycle_quality["structured"] < options.min_structured_similarity
                ):
                    reasons.append("roundtrip_structured_mismatch")
                if reasons:
                    job["status"] = "rejected"
                    job["rejection_reasons"] = reasons
                else:
                    job["status"] = "accepted"
    else:
        for job in pending:
            if job["status"] == "forward_pass":
                job["status"] = "accepted"
    return results


def _verify_committed_result_part(
    result_path: Path,
    raw_rows: Sequence[bytes],
    *,
    source_start_index: int,
    translator: TranslatorLike,
    options: QueueTranslationOptions,
    expected_directions: Sequence[tuple[str, str]],
) -> None:
    """Re-execute the model policy; local JSON digests alone are not attestation."""

    expected_rows = _process_raw_rows(
        raw_rows,
        start_index=source_start_index,
        translator=translator,
        options=options,
        expected_directions=expected_directions,
    )
    observed_rows: list[dict[str, Any]] = []
    with result_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            loaded: object = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError(f"invalid result record in {result_path}:{line_number}")
            observed_rows.append(loaded)
    if len(observed_rows) != len(expected_rows):
        raise ValueError(f"committed result part row count changed: {result_path}")
    for line_number, (observed, expected) in enumerate(
        zip(observed_rows, expected_rows, strict=True),
        start=1,
    ):
        observed_without_run = {key: value for key, value in observed.items() if key != "run_id"}
        if observed_without_run != expected:
            raise ValueError(
                "committed queue result cannot be reproduced by the recorded Translator and "
                f"policy: {result_path}:{line_number}"
            )


def _source_identity(
    path: Path,
    previous: Mapping[str, Any] | None,
    *,
    force_hash: bool = False,
) -> dict[str, Any]:
    del previous, force_hash
    before = path.stat()
    resolved = str(path.resolve())
    digest = sha256_file(path)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if before_identity != after_identity:
        raise ValueError(f"input source changed while hashing: {path}")
    return {
        "path": resolved,
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
        "device": after.st_dev,
        "inode": after.st_ino,
        "ctime_ns": after.st_ctime_ns,
        "nlink": after.st_nlink,
    }


def _assert_source_content_identity(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Rehash the complete source immediately before committing one part."""

    observed = _source_identity(path, None, force_hash=True)
    if any(observed.get(field) != expected.get(field) for field in _CONTENT_IDENTITY_FIELDS):
        raise ValueError("input source content changed during queue translation")
    return observed


def _source_snapshot_runtime_unchanged(path: Path, expected: Mapping[str, Any]) -> bool:
    """Trust an unchanged private snapshot inode until the final full hash barrier."""

    fields = ("device", "inode", "mtime_ns", "ctime_ns", "nlink")
    if type(expected.get("size")) is not int or any(
        type(expected.get(field)) is not int for field in fields
    ):
        return False
    metadata = os.lstat(path)
    return (
        metadata.st_nlink == 1
        and metadata.st_size == expected["size"]
        and stat.S_ISREG(metadata.st_mode)
        and all(
            expected[field] == observed
            for field, observed in (
                ("device", metadata.st_dev),
                ("inode", metadata.st_ino),
                ("mtime_ns", metadata.st_mtime_ns),
                ("ctime_ns", metadata.st_ctime_ns),
                ("nlink", metadata.st_nlink),
            )
        )
    )


def _open_snapshot_runtime_identity(handle: BinaryIO, path: Path) -> tuple[int, ...]:
    """Capture a cheap immutable-snapshot identity without rehashing all bytes."""

    opened = os.fstat(handle.fileno())
    linked = os.lstat(path)
    opened_identity = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
        opened.st_nlink,
        getattr(opened, "st_file_attributes", 0),
    )
    linked_identity = (
        linked.st_dev,
        linked.st_ino,
        linked.st_size,
        linked.st_mtime_ns,
        linked.st_ctime_ns,
        linked.st_nlink,
        getattr(linked, "st_file_attributes", 0),
    )
    if opened_identity != linked_identity or opened.st_nlink != 1:
        raise ValueError("private queue source snapshot path or inode changed")
    return opened_identity


def _assert_open_snapshot_unchanged(
    handle: BinaryIO,
    path: Path,
    expected: tuple[int, ...],
) -> None:
    if _open_snapshot_runtime_identity(handle, path) != expected:
        raise ValueError("private queue source snapshot changed during shard processing")


def _ensure_source_snapshot(
    input_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Materialize the queue bytes processed by this run in a private snapshot."""

    _cleanup_orphaned_initialization_temps(output_dir)
    source_before = _source_identity(input_path, None, force_hash=True)
    snapshot_path = output_dir / SOURCE_SNAPSHOT_FILENAME
    if snapshot_path.exists():
        _assert_plain_file(snapshot_path, label="private queue source snapshot")
        snapshot = _source_identity(snapshot_path, None, force_hash=True)
        if any(snapshot.get(field) != source_before.get(field) for field in ("size", "sha256")):
            raise ValueError(
                "private queue source snapshot differs from the requested input; "
                "use a new output directory"
            )
        os.chmod(snapshot_path, 0o444)
        return source_before, snapshot_path, snapshot

    descriptor, temporary = _secure_temporary_path(snapshot_path)
    digest = hashlib.sha256()
    copied_size = 0
    try:
        with (
            input_path.open("rb") as source_handle,
            os.fdopen(
                descriptor,
                "wb",
            ) as snapshot_handle,
        ):
            descriptor = -1
            while block := source_handle.read(8 * 1024 * 1024):
                snapshot_handle.write(block)
                digest.update(block)
                copied_size += len(block)
            snapshot_handle.flush()
            os.fsync(snapshot_handle.fileno())
        source_after = _source_identity(input_path, None, force_hash=True)
        if any(
            source_after.get(field) != source_before.get(field)
            for field in _CONTENT_IDENTITY_FIELDS
        ):
            raise ValueError("input source changed while its private snapshot was being created")
        if copied_size != source_after["size"] or digest.hexdigest() != source_after["sha256"]:
            raise ValueError("private queue source snapshot does not match the input bytes")
        temporary_identity = _source_identity(temporary, None, force_hash=True)
        if any(
            temporary_identity.get(field) != source_after.get(field) for field in ("size", "sha256")
        ):
            raise ValueError("private queue source snapshot changed before publication")
        _publish_no_replace(temporary, snapshot_path)
        _assert_plain_file(snapshot_path, label="private queue source snapshot")
        os.chmod(snapshot_path, 0o444)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _durable_unlink(temporary, missing_ok=True)
    snapshot = _source_identity(snapshot_path, None, force_hash=True)
    if any(snapshot.get(field) != source_after.get(field) for field in ("size", "sha256")):
        raise ValueError("published private queue source snapshot does not match the input bytes")
    return source_after, snapshot_path, snapshot


def _resolve_source_snapshot(
    manifest: Mapping[str, Any],
    *,
    input_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Verify an existing run's original input and private processing snapshot."""

    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("queue manifest has no valid configuration")
    recorded_source = configuration.get("source")
    recorded_snapshot = configuration.get("source_snapshot")
    if not isinstance(recorded_source, Mapping) or not isinstance(recorded_snapshot, Mapping):
        raise ValueError(
            "queue manifest predates immutable source snapshots; start a new output directory"
        )
    source = _assert_source_content_identity(input_path, recorded_source)
    snapshot_path = output_dir / SOURCE_SNAPSHOT_FILENAME
    recorded_path = Path(str(recorded_snapshot.get("path", "")))
    if recorded_path.resolve() != snapshot_path.resolve():
        raise ValueError("queue source snapshot path does not match its manifest")
    _assert_plain_file(snapshot_path, label="private queue source snapshot")
    snapshot_metadata = os.lstat(snapshot_path)
    if snapshot_metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        os.chmod(snapshot_path, 0o444)
    if _source_snapshot_runtime_unchanged(snapshot_path, recorded_snapshot):
        snapshot = dict(recorded_snapshot)
    else:
        snapshot = _assert_source_content_identity(snapshot_path, recorded_snapshot)
    if any(source.get(field) != snapshot.get(field) for field in ("size", "sha256")):
        raise ValueError("queue input and private source snapshot no longer match")
    return source, snapshot_path, snapshot


def _new_manifest(
    *,
    source: Mapping[str, Any],
    source_snapshot: Mapping[str, Any] | None = None,
    source_index: Mapping[str, Any] | None = None,
    options: QueueTranslationOptions,
    run_metadata: Mapping[str, Any],
    accepted_dir: Path,
    accepted_shard_prefix: str | None,
    teacher_pilot_rows: int | None,
    runtime_verification: str | None = None,
    legacy_public_binding: tuple[str, str] | None = None,
) -> dict[str, Any]:
    configuration = {
        "pipeline_version": PIPELINE_VERSION,
        "source": dict(source),
        "options": asdict(options),
        "run_metadata": dict(run_metadata),
        "accepted_dir": str(accepted_dir.resolve()),
    }
    if source_snapshot is not None:
        configuration["source_snapshot"] = dict(source_snapshot)
    if source_index is not None:
        configuration["source_index"] = dict(source_index)
    if runtime_verification is not None:
        configuration["runtime_verification"] = runtime_verification
    if accepted_shard_prefix is not None:
        configuration["accepted_shard_prefix"] = accepted_shard_prefix
    if legacy_public_binding is not None:
        legacy_run_id, legacy_run_signature = legacy_public_binding
        configuration.update(
            {
                "legacy_public_marker": LEGACY_PUBLIC_MARKER,
                "legacy_run_id": legacy_run_id,
                "legacy_run_signature": legacy_run_signature,
            }
        )
    signature = _stable_digest(_signature_configuration(configuration))
    return {
        "schema": MANIFEST_SCHEMA,
        "signature_version": SIGNATURE_VERSION,
        "run_id": signature[:16],
        "run_signature": signature,
        "integrity": dict(_LOCAL_INTEGRITY_DESCRIPTOR),
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "progress": {
            "completed_rows": 0,
            "source_byte_offset": 0,
            "next_part": 0,
            "complete": False,
        },
        "stats": {
            "processed": 0,
            "generated": 0,
            "accepted": 0,
            "rejected": 0,
            "errors": 0,
            "skipped_existing": 0,
        },
        "teacher_review": (
            {
                "pilot_rows": teacher_pilot_rows,
                "review_required": False,
                "approved": False,
                "approved_at": None,
                "approved_by": None,
            }
            if teacher_pilot_rows is not None
            else None
        ),
        "parts": [],
        "training_set": None,
    }


def _validate_or_register_parts(
    manifest: dict[str, Any],
    *,
    input_path: Path,
    output_dir: Path,
    accepted_dir: Path,
    input_stem: str,
    run_lineage: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    allow_legacy_registration: bool,
    translator: TranslatorLike,
    options: QueueTranslationOptions,
    expected_directions: Sequence[tuple[str, str]],
    replay_unstaged_parts: bool,
    replay_all_parts: bool,
    replay_before_part: int | None = None,
) -> bool:
    """Verify committed shards, registering legacy v1 shards on first resume."""

    progress = manifest["progress"]
    next_part = int(progress["next_part"])
    if replay_before_part is not None and (
        type(replay_before_part) is not int or not 0 <= replay_before_part <= next_part
    ):
        raise ValueError("replay_before_part must be a committed part boundary")
    parts = manifest.get("parts")
    changed = False
    legacy_public_artifacts: dict[int, tuple[Path, dict[str, Any]]] = {}
    if parts is None:
        if not allow_legacy_registration:
            raise ValueError(
                "current queue manifest is missing its committed part ledger; "
                "refusing legacy downgrade"
            )
        parts = []
        manifest["parts"] = parts
        changed = True
        source_start = 0
        for part_index in range(next_part):
            result_path = output_dir / f"part-{part_index:06d}.jsonl"
            accepted_path = _accepted_shard_path(
                manifest,
                accepted_dir=accepted_dir,
                input_stem=input_stem,
                part_index=part_index,
            )
            legacy_public_path = _legacy_public_accepted_path(
                manifest,
                accepted_dir=accepted_dir,
                input_stem=input_stem,
                part_index=part_index,
            )
            if not result_path.is_file():
                raise FileNotFoundError(f"legacy result shard is missing: {result_path}")
            result_artifact = _jsonl_artifact(result_path)
            legacy_public_artifact = (
                _jsonl_artifact(legacy_public_path) if legacy_public_path.is_file() else None
            )
            if accepted_path.is_file():
                accepted_artifact = _jsonl_artifact(accepted_path)
                # A previous migration attempt may have normalized the private
                # copy and crashed before removing the legacy public source.
                # Semantic replay below authenticates the private copy against
                # the immutable queue and model; byte equality with the older
                # row shape is neither expected nor required for safe recovery.
            elif legacy_public_artifact is not None:
                accepted_artifact = _copy_plain_file_no_replace(
                    legacy_public_path,
                    accepted_path,
                    expected=legacy_public_artifact,
                )
            else:
                raise FileNotFoundError(
                    f"legacy accepted shard and private migration are missing: {legacy_public_path}"
                )
            if legacy_public_artifact is not None:
                legacy_public_artifacts[part_index] = (
                    legacy_public_path,
                    legacy_public_artifact,
                )
            parts.append(
                {
                    "part": part_index,
                    "source_start_index": source_start,
                    "source_rows": result_artifact["rows"],
                    "result": result_artifact,
                    "accepted": accepted_artifact,
                    "published": True,
                }
            )
            source_start += int(result_artifact["rows"])
    if not isinstance(parts, list) or len(parts) != next_part:
        raise ValueError("queue manifest part count does not match progress.next_part")

    total_result_rows = 0
    total_accepted_rows = 0
    total_generated_rows = 0
    total_status_counts: Counter[str] = Counter()
    expected_source_start = 0
    with input_path.open("rb") as source_handle:
        source_size = os.fstat(source_handle.fileno()).st_size
        for part_index, part in enumerate(parts):
            if (
                not isinstance(part, dict)
                or type(part.get("part")) is not int
                or part.get("part") != part_index
            ):
                raise ValueError("queue manifest contains an invalid or out-of-order part")
            if part.get("published") is not True:
                raise ValueError(f"accepted part {part_index:06d} is not published")
            result_path = output_dir / f"part-{part_index:06d}.jsonl"
            accepted_path = _accepted_shard_path(
                manifest,
                accepted_dir=accepted_dir,
                input_stem=input_stem,
                part_index=part_index,
            )
            recorded_result = part.get("result")
            if not isinstance(recorded_result, Mapping):
                raise ValueError(f"result part {part_index:06d} has no artifact metadata")
            recorded_result_path = Path(str(recorded_result.get("path", "")))
            if recorded_result_path.resolve() != result_path.resolve():
                raise ValueError(
                    f"result part {part_index:06d} path does not match its queue manifest"
                )
            recorded_accepted = part.get("accepted")
            if not isinstance(recorded_accepted, Mapping):
                raise ValueError(f"accepted part {part_index:06d} has no artifact metadata")
            recorded_accepted_path = Path(str(recorded_accepted.get("path", "")))
            if recorded_accepted_path.resolve() != accepted_path.resolve():
                raise ValueError(
                    f"accepted part {part_index:06d} path does not match its queue manifest"
                )
            source_rows_count = _exact_non_negative_integer(
                part.get("source_rows"),
                field=f"queue manifest parts[{part_index}].source_rows",
            )
            source_start_index = _exact_non_negative_integer(
                part.get("source_start_index"),
                field=f"queue manifest parts[{part_index}].source_start_index",
            )
            if source_start_index != expected_source_start:
                raise ValueError(f"result part {part_index:06d} source range is not contiguous")
            should_replay = (
                allow_legacy_registration
                or replay_all_parts
                or (replay_before_part is not None and part_index < replay_before_part)
                or (
                    replay_unstaged_parts
                    and recorded_accepted.get("rows") != 0
                    and part.get("training") is None
                )
            )
            status_counts = part.get("status_counts")
            generated_rows = part.get("generated_rows")
            source_end_byte_offset = part.get("source_end_byte_offset")
            can_use_runtime_index = (
                not should_replay
                and not allow_legacy_registration
                and type(source_end_byte_offset) is int
                and isinstance(status_counts, Mapping)
                and type(generated_rows) is int
                and _artifact_runtime_unchanged(result_path, recorded_result)
                and _artifact_runtime_unchanged(accepted_path, recorded_accepted)
            )
            if can_use_runtime_index:
                typed_source_end = cast(int, source_end_byte_offset)
                typed_status_counts = cast(Mapping[str, Any], status_counts)
                typed_generated_rows = cast(int, generated_rows)
                result = dict(recorded_result)
                accepted = dict(recorded_accepted)
                if source_rows_count != result["rows"]:
                    raise ValueError(f"result part {part_index:06d} source row count mismatch")
                current_offset = source_handle.tell()
                if typed_source_end <= current_offset or typed_source_end > source_size:
                    raise ValueError(
                        f"result part {part_index:06d} has an invalid source byte boundary"
                    )
                source_handle.seek(typed_source_end)
                semantic_counts = {
                    status: _exact_non_negative_integer(
                        typed_status_counts.get(status),
                        field=f"queue manifest parts[{part_index}].status_counts.{status}",
                    )
                    for status in ("accepted", "rejected", "error", "skipped_existing")
                }
                semantic_generated_rows = typed_generated_rows
            else:
                result = _validate_artifact(
                    recorded_result,
                    expected_path=result_path,
                    label=f"result part {part_index:06d}",
                )
                if not accepted_path.is_file():
                    raise FileNotFoundError(
                        f"accepted part {part_index:06d} is missing: {accepted_path}"
                    )
                _assert_plain_file(
                    accepted_path,
                    label=f"accepted part {part_index:06d}",
                )
                observed_accepted_before = _jsonl_artifact(accepted_path)
                if source_rows_count != result["rows"]:
                    raise ValueError(f"result part {part_index:06d} source row count mismatch")
                (
                    semantic_counts,
                    semantic_generated_rows,
                    normalized_rows,
                    accepted_changed,
                    source_rows,
                ) = _part_semantics(
                    result_path,
                    accepted_path,
                    source_start_index=expected_source_start,
                    run_id=str(manifest["run_id"]),
                    run_lineage=run_lineage,
                    source_identity=source_identity,
                    source_handle=source_handle,
                )
                observed_source_end = source_handle.tell()
                if source_end_byte_offset is None:
                    part["source_end_byte_offset"] = observed_source_end
                    changed = True
                elif (
                    type(source_end_byte_offset) is not int
                    or source_end_byte_offset != observed_source_end
                ):
                    raise ValueError(
                        f"result part {part_index:06d} source byte boundary is invalid"
                    )
                if should_replay:
                    _verify_committed_result_part(
                        result_path,
                        source_rows,
                        source_start_index=expected_source_start,
                        translator=translator,
                        options=options,
                        expected_directions=expected_directions,
                    )
                canonical_accepted_digest = hashlib.sha256()
                canonical_accepted_size = 0
                for normalized_row in normalized_rows:
                    encoded = (
                        json.dumps(
                            normalized_row,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                    canonical_accepted_digest.update(encoded)
                    canonical_accepted_size += len(encoded)
                accepted_changed = accepted_changed or (
                    observed_accepted_before["size"] != canonical_accepted_size
                    or observed_accepted_before["sha256"] != canonical_accepted_digest.hexdigest()
                )
                if accepted_changed:
                    os.chmod(accepted_path, 0o600)
                    try:
                        _atomic_write_jsonl(accepted_path, normalized_rows)
                    finally:
                        if accepted_path.exists():
                            os.chmod(accepted_path, 0o444)
                accepted = _jsonl_artifact(accepted_path)
                os.chmod(result_path, 0o444)
                os.chmod(accepted_path, 0o444)
                result = _refresh_artifact_runtime(result_path, result)
                accepted = _refresh_artifact_runtime(accepted_path, accepted)
                if dict(recorded_result) != result:
                    part["result"] = result
                    changed = True
                if dict(recorded_accepted) != accepted:
                    part["accepted"] = accepted
                    changed = True
            if not isinstance(status_counts, Mapping):
                part["status_counts"] = semantic_counts
                status_counts = semantic_counts
                changed = True
            elif any(
                _exact_non_negative_integer(
                    status_counts.get(status),
                    field=f"queue manifest parts[{part_index}].status_counts.{status}",
                )
                != semantic_counts[status]
                for status in ("accepted", "rejected", "error", "skipped_existing")
            ):
                raise ValueError(f"result part {part_index:06d} status counts contradict its rows")
            if generated_rows is None:
                part["generated_rows"] = semantic_generated_rows
                generated_rows = semantic_generated_rows
                changed = True
            elif type(generated_rows) is not int or generated_rows != semantic_generated_rows:
                raise ValueError(
                    f"result part {part_index:06d} generated row count contradicts its rows"
                )
            normalized_counts = {
                status: _exact_non_negative_integer(
                    status_counts.get(status),
                    field=f"queue manifest parts[{part_index}].status_counts.{status}",
                )
                for status in ("accepted", "rejected", "error", "skipped_existing")
            }
            if sum(normalized_counts.values()) != result["rows"]:
                raise ValueError(f"result part {part_index:06d} status counts do not match rows")
            if normalized_counts["accepted"] != accepted["rows"]:
                raise ValueError(
                    f"accepted part {part_index:06d} row count does not match statuses"
                )
            if type(generated_rows) is not int or not 0 <= generated_rows <= result["rows"]:
                raise ValueError(f"result part {part_index:06d} has invalid generated row count")
            total_result_rows += result["rows"]
            total_accepted_rows += accepted["rows"]
            total_generated_rows += generated_rows
            total_status_counts.update(normalized_counts)
            expected_source_start += result["rows"]
        if source_handle.tell() != progress["source_byte_offset"]:
            raise ValueError(
                "committed source queue byte range does not match progress.source_byte_offset"
            )
        if progress.get("complete") is True and source_handle.read(1):
            raise ValueError("queue manifest marks an unconsumed source tail as complete")

    if total_result_rows != progress["completed_rows"]:
        raise ValueError("committed result rows do not match progress.completed_rows")
    stats = cast(dict[str, Any], manifest["stats"])
    if total_result_rows != stats["processed"]:
        raise ValueError("committed result rows do not match stats.processed")
    if total_accepted_rows != stats["accepted"]:
        raise ValueError("committed accepted rows do not match stats.accepted")
    for status, stat_name in (
        ("rejected", "rejected"),
        ("error", "errors"),
        ("skipped_existing", "skipped_existing"),
    ):
        if total_status_counts[status] != stats[stat_name]:
            raise ValueError(f"committed {status} rows do not match stats.{stat_name}")
    if "generated" not in stats:
        stats["generated"] = total_generated_rows
        changed = True
    elif total_generated_rows != stats["generated"]:
        raise ValueError("committed generated rows do not match stats.generated")
    for part_index, (legacy_public_path, legacy_public_artifact) in legacy_public_artifacts.items():
        if not _artifact_content_matches(legacy_public_path, legacy_public_artifact):
            raise ValueError(
                f"legacy public accepted part {part_index:06d} changed during migration"
            )
        os.chmod(legacy_public_path, 0o600)
        _durable_unlink(legacy_public_path)
        changed = True
    return changed


def _synchronize_teacher_review_state(manifest: dict[str, Any]) -> bool:
    review = manifest.get("teacher_review")
    if not isinstance(review, dict):
        return False
    expected = _expected_teacher_review_required(manifest)
    if review.get("review_required") is expected:
        return False
    review["review_required"] = expected
    return True


def _configure_teacher_review(
    manifest: dict[str, Any],
    *,
    teacher_pilot_rows: int | None,
    approve_teacher: bool,
    approval_actor: str | None,
    existing_manifest: bool,
) -> bool:
    """Restore/freeze the pilot policy and record explicit post-pilot approval."""

    stats = manifest["stats"]
    changed = False
    if "generated" not in stats:
        stats["generated"] = int(stats.get("accepted", 0)) + int(stats.get("rejected", 0))
        changed = True
    review = manifest.get("teacher_review")
    if review is None and teacher_pilot_rows is not None:
        review = {
            "pilot_rows": teacher_pilot_rows,
            "review_required": False,
            "approved": False,
            "approved_at": None,
            "approved_by": None,
        }
        manifest["teacher_review"] = review
        changed = True
    elif isinstance(review, dict) and teacher_pilot_rows is not None:
        if int(review["pilot_rows"]) != teacher_pilot_rows:
            raise ValueError(
                "teacher pilot size is fixed by the manifest; "
                "use the original value or a new output directory"
            )
    elif review is not None and not isinstance(review, dict):
        raise ValueError("invalid teacher_review in queue manifest")

    changed = _synchronize_teacher_review_state(manifest) or changed

    if approve_teacher:
        if not existing_manifest or not isinstance(review, dict):
            raise ValueError("a teacher cannot be approved before a pilot run")
        if not _expected_teacher_review_required(manifest):
            raise ValueError("the teacher pilot is not complete or has no reviewable output")
        if review.get("approved") is False:
            review = cast(dict[str, Any], review)
            actor = canonical_text(approval_actor or "")
            if not actor:
                raise ValueError("approval_actor is required to approve a teacher")
            review["approved"] = True
            review["approved_at"] = datetime.now(UTC).isoformat()
            review["approved_by"] = actor
            changed = True
    return changed


def _remaining_teacher_pilot_rows(manifest: Mapping[str, Any]) -> int | None:
    review = manifest.get("teacher_review")
    if not isinstance(review, Mapping) or review.get("approved") is True:
        return None
    return max(
        0,
        int(review["pilot_rows"]) - int(manifest["stats"].get("generated", 0)),
    )


def _translate_queue_unlocked(
    input_path: str | Path,
    output_dir: str | Path,
    translator: TranslatorLike,
    *,
    accepted_dir: str | Path,
    options: QueueTranslationOptions | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    max_rows: int | None = None,
    teacher_pilot_rows: int | None = None,
    approve_teacher: bool = False,
    approval_actor: str | None = None,
    allow_unverified_translator: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Translate pending queue rows into atomic result and training shards."""

    options = options or QueueTranslationOptions()
    options.validate()
    if max_rows is not None and (type(max_rows) is not int or max_rows <= 0):
        raise ValueError("max_rows must be a positive integer or None")
    if teacher_pilot_rows is not None and (
        type(teacher_pilot_rows) is not int or teacher_pilot_rows <= 0
    ):
        raise ValueError("teacher_pilot_rows must be a positive integer or None")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    accepted_dir = Path(accepted_dir)
    artifact_direction_list = _artifact_translation_directions(translator)
    artifact_directions = set(artifact_direction_list)
    manifest_path = output_dir / "manifest.json"
    existing: dict[str, Any] | None = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict) or loaded.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"unsupported queue manifest: {manifest_path}")
        existing = loaded
        _manifest_run_id(existing)
        _validate_manifest_control_state(existing)
    legacy_manifest = existing is not None and _is_exact_legacy_manifest(existing)
    if existing is not None and existing.get("parts") is None and not legacy_manifest:
        raise ValueError(
            "current queue manifest is missing its committed part ledger; refusing legacy downgrade"
        )
    if existing is not None and run_metadata is None:
        existing_configuration = existing.get("configuration")
        inherited_run_metadata = (
            existing_configuration.get("run_metadata")
            if isinstance(existing_configuration, Mapping)
            else None
        )
        if not isinstance(inherited_run_metadata, Mapping):
            raise ValueError("queue manifest has no valid run_metadata")
        effective_run_metadata = dict(cast(Mapping[str, Any], inherited_run_metadata))
    elif not isinstance(run_metadata, Mapping):
        raise ValueError("new queue runs require explicit immutable run_metadata")
    else:
        effective_run_metadata = dict(run_metadata)
    if existing is None:
        accepted_shard_prefix: str | None = ACCEPTED_SHARD_PREFIX
    elif legacy_manifest:
        accepted_shard_prefix = LEGACY_ACCEPTED_SHARD_PREFIX
    else:
        existing_configuration = existing.get("configuration")
        accepted_shard_prefix = _manifest_accepted_shard_prefix(existing)
    snapshot_migrated = False
    recorded_signature: str | None = None
    legacy_public_binding: tuple[str, str] | None = None
    if existing is not None and accepted_shard_prefix == LEGACY_ACCEPTED_SHARD_PREFIX:
        existing_configuration = cast(Mapping[str, Any], existing["configuration"])
        if not legacy_manifest:
            legacy_id = existing_configuration.get("legacy_run_id")
            legacy_signature = existing_configuration.get("legacy_run_signature")
            if isinstance(legacy_id, str) and isinstance(legacy_signature, str):
                legacy_public_binding = (legacy_id, legacy_signature)
    if existing is None:
        _cleanup_orphaned_initialization_temps(output_dir)
        non_lock_entries = [
            entry
            for entry in output_dir.iterdir()
            if entry.name
            not in {
                RUN_LOCK_FILENAME,
                ACCEPTED_LOCK_FILENAME,
                SOURCE_SNAPSHOT_FILENAME,
                SOURCE_INDEX_FILENAME,
            }
        ]
        if non_lock_entries:
            raise FileExistsError(f"{output_dir} is not empty and has no compatible queue manifest")
        source, source_snapshot_path, source_snapshot = _ensure_source_snapshot(
            input_path,
            output_dir,
        )
    else:
        existing_configuration = existing.get("configuration")
        if not isinstance(existing_configuration, dict):
            raise ValueError("queue manifest has no valid configuration")
        stable_before_migration = _stable_digest(_signature_configuration(existing_configuration))
        legacy_before_migration = _stable_digest(existing_configuration)
        raw_recorded_signature = existing.get("run_signature")
        if not isinstance(raw_recorded_signature, str) or raw_recorded_signature not in {
            stable_before_migration,
            legacy_before_migration,
        }:
            raise ValueError("queue manifest configuration digest is invalid")
        recorded_signature = raw_recorded_signature
        _validate_run_signature_binding(existing)
        if not isinstance(existing_configuration.get("source_snapshot"), Mapping):
            source, source_snapshot_path, source_snapshot = _ensure_source_snapshot(
                input_path,
                output_dir,
            )
            existing_configuration["source_snapshot"] = dict(source_snapshot)
            snapshot_migrated = True
        else:
            source, source_snapshot_path, source_snapshot = _resolve_source_snapshot(
                existing,
                input_path=input_path,
                output_dir=output_dir,
            )
        if legacy_manifest:
            legacy_public_binding = (_manifest_run_id(existing), raw_recorded_signature)
            existing_configuration.update(
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "accepted_shard_prefix": LEGACY_ACCEPTED_SHARD_PREFIX,
                    "legacy_public_marker": LEGACY_PUBLIC_MARKER,
                    "legacy_run_id": legacy_public_binding[0],
                    "legacy_run_signature": legacy_public_binding[1],
                }
            )
            snapshot_migrated = True
    existing_index = None
    mutable_existing_configuration: dict[str, Any] | None = None
    if existing is not None:
        raw_configuration = existing["configuration"]
        if not isinstance(raw_configuration, dict):
            raise ValueError("queue manifest configuration must be a JSON object")
        mutable_existing_configuration = cast(dict[str, Any], raw_configuration)
        raw_index = mutable_existing_configuration.get("source_index")
        if isinstance(raw_index, Mapping):
            existing_index = cast(Mapping[str, Any], raw_index)
    source_index_path, source_index, requested_directions = _ensure_source_index(
        source_snapshot_path,
        output_dir,
        source_snapshot,
        expected=existing_index,
    )
    if mutable_existing_configuration is not None and existing_index is None:
        mutable_existing_configuration["source_index"] = dict(source_index)
        snapshot_migrated = True
    _validate_requested_directions(
        requested_directions,
        artifact_directions,
        roundtrip_enabled=options.roundtrip_enabled,
    )
    _validate_target_script_policies(requested_directions, options)
    run_lineage = _validated_run_lineage(
        effective_run_metadata,
        translator,
        artifact_direction_list,
        legacy_manifest=legacy_manifest,
        allow_unverified_translator=allow_unverified_translator,
    )
    if legacy_manifest:
        # A migrated manifest must be independently resumable under the current
        # schema. Persist every field that legacy validation safely derived from
        # the bound translator instead of requiring the old relaxation again.
        effective_run_metadata = {
            **effective_run_metadata,
            "tokenizer_metadata": run_lineage["tokenizer_metadata"],
            "translation_directions": run_lineage["translation_directions"],
            "translation_graph_source": run_lineage["translation_graph_source"],
        }
        if mutable_existing_configuration is None:
            raise AssertionError("legacy migration requires a mutable configuration")
        mutable_existing_configuration["run_metadata"] = dict(effective_run_metadata)
        snapshot_migrated = True
    candidate = _new_manifest(
        source=source,
        source_snapshot=source_snapshot,
        source_index=source_index,
        options=options,
        run_metadata=effective_run_metadata,
        accepted_dir=accepted_dir,
        accepted_shard_prefix=accepted_shard_prefix,
        teacher_pilot_rows=teacher_pilot_rows,
        runtime_verification=str(run_lineage["runtime_verification"]),
        legacy_public_binding=legacy_public_binding,
    )
    resume_metadata_changed = False
    if existing is None:
        manifest = candidate
    else:
        existing_configuration = existing.get("configuration")
        if not isinstance(existing_configuration, dict):
            raise ValueError("queue manifest has no valid configuration")
        stable_existing_signature = _stable_digest(_signature_configuration(existing_configuration))
        if "runtime_verification" not in existing_configuration:
            existing_configuration["runtime_verification"] = run_lineage["runtime_verification"]
            stable_existing_signature = _stable_digest(
                _signature_configuration(existing_configuration)
            )
            snapshot_migrated = True
        if stable_existing_signature != candidate["run_signature"]:
            raise ValueError(
                "queue resume configuration changed; use a new output directory "
                "for a different source, model, or quality policy"
            )
        manifest = existing
        resume_metadata_changed = (
            recorded_signature != stable_existing_signature
            or manifest.get("signature_version") != SIGNATURE_VERSION
            or manifest["configuration"].get("source") != source
            or manifest["configuration"].get("source_snapshot") != source_snapshot
            or snapshot_migrated
            or manifest.get("integrity") != _LOCAL_INTEGRITY_DESCRIPTOR
        )
        manifest["run_signature"] = stable_existing_signature
        manifest["signature_version"] = SIGNATURE_VERSION
        manifest["integrity"] = dict(_LOCAL_INTEGRITY_DESCRIPTOR)
        manifest["configuration"]["source"] = source
        manifest["configuration"]["source_snapshot"] = source_snapshot
        manifest["configuration"]["source_index"] = source_index
        manifest.setdefault("training_set", None)
    _validate_run_signature_binding(manifest)
    _validate_manifest_control_state(manifest)
    _claim_accepted_namespace(
        manifest,
        output_dir=output_dir,
        accepted_dir=accepted_dir,
        input_stem=input_path.stem,
    )
    if existing is not None:
        recovery_changed = _recover_pending_accepted_parts(
            manifest,
            manifest_path=manifest_path,
            accepted_dir=accepted_dir,
            input_stem=input_path.stem,
        )
    else:
        recovery_changed = False
    historical_part_boundary = _exact_non_negative_integer(
        cast(Mapping[str, Any], manifest["progress"]).get("next_part"),
        field="queue manifest progress.next_part",
    )
    startup_replays_every_historical_part = legacy_manifest or (
        existing is not None
        and (
            approve_teacher
            or (
                cast(Mapping[str, Any], manifest["progress"]).get("complete") is True
                and manifest.get("training_set") is None
            )
        )
    )
    validation_changed = _validate_or_register_parts(
        manifest,
        input_path=source_snapshot_path,
        output_dir=output_dir,
        accepted_dir=accepted_dir,
        input_stem=input_path.stem,
        run_lineage=run_lineage,
        source_identity=source,
        allow_legacy_registration=legacy_manifest,
        translator=translator,
        options=options,
        expected_directions=artifact_direction_list,
        replay_unstaged_parts=(
            existing is not None
            and (
                manifest.get("teacher_review") is None
                or (
                    isinstance(manifest.get("teacher_review"), Mapping)
                    and cast(Mapping[str, Any], manifest["teacher_review"]).get("approved") is True
                )
            )
        ),
        replay_all_parts=startup_replays_every_historical_part,
    )
    parts_changed = recovery_changed or validation_changed
    review_changed = _configure_teacher_review(
        manifest,
        teacher_pilot_rows=teacher_pilot_rows,
        approve_teacher=approve_teacher,
        approval_actor=approval_actor,
        existing_manifest=existing is not None,
    )
    _validate_manifest_control_state(manifest)
    if existing is None or resume_metadata_changed or parts_changed or review_changed:
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_write_json(manifest_path, manifest)
    training_changed = _stage_training_parts(
        manifest,
        accepted_dir=accepted_dir,
        input_stem=input_path.stem,
        verify_content=cast(Mapping[str, Any], manifest["progress"]).get("complete") is True,
    )
    training_changed = (
        _publish_complete_training_set(
            manifest,
            accepted_dir=accepted_dir,
            input_stem=input_path.stem,
        )
        or training_changed
    )
    if training_changed:
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_write_json(manifest_path, manifest)
    progress = manifest["progress"]
    if progress.get("complete") is True:
        return manifest

    processed_this_call = 0
    with ExitStack() as processing_resources:
        source_handle = processing_resources.enter_context(source_snapshot_path.open("rb"))
        index_uri = f"{source_index_path.resolve(strict=True).as_uri()}?mode=ro&immutable=1"
        source_index_database = sqlite3.connect(index_uri, uri=True)
        processing_resources.callback(source_index_database.close)
        snapshot_runtime_identity = _open_snapshot_runtime_identity(
            source_handle,
            source_snapshot_path,
        )
        source_handle.seek(int(progress["source_byte_offset"]))
        while max_rows is None or processed_this_call < max_rows:
            pilot_remaining = _remaining_teacher_pilot_rows(manifest)
            if pilot_remaining == 0:
                if _synchronize_teacher_review_state(manifest):
                    manifest["updated_at"] = datetime.now(UTC).isoformat()
                    _atomic_write_json(manifest_path, manifest)
                break
            remaining = (
                options.shard_size
                if max_rows is None
                else min(options.shard_size, max_rows - processed_this_call)
            )
            if pilot_remaining is not None:
                remaining = min(remaining, pilot_remaining)
            start_index = int(progress["completed_rows"])
            raw_rows: list[bytes] = []
            for _ in range(remaining):
                raw = source_handle.readline()
                if not raw:
                    break
                raw_rows.append(raw)
            if not raw_rows:
                final_snapshot = _assert_source_content_identity(
                    source_snapshot_path,
                    source_snapshot,
                )
                manifest["configuration"]["source_snapshot"] = final_snapshot
                progress["complete"] = True
                _synchronize_teacher_review_state(manifest)
                manifest["updated_at"] = datetime.now(UTC).isoformat()
                _atomic_write_json(manifest_path, manifest)
                replay_changed = False
                if historical_part_boundary and not startup_replays_every_historical_part:
                    replay_changed = _validate_or_register_parts(
                        manifest,
                        input_path=source_snapshot_path,
                        output_dir=output_dir,
                        accepted_dir=accepted_dir,
                        input_stem=input_path.stem,
                        run_lineage=run_lineage,
                        source_identity=source,
                        allow_legacy_registration=False,
                        translator=translator,
                        options=options,
                        expected_directions=artifact_direction_list,
                        replay_unstaged_parts=False,
                        replay_all_parts=False,
                        replay_before_part=historical_part_boundary,
                    )
                stage_changed = _stage_training_parts(
                    manifest,
                    accepted_dir=accepted_dir,
                    input_stem=input_path.stem,
                    verify_content=True,
                )
                if (
                    _publish_complete_training_set(
                        manifest,
                        accepted_dir=accepted_dir,
                        input_stem=input_path.stem,
                    )
                    or replay_changed
                    or stage_changed
                ):
                    manifest["updated_at"] = datetime.now(UTC).isoformat()
                    _atomic_write_json(manifest_path, manifest)
                break

            results = _process_raw_rows(
                raw_rows,
                start_index=start_index,
                translator=translator,
                options=options,
                source_index_database=source_index_database,
                expected_directions=artifact_direction_list,
            )

            part = int(progress["next_part"])
            run_id = str(manifest["run_id"])
            accepted_rows: list[dict[str, Any]] = []
            for result in results:
                result["run_id"] = run_id
                if result["status"] != "accepted":
                    continue
                accepted_rows.append(
                    _canonical_accepted_row(
                        result,
                        run_id=run_id,
                        run_lineage=run_lineage,
                        source_identity=source,
                    )
                )

            counts = Counter(result["status"] for result in results)
            status_counts = {
                status: counts[status]
                for status in ("accepted", "rejected", "error", "skipped_existing")
            }
            generated_rows = sum(
                result.get("translation") is not None and result["status"] != "skipped_existing"
                for result in results
            )
            result_path = output_dir / f"part-{part:06d}.jsonl"
            accepted_path = _accepted_shard_path(
                manifest,
                accepted_dir=accepted_dir,
                input_stem=input_path.stem,
                part_index=part,
            )
            pending_accepted_path = _pending_accepted_path(accepted_path)
            if accepted_path.exists():
                raise FileExistsError(
                    "uncommitted accepted shard target already exists; "
                    f"refusing to overwrite {accepted_path}"
                )
            _atomic_write_jsonl(result_path, results)
            _atomic_write_jsonl(pending_accepted_path, accepted_rows)
            accepted_artifact = _jsonl_artifact(pending_accepted_path)
            accepted_artifact["path"] = str(accepted_path.resolve())
            position = source_handle.tell()
            source_exhausted = source_handle.read(1) == b""
            source_handle.seek(position)
            _assert_open_snapshot_unchanged(
                source_handle,
                source_snapshot_path,
                snapshot_runtime_identity,
            )
            manifest["parts"].append(
                {
                    "part": part,
                    "source_start_index": start_index,
                    "source_rows": len(raw_rows),
                    "source_end_byte_offset": position,
                    "result": _jsonl_artifact(result_path),
                    "accepted": accepted_artifact,
                    "status_counts": status_counts,
                    "generated_rows": generated_rows,
                    # Local JSON digests detect corruption but are not external
                    # attestation. Current runs bind the complete inference
                    # policy and artifact identities in the run signature.
                    "published": False,
                }
            )

            stats = manifest["stats"]
            stats["processed"] += len(results)
            stats["generated"] += generated_rows
            stats["accepted"] += counts["accepted"]
            stats["rejected"] += counts["rejected"]
            stats["errors"] += counts["error"]
            stats["skipped_existing"] += counts["skipped_existing"]
            progress["completed_rows"] += len(raw_rows)
            progress["source_byte_offset"] = source_handle.tell()
            progress["next_part"] += 1
            processed_this_call += len(raw_rows)
            progress["complete"] = source_exhausted
            _synchronize_teacher_review_state(manifest)
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            _atomic_write_json(manifest_path, manifest)
            _publish_no_replace(pending_accepted_path, accepted_path)
            _assert_open_snapshot_unchanged(
                source_handle,
                source_snapshot_path,
                snapshot_runtime_identity,
            )
            _assert_plain_file(accepted_path, label=f"published accepted part {part:06d}")
            if not _artifact_content_matches(accepted_path, accepted_artifact):
                raise ValueError(f"published accepted shard failed verification: {accepted_path}")
            manifest["parts"][-1]["published"] = True
            os.chmod(accepted_path, 0o444)
            os.chmod(result_path, 0o444)
            manifest["parts"][-1]["accepted"] = _refresh_artifact_runtime(
                accepted_path,
                cast(Mapping[str, Any], manifest["parts"][-1]["accepted"]),
            )
            manifest["parts"][-1]["result"] = _refresh_artifact_runtime(
                result_path,
                cast(Mapping[str, Any], manifest["parts"][-1]["result"]),
            )
            if source_exhausted:
                final_snapshot = _assert_source_content_identity(
                    source_snapshot_path,
                    source_snapshot,
                )
                manifest["configuration"]["source_snapshot"] = final_snapshot
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            _atomic_write_json(manifest_path, manifest)
            publication_validation_changed = False
            if (
                progress["complete"]
                and historical_part_boundary
                and not startup_replays_every_historical_part
            ):
                # Local shard digests are corruption checks, not authentication.
                # Reproduce every model decision loaded from an earlier process
                # exactly once at the publication boundary. Parts created in
                # this call are already authenticated by their live inference.
                publication_validation_changed = _validate_or_register_parts(
                    manifest,
                    input_path=source_snapshot_path,
                    output_dir=output_dir,
                    accepted_dir=accepted_dir,
                    input_stem=input_path.stem,
                    run_lineage=run_lineage,
                    source_identity=source,
                    allow_legacy_registration=False,
                    translator=translator,
                    options=options,
                    expected_directions=artifact_direction_list,
                    replay_unstaged_parts=False,
                    replay_all_parts=False,
                    replay_before_part=historical_part_boundary,
                )
                if publication_validation_changed:
                    manifest["updated_at"] = datetime.now(UTC).isoformat()
                    _atomic_write_json(manifest_path, manifest)
            staged_current_part = _stage_training_parts(
                manifest,
                accepted_dir=accepted_dir,
                input_stem=input_path.stem,
                part_indices=None if progress["complete"] else (part,),
                verify_content=progress["complete"],
            )
            published_training_set = _publish_complete_training_set(
                manifest,
                accepted_dir=accepted_dir,
                input_stem=input_path.stem,
            )
            if publication_validation_changed or staged_current_part or published_training_set:
                manifest["updated_at"] = datetime.now(UTC).isoformat()
                _atomic_write_json(manifest_path, manifest)
            if log is not None:
                log(
                    f"part {part:06d}: {len(results):,} rows, "
                    f"{counts['accepted']:,} accepted, {counts['rejected']:,} rejected, "
                    f"{counts['error']:,} errors; total {stats['processed']:,}"
                )
            if progress["complete"]:
                break
    return manifest


def translate_queue(
    input_path: str | Path,
    output_dir: str | Path,
    translator: TranslatorLike,
    *,
    accepted_dir: str | Path,
    options: QueueTranslationOptions | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    max_rows: int | None = None,
    teacher_pilot_rows: int | None = None,
    approve_teacher: bool = False,
    approval_actor: str | None = None,
    allow_unverified_translator: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Translate a queue under a single-writer lock."""

    output_path = Path(output_dir)
    accepted_path = Path(accepted_dir)
    output_resolved = output_path.resolve()
    accepted_resolved = accepted_path.resolve()
    if output_resolved.is_relative_to(accepted_resolved) or accepted_resolved.is_relative_to(
        output_resolved
    ):
        raise ValueError(
            "queue output_dir and accepted_dir must be separate, non-nested directories"
        )
    with ExitStack() as locks:
        locks.enter_context(_queue_run_lock(output_path))
        locks.enter_context(_accepted_run_lock(accepted_path))
        return _translate_queue_unlocked(
            input_path,
            output_dir,
            translator,
            accepted_dir=accepted_dir,
            options=options,
            run_metadata=run_metadata,
            max_rows=max_rows,
            teacher_pilot_rows=teacher_pilot_rows,
            approve_teacher=approve_teacher,
            approval_actor=approval_actor,
            allow_unverified_translator=allow_unverified_translator,
            log=log,
        )
