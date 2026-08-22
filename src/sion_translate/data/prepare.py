from __future__ import annotations

import hashlib
import glob
import json
import math
import multiprocessing
import os
import re
import shutil
import sqlite3
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, BinaryIO, Sequence, TypeAlias, cast

import numpy as np

from sion_translate.fingerprint import (
    PREPROCESSING_SCHEMA,
    DatasetFingerprint,
    FileFingerprint,
    file_sha256,
)
from sion_translate.performance import bounded_ordered_map, build_cpu_plan
from sion_translate.splitting import (
    TargetSplitGuard,
    choose_split_for_key,
    endpoint_split_digest,
    endpoint_split_key,
)
from sion_translate.structured import protect_shared_structured_spans
from sion_translate.synthetic import (
    DEFAULT_SYNTHETIC_PREFIXES,
    DEFAULT_SYNTHETIC_SAMPLING_WEIGHT,
    normalize_synthetic_prefixes,
    synthetic_path,
    synthetic_record,
)
from sion_translate.tokenizer import SLOT_SYMBOLS, SionTokenizer

from .quality import (
    QualityPolicy,
    apply_record_quality_profile,
    assess_pair,
    canonical_text,
    dedup_key,
)
from .integrity import (
    build_dataset_artifact_inventory,
    validate_dataset_artifact_inventory,
)
from .record_metadata import (
    RECORD_METADATA_DATA_SUFFIX,
    RECORD_METADATA_FIELDS,
    RECORD_METADATA_FORMAT,
    RECORD_METADATA_INDEX_DTYPE,
    RECORD_METADATA_INDEX_SUFFIX,
    decode_record_metadata,
    encode_record_metadata,
    resolve_record_training_direction,
)
from .records import (
    expand_parallel_record,
    languages_from_pairs,
    normalize_language_pairs,
    normalize_translation_directions,
)

INDEX_FORMAT = "sion-indexed-parallel-v6"
RAW_FINGERPRINT_FILENAME = "raw_fingerprint.json"
PREPARE_COMPLETION_FILENAME = ".sion-prepare-complete.json"
PREPARE_COMPLETION_SCHEMA = "sion-prepare-completion-v1"

INDEX_DTYPE = np.dtype(
    [
        ("src_offset", "<u8"),
        ("src_length", "<u4"),
        ("tgt_offset", "<u8"),
        ("tgt_length", "<u4"),
        ("src_register", "u1"),
        ("tgt_register", "u1"),
        ("src_language_id", "<u2"),
        ("tgt_language_id", "<u2"),
        ("source_id", "<u2"),
        ("quality_score", "u1"),
        ("synthetic", "u1"),
        # 1 when the reverse direction must not be trained, either because the
        # global graph is one-way or this row carries a scoped training edge.
        ("forward_only", "u1"),
    ]
)


@dataclass
class PrepareStats:
    physical_lines: int = 0
    valid_pairs: int = 0
    invalid_json: int = 0
    invalid_utf8: int = 0
    invalid_record: int = 0
    missing_text: int = 0
    non_string: int = 0
    invalid_language: int = 0
    unaligned_lists: int = 0
    duplicates: int = 0
    quality_filtered: int = 0
    too_short: int = 0
    identical_text: int = 0
    length_ratio_outlier: int = 0
    language_mismatch: int = 0
    control_characters: int = 0
    excessive_repetition: int = 0
    structured_span_warnings: int = 0
    ja_no_kana_warnings: int = 0
    split_conflicts: int = 0
    too_long: int = 0
    train: int = 0
    validation: int = 0
    test: int = 0
    ko_tokens: int = 0
    ja_tokens: int = 0
    quality_score_sum: int = 0
    synthetic_pairs: int = 0
    forward_only_pairs: int = 0


def infer_register(text: str, language: str) -> int:
    """문말 표현으로 존댓말 단계(register)를 추정합니다.

    한국어/일본어에만 규칙이 있으며, 그 외 언어는 0(미상)을 반환합니다.
    """
    stripped = text.rstrip(" \t\r\n.!?。！？\"'“”‘’")
    if language == "ko":
        if re.search(r"(옵니다|사옵니다|드리겠습니다|주시기 바랍니다|하십시오)$", stripped):
            return 3
        if re.search(r"(습니다|ㅂ니다|입니다|합니다|됩니다|세요|어요|아요|예요|이에요)$", stripped):
            return 2
        if re.search(r"(다|한다|했다|이다|한다면|하자|해)$", stripped):
            return 1
    elif language == "ja":
        if re.search(r"(でございます|いたします|くださいませ|なさいます)$", stripped):
            return 3
        if re.search(r"(です|ます|でした|ません|ください)$", stripped):
            return 2
        if re.search(r"(だ|である|する|した|ない|よう)$", stripped):
            return 1
    return 0


def protect_shared_spans(ko: str, ja: str, maximum: int = 64) -> tuple[str, str]:
    """Replace exact shared structured spans with reversible slot symbols.

    The serving preprocessor keeps the slot->surface map. The training shards only
    need the stable symbols, which teach TETM to preserve rather than hallucinate.
    """

    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    return protect_shared_structured_spans(
        ko,
        ja,
        slot_symbols=SLOT_SYMBOLS[:maximum],
    )


class ShardWriter:
    def __init__(
        self,
        root: Path,
        split: str,
        shard_size: int,
        language_to_id: dict[str, int],
    ):
        self.root = root / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = shard_size
        self.language_to_id = language_to_id
        self.shard_index = 0
        self.records: list[tuple[int, ...]] = []
        self.record_metadata: list[bytes] = []
        self.src_offset = 0
        self.tgt_offset = 0
        self.total_records = 0
        self._src_handle: BinaryIO | None = None
        self._tgt_handle: BinaryIO | None = None
        self._open_shard()

    def _prefix(self) -> str:
        return f"{self.shard_index:05d}"

    def _open_shard(self) -> None:
        self._src_handle = (self.root / f"{self._prefix()}.src.bin").open("wb")
        self._tgt_handle = (self.root / f"{self._prefix()}.tgt.bin").open("wb")
        self.records = []
        self.record_metadata = []
        self.src_offset = 0
        self.tgt_offset = 0

    def add(
        self,
        src_ids: Sequence[int],
        tgt_ids: Sequence[int],
        src_register: int,
        tgt_register: int,
        src_language: str,
        tgt_language: str,
        source_id: int,
        quality_score: int,
        synthetic: bool,
        forward_only: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> None:
        assert self._src_handle is not None and self._tgt_handle is not None
        metadata_payload = encode_record_metadata(metadata)
        src_array = np.asarray(src_ids, dtype=np.uint32)
        tgt_array = np.asarray(tgt_ids, dtype=np.uint32)
        src_array.tofile(self._src_handle)
        tgt_array.tofile(self._tgt_handle)
        self.records.append(
            (
                self.src_offset,
                len(src_array),
                self.tgt_offset,
                len(tgt_array),
                src_register,
                tgt_register,
                self.language_to_id[src_language],
                self.language_to_id[tgt_language],
                source_id,
                quality_score,
                int(synthetic),
                int(forward_only),
            )
        )
        self.record_metadata.append(metadata_payload)
        self.src_offset += len(src_array)
        self.tgt_offset += len(tgt_array)
        self.total_records += 1
        if len(self.records) >= self.shard_size:
            self._finish_shard()
            self.shard_index += 1
            self._open_shard()

    def _finish_shard(self) -> None:
        assert self._src_handle is not None and self._tgt_handle is not None
        self._src_handle.close()
        self._tgt_handle.close()
        if self.records:
            np.save(
                self.root / f"{self._prefix()}.idx.npy",
                np.asarray(self.records, dtype=INDEX_DTYPE),
                allow_pickle=False,
            )
            if any(self.record_metadata):
                metadata_rows: list[tuple[int, int]] = []
                offset = 0
                data_path = self.root / f"{self._prefix()}{RECORD_METADATA_DATA_SUFFIX}"
                with data_path.open("wb") as handle:
                    for payload in self.record_metadata:
                        handle.write(payload)
                        metadata_rows.append((offset, len(payload)))
                        offset += len(payload)
                np.save(
                    self.root / f"{self._prefix()}{RECORD_METADATA_INDEX_SUFFIX}",
                    np.asarray(metadata_rows, dtype=RECORD_METADATA_INDEX_DTYPE),
                    allow_pickle=False,
                )
        else:
            for side in ("src", "tgt"):
                (self.root / f"{self._prefix()}.{side}.bin").unlink(missing_ok=True)
        self._src_handle = None
        self._tgt_handle = None

    def close(self) -> None:
        if self._src_handle is not None:
            self._finish_shard()


_QUALITY_REASON_FIELDS = {
    "too_short": "too_short",
    "identical_text": "identical_text",
    "length_ratio": "length_ratio_outlier",
    "ko_script_mismatch": "language_mismatch",
    "ja_script_mismatch": "language_mismatch",
    "control_characters": "control_characters",
    "excessive_repetition": "excessive_repetition",
}


_PrepareBatchInput: TypeAlias = tuple[
    int,
    list[bytes],
    QualityPolicy,
    bool,
    tuple[tuple[str, str], ...],
]
_PrepareEvent: TypeAlias = tuple[str, tuple[Any, ...]]

_PREPARE_WORKER_TOKENIZER: SionTokenizer | None = None


def _initialize_prepare_worker(tokenizer_model: str) -> None:
    global _PREPARE_WORKER_TOKENIZER
    _PREPARE_WORKER_TOKENIZER = SionTokenizer(  # pyright: ignore[reportConstantRedefinition]
        tokenizer_model
    )


def _process_prepare_batch(args: _PrepareBatchInput) -> list[_PrepareEvent]:
    """CPU-heavy, order-preserving row work executed in worker processes."""

    source_id, rows, quality_policy, filter_quality, language_pairs = args
    tokenizer = _PREPARE_WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("prepare worker tokenizer was not initialized")
    output: list[_PrepareEvent] = []
    for raw_line in rows:
        output.append(("physical_line", (source_id,)))
        record_group_key = hashlib.sha256(raw_line.strip()).hexdigest()
        try:
            line = raw_line.decode("utf-8-sig")
        except UnicodeDecodeError:
            output.append(("invalid_utf8", (source_id,)))
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            output.append(("invalid_json", (source_id,)))
            continue
        record_is_synthetic = synthetic_record(row)
        expansion = expand_parallel_record(row, language_pairs)
        for issue in expansion.issues:
            output.append((issue, (source_id,)))
        for pair in expansion.pairs:
            language_pair = (pair.language_a, pair.language_b)
            text_a, text_b = canonical_text(pair.text_a), canonical_text(pair.text_b)
            assessment = assess_pair(
                text_a,
                text_b,
                quality_policy,
                languages=language_pair,
            )
            assessment = apply_record_quality_profile(
                assessment,
                pair.metadata.get("quality_profile"),
            )
            unsafe = "control_characters" in assessment.rejection_reasons
            if not assessment.accepted and (filter_quality or unsafe):
                output.append(
                    (
                        "quality_filtered",
                        (
                            source_id,
                            assessment.rejection_reasons,
                            assessment.warning_reasons,
                        ),
                    )
                )
                continue
            encoded_a, encoded_b = protect_shared_spans(text_a, text_b)
            output.append(
                (
                    "candidate",
                    (
                        source_id,
                        record_group_key,
                        record_is_synthetic,
                        pair.language_a,
                        pair.language_b,
                        pair.metadata,
                        text_a,
                        text_b,
                        tokenizer.encode(encoded_a),
                        tokenizer.encode(encoded_b),
                        infer_register(text_a, pair.language_a),
                        infer_register(text_b, pair.language_b),
                        assessment.score,
                        assessment.rejection_reasons,
                        assessment.warning_reasons,
                    ),
                )
            )
    return output


def _prepare_input_batches(
    paths: Sequence[Path],
    quality_policy: QualityPolicy,
    filter_quality: bool,
    language_pairs: tuple[tuple[str, str], ...],
    train_only_prefixes: tuple[str, ...],
    batch_size: int = 512,
) -> Iterator[_PrepareBatchInput]:
    # Real rows own duplicate precedence even when a synthetic filename sorts
    # first or a single source mixes row-level ``synthetic`` markers. Reading
    # mixed files twice avoids buffering an unbounded corpus merely to reorder
    # it; every line is yielded in exactly one pass.
    for synthetic_pass in (False, True):
        for source_id, path in enumerate(paths):
            path_is_synthetic = synthetic_path(path, train_only_prefixes)
            if path_is_synthetic != synthetic_pass and path_is_synthetic:
                continue
            with path.open("rb") as handle:
                rows: list[bytes] = []
                for raw_line in handle:
                    row_is_synthetic = path_is_synthetic
                    if not path_is_synthetic:
                        try:
                            decoded = raw_line.decode("utf-8-sig")
                            row_is_synthetic = synthetic_record(json.loads(decoded))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            row_is_synthetic = False
                    if row_is_synthetic != synthetic_pass:
                        continue
                    rows.append(raw_line)
                    if len(rows) >= batch_size:
                        yield source_id, rows, quality_policy, filter_quality, language_pairs
                        rows = []
                if rows:
                    yield source_id, rows, quality_policy, filter_quality, language_pairs


def _increment(stats: PrepareStats, field: str, amount: int = 1) -> None:
    setattr(stats, field, getattr(stats, field) + amount)


def _record_quality_reasons(targets: Sequence[PrepareStats], reasons: Sequence[str]) -> None:
    fields = {
        _QUALITY_REASON_FIELDS[reason] for reason in reasons if reason in _QUALITY_REASON_FIELDS
    }
    for stats in targets:
        for field in fields:
            _increment(stats, field)


class _MemoryDigestSet:
    def __init__(self):
        self.values: set[bytes] = set()

    def add_if_new(self, digest: bytes) -> bool:
        if digest in self.values:
            return False
        self.values.add(digest)
        return True

    def close(self) -> None:
        self.values.clear()


class _SqliteDigestSet:
    """Disk-backed exact digest set for corpora too large for a Python set."""

    def __init__(self, path: Path):
        self.path = path
        self.connection: sqlite3.Connection | None = sqlite3.connect(path)
        connection = self.connection
        assert connection is not None
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        connection.execute("CREATE TABLE digests (digest BLOB PRIMARY KEY) WITHOUT ROWID")
        connection.execute("BEGIN IMMEDIATE")

    def add_if_new(self, digest: bytes) -> bool:
        if self.connection is None:
            raise RuntimeError("digest store is closed")
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO digests(digest) VALUES (?)",
            (sqlite3.Binary(digest),),
        )
        return cursor.rowcount == 1

    def close(self) -> None:
        if self.connection is not None:
            self.connection.commit()
            self.connection.close()
            self.connection = None


# 합성 데이터가 든 입력 파일의 접두어. 이런 파일은 train split 에만 넣습니다 —
# 역번역이나 이어붙이기로 만든 예제가 holdout 에 들어가면 점수가 실제 번역 품질이
# 아니라 합성 규칙을 재게 됩니다.
DEFAULT_TRAIN_ONLY_PREFIXES: tuple[str, ...] = DEFAULT_SYNTHETIC_PREFIXES


@dataclass(frozen=True)
class _FileSnapshot:
    resolved_path: str
    size: int
    sha256: str
    modified_ns: int
    changed_ns: int
    device: int
    inode: int
    file_attributes: int


def _path_exists(path: Path) -> bool:
    """Return true for ordinary paths and dangling links alike."""

    return os.path.lexists(path)


def _is_reparse_stat(value: os.stat_result) -> bool:
    return bool(
        stat.S_ISLNK(value.st_mode)
        or getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _assert_no_reparse_components(path: Path, *, role: str) -> None:
    """Reject symlinks and Windows reparse points anywhere in an existing path."""

    absolute = _absolute_path(path)
    anchor = Path(absolute.anchor)
    current = anchor
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        current /= part
        if not _path_exists(current):
            continue
        try:
            current_stat = os.lstat(current)
        except OSError as error:
            raise OSError(f"Cannot inspect {role} path component: {current}") from error
        if _is_reparse_stat(current_stat):
            raise ValueError(f"{role} cannot traverse a symlink or reparse point: {current}")


def _regular_file_stat(path: Path, *, role: str) -> os.stat_result:
    _assert_no_reparse_components(path, role=role)
    try:
        value = os.lstat(path)
    except OSError as error:
        raise OSError(f"Cannot inspect {role}: {path}") from error
    if _is_reparse_stat(value) or not stat.S_ISREG(value.st_mode):
        raise ValueError(f"{role} must be a regular file without reparse points: {path}")
    return value


def _snapshot_file(path: Path, *, role: str) -> _FileSnapshot:
    """Hash a regular file and reject replacement or mutation during hashing."""

    before = _regular_file_stat(path, role=role)
    try:
        resolved_before = path.resolve(strict=True)
        content_hash = file_sha256(path)
        resolved_after = path.resolve(strict=True)
    except OSError as error:
        raise OSError(f"Cannot hash {role}: {path}") from error
    after = _regular_file_stat(path, role=role)
    identity_before = (
        str(resolved_before),
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_dev,
        before.st_ino,
        getattr(before, "st_file_attributes", 0),
    )
    identity_after = (
        str(resolved_after),
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_dev,
        after.st_ino,
        getattr(after, "st_file_attributes", 0),
    )
    if identity_after != identity_before:
        raise RuntimeError(f"{role} changed while its SHA-256 snapshot was captured: {path}")
    return _FileSnapshot(
        resolved_path=str(resolved_after),
        size=after.st_size,
        sha256=content_hash,
        modified_ns=after.st_mtime_ns,
        changed_ns=after.st_ctime_ns,
        device=after.st_dev,
        inode=after.st_ino,
        file_attributes=getattr(after, "st_file_attributes", 0),
    )


def _secure_expand_inputs(patterns: Sequence[str]) -> list[Path]:
    """Expand the public input syntax without resolving links before validation."""

    candidates: set[Path] = set()
    for pattern in patterns:
        candidate = Path(pattern)
        if any(character in pattern for character in "*?[]"):
            candidates.update(Path(match) for match in glob.glob(pattern))
            continue
        if _path_exists(candidate):
            _assert_no_reparse_components(candidate, role="dataset input")
            candidate_stat = os.lstat(candidate)
            if stat.S_ISDIR(candidate_stat.st_mode):
                candidates.update(candidate.glob("*.jsonl"))
                continue
        candidates.add(candidate)

    resolved: set[Path] = set()
    for candidate in candidates:
        if not _path_exists(candidate):
            continue
        _regular_file_stat(candidate, role="dataset input")
        resolved.add(candidate.resolve(strict=True))
    return sorted(resolved, key=lambda item: (item.name, str(item)))


def _capture_input_snapshots(
    input_patterns: Sequence[str],
) -> tuple[list[Path], tuple[_FileSnapshot, ...]]:
    paths = _secure_expand_inputs(input_patterns)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {input_patterns}")
    snapshots = tuple(_snapshot_file(path, role="dataset input") for path in paths)
    names = [Path(snapshot.resolved_path).name for snapshot in snapshots]
    if len(names) != len(set(names)):
        raise ValueError("dataset input file names must be unique")
    return paths, snapshots


def _verify_input_snapshots(
    input_patterns: Sequence[str],
    expected_paths: Sequence[Path],
    expected_sources: tuple[_FileSnapshot, ...],
    tokenizer_path: Path,
    expected_tokenizer: _FileSnapshot,
) -> None:
    try:
        actual_paths, actual_sources = _capture_input_snapshots(input_patterns)
        actual_tokenizer = _snapshot_file(tokenizer_path, role="tokenizer model")
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError("Dataset inputs or tokenizer changed during preparation") from error
    if list(actual_paths) != list(expected_paths) or actual_sources != expected_sources:
        raise RuntimeError("Dataset input file set or bytes changed during preparation")
    if actual_tokenizer != expected_tokenizer:
        raise RuntimeError("Tokenizer bytes changed during dataset preparation")


def prepare_preprocessing_options(
    *,
    shard_size: int = 100_000,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    max_tokens_per_side: int = 510,
    quality_policy: QualityPolicy | None = None,
    filter_quality: bool = True,
    prevent_target_leakage: bool = True,
    approximate_split: bool = False,
    dedup_backend: str = "sqlite",
    source_only_languages: Sequence[str] = (),
    translation_directions: Sequence[Sequence[str]] = (),
    train_only_prefixes: Sequence[str] = DEFAULT_TRAIN_ONLY_PREFIXES,
    managed_augmentation_prefix: str | None = None,
    synthetic_sampling_weight: float = DEFAULT_SYNTHETIC_SAMPLING_WEIGHT,
    language_pair_count: int = 1,
) -> dict[str, Any]:
    """Return the fingerprint/manifest contract for every output-affecting option."""

    quality_policy = quality_policy or QualityPolicy()
    endpoint_key_schema = (
        "language-prefixed-minhash-char5-v1" if approximate_split else "language-prefixed-exact-v1"
    )
    split_key_schema = "record-sha256-v1" if language_pair_count > 1 else endpoint_key_schema
    options = {
        "approximate_split": approximate_split,
        "dedup_backend": dedup_backend,
        "endpoint_leakage_guard": "language-endpoint-bloom-v2",
        "endpoint_leakage_key": endpoint_key_schema,
        "filter_quality": filter_quality,
        "index_dtype": INDEX_DTYPE.descr,
        "max_tokens_per_side": max_tokens_per_side,
        "prevent_target_leakage": prevent_target_leakage,
        "quality_policy": quality_policy.to_dict(),
        "record_metadata_fields": list(RECORD_METADATA_FIELDS),
        "record_metadata_format": RECORD_METADATA_FORMAT,
        "record_metadata_index_dtype": RECORD_METADATA_INDEX_DTYPE.descr,
        "shard_size": shard_size,
        "source_only_languages": list(source_only_languages),
        "translation_directions": [list(direction) for direction in translation_directions],
        "split_key": split_key_schema,
        "synthetic_sampling_weight": synthetic_sampling_weight,
        "test_fraction": test_fraction,
        "train_only_prefixes": list(train_only_prefixes),
        "validation_fraction": validation_fraction,
    }
    if managed_augmentation_prefix is not None:
        options["managed_augmentation_prefix"] = managed_augmentation_prefix
    # Normalize tuples and NumPy dtype descriptors through the same JSON
    # representation persisted by both the fingerprint sidecar and manifest.
    normalized: object = json.loads(_canonical_json(options))
    if not isinstance(normalized, dict):
        raise AssertionError("prepare preprocessing options must normalize to an object")
    return cast(dict[str, Any], normalized)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _fingerprint_from_snapshots(
    source_snapshots: tuple[_FileSnapshot, ...],
    tokenizer_snapshot: _FileSnapshot,
    *,
    language_pairs: tuple[tuple[str, str], ...],
    preprocessing_options: Mapping[str, Any],
) -> DatasetFingerprint:
    return DatasetFingerprint(
        files=tuple(
            FileFingerprint(
                name=Path(snapshot.resolved_path).name,
                size=snapshot.size,
                sha256=snapshot.sha256,
            )
            for snapshot in source_snapshots
        ),
        language_pairs=language_pairs,
        tokenizer_sha256=tokenizer_snapshot.sha256,
        preprocessing_schema=PREPROCESSING_SCHEMA,
        preprocessing_options_json=_canonical_json(preprocessing_options),
    )


def _write_json_durable(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json_object(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {role}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return cast(dict[str, Any], value)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_regular_directory(path: Path, *, role: str) -> None:
    _assert_no_reparse_components(path, role=role)
    try:
        value = os.lstat(path)
    except OSError as error:
        raise OSError(f"Cannot inspect {role}: {path}") from error
    if _is_reparse_stat(value) or not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"{role} must be a regular directory: {path}")


def _fsync_staging_tree(root: Path) -> None:
    _assert_regular_directory(root, role="dataset staging directory")
    directories = [root]
    for artifact in root.rglob("*"):
        _assert_no_reparse_components(artifact, role="dataset staging artifact")
        artifact_stat = os.lstat(artifact)
        if _is_reparse_stat(artifact_stat):
            raise RuntimeError(f"Dataset staging contains a reparse point: {artifact}")
        if stat.S_ISDIR(artifact_stat.st_mode):
            directories.append(artifact)
        elif stat.S_ISREG(artifact_stat.st_mode):
            with artifact.open("r+b") as handle:
                os.fsync(handle.fileno())
        else:
            raise RuntimeError(f"Dataset staging contains a non-regular artifact: {artifact}")
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _refuse_or_remove_empty_output(output_dir: Path) -> None:
    if not _path_exists(output_dir):
        return
    _assert_regular_directory(output_dir, role="dataset output")
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory so stale shards cannot mix with this run."
        )
    output_dir.rmdir()


def _staging_candidates(output_dir: Path) -> list[Path]:
    prefix = f".{output_dir.name}.staging-"
    if not output_dir.parent.is_dir():
        return []
    return sorted(
        (
            candidate
            for candidate in output_dir.parent.iterdir()
            if candidate.name.startswith(prefix)
        ),
        key=lambda candidate: candidate.name,
    )


def _quarantine_staging(candidate: Path, output_dir: Path) -> None:
    """Move an invalid regular staging tree aside before recursive cleanup."""

    _assert_regular_directory(candidate, role="orphan dataset staging")
    pending = [candidate]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise RuntimeError(f"Cannot inspect orphan dataset staging: {directory}") from error
        for entry in entries:
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise RuntimeError(
                    f"Cannot inspect orphan dataset staging artifact: {entry.path}"
                ) from error
            if _is_reparse_stat(entry_stat):
                raise RuntimeError(
                    "Refusing to clean orphan staging containing a symlink/reparse point: "
                    f"{entry.path}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                pending.append(Path(entry.path))
            elif not stat.S_ISREG(entry_stat.st_mode):
                raise RuntimeError(f"Refusing to clean non-regular staging artifact: {entry.path}")
    while True:
        quarantine = output_dir.with_name(f".{output_dir.name}.rejected-{uuid.uuid4().hex}")
        if not _path_exists(quarantine):
            break
    os.rename(candidate, quarantine)
    _fsync_directory(output_dir.parent)
    shutil.rmtree(quarantine)
    _fsync_directory(output_dir.parent)


def _discard_private_staging(staging_dir: Path, output_dir: Path) -> None:
    if not _path_exists(staging_dir):
        return
    _quarantine_staging(staging_dir, output_dir)


def _validate_staging_tree_shape(staging_dir: Path) -> None:
    _assert_regular_directory(staging_dir, role="dataset staging")
    allowed = {
        "train",
        "validation",
        "test",
        RAW_FINGERPRINT_FILENAME,
        "manifest.json",
        PREPARE_COMPLETION_FILENAME,
    }
    actual = {candidate.name for candidate in staging_dir.iterdir()}
    if actual != allowed:
        raise ValueError(
            "Dataset staging top-level artifacts differ from the complete contract: "
            f"missing={sorted(allowed - actual)}, unexpected={sorted(actual - allowed)}"
        )
    for split in ("train", "validation", "test"):
        _assert_regular_directory(staging_dir / split, role=f"dataset {split} split")
    for filename in (RAW_FINGERPRINT_FILENAME, "manifest.json", PREPARE_COMPLETION_FILENAME):
        _regular_file_stat(staging_dir / filename, role=f"dataset {filename}")


def _prepare_stats_from_manifest(manifest: Mapping[str, Any]) -> PrepareStats:
    raw_stats: object = manifest.get("stats")
    if not isinstance(raw_stats, Mapping):
        raise ValueError("Dataset manifest stats must be an object")
    stats_mapping = cast(Mapping[object, object], raw_stats)
    expected_fields = {field.name for field in fields(PrepareStats)}
    if set(stats_mapping) != expected_fields:
        raise ValueError("Dataset manifest stats fields differ from PrepareStats")
    normalized: dict[str, int] = {}
    for name in expected_fields:
        value = stats_mapping[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Dataset manifest stat is invalid: {name}")
        normalized[name] = value
    return PrepareStats(**normalized)


def _validate_manifest_sources(
    manifest: Mapping[str, Any],
    source_snapshots: tuple[_FileSnapshot, ...],
    train_only_prefixes: tuple[str, ...],
    stats: PrepareStats,
) -> tuple[PrepareStats, ...]:
    raw_sources: object = manifest.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("Dataset manifest source list differs from its input snapshot")
    source_values = cast(list[object], raw_sources)
    if len(source_values) != len(source_snapshots):
        raise ValueError("Dataset manifest source list differs from its input snapshot")
    accumulated = {field.name: 0 for field in fields(PrepareStats)}
    validated_stats: list[PrepareStats] = []
    for source_id, (raw_source, snapshot) in enumerate(
        zip(source_values, source_snapshots, strict=True)
    ):
        if not isinstance(raw_source, Mapping):
            raise ValueError(f"Dataset manifest source {source_id} must be an object")
        source_mapping = cast(Mapping[object, object], raw_source)
        if set(source_mapping) != {
            "id",
            "name",
            "path",
            "synthetic_file",
            "stats",
            "mean_quality_score",
        }:
            raise ValueError(f"Dataset manifest source {source_id} fields are invalid")
        source_path = Path(snapshot.resolved_path)
        if (
            source_mapping.get("id") != source_id
            or source_mapping.get("name") != source_path.name
            or source_mapping.get("path") != snapshot.resolved_path
            or source_mapping.get("synthetic_file")
            != synthetic_path(source_path, train_only_prefixes)
        ):
            raise ValueError(f"Dataset manifest source {source_id} identity is invalid")
        source_stats = _prepare_stats_from_manifest({"stats": source_mapping.get("stats")})
        validated_stats.append(source_stats)
        for field in fields(PrepareStats):
            accumulated[field.name] += getattr(source_stats, field.name)
        expected_mean = source_stats.quality_score_sum / max(source_stats.valid_pairs, 1)
        raw_mean = source_mapping.get("mean_quality_score")
        if (
            isinstance(raw_mean, bool)
            or not isinstance(raw_mean, (int, float))
            or not math.isclose(float(raw_mean), expected_mean, rel_tol=1e-12, abs_tol=1e-15)
        ):
            raise ValueError(f"Dataset manifest source {source_id} mean score is invalid")
    if accumulated != asdict(stats):
        raise ValueError("Dataset manifest per-source stats do not add up to total stats")
    return tuple(validated_stats)


def _validate_indexed_payload_semantics(
    staging_dir: Path,
    *,
    stats: PrepareStats,
    source_stats: tuple[PrepareStats, ...],
    languages: tuple[str, ...],
    normalized_pairs: tuple[tuple[str, str], ...],
    translation_directions: tuple[tuple[str, str], ...],
    source_only: tuple[str, ...],
) -> None:
    """Validate every field consumed later by the indexed dataset loader."""

    source_count = len(source_stats)
    source_only_set = set(source_only)
    allowed_language_pairs: set[tuple[int, int]] = set()
    language_to_id = {language: index for index, language in enumerate(languages)}
    direction_set = set(translation_directions)
    for source_language, target_language in translation_directions:
        allowed_language_pairs.add(
            (
                language_to_id[source_language],
                language_to_id[target_language],
            )
        )

    source_rows = np.zeros(source_count, dtype=np.int64)
    source_synthetic = np.zeros(source_count, dtype=np.int64)
    source_forward_only = np.zeros(source_count, dtype=np.int64)
    source_quality = np.zeros(source_count, dtype=np.int64)
    source_src_tokens = np.zeros(source_count, dtype=np.int64)
    source_tgt_tokens = np.zeros(source_count, dtype=np.int64)
    source_split_rows = {
        split: np.zeros(source_count, dtype=np.int64) for split in ("train", "validation", "test")
    }
    split_rows: dict[str, int] = {}

    for split in ("train", "validation", "test"):
        split_dir = staging_dir / split
        index_paths = sorted(split_dir.glob("*.idx.npy"))
        expected_artifacts: set[str] = set()
        split_total = 0
        for index_path in index_paths:
            prefix = index_path.name.removesuffix(".idx.npy")
            src_path = split_dir / f"{prefix}.src.bin"
            tgt_path = split_dir / f"{prefix}.tgt.bin"
            for token_path in (src_path, tgt_path):
                token_stat = _regular_file_stat(token_path, role="dataset token shard")
                if token_stat.st_size % np.dtype(np.uint32).itemsize:
                    raise ValueError(f"Dataset token shard byte length is invalid: {token_path}")
            try:
                index = np.load(index_path, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ValueError(f"Cannot read dataset index: {index_path}") from error
            if index.dtype != INDEX_DTYPE:
                raise ValueError(f"Dataset index dtype is invalid: {index_path}")

            src_offsets = np.asarray(index["src_offset"], dtype=np.uint64)
            src_lengths = np.asarray(index["src_length"], dtype=np.uint64)
            tgt_offsets = np.asarray(index["tgt_offset"], dtype=np.uint64)
            tgt_lengths = np.asarray(index["tgt_length"], dtype=np.uint64)
            expected_src_offsets = np.concatenate(
                (np.zeros(1, dtype=np.uint64), np.cumsum(src_lengths[:-1], dtype=np.uint64))
            )
            expected_tgt_offsets = np.concatenate(
                (np.zeros(1, dtype=np.uint64), np.cumsum(tgt_lengths[:-1], dtype=np.uint64))
            )
            if not np.array_equal(src_offsets, expected_src_offsets) or not np.array_equal(
                tgt_offsets,
                expected_tgt_offsets,
            ):
                raise ValueError(f"Dataset token offsets are not contiguous: {index_path}")
            src_tokens = int(src_lengths.sum(dtype=np.uint64))
            tgt_tokens = int(tgt_lengths.sum(dtype=np.uint64))
            if src_tokens * np.dtype(np.uint32).itemsize != src_path.stat().st_size:
                raise ValueError(f"Dataset source token offsets exceed their shard: {index_path}")
            if tgt_tokens * np.dtype(np.uint32).itemsize != tgt_path.stat().st_size:
                raise ValueError(f"Dataset target token offsets exceed their shard: {index_path}")

            source_ids = np.asarray(index["source_id"], dtype=np.int64)
            src_language_ids = np.asarray(index["src_language_id"], dtype=np.int64)
            tgt_language_ids = np.asarray(index["tgt_language_id"], dtype=np.int64)
            synthetic = np.asarray(index["synthetic"], dtype=np.int64)
            forward_only = np.asarray(index["forward_only"], dtype=np.int64)
            quality = np.asarray(index["quality_score"], dtype=np.int64)
            if source_ids.size and (
                int(source_ids.min()) < 0 or int(source_ids.max()) >= source_count
            ):
                raise ValueError(f"Dataset index source_id is outside the manifest: {index_path}")
            for values, name, upper_bound in (
                (src_language_ids, "src_language_id", len(languages)),
                (tgt_language_ids, "tgt_language_id", len(languages)),
            ):
                if values.size and (int(values.min()) < 0 or int(values.max()) >= upper_bound):
                    raise ValueError(f"Dataset index {name} is outside the manifest: {index_path}")
            if not bool(np.isin(synthetic, (0, 1)).all()) or not bool(
                np.isin(forward_only, (0, 1)).all()
            ):
                raise ValueError(f"Dataset index boolean flags are invalid: {index_path}")
            if quality.size and (int(quality.min()) < 0 or int(quality.max()) > 100):
                raise ValueError(f"Dataset index quality score is invalid: {index_path}")
            if not bool(
                np.isin(np.asarray(index["src_register"], dtype=np.int64), (0, 1, 2, 3)).all()
            ) or not bool(
                np.isin(np.asarray(index["tgt_register"], dtype=np.int64), (0, 1, 2, 3)).all()
            ):
                raise ValueError(f"Dataset index register is invalid: {index_path}")
            scoped_direction_rows: dict[int, list[str]] = {}
            for row_id, (src_language_id, tgt_language_id, forward_flag) in enumerate(
                zip(
                    src_language_ids,
                    tgt_language_ids,
                    forward_only,
                    strict=True,
                )
            ):
                pair = (int(src_language_id), int(tgt_language_id))
                if pair not in allowed_language_pairs:
                    raise ValueError(f"Dataset index language pair is not configured: {index_path}")
                source_language = languages[pair[0]]
                target_language = languages[pair[1]]
                reverse_trained = (target_language, source_language) in direction_set
                if (not bool(forward_flag) and not reverse_trained) or (
                    target_language in source_only_set
                ):
                    raise ValueError(
                        f"Dataset index translation direction is invalid: {index_path}"
                    )
                if bool(forward_flag) and reverse_trained:
                    scoped_direction_rows[row_id] = [source_language, target_language]

            row_count = len(index)
            split_total += row_count
            counts = np.bincount(source_ids, minlength=source_count)[:source_count]
            source_rows += counts
            source_split_rows[split] += counts
            source_synthetic += np.bincount(
                source_ids,
                weights=synthetic,
                minlength=source_count,
            )[:source_count].astype(np.int64)
            source_forward_only += np.bincount(
                source_ids,
                weights=forward_only,
                minlength=source_count,
            )[:source_count].astype(np.int64)
            source_quality += np.bincount(
                source_ids,
                weights=quality,
                minlength=source_count,
            )[:source_count].astype(np.int64)
            source_src_tokens += np.bincount(
                source_ids,
                weights=src_lengths,
                minlength=source_count,
            )[:source_count].astype(np.int64)
            source_tgt_tokens += np.bincount(
                source_ids,
                weights=tgt_lengths,
                minlength=source_count,
            )[:source_count].astype(np.int64)

            metadata_index_path = split_dir / f"{prefix}{RECORD_METADATA_INDEX_SUFFIX}"
            metadata_data_path = split_dir / f"{prefix}{RECORD_METADATA_DATA_SUFFIX}"
            metadata_present = _path_exists(metadata_index_path) or _path_exists(metadata_data_path)
            expected_artifacts.update({index_path.name, src_path.name, tgt_path.name})
            if metadata_present:
                _regular_file_stat(metadata_index_path, role="dataset record metadata index")
                metadata_stat = _regular_file_stat(
                    metadata_data_path,
                    role="dataset record metadata payload",
                )
                metadata_index = np.load(metadata_index_path, allow_pickle=False)
                if (
                    metadata_index.dtype != RECORD_METADATA_INDEX_DTYPE
                    or len(metadata_index) != row_count
                ):
                    raise ValueError(
                        f"Dataset record metadata index is invalid: {metadata_index_path}"
                    )
                metadata_offsets = np.asarray(metadata_index["offset"], dtype=np.uint64)
                metadata_lengths = np.asarray(metadata_index["length"], dtype=np.uint64)
                expected_metadata_offsets = np.concatenate(
                    (
                        np.zeros(1, dtype=np.uint64),
                        np.cumsum(metadata_lengths[:-1], dtype=np.uint64),
                    )
                )
                if not np.array_equal(metadata_offsets, expected_metadata_offsets):
                    raise ValueError(
                        f"Dataset record metadata offsets are not contiguous: {metadata_index_path}"
                    )
                if int(metadata_lengths.sum(dtype=np.uint64)) != metadata_stat.st_size:
                    raise ValueError(
                        f"Dataset record metadata offsets exceed their payload: {metadata_index_path}"
                    )
                if scoped_direction_rows:
                    metadata_store = np.memmap(metadata_data_path, dtype=np.uint8, mode="r")
                    for row_id, expected_direction in scoped_direction_rows.items():
                        metadata_row = metadata_index[row_id]
                        offset = int(metadata_row["offset"])
                        length = int(metadata_row["length"])
                        payload = np.asarray(
                            metadata_store[offset : offset + length],
                            dtype=np.uint8,
                        ).tobytes()
                        if (
                            decode_record_metadata(payload).get("training_direction")
                            != expected_direction
                        ):
                            raise ValueError(
                                "Dataset row-scoped direction lacks matching metadata: "
                                f"{index_path} row={row_id}"
                            )
                expected_artifacts.update({metadata_index_path.name, metadata_data_path.name})
            elif scoped_direction_rows:
                raise ValueError(f"Dataset row-scoped directions require metadata: {index_path}")
        actual_artifacts = {path.name for path in split_dir.iterdir()}
        if actual_artifacts != expected_artifacts:
            raise ValueError(
                f"Dataset split artifacts are incomplete or unexpected: {split}; "
                f"missing={sorted(expected_artifacts - actual_artifacts)}, "
                f"unexpected={sorted(actual_artifacts - expected_artifacts)}"
            )
        split_rows[split] = split_total

    if split_rows != {
        "train": stats.train,
        "validation": stats.validation,
        "test": stats.test,
    }:
        raise ValueError("Dataset manifest split counts differ from indexed payload rows")
    if stats.valid_pairs != stats.train + stats.validation + stats.test:
        raise ValueError("Dataset manifest total valid pairs differ from its split counts")
    for source_id, expected in enumerate(source_stats):
        derived = {
            "valid_pairs": int(source_rows[source_id]),
            "train": int(source_split_rows["train"][source_id]),
            "validation": int(source_split_rows["validation"][source_id]),
            "test": int(source_split_rows["test"][source_id]),
            "synthetic_pairs": int(source_synthetic[source_id]),
            "forward_only_pairs": int(source_forward_only[source_id]),
            "quality_score_sum": int(source_quality[source_id]),
            "ko_tokens": int(source_src_tokens[source_id]),
            "ja_tokens": int(source_tgt_tokens[source_id]),
        }
        for name, value in derived.items():
            if getattr(expected, name) != value:
                raise ValueError(
                    f"Dataset manifest source {source_id} {name} differs from indexed payload"
                )


def _completion_payload(staging_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        "schema": PREPARE_COMPLETION_SCHEMA,
        "manifest_sha256": file_sha256(staging_dir / "manifest.json"),
        "raw_fingerprint_sha256": file_sha256(staging_dir / RAW_FINGERPRINT_FILENAME),
        "artifact_inventory_sha256": hashlib.sha256(
            _canonical_json(manifest["artifact_inventory"]).encode("utf-8")
        ).hexdigest(),
    }


def _validate_complete_staging(
    staging_dir: Path,
    *,
    expected_fingerprint: DatasetFingerprint,
    source_snapshots: tuple[_FileSnapshot, ...],
    tokenizer_snapshot: _FileSnapshot,
    normalized_pairs: tuple[tuple[str, str], ...],
    translation_directions: tuple[tuple[str, str], ...],
    languages: tuple[str, ...],
    source_only: tuple[str, ...],
    train_only_prefixes: tuple[str, ...],
    preprocessing_options: Mapping[str, Any],
) -> PrepareStats:
    _validate_staging_tree_shape(staging_dir)
    watched = tuple(
        _snapshot_file(staging_dir / filename, role=f"dataset staging {filename}")
        for filename in (
            RAW_FINGERPRINT_FILENAME,
            "manifest.json",
            PREPARE_COMPLETION_FILENAME,
        )
    )
    fingerprint = _read_json_object(
        staging_dir / RAW_FINGERPRINT_FILENAME,
        role="dataset raw fingerprint",
    )
    expected_fingerprint_payload = expected_fingerprint.to_dict()
    if fingerprint != expected_fingerprint_payload:
        raise ValueError("Dataset staging raw fingerprint differs from the expected contract")
    manifest = _read_json_object(staging_dir / "manifest.json", role="dataset manifest")
    if manifest.get("format") != INDEX_FORMAT:
        raise ValueError("Dataset staging manifest format is unsupported")
    if manifest.get("fingerprint") != expected_fingerprint_payload:
        raise ValueError("Dataset manifest fingerprint differs from its raw sidecar")
    if manifest.get("preprocessing_schema") != PREPROCESSING_SCHEMA:
        raise ValueError("Dataset manifest preprocessing schema is invalid")
    if manifest.get("preprocessing_options") != dict(preprocessing_options):
        raise ValueError("Dataset manifest preprocessing options differ from its raw sidecar")
    if manifest.get("language_pairs") != [list(pair) for pair in normalized_pairs]:
        raise ValueError("Dataset manifest language pairs differ from the expected contract")
    if manifest.get("language_pair") != list(normalized_pairs[0]):
        raise ValueError("Dataset manifest primary language pair differs from the contract")
    if manifest.get("translation_directions") != [
        list(direction) for direction in translation_directions
    ]:
        raise ValueError("Dataset manifest translation directions differ from the contract")
    if manifest.get("languages") != list(languages):
        raise ValueError("Dataset manifest language ordering differs from the expected contract")
    if manifest.get("language_to_id") != {
        language: index for index, language in enumerate(languages)
    }:
        raise ValueError("Dataset manifest language_to_id is invalid")
    if manifest.get("source_only_languages") != list(source_only):
        raise ValueError("Dataset manifest source-only languages differ from the contract")
    if manifest.get("inputs") != [snapshot.resolved_path for snapshot in source_snapshots]:
        raise ValueError("Dataset manifest input paths differ from the snapshotted sources")
    if manifest.get("tokenizer_model") != tokenizer_snapshot.resolved_path:
        raise ValueError("Dataset manifest tokenizer path differs from the snapshot")
    if manifest.get("train_only_prefixes") != list(train_only_prefixes):
        raise ValueError("Dataset manifest synthetic prefixes differ from the contract")
    if manifest.get("storage_sides") != ["src", "tgt"]:
        raise ValueError("Dataset manifest storage-side contract is invalid")
    if manifest.get("index_dtype") != preprocessing_options["index_dtype"]:
        raise ValueError("Dataset manifest index dtype contradicts preprocessing options")
    if manifest.get("record_metadata") != {
        "format": preprocessing_options["record_metadata_format"],
        "fields": preprocessing_options["record_metadata_fields"],
        "optional": True,
        "index_suffix": RECORD_METADATA_INDEX_SUFFIX,
        "data_suffix": RECORD_METADATA_DATA_SUFFIX,
        "index_dtype": preprocessing_options["record_metadata_index_dtype"],
    }:
        raise ValueError("Dataset manifest record-metadata contract is invalid")
    if manifest.get("synthetic_policy") != {
        "record_field": "synthetic",
        "train_only": True,
        "sampling_weight": preprocessing_options["synthetic_sampling_weight"],
        "prefixes": list(train_only_prefixes),
    }:
        raise ValueError("Dataset manifest synthetic policy contradicts preprocessing options")
    if manifest.get("atomic_build") is not True:
        raise ValueError("Dataset manifest is not marked as an atomic generation")
    direct_option_fields = {
        "approximate_split": "approximate_split",
        "dedup_backend": "dedup_backend",
        "endpoint_leakage_key": "endpoint_leakage_key",
        "shard_size": "shard_size",
        "test_fraction": "test_fraction",
        "validation_fraction": "validation_fraction",
    }
    for manifest_name, option_name in direct_option_fields.items():
        if manifest.get(manifest_name) != preprocessing_options[option_name]:
            raise ValueError(f"Dataset manifest {manifest_name} contradicts preprocessing options")
    if manifest.get("quality_filter_enabled") != preprocessing_options["filter_quality"]:
        raise ValueError("Dataset manifest quality filter contradicts preprocessing options")
    if manifest.get("quality_policy") != preprocessing_options["quality_policy"]:
        raise ValueError("Dataset manifest quality policy contradicts preprocessing options")
    if (
        manifest.get("target_leakage_guard_enabled")
        != preprocessing_options["prevent_target_leakage"]
    ):
        raise ValueError("Dataset manifest leakage guard contradicts preprocessing options")
    if manifest.get("target_leakage_guard") != preprocessing_options["endpoint_leakage_guard"]:
        raise ValueError("Dataset manifest leakage guard schema contradicts preprocessing options")
    if manifest.get("split_key") != preprocessing_options["split_key"]:
        raise ValueError("Dataset manifest split key contradicts preprocessing options")
    stats = _prepare_stats_from_manifest(manifest)
    source_stats = _validate_manifest_sources(
        manifest,
        source_snapshots,
        train_only_prefixes,
        stats,
    )
    expected_mean = stats.quality_score_sum / max(stats.valid_pairs, 1)
    raw_mean = manifest.get("mean_quality_score")
    if (
        isinstance(raw_mean, bool)
        or not isinstance(raw_mean, (int, float))
        or not math.isclose(float(raw_mean), expected_mean, rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise ValueError("Dataset manifest mean quality score is invalid")
    _validate_indexed_payload_semantics(
        staging_dir,
        stats=stats,
        source_stats=source_stats,
        languages=languages,
        normalized_pairs=normalized_pairs,
        translation_directions=translation_directions,
        source_only=source_only,
    )
    validate_dataset_artifact_inventory(staging_dir, manifest)
    completion = _read_json_object(
        staging_dir / PREPARE_COMPLETION_FILENAME,
        role="dataset completion marker",
    )
    expected_completion = _completion_payload(staging_dir, manifest)
    if completion != expected_completion:
        raise ValueError("Dataset staging completion marker does not authenticate its generation")
    verified = tuple(
        _snapshot_file(staging_dir / filename, role=f"dataset staging {filename}")
        for filename in (
            RAW_FINGERPRINT_FILENAME,
            "manifest.json",
            PREPARE_COMPLETION_FILENAME,
        )
    )
    if verified != watched:
        raise RuntimeError("Dataset staging metadata changed while it was authenticated")
    _validate_staging_tree_shape(staging_dir)
    return stats


def _publish_staged_directory(
    staging_dir: Path,
    output_dir: Path,
    *,
    before_rename: Callable[[], None] | None = None,
) -> None:
    if _path_exists(output_dir):
        raise FileExistsError(f"Dataset output appeared while publishing: {output_dir}")
    _fsync_staging_tree(staging_dir)
    _fsync_directory(output_dir.parent)
    if before_rename is not None:
        before_rename()
    if _path_exists(output_dir):
        raise FileExistsError(f"Dataset output appeared while publishing: {output_dir}")
    renamed = False
    try:
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
                    "Dataset publication durability failed and staging rollback also failed"
                ) from rollback_error
            raise
        if _path_exists(output_dir):
            raise FileExistsError(
                f"Dataset output appeared while publishing: {output_dir}"
            ) from error
        raise


def _recover_or_clean_staging(
    output_dir: Path,
    *,
    input_patterns: Sequence[str],
    expected_paths: Sequence[Path],
    source_snapshots: tuple[_FileSnapshot, ...],
    tokenizer_path: Path,
    tokenizer_snapshot: _FileSnapshot,
    expected_fingerprint: DatasetFingerprint,
    normalized_pairs: tuple[tuple[str, str], ...],
    translation_directions: tuple[tuple[str, str], ...],
    languages: tuple[str, ...],
    source_only: tuple[str, ...],
    train_only_prefixes: tuple[str, ...],
    preprocessing_options: Mapping[str, Any],
) -> PrepareStats | None:
    valid: list[tuple[Path, PrepareStats]] = []
    for candidate in _staging_candidates(output_dir):
        try:
            candidate_stat = os.lstat(candidate)
        except OSError as error:
            raise RuntimeError(f"Cannot inspect orphan dataset staging: {candidate}") from error
        if _is_reparse_stat(candidate_stat) or not stat.S_ISDIR(candidate_stat.st_mode):
            raise RuntimeError(
                f"Refusing unsafe orphan staging path (symlink/reparse/non-directory): {candidate}"
            )
        try:
            recovered_stats = _validate_complete_staging(
                candidate,
                expected_fingerprint=expected_fingerprint,
                source_snapshots=source_snapshots,
                tokenizer_snapshot=tokenizer_snapshot,
                normalized_pairs=normalized_pairs,
                translation_directions=translation_directions,
                languages=languages,
                source_only=source_only,
                train_only_prefixes=train_only_prefixes,
                preprocessing_options=preprocessing_options,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            _quarantine_staging(candidate, output_dir)
            continue
        valid.append((candidate, recovered_stats))
    if not valid:
        return None
    selected, recovered_stats = max(valid, key=lambda item: item[0].name)
    for candidate, _ in valid:
        if candidate != selected:
            _quarantine_staging(candidate, output_dir)
    _verify_input_snapshots(
        input_patterns,
        expected_paths,
        source_snapshots,
        tokenizer_path,
        tokenizer_snapshot,
    )
    _validate_complete_staging(
        selected,
        expected_fingerprint=expected_fingerprint,
        source_snapshots=source_snapshots,
        tokenizer_snapshot=tokenizer_snapshot,
        normalized_pairs=normalized_pairs,
        translation_directions=translation_directions,
        languages=languages,
        source_only=source_only,
        train_only_prefixes=train_only_prefixes,
        preprocessing_options=preprocessing_options,
    )
    _publish_staged_directory(
        selected,
        output_dir,
        before_rename=lambda: _verify_input_snapshots(
            input_patterns,
            expected_paths,
            source_snapshots,
            tokenizer_path,
            tokenizer_snapshot,
        ),
    )
    return recovered_stats


def prepare_dataset(
    input_patterns: Sequence[str],
    tokenizer_model: str | Path,
    output_dir: str | Path,
    *,
    shard_size: int = 100_000,
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    max_tokens_per_side: int = 510,
    quality_policy: QualityPolicy | None = None,
    filter_quality: bool = True,
    prevent_target_leakage: bool = True,
    approximate_split: bool = False,
    dedup_backend: str = "sqlite",
    language_pair: Sequence[str] = ("ko", "ja"),
    source_only_languages: Sequence[str] = (),
    language_pairs: Sequence[Sequence[str]] | None = None,
    translation_directions: Sequence[Sequence[str]] | None = None,
    train_only_prefixes: Sequence[str] = DEFAULT_TRAIN_ONLY_PREFIXES,
    managed_augmentation_prefix: str | None = None,
    synthetic_sampling_weight: float = DEFAULT_SYNTHETIC_SAMPLING_WEIGHT,
    num_workers: int | None = None,
    expected_fingerprint: DatasetFingerprint | None = None,
) -> PrepareStats:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Validation and test fractions must be non-negative")
    if validation_fraction + test_fraction >= 0.5:
        raise ValueError("Validation and test fractions are unexpectedly large")
    if max_tokens_per_side < 1:
        raise ValueError("max_tokens_per_side must be positive")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if dedup_backend not in {"sqlite", "memory"}:
        raise ValueError("dedup_backend must be either 'sqlite' or 'memory'")
    if not 0.0 <= synthetic_sampling_weight <= 1.0:
        raise ValueError("synthetic_sampling_weight must be in [0, 1]")

    quality_policy = quality_policy or QualityPolicy()
    quality_policy.validate()
    normalized_pairs = normalize_language_pairs(language_pair, language_pairs)
    if not normalized_pairs:
        raise ValueError("at least one language pair is required")
    primary_pair = next(iter(normalized_pairs))
    train_only_prefixes = normalize_synthetic_prefixes(train_only_prefixes)
    if managed_augmentation_prefix is not None:
        managed_augmentation_prefix = str(managed_augmentation_prefix)
        if managed_augmentation_prefix not in train_only_prefixes:
            raise ValueError(
                "managed_augmentation_prefix must also be a train-only synthetic prefix"
            )
    languages = languages_from_pairs(normalized_pairs)
    language_to_id = {language: index for index, language in enumerate(languages)}
    source_only = tuple(dict.fromkeys(str(language) for language in source_only_languages))
    unknown_source_only = sorted(set(source_only) - set(languages))
    if unknown_source_only:
        raise ValueError(
            "source_only_languages must appear in the configured language pairs; "
            f"{unknown_source_only} do not"
        )
    for pair in normalized_pairs:
        if pair[0] in source_only and pair[1] in source_only:
            raise ValueError(
                "at most one side of a language pair may be source-only; both sides "
                f"of {list(pair)!r} are source-only"
            )
    normalized_directions = normalize_translation_directions(
        normalized_pairs,
        translation_directions,
        source_only_languages=source_only,
    )
    direction_set = frozenset(normalized_directions)

    preprocessing_options = prepare_preprocessing_options(
        shard_size=shard_size,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        max_tokens_per_side=max_tokens_per_side,
        quality_policy=quality_policy,
        filter_quality=filter_quality,
        prevent_target_leakage=prevent_target_leakage,
        approximate_split=approximate_split,
        dedup_backend=dedup_backend,
        source_only_languages=source_only,
        translation_directions=normalized_directions,
        train_only_prefixes=train_only_prefixes,
        managed_augmentation_prefix=managed_augmentation_prefix,
        synthetic_sampling_weight=synthetic_sampling_weight,
        language_pair_count=len(normalized_pairs),
    )
    endpoint_key_schema = cast(str, preprocessing_options["endpoint_leakage_key"])
    split_key_schema = cast(str, preprocessing_options["split_key"])
    paths, source_snapshots = _capture_input_snapshots(input_patterns)
    if managed_augmentation_prefix is not None:
        # Imported lazily because augmentation accounting reuses this module's
        # preprocessing contract. Official train/prepare callers hold the raw
        # directory lease while this validates and crash-recovers the ledger.
        from sion_translate.augmentation import load_augmentation_registry

        for parent in sorted({path.parent for path in paths}, key=str):
            load_augmentation_registry(parent, managed_augmentation_prefix, ())
    if len(paths) > np.iinfo(np.uint16).max:
        raise ValueError("Too many input files for the uint16 source_id field")
    tokenizer_path = _absolute_path(Path(tokenizer_model))
    tokenizer_snapshot = _snapshot_file(tokenizer_path, role="tokenizer model")
    observed_fingerprint = _fingerprint_from_snapshots(
        source_snapshots,
        tokenizer_snapshot,
        language_pairs=normalized_pairs,
        preprocessing_options=preprocessing_options,
    )
    if expected_fingerprint is not None and expected_fingerprint != observed_fingerprint:
        raise RuntimeError(
            "Caller dataset fingerprint differs from the exact preparation input contract"
        )
    dataset_fingerprint = expected_fingerprint or observed_fingerprint

    # Validate the model and all required language tags before a large worker pool.
    tokenizer = SionTokenizer(tokenizer_path)
    missing_languages = sorted(set(languages) - set(tokenizer.languages))
    if missing_languages:
        raise ValueError(
            "Tokenizer is missing configured language tags: "
            f"{missing_languages}; retrain it with language_pairs={normalized_pairs!r}"
        )

    output_dir = _absolute_path(Path(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _assert_regular_directory(output_dir.parent, role="dataset output parent")
    _refuse_or_remove_empty_output(output_dir)
    recovered = _recover_or_clean_staging(
        output_dir,
        input_patterns=input_patterns,
        expected_paths=paths,
        source_snapshots=source_snapshots,
        tokenizer_path=tokenizer_path,
        tokenizer_snapshot=tokenizer_snapshot,
        expected_fingerprint=dataset_fingerprint,
        normalized_pairs=normalized_pairs,
        translation_directions=normalized_directions,
        languages=languages,
        source_only=source_only,
        train_only_prefixes=train_only_prefixes,
        preprocessing_options=preprocessing_options,
    )
    if recovered is not None:
        return recovered
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging-{uuid.uuid4().hex}")
    staging_dir.mkdir(exist_ok=False)
    try:
        writers = {
            split: ShardWriter(staging_dir, split, shard_size, language_to_id)
            for split in ("train", "validation", "test")
        }
        digest_store = (
            _SqliteDigestSet(staging_dir / ".dedup.sqlite3")
            if dedup_backend == "sqlite"
            else _MemoryDigestSet()
        )
    except BaseException:
        _discard_private_staging(staging_dir, output_dir)
        raise
    stats = PrepareStats()
    per_source_stats = [PrepareStats() for _ in paths]
    estimated_pairs = max(1, sum(snapshot.size for snapshot in source_snapshots) // 200)
    target_split_guard = (
        TargetSplitGuard(estimated_pairs, validation_fraction, test_fraction)
        if prevent_target_leakage
        else None
    )

    workers = num_workers or build_cpu_plan(input_files=len(paths)).dataset_workers
    inputs = _prepare_input_batches(
        paths,
        quality_policy,
        filter_quality,
        normalized_pairs,
        train_only_prefixes,
    )
    try:
        if workers <= 1:
            _initialize_prepare_worker(str(tokenizer_path))
            processed_batches = map(_process_prepare_batch, inputs)
            executor = None
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_prepare_worker,
                initargs=(str(tokenizer_path),),
            )
            processed_batches = bounded_ordered_map(
                executor, _process_prepare_batch, inputs, max_pending=workers * 2
            )
    except BaseException:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        try:
            digest_store.close()
        except Exception:
            pass
        _discard_private_staging(staging_dir, output_dir)
        raise

    try:
        for batch in processed_batches:
            for status, payload in batch:
                source_id = int(payload[0])
                source_stats = per_source_stats[source_id]
                targets = (stats, source_stats)
                if status == "physical_line":
                    for target in targets:
                        _increment(target, "physical_lines")
                    continue

                if status in {
                    "invalid_utf8",
                    "invalid_json",
                    "invalid_record",
                    "non_string",
                    "missing_text",
                    "invalid_language",
                    "unaligned_lists",
                }:
                    for target in targets:
                        _increment(target, status)
                    continue

                if status == "quality_filtered":
                    _, rejection_reasons, warning_reasons = payload
                    _record_quality_reasons(targets, rejection_reasons)
                    if "structured_span_mismatch" in warning_reasons:
                        for target in targets:
                            _increment(target, "structured_span_warnings")
                    if "ja_no_kana" in warning_reasons:
                        for target in targets:
                            _increment(target, "ja_no_kana_warnings")
                    for target in targets:
                        _increment(target, "quality_filtered")
                    continue

                (
                    _,
                    record_group_key,
                    record_is_synthetic,
                    language_a,
                    language_b,
                    metadata,
                    text_a,
                    text_b,
                    ids_a,
                    ids_b,
                    register_a,
                    register_b,
                    quality_score,
                    rejection_reasons,
                    warning_reasons,
                ) = payload
                dedup_language_a, dedup_language_b = language_a, language_b
                dedup_text_a, dedup_text_b = text_a, text_b
                _record_quality_reasons(targets, rejection_reasons)
                if "structured_span_mismatch" in warning_reasons:
                    for target in targets:
                        _increment(target, "structured_span_warnings")
                if "ja_no_kana" in warning_reasons:
                    for target in targets:
                        _increment(target, "ja_no_kana_warnings")

                # Side A is the physical direction exposed as virtual direction
                # zero. A row-scoped edge is used by backtranslation so the
                # synthetic source -> real target example is never virtualized
                # into a pseudo-label reverse example, even when the global pair
                # is bidirectional.
                row_direction = resolve_record_training_direction(
                    metadata,
                    (language_a, language_b),
                    direction_set,
                )
                if row_direction is not None:
                    metadata = dict(metadata)
                    metadata["training_direction"] = list(row_direction)
                storage_direction = row_direction or (
                    (language_a, language_b)
                    if (language_a, language_b) in direction_set
                    else (language_b, language_a)
                )
                if (language_a, language_b) != storage_direction:
                    language_a, language_b = language_b, language_a
                    text_a, text_b = text_b, text_a
                    ids_a, ids_b = ids_b, ids_a
                    register_a, register_b = register_b, register_a
                forward_only = (
                    row_direction is not None
                    or (
                        language_b,
                        language_a,
                    )
                    not in direction_set
                )

                # A row that can never be written must not reserve endpoint
                # ownership and suppress a usable row encountered later.
                if len(ids_a) > max_tokens_per_side or len(ids_b) > max_tokens_per_side:
                    for target in targets:
                        _increment(target, "too_long")
                    continue

                # Approve the row's partition before it can affect pair dedup.
                # Both language-scoped endpoints participate so a surface
                # cannot leak when it changes role across configured pairs.
                is_synthetic = record_is_synthetic or synthetic_path(
                    paths[source_id], train_only_prefixes
                )
                if is_synthetic:
                    split = "train"
                elif len(normalized_pairs) > 1:
                    split_key = f"record\0{record_group_key}"
                    split = choose_split_for_key(
                        split_key,
                        validation_fraction,
                        test_fraction,
                    )
                else:
                    split_key = endpoint_split_key(
                        language_a,
                        text_a,
                        approximate=approximate_split,
                    )
                    split = choose_split_for_key(
                        split_key,
                        validation_fraction,
                        test_fraction,
                    )
                if target_split_guard is not None:
                    endpoint_digests = (
                        endpoint_split_digest(
                            language_a,
                            text_a,
                            approximate=approximate_split,
                        ),
                        endpoint_split_digest(
                            language_b,
                            text_b,
                            approximate=approximate_split,
                        ),
                    )
                    if not target_split_guard.accept_many(split, endpoint_digests):
                        for target in targets:
                            _increment(target, "split_conflicts")
                        continue

                supervised_directions = [(language_a, language_b)]
                if not forward_only:
                    supervised_directions.append((language_b, language_a))
                claimed_directions: list[tuple[str, str]] = []
                for supervised_direction in supervised_directions:
                    direction_key = (
                        f"{dedup_language_a}\0{dedup_key(dedup_text_a)}\0"
                        f"{dedup_language_b}\0{dedup_key(dedup_text_b)}\0"
                        f"{supervised_direction[0]}\0{supervised_direction[1]}"
                    ).encode("utf-8")
                    direction_digest = hashlib.sha256(direction_key).digest()[:16]
                    if digest_store.add_if_new(direction_digest):
                        claimed_directions.append(supervised_direction)
                if not claimed_directions:
                    for target in targets:
                        _increment(target, "duplicates")
                    continue
                if len(claimed_directions) == 1 and len(supervised_directions) == 2:
                    claimed_direction = claimed_directions[0]
                    if (language_a, language_b) != claimed_direction:
                        language_a, language_b = language_b, language_a
                        text_a, text_b = text_b, text_a
                        ids_a, ids_b = ids_b, ids_a
                        register_a, register_b = register_b, register_a
                    forward_only = True
                    metadata = dict(metadata)
                    metadata["training_direction"] = list(claimed_direction)

                writers[split].add(
                    ids_a,
                    ids_b,
                    register_a,
                    register_b,
                    language_a,
                    language_b,
                    source_id,
                    quality_score,
                    is_synthetic,
                    forward_only,
                    metadata,
                )
                for target in targets:
                    _increment(target, split)
                    _increment(target, "valid_pairs")
                    _increment(target, "ko_tokens", len(ids_a))
                    _increment(target, "ja_tokens", len(ids_b))
                    _increment(target, "quality_score_sum", quality_score)
                    if is_synthetic:
                        _increment(target, "synthetic_pairs")
                    if forward_only:
                        _increment(target, "forward_only_pairs")
    except BaseException:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        try:
            digest_store.close()
        except Exception:
            pass
        _discard_private_staging(staging_dir, output_dir)
        raise
    else:
        if executor is not None:
            executor.shutdown()

    try:
        # This is the first complete post-worker boundary. Hashing again, rather
        # than trusting size/mtime, detects in-place same-length rewrites.
        _verify_input_snapshots(
            input_patterns,
            paths,
            source_snapshots,
            tokenizer_path,
            tokenizer_snapshot,
        )
        for writer in writers.values():
            writer.close()
        digest_store.close()
        (staging_dir / ".dedup.sqlite3").unlink(missing_ok=True)
    except BaseException:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        try:
            digest_store.close()
        except Exception:
            pass
        _discard_private_staging(staging_dir, output_dir)
        raise

    try:
        # The fingerprint belongs to this generation, so it must enter staging
        # before inventory construction and publication rather than being added
        # later by the caller in a second transaction.
        _write_json_durable(
            staging_dir / RAW_FINGERPRINT_FILENAME,
            dataset_fingerprint.to_dict(),
        )
        artifact_inventory = build_dataset_artifact_inventory(staging_dir)
        manifest = {
            "format": INDEX_FORMAT,
            "language_pair": list(primary_pair),
            "language_pairs": [list(pair) for pair in normalized_pairs],
            "translation_directions": [list(direction) for direction in normalized_directions],
            "languages": list(languages),
            "language_to_id": language_to_id,
            "source_only_languages": list(source_only),
            "storage_sides": ["src", "tgt"],
            "record_metadata": {
                "format": RECORD_METADATA_FORMAT,
                "fields": list(RECORD_METADATA_FIELDS),
                "optional": True,
                "index_suffix": RECORD_METADATA_INDEX_SUFFIX,
                "data_suffix": RECORD_METADATA_DATA_SUFFIX,
                "index_dtype": RECORD_METADATA_INDEX_DTYPE.descr,
            },
            "train_only_prefixes": list(train_only_prefixes),
            "synthetic_policy": {
                "record_field": "synthetic",
                "train_only": True,
                "sampling_weight": synthetic_sampling_weight,
                "prefixes": list(train_only_prefixes),
            },
            "tokenizer_model": tokenizer_snapshot.resolved_path,
            "fingerprint": dataset_fingerprint.to_dict(),
            "preprocessing_schema": PREPROCESSING_SCHEMA,
            "preprocessing_options": preprocessing_options,
            "inputs": [snapshot.resolved_path for snapshot in source_snapshots],
            "sources": [
                {
                    "id": source_id,
                    "name": path.name,
                    "path": source_snapshots[source_id].resolved_path,
                    "synthetic_file": synthetic_path(path, train_only_prefixes),
                    "stats": asdict(per_source_stats[source_id]),
                    "mean_quality_score": (
                        per_source_stats[source_id].quality_score_sum
                        / max(per_source_stats[source_id].valid_pairs, 1)
                    ),
                }
                for source_id, path in enumerate(paths)
            ],
            "index_dtype": INDEX_DTYPE.descr,
            "stats": asdict(stats),
            "mean_quality_score": stats.quality_score_sum / max(stats.valid_pairs, 1),
            "shard_size": shard_size,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "quality_filter_enabled": filter_quality,
            "quality_policy": quality_policy.to_dict(),
            "target_leakage_guard_enabled": prevent_target_leakage,
            "target_leakage_guard": "language-endpoint-bloom-v2",
            "endpoint_leakage_key": endpoint_key_schema,
            "approximate_split": approximate_split,
            "split_key": split_key_schema,
            "dedup_backend": dedup_backend,
            "atomic_build": True,
            "artifact_inventory": artifact_inventory,
        }
        _write_json_durable(staging_dir / "manifest.json", manifest)
        _write_json_durable(
            staging_dir / PREPARE_COMPLETION_FILENAME,
            _completion_payload(staging_dir, manifest),
        )
        _validate_complete_staging(
            staging_dir,
            expected_fingerprint=dataset_fingerprint,
            source_snapshots=source_snapshots,
            tokenizer_snapshot=tokenizer_snapshot,
            normalized_pairs=normalized_pairs,
            translation_directions=normalized_directions,
            languages=languages,
            source_only=source_only,
            train_only_prefixes=train_only_prefixes,
            preprocessing_options=preprocessing_options,
        )
        # Last gate before rename. This catches a newly discovered wildcard
        # source as well as content replacement after worker completion.
        _verify_input_snapshots(
            input_patterns,
            paths,
            source_snapshots,
            tokenizer_path,
            tokenizer_snapshot,
        )
        _publish_staged_directory(
            staging_dir,
            output_dir,
            before_rename=lambda: _verify_input_snapshots(
                input_patterns,
                paths,
                source_snapshots,
                tokenizer_path,
                tokenizer_snapshot,
            ),
        )
    except BaseException:
        _discard_private_staging(staging_dir, output_dir)
        raise
    return stats
