from __future__ import annotations

import errno
import gzip
import glob
import hashlib
import hmac
from importlib.metadata import version as package_version
import json
import math
import multiprocessing
import os
import re
import shutil
import sqlite3
import stat
import sys
import unicodedata
import uuid
import warnings
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, BinaryIO, Sequence, TypeAlias, cast

import numpy as np

from sion_translate.fingerprint import (
    PREPROCESSING_SCHEMA,
    DatasetFingerprint,
    FileFingerprint,
    file_sha256,
)
from sion_translate.language_tags import canonicalize_language_tags, parse_language_tag
from sion_translate.locking import _exclusive_lock  # pyright: ignore[reportPrivateUsage]
from sion_translate.performance import bounded_ordered_map, build_cpu_plan
from sion_translate.revision import DRAFT_SEPARATOR
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
PREPARE_PROGRESS_SCHEMA = "sion-prepare-worker-progress-v2"
PREPARE_PROGRESS_CONTRACT_FILENAME = "contract.json"
PREPARE_PROGRESS_EPOCH_FILENAME = "generation.json"
PREPARE_PROGRESS_EPOCH_SCHEMA = "sion-prepare-worker-generation-v1"
PREPARE_PROGRESS_WRITER_LOCK_FILENAME = "writer.lock"
PREPARE_PROGRESS_CHUNK_SCHEMA = "sion-prepare-worker-chunk-v2"
PREPARE_WORKER_ALGORITHM_SCHEMA = "sion-prepare-worker-events-v3"
PREPARE_BATCH_SIZE = 512
PREPARE_OUTPUT_LOCK_SCHEMA = "sion-prepare-output-lock-v1"
PREPARE_STATS_SCHEMA_V1 = "sion-prepare-stats-src-tgt-v1"
PREPARE_STATS_SCHEMA_V2 = "sion-prepare-stats-src-tgt-v2"
PREPARE_STATS_SCHEMA = "sion-prepare-stats-src-tgt-v3"
DATASET_SPLITS = ("train", "validation", "test", "refinement_evidence")

# These limits bound every supported language graph without naming any
# language. They turn adversarial nested records into an early, reproducible
# contract error instead of letting one JSON line exhaust worker memory or the
# output filesystem. Ordinary corpora remain far below all four limits.
PREPARE_MAX_RAW_LINE_BYTES = 16 * 1024 * 1024
PREPARE_MAX_BATCH_RAW_BYTES = 64 * 1024 * 1024
PREPARE_MAX_EXPANDED_PAIRS_PER_LINE = 1_024
PREPARE_MAX_RECORD_METADATA_BYTES = 256 * 1024
PREPARE_MAX_CHUNK_UNCOMPRESSED_BYTES = 256 * 1024 * 1024

# The v1.5 package used to calibrate this preflight contained 17,319,801,659
# bytes of training JSONL and 1,757,541,972 bytes of prepared shards (a 0.102
# final/input ratio); deflating the same JSONL used a 0.331 ratio. The larger
# 0.85 and 0.55 estimates below leave substantial room for multilingual rows,
# token-ID JSON, and per-row metadata. They are capacity estimates rather than
# mathematical upper bounds, so the fixed reserve and per-chunk/shard checks
# still stop a pathological corpus before it consumes cleanup space.
_PREPARE_FINAL_BYTES_PER_INPUT_BYTE = 0.85
_PREPARE_CACHE_BYTES_PER_INPUT_BYTE = 0.55
_PREPARE_SPACE_RESERVE_BYTES = 512 * 1024 * 1024
_PREPARE_FIXED_FINAL_OVERHEAD_BYTES = 32 * 1024 * 1024
_PREPARE_PER_SHARD_OVERHEAD_BYTES = 128 * 1024
_PREPARE_SQLITE_BYTES_PER_CANDIDATE = 192

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
SHARED_TARGET_INDEX_DTYPE = np.dtype([*INDEX_DTYPE.descr, ("target_shared", "u1")])


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
    reserved_draft_separator: int = 0
    excessive_repetition: int = 0
    structured_span_rejections: int = 0
    structured_span_warnings: int = 0
    ja_no_kana_warnings: int = 0
    split_conflicts: int = 0
    too_long: int = 0
    train: int = 0
    validation: int = 0
    test: int = 0
    refinement_evidence: int = 0
    src_tokens: int = 0
    tgt_tokens: int = 0
    quality_score_sum: int = 0
    synthetic_pairs: int = 0
    forward_only_pairs: int = 0


# This tuple is the deployed v1 wire contract. Do not append a new in-memory
# counter here: define a new statistics schema and an explicit migration map.
# The import-time parity check below turns an accidental PrepareStats change
# into a developer-visible error instead of silently changing persisted v1
# manifests or making authenticated older artifacts unreadable.
_PREPARE_STATS_V1_FIELDS = (
    "physical_lines",
    "valid_pairs",
    "invalid_json",
    "invalid_utf8",
    "invalid_record",
    "missing_text",
    "non_string",
    "invalid_language",
    "unaligned_lists",
    "duplicates",
    "quality_filtered",
    "too_short",
    "identical_text",
    "length_ratio_outlier",
    "language_mismatch",
    "control_characters",
    "excessive_repetition",
    "structured_span_rejections",
    "structured_span_warnings",
    "ja_no_kana_warnings",
    "split_conflicts",
    "too_long",
    "train",
    "validation",
    "test",
    "src_tokens",
    "tgt_tokens",
    "quality_score_sum",
    "synthetic_pairs",
    "forward_only_pairs",
)
_PREPARE_STATS_V2_FIELDS = (
    *_PREPARE_STATS_V1_FIELDS[:25],
    "refinement_evidence",
    *_PREPARE_STATS_V1_FIELDS[25:],
)
_PREPARE_STATS_V3_FIELDS = (
    "physical_lines",
    "valid_pairs",
    "invalid_json",
    "invalid_utf8",
    "invalid_record",
    "missing_text",
    "non_string",
    "invalid_language",
    "unaligned_lists",
    "duplicates",
    "quality_filtered",
    "too_short",
    "identical_text",
    "length_ratio_outlier",
    "language_mismatch",
    "control_characters",
    "reserved_draft_separator",
    "excessive_repetition",
    "structured_span_rejections",
    "structured_span_warnings",
    "ja_no_kana_warnings",
    "split_conflicts",
    "too_long",
    "train",
    "validation",
    "test",
    "refinement_evidence",
    "src_tokens",
    "tgt_tokens",
    "quality_score_sum",
    "synthetic_pairs",
    "forward_only_pairs",
)
if tuple(field.name for field in fields(PrepareStats)) != _PREPARE_STATS_V3_FIELDS:
    raise RuntimeError("PrepareStats changed without a new explicit persisted statistics schema")
_PREPARE_STATS_V1_FIELD_MAP = tuple((name, name) for name in _PREPARE_STATS_V1_FIELDS)
_PREPARE_STATS_V2_FIELD_MAP = tuple((name, name) for name in _PREPARE_STATS_V2_FIELDS)
_PREPARE_STATS_V3_FIELD_MAP = tuple((name, name) for name in _PREPARE_STATS_V3_FIELDS)
_PREPARE_STATS_LEGACY_FIELD_MAP = tuple(
    (
        {"src_tokens": "ko_tokens", "tgt_tokens": "ja_tokens"}.get(name, name),
        name,
    )
    for name in _PREPARE_STATS_V1_FIELDS
)


def prepare_stats_schema_from_manifest(
    manifest: Mapping[str, object],
    *,
    role: str,
) -> str | None:
    """Return the manifest-wide storage-side statistics schema.

    Manifests written before this schema marker used ``ko_tokens`` and
    ``ja_tokens`` for the physical source and target sides even when the
    configured graph contained other languages. A missing marker is therefore
    the only accepted legacy representation. An explicit null or an unknown
    marker is rejected instead of being guessed.
    """

    if "stats_schema" not in manifest:
        return None
    value = manifest.get("stats_schema")
    if not isinstance(value, str) or value not in {
        PREPARE_STATS_SCHEMA_V1,
        PREPARE_STATS_SCHEMA_V2,
        PREPARE_STATS_SCHEMA,
    }:
        raise ValueError(f"{role} stats_schema is unsupported")
    return value


def validated_prepare_stats(
    value: object,
    *,
    stats_schema: str | None,
    role: str,
) -> PrepareStats:
    """Validate and normalize one total or per-source statistics object."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{role} stats must be an object")
    raw = cast(Mapping[object, object], value)
    if stats_schema is None:
        field_map = _PREPARE_STATS_LEGACY_FIELD_MAP
    elif stats_schema == PREPARE_STATS_SCHEMA_V1:
        field_map = _PREPARE_STATS_V1_FIELD_MAP
    elif stats_schema == PREPARE_STATS_SCHEMA_V2:
        field_map = _PREPARE_STATS_V2_FIELD_MAP
    elif stats_schema == PREPARE_STATS_SCHEMA:
        field_map = _PREPARE_STATS_V3_FIELD_MAP
    else:
        raise ValueError(f"{role} stats_schema is unsupported")
    expected_fields = {serialized_name for serialized_name, _ in field_map}
    if set(raw) != expected_fields:
        raise ValueError(f"{role} stats fields do not match their schema")

    normalized: dict[str, int] = {}
    for serialized_name, internal_name in field_map:
        field_value = raw[serialized_name]
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 0:
            raise ValueError(f"{role} stat is invalid: {serialized_name}")
        normalized[internal_name] = field_value
    return PrepareStats(**normalized)


def infer_register(text: str, language: str) -> int:
    """Infer a coarse politeness register from sentence-final expressions.

    Rules currently cover Korean and Japanese. Every other language returns
    zero, which means unknown rather than neutral.
    """
    primary_language = parse_language_tag(language, field="register language").language
    stripped = text.rstrip(" \t\r\n.!?。！？\"'“”‘’")
    if primary_language == "ko":
        if re.search(r"(옵니다|사옵니다|드리겠습니다|주시기 바랍니다|하십시오)$", stripped):
            return 3
        if re.search(r"(습니다|ㅂ니다|입니다|합니다|됩니다|세요|어요|아요|예요|이에요)$", stripped):
            return 2
        if re.search(r"(다|한다|했다|이다|한다면|하자|해)$", stripped):
            return 1
    elif primary_language == "ja":
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
        *,
        shared_targets: bool = False,
        maximum_shard_bytes: int | None = None,
    ):
        self.root = root / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = shard_size
        self.language_to_id = language_to_id
        self.shared_targets = shared_targets
        self.maximum_shard_bytes = maximum_shard_bytes
        self.index_dtype = SHARED_TARGET_INDEX_DTYPE if shared_targets else INDEX_DTYPE
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
        if self.maximum_shard_bytes is not None:
            _ensure_prepare_write_reserve(
                self.root,
                self.maximum_shard_bytes,
                role=f"next {self.split} dataset shard",
            )
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
        shared_target: bool = False,
    ) -> None:
        assert self._src_handle is not None and self._tgt_handle is not None
        metadata_payload = encode_record_metadata(metadata)
        src_array = np.asarray(src_ids, dtype=np.uint32)
        tgt_array = np.asarray(tgt_ids, dtype=np.uint32)
        if shared_target:
            if not self.shared_targets:
                raise ValueError("shared targets require a shared-target shard writer")
            if (
                src_language != tgt_language
                or src_register != tgt_register
                or not np.array_equal(src_array, tgt_array)
            ):
                raise ValueError("a shared target must be identical to its source contract")
        src_array.tofile(self._src_handle)
        if not shared_target:
            tgt_array.tofile(self._tgt_handle)
        record = (
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
        self.records.append((*record, int(shared_target)) if self.shared_targets else record)
        self.record_metadata.append(metadata_payload)
        self.src_offset += len(src_array)
        if not shared_target:
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
                np.asarray(self.records, dtype=self.index_dtype),
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
    "reserved_draft_separator": "reserved_draft_separator",
    "excessive_repetition": "excessive_repetition",
    "structured_span_mismatch": "structured_span_rejections",
}


_PrepareBatchInput: TypeAlias = tuple[
    int,
    list[tuple[int, bytes]],
    QualityPolicy,
    bool,
    tuple[tuple[str, str], ...],
    int,
    str,
    tuple[tuple[str, str], ...],
]
_PrepareEvent: TypeAlias = tuple[str, tuple[Any, ...]]


@dataclass(frozen=True)
class _PrepareBatchDescriptor:
    """Content-bound coordinates for one deterministic worker batch."""

    synthetic_pass: bool
    source_id: int
    batch_index: int
    start_offset: int
    end_offset: int
    row_count: int
    raw_bytes: int
    raw_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _PrepareBatchJob:
    descriptor: _PrepareBatchDescriptor
    batch: _PrepareBatchInput
    progress_chunks_dir: str
    progress_contract_sha256: str
    generation_epoch: int


@dataclass(frozen=True)
class _PrepareProgress:
    root: Path
    chunks_dir: Path
    contract_sha256: str
    generation_epoch: int = 0


@dataclass
class _PrepareCapacitySummary:
    """Exact pre-dedup payload totals emitted by completed worker chunks."""

    candidate_rows: int = 0
    token_ids: int = 0
    metadata_bytes: int = 0

    def add_events(self, events: Sequence[_PrepareEvent]) -> None:
        for status, payload in events:
            if status != "candidate":
                continue
            metadata = cast(dict[str, object], payload[5])
            ids_a = cast(list[int], payload[8])
            ids_b = cast(list[int], payload[9])
            self.candidate_rows += 1
            self.token_ids += len(ids_a) + len(ids_b)
            self.metadata_bytes += len(encode_record_metadata(metadata))


class _PrepareProgressError(ValueError):
    """Raised when persisted worker progress cannot be trusted."""


_PREPARE_WORKER_TOKENIZER: SionTokenizer | None = None


def _initialize_prepare_worker(tokenizer_model: str) -> None:
    global _PREPARE_WORKER_TOKENIZER
    _PREPARE_WORKER_TOKENIZER = SionTokenizer(  # pyright: ignore[reportConstantRedefinition]
        tokenizer_model
    )


def _pair_has_unauthenticated_draft_separator(
    text_a: str,
    text_b: str,
    *,
    language_pair: tuple[str, str],
    metadata: Mapping[str, object],
    source_name: str,
    translation_directions: tuple[tuple[str, str], ...],
) -> bool:
    """Reject reserved revision syntax unless the exact source edge authenticates it."""

    if DRAFT_SEPARATOR not in text_a and DRAFT_SEPARATOR not in text_b:
        return False

    try:
        direction = resolve_record_training_direction(
            metadata,
            language_pair,
            frozenset(translation_directions),
        )
    except ValueError:
        return True
    if direction is None:
        return True

    if direction == language_pair:
        source_text, target_text = text_a, text_b
    elif direction == (language_pair[1], language_pair[0]):
        source_text, target_text = text_b, text_a
    else:
        return True

    provenance = metadata.get("provenance")
    transformation = (
        cast(Mapping[object, object], provenance).get("transformation")
        if isinstance(provenance, Mapping)
        else None
    )
    filename_marked = source_name.startswith("revise_")
    provenance_marked = transformation == "revision"
    if filename_marked and transformation is not None and not provenance_marked:
        return True
    if not (filename_marked or provenance_marked):
        return True

    # A decoder target can never carry model-control syntax. A revision source
    # carries exactly one separator between two non-empty text segments.
    if DRAFT_SEPARATOR in target_text or source_text.count(DRAFT_SEPARATOR) != 1:
        return True
    original_source, _, draft = source_text.partition(DRAFT_SEPARATOR)
    return not original_source.strip() or not draft.strip()


def _process_prepare_batch(args: _PrepareBatchInput) -> list[_PrepareEvent]:
    """CPU-heavy, order-preserving row work executed in worker processes."""

    (
        source_id,
        rows,
        quality_policy,
        filter_quality,
        language_pairs,
        max_tokens_per_side,
        source_name,
        translation_directions,
    ) = args
    tokenizer = _PREPARE_WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("prepare worker tokenizer was not initialized")
    output: list[_PrepareEvent] = []
    candidate_event_bytes = 0
    for source_offset, raw_line in rows:
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
        if len(expansion.pairs) > PREPARE_MAX_EXPANDED_PAIRS_PER_LINE:
            raise ValueError(
                "A physical JSONL record expands beyond the configured safety limit: "
                f"source_id={source_id}, byte_offset={source_offset}, "
                f"expanded_pairs={len(expansion.pairs):,}, "
                f"limit={PREPARE_MAX_EXPANDED_PAIRS_PER_LINE:,}"
            )
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
            if _pair_has_unauthenticated_draft_separator(
                text_a,
                text_b,
                language_pair=language_pair,
                metadata=pair.metadata,
                source_name=source_name,
                translation_directions=translation_directions,
            ):
                assessment = replace(
                    assessment,
                    accepted=False,
                    rejection_reasons=tuple(
                        dict.fromkeys((*assessment.rejection_reasons, "reserved_draft_separator"))
                    ),
                )
            unsafe_reasons = {
                "control_characters",
                "reserved_draft_separator",
                "structured_span_mismatch",
            }
            unsafe = bool(unsafe_reasons.intersection(assessment.rejection_reasons))
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
            metadata_bytes = encode_record_metadata(pair.metadata)
            if len(metadata_bytes) > PREPARE_MAX_RECORD_METADATA_BYTES:
                raise ValueError(
                    "A physical JSONL record contains supported metadata beyond the "
                    "configured safety limit: "
                    f"source_id={source_id}, byte_offset={source_offset}, "
                    f"metadata_bytes={len(metadata_bytes):,}, "
                    f"limit={PREPARE_MAX_RECORD_METADATA_BYTES:,}"
                )
            encoded_a, encoded_b = protect_shared_spans(text_a, text_b)
            ids_a = tokenizer.encode(encoded_a)
            ids_b = tokenizer.encode(encoded_b)
            if len(ids_a) > max_tokens_per_side or len(ids_b) > max_tokens_per_side:
                output.append(
                    (
                        "too_long",
                        (
                            source_id,
                            assessment.rejection_reasons,
                            assessment.warning_reasons,
                        ),
                    )
                )
                continue
            candidate: _PrepareEvent = (
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
                    ids_a,
                    ids_b,
                    infer_register(text_a, pair.language_a),
                    infer_register(text_b, pair.language_b),
                    assessment.score,
                    assessment.rejection_reasons,
                    assessment.warning_reasons,
                ),
            )
            candidate_event_bytes += len(
                _prepare_event_json([candidate[0], list(candidate[1])]).encode("utf-8")
            )
            if candidate_event_bytes > PREPARE_MAX_CHUNK_UNCOMPRESSED_BYTES - 1024 * 1024:
                raise ValueError(
                    "A prepare batch expands beyond the deterministic worker-chunk limit: "
                    f"source_id={source_id}, byte_offset={source_offset}, "
                    f"limit={PREPARE_MAX_CHUNK_UNCOMPRESSED_BYTES:,} bytes"
                )
            output.append(candidate)
    return output


def _prepare_batch_records(
    paths: Sequence[Path],
    quality_policy: QualityPolicy,
    filter_quality: bool,
    language_pairs: tuple[tuple[str, str], ...],
    train_only_prefixes: tuple[str, ...],
    max_tokens_per_side: int,
    translation_directions: tuple[tuple[str, str], ...],
    batch_size: int | None = None,
) -> Iterator[tuple[_PrepareBatchDescriptor, _PrepareBatchInput]]:
    if batch_size is None:
        batch_size = PREPARE_BATCH_SIZE
    if batch_size < 1:
        raise ValueError("prepare batch size must be positive")
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
                rows: list[tuple[int, bytes]] = []
                batch_index = 0
                start_offset: int | None = None
                end_offset = 0
                batch_raw_bytes = 0
                raw_digest = hashlib.sha256()
                while True:
                    line_offset = handle.tell()
                    raw_line = handle.readline()
                    if not raw_line:
                        break
                    if len(raw_line) > PREPARE_MAX_RAW_LINE_BYTES:
                        raise ValueError(
                            "A JSONL physical line exceeds the preparation safety limit: "
                            f"path={path}, byte_offset={line_offset}, "
                            f"line_bytes={len(raw_line):,}, "
                            f"limit={PREPARE_MAX_RAW_LINE_BYTES:,}"
                        )
                    row_is_synthetic = path_is_synthetic
                    if not path_is_synthetic:
                        try:
                            decoded = raw_line.decode("utf-8-sig")
                            row_is_synthetic = synthetic_record(json.loads(decoded))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            row_is_synthetic = False
                    if row_is_synthetic != synthetic_pass:
                        continue
                    if rows and batch_raw_bytes + len(raw_line) > PREPARE_MAX_BATCH_RAW_BYTES:
                        assert start_offset is not None
                        descriptor = _PrepareBatchDescriptor(
                            synthetic_pass=synthetic_pass,
                            source_id=source_id,
                            batch_index=batch_index,
                            start_offset=start_offset,
                            end_offset=end_offset,
                            row_count=len(rows),
                            raw_bytes=batch_raw_bytes,
                            raw_sha256=raw_digest.hexdigest(),
                        )
                        yield (
                            descriptor,
                            (
                                source_id,
                                rows,
                                quality_policy,
                                filter_quality,
                                language_pairs,
                                max_tokens_per_side,
                                path.name,
                                translation_directions,
                            ),
                        )
                        batch_index += 1
                        rows = []
                        start_offset = None
                        batch_raw_bytes = 0
                        raw_digest = hashlib.sha256()
                    if start_offset is None:
                        start_offset = line_offset
                    rows.append((line_offset, raw_line))
                    batch_raw_bytes += len(raw_line)
                    raw_digest.update(len(raw_line).to_bytes(8, "little"))
                    raw_digest.update(raw_line)
                    end_offset = handle.tell()
                    if len(rows) >= batch_size:
                        assert start_offset is not None
                        descriptor = _PrepareBatchDescriptor(
                            synthetic_pass=synthetic_pass,
                            source_id=source_id,
                            batch_index=batch_index,
                            start_offset=start_offset,
                            end_offset=end_offset,
                            row_count=len(rows),
                            raw_bytes=batch_raw_bytes,
                            raw_sha256=raw_digest.hexdigest(),
                        )
                        yield (
                            descriptor,
                            (
                                source_id,
                                rows,
                                quality_policy,
                                filter_quality,
                                language_pairs,
                                max_tokens_per_side,
                                path.name,
                                translation_directions,
                            ),
                        )
                        batch_index += 1
                        rows = []
                        start_offset = None
                        batch_raw_bytes = 0
                        raw_digest = hashlib.sha256()
                if rows:
                    assert start_offset is not None
                    descriptor = _PrepareBatchDescriptor(
                        synthetic_pass=synthetic_pass,
                        source_id=source_id,
                        batch_index=batch_index,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        row_count=len(rows),
                        raw_bytes=batch_raw_bytes,
                        raw_sha256=raw_digest.hexdigest(),
                    )
                    yield (
                        descriptor,
                        (
                            source_id,
                            rows,
                            quality_policy,
                            filter_quality,
                            language_pairs,
                            max_tokens_per_side,
                            path.name,
                            translation_directions,
                        ),
                    )


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


# Input prefixes that identify synthetic examples. Keep these sources in the
# training split because placing backtranslations or concatenations in a
# holdout would measure the generation rule instead of translation quality.
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
    refinement_evidence_fraction: float = 0.0,
    source_only_synthetic_evidence_files: Sequence[str] = (),
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
        "refinement_evidence_fraction": refinement_evidence_fraction,
        "shard_size": shard_size,
        "source_only_languages": list(source_only_languages),
        "translation_directions": [list(direction) for direction in translation_directions],
        "split_key": split_key_schema,
        "synthetic_sampling_weight": synthetic_sampling_weight,
        "source_only_synthetic_evidence_files": list(source_only_synthetic_evidence_files),
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


def _prepare_event_json(value: object) -> str:
    """Encode raw-record-derived events without losing permissive JSON values."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        # Python's input decoder has historically accepted NaN and infinities.
        # Preserve those values inside optional metadata so checkpointing does
        # not turn a formerly processable row into a fatal whole-corpus error.
        allow_nan=True,
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


def _prepare_chunk_filename(descriptor: _PrepareBatchDescriptor) -> str:
    pass_index = 1 if descriptor.synthetic_pass else 0
    return (
        f"pass-{pass_index}-source-{descriptor.source_id:05d}-"
        f"batch-{descriptor.batch_index:09d}.json.gz"
    )


def _prepare_descriptor_from_json(value: object) -> _PrepareBatchDescriptor:
    if not isinstance(value, Mapping):
        raise _PrepareProgressError("prepare progress chunk descriptor must be an object")
    raw = cast(Mapping[object, object], value)
    expected_fields = {
        "synthetic_pass",
        "source_id",
        "batch_index",
        "start_offset",
        "end_offset",
        "row_count",
        "raw_bytes",
        "raw_sha256",
    }
    if set(raw) != expected_fields:
        raise _PrepareProgressError("prepare progress chunk descriptor fields are invalid")
    synthetic_pass = raw["synthetic_pass"]
    if not isinstance(synthetic_pass, bool):
        raise _PrepareProgressError("prepare progress synthetic_pass must be boolean")
    integer_values: dict[str, int] = {}
    for name in (
        "source_id",
        "batch_index",
        "start_offset",
        "end_offset",
        "row_count",
        "raw_bytes",
    ):
        item = raw[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise _PrepareProgressError(f"prepare progress {name} must be non-negative")
        integer_values[name] = item
    if integer_values["row_count"] < 1:
        raise _PrepareProgressError("prepare progress row_count must be positive")
    if integer_values["end_offset"] <= integer_values["start_offset"]:
        raise _PrepareProgressError("prepare progress byte offsets are invalid")
    if not 0 < integer_values["raw_bytes"] <= PREPARE_MAX_BATCH_RAW_BYTES:
        raise _PrepareProgressError("prepare progress raw byte count is invalid")
    if integer_values["raw_bytes"] > (
        integer_values["end_offset"] - integer_values["start_offset"]
    ):
        raise _PrepareProgressError("prepare progress raw bytes exceed their source span")
    raw_sha256 = raw["raw_sha256"]
    if (
        not isinstance(raw_sha256, str)
        or len(raw_sha256) != 64
        or any(character not in "0123456789abcdef" for character in raw_sha256)
    ):
        raise _PrepareProgressError("prepare progress raw SHA-256 is invalid")
    return _PrepareBatchDescriptor(
        synthetic_pass=synthetic_pass,
        source_id=integer_values["source_id"],
        batch_index=integer_values["batch_index"],
        start_offset=integer_values["start_offset"],
        end_offset=integer_values["end_offset"],
        row_count=integer_values["row_count"],
        raw_bytes=integer_values["raw_bytes"],
        raw_sha256=raw_sha256,
    )


def _prepare_event_payload(
    value: object,
    *,
    descriptor: _PrepareBatchDescriptor,
    language_pairs: tuple[tuple[str, str], ...],
    max_tokens_per_side: int,
) -> list[_PrepareEvent]:
    """Validate cached worker events before the reducer is allowed to consume them."""

    if not isinstance(value, list):
        raise _PrepareProgressError("prepare progress events must be an array")
    simple_statuses = {
        "physical_line",
        "invalid_utf8",
        "invalid_json",
        "invalid_record",
        "non_string",
        "missing_text",
        "invalid_language",
        "unaligned_lists",
    }
    allowed_pairs = set(language_pairs)
    validated: list[_PrepareEvent] = []
    physical_lines = 0
    for raw_event in cast(list[object], value):
        if not isinstance(raw_event, list):
            raise _PrepareProgressError("prepare progress event shape is invalid")
        event_items = cast(list[object], raw_event)
        if len(event_items) != 2:
            raise _PrepareProgressError("prepare progress event shape is invalid")
        status, raw_payload = event_items
        if not isinstance(status, str) or not isinstance(raw_payload, list):
            raise _PrepareProgressError("prepare progress event types are invalid")
        payload = cast(list[object], raw_payload)
        if not payload:
            raise _PrepareProgressError("prepare progress event payload is empty")
        source_id = payload[0]
        if (
            isinstance(source_id, bool)
            or not isinstance(source_id, int)
            or source_id != descriptor.source_id
        ):
            raise _PrepareProgressError("prepare progress event source identity is invalid")
        if status in simple_statuses:
            if len(payload) != 1:
                raise _PrepareProgressError(f"prepare progress {status} payload is invalid")
            if status == "physical_line":
                physical_lines += 1
        elif status in {"quality_filtered", "too_long"}:
            if len(payload) != 3:
                raise _PrepareProgressError(f"prepare progress {status} event is invalid")
            for reasons in payload[1:]:
                if not isinstance(reasons, list) or not all(
                    isinstance(reason, str) for reason in cast(list[object], reasons)
                ):
                    raise _PrepareProgressError(f"prepare progress {status} reasons are invalid")
        elif status == "candidate":
            if len(payload) != 15:
                raise _PrepareProgressError("prepare progress candidate payload is invalid")
            group_key, synthetic, language_a, language_b, metadata = payload[1:6]
            text_a, text_b, ids_a, ids_b = payload[6:10]
            register_a, register_b, quality_score = payload[10:13]
            if (
                not isinstance(group_key, str)
                or len(group_key) != 64
                or any(character not in "0123456789abcdef" for character in group_key)
                or not isinstance(synthetic, bool)
                or not isinstance(language_a, str)
                or not isinstance(language_b, str)
                or (language_a, language_b) not in allowed_pairs
                or not isinstance(metadata, dict)
                or not isinstance(text_a, str)
                or not isinstance(text_b, str)
            ):
                raise _PrepareProgressError("prepare progress candidate identity is invalid")
            try:
                metadata_size = len(encode_record_metadata(cast(dict[str, object], metadata)))
            except (TypeError, ValueError) as error:
                raise _PrepareProgressError(
                    "prepare progress candidate metadata is invalid"
                ) from error
            if metadata_size > PREPARE_MAX_RECORD_METADATA_BYTES:
                raise _PrepareProgressError("prepare progress candidate metadata exceeds its bound")
            for token_ids in (ids_a, ids_b):
                if not isinstance(token_ids, list):
                    raise _PrepareProgressError("prepare progress token IDs are invalid")
                normalized_token_ids = cast(list[object], token_ids)
                if not all(
                    not isinstance(token_id, bool)
                    and isinstance(token_id, int)
                    and 0 <= token_id <= np.iinfo(np.uint32).max
                    for token_id in normalized_token_ids
                ):
                    raise _PrepareProgressError("prepare progress token IDs are invalid")
                if len(normalized_token_ids) > max_tokens_per_side:
                    raise _PrepareProgressError(
                        "prepare progress candidate token IDs exceed the preprocessing contract"
                    )
            if any(
                isinstance(register, bool)
                or not isinstance(register, int)
                or register not in (0, 1, 2, 3)
                for register in (register_a, register_b)
            ):
                raise _PrepareProgressError("prepare progress register is invalid")
            if (
                isinstance(quality_score, bool)
                or not isinstance(quality_score, int)
                or not 0 <= quality_score <= 100
            ):
                raise _PrepareProgressError("prepare progress quality score is invalid")
            for reasons in payload[13:]:
                if not isinstance(reasons, list) or not all(
                    isinstance(reason, str) for reason in cast(list[object], reasons)
                ):
                    raise _PrepareProgressError("prepare progress candidate reasons are invalid")
        else:
            raise _PrepareProgressError(f"prepare progress event status is invalid: {status!r}")
        validated.append((status, tuple(payload)))
    if physical_lines != descriptor.row_count:
        raise _PrepareProgressError(
            "prepare progress physical-line count does not match its source batch"
        )
    return validated


def _prepare_chunk_uncompressed_limit(descriptor: _PrepareBatchDescriptor) -> int:
    del descriptor
    # Upstream line, fanout, metadata, and token limits cap production before
    # serialization. This absolute ceiling makes recovery independent of a
    # language graph's shape while still rejecting gzip bombs.
    return PREPARE_MAX_CHUNK_UNCOMPRESSED_BYTES


def _load_prepare_chunk(
    path: Path,
    *,
    expected_descriptor: _PrepareBatchDescriptor,
    progress_contract_sha256: str,
    language_pairs: tuple[tuple[str, str], ...],
    max_tokens_per_side: int,
) -> tuple[_PrepareBatchDescriptor, list[_PrepareEvent]]:
    try:
        _regular_file_stat(path, role="prepare progress chunk")
    except (OSError, ValueError) as error:
        raise _PrepareProgressError(f"prepare progress chunk is unsafe: {path}") from error
    try:
        with gzip.open(path, "rb") as handle:
            limit = _prepare_chunk_uncompressed_limit(expected_descriptor)
            encoded = handle.read(limit + 1)
    except (OSError, EOFError) as error:
        raise _PrepareProgressError(f"cannot decompress prepare progress chunk: {path}") from error
    if len(encoded) > limit:
        raise _PrepareProgressError(f"prepare progress chunk expands beyond its bound: {path}")
    try:
        raw_document: object = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _PrepareProgressError(f"cannot decode prepare progress chunk: {path}") from error
    if not isinstance(raw_document, Mapping):
        raise _PrepareProgressError(f"prepare progress chunk must be an object: {path}")
    document = cast(Mapping[object, object], raw_document)
    if set(document) != {
        "schema",
        "contract_sha256",
        "descriptor",
        "integrity_sha256",
        "events",
    }:
        raise _PrepareProgressError(f"prepare progress chunk fields are invalid: {path}")
    if document["schema"] != PREPARE_PROGRESS_CHUNK_SCHEMA:
        raise _PrepareProgressError(f"prepare progress chunk schema is incompatible: {path}")
    if document["contract_sha256"] != progress_contract_sha256:
        raise _PrepareProgressError(f"prepare progress chunk contract is incompatible: {path}")
    descriptor = _prepare_descriptor_from_json(document["descriptor"])
    if descriptor != expected_descriptor:
        raise _PrepareProgressError(f"prepare progress chunk source coordinates changed: {path}")
    if path.name != _prepare_chunk_filename(descriptor):
        raise _PrepareProgressError(f"prepare progress chunk filename is invalid: {path}")
    raw_events = document["events"]
    payload_digest = document["integrity_sha256"]
    integrity_payload = {
        "contract_sha256": document["contract_sha256"],
        "descriptor": document["descriptor"],
        "events": raw_events,
    }
    if not isinstance(payload_digest, str) or not hmac.compare_digest(
        payload_digest,
        hashlib.sha256(_prepare_event_json(integrity_payload).encode("utf-8")).hexdigest(),
    ):
        raise _PrepareProgressError(f"prepare progress chunk integrity digest is invalid: {path}")
    return descriptor, _prepare_event_payload(
        raw_events,
        descriptor=descriptor,
        language_pairs=language_pairs,
        max_tokens_per_side=max_tokens_per_side,
    )


def _write_prepare_chunk(job: _PrepareBatchJob, events: list[_PrepareEvent]) -> None:
    chunks_dir = Path(job.progress_chunks_dir)
    _assert_regular_directory(chunks_dir, role="prepare progress chunks")
    path = chunks_dir / _prepare_chunk_filename(job.descriptor)
    raw_events = [[status, list(payload)] for status, payload in events]
    # Validate the exact representation that will be recovered, rather than
    # trusting an in-memory worker result that JSON might coerce unexpectedly.
    _prepare_event_payload(
        json.loads(_prepare_event_json(raw_events)),
        descriptor=job.descriptor,
        language_pairs=job.batch[4],
        max_tokens_per_side=job.batch[5],
    )
    document = {
        "schema": PREPARE_PROGRESS_CHUNK_SCHEMA,
        "contract_sha256": job.progress_contract_sha256,
        "descriptor": job.descriptor.to_dict(),
        "events": raw_events,
    }
    integrity_payload = {
        "contract_sha256": document["contract_sha256"],
        "descriptor": document["descriptor"],
        "events": raw_events,
    }
    document["integrity_sha256"] = hashlib.sha256(
        _prepare_event_json(integrity_payload).encode("utf-8")
    ).hexdigest()
    encoded = (_prepare_event_json(document) + "\n").encode("utf-8")
    if len(encoded) > _prepare_chunk_uncompressed_limit(job.descriptor):
        raise RuntimeError("prepare worker output exceeded its deterministic chunk bound")
    progress = _PrepareProgress(
        root=chunks_dir.parent,
        chunks_dir=chunks_dir,
        contract_sha256=job.progress_contract_sha256,
        generation_epoch=job.generation_epoch,
    )
    with _prepare_writer_lease(progress.root):
        observed_epoch = _read_prepare_generation_epoch(progress)
        if observed_epoch != job.generation_epoch:
            raise _PrepareProgressError("prepare worker belongs to a superseded parent generation")
        if _path_exists(path):
            _, winner = _load_prepare_chunk(
                path,
                expected_descriptor=job.descriptor,
                progress_contract_sha256=job.progress_contract_sha256,
                language_pairs=job.batch[4],
                max_tokens_per_side=job.batch[5],
            )
            winner_json = _prepare_event_json(
                [[status, list(payload)] for status, payload in winner]
            )
            if not hmac.compare_digest(winner_json, _prepare_event_json(raw_events)):
                raise _PrepareProgressError(
                    f"competing prepare worker published different events: {path}"
                )
            return
        # Reserve inspection and publication share one kernel lease, so two
        # workers cannot both spend the cleanup reserve observed by one check.
        _ensure_prepare_write_reserve(chunks_dir, len(encoded))
        temporary = chunks_dir / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as raw_handle:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=6,
                    fileobj=raw_handle,
                    mtime=0,
                ) as compressed:
                    compressed.write(encoded)
                raw_handle.flush()
                os.fsync(raw_handle.fileno())
            # A hard-link publication is atomic and refuses an existing name on
            # both POSIX and Windows. The final checkpoint is therefore immutable.
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                _, winner = _load_prepare_chunk(
                    path,
                    expected_descriptor=job.descriptor,
                    progress_contract_sha256=job.progress_contract_sha256,
                    language_pairs=job.batch[4],
                    max_tokens_per_side=job.batch[5],
                )
                winner_json = _prepare_event_json(
                    [[status, list(payload)] for status, payload in winner]
                )
                if not hmac.compare_digest(winner_json, _prepare_event_json(raw_events)):
                    raise _PrepareProgressError(
                        f"competing prepare worker published different events: {path}"
                    ) from error
            _fsync_directory(chunks_dir)
        finally:
            temporary.unlink(missing_ok=True)


def _process_prepare_job(job: _PrepareBatchJob) -> list[_PrepareEvent]:
    path = Path(job.progress_chunks_dir) / _prepare_chunk_filename(job.descriptor)
    if _path_exists(path):
        _, events = _load_prepare_chunk(
            path,
            expected_descriptor=job.descriptor,
            progress_contract_sha256=job.progress_contract_sha256,
            language_pairs=job.batch[4],
            max_tokens_per_side=job.batch[5],
        )
        return events
    events = _process_prepare_batch(job.batch)
    _write_prepare_chunk(job, events)
    return events


def _write_json_durable(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _replace_json_durable(path: Path, value: object) -> None:
    """Atomically replace a small JSON control file and sync its directory."""

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {role}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return cast(dict[str, Any], value)


def _read_prepare_generation_epoch(progress: _PrepareProgress) -> int:
    document = _read_json_object(
        progress.root / PREPARE_PROGRESS_EPOCH_FILENAME,
        role="prepare progress generation",
    )
    if set(document) != {"schema", "epoch"}:
        raise _PrepareProgressError("prepare progress generation fields are invalid")
    epoch = document.get("epoch")
    if (
        document.get("schema") != PREPARE_PROGRESS_EPOCH_SCHEMA
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
    ):
        raise _PrepareProgressError("prepare progress generation is invalid")
    return epoch


def _prepare_writer_conflict_message(root: Path, holder: str) -> str:
    return (
        f"prepare progress still has an active worker writer: {root}\n"
        f"  current holder: {holder}\n"
        "Wait for the writer to finish; its completed checkpoint remains reusable."
    )


@contextmanager  # pyright: ignore[reportDeprecated]
def _prepare_writer_lease(progress_root: Path) -> Iterator[Path]:
    with _exclusive_lock(
        progress_root,
        filename=PREPARE_PROGRESS_WRITER_LOCK_FILENAME,
        conflict_message=_prepare_writer_conflict_message,
        timeout=60.0,
        poll_interval=0.1,
    ) as lock_path:
        yield lock_path


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
            with os.scandir(directory) as scanner:
                entries = [
                    (Path(entry.path), entry.stat(follow_symlinks=False)) for entry in scanner
                ]
        except OSError as error:
            raise RuntimeError(f"Cannot inspect orphan dataset staging: {directory}") from error
        for entry_path, entry_stat in entries:
            if _is_reparse_stat(entry_stat):
                raise RuntimeError(
                    "Refusing to clean orphan staging containing a symlink/reparse point: "
                    f"{entry_path}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                pending.append(entry_path)
            elif not stat.S_ISREG(entry_stat.st_mode):
                raise RuntimeError(f"Refusing to clean non-regular staging artifact: {entry_path}")
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


def _prepare_progress_disk_bytes(progress: _PrepareProgress) -> int:
    """Return checkpoint file sizes without following filesystem links."""

    total = 0
    for root in (progress.root, progress.chunks_dir):
        _assert_regular_directory(root, role="prepare progress directory")
    for path in (
        progress.root / PREPARE_PROGRESS_CONTRACT_FILENAME,
        progress.root / PREPARE_PROGRESS_EPOCH_FILENAME,
        *progress.chunks_dir.iterdir(),
    ):
        total += _regular_file_stat(path, role="prepare progress file").st_size
    return total


def _ensure_prepare_write_reserve(
    directory: Path,
    pending_bytes: int,
    *,
    role: str = "prepared worker checkpoint",
) -> None:
    """Keep enough free capacity for cleanup and filesystem metadata."""

    if pending_bytes < 0:
        raise ValueError("pending prepare bytes must be non-negative")
    free_bytes = shutil.disk_usage(directory).free
    required = pending_bytes + _PREPARE_SPACE_RESERVE_BYTES
    if free_bytes < required:
        raise OSError(
            errno.ENOSPC,
            f"Insufficient free disk space for {role}: "
            f"need at least {required:,} bytes, found {free_bytes:,}",
        )


def _ensure_prepare_capacity(
    output_dir: Path,
    source_snapshots: tuple[_FileSnapshot, ...],
    progress: _PrepareProgress,
) -> None:
    """Reject a build that is likely to exhaust its output filesystem."""

    source_bytes = sum(snapshot.size for snapshot in source_snapshots)
    estimated_final_bytes = math.ceil(source_bytes * _PREPARE_FINAL_BYTES_PER_INPUT_BYTE)
    estimated_cache_bytes = math.ceil(source_bytes * _PREPARE_CACHE_BYTES_PER_INPUT_BYTE)
    cached_bytes = _prepare_progress_disk_bytes(progress)
    remaining_cache_bytes = max(0, estimated_cache_bytes - cached_bytes)
    required = estimated_final_bytes + remaining_cache_bytes + _PREPARE_SPACE_RESERVE_BYTES
    free_bytes = shutil.disk_usage(output_dir.parent).free
    if free_bytes < required:
        raise OSError(
            errno.ENOSPC,
            "Insufficient free disk space for resumable translation dataset preparation: "
            f"estimated required={required:,} bytes "
            f"(final={estimated_final_bytes:,}, remaining checkpoint={remaining_cache_bytes:,}, "
            f"reserve={_PREPARE_SPACE_RESERVE_BYTES:,}), available={free_bytes:,}; "
            f"reusable checkpoint={cached_bytes:,}",
        )


def _ensure_prepare_final_capacity(
    output_dir: Path,
    summary: _PrepareCapacitySummary,
    *,
    shard_size: int,
    dedup_backend: str,
    translation_directions: tuple[tuple[str, str], ...],
) -> int:
    """Gate staging on exact worker totals plus conservative format overhead."""

    maximum_direction_payload = max(
        (
            len(encode_record_metadata({"training_direction": list(direction)}))
            for direction in translation_directions
        ),
        default=0,
    )
    row_bytes = summary.candidate_rows * (
        INDEX_DTYPE.itemsize + RECORD_METADATA_INDEX_DTYPE.itemsize + maximum_direction_payload
    )
    token_bytes = summary.token_ids * np.dtype(np.uint32).itemsize
    # At most three additional nearly empty shards arise when candidates split
    # over train, validation, test, and refinement evidence. The allowance covers NumPy
    # headers, directory entries, inventory records, and filesystem rounding.
    maximum_shards = math.ceil(summary.candidate_rows / shard_size) + 3
    shard_overhead = maximum_shards * _PREPARE_PER_SHARD_OVERHEAD_BYTES
    sqlite_overhead = (
        summary.candidate_rows * _PREPARE_SQLITE_BYTES_PER_CANDIDATE
        if dedup_backend == "sqlite"
        else 0
    )
    staging_capacity_plan = (
        token_bytes
        + row_bytes
        + summary.metadata_bytes
        + shard_overhead
        + sqlite_overhead
        + _PREPARE_FIXED_FINAL_OVERHEAD_BYTES
    )
    required = staging_capacity_plan + _PREPARE_SPACE_RESERVE_BYTES
    free_bytes = shutil.disk_usage(output_dir.parent).free
    if free_bytes < required:
        raise OSError(
            errno.ENOSPC,
            "Insufficient free disk space after translation worker checkpointing: "
            f"need at least {required:,} bytes "
            f"(planned staging={staging_capacity_plan:,}, "
            f"candidate rows={summary.candidate_rows:,}, "
            f"candidate token IDs={summary.token_ids:,}, "
            f"reserve={_PREPARE_SPACE_RESERVE_BYTES:,}), available={free_bytes:,}. "
            "The completed worker checkpoints are reusable after space is freed.",
        )
    return staging_capacity_plan


def _prepare_progress_contract(
    dataset_fingerprint: DatasetFingerprint,
    source_snapshots: tuple[_FileSnapshot, ...],
    tokenizer_snapshot: _FileSnapshot,
    *,
    normalized_pairs: tuple[tuple[str, str], ...],
    preprocessing_options: Mapping[str, Any],
) -> tuple[dict[str, object], str]:
    """Bind cached worker output to every input that can change its events."""

    contract: dict[str, object] = {
        "schema": PREPARE_PROGRESS_SCHEMA,
        "chunk_schema": PREPARE_PROGRESS_CHUNK_SCHEMA,
        "worker_algorithm_schema": PREPARE_WORKER_ALGORITHM_SCHEMA,
        "worker_runtime": {
            "python": ".".join(str(item) for item in sys.version_info[:3]),
            "sentencepiece": package_version("sentencepiece"),
            "unicode": unicodedata.unidata_version,
        },
        "batch_size": PREPARE_BATCH_SIZE,
        "maximum_raw_line_bytes": PREPARE_MAX_RAW_LINE_BYTES,
        "maximum_batch_raw_bytes": PREPARE_MAX_BATCH_RAW_BYTES,
        "maximum_expanded_pairs_per_line": PREPARE_MAX_EXPANDED_PAIRS_PER_LINE,
        "maximum_record_metadata_bytes": PREPARE_MAX_RECORD_METADATA_BYTES,
        "maximum_chunk_uncompressed_bytes": PREPARE_MAX_CHUNK_UNCOMPRESSED_BYTES,
        "dataset_format": INDEX_FORMAT,
        "preprocessing_schema": PREPROCESSING_SCHEMA,
        "fingerprint": dataset_fingerprint.to_dict(),
        "preprocessing_options": json.loads(_canonical_json(preprocessing_options)),
        "language_pairs": [list(pair) for pair in normalized_pairs],
        "input_scan": "real-before-synthetic-binary-lines-v1",
        "sources": [
            {
                "id": source_id,
                "resolved_path": snapshot.resolved_path,
                "size": snapshot.size,
                "sha256": snapshot.sha256,
            }
            for source_id, snapshot in enumerate(source_snapshots)
        ],
        "tokenizer": {
            "resolved_path": tokenizer_snapshot.resolved_path,
            "size": tokenizer_snapshot.size,
            "sha256": tokenizer_snapshot.sha256,
        },
    }
    digest = hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()
    envelope: dict[str, object] = {
        "schema": PREPARE_PROGRESS_SCHEMA,
        "contract_sha256": digest,
        "contract": contract,
    }
    return envelope, digest


def _prepare_progress_candidates(output_dir: Path) -> list[Path]:
    prefix = f".{output_dir.name}.prepare-progress-"
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


def _prepare_output_lock_filename(output_dir: Path) -> str:
    """Return one portable lock identity for an exact dataset output path."""

    identity = os.path.normcase(str(_absolute_path(output_dir))).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return f".{PREPARE_OUTPUT_LOCK_SCHEMA}-{digest}.lock"


def _prepare_output_conflict_message(
    output_dir: Path,
) -> Callable[[Path, str], str]:
    def describe(_root: Path, holder: str) -> str:
        return (
            f"dataset preparation output is locked by another process: {output_dir}\n"
            f"  current holder: {holder}\n"
            "  Wait for that preparation to finish or choose a different output directory."
        )

    return describe


@contextmanager  # pyright: ignore[reportDeprecated]
def _prepare_output_lock(output_dir: Path) -> Iterator[Path]:
    """Hold the output-specific lease without overlapping the artifact-root lock."""

    with _exclusive_lock(
        output_dir.parent,
        filename=_prepare_output_lock_filename(output_dir),
        conflict_message=_prepare_output_conflict_message(output_dir),
    ) as lock_path:
        yield lock_path


def _validate_prepare_progress_shape(
    progress: _PrepareProgress,
    expected_contract: Mapping[str, object],
) -> None:
    _assert_regular_directory(progress.root, role="prepare progress directory")
    actual_top_level = {entry.name for entry in progress.root.iterdir()}
    required_top_level = {
        PREPARE_PROGRESS_CONTRACT_FILENAME,
        PREPARE_PROGRESS_EPOCH_FILENAME,
        "chunks",
    }
    if actual_top_level not in (
        required_top_level,
        required_top_level | {PREPARE_PROGRESS_WRITER_LOCK_FILENAME},
    ):
        raise _PrepareProgressError("prepare progress top-level artifacts are invalid")
    _assert_regular_directory(progress.chunks_dir, role="prepare progress chunks")
    _regular_file_stat(
        progress.root / PREPARE_PROGRESS_EPOCH_FILENAME,
        role="prepare progress generation",
    )
    if PREPARE_PROGRESS_WRITER_LOCK_FILENAME in actual_top_level:
        _regular_file_stat(
            progress.root / PREPARE_PROGRESS_WRITER_LOCK_FILENAME,
            role="prepare progress writer lock",
        )
    observed_contract = _read_json_object(
        progress.root / PREPARE_PROGRESS_CONTRACT_FILENAME,
        role="prepare progress contract",
    )
    if observed_contract != expected_contract:
        raise _PrepareProgressError("prepare progress contract is incompatible")

    removed_temporary = False
    chunk_pattern = re.compile(r"pass-[01]-source-\d{5}-batch-\d{9}\.json\.gz")
    temporary_pattern = re.compile(
        r"\.pass-[01]-source-\d{5}-batch-\d{9}\.json\.gz\.[0-9a-f]{32}\.tmp"
    )
    for entry in progress.chunks_dir.iterdir():
        try:
            entry_stat = os.lstat(entry)
        except OSError as error:
            raise _PrepareProgressError(
                f"cannot inspect prepare progress entry: {entry}"
            ) from error
        if _is_reparse_stat(entry_stat) or not stat.S_ISREG(entry_stat.st_mode):
            raise _PrepareProgressError(f"prepare progress entry is unsafe: {entry}")
        if temporary_pattern.fullmatch(entry.name):
            entry.unlink()
            removed_temporary = True
            continue
        if not chunk_pattern.fullmatch(entry.name):
            raise _PrepareProgressError(f"prepare progress entry is unexpected: {entry}")
    if removed_temporary:
        _fsync_directory(progress.chunks_dir)


def _validate_prepare_progress_inventory(
    progress: _PrepareProgress,
    expected_chunk_names: set[str],
) -> None:
    """Reject checkpoint files that were not produced by this exact scan."""

    _assert_regular_directory(progress.chunks_dir, role="prepare progress chunks")
    actual_chunk_names = {entry.name for entry in progress.chunks_dir.iterdir()}
    if actual_chunk_names != expected_chunk_names:
        missing = sorted(expected_chunk_names - actual_chunk_names)
        unexpected = sorted(actual_chunk_names - expected_chunk_names)
        raise _PrepareProgressError(
            "prepare progress inventory differs from the deterministic input scan; "
            f"missing={missing[:3]!r}, unexpected={unexpected[:3]!r}"
        )


def _initialize_prepare_progress(
    output_dir: Path,
    expected_contract: dict[str, object],
    contract_sha256: str,
) -> _PrepareProgress:
    suffix = contract_sha256[:24]
    selected = output_dir.with_name(f".{output_dir.name}.prepare-progress-{suffix}")
    for candidate in _prepare_progress_candidates(output_dir):
        if candidate != selected:
            _quarantine_staging(candidate, output_dir)
    progress = _PrepareProgress(
        root=selected,
        chunks_dir=selected / "chunks",
        contract_sha256=contract_sha256,
    )
    if _path_exists(selected):
        try:
            with _prepare_writer_lease(progress.root):
                _validate_prepare_progress_shape(progress, expected_contract)
                generation_epoch = _read_prepare_generation_epoch(progress) + 1
                _replace_json_durable(
                    progress.root / PREPARE_PROGRESS_EPOCH_FILENAME,
                    {
                        "schema": PREPARE_PROGRESS_EPOCH_SCHEMA,
                        "epoch": generation_epoch,
                    },
                )
            return _PrepareProgress(
                root=progress.root,
                chunks_dir=progress.chunks_dir,
                contract_sha256=progress.contract_sha256,
                generation_epoch=generation_epoch,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _quarantine_staging(selected, output_dir)
            warnings.warn(
                f"Discarded incompatible or damaged prepare progress: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
    selected.mkdir(exist_ok=False)
    progress.chunks_dir.mkdir(exist_ok=False)
    _write_json_durable(
        selected / PREPARE_PROGRESS_CONTRACT_FILENAME,
        expected_contract,
    )
    _write_json_durable(
        selected / PREPARE_PROGRESS_EPOCH_FILENAME,
        {
            "schema": PREPARE_PROGRESS_EPOCH_SCHEMA,
            "epoch": 1,
        },
    )
    _fsync_directory(progress.chunks_dir)
    _fsync_directory(selected)
    _fsync_directory(output_dir.parent)
    return _PrepareProgress(
        root=progress.root,
        chunks_dir=progress.chunks_dir,
        contract_sha256=progress.contract_sha256,
        generation_epoch=1,
    )


def _discard_prepare_progress(progress: _PrepareProgress, output_dir: Path) -> None:
    if not _path_exists(progress.root):
        return
    try:
        try:
            _quarantine_staging(progress.root, output_dir)
        except PermissionError as error:
            if os.name != "nt" or getattr(error, "winerror", None) not in {5, 32}:
                raise
            # Windows may deny a directory rename briefly after gzip I/O even
            # after all in-process handles have closed. The tree was fully
            # checked before rename, so direct removal is the safe platform
            # fallback.
            shutil.rmtree(progress.root)
            _fsync_directory(output_dir.parent)
    except (OSError, RuntimeError, ValueError) as error:
        warnings.warn(
            f"Could not remove completed prepare progress at {progress.root}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )


def _publication_failure_is_resumable(
    error: BaseException,
    staging_dir: Path,
    output_dir: Path,
) -> bool:
    """Keep a complete generation after an ordinary publication I/O failure."""

    if (
        not isinstance(error, OSError)
        or isinstance(error, FileExistsError)
        or not _path_exists(staging_dir)
        or _path_exists(output_dir)
    ):
        return False
    try:
        _assert_regular_directory(staging_dir, role="dataset staging")
        _regular_file_stat(
            staging_dir / PREPARE_COMPLETION_FILENAME,
            role="dataset completion marker",
        )
    except (OSError, ValueError):
        return False
    return True


def _validate_staging_tree_shape(staging_dir: Path) -> None:
    _assert_regular_directory(staging_dir, role="dataset staging")
    allowed = {
        *DATASET_SPLITS,
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
    for split in DATASET_SPLITS:
        _assert_regular_directory(staging_dir / split, role=f"dataset {split} split")
    for filename in (RAW_FINGERPRINT_FILENAME, "manifest.json", PREPARE_COMPLETION_FILENAME):
        _regular_file_stat(staging_dir / filename, role=f"dataset {filename}")


def _prepare_stats_from_manifest(manifest: Mapping[str, Any]) -> PrepareStats:
    stats_schema = prepare_stats_schema_from_manifest(manifest, role="Dataset manifest")
    return validated_prepare_stats(
        manifest.get("stats"),
        stats_schema=stats_schema,
        role="Dataset manifest",
    )


def _validate_manifest_sources(
    manifest: Mapping[str, Any],
    source_snapshots: tuple[_FileSnapshot, ...],
    train_only_prefixes: tuple[str, ...],
    stats: PrepareStats,
) -> tuple[PrepareStats, ...]:
    stats_schema = prepare_stats_schema_from_manifest(manifest, role="Dataset manifest")
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
        source_stats = validated_prepare_stats(
            source_mapping.get("stats"),
            stats_schema=stats_schema,
            role=f"Dataset manifest source {source_id}",
        )
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
    synthetic_evidence_source_ids: frozenset[int],
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
    source_split_rows = {split: np.zeros(source_count, dtype=np.int64) for split in DATASET_SPLITS}
    split_rows: dict[str, int] = {}

    for split in DATASET_SPLITS:
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
            for row_id, (
                src_language_id,
                tgt_language_id,
                source_id,
                forward_flag,
                synthetic_flag,
            ) in enumerate(
                zip(
                    src_language_ids,
                    tgt_language_ids,
                    source_ids,
                    forward_only,
                    synthetic,
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
                if split in {"validation", "test"} and bool(synthetic_flag):
                    raise ValueError(
                        f"Dataset index synthetic rows must not enter {split}: {index_path}"
                    )
                if (
                    split == "refinement_evidence"
                    and bool(synthetic_flag)
                    and not (
                        int(source_id) in synthetic_evidence_source_ids
                        and source_language in source_only_set
                        and target_language not in source_only_set
                        and bool(forward_flag)
                    )
                ):
                    raise ValueError(
                        "Dataset synthetic refinement evidence contradicts its exact-source "
                        f"and one-way graph policy: {index_path}"
                    )
                if (
                    split == "refinement_evidence"
                    and not bool(synthetic_flag)
                    and int(source_id) in synthetic_evidence_source_ids
                ):
                    raise ValueError(
                        "Dataset exact synthetic evidence source produced an unmarked row: "
                        f"{index_path}"
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
        "refinement_evidence": stats.refinement_evidence,
    }:
        raise ValueError("Dataset manifest split counts differ from indexed payload rows")
    if stats.valid_pairs != (
        stats.train + stats.validation + stats.test + stats.refinement_evidence
    ):
        raise ValueError("Dataset manifest total valid pairs differ from its split counts")
    for source_id, expected in enumerate(source_stats):
        derived = {
            "valid_pairs": int(source_rows[source_id]),
            "train": int(source_split_rows["train"][source_id]),
            "validation": int(source_split_rows["validation"][source_id]),
            "test": int(source_split_rows["test"][source_id]),
            "refinement_evidence": int(source_split_rows["refinement_evidence"][source_id]),
            "synthetic_pairs": int(source_synthetic[source_id]),
            "forward_only_pairs": int(source_forward_only[source_id]),
            "quality_score_sum": int(source_quality[source_id]),
            "src_tokens": int(source_src_tokens[source_id]),
            "tgt_tokens": int(source_tgt_tokens[source_id]),
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
    raw_evidence_sources = preprocessing_options.get("source_only_synthetic_evidence_files")
    if not isinstance(raw_evidence_sources, list) or any(
        not isinstance(name, str) for name in cast(list[object], raw_evidence_sources)
    ):
        raise ValueError("Dataset preprocessing synthetic evidence sources are invalid")
    evidence_source_names = cast(list[str], raw_evidence_sources)
    if any(
        Path(name).name != name or "/" in name or "\\" in name for name in evidence_source_names
    ) or len({name.casefold() for name in evidence_source_names}) != len(evidence_source_names):
        raise ValueError("Dataset preprocessing synthetic evidence basenames are invalid")
    if manifest.get("synthetic_policy") != {
        "record_field": "synthetic",
        "train_only_by_default": True,
        "source_only_holdout_enabled": bool(evidence_source_names),
        "source_only_holdout_purpose": "relative-refinement-evidence-only-v1",
        "source_only_evidence_files": evidence_source_names,
        "source_only_target_overlap": "allowed-for-relative-evidence-only-v1",
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
        "refinement_evidence_fraction": "refinement_evidence_fraction",
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
    evidence_name_to_id = {
        Path(snapshot.resolved_path).name: source_id
        for source_id, snapshot in enumerate(source_snapshots)
    }
    missing_evidence_sources = sorted(set(evidence_source_names) - set(evidence_name_to_id))
    if missing_evidence_sources:
        raise ValueError(
            "Dataset preprocessing synthetic evidence sources are absent from the manifest: "
            f"{missing_evidence_sources}"
        )
    synthetic_evidence_source_ids = frozenset(
        evidence_name_to_id[name] for name in evidence_source_names
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
        synthetic_evidence_source_ids=synthetic_evidence_source_ids,
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


def _prepare_dataset_locked(
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
    language_pair: Sequence[str] | None = None,
    source_only_languages: Sequence[str] = (),
    language_pairs: Sequence[Sequence[str]] | None = None,
    translation_directions: Sequence[Sequence[str]] | None = None,
    train_only_prefixes: Sequence[str] = DEFAULT_TRAIN_ONLY_PREFIXES,
    managed_augmentation_prefix: str | None = None,
    synthetic_sampling_weight: float = DEFAULT_SYNTHETIC_SAMPLING_WEIGHT,
    refinement_evidence_fraction: float = 0.0,
    source_only_synthetic_evidence_files: Sequence[str] = (),
    num_workers: int | None = None,
    expected_fingerprint: DatasetFingerprint | None = None,
) -> PrepareStats:
    if validation_fraction < 0 or test_fraction < 0 or refinement_evidence_fraction < 0:
        raise ValueError("Split fractions must be non-negative")
    if validation_fraction + test_fraction + refinement_evidence_fraction >= 0.5:
        raise ValueError("Held-out split fractions are unexpectedly large")
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
    source_only = canonicalize_language_tags(
        list(source_only_languages),
        field="source_only_languages",
        reject_duplicates=False,
    )
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
    evidence_source_names: list[str] = []
    evidence_source_casefolded: set[str] = set()
    for index, raw_name in enumerate(source_only_synthetic_evidence_files):
        if not isinstance(raw_name, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("source_only_synthetic_evidence_files must contain strings")
        name = raw_name.strip()
        if (
            not name
            or name != raw_name
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or not name.casefold().endswith(".jsonl")
        ):
            raise ValueError("source_only_synthetic_evidence_files must contain exact basenames")
        folded = name.casefold()
        if folded in evidence_source_casefolded:
            raise ValueError(
                "source_only_synthetic_evidence_files contains a case-insensitive "
                f"duplicate at index {index}"
            )
        evidence_source_casefolded.add(folded)
        evidence_source_names.append(name)
    if evidence_source_names and refinement_evidence_fraction <= 0.0:
        raise ValueError(
            "source_only_synthetic_evidence_files requires a positive refinement_evidence_fraction"
        )
    if evidence_source_names and not source_only:
        raise ValueError("source_only_synthetic_evidence_files requires source_only_languages")

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
        refinement_evidence_fraction=refinement_evidence_fraction,
        source_only_synthetic_evidence_files=evidence_source_names,
        language_pair_count=len(normalized_pairs),
    )
    endpoint_key_schema = cast(str, preprocessing_options["endpoint_leakage_key"])
    split_key_schema = cast(str, preprocessing_options["split_key"])
    paths, source_snapshots = _capture_input_snapshots(input_patterns)
    path_by_name = {path.name: path for path in paths}
    missing_evidence_sources = sorted(set(evidence_source_names) - set(path_by_name))
    if missing_evidence_sources:
        raise ValueError(
            "source-only synthetic evidence files are absent from dataset inputs: "
            f"{missing_evidence_sources}"
        )
    non_synthetic_evidence_sources = [
        name
        for name in evidence_source_names
        if not synthetic_path(path_by_name[name], train_only_prefixes)
    ]
    if non_synthetic_evidence_sources:
        raise ValueError(
            "source-only synthetic evidence files must match an authenticated synthetic "
            f"prefix: {non_synthetic_evidence_sources}"
        )
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
    _assert_regular_directory(output_dir.parent, role="dataset output parent")
    progress_contract, progress_contract_sha256 = _prepare_progress_contract(
        dataset_fingerprint,
        source_snapshots,
        tokenizer_snapshot,
        normalized_pairs=normalized_pairs,
        preprocessing_options=preprocessing_options,
    )
    if _path_exists(output_dir):
        try:
            completed_stats = _validate_complete_staging(
                output_dir,
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
        except (OSError, RuntimeError, TypeError, ValueError):
            _refuse_or_remove_empty_output(output_dir)
        else:
            _verify_input_snapshots(
                input_patterns,
                paths,
                source_snapshots,
                tokenizer_path,
                tokenizer_snapshot,
            )
            completed_progress = _PrepareProgress(
                root=output_dir.with_name(
                    f".{output_dir.name}.prepare-progress-{progress_contract_sha256[:24]}"
                ),
                chunks_dir=output_dir.with_name(
                    f".{output_dir.name}.prepare-progress-{progress_contract_sha256[:24]}"
                )
                / "chunks",
                contract_sha256=progress_contract_sha256,
            )
            _discard_prepare_progress(completed_progress, output_dir)
            return completed_stats
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
        completed_progress = _PrepareProgress(
            root=output_dir.with_name(
                f".{output_dir.name}.prepare-progress-{progress_contract_sha256[:24]}"
            ),
            chunks_dir=output_dir.with_name(
                f".{output_dir.name}.prepare-progress-{progress_contract_sha256[:24]}"
            )
            / "chunks",
            contract_sha256=progress_contract_sha256,
        )
        _discard_prepare_progress(completed_progress, output_dir)
        return recovered
    progress = _initialize_prepare_progress(
        output_dir,
        progress_contract,
        progress_contract_sha256,
    )
    _ensure_prepare_capacity(output_dir, source_snapshots, progress)
    workers = num_workers or build_cpu_plan(input_files=len(paths)).dataset_workers
    expected_chunk_names: set[str] = set()
    expected_descriptors: list[_PrepareBatchDescriptor] = []

    def prepare_jobs() -> Iterator[_PrepareBatchJob]:
        for descriptor, batch in _prepare_batch_records(
            paths,
            quality_policy,
            filter_quality,
            normalized_pairs,
            train_only_prefixes,
            max_tokens_per_side,
            normalized_directions,
        ):
            chunk_name = _prepare_chunk_filename(descriptor)
            if chunk_name in expected_chunk_names:
                raise RuntimeError(f"duplicate deterministic prepare batch: {chunk_name}")
            expected_chunk_names.add(chunk_name)
            expected_descriptors.append(descriptor)
            yield _PrepareBatchJob(
                descriptor=descriptor,
                batch=batch,
                progress_chunks_dir=str(progress.chunks_dir),
                progress_contract_sha256=progress.contract_sha256,
                generation_epoch=progress.generation_epoch,
            )

    inputs = prepare_jobs()
    executor: ProcessPoolExecutor | None = None
    capacity_summary = _PrepareCapacitySummary()
    try:
        if workers <= 1:
            _initialize_prepare_worker(str(tokenizer_path))
            processed_batches = map(_process_prepare_job, inputs)
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_prepare_worker,
                initargs=(str(tokenizer_path),),
            )
            processed_batches = bounded_ordered_map(
                executor, _process_prepare_job, inputs, max_pending=workers * 2
            )
        for batch in processed_batches:
            capacity_summary.add_events(batch)
    except BaseException as error:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if isinstance(error, _PrepareProgressError):
            _discard_prepare_progress(progress, output_dir)
        raise
    else:
        if executor is not None:
            executor.shutdown()
        executor = None

    try:
        _validate_prepare_progress_inventory(progress, expected_chunk_names)
        # Finish and integrity-check every worker checkpoint before allocating any
        # staging shard. A capacity failure therefore preserves all expensive
        # tokenization work for the next invocation.
        _verify_input_snapshots(
            input_patterns,
            paths,
            source_snapshots,
            tokenizer_path,
            tokenizer_snapshot,
        )
        _ensure_prepare_final_capacity(
            output_dir,
            capacity_summary,
            shard_size=shard_size,
            dedup_backend=dedup_backend,
            translation_directions=normalized_directions,
        )
    except BaseException as error:
        if isinstance(error, _PrepareProgressError):
            _discard_prepare_progress(progress, output_dir)
        raise

    staging_dir = output_dir.with_name(f".{output_dir.name}.staging-{uuid.uuid4().hex}")
    staging_dir.mkdir(exist_ok=False)
    maximum_shard_bytes = shard_size * (
        max_tokens_per_side * np.dtype(np.uint32).itemsize * 2
        + INDEX_DTYPE.itemsize
        + RECORD_METADATA_INDEX_DTYPE.itemsize
        + 512
    )
    # Four split writers stay open concurrently. The exact pre-dedup capacity
    # plan above covers the complete staging generation; this smaller rolling
    # reserve still catches external disk consumption between shard openings.
    concurrent_shard_reserve = maximum_shard_bytes * len(DATASET_SPLITS)
    writers: dict[str, ShardWriter] = {}
    try:
        for split in DATASET_SPLITS:
            writers[split] = ShardWriter(
                staging_dir,
                split,
                shard_size,
                language_to_id,
                maximum_shard_bytes=concurrent_shard_reserve,
            )
        digest_store = (
            _SqliteDigestSet(staging_dir / ".dedup.sqlite3")
            if dedup_backend == "sqlite"
            else _MemoryDigestSet()
        )
    except BaseException:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        _discard_private_staging(staging_dir, output_dir)
        raise
    stats = PrepareStats()
    per_source_stats = [PrepareStats() for _ in paths]
    estimated_pairs = max(1, sum(snapshot.size for snapshot in source_snapshots) // 200)
    target_split_guard = (
        TargetSplitGuard(
            estimated_pairs,
            validation_fraction,
            test_fraction,
            refinement_evidence_fraction,
        )
        if prevent_target_leakage
        else None
    )
    evidence_source_name_set = frozenset(evidence_source_names)

    def replay_batches() -> Iterator[list[_PrepareEvent]]:
        for descriptor in expected_descriptors:
            _, events = _load_prepare_chunk(
                progress.chunks_dir / _prepare_chunk_filename(descriptor),
                expected_descriptor=descriptor,
                progress_contract_sha256=progress.contract_sha256,
                language_pairs=normalized_pairs,
                max_tokens_per_side=max_tokens_per_side,
            )
            yield events

    try:
        for batch in replay_batches():
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

                if status == "too_long":
                    _, rejection_reasons, warning_reasons = payload
                    _record_quality_reasons(targets, rejection_reasons)
                    if "structured_span_mismatch" in warning_reasons:
                        for target in targets:
                            _increment(target, "structured_span_warnings")
                    if "ja_no_kana" in warning_reasons:
                        for target in targets:
                            _increment(target, "ja_no_kana_warnings")
                    for target in targets:
                        _increment(target, "too_long")
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
                if len(normalized_pairs) > 1:
                    split_key = f"record\0{record_group_key}"
                else:
                    split_key = endpoint_split_key(
                        language_a,
                        text_a,
                        approximate=approximate_split,
                    )
                candidate_split = choose_split_for_key(
                    split_key,
                    validation_fraction,
                    test_fraction,
                    refinement_evidence_fraction,
                )
                if is_synthetic:
                    selected_evidence_group = bool(
                        paths[source_id].name in evidence_source_name_set
                        and candidate_split == "refinement_evidence"
                    )
                    eligible_source_only_edge = bool(
                        language_a in source_only and language_b not in source_only and forward_only
                    )
                    if selected_evidence_group and eligible_source_only_edge:
                        split = "refinement_evidence"
                    elif selected_evidence_group:
                        # One multilingual synthetic record may also expand into
                        # an ordinary sibling pair. Withhold that sibling rather
                        # than leaking the same provenance group back into train.
                        for target in targets:
                            _increment(target, "split_conflicts")
                        continue
                    else:
                        split = "train"
                else:
                    split = candidate_split
                if target_split_guard is not None:
                    source_endpoint_digest = endpoint_split_digest(
                        language_a,
                        text_a,
                        approximate=approximate_split,
                    )
                    if split == "refinement_evidence" and is_synthetic:
                        accepted_by_split_guard = target_split_guard.accept_refinement_evidence_with_training_target_overlap(
                            isolated_digests=(source_endpoint_digest,),
                            training_overlap_digests=(
                                endpoint_split_digest(
                                    language_b,
                                    text_b,
                                    approximate=approximate_split,
                                ),
                            ),
                        )
                    else:
                        accepted_by_split_guard = target_split_guard.accept_many(
                            split,
                            (
                                source_endpoint_digest,
                                endpoint_split_digest(
                                    language_b,
                                    text_b,
                                    approximate=approximate_split,
                                ),
                            ),
                        )
                    if not accepted_by_split_guard:
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
                    _increment(target, "src_tokens", len(ids_a))
                    _increment(target, "tgt_tokens", len(ids_b))
                    _increment(target, "quality_score_sum", quality_score)
                    if is_synthetic:
                        _increment(target, "synthetic_pairs")
                    if forward_only:
                        _increment(target, "forward_only_pairs")
    except BaseException as error:
        if executor is not None:
            # A failed future can leave later batches writing gzip chunks. Wait
            # for those handles to close before quarantining the progress tree,
            # especially on Windows where open files prevent directory moves.
            executor.shutdown(wait=True, cancel_futures=True)
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
        if isinstance(error, _PrepareProgressError):
            _discard_prepare_progress(progress, output_dir)
        raise
    else:
        if executor is not None:
            executor.shutdown()

    try:
        _validate_prepare_progress_inventory(progress, expected_chunk_names)
    except BaseException as error:
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
        if isinstance(error, _PrepareProgressError):
            _discard_prepare_progress(progress, output_dir)
        raise

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
            "stats_schema": PREPARE_STATS_SCHEMA,
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
                "train_only_by_default": True,
                "source_only_holdout_enabled": bool(evidence_source_names),
                "source_only_holdout_purpose": "relative-refinement-evidence-only-v1",
                "source_only_evidence_files": list(evidence_source_names),
                "source_only_target_overlap": "allowed-for-relative-evidence-only-v1",
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
            "refinement_evidence_fraction": refinement_evidence_fraction,
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
    except BaseException as error:
        # A completion marker exists only after payload validation. Preserve
        # that authenticated generation across ENOSPC, sharing violations, or
        # another ordinary publication I/O error so the next run can publish it
        # without repeating tokenization. Contract drift and output collisions
        # remain non-resumable and are cleaned immediately.
        if not _publication_failure_is_resumable(error, staging_dir, output_dir):
            _discard_private_staging(staging_dir, output_dir)
        raise
    _discard_prepare_progress(progress, output_dir)
    return stats


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
    language_pair: Sequence[str] | None = None,
    source_only_languages: Sequence[str] = (),
    language_pairs: Sequence[Sequence[str]] | None = None,
    translation_directions: Sequence[Sequence[str]] | None = None,
    train_only_prefixes: Sequence[str] = DEFAULT_TRAIN_ONLY_PREFIXES,
    managed_augmentation_prefix: str | None = None,
    synthetic_sampling_weight: float = DEFAULT_SYNTHETIC_SAMPLING_WEIGHT,
    refinement_evidence_fraction: float = 0.0,
    source_only_synthetic_evidence_files: Sequence[str] = (),
    num_workers: int | None = None,
    expected_fingerprint: DatasetFingerprint | None = None,
) -> PrepareStats:
    """Prepare one dataset generation under an output-scoped process lock."""

    normalized_output = _absolute_path(Path(output_dir))
    with _prepare_output_lock(normalized_output):
        return _prepare_dataset_locked(
            input_patterns,
            tokenizer_model,
            normalized_output,
            shard_size=shard_size,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            max_tokens_per_side=max_tokens_per_side,
            quality_policy=quality_policy,
            filter_quality=filter_quality,
            prevent_target_leakage=prevent_target_leakage,
            approximate_split=approximate_split,
            dedup_backend=dedup_backend,
            language_pair=language_pair,
            source_only_languages=source_only_languages,
            language_pairs=language_pairs,
            translation_directions=translation_directions,
            train_only_prefixes=train_only_prefixes,
            managed_augmentation_prefix=managed_augmentation_prefix,
            synthetic_sampling_weight=synthetic_sampling_weight,
            refinement_evidence_fraction=refinement_evidence_fraction,
            source_only_synthetic_evidence_files=source_only_synthetic_evidence_files,
            num_workers=num_workers,
            expected_fingerprint=expected_fingerprint,
        )
