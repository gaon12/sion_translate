from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from sion_translate.fingerprint import PREPROCESSING_SCHEMA, build_dataset_fingerprint
from sion_translate.performance import bounded_ordered_map, build_cpu_plan
from sion_translate.splitting import TargetSplitGuard, choose_split_for_key
from sion_translate.structured import protect_shared_structured_spans
from sion_translate.synthetic import (
    DEFAULT_SYNTHETIC_PREFIXES,
    DEFAULT_SYNTHETIC_SAMPLING_WEIGHT,
    normalize_synthetic_prefixes,
    synthetic_path,
    synthetic_record,
)
from sion_translate.tokenizer import SLOT_SYMBOLS, SionTokenizer, expand_inputs

from .quality import QualityPolicy, assess_pair, canonical_text, dedup_key
from .records import (
    expand_parallel_record,
    languages_from_pairs,
    normalize_language_pairs,
)

INDEX_FORMAT = "sion-indexed-parallel-v5"

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
        # 1 when the reverse direction must never be trained, because it would
        # put a source-only language (한본어 kj) on the target side.
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
        self.src_offset = 0
        self.tgt_offset = 0
        self.total_records = 0
        self._src_handle = None
        self._tgt_handle = None
        self._open_shard()

    def _prefix(self) -> str:
        return f"{self.shard_index:05d}"

    def _open_shard(self) -> None:
        self._src_handle = (self.root / f"{self._prefix()}.src.bin").open("wb")
        self._tgt_handle = (self.root / f"{self._prefix()}.tgt.bin").open("wb")
        self.records = []
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
    ) -> None:
        assert self._src_handle is not None and self._tgt_handle is not None
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


_PREPARE_WORKER_TOKENIZER: SionTokenizer | None = None


def _initialize_prepare_worker(tokenizer_model: str) -> None:
    global _PREPARE_WORKER_TOKENIZER
    _PREPARE_WORKER_TOKENIZER = SionTokenizer(tokenizer_model)


def _process_prepare_batch(args):
    """CPU-heavy, order-preserving row work executed in worker processes."""

    source_id, rows, quality_policy, filter_quality, language_pairs = args
    tokenizer = _PREPARE_WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("prepare worker tokenizer was not initialized")
    output = []
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
    batch_size: int = 512,
):
    for source_id, path in enumerate(paths):
        with path.open("rb") as handle:
            rows: list[bytes] = []
            for raw_line in handle:
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
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        self.connection.execute("CREATE TABLE digests (digest BLOB PRIMARY KEY) WITHOUT ROWID")
        self.connection.execute("BEGIN IMMEDIATE")

    def add_if_new(self, digest: bytes) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO digests(digest) VALUES (?)",
            (sqlite3.Binary(digest),),
        )
        return cursor.rowcount == 1

    def close(self) -> None:
        if self.connection is not None:
            self.connection.commit()
            self.connection.close()
            self.connection = None  # type: ignore[assignment]


# 합성 데이터가 든 입력 파일의 접두어. 이런 파일은 train split 에만 넣습니다 —
# 역번역이나 이어붙이기로 만든 예제가 holdout 에 들어가면 점수가 실제 번역 품질이
# 아니라 합성 규칙을 재게 됩니다.
DEFAULT_TRAIN_ONLY_PREFIXES: tuple[str, ...] = DEFAULT_SYNTHETIC_PREFIXES


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
    dedup_backend: str = "sqlite",
    language_pair: Sequence[str] = ("ko", "ja"),
    source_only_languages: Sequence[str] = (),
    language_pairs: Sequence[Sequence[str]] | None = None,
    train_only_prefixes: Sequence[str] = DEFAULT_TRAIN_ONLY_PREFIXES,
    synthetic_sampling_weight: float = DEFAULT_SYNTHETIC_SAMPLING_WEIGHT,
    num_workers: int | None = None,
) -> PrepareStats:
    paths = expand_inputs(input_patterns)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {input_patterns}")
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Validation and test fractions must be non-negative")
    if validation_fraction + test_fraction >= 0.5:
        raise ValueError("Validation and test fractions are unexpectedly large")
    if max_tokens_per_side < 1:
        raise ValueError("max_tokens_per_side must be positive")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if len(paths) > np.iinfo(np.uint16).max:
        raise ValueError("Too many input files for the uint16 source_id field")
    if dedup_backend not in {"sqlite", "memory"}:
        raise ValueError("dedup_backend must be either 'sqlite' or 'memory'")
    if not 0.0 <= synthetic_sampling_weight <= 1.0:
        raise ValueError("synthetic_sampling_weight must be in [0, 1]")

    quality_policy = quality_policy or QualityPolicy()
    quality_policy.validate()
    normalized_pairs = normalize_language_pairs(language_pair, language_pairs)
    train_only_prefixes = normalize_synthetic_prefixes(train_only_prefixes)
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
    source_only_set = frozenset(source_only)

    # Validate the model and all required language tags before a large worker pool.
    tokenizer = SionTokenizer(tokenizer_model)
    missing_languages = sorted(set(languages) - set(tokenizer.languages))
    if missing_languages:
        raise ValueError(
            "Tokenizer is missing configured language tags: "
            f"{missing_languages}; retrain it with language_pairs={normalized_pairs!r}"
        )
    dataset_fingerprint = build_dataset_fingerprint(
        paths,
        language_pairs=normalized_pairs,
        tokenizer_model=tokenizer_model,
        preprocessing_schema=PREPROCESSING_SCHEMA,
        preprocessing_options={
            "dedup_backend": dedup_backend,
            "filter_quality": filter_quality,
            "index_dtype": INDEX_DTYPE.descr,
            "max_tokens_per_side": max_tokens_per_side,
            "prevent_target_leakage": prevent_target_leakage,
            "quality_policy": quality_policy.to_dict(),
            "shard_size": shard_size,
            "source_only_languages": list(source_only),
            "synthetic_sampling_weight": synthetic_sampling_weight,
            "test_fraction": test_fraction,
            "train_only_prefixes": list(train_only_prefixes),
            "validation_fraction": validation_fraction,
        },
    )
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use a new directory so stale shards cannot mix with this run."
            )
        output_dir.rmdir()
    staging_dir = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    staging_dir.mkdir()
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
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    stats = PrepareStats()
    per_source_stats = [PrepareStats() for _ in paths]
    estimated_pairs = max(1, sum(path.stat().st_size for path in paths) // 200)
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
    )
    if workers <= 1:
        _initialize_prepare_worker(str(tokenizer_model))
        processed_batches = map(_process_prepare_batch, inputs)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_prepare_worker,
            initargs=(str(tokenizer_model),),
        )
        processed_batches = bounded_ordered_map(
            executor, _process_prepare_batch, inputs, max_pending=workers * 2
        )

    try:
        for batch in processed_batches:
            for status, payload in batch:
                source_id = payload[0]
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
                _record_quality_reasons(targets, rejection_reasons)
                if "structured_span_mismatch" in warning_reasons:
                    for target in targets:
                        _increment(target, "structured_span_warnings")
                if "ja_no_kana" in warning_reasons:
                    for target in targets:
                        _increment(target, "ja_no_kana_warnings")

                # A source-only language must sit on side A so that direction 0
                # translates out of it. Swapping here, before dedup and split
                # keying, keeps every downstream key consistent with what is
                # actually written to the shard.
                forward_only = False
                if source_only_set:
                    if language_b in source_only_set:
                        language_a, language_b = language_b, language_a
                        text_a, text_b = text_b, text_a
                        ids_a, ids_b = ids_b, ids_a
                        register_a, register_b = register_b, register_a
                    forward_only = language_a in source_only_set

                pair_key = (
                    f"{language_a}\0{dedup_key(text_a)}\0{language_b}\0{dedup_key(text_b)}"
                ).encode("utf-8")
                pair_digest = hashlib.sha256(pair_key).digest()[:16]
                if not digest_store.add_if_new(pair_digest):
                    for target in targets:
                        _increment(target, "duplicates")
                    continue

                if len(ids_a) > max_tokens_per_side or len(ids_b) > max_tokens_per_side:
                    for target in targets:
                        _increment(target, "too_long")
                    continue

                # Group all translations sharing a source text in one split.
                is_synthetic = record_is_synthetic or synthetic_path(
                    paths[source_id], train_only_prefixes
                )
                if is_synthetic:
                    split = "train"
                else:
                    source_key = dedup_key(text_a)
                    if len(normalized_pairs) > 1:
                        source_key = f"record\0{record_group_key}"
                    split = choose_split_for_key(
                        source_key,
                        validation_fraction,
                        test_fraction,
                    )
                if target_split_guard is not None:
                    target_key = dedup_key(text_b)
                    if len(normalized_pairs) > 1:
                        target_key = f"{language_b}\0{target_key}"
                    target_digest = hashlib.sha256(target_key.encode("utf-8")).digest()
                    if not target_split_guard.accept(split, target_digest):
                        for target in targets:
                            _increment(target, "split_conflicts")
                        continue
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
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    else:
        if executor is not None:
            executor.shutdown()

    try:
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
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    manifest = {
        "format": INDEX_FORMAT,
        "language_pair": list(normalized_pairs[0]),
        "language_pairs": [list(pair) for pair in normalized_pairs],
        "languages": list(languages),
        "language_to_id": language_to_id,
        "source_only_languages": list(source_only),
        "storage_sides": ["src", "tgt"],
        "train_only_prefixes": list(train_only_prefixes),
        "synthetic_policy": {
            "record_field": "synthetic",
            "train_only": True,
            "sampling_weight": synthetic_sampling_weight,
            "prefixes": list(train_only_prefixes),
        },
        "tokenizer_model": str(Path(tokenizer_model).resolve()),
        "fingerprint": dataset_fingerprint.to_dict(),
        "inputs": [str(path) for path in paths],
        "sources": [
            {
                "id": source_id,
                "name": path.name,
                "path": str(path),
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
        "target_leakage_guard": "bloom-v1",
        "dedup_backend": dedup_backend,
        "atomic_build": True,
    }
    try:
        with (staging_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        staging_dir.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return stats
