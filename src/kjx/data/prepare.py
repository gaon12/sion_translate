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

from kjx.tokenizer import KJTokenizer, expand_inputs
from kjx.splitting import TargetSplitGuard, choose_split_for_key
from kjx.performance import bounded_ordered_map, build_cpu_plan

from .quality import QualityPolicy, assess_pair, canonical_text, dedup_key


INDEX_DTYPE = np.dtype(
    [
        ("ko_offset", "<u8"),
        ("ko_length", "<u4"),
        ("ja_offset", "<u8"),
        ("ja_length", "<u4"),
        ("ko_register", "u1"),
        ("ja_register", "u1"),
        ("source_id", "<u2"),
        ("quality_score", "u1"),
    ]
)

PROTECTED_PATTERN = re.compile(
    r"https?://[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"\{[A-Za-z_][A-Za-z0-9_.-]*\}|%[A-Za-z]|"
    r"(?<![A-Za-z0-9_.-])[A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*"
    r"(?![A-Za-z0-9_.-])|"
    r"(?<![A-Za-z0-9])\d[\d,.:/%+\-]*\d(?![A-Za-z0-9])|"
    r"(?<![A-Za-z0-9])\d(?![A-Za-z0-9])"
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

    ko_spans = {match.group(0) for match in PROTECTED_PATTERN.finditer(ko)}
    ja_spans = {match.group(0) for match in PROTECTED_PATTERN.finditer(ja)}
    shared = sorted(ko_spans & ja_spans, key=lambda value: (-len(value), value))[:maximum]
    replacements = {value: f"<slot_{index}>" for index, value in enumerate(shared)}

    def replace_matches(text: str) -> str:
        pieces: list[str] = []
        cursor = 0
        for match in PROTECTED_PATTERN.finditer(text):
            replacement = replacements.get(match.group(0))
            if replacement is None:
                continue
            pieces.append(text[cursor : match.start()])
            pieces.append(replacement)
            cursor = match.end()
        pieces.append(text[cursor:])
        return "".join(pieces)

    return replace_matches(ko), replace_matches(ja)


class ShardWriter:
    def __init__(
        self,
        root: Path,
        split: str,
        shard_size: int,
        side_names: Sequence[str] = ("ko", "ja"),
    ):
        self.root = root / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = shard_size
        # 토큰 bin 파일 이름에 쓰이는 언어 이름 (예: 00000.en.bin)
        self.side_names = tuple(side_names)
        self.shard_index = 0
        self.records: list[tuple[int, int, int, int, int, int, int, int]] = []
        self.ko_offset = 0
        self.ja_offset = 0
        self.total_records = 0
        self._ko_handle = None
        self._ja_handle = None
        self._open_shard()

    def _prefix(self) -> str:
        return f"{self.shard_index:05d}"

    def _open_shard(self) -> None:
        self._ko_handle = (self.root / f"{self._prefix()}.{self.side_names[0]}.bin").open("wb")
        self._ja_handle = (self.root / f"{self._prefix()}.{self.side_names[1]}.bin").open("wb")
        self.records = []
        self.ko_offset = 0
        self.ja_offset = 0

    def add(
        self,
        ko_ids: Sequence[int],
        ja_ids: Sequence[int],
        ko_register: int,
        ja_register: int,
        source_id: int,
        quality_score: int,
    ) -> None:
        assert self._ko_handle is not None and self._ja_handle is not None
        ko_array = np.asarray(ko_ids, dtype=np.uint32)
        ja_array = np.asarray(ja_ids, dtype=np.uint32)
        ko_array.tofile(self._ko_handle)
        ja_array.tofile(self._ja_handle)
        self.records.append(
            (
                self.ko_offset,
                len(ko_array),
                self.ja_offset,
                len(ja_array),
                ko_register,
                ja_register,
                source_id,
                quality_score,
            )
        )
        self.ko_offset += len(ko_array)
        self.ja_offset += len(ja_array)
        self.total_records += 1
        if len(self.records) >= self.shard_size:
            self._finish_shard()
            self.shard_index += 1
            self._open_shard()

    def _finish_shard(self) -> None:
        assert self._ko_handle is not None and self._ja_handle is not None
        self._ko_handle.close()
        self._ja_handle.close()
        if self.records:
            np.save(
                self.root / f"{self._prefix()}.idx.npy",
                np.asarray(self.records, dtype=INDEX_DTYPE),
                allow_pickle=False,
            )
        else:
            for side in self.side_names:
                (self.root / f"{self._prefix()}.{side}.bin").unlink(missing_ok=True)
        self._ko_handle = None
        self._ja_handle = None

    def close(self) -> None:
        if self._ko_handle is not None:
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


_PREPARE_WORKER_TOKENIZER: KJTokenizer | None = None


def _initialize_prepare_worker(tokenizer_model: str) -> None:
    global _PREPARE_WORKER_TOKENIZER
    _PREPARE_WORKER_TOKENIZER = KJTokenizer(tokenizer_model)


def _process_prepare_batch(args):
    """CPU-heavy, order-preserving row work executed in worker processes."""

    source_id, rows, quality_policy, filter_quality, language_pair = args
    tokenizer = _PREPARE_WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("prepare worker tokenizer was not initialized")
    key_a, key_b = language_pair
    output = []
    for raw_line in rows:
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
        if not isinstance(row, dict):
            output.append(("invalid_record", (source_id,)))
            continue
        raw_a, raw_b = row.get(key_a), row.get(key_b)
        if not isinstance(raw_a, str) or not isinstance(raw_b, str):
            output.append(("non_string", (source_id,)))
            continue
        text_a, text_b = canonical_text(raw_a), canonical_text(raw_b)
        if not text_a or not text_b:
            output.append(("missing_text", (source_id,)))
            continue
        assessment = assess_pair(text_a, text_b, quality_policy, languages=language_pair)
        unsafe = "control_characters" in assessment.rejection_reasons
        if not assessment.accepted and (filter_quality or unsafe):
            output.append(
                (
                    "quality_filtered",
                    (source_id, assessment.rejection_reasons, assessment.warning_reasons),
                )
            )
            continue
        encoded_a, encoded_b = protect_shared_spans(text_a, text_b)
        output.append(
            (
                "candidate",
                (
                    source_id,
                    text_a,
                    text_b,
                    tokenizer.encode(encoded_a),
                    tokenizer.encode(encoded_b),
                    infer_register(text_a, key_a),
                    infer_register(text_b, key_b),
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
    language_pair: tuple[str, str],
    batch_size: int = 512,
):
    for source_id, path in enumerate(paths):
        with path.open("rb") as handle:
            rows: list[bytes] = []
            for raw_line in handle:
                rows.append(raw_line)
                if len(rows) >= batch_size:
                    yield source_id, rows, quality_policy, filter_quality, language_pair
                    rows = []
            if rows:
                yield source_id, rows, quality_policy, filter_quality, language_pair


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
    train_only_prefixes: Sequence[str] = ("bt_",),
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

    quality_policy = quality_policy or QualityPolicy()
    quality_policy.validate()

    # Validate the model in the parent before starting a large worker pool.
    KJTokenizer(tokenizer_model)
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
            split: ShardWriter(staging_dir, split, shard_size, side_names=language_pair)
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
    inputs = _prepare_input_batches(paths, quality_policy, filter_quality, tuple(language_pair))
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

    key_a, key_b = language_pair
    try:
        for batch in processed_batches:
            for status, payload in batch:
                source_id = payload[0]
                source_stats = per_source_stats[source_id]
                targets = (stats, source_stats)
                for target in targets:
                    _increment(target, "physical_lines")

                if status in {
                    "invalid_utf8",
                    "invalid_json",
                    "invalid_record",
                    "non_string",
                    "missing_text",
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
                    ko,
                    ja,
                    ko_ids,
                    ja_ids,
                    ko_register,
                    ja_register,
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

                pair_key = f"{dedup_key(ko)}\0{dedup_key(ja)}".encode("utf-8")
                pair_digest = hashlib.sha256(pair_key).digest()[:16]
                if not digest_store.add_if_new(pair_digest):
                    for target in targets:
                        _increment(target, "duplicates")
                    continue

                if len(ko_ids) > max_tokens_per_side or len(ja_ids) > max_tokens_per_side:
                    for target in targets:
                        _increment(target, "too_long")
                    continue

                # Group all translations sharing a source text in one split.
                force_train_split = paths[source_id].name.startswith(tuple(train_only_prefixes))
                if force_train_split:
                    split = "train"
                else:
                    split = choose_split_for_key(dedup_key(ko), validation_fraction, test_fraction)
                if target_split_guard is not None:
                    target_digest = hashlib.sha256(dedup_key(ja).encode("utf-8")).digest()
                    if not target_split_guard.accept(split, target_digest):
                        for target in targets:
                            _increment(target, "split_conflicts")
                        continue
                writers[split].add(
                    ko_ids, ja_ids, ko_register, ja_register, source_id, quality_score
                )
                for target in targets:
                    _increment(target, split)
                    _increment(target, "valid_pairs")
                    _increment(target, "ko_tokens", len(ko_ids))
                    _increment(target, "ja_tokens", len(ja_ids))
                    _increment(target, "quality_score_sum", quality_score)
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
        "format": "kjx-indexed-parallel-v2",
        "language_pair": list(language_pair),
        "train_only_prefixes": list(train_only_prefixes),
        "tokenizer_model": str(Path(tokenizer_model).resolve()),
        "inputs": [str(path) for path in paths],
        "sources": [
            {
                "id": source_id,
                "name": path.name,
                "path": str(path),
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
