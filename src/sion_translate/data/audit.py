from __future__ import annotations

import glob
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .quality import QualityPolicy, assess_pair, canonical_text, dedup_key


_QUALITY_REJECTION_REASONS = (
    "control_characters",
    "excessive_repetition",
    "identical_text",
    "ja_script_mismatch",
    "ko_script_mismatch",
    "length_ratio",
    "too_short",
)
_QUALITY_WARNING_REASONS = ("length_ratio", "structured_span_mismatch")


def _expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        candidate = Path(pattern)
        if any(character in pattern for character in "*?[]"):
            paths.update(Path(match) for match in glob.glob(pattern))
        elif candidate.is_dir():
            paths.update(candidate.glob("*.jsonl"))
        else:
            paths.add(candidate)
    resolved = sorted(path.resolve() for path in paths if path.is_file())
    if not resolved:
        raise FileNotFoundError(f"No JSONL files matched: {list(patterns)}")
    return resolved


class HyperLogLog:
    """Small, dependency-free HyperLogLog cardinality estimator."""

    def __init__(self, precision: int = 14):
        if not 4 <= precision <= 18:
            raise ValueError("hll_precision must be between 4 and 18")
        self.precision = precision
        self.register_count = 1 << precision
        self.registers = [0] * self.register_count

    def add_digest(self, digest: bytes) -> None:
        value = int.from_bytes(digest[:8], "big")
        remaining_bits = 64 - self.precision
        register = value >> remaining_bits
        suffix = value & ((1 << remaining_bits) - 1)
        rank = (
            remaining_bits - suffix.bit_length() + 1
            if suffix
            else remaining_bits + 1
        )
        if rank > self.registers[register]:
            self.registers[register] = rank

    def estimate(self) -> float:
        registers = self.register_count
        if registers == 16:
            alpha = 0.673
        elif registers == 32:
            alpha = 0.697
        elif registers == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1.0 + 1.079 / registers)
        raw = alpha * registers * registers / sum(
            2.0 ** (-value) for value in self.registers
        )
        empty = self.registers.count(0)
        if raw <= 2.5 * registers and empty:
            return registers * math.log(registers / empty)
        return raw


class _ExactUniqueTracker:
    def __init__(self, limit: int):
        if limit <= 0:
            raise ValueError("exact_unique_limit must be positive")
        self.limit = limit
        self.values: set[bytes] | None = set()

    def add(self, digest: bytes) -> None:
        if self.values is None or digest in self.values:
            return
        if len(self.values) >= self.limit:
            self.values = None
            return
        self.values.add(digest)

    @property
    def available(self) -> bool:
        return self.values is not None

    @property
    def count(self) -> int | None:
        return len(self.values) if self.values is not None else None


class _LengthReservoir:
    def __init__(self, size: int, seed: int):
        if size <= 0:
            raise ValueError("sample_size must be positive")
        self.size = size
        self.random = random.Random(seed)
        self.seen = 0
        self.values: list[tuple[int, int]] = []

    def add(self, ko_length: int, ja_length: int) -> None:
        self.seen += 1
        value = (ko_length, ja_length)
        if len(self.values) < self.size:
            self.values.append(value)
            return
        position = self.random.randrange(self.seen)
        if position < self.size:
            self.values[position] = value


def _nearest_rank(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _preview(value: Any, limit: int) -> str:
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            rendered = repr(value)
    # JSON can legally decode escaped lone surrogates, but UTF-8 files cannot
    # encode them. Preserve a safe escaped preview instead of crashing report IO.
    safe = rendered.encode("utf-8", errors="backslashreplace").decode("utf-8")
    return safe[:limit]


class _AuditAccumulator:
    def __init__(
        self,
        *,
        sample_size: int,
        seed: int,
        hll_precision: int,
        exact_unique_limit: int,
        max_issue_examples: int,
    ):
        self.bytes = 0
        self.rows = 0
        self.valid = 0
        self.invalid = 0
        self.missing = 0
        self.non_string = 0
        self.invalid_reasons: Counter[str] = Counter()
        self.signals: Counter[str] = Counter()
        self.warning_signals: Counter[str] = Counter()
        self.quality_pass = 0
        self.ko_total_chars = 0
        self.ja_total_chars = 0
        self.lengths = _LengthReservoir(sample_size, seed)
        self.hll = HyperLogLog(hll_precision)
        self.exact_unique = _ExactUniqueTracker(exact_unique_limit)
        self.max_issue_examples = max_issue_examples
        self.issue_examples: list[dict[str, Any]] = []

    def add_example(self, example: dict[str, Any]) -> None:
        if len(self.issue_examples) < self.max_issue_examples:
            self.issue_examples.append(example)

    def record_invalid(self, reason: str, example: dict[str, Any]) -> None:
        self.rows += 1
        self.invalid += 1
        self.invalid_reasons[reason] += 1
        self.add_example(example)

    def record_missing(self, example: dict[str, Any]) -> None:
        self.rows += 1
        self.missing += 1
        self.add_example(example)

    def record_non_string(self, example: dict[str, Any]) -> None:
        self.rows += 1
        self.non_string += 1
        self.add_example(example)

    def record_valid(
        self,
        ko: str,
        ja: str,
        pair_digest: bytes,
        rejection_reasons: list[str],
        warning_reasons: list[str],
        example: dict[str, Any],
    ) -> None:
        self.rows += 1
        self.valid += 1
        self.ko_total_chars += len(ko)
        self.ja_total_chars += len(ja)
        self.lengths.add(len(ko), len(ja))
        self.hll.add_digest(pair_digest)
        self.exact_unique.add(pair_digest)
        for issue in rejection_reasons:
            self.signals[issue] += 1
        for warning in warning_reasons:
            self.warning_signals[warning] += 1
        if rejection_reasons or warning_reasons:
            self.add_example(example)
        if not rejection_reasons:
            self.quality_pass += 1

    def _length_report(self, side: str) -> dict[str, Any]:
        if side == "ko":
            total = self.ko_total_chars
            sampled = [value[0] for value in self.lengths.values]
        elif side == "ja":
            total = self.ja_total_chars
            sampled = [value[1] for value in self.lengths.values]
        else:
            total = self.ko_total_chars + self.ja_total_chars
            sampled = [sum(value) for value in self.lengths.values]
        return {
            "count": self.valid,
            "total_chars": total,
            "mean_chars": round(total / self.valid, 4) if self.valid else 0.0,
            "sample_count": len(sampled),
            "sampled_percentiles_nearest_rank": {
                "p50": _nearest_rank(sampled, 0.50),
                "p95": _nearest_rank(sampled, 0.95),
                "p99": _nearest_rank(sampled, 0.99),
            },
        }

    def report(self) -> dict[str, Any]:
        estimate = max(0, round(self.hll.estimate()))
        signal_names = sorted(set(_QUALITY_REJECTION_REASONS) | self.signals.keys())
        warning_names = sorted(
            set(_QUALITY_WARNING_REASONS) | self.warning_signals.keys()
        )
        return {
            "bytes": self.bytes,
            "rows": self.rows,
            "valid": self.valid,
            "invalid": self.invalid,
            "missing": self.missing,
            "non_string": self.non_string,
            "invalid_breakdown": dict(sorted(self.invalid_reasons.items())),
            "character_lengths": {
                "ko": self._length_report("ko"),
                "ja": self._length_report("ja"),
                "pair": self._length_report("pair"),
            },
            "unique_pairs": {
                "exact_count": self.exact_unique.count,
                "exact_count_available": self.exact_unique.available,
                "exact_tracking_limit": self.exact_unique.limit,
                "hyperloglog_estimate": {
                    "label": "approximate_unique_pair_estimate",
                    "count": estimate,
                    "precision": self.hll.precision,
                    "registers": self.hll.register_count,
                    "relative_standard_error": round(
                        1.04 / math.sqrt(self.hll.register_count), 6
                    ),
                },
            },
            "signals": {name: self.signals[name] for name in signal_names},
            "warning_signals": {
                name: self.warning_signals[name] for name in warning_names
            },
            "quality_pass_count": self.quality_pass,
            "quality_pass_rate": _rate(self.quality_pass, self.valid),
            "quality_rate_denominator": "valid_rows",
            "issue_examples": self.issue_examples,
        }


def _stable_file_seed(seed: int, path: Path) -> int:
    digest = hashlib.sha256(f"{seed}\0{path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _pair_assessment(
    ko: str,
    ja: str,
    *,
    policy: QualityPolicy,
) -> tuple[list[str], list[str]]:
    assessment = assess_pair(ko, ja, policy)
    return list(assessment.rejection_reasons), list(assessment.warning_reasons)


def audit_dataset(
    input_patterns: Sequence[str],
    *,
    max_length_ratio: float = 5.0,
    sample_size: int = 100_000,
    seed: int = 20260711,
    hll_precision: int = 14,
    exact_unique_limit: int = 100_000,
    max_issue_examples: int = 5,
    issue_preview_chars: int = 160,
    min_chars_per_side: int = 2,
    min_language_fraction: float = 0.10,
    min_language_check_chars: int = 4,
) -> dict[str, Any]:
    """Audit raw parallel JSONL files with bounded working memory.

    Exact counters and character totals are streaming. Percentiles use reservoir
    samples, and unique-pair cardinality is explicitly reported as a HyperLogLog
    estimate once the bounded exact tracker reaches its configured limit.
    """

    if max_length_ratio <= 1.0:
        raise ValueError("max_length_ratio must be greater than 1")
    if max_issue_examples < 0:
        raise ValueError("max_issue_examples must be non-negative")
    if issue_preview_chars <= 0:
        raise ValueError("issue_preview_chars must be positive")
    quality_policy = QualityPolicy(
        min_chars_per_side=min_chars_per_side,
        max_length_ratio=max_length_ratio,
        min_language_fraction=min_language_fraction,
        min_language_check_chars=min_language_check_chars,
    )
    quality_policy.validate()

    paths = _expand_inputs(input_patterns)
    global_stats = _AuditAccumulator(
        sample_size=sample_size,
        seed=seed,
        hll_precision=hll_precision,
        exact_unique_limit=exact_unique_limit,
        max_issue_examples=max_issue_examples,
    )
    file_reports: list[dict[str, Any]] = []

    for path in paths:
        file_stats = _AuditAccumulator(
            sample_size=sample_size,
            seed=_stable_file_seed(seed, path),
            hll_precision=hll_precision,
            exact_unique_limit=exact_unique_limit,
            max_issue_examples=max_issue_examples,
        )
        file_bytes = path.stat().st_size
        file_stats.bytes = file_bytes
        global_stats.bytes += file_bytes

        with path.open("rb") as handle:
            for row_number, raw_line in enumerate(handle, start=1):
                base_example = {"source": str(path), "row": row_number}
                try:
                    encoding = "utf-8-sig" if row_number == 1 else "utf-8"
                    line = raw_line.decode(encoding)
                    record = json.loads(line)
                except UnicodeDecodeError:
                    example = {
                        **base_example,
                        "issues": ["invalid_utf8"],
                        "line_preview": raw_line[:issue_preview_chars].decode(
                            "utf-8", errors="replace"
                        ),
                    }
                    file_stats.record_invalid("invalid_utf8", example)
                    global_stats.record_invalid("invalid_utf8", example)
                    continue
                except json.JSONDecodeError:
                    example = {
                        **base_example,
                        "issues": ["invalid_json"],
                        "line_preview": line.strip()[:issue_preview_chars],
                    }
                    file_stats.record_invalid("invalid_json", example)
                    global_stats.record_invalid("invalid_json", example)
                    continue

                if not isinstance(record, dict):
                    example = {
                        **base_example,
                        "issues": ["invalid_record_type"],
                        "line_preview": _preview(record, issue_preview_chars),
                    }
                    file_stats.record_invalid("invalid_record_type", example)
                    global_stats.record_invalid("invalid_record_type", example)
                    continue

                if "ko" not in record or "ja" not in record:
                    example = {
                        **base_example,
                        "issues": ["missing_text"],
                        "ko_preview": _preview(record.get("ko"), issue_preview_chars),
                        "ja_preview": _preview(record.get("ja"), issue_preview_chars),
                    }
                    file_stats.record_missing(example)
                    global_stats.record_missing(example)
                    continue
                if not isinstance(record["ko"], str) or not isinstance(record["ja"], str):
                    example = {
                        **base_example,
                        "issues": ["non_string_text"],
                        "ko_preview": _preview(record["ko"], issue_preview_chars),
                        "ja_preview": _preview(record["ja"], issue_preview_chars),
                    }
                    file_stats.record_non_string(example)
                    global_stats.record_non_string(example)
                    continue

                ko = canonical_text(record["ko"])
                ja = canonical_text(record["ja"])
                if not ko or not ja:
                    example = {
                        **base_example,
                        "issues": ["missing_text"],
                        "ko_preview": _preview(ko, issue_preview_chars),
                        "ja_preview": _preview(ja, issue_preview_chars),
                    }
                    file_stats.record_missing(example)
                    global_stats.record_missing(example)
                    continue

                pair_key = f"{dedup_key(ko)}\0{dedup_key(ja)}".encode("utf-8")
                pair_digest = hashlib.sha256(pair_key).digest()[:16]
                rejection_reasons, warning_reasons = _pair_assessment(
                    ko,
                    ja,
                    policy=quality_policy,
                )
                example = {
                    **base_example,
                    "issues": rejection_reasons,
                    "warnings": warning_reasons,
                    "ko_preview": _preview(ko, issue_preview_chars),
                    "ja_preview": _preview(ja, issue_preview_chars),
                }
                file_stats.record_valid(
                    ko,
                    ja,
                    pair_digest,
                    rejection_reasons,
                    warning_reasons,
                    example,
                )
                global_stats.record_valid(
                    ko,
                    ja,
                    pair_digest,
                    rejection_reasons,
                    warning_reasons,
                    example,
                )

        report = {"source": str(path), **file_stats.report()}
        file_reports.append(report)

    global_report = global_stats.report()
    for report in file_reports:
        report["source_share"] = {
            "bytes": _rate(report["bytes"], global_report["bytes"]),
            "rows": _rate(report["rows"], global_report["rows"]),
            "valid": _rate(report["valid"], global_report["valid"]),
        }

    return {
        "schema": "sion-raw-dataset-audit-v1",
        "parameters": {
            "max_length_ratio": max_length_ratio,
            "sample_size": sample_size,
            "seed": seed,
            "hll_precision": hll_precision,
            "exact_unique_limit": exact_unique_limit,
            "max_issue_examples": max_issue_examples,
            "issue_preview_chars": issue_preview_chars,
            "quality_policy": quality_policy.to_dict(),
        },
        "global": {"file_count": len(paths), **global_report},
        "files": file_reports,
    }


__all__ = ["HyperLogLog", "audit_dataset"]
