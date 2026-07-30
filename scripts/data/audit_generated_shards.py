#!/usr/bin/env python3
"""Gate generated parallel shards on template collapse and semantic misalignment.

``QualityPolicy`` judges one pair at a time, so it can only see script ratios,
length ratios, control characters and character repetition. Generated corpora
fail in ways that are invisible at that granularity:

- a shard restates a handful of sentence frames thousands of times, so the row
  count overstates how much the model can learn from it;
- source and target clause pools are recombined independently, so every row
  looks well-formed while no row is actually a translation of its source;
- held-out rows are recombinations of training rows, so the split carries no
  information and the holdout score is free.

This tool measures those properties per shard and exits non-zero when a
threshold is violated, so it can run in front of ``sion-prepare-data``.

Usage::

    python scripts/data/audit_generated_shards.py data/data4*.jsonl
    python scripts/data/audit_generated_shards.py --json report.json data/data51.jsonl
    python scripts/data/audit_generated_shards.py --min-skeleton-ttr 0.3 data/data44.jsonl

Exit codes: 0 every shard passed, 1 a threshold was violated, 2 bad input.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Iterator

from sion_translate.data.quality import canonical_text
from sion_translate.splitting import choose_split_for_key, normalized_split_key


# A quoted span is the unit generated corpora vary while holding the frame
# fixed, so it has to be blanked out before frames can be compared.
_QUOTED = re.compile(r"[\"“‘'][^\"”’']{1,120}[\"”’']")
_DIGITS = re.compile(r"\d")
_HANGUL = re.compile(r"[가-힣ㄱ-ㅣ]")
_KANA = re.compile(r"[぀-ヿｦ-ﾟ]")
_HAN = re.compile(r"[一-鿿]")


@dataclass(frozen=True)
class Thresholds:
    """Minimum acceptable diversity and maximum acceptable contamination."""

    min_skeleton_ttr: float = 0.50
    min_quoted_ttr: float = 0.05
    max_duplicate_source: float = 0.10
    max_conflicting_source: float = 0.02
    max_foreign_script_target: float = 0.02
    max_near_duplicate_leak: float = 0.10

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, float) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a float in [0, 1]; got {value!r}")


@dataclass
class ShardReport:
    """Per-shard measurements plus the thresholds the shard violated."""

    path: str
    rows: int = 0
    unreadable_rows: int = 0
    distinct_sources: int = 0
    skeleton_count: int = 0
    skeleton_ttr: float = 0.0
    quoted_total: int = 0
    quoted_unique: int = 0
    quoted_ttr: float = 1.0
    duplicate_source: float = 0.0
    conflicting_source: float = 0.0
    max_targets_per_source: int = 0
    foreign_script_target: float = 0.0
    foreign_script_source: float = 0.0
    held_out_rows: int = 0
    near_duplicate_leak: float = 0.0
    top_skeletons: list[tuple[str, int]] = field(default_factory=list)
    top_quoted: list[tuple[str, int]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


def skeleton(text: str) -> str:
    """Blank quoted spans and digits so reused sentence frames collapse together."""

    return _DIGITS.sub("#", _QUOTED.sub("<Q>", text))


def iter_rows(path: Path, *, source_key: str, target_key: str) -> Iterator[tuple[str, str]]:
    """Yield canonicalized pairs, or ``("", "")`` for a row that cannot be read."""

    with path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                yield "", ""
                continue
            if not isinstance(row, dict):
                yield "", ""
                continue
            source = row.get(source_key)
            target = row.get(target_key)
            if not isinstance(source, str) or not isinstance(target, str):
                yield "", ""
                continue
            yield canonical_text(source), canonical_text(target)


def _foreign_script_probe(target_language: str):
    """Return the predicate for "this target contains the wrong script".

    Hardcoding "Hangul is foreign" only works when the target is Japanese. It
    reports every row as contaminated when auditing kj->ko, where the target is
    supposed to be Korean.
    """

    if target_language == "ja":
        return lambda text: bool(_HANGUL.search(text))
    if target_language == "ko":
        return lambda text: bool(_KANA.search(text) or _HAN.search(text))
    if target_language == "none":
        return lambda text: False
    raise ValueError(f"target_language must be ja, ko or none; got {target_language!r}")


def audit_shard(
    path: Path,
    thresholds: Thresholds | None = None,
    *,
    source_key: str = "ko",
    target_key: str = "ja",
    target_language: str = "ja",
    validation_fraction: float = 0.005,
    test_fraction: float = 0.005,
    examples: int = 3,
) -> ShardReport:
    """Measure one JSONL shard and record which thresholds it violates."""

    thresholds = thresholds or Thresholds()
    thresholds.validate()
    if examples < 0:
        raise ValueError("examples must be non-negative")
    is_foreign = _foreign_script_probe(target_language)
    if not path.is_file():
        raise FileNotFoundError(path)

    report = ShardReport(path=str(path))
    skeletons: Counter[str] = Counter()
    quoted: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    targets_by_source: defaultdict[str, set[str]] = defaultdict(set)
    train_skeletons: set[str] = set()
    held_out_skeletons: list[str] = []
    foreign_target = 0
    foreign_source = 0

    for source, target in iter_rows(path, source_key=source_key, target_key=target_key):
        report.rows += 1
        if not source or not target:
            report.unreadable_rows += 1
            continue
        sources[source] += 1
        targets_by_source[source].add(target)
        source_skeleton = skeleton(source)
        skeletons[source_skeleton] += 1
        for span in _QUOTED.findall(source):
            quoted[span] += 1
        # The target must be monolingual in its own language. A code-mixed
        # source is legitimate for 한본어, so that side is reported only.
        if is_foreign(target):
            foreign_target += 1
        if _KANA.search(source) or _HAN.search(source):
            foreign_source += 1
        split = choose_split_for_key(
            normalized_split_key(source),
            validation_fraction,
            test_fraction,
        )
        if split == "train":
            train_skeletons.add(source_skeleton)
        else:
            held_out_skeletons.append(source_skeleton)

    usable = report.rows - report.unreadable_rows
    if usable <= 0:
        report.violations.append(f"usable_rows {usable} < 1")
        return report

    report.distinct_sources = len(sources)
    report.skeleton_count = len(skeletons)
    report.skeleton_ttr = len(skeletons) / usable
    report.quoted_total = sum(quoted.values())
    report.quoted_unique = len(quoted)
    if report.quoted_total:
        report.quoted_ttr = report.quoted_unique / report.quoted_total
    report.duplicate_source = 1.0 - len(sources) / usable
    conflicting = sum(1 for targets in targets_by_source.values() if len(targets) > 1)
    report.conflicting_source = conflicting / len(targets_by_source)
    report.max_targets_per_source = max(len(targets) for targets in targets_by_source.values())
    report.foreign_script_target = foreign_target / usable
    report.foreign_script_source = foreign_source / usable
    report.held_out_rows = len(held_out_skeletons)
    if held_out_skeletons:
        leaked = sum(1 for value in held_out_skeletons if value in train_skeletons)
        report.near_duplicate_leak = leaked / len(held_out_skeletons)
    report.top_skeletons = [(text[:160], count) for text, count in skeletons.most_common(examples)]
    report.top_quoted = [(text[:120], count) for text, count in quoted.most_common(examples)]

    checks = (
        ("skeleton_ttr", report.skeleton_ttr, thresholds.min_skeleton_ttr, "<"),
        ("quoted_ttr", report.quoted_ttr, thresholds.min_quoted_ttr, "<"),
        ("duplicate_source", report.duplicate_source, thresholds.max_duplicate_source, ">"),
        (
            "conflicting_source",
            report.conflicting_source,
            thresholds.max_conflicting_source,
            ">",
        ),
        (
            "foreign_script_target",
            report.foreign_script_target,
            thresholds.max_foreign_script_target,
            ">",
        ),
        (
            "near_duplicate_leak",
            report.near_duplicate_leak,
            thresholds.max_near_duplicate_leak,
            ">",
        ),
    )
    for name, value, limit, direction in checks:
        if (direction == "<" and value < limit) or (direction == ">" and value > limit):
            report.violations.append(f"{name} {value:.3f} {direction} {limit:.3f}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate generated parallel shards on template collapse and misalignment."
    )
    parser.add_argument("paths", nargs="+", help="JSONL shards to audit")
    parser.add_argument("--json", dest="json_out", help="write the full report to this path")
    parser.add_argument("--source-key", default="ko", help="JSON key holding the source text")
    parser.add_argument("--target-key", default="ja", help="JSON key holding the target text")
    parser.add_argument(
        "--target-language",
        default="ja",
        choices=("ja", "ko", "none"),
        help="language the target is expected to be monolingual in",
    )
    parser.add_argument("--examples", type=int, default=3, help="worst offenders to print")
    parser.add_argument("--min-skeleton-ttr", type=float, default=0.50)
    parser.add_argument("--min-quoted-ttr", type=float, default=0.05)
    parser.add_argument("--max-duplicate-source", type=float, default=0.10)
    parser.add_argument("--max-conflicting-source", type=float, default=0.02)
    parser.add_argument("--max-foreign-script-target", type=float, default=0.02)
    parser.add_argument("--max-near-duplicate-leak", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = Thresholds(
        min_skeleton_ttr=args.min_skeleton_ttr,
        min_quoted_ttr=args.min_quoted_ttr,
        max_duplicate_source=args.max_duplicate_source,
        max_conflicting_source=args.max_conflicting_source,
        max_foreign_script_target=args.max_foreign_script_target,
        max_near_duplicate_leak=args.max_near_duplicate_leak,
    )
    try:
        thresholds.validate()
    except ValueError as error:
        print(f"invalid threshold: {error}", file=sys.stderr)
        return 2
    if args.examples < 0:
        print("--examples must be non-negative", file=sys.stderr)
        return 2

    reports: list[ShardReport] = []
    for raw_path in args.paths:
        try:
            reports.append(
                audit_shard(
                    Path(raw_path),
                    thresholds,
                    source_key=args.source_key,
                    target_key=args.target_key,
                    target_language=args.target_language,
                    examples=args.examples,
                )
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"{raw_path}: cannot audit ({error})", file=sys.stderr)
            return 2

    for report in reports:
        verdict = "PASS" if report.passed else "FAIL"
        print(
            f"{Path(report.path).name:26} rows={report.rows:>8,} "
            f"skelTTR={report.skeleton_ttr:.3f} quotTTR={report.quoted_ttr:.3f} "
            f"dupSrc={100 * report.duplicate_source:5.1f}% "
            f"leak={100 * report.near_duplicate_leak:5.1f}% {verdict}"
        )
        for violation in report.violations:
            print(f"    ! {violation}")
        for text, count in report.top_skeletons:
            if count > 1:
                print(f"      skeleton x{count:<6,} {text!r}")
        for text, count in report.top_quoted:
            if count > 1:
                print(f"      quoted   x{count:<6,} {text!r}")

    if args.json_out:
        payload = [asdict(report) for report in reports]
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0 if all(report.passed for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
