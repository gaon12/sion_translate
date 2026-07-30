#!/usr/bin/env python3
"""Report and remove rows that sexualise a minor.

The adult visual-novel shards are a legitimate translation domain, but the genre
routinely uses school settings, so the corpus is checked rather than assumed
clean. A row is removed only when a child marker and a sexual marker occur
together, pooled across both sides of the pair - see
:mod:`sion_translate.content_screen` for why the marker lists stop short of
高校生 / 고등학생.

The report names the markers that fired and the row identifiers, never the
matched sentence: the point is to get the rows out of the corpus, not to
reproduce them in a second file.

Usage::

    python scripts/data/screen_protected_content.py \
        --output data/data54.jsonl \
        --rejected-ids reports/data54-screened-ids.txt \
        --report reports/data54-screen.json \
        staging/data54.candidate.jsonl

Omit ``--output`` to report without writing a filtered file.

Exit codes: 0 clean or filtered, 2 bad input, 3 flagged rows found and no
``--output`` was given to remove them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from sion_translate.content_screen import CorpusScreenReport, known_languages, screen_pair


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            row.setdefault("_line", number)
            rows.append(row)
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = {key: value for key, value in row.items() if key != "_line"}
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def row_identifier(row: dict[str, Any], source_key: str) -> str:
    for key in ("resource_id", "family_id", "document_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return f"line:{row.get('_line', '?')}"


def screen_file(
    path: Path,
    *,
    source_key: str,
    target_key: str,
    source_language: str,
    target_language: str,
) -> tuple[CorpusScreenReport, list[dict[str, Any]], list[dict[str, Any]]]:
    report = CorpusScreenReport()
    child_counts: Counter[str] = Counter()
    sexual_counts: Counter[str] = Counter()
    age_counts: Counter[int] = Counter()
    kept: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []

    for row in read_rows(path):
        report.rows += 1
        source = row.get(source_key)
        target = row.get(target_key)
        if not isinstance(source, str) or not isinstance(target, str):
            kept.append(row)
            continue
        result = screen_pair(
            source,
            target,
            source_language=source_language,
            target_language=target_language,
        )
        if not result.flagged:
            kept.append(row)
            continue
        report.flagged += 1
        flagged.append(row)
        for marker in result.child_markers:
            child_counts[marker] += 1
        for marker in result.sexual_markers:
            sexual_counts[marker] += 1
        for age in result.ages:
            age_counts[age] += 1
        if len(report.flagged_row_ids) < 200:
            report.flagged_row_ids.append(row_identifier(row, source_key))

    report.child_marker_counts = dict(child_counts.most_common())
    report.sexual_marker_counts = dict(sexual_counts.most_common())
    report.age_counts = dict(sorted(age_counts.items()))
    return report, kept, flagged


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="write the rows that pass the screen here")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--rejected-ids", type=Path, help="write flagged row identifiers here")
    parser.add_argument("--source-key", default="ko")
    parser.add_argument("--target-key", default="ja")
    parser.add_argument(
        "--source-language",
        default="ko",
        help=f"marker table to use; configured: {', '.join(known_languages())}",
    )
    parser.add_argument("--target-language", default="ja")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.is_file():
        print(f"{args.input}: not a file", file=sys.stderr)
        return 2

    try:
        report, kept, flagged = screen_file(
            args.input,
            source_key=args.source_key,
            target_key=args.target_key,
            source_language=args.source_language,
            target_language=args.target_language,
        )
    except (OSError, ValueError) as error:
        print(f"{args.input}: cannot screen ({error})", file=sys.stderr)
        return 2

    print(
        f"{args.input.name:34} {report.rows:>8,} rows  "
        f"flagged {report.flagged:,} ({100.0 * report.flagged_rate:.4f}%)"
    )
    if report.child_marker_counts:
        print("  child markers that fired:")
        for marker, count in list(report.child_marker_counts.items())[:12]:
            print(f"    {count:6,}  {marker}")
    if report.age_counts:
        print(f"  ages that fired: {report.age_counts}")
    if report.sexual_marker_counts:
        print("  sexual markers that fired:")
        for marker, count in list(report.sexual_marker_counts.items())[:12]:
            print(f"    {count:6,}  {marker}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.rejected_ids and flagged:
        args.rejected_ids.parent.mkdir(parents=True, exist_ok=True)
        args.rejected_ids.write_text(
            "\n".join(row_identifier(row, args.source_key) for row in flagged) + "\n",
            encoding="utf-8",
        )

    if args.output:
        write_rows(args.output, kept)
        print(f"  wrote {len(kept):,} rows to {args.output}")
        return 0
    return 3 if report.flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
