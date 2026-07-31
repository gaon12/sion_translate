#!/usr/bin/env python3
"""Report shards whose JSONL structure no shipped language pair can read.

A shard whose keys do not match any configured pair contributes nothing and says
nothing about it: ``iter_parallel_text`` simply yields no sentences for it. That
is silent data loss, and it is easy to cause - one shard in this corpus arrived
with the keys ``한국어``/``일본어`` instead of ``ko``/``ja`` and would have been
dropped whole.

The check is language-generic. It reads the configured pairs and inspects a
bounded sample from each file for a structurally valid parallel record. It does
not run quality filtering, split assignment, or duplicate detection: those
operations are expensive and can reject otherwise correctly keyed records.

Usage::

    python scripts/data/check_shard_keys.py                     # uses sion_translate.yaml
    python scripts/data/check_shard_keys.py --config other.yaml
    python scripts/data/check_shard_keys.py --pair ko ja --pair kj ko data/*.jsonl

Exit codes: 0 every shard is readable, 1 at least one yields nothing, 2 bad input.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Sequence

from sion_translate.data.records import expand_parallel_record, normalize_language_pairs


DEFAULT_SCAN_LINES = 2000


def observed_keys(path: Path, *, limit: int = DEFAULT_SCAN_LINES) -> list[str]:
    """The JSON keys this file actually uses, most common first."""

    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                counts.update(key for key, value in row.items() if isinstance(value, str))
    return [key for key, _ in counts.most_common()]


def inspect_shard(
    path: Path,
    *,
    pairs: Sequence[Sequence[str]],
    limit: int = DEFAULT_SCAN_LINES,
) -> tuple[bool, list[str], int]:
    """Return whether a bounded sample contains a configured parallel record."""

    if limit < 1:
        raise ValueError("scan line limit must be positive")
    counts: Counter[str] = Counter()
    scanned = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if scanned >= limit:
                break
            scanned += 1
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                counts.update(key for key, value in row.items() if isinstance(value, str))
            if expand_parallel_record(row, pairs).pairs:
                return True, [key for key, _ in counts.most_common()], scanned
    return False, [key for key, _ in counts.most_common()], scanned


def configured_pairs(
    config_path: Path | None,
    explicit: Sequence[Sequence[str]],
) -> tuple[tuple[str, str], ...]:
    if explicit:
        return normalize_language_pairs(language_pairs=explicit)
    from sion_translate.config import load_config

    path = config_path or Path("sion_translate.yaml")
    return normalize_language_pairs(
        language_pairs=load_config(path).data.configured_language_pairs()
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="JSONL files (default: data/*.jsonl)")
    parser.add_argument("--config", type=Path, help="config to read language pairs from")
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        default=[],
        metavar=("LANG_A", "LANG_B"),
        help="language pair, repeatable; overrides --config",
    )
    parser.add_argument(
        "--scan-lines",
        type=int,
        default=DEFAULT_SCAN_LINES,
        metavar="N",
        help=f"maximum physical lines inspected per shard (default: {DEFAULT_SCAN_LINES})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = args.inputs or sorted(Path("data").glob("*.jsonl"))
    if not paths:
        print("no JSONL files found", file=sys.stderr)
        return 2

    try:
        pairs = configured_pairs(args.config, args.pair)
    except (OSError, ValueError) as error:
        print(f"cannot read language pairs ({error})", file=sys.stderr)
        return 2

    languages = sorted({language for pair in pairs for language in pair})
    print(f"configured languages: {', '.join(languages)}")
    print(f"pairs: {', '.join('->'.join(pair) for pair in pairs)}")
    print()

    unreadable: list[tuple[Path, list[str], int]] = []
    for path in paths:
        try:
            readable, keys, scanned = inspect_shard(
                path,
                pairs=pairs,
                limit=args.scan_lines,
            )
        except (OSError, ValueError) as error:
            print(f"{path.name:36} cannot read ({error})", file=sys.stderr)
            return 2
        if not readable:
            unreadable.append((path, keys, scanned))
            print(
                f"  {path.name:36} no pair in {scanned:>6,} sampled line(s)"
                "   <-- structurally unreadable"
            )
        else:
            print(f"  {path.name:36} readable after {scanned:>6,} line(s)")

    print()
    if not unreadable:
        print(
            "every shard sample contains a structurally readable record "
            "(quality filtering was intentionally not run)."
        )
        return 0

    print()
    print(f"{len(unreadable)} shard sample(s) contain no configured pair:")
    for path, keys, scanned in unreadable:
        print(f"  {path}")
        print(f"      lines sampled: {scanned:,}")
        print(f"      keys present : {', '.join(keys) if keys else '(none)'}")
        print(f"      keys expected: {', '.join(languages)}")
    print()
    print(
        "Rename the keys in the JSONL to match a configured language. Adding the "
        "existing key as a language is usually not an option: a language key must "
        "be 1-16 ASCII alphanumeric characters starting with a letter, so a key "
        "like 한국어 can never be one. Leaving this as is drops the file silently."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
