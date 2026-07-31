#!/usr/bin/env python3
"""Report shards whose JSONL keys no shipped language pair can read.

A shard whose keys do not match any configured pair contributes nothing and says
nothing about it: ``iter_parallel_text`` simply yields no sentences for it. That
is silent data loss, and it is easy to cause - one shard in this corpus arrived
with the keys ``한국어``/``일본어`` instead of ``ko``/``ja`` and would have been
dropped whole.

The check is language-generic. It reads the configured pairs, asks how many
sentences each file actually yields, and reports the files that yield none along
with the keys they do carry, so the fix is obvious.

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

from sion_translate.tokenizer import iter_parallel_text


def observed_keys(path: Path, *, limit: int = 2000) -> list[str]:
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


def configured_pairs(config_path: Path | None, explicit: Sequence[Sequence[str]]) -> tuple:
    if explicit:
        return tuple(tuple(pair) for pair in explicit)
    from sion_translate.config import load_config

    path = config_path or Path("sion_translate.yaml")
    return load_config(path).data.configured_language_pairs()


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

    unreadable: list[tuple[Path, list[str]]] = []
    total = 0
    for path in paths:
        try:
            sentences = sum(
                1 for _ in iter_parallel_text([path], language_pairs=pairs, num_workers=1)
            )
        except (OSError, ValueError) as error:
            print(f"{path.name:36} cannot read ({error})", file=sys.stderr)
            return 2
        total += sentences
        if sentences == 0:
            keys = observed_keys(path)
            unreadable.append((path, keys))
            print(f"  {path.name:36} {sentences:>10,}   <-- yields nothing")
        else:
            print(f"  {path.name:36} {sentences:>10,}")

    print()
    print(f"total sentences: {total:,}")
    if not unreadable:
        print("every shard is readable with the configured pairs.")
        return 0

    print()
    print(f"{len(unreadable)} shard(s) contribute nothing:")
    for path, keys in unreadable:
        print(f"  {path}")
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
