#!/usr/bin/env python3
"""Build a deterministic, source-balanced Korean-English training shard."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any


HANGUL_RE = re.compile(r"[가-힣]")
LATIN_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = SPACE_RE.sub(" ", value).strip()
    if len(text) < 2 or len(text) > 6_000:
        return None
    return text


def valid_pair(row: Any) -> tuple[str, str] | None:
    if not isinstance(row, dict):
        return None
    ko = clean_text(row.get("ko"))
    en = clean_text(row.get("en"))
    if ko is None or en is None or not HANGUL_RE.search(ko) or not LATIN_RE.search(en):
        return None
    if ko.casefold() == en.casefold():
        return None
    shorter = min(len(ko), len(en))
    longer = max(len(ko), len(en))
    if shorter < 4 and longer > 80:
        return None
    if longer / max(shorter, 1) > 8:
        return None
    return ko, en


def count_valid(path: Path) -> tuple[int, Counter[str]]:
    valid = 0
    rejected: Counter[str] = Counter()
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rejected["invalid_json"] += 1
                continue
            if valid_pair(row) is None:
                rejected["quality_filter"] += 1
                continue
            valid += 1
    return valid, rejected


def balanced_allocations(counts: list[int], target: int) -> list[int]:
    if target < 0:
        raise ValueError("target must be non-negative")
    if target > sum(counts):
        raise ValueError(f"target {target:,} exceeds {sum(counts):,} valid rows")
    low, high = 0, max(counts, default=0)
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(count, middle) for count in counts) <= target:
            low = middle
        else:
            high = middle - 1
    allocations = [min(count, low) for count in counts]
    remainder = target - sum(allocations)
    for index, count in enumerate(counts):
        if remainder == 0:
            break
        if allocations[index] < count:
            allocations[index] += 1
            remainder -= 1
    if remainder:
        raise AssertionError("allocation did not consume target")
    return allocations


def selected_valid_indexes(valid_count: int, allocation: int) -> set[int]:
    if allocation == valid_count:
        return set(range(valid_count))
    return {(offset * valid_count) // allocation for offset in range(allocation)}


def build(inputs: list[Path], output: Path, target: int) -> dict[str, Any]:
    scans = [count_valid(path) for path in inputs]
    counts = [valid for valid, _ in scans]
    allocations = balanced_allocations(counts, target)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    digest = sha256()
    output_bytes = 0
    written = 0
    per_source: list[dict[str, Any]] = []

    with temporary.open("wb") as destination:
        for path, (valid_count, rejected), allocation in zip(
            inputs, scans, allocations, strict=True
        ):
            selected = selected_valid_indexes(valid_count, allocation)
            valid_index = 0
            source_written = 0
            with path.open(encoding="utf-8-sig") as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pair = valid_pair(row)
                    if pair is None:
                        continue
                    if valid_index in selected:
                        encoded = (
                            json.dumps(
                                {"ko": pair[0], "en": pair[1]},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")
                        destination.write(encoded)
                        digest.update(encoded)
                        output_bytes += len(encoded)
                        written += 1
                        source_written += 1
                    valid_index += 1
            if source_written != allocation:
                raise AssertionError(
                    f"selection mismatch for {path}: {source_written} != {allocation}"
                )
            per_source.append(
                {
                    "input": str(path.resolve()),
                    "valid_rows": valid_count,
                    "selected_rows": source_written,
                    "rejected": dict(rejected),
                }
            )

    if written != target:
        raise AssertionError(f"output count mismatch: {written} != {target}")
    temporary.replace(output)
    report = {
        "languages": ["ko", "en"],
        "strategy": "equal-cap deterministic sampling across every registered source",
        "row_count": written,
        "output_bytes": output_bytes,
        "sha256": digest.hexdigest(),
        "output": str(output.resolve()),
        "sources": per_source,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/data60_koen.jsonl"))
    parser.add_argument("--target", type=int, default=8_530_672)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build([path.resolve() for path in args.inputs], args.output.resolve(), args.target)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
