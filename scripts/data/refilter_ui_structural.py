#!/usr/bin/env python3
"""Re-apply deterministic UI-pair invariants to an existing JSONL file."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from rebuild_verified_parallel import _ui_pair_rejection


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_file(input_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    input_sha256 = sha256_file(input_path)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    input_rows = 0
    written_rows = 0
    rejected: Counter[str] = Counter()
    rejected_examples: list[dict[str, Any]] = []

    with (
        input_path.open("r", encoding="utf-8-sig") as source,
        temporary.open("w", encoding="utf-8", newline="\n") as destination,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            input_rows += 1
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected object")
            ko, ja = row.get("ko"), row.get("ja")
            if not isinstance(ko, str) or not isinstance(ja, str):
                raise ValueError(f"Line {line_number}: missing ko/ja strings")
            reason = _ui_pair_rejection(ko, ja, str(row.get("resource_id", "")))
            if reason:
                rejected[reason] += 1
                if len(rejected_examples) < 100:
                    rejected_examples.append(
                        {
                            "line": line_number,
                            "reason": reason,
                            "resource_id": row.get("resource_id"),
                            "ko": ko,
                            "ja": ja,
                        }
                    )
                continue
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written_rows += 1

    temporary.replace(output_path)
    report: dict[str, Any] = {
        "schema": "sion-ui-structural-refilter-v1",
        "input": str(input_path),
        "input_rows": input_rows,
        "input_sha256": input_sha256,
        "output": str(output_path),
        "written_rows": written_rows,
        "removed_rows": input_rows - written_rows,
        "output_sha256": sha256_file(output_path),
        "rejected_by_reason": dict(rejected),
        "rejected_examples": rejected_examples,
        "checks": [
            "expected Korean and Japanese scripts",
            "matching placeholders including Fluent term references",
            "matching ASCII numeric tokens",
            "length and access-key guards",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = filter_file(
        args.input.resolve(),
        args.output.resolve(),
        args.report.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
