#!/usr/bin/env python3
"""Count effective Korean/English/Japanese parallel pairs in JSONL shards.

A three-language record contributes one usable example to each of ko-ja,
ko-en, and en-ja.  The report keeps both exact language signatures and those
effective pair totals so corpus balancing decisions are explicit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable


LANGUAGES = ("ko", "en", "ja")
PAIRS = (("ko", "ja"), ("ko", "en"), ("en", "ja"))


def populated_languages(row: Any) -> tuple[str, ...]:
    if not isinstance(row, dict):
        return ()
    return tuple(
        language
        for language in LANGUAGES
        if isinstance(row.get(language), str) and row[language].strip()
    )


def audit(paths: Iterable[Path]) -> dict[str, Any]:
    effective: Counter[str] = Counter()
    signatures: Counter[str] = Counter()
    per_file: list[dict[str, Any]] = []
    total_rows = 0
    malformed_rows = 0

    for path in paths:
        file_effective: Counter[str] = Counter()
        file_signatures: Counter[str] = Counter()
        file_rows = 0
        file_malformed = 0
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                file_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    file_malformed += 1
                    continue
                languages = populated_languages(row)
                signature = "-".join(languages) if languages else "none"
                file_signatures[signature] += 1
                for first, second in PAIRS:
                    if first in languages and second in languages:
                        file_effective[f"{first}-{second}"] += 1

        total_rows += file_rows
        malformed_rows += file_malformed
        effective.update(file_effective)
        signatures.update(file_signatures)
        per_file.append(
            {
                "path": str(path),
                "rows": file_rows,
                "malformed_rows": file_malformed,
                "signatures": dict(sorted(file_signatures.items())),
                "effective_pairs": dict(sorted(file_effective.items())),
            }
        )

    baseline = effective["ko-ja"]
    ratios = {
        pair: (count / baseline if baseline else None) for pair, count in sorted(effective.items())
    }
    return {
        "total_rows": total_rows,
        "malformed_rows": malformed_rows,
        "signatures": dict(sorted(signatures.items())),
        "effective_pairs": dict(sorted(effective.items())),
        "relative_to_ko_ja": ratios,
        "files": per_file,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.inputs or sorted(Path("data").glob("*.jsonl"))
    report = audit(path.resolve() for path in paths)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
