"""Report number, sign, unit and script preservation for a translation file.

These are the defects chrF does not see. Scoring is against the source rather
than a reference, so this runs without gold translations and can gate
machine-translated data before it enters a corpus.

Input is JSONL with one object per line. Keys are configurable; the defaults
match both the translation-queue result format and a plain parallel shard::

    sion-check-preservation --target-scripts ja translated.jsonl
    sion-check-preservation --source-key kj --target-key ko \\
        --target-scripts ko data/synthetic_hanboneo.jsonl
    sion-check-preservation --json report.json --max-violation-rate 0.02 out.jsonl

Exit codes: 0 within thresholds, 1 a threshold was exceeded, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sion_translate.console import configure_stdio
from sion_translate.preservation import check_corpus, format_report
from sion_translate.scripts_registry import resolve_scripts

DEFAULT_SOURCE_KEYS = ("source", "ko", "src")
DEFAULT_TARGET_KEYS = ("translation", "hypothesis", "ja", "tgt")


def script_list(value: str) -> tuple[str, ...]:
    """Parse a comma-separated script/language list."""

    names = tuple(part.strip() for part in value.split(",") if part.strip())
    resolve_scripts(names)
    return names


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as handle:
        for number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{number} is not valid JSON ({error})") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            rows.append(row)
    return rows


def pick(row: dict[str, object], keys: tuple[str, ...], *, line: int, role: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"line {line} has no usable {role} field; tried {list(keys)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="JSONL files to check")
    parser.add_argument(
        "--source-key",
        action="append",
        default=None,
        metavar="KEY",
        help=f"source field, repeatable in priority order (default: {' '.join(DEFAULT_SOURCE_KEYS)})",
    )
    parser.add_argument(
        "--target-key",
        action="append",
        default=None,
        metavar="KEY",
        help=f"target field, repeatable (default: {' '.join(DEFAULT_TARGET_KEYS)})",
    )
    parser.add_argument(
        "--target-scripts",
        type=script_list,
        default=(),
        metavar="LIST",
        help=(
            "writing systems the target may use: script names or language "
            "shorthands, comma separated (ko / ja / kana,han). "
            "omit to skip the script check"
        ),
    )
    parser.add_argument("--examples", type=int, default=5, help="failing examples to print")
    parser.add_argument("--json", dest="json_out", help="write the full report here")
    parser.add_argument(
        "--max-violation-rate",
        type=float,
        help="fail when any check exceeds this fraction of sentences",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = build_parser().parse_args(argv)
    if args.examples < 0:
        print("--examples must be non-negative", file=sys.stderr)
        return 2
    if args.max_violation_rate is not None and not 0.0 <= args.max_violation_rate <= 1.0:
        print("--max-violation-rate must be in [0, 1]", file=sys.stderr)
        return 2

    source_keys = tuple(args.source_key or DEFAULT_SOURCE_KEYS)
    target_keys = tuple(args.target_key or DEFAULT_TARGET_KEYS)

    reports: list[dict[str, object]] = []
    failed = False
    for raw_path in args.paths:
        path = Path(raw_path)
        try:
            rows = read_rows(path)
            sources = [
                pick(row, source_keys, line=index, role="source")
                for index, row in enumerate(rows, start=1)
            ]
            targets = [
                pick(row, target_keys, line=index, role="target")
                for index, row in enumerate(rows, start=1)
            ]
            counts = check_corpus(
                sources,
                targets,
                target_scripts=args.target_scripts,
                examples=args.examples,
            )
        except (OSError, ValueError) as error:
            print(f"{raw_path}: cannot check ({error})", file=sys.stderr)
            return 2

        print(format_report(counts, title=path.name))
        payload = counts.to_dict()
        payload["path"] = str(path)
        reports.append(payload)

        if args.max_violation_rate is not None and counts.sentences:
            for name in ("number", "sign", "unit", "script"):
                rate = getattr(counts, f"{name}_violations") / counts.sentences
                if rate > args.max_violation_rate:
                    print(
                        f"    ! {name} violation rate {rate:.3f} > {args.max_violation_rate:.3f}",
                        file=sys.stderr,
                    )
                    failed = True

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
