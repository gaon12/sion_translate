from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sion_translate.console import configure_stdio
from sion_translate.data.audit import audit_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream raw parallel JSONL files and report data quality"
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL paths or globs")
    parser.add_argument("--output", help="Write the JSON report to this path")
    parser.add_argument(
        "--language-pair",
        nargs=2,
        action="append",
        required=True,
        metavar=("LANG_A", "LANG_B"),
        help="JSONL physical language pair to audit; repeat for multilingual corpora",
    )
    parser.add_argument("--max-ratio", type=float, default=5.0)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--hll-precision", type=int, default=14)
    parser.add_argument("--exact-unique-limit", type=int, default=100_000)
    parser.add_argument("--max-issue-examples", type=int, default=5)
    parser.add_argument("--min-chars-per-side", type=int, default=2)
    parser.add_argument("--min-language-fraction", type=float, default=0.10)
    parser.add_argument("--script-min-chars", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    configure_stdio()
    args = build_parser().parse_args(argv)
    report = audit_dataset(
        args.input,
        language_pairs=args.language_pair,
        max_length_ratio=args.max_ratio,
        sample_size=args.sample_size,
        seed=args.seed,
        hll_precision=args.hll_precision,
        exact_unique_limit=args.exact_unique_limit,
        max_issue_examples=args.max_issue_examples,
        min_chars_per_side=args.min_chars_per_side,
        min_language_fraction=args.min_language_fraction,
        min_language_check_chars=args.script_min_chars,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
