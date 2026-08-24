"""Compare JSONL outputs from multiple translation systems consistently."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import time

from sion_translate.comparison import (
    category_results_as_markdown,
    load_comparison_cases,
    load_system_translations,
    save_comparison,
    score_system_categories,
    score_systems,
)
from sion_translate.console import configure_stdio
from sion_translate.evaluation import results_as_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare translation output JSONL files")
    parser.add_argument("--cases", required=True, help="Comparison cases in JSONL format")
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=FILE",
        help="System name and translation-output JSONL; may be specified repeatedly",
    )
    parser.add_argument(
        "--output",
        help="Result path without an extension (default: reports/comparison-<timestamp>)",
    )
    return parser


def parse_system_specs(specs: Sequence[str]) -> list[tuple[str, str]]:
    """Parse display labels while rejecting case-insensitive identity collisions."""

    parsed: list[tuple[str, str]] = []
    seen_identities: set[str] = set()
    for spec in specs:
        raw_name, separator, raw_path = spec.partition("=")
        name = raw_name.strip()
        path = raw_path.strip()
        if not separator or not name or not path:
            raise ValueError(f"--system must use NAME=FILE format: {spec}")
        identity = name.casefold()
        if identity in seen_identities:
            raise ValueError(f"Duplicate system name: {name}")
        seen_identities.add(identity)
        parsed.append((name, path))
    return parsed


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    try:
        system_specs = parse_system_specs(args.system)
        cases = load_comparison_cases(args.cases)
        systems: dict[str, dict[str, str]] = {}
        for name, path in system_specs:
            systems[name] = load_system_translations(path, cases)
        results = score_systems(cases, systems)
        category_results = score_system_categories(cases, systems)
        output = args.output or f"reports/comparison-{time.strftime('%Y%m%d-%H%M%S')}"
        save_comparison(
            output,
            cases,
            systems,
            results,
            category_results=category_results,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(results_as_markdown(results))
    print()
    print(category_results_as_markdown(category_results))
    print(f"Saved: {output}.json / {output}.md")


if __name__ == "__main__":
    main()
