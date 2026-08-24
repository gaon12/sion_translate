"""여러 번역 시스템의 JSONL 출력을 동일 조건으로 비교한다."""

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
    parser.add_argument("--cases", required=True, help="비교 문장 JSONL")
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        metavar="NAME=FILE",
        help="시스템 이름과 번역 출력 JSONL (여러 번 지정)",
    )
    parser.add_argument("--output", help="결과 경로 확장자 제외 (기본: reports/comparison-시각)")
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
            raise ValueError(f"--system 형식은 NAME=FILE 입니다: {spec}")
        identity = name.casefold()
        if identity in seen_identities:
            raise ValueError(f"중복 시스템 이름: {name}")
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
    print(f"저장: {output}.json / {output}.md")


if __name__ == "__main__":
    main()
