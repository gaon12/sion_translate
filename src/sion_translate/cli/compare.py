"""여러 번역 시스템의 JSONL 출력을 동일 조건으로 비교한다."""

from __future__ import annotations

import argparse
import time

from sion_translate.comparison import (
    load_comparison_cases,
    load_system_translations,
    save_comparison,
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


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    try:
        cases = load_comparison_cases(args.cases)
        systems: dict[str, dict[str, str]] = {}
        for spec in args.system:
            name, separator, path = spec.partition("=")
            if not separator or not name.strip() or not path.strip():
                raise ValueError(f"--system 형식은 NAME=FILE 입니다: {spec}")
            name = name.strip()
            if name in systems:
                raise ValueError(f"중복 시스템 이름: {name}")
            systems[name] = load_system_translations(path.strip(), cases)
        results = score_systems(cases, systems)
        output = args.output or f"reports/comparison-{time.strftime('%Y%m%d-%H%M%S')}"
        save_comparison(output, cases, systems, results)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(results_as_markdown(results))
    print(f"저장: {output}.json / {output}.md")


if __name__ == "__main__":
    main()
