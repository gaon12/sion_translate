# argparse namespaces and report serializers are runtime-shaped.
# pyright: reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sion_translate.console import configure_stdio
from sion_translate.data.monolingual import discover_monolingual_sources
from sion_translate.token_audit import (
    audit_indexed_token_exposure,
    audit_monolingual_token_exposure,
    audit_token_exposure,
    combine_target_exposure,
)


def configured_pairs(args) -> list[list[str]]:
    return args.language_pairs or [args.language_pair]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit byte fallback and decoder-target token exposure"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", nargs="+", help="JSONL files or glob patterns")
    source.add_argument(
        "--dataset",
        help="already indexed dataset root; counts stored decoder targets without re-tokenizing",
    )
    parser.add_argument("--tokenizer", required=True, help="SentencePiece .model path")
    parser.add_argument(
        "--split",
        default="train",
        help="indexed dataset split to scan with --dataset (default: train)",
    )
    parser.add_argument("--language-pair", nargs=2, default=["ko", "ja"])
    parser.add_argument("--language-pairs", nargs=2, action="append")
    parser.add_argument("--source-only-language", nargs="+", default=[])
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-physical-pairs",
        type=int,
        default=0,
        help="0 scans all accepted pairs; a positive value is a prefix preflight",
    )
    parser.add_argument("--rare-threshold", type=int, default=25)
    parser.add_argument("--max-piece-examples", type=int, default=50)
    parser.add_argument("--filter-quality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--monolingual-corpus",
        help=(
            "foundation 단일어 코퍼스 루트 (예: data/corpus). 지정하면 그 단계가 "
            "주는 디코더 타깃 노출을 함께 세고 두 단계를 합친 판정을 냅니다"
        ),
    )
    parser.add_argument(
        "--monolingual-max-lines",
        type=int,
        default=0,
        help="0 이면 전량 스캔, 양수는 언어별 prefix 표본 (빠른 preflight 용)",
    )
    parser.add_argument("--output", help="JSON report path (default: stdout)")
    parser.add_argument(
        "--fail-byte-rate",
        type=float,
        default=None,
        help="exit non-zero if any language exceeds this byte-fallback token rate",
    )
    parser.add_argument(
        "--fail-rare-pieces",
        type=int,
        default=None,
        help="exit non-zero if global rare target pieces exceed this count",
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    if args.fail_byte_rate is not None and not 0.0 <= args.fail_byte_rate <= 1.0:
        raise SystemExit("--fail-byte-rate must be in [0, 1]")
    if args.fail_rare_pieces is not None and args.fail_rare_pieces < 0:
        raise SystemExit("--fail-rare-pieces must be non-negative")
    if args.dataset:
        if args.max_physical_pairs:
            raise SystemExit("--max-physical-pairs applies only to raw --input scans")
        report = audit_indexed_token_exposure(
            args.dataset,
            args.tokenizer,
            split=args.split,
            bidirectional=args.bidirectional,
            rare_threshold=args.rare_threshold,
            max_piece_examples=args.max_piece_examples,
        )
    else:
        report = audit_token_exposure(
            args.input,
            args.tokenizer,
            language_pair=args.language_pair,
            language_pairs=args.language_pairs,
            source_only_languages=args.source_only_language,
            bidirectional=args.bidirectional,
            max_physical_pairs=args.max_physical_pairs,
            rare_threshold=args.rare_threshold,
            max_piece_examples=args.max_piece_examples,
            filter_quality=args.filter_quality,
            return_counts=bool(args.monolingual_corpus),
        )

    if args.monolingual_corpus:
        if args.dataset:
            raise SystemExit(
                "--monolingual-corpus 는 --input 스캔과 함께 씁니다 "
                "(--dataset 은 이미 색인된 번역 데이터셋입니다)"
            )
        languages = sorted({language for pair in configured_pairs(args) for language in pair})
        monolingual = audit_monolingual_token_exposure(
            discover_monolingual_sources(args.monolingual_corpus, languages),
            args.tokenizer,
            rare_threshold=args.rare_threshold,
            max_piece_examples=args.max_piece_examples,
            max_lines_per_language=args.monolingual_max_lines,
        )
        report["combined_stages"] = combine_target_exposure(
            report["global_target_counts"],
            monolingual["counts"],
            args.tokenizer,
            rare_threshold=args.rare_threshold,
            max_piece_examples=args.max_piece_examples,
        )
        # 원시 count 벡터는 어휘 크기의 정수 배열이라 보고서에 싣지 않습니다.
        monolingual.pop("counts", None)
        report["monolingual"] = monolingual
    report.pop("global_target_counts", None)

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    failures: list[str] = []
    if args.fail_byte_rate is not None:
        for language, values in report["languages"].items():
            if values["byte_fallback_rate"] > args.fail_byte_rate:
                failures.append(f"{language} byte_fallback_rate={values['byte_fallback_rate']:.8f}")
    rare = report["global_target_frequency"]["rare_observed_pieces"]
    if args.fail_rare_pieces is not None and rare > args.fail_rare_pieces:
        failures.append(f"rare_observed_pieces={rare}")
    if failures:
        raise SystemExit("token exposure audit failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
