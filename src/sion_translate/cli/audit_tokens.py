# argparse namespaces and report serializers are runtime-shaped.
# pyright: reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from sion_translate.console import configure_stdio
from sion_translate.data.monolingual import discover_monolingual_sources
from sion_translate.token_audit import (
    audit_indexed_token_exposure,
    audit_monolingual_token_exposure,
    audit_token_exposure,
    combine_target_exposure,
)


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
        default=None,
        help="indexed dataset split to scan with --dataset (default: train)",
    )
    parser.add_argument(
        "--language-pair",
        "--language-pairs",
        dest="language_pairs",
        nargs=2,
        action="append",
        metavar=("LANG_A", "LANG_B"),
        help=(
            "physical pair to scan; repeat for multiple pairs (derived from "
            "--translation-direction when omitted)"
        ),
    )
    parser.add_argument(
        "--translation-direction",
        nargs=2,
        action="append",
        metavar=("SOURCE", "TARGET"),
        help="authenticated ordered training edge; required and repeatable for raw --input",
    )
    parser.add_argument(
        "--source-only-language",
        nargs="+",
        default=None,
        help=(
            "deprecated raw-audit compatibility assertion; the resulting legacy policy must "
            "exactly match the explicit --translation-direction graph"
        ),
    )
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "legacy indexed override, or deprecated raw-audit assertion that must match the "
            "explicit direction graph"
        ),
    )
    parser.add_argument(
        "--train-only-prefix",
        action="append",
        default=None,
        metavar="PREFIX",
        help=(
            "additional raw-input filename prefix whose unscoped records fail closed as "
            "synthetic; repeat for every custom preparation prefix"
        ),
    )
    parser.add_argument(
        "--max-physical-pairs",
        type=int,
        default=0,
        help="0 scans all accepted pairs; a positive value is a prefix preflight",
    )
    parser.add_argument("--rare-threshold", type=int, default=25)
    parser.add_argument("--max-piece-examples", type=int, default=50)
    parser.add_argument(
        "--filter-quality",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="raw --input only (default: enabled)",
    )
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
        help=(
            "exit non-zero if target pieces below the threshold, including unused pieces, "
            "exceed this count"
        ),
    )
    return parser


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    if args.fail_byte_rate is not None and not 0.0 <= args.fail_byte_rate <= 1.0:
        raise SystemExit("--fail-byte-rate must be in [0, 1]")
    if args.fail_rare_pieces is not None and args.fail_rare_pieces < 0:
        raise SystemExit("--fail-rare-pieces must be non-negative")
    if args.monolingual_max_lines < 0:
        raise SystemExit("--monolingual-max-lines must be non-negative")
    if args.monolingual_max_lines and not args.monolingual_corpus:
        raise SystemExit("--monolingual-max-lines requires --monolingual-corpus")
    if args.dataset:
        if args.max_physical_pairs != 0:
            raise SystemExit("--max-physical-pairs applies only to raw --input scans")
        if args.filter_quality is not None:
            raise SystemExit("--filter-quality applies only to raw --input scans")
        if args.monolingual_corpus:
            raise SystemExit(
                "--monolingual-corpus 는 --input 스캔과 함께 씁니다 "
                "(--dataset 은 이미 색인된 번역 데이터셋입니다)"
            )
        if (
            args.language_pairs
            or args.translation_direction
            or args.source_only_language is not None
            or args.train_only_prefix is not None
        ):
            raise SystemExit(
                "--language-pair, --translation-direction, --source-only-language, and "
                "--train-only-prefix apply only to raw --input scans"
            )
        report = audit_indexed_token_exposure(
            args.dataset,
            args.tokenizer,
            split=args.split or "train",
            bidirectional=args.bidirectional,
            rare_threshold=args.rare_threshold,
            max_piece_examples=args.max_piece_examples,
        )
    else:
        if args.split is not None:
            raise SystemExit("--split applies only to indexed --dataset scans")
        if not args.translation_direction:
            raise SystemExit(
                "raw --input scans require at least one --translation-direction SOURCE TARGET"
            )
        report = audit_token_exposure(
            args.input,
            args.tokenizer,
            language_pairs=args.language_pairs,
            translation_directions=args.translation_direction,
            source_only_languages=args.source_only_language,
            bidirectional=args.bidirectional,
            train_only_prefixes=args.train_only_prefix or (),
            max_physical_pairs=args.max_physical_pairs,
            rare_threshold=args.rare_threshold,
            max_piece_examples=args.max_piece_examples,
            filter_quality=True if args.filter_quality is None else args.filter_quality,
            return_counts=bool(args.monolingual_corpus),
        )

    if args.monolingual_corpus:
        languages = sorted(report["languages"])
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
        _atomic_write_text(Path(args.output), rendered)
    else:
        print(rendered, end="")

    failures: list[str] = []
    if report["physical_pairs"] == 0:
        failures.append("physical_pairs=0")
    empty_directions = sorted(
        direction for direction, values in report["directions"].items() if values["examples"] == 0
    )
    if empty_directions:
        failures.append("zero-example-directions=" + ",".join(empty_directions))
    if args.fail_byte_rate is not None:
        for language, values in report["languages"].items():
            if values["byte_fallback_rate"] > args.fail_byte_rate:
                failures.append(f"{language} byte_fallback_rate={values['byte_fallback_rate']:.8f}")
    below_threshold = (
        report["combined_stages"]["still_below_threshold"]
        if "combined_stages" in report
        else report["global_target_frequency"]["below_threshold_pieces"]
    )
    if args.fail_rare_pieces is not None and below_threshold > args.fail_rare_pieces:
        failures.append(f"below_threshold_pieces={below_threshold}")
    if failures:
        raise SystemExit("token exposure audit failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
