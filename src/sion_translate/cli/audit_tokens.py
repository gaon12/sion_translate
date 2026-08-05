from __future__ import annotations

import argparse
import json
from pathlib import Path

from sion_translate.console import configure_stdio
from sion_translate.token_audit import audit_token_exposure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit byte fallback and decoder-target token exposure"
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL files or glob patterns")
    parser.add_argument("--tokenizer", required=True, help="SentencePiece .model path")
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
    )
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
