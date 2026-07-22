from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from kjx.data.prepare import prepare_dataset
from kjx.data.quality import QualityPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build indexed KJ-X training shards")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--validation-fraction", type=float, default=0.005)
    parser.add_argument("--test-fraction", type=float, default=0.005)
    parser.add_argument("--max-tokens-per-side", type=int, default=510)
    parser.add_argument("--workers", type=int, default=None, help="전처리 프로세스 수 (기본: 자동)")
    parser.add_argument("--min-chars-per-side", type=int, default=2)
    parser.add_argument("--max-length-ratio", type=float, default=5.0)
    parser.add_argument("--min-language-fraction", type=float, default=0.10)
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Record quality signals but keep pairs that fail conservative filters",
    )
    parser.add_argument(
        "--allow-target-leakage",
        action="store_true",
        help="Disable the bounded-memory target-side split leakage guard",
    )
    parser.add_argument(
        "--dedup-backend",
        choices=("sqlite", "memory"),
        default="sqlite",
        help="Use bounded-RAM exact dedup by default; memory is faster for tiny corpora",
    )
    parser.add_argument(
        "--language-pair",
        nargs=2,
        default=["ko", "ja"],
        metavar=("LANG_A", "LANG_B"),
        help="JSONL 키 이름 (기본: ko ja)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = prepare_dataset(
        args.input,
        args.tokenizer,
        args.output_dir,
        shard_size=args.shard_size,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        max_tokens_per_side=args.max_tokens_per_side,
        quality_policy=QualityPolicy(
            min_chars_per_side=args.min_chars_per_side,
            max_length_ratio=args.max_length_ratio,
            min_language_fraction=args.min_language_fraction,
        ),
        filter_quality=not args.no_quality_filter,
        prevent_target_leakage=not args.allow_target_leakage,
        dedup_backend=args.dedup_backend,
        language_pair=args.language_pair,
        num_workers=args.workers,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
