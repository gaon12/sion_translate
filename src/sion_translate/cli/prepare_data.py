from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from sion_translate.console import configure_stdio
from sion_translate.data.prepare import DEFAULT_TRAIN_ONLY_PREFIXES, prepare_dataset
from sion_translate.data.quality import QualityPolicy
from sion_translate.locking import artifact_locks


def _input_lock_roots(inputs: list[str]) -> tuple[Path, ...]:
    roots: set[Path] = set()
    for raw_input in inputs:
        path = Path(raw_input)
        if any(character in raw_input for character in "*?[]"):
            roots.add(path.resolve().parent)
        elif path.is_dir():
            roots.add(path.resolve())
        else:
            roots.add(path.resolve().parent)
    return tuple(roots)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build indexed sion_translate training shards")
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
    parser.add_argument(
        "--language-pairs",
        nargs=2,
        action="append",
        metavar=("LANG_A", "LANG_B"),
        help="여러 언어쌍을 한 데이터셋에 넣을 때 반복 지정",
    )
    parser.add_argument(
        "--translation-direction",
        nargs=2,
        action="append",
        metavar=("SOURCE", "TARGET"),
        help="실제로 학습할 방향을 반복 지정 (생략하면 기존 양방향/source-only 정책)",
    )
    parser.add_argument(
        "--approximate-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "split 배정을 문자 5-gram MinHash 버킷으로 수행해 근사 중복이 "
            "train 과 holdout 을 넘나들지 못하게 합니다 (기본: 활성)"
        ),
    )
    parser.add_argument(
        "--exact-split",
        action="store_false",
        dest="approximate_split",
        help="과거 실험 재현용 완전일치 split (근사 중복 누수 위험)",
    )
    parser.add_argument(
        "--source-only-language",
        nargs="+",
        default=[],
        metavar="LANG",
        help=(
            "원문으로만 쓰고 번역 결과로는 내보내지 않는 언어 "
            "(예: 한본어 kj). 이 언어가 든 쌍은 단방향으로만 학습됩니다"
        ),
    )
    parser.add_argument(
        "--train-only-prefix",
        nargs="+",
        default=list(DEFAULT_TRAIN_ONLY_PREFIXES),
        metavar="PREFIX",
        help=(
            "이 접두어로 시작하는 입력 파일은 train split 에만 넣습니다 "
            f"(기본: {' '.join(DEFAULT_TRAIN_ONLY_PREFIXES)}). 합성 데이터로 "
            "holdout 점수가 올라가는 것을 막습니다"
        ),
    )
    parser.add_argument(
        "--managed-augmentation-prefix",
        default="bt_",
        help=(
            "이 접두사의 JSONL은 sion-augment ledger가 소유한 shard만 허용합니다 "
            "(빈 문자열이면 검증 비활성)"
        ),
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    with artifact_locks(
        (
            *_input_lock_roots(args.input),
            Path(args.tokenizer).resolve().parent,
            Path(args.output_dir).resolve().parent,
        )
    ):
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
            approximate_split=args.approximate_split,
            dedup_backend=args.dedup_backend,
            language_pair=args.language_pair,
            language_pairs=args.language_pairs,
            translation_directions=args.translation_direction,
            source_only_languages=args.source_only_language,
            train_only_prefixes=args.train_only_prefix,
            managed_augmentation_prefix=args.managed_augmentation_prefix or None,
            num_workers=args.workers,
        )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
