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
    parser.add_argument(
        "--refinement-evidence-fraction",
        type=float,
        default=0.0,
        help=(
            "Reserve a dedicated split for relative candidate-refinement evidence. "
            "This split never contributes to ordinary validation or test metrics."
        ),
    )
    parser.add_argument("--max-tokens-per-side", type=int, default=510)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of preprocessing processes (default: automatic)",
    )
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
    language_group = parser.add_mutually_exclusive_group(required=True)
    language_group.add_argument(
        "--language-pair",
        nargs=2,
        metavar=("LANG_A", "LANG_B"),
        help="One physical parallel language pair",
    )
    language_group.add_argument(
        "--language-pairs",
        nargs=2,
        action="append",
        metavar=("LANG_A", "LANG_B"),
        help="Repeat to include multiple physical parallel pairs in one dataset",
    )
    parser.add_argument(
        "--translation-direction",
        nargs=2,
        action="append",
        metavar=("SOURCE", "TARGET"),
        help=(
            "Repeat for each direction to train. When omitted, use the existing "
            "bidirectional/source-only policy."
        ),
    )
    parser.add_argument(
        "--approximate-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Assign splits by character 5-gram MinHash bucket so approximate "
            "duplicates cannot cross between train and holdout (default: enabled)"
        ),
    )
    parser.add_argument(
        "--exact-split",
        action="store_false",
        dest="approximate_split",
        help=(
            "Use exact-match splits to reproduce older experiments; approximate "
            "duplicates may leak across splits"
        ),
    )
    parser.add_argument(
        "--source-only-language",
        nargs="+",
        default=[],
        metavar="LANG",
        help=(
            "Language used only as source text and never as a translation target "
            "(for example, mixed Korean-Japanese '한본어' with tag kj). Pairs "
            "containing this language are trained in one direction only."
        ),
    )
    parser.add_argument(
        "--train-only-prefix",
        nargs="+",
        default=list(DEFAULT_TRAIN_ONLY_PREFIXES),
        metavar="PREFIX",
        help=(
            "Restrict input files with this prefix to the training split "
            f"(default: {' '.join(DEFAULT_TRAIN_ONLY_PREFIXES)}). This prevents "
            "synthetic data from inflating holdout scores."
        ),
    )
    parser.add_argument(
        "--source-only-synthetic-evidence-file",
        action="append",
        default=[],
        metavar="BASENAME",
        help=(
            "Exact synthetic input basename allowed to contribute one-way source-only "
            "rows to refinement evidence. Repeat explicitly for every reviewed file."
        ),
    )
    parser.add_argument(
        "--managed-augmentation-prefix",
        default="bt_",
        help=(
            "For JSONL files with this prefix, accept only shards owned by a "
            "sion-augment ledger (an empty string disables validation)"
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
            refinement_evidence_fraction=args.refinement_evidence_fraction,
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
            source_only_synthetic_evidence_files=(args.source_only_synthetic_evidence_file),
            managed_augmentation_prefix=args.managed_augmentation_prefix or None,
            num_workers=args.workers,
        )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
