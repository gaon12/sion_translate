from __future__ import annotations

import argparse
from pathlib import Path

from sion_translate.console import configure_stdio
from sion_translate.locking import artifact_locks
from sion_translate.synthetic import DEFAULT_SYNTHETIC_PREFIXES
from sion_translate.tokenizer import (
    DEFAULT_TOKENIZER_INPUT_SENTENCE_SIZE,
    DEFAULT_TOKENIZER_SAMPLING_ALPHA,
    train_tokenizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the from-scratch sion_translate tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="JSONL files or glob patterns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=48000)
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=DEFAULT_TOKENIZER_INPUT_SENTENCE_SIZE,
        help=(
            "Maximum sentences exposed to SentencePiece after deterministic "
            f"language-stratified sampling (default: {DEFAULT_TOKENIZER_INPUT_SENTENCE_SIZE:,}). "
            "Use 0 only on a host provisioned to process the full corpus."
        ),
    )
    parser.add_argument(
        "--sampling-alpha",
        type=float,
        default=DEFAULT_TOKENIZER_SAMPLING_ALPHA,
        help=(
            "Language sampling exponent in (0, 1]. Lower values give relatively "
            "more sample capacity to low-resource languages."
        ),
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=0,
        help="Seed that rotates deterministic per-language systematic samples.",
    )
    parser.add_argument(
        "--required-character-min-occurrences",
        type=int,
        default=25,
        help=(
            "Reserve characters observed at least this many times to avoid byte "
            "fallback. Use 0 to disable the frequency floor."
        ),
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9999,
        help=(
            "Fraction of corpus character frequency covered directly by the "
            "vocabulary. The default leaves a meaningful byte-fallback tail."
        ),
    )
    parser.add_argument("--seed-sentencepiece-size", type=int, default=1_000_000)
    parser.add_argument(
        "--workers", type=int, default=None, help="Preprocessing process count (default: auto)"
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="SentencePiece thread count (default: auto)"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.005)
    parser.add_argument("--test-fraction", type=float, default=0.005)
    language_group = parser.add_mutually_exclusive_group(required=True)
    language_group.add_argument(
        "--language-pair",
        nargs=2,
        metavar=("LANG_A", "LANG_B"),
        help="One physical parallel language pair.",
    )
    language_group.add_argument(
        "--language-pairs",
        nargs=2,
        action="append",
        metavar=("LANG_A", "LANG_B"),
        help="Repeat this option for every physical parallel language pair.",
    )
    parser.add_argument(
        "--translation-direction",
        nargs=2,
        action="append",
        metavar=("SOURCE", "TARGET"),
        help="Repeat for every trained source-to-target direction.",
    )
    parser.add_argument(
        "--approximate-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same approximate MinHash split policy as dataset preparation.",
    )
    parser.add_argument(
        "--exact-split",
        action="store_false",
        dest="approximate_split",
        help="Use exact endpoint matching for legacy experiment reproduction.",
    )
    parser.add_argument(
        "--source-only-language",
        nargs="+",
        default=[],
        metavar="LANG",
        help="Languages used only as sources; must match dataset preparation.",
    )
    parser.add_argument(
        "--train-only-prefix",
        nargs="+",
        default=list(DEFAULT_SYNTHETIC_PREFIXES),
        metavar="PREFIX",
        help="Prefixes that identify synthetic files restricted to the train split.",
    )
    parser.add_argument(
        "--no-split-digits",
        dest="split_digits",
        action="store_false",
        help=(
            "Allow multi-digit token pieces (not recommended). This increases the "
            "risk of changing amounts, dates, and measurements during translation."
        ),
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    with artifact_locks((Path(args.output_dir),)):
        model_path = train_tokenizer(
            args.input,
            args.output_dir,
            vocab_size=args.vocab_size,
            input_sentence_size=args.input_sentence_size,
            sampling_alpha=args.sampling_alpha,
            sampling_seed=args.sampling_seed,
            character_coverage=args.character_coverage,
            required_character_min_occurrences=args.required_character_min_occurrences,
            seed_sentencepiece_size=args.seed_sentencepiece_size,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            language_pair=args.language_pair,
            language_pairs=args.language_pairs,
            translation_directions=args.translation_direction,
            approximate_split=args.approximate_split,
            source_only_languages=args.source_only_language,
            train_only_prefixes=args.train_only_prefix,
            num_workers=args.workers,
            num_threads=args.threads,
            split_digits=args.split_digits,
        )
    print(model_path)


if __name__ == "__main__":
    main()
