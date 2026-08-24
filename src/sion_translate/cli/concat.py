"""Build multi-sentence training examples by joining unrelated translation pairs.

    sion-concat --input "data/*.jsonl" --output data/concat_multi.jsonl --count 200000

When the output filename starts with ``concat_``, the default
``sion-prepare-data --train-only-prefix`` policy restricts it to the training
split. This prevents synthetic examples from inflating holdout scores.
"""

# CLI registries and argparse namespaces expose dynamic callables.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path

from sion_translate.concat import (
    SEPARATORS,
    build_concatenations,
    read_records,
    write_concatenations,
)
from sion_translate.console import configure_stdio
from sion_translate.data.prepare import DEFAULT_TRAIN_ONLY_PREFIXES
from sion_translate.tokenizer import expand_inputs


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concatenate unrelated pairs into multi-sentence training examples"
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL files or glob patterns")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--count", type=_nonnegative_int, required=True, help="Number of examples to build"
    )
    parser.add_argument("--min-sentences", type=_positive_int, default=2)
    parser.add_argument("--max-sentences", type=_positive_int, default=4)
    parser.add_argument(
        "--separator",
        default="space",
        choices=sorted(SEPARATORS),
        help=(
            "space joins text with spaces to resemble normal input; seg marks "
            "boundaries explicitly with the <seg> control token"
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=_positive_int,
        default=480,
        help=(
            "Maximum characters per side (default: 480). Discard examples that "
            "a training shard would truncate."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        help=(
            "SentencePiece model path. When provided, --max-tokens is measured "
            "using exact token counts."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=None,
        help=(
            "Maximum tokens per side; requires --tokenizer and normally matches "
            "training max_tokens_per_side."
        ),
    )
    parser.add_argument(
        "--language-pair",
        nargs=2,
        required=True,
        metavar=("LANG_A", "LANG_B"),
        help="JSONL language keys (LANG_A LANG_B)",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    if args.max_tokens is not None and not args.tokenizer:
        raise SystemExit("--max-tokens requires --tokenizer.")

    count_tokens: Callable[[str], int] | None = None
    if args.tokenizer:
        from sion_translate.tokenizer import SionTokenizer

        tokenizer = SionTokenizer(args.tokenizer)

        def _count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        count_tokens = _count_tokens

    paths = expand_inputs(args.input)
    if not paths:
        raise SystemExit(f"Could not find any input JSONL files: {args.input}")

    pairs = list(read_records(paths, args.language_pair))
    if not pairs:
        raise SystemExit("No readable translation pairs were found.")

    examples, stats = build_concatenations(
        pairs,
        count=args.count,
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        separator=args.separator,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
        count_tokens=count_tokens,
        seed=args.seed,
    )
    written = write_concatenations(args.output, examples, args.language_pair)

    output = Path(args.output)
    if not output.name.startswith(DEFAULT_TRAIN_ONLY_PREFIXES):
        print(
            f"[sion] Warning: {output.name} does not start with "
            f"{' / '.join(DEFAULT_TRAIN_ONLY_PREFIXES)}. Synthetic examples may "
            "therefore enter validation/test and inflate holdout scores. Rename "
            "the file or update sion-prepare-data --train-only-prefix."
        )
    print(f"[sion] Wrote {written} examples to {output}")
    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
