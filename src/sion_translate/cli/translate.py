"""Translate in either trained direction with an exported model.

    sion-translate --to ja "안녕하세요"        # Korean → Japanese
    sion-translate --to ko "こんにちは"         # Japanese → Korean
    cat input.txt | sion-translate --to ja     # File/pipe input, one sentence per line

When no model is specified, the command discovers it in a runs/.../exports
directory, preferring the best EMA weights because they usually offer the best
quality. Language pairs are read from model provenance, so an en-de model can be
selected with options such as --to de.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from sion_translate.config import config_from_raw, load_raw_config
from sion_translate.console import configure_stdio
from sion_translate.generation import (
    DEFAULT_LENGTH_PENALTY,
    DEFAULT_MAX_OUTPUT_LENGTH_RATIO,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_NUM_BEAMS,
)
from sion_translate.glossary import load_glossary
from sion_translate.inference import Translator, find_exported_model
from sion_translate.iterative import refine_batch, summarize
from sion_translate.language_tags import (
    LanguageTagError,
    canonicalize_language_pair,
    canonicalize_language_tag,
)
from sion_translate.rerank import STRATEGIES as RERANK_STRATEGIES

DEFAULT_CONFIG_FILE = "sion_translate.yaml"


def _canonical_cli_language(value: str | None, *, option: str) -> str | None:
    if value is None:
        return None
    try:
        return canonicalize_language_tag(value, field=option)
    except LanguageTagError as error:
        raise SystemExit(str(error)) from error


def _canonical_model_directions(
    trained_directions: Sequence[Sequence[str]],
) -> tuple[tuple[str, str], ...]:
    directions: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_direction in enumerate(trained_directions):
        try:
            direction = canonicalize_language_pair(
                raw_direction,
                field=f"model translation_directions[{index}]",
            )
        except LanguageTagError as error:
            raise SystemExit(str(error)) from error
        if direction in seen:
            raise SystemExit(
                "Model translation_directions contains a duplicate after BCP 47 "
                f"normalization: {direction[0]}→{direction[1]}"
            )
        seen.add(direction)
        directions.append(direction)
    return tuple(directions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate with a trained sion_translate model")
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to translate; when omitted, read one sentence per line from stdin",
    )
    parser.add_argument(
        "--from",
        dest="source",
        help="Source language; required for multilingual models when ambiguous",
    )
    parser.add_argument(
        "--to",
        dest="target",
        help="Target language; may be omitted only when exactly one trained direction exists",
    )
    parser.add_argument("--model", help="Exported model path (default: discover from exports)")
    parser.add_argument(
        "--int8",
        action="store_true",
        help=(
            "Use the CPU-only INT8 model to reduce file size and memory use; "
            "this does not improve speed"
        ),
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=DEFAULT_NUM_BEAMS,
        help="number of beams; use 1 for greedy decoding (default: 4)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=0,
        help=(
            "Number of stochastic candidates in addition to the beam result "
            "(0 disables reranking). This can improve quality by spending more "
            "inference compute without retraining."
        ),
    )
    parser.add_argument(
        "--rerank",
        default="mbr+qe",
        choices=RERANK_STRATEGIES,
        help="Candidate-selection strategy (default: mbr+qe); used only with --candidates >= 1",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Candidate sampling temperature (default: 0.3; outperformed 0.7 on holdout)",
    )
    parser.add_argument(
        "--top-k", type=int, default=0, help="Top-k candidate sampling limit (0 is unlimited)"
    )
    parser.add_argument(
        "--revise-rounds",
        type=int,
        default=0,
        help=(
            "Maximum draft-revision rounds (0 disables revision). Translate easy "
            "sentences once and revise only results below the acceptance threshold. "
            "This is useful only for a model trained with sion-revise-data output."
        ),
    )
    parser.add_argument(
        "--accept-score",
        type=float,
        default=0.95,
        help="Do not revise when the QE score reaches this value (default: 0.95)",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.01,
        help="Stop when one round improves QE by less than this value (default: 0.01)",
    )
    parser.add_argument("--length-penalty", type=float, default=DEFAULT_LENGTH_PENALTY)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--reasoning-level",
        type=int,
        choices=range(10),
        default=None,
        metavar="0-9",
        help=(
            "Internal verification/candidate-refinement endpoint (0 translates "
            "directly; 1-9 selects a monotonic stage of the configured iterations, "
            "and all 1-9 values are equivalent with one iteration). When omitted, "
            "use the checkpoint default shared by training and validation."
        ),
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=DEFAULT_NO_REPEAT_NGRAM_SIZE,
        help="forbid repeated n-grams of this size; use 0 to disable (default: 4)",
    )
    parser.add_argument(
        "--max-output-length-ratio",
        type=float,
        default=DEFAULT_MAX_OUTPUT_LENGTH_RATIO,
        help="maximum output/source token ratio, plus a separate 16-token margin",
    )
    parser.add_argument(
        "--glossary",
        help=(
            "Glossary JSON path used to force configured term mappings "
            "(default: configuration data.glossary)"
        ),
    )
    parser.add_argument(
        "--no-glossary",
        action="store_true",
        help="Disable the glossary even when one is configured",
    )
    parser.add_argument("--config", help=f"Configuration file (default: {DEFAULT_CONFIG_FILE})")
    return parser


def resolve_translation_target(
    requested: str | None,
    source_language: str | None,
    trained_directions: Sequence[Sequence[str]],
) -> str:
    """Compatibility wrapper returning the target of an authenticated direction."""

    return resolve_translation_direction(
        requested,
        source_language,
        trained_directions,
    )[1]


def resolve_translation_direction(
    requested_target: str | None,
    requested_source: str | None,
    trained_directions: Sequence[Sequence[str]],
) -> tuple[str, str]:
    """Resolve both endpoints, including a uniquely implied missing endpoint."""

    directions = _canonical_model_directions(trained_directions)
    if not directions:
        raise SystemExit("The model has no authenticated translation_directions.")
    source = _canonical_cli_language(requested_source, option="--from")
    target = _canonical_cli_language(requested_target, option="--to")
    supported = ", ".join(f"{edge_source}→{edge_target}" for edge_source, edge_target in directions)

    if source is None and target is None:
        if len(directions) != 1:
            raise SystemExit(
                "The model has multiple trained directions. Specify --from LANG "
                f"or --to LANG (supported: {supported})"
            )
        return directions[0]
    if source is not None and target is not None:
        if (source, target) in set(directions):
            return source, target
        raise SystemExit(f"{source}→{target} is not a trained direction (supported: {supported})")
    if source is not None:
        outgoing = [direction for direction in directions if direction[0] == source]
        if not outgoing:
            raise SystemExit(
                f"No trained direction starts from --from {source} (supported: {supported})"
            )
        if len(outgoing) > 1:
            choices = ", ".join(
                f"{edge_source}→{edge_target}" for edge_source, edge_target in outgoing
            )
            raise SystemExit(
                f"--from {source} has multiple possible targets. Specify --to LANG "
                f"(supported: {choices})"
            )
        return outgoing[0]

    assert target is not None
    incoming = [direction for direction in directions if direction[1] == target]
    if not incoming:
        raise SystemExit(f"--to {target} is not a trained target (supported: {supported})")
    if len(incoming) > 1:
        choices = ", ".join(f"{edge_source}→{edge_target}" for edge_source, edge_target in incoming)
        raise SystemExit(
            f"--to {target} has multiple possible sources. Specify --from LANG "
            f"(supported: {choices})"
        )
    return incoming[0]


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    # Read the tokenizer location and output directory from the configuration.
    config_path = args.config or (
        DEFAULT_CONFIG_FILE if Path(DEFAULT_CONFIG_FILE).exists() else None
    )
    config = config_from_raw(load_raw_config(config_path) if config_path else {})

    model_path = args.model or find_exported_model(config.training.output_dir, int8=args.int8)
    translator = Translator(model_path, config.data.tokenizer_model)

    source_language, target = resolve_translation_direction(
        args.target,
        args.source,
        translator.translation_directions,
    )

    # Glossary precedence: --glossary, then data.glossary; --no-glossary disables both.
    glossary = None
    glossary_path = None if args.no_glossary else (args.glossary or config.data.glossary)
    if glossary_path:
        glossary = load_glossary(glossary_path)
        print(
            f"[sion] Applying glossary: {glossary_path} ({len(glossary)} terms)",
            file=sys.stderr,
            flush=True,
        )

    lines = args.text if args.text else [line.rstrip("\n") for line in sys.stdin]
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise SystemExit("No text was provided for translation.")

    print(
        f"[sion] Model: {model_path}; translating to {target}",
        file=sys.stderr,
        flush=True,
    )
    if args.candidates > 0:
        print(
            f"[sion] Reranking {args.candidates + 1} candidates (1 beam + "
            f"{args.candidates} samples) with {args.rerank}",
            file=sys.stderr,
            flush=True,
        )
    translations = translator.translate(
        lines,
        source_language=source_language,
        target_language=target,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        max_new_tokens=args.max_new_tokens,
        glossary=glossary,
        num_candidates=args.candidates,
        rerank=args.rerank,
        temperature=args.temperature,
        top_k=args.top_k,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_output_length_ratio=args.max_output_length_ratio,
        reasoning_level=args.reasoning_level,
    )

    if args.revise_rounds > 0:
        if translator.tokenizer.draft_id is None:
            raise SystemExit(
                "This tokenizer has no <draft> control token, so --revise-rounds "
                "cannot be used. Retrain the tokenizer with sion-train-tokenizer "
                "and include data produced by sion-revise-data in training."
            )

        def revise_batch(sources: Sequence[str], drafts: Sequence[str]) -> list[str]:
            return translator.revise(
                sources,
                drafts,
                source_language=source_language,
                target_language=target,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                max_new_tokens=args.max_new_tokens,
                reasoning_level=args.reasoning_level,
            )

        results = refine_batch(
            lines,
            translations,
            revise_batch,
            target_language=target,
            accept_score=args.accept_score,
            min_gain=args.min_gain,
            max_rounds=args.revise_rounds,
        )
        translations = [result.text for result in results]
        print(
            f"[sion] Iterative revision: {json.dumps(summarize(results), ensure_ascii=False)}",
            file=sys.stderr,
            flush=True,
        )

    for translated in translations:
        print(translated, flush=True)


if __name__ == "__main__":
    main()
