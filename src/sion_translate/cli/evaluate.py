"""Measure chrF/BLEU on a fixed translation evaluation set.

    sion-evaluate                          # Evaluate every direction on the test split
    sion-evaluate --benchmark flores.jsonl # Evaluate an external JSONL benchmark
    sion-evaluate --direction ko ja \
      --compare deepl=deepl_out.txt \
      --compare google=google_out.txt     # Compare external-service output

To compare an external service:
    1. Export evaluation sources with
       sion-evaluate --direction ko ja --export-sources src.txt.
    2. Translate those sources with DeepL, Google, Papago, or another service,
       then save one translation per line.
    3. Pass --compare SERVICE=OUTPUT_FILE to score it against the same references
       with the same metrics.

Results are printed as a terminal table and saved under
reports/evaluation-*.json/.md.
"""

# CLI registry callables are discovered dynamically.
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from sion_translate.config import config_from_raw, load_raw_config
from sion_translate.console import configure_stdio
from sion_translate.glossary import load_glossary
from sion_translate.evaluation import (
    DirectionResult,
    load_benchmark_pairs,
    load_split_pairs,
    number_preservation_details,
    results_as_markdown,
    save_results,
    score_translations,
)
from sion_translate.inference import Translator, find_exported_model
from sion_translate.language_tags import LanguageTagError, canonicalize_language_pair

DEFAULT_CONFIG_FILE = "sion_translate.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate translation quality (chrF/BLEU)")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (default: test)")
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help=(
            "External benchmark JSONL path. When provided, it replaces --split; "
            "may be specified repeatedly."
        ),
    )
    parser.add_argument(
        "--direction",
        nargs="+",
        metavar="LANG",
        help=(
            "Evaluation direction: --direction SOURCE TARGET (default: every "
            "trained direction). Simple tags without hyphens also accept the "
            "legacy ko-ja form."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum evaluation sentences per direction",
    )
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", help="Exported model path (default: discover from exports)")
    parser.add_argument("--int8", action="store_true", help="Evaluate the INT8 quantized model")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="External-system output file, one translation per line; requires --direction",
    )
    parser.add_argument(
        "--export-sources",
        help=("Save evaluation sources to this file for an external service; requires --direction"),
    )
    parser.add_argument("--output", help="Result path (default: reports/evaluation-<timestamp>)")
    parser.add_argument(
        "--glossary",
        help=(
            "Glossary JSON path used to force terms in sion_translate output "
            "(default: configuration data.glossary)"
        ),
    )
    parser.add_argument(
        "--no-glossary",
        action="store_true",
        help="Disable the glossary for evaluation even when one is configured",
    )
    parser.add_argument("--config", help=f"Configuration file (default: {DEFAULT_CONFIG_FILE})")
    return parser


def log(message: str) -> None:
    print(f"[sion] {message}", flush=True)


def resolve_evaluation_directions(
    requested: str | Sequence[str] | None,
    trained_directions: Sequence[Sequence[str]],
) -> list[tuple[str, str]]:
    """Resolve a structured CLI request against the model's exact trained graph."""

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
                f"normalization: {_direction_label(*direction)}"
            )
        seen.add(direction)
        directions.append(direction)

    if not directions:
        raise SystemExit("The model has no authenticated translation_directions.")

    tokens = (requested,) if isinstance(requested, str) else tuple(requested or ())
    if not tokens or tokens == ("both",):
        return directions

    if len(tokens) == 2:
        try:
            selected = canonicalize_language_pair(tokens, field="--direction")
        except LanguageTagError as error:
            raise SystemExit(str(error)) from error
        if selected in seen:
            return [selected]
    elif len(tokens) == 1:
        # Backwards compatibility is limited to labels whose two members do
        # not contain a hyphen.  Such labels have exactly one separator and
        # cannot collide with an extended or private-use BCP 47 identity.
        legacy = tokens[0]
        matches = [
            direction
            for direction in directions
            if "-" not in direction[0]
            and "-" not in direction[1]
            and f"{direction[0]}-{direction[1]}" == legacy
        ]
        if len(matches) == 1:
            return matches

    valid = ", ".join(f"--direction {source} {target}" for source, target in directions)
    raise SystemExit(f"--direction requires two language tags: SOURCE TARGET (supported: {valid})")


def _direction_label(source_language: str, target_language: str) -> str:
    """Return a collision-free label while preserving legacy simple labels."""

    if "-" not in source_language and "-" not in target_language:
        return f"{source_language}-{target_language}"
    return f"{source_language}→{target_language}"


def _read_comparison_lines(file_path: str | Path, *, expected_count: int) -> list[str]:
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()
    if len(lines) != expected_count:
        raise SystemExit(
            f"{file_path}: {len(lines)} translation lines != {expected_count} "
            "evaluation pairs; the line count must match the evaluation set exactly"
        )
    return lines


def _parse_comparison_specs(specs: Sequence[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    for spec in specs:
        raw_name, separator, raw_path = spec.partition("=")
        name = raw_name.strip()
        file_path = raw_path.strip()
        if not separator or not name or not file_path:
            raise SystemExit(f"--compare must use NAME=FILE format: {spec}")
        identity = name.casefold()
        if identity == "sion":
            raise SystemExit(
                "The --compare system name 'sion' is reserved for the built-in model label"
            )
        if identity in seen_names:
            raise SystemExit(f"Duplicate --compare system name: {name}")
        seen_names.add(identity)
        parsed.append((name, file_path))
    return parsed


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    config_path = args.config or (
        DEFAULT_CONFIG_FILE if Path(DEFAULT_CONFIG_FILE).exists() else None
    )
    config = config_from_raw(load_raw_config(config_path) if config_path else {})
    model_path = args.model or find_exported_model(config.training.output_dir, int8=args.int8)
    translator = Translator(model_path, config.data.tokenizer_model)

    # ── Resolve evaluation directions ────────────────────────────────────
    directions = resolve_evaluation_directions(args.direction, translator.translation_directions)
    if (args.compare or args.export_sources) and len(directions) != 1:
        raise SystemExit(
            "--compare / --export-sources require --direction to select exactly one direction"
        )
    comparison_specs = _parse_comparison_specs(args.compare)

    # ── Optional glossary ────────────────────────────────────────────────
    glossary = None
    glossary_path = None if args.no_glossary else (args.glossary or config.data.glossary)
    if glossary_path:
        glossary = load_glossary(glossary_path)
        log(f"Applying glossary: {glossary_path} ({len(glossary)} terms)")

    # ── Load evaluation pairs ────────────────────────────────────────────
    if args.benchmark:
        log(f"Loading benchmark: {', '.join(args.benchmark)}")
        pairs = load_benchmark_pairs(
            args.benchmark,
            translator.language_pairs,
            translation_directions=translator.translation_directions,
            max_samples_per_direction=args.max_samples,
        )
        eval_set_name = ";".join(args.benchmark)
    else:
        log(f"Loading internal {args.split} split (a holdout never exposed to training)")
        pairs = load_split_pairs(
            config.data.dataset_dir,
            args.split,
            translator.tokenizer,
            model_language_pairs=translator.language_pairs,
            max_samples_per_direction=args.max_samples,
        )
        eval_set_name = f"dataset:{args.split}"

    # ── Optionally export sources for an external service ────────────────
    if args.export_sources:
        direction = directions[0]
        sources = [source for source, _ in pairs.get(direction, [])]
        Path(args.export_sources).write_text("\n".join(sources) + "\n", encoding="utf-8")
        log(
            f"Saved {len(sources)} source sentences to {args.export_sources}. "
            "Translate them with the external service, then pass its output via --compare."
        )

    # ── Run evaluation ───────────────────────────────────────────────────
    results: list[DirectionResult] = []
    for source_language, target_language in directions:
        samples = pairs.get((source_language, target_language), [])
        if not samples:
            log(
                f"{_direction_label(source_language, target_language)}: "
                "no evaluation pairs; skipping"
            )
            continue
        sources = [source for source, _ in samples]
        references = [reference for _, reference in samples]
        direction_name = _direction_label(source_language, target_language)

        log(f"{direction_name}: translating {len(samples)} sentences (beam {args.num_beams})...")
        started = time.perf_counter()
        hypotheses = translator.translate(
            sources,
            source_language=source_language,
            target_language=target_language,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            glossary=glossary,
        )
        elapsed = time.perf_counter() - started
        chrf, bleu, tokenize = score_translations(
            hypotheses, references, target_language=target_language
        )
        number_result = number_preservation_details(hypotheses, sources=sources)
        results.append(
            DirectionResult(
                system="sion",
                direction=direction_name,
                samples=len(samples),
                chrf=chrf,
                bleu=bleu,
                bleu_tokenize=tokenize,
                number_f1=number_result.f1,
                number_exact=number_result.exact,
                number_samples=number_result.samples,
                number_inventions=number_result.inventions,
            )
        )
        number_summary = (
            f"number F1 {number_result.f1:.2f} "
            f"(exact {number_result.exact}/{number_result.samples}, "
            f"inventions {number_result.inventions})"
            if number_result.samples
            else "no sentences with numbers"
        )
        log(
            f"{direction_name}: chrF {chrf:.2f} / BLEU {bleu:.2f} / "
            f"{number_summary} ({elapsed:.0f}s)"
        )

        # Score external output against the same references and metrics.
        for name, file_path in comparison_specs:
            hypotheses = _read_comparison_lines(
                file_path,
                expected_count=len(references),
            )
            chrf, bleu, tokenize = score_translations(
                hypotheses, references, target_language=target_language
            )
            number_result = number_preservation_details(hypotheses, sources=sources)
            results.append(
                DirectionResult(
                    system=name,
                    direction=direction_name,
                    samples=len(references),
                    chrf=chrf,
                    bleu=bleu,
                    bleu_tokenize=tokenize,
                    number_f1=number_result.f1,
                    number_exact=number_result.exact,
                    number_samples=number_result.samples,
                    number_inventions=number_result.inventions,
                )
            )

    if not results:
        raise SystemExit("No data is available for evaluation.")

    # ── Print and save results ───────────────────────────────────────────
    print()
    print(results_as_markdown(results))
    output = args.output or f"reports/evaluation-{time.strftime('%Y%m%d-%H%M%S')}"
    save_results(
        results,
        output,
        metadata={
            "model": str(model_path),
            "eval_set": eval_set_name,
            "num_beams": args.num_beams,
            "max_samples": args.max_samples,
            "language_pairs": [list(pair) for pair in translator.language_pairs],
            "translation_directions": [
                list(direction) for direction in translator.translation_directions
            ],
            "evaluated_directions": [list(direction) for direction in directions],
        },
    )
    log(f"Saved: {output}.json / {output}.md")


if __name__ == "__main__":
    main()
