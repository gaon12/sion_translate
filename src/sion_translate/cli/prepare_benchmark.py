"""Convert FLORES-200 into the JSONL format consumed by ``sion-evaluate``.

FLORES-200 is widely used in research and commercial translation benchmarks,
so it provides a useful basis for comparing scores with external systems.

Recommended offline usage: download and extract a FLORES-200 distribution,
then pass its directory:

    sion-prepare-benchmark --flores-dir flores200_dataset --split devtest

Alternatively, when Hugging Face ``datasets`` is installed:

    sion-prepare-benchmark --hf --split devtest

The default output is ``benchmarks/flores_<a>_<b>.<split>.jsonl`` and can be
evaluated directly:

    sion-evaluate --benchmark benchmarks/flores_ko_ja.devtest.jsonl

The selected language pair comes from the configured language graph unless
``--language-pair`` chooses one of its edges explicitly.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sion_translate.benchmark import (
    flores_code,
    pairs_from_hf_datasets,
    pairs_from_local_flores,
    validate_flores_path_component,
    write_jsonl,
)
from sion_translate.config import config_from_raw, load_raw_config
from sion_translate.console import configure_stdio
from sion_translate.language_tags import (
    LanguageTagError,
    canonicalize_language_pair,
    canonicalize_language_tag,
)

DEFAULT_CONFIG_FILE = "sion_translate.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert FLORES-200 to sion-evaluate JSONL")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--flores-dir", help="Extracted local FLORES-200 directory")
    source.add_argument("--hf", action="store_true", help="Download with Hugging Face datasets")
    parser.add_argument(
        "--split",
        default="devtest",
        choices=("dev", "devtest"),
        help="FLORES split (default: devtest for standard evaluation)",
    )
    parser.add_argument(
        "--flores-code",
        action="append",
        default=[],
        metavar="LANG=CODE",
        help="Override a language's FLORES code, for example ko=kor_Hang",
    )
    parser.add_argument(
        "--output", help="Output JSONL path (default: benchmarks/flores_<a>_<b>.<split>.jsonl)"
    )
    parser.add_argument(
        "--language-pair",
        nargs=2,
        metavar=("LANG_A", "LANG_B"),
        help="One configured language pair to convert",
    )
    parser.add_argument("--config", help=f"Configuration file (default: {DEFAULT_CONFIG_FILE})")
    return parser


def log(message: str) -> None:
    print(f"[sion] {message}", flush=True)


def resolve_benchmark_language_pair(
    requested: Sequence[str] | None,
    configured_pairs: Sequence[Sequence[str]],
) -> tuple[str, str]:
    """Canonicalize one requested physical pair and reject alias collisions."""

    canonical_pairs: list[tuple[str, str]] = []
    seen_edges: set[frozenset[str]] = set()
    for index, raw_pair in enumerate(configured_pairs):
        try:
            pair = canonicalize_language_pair(
                raw_pair,
                field=f"configured language_pairs[{index}]",
            )
        except LanguageTagError as error:
            raise SystemExit(str(error)) from error
        edge = frozenset(pair)
        if edge in seen_edges:
            raise SystemExit(
                "Configured language_pairs contains duplicate physical edges after "
                f"BCP 47 canonicalization: {pair!r}"
            )
        seen_edges.add(edge)
        canonical_pairs.append(pair)

    if requested is not None:
        try:
            pair = canonicalize_language_pair(requested, field="--language-pair")
        except LanguageTagError as error:
            raise SystemExit(str(error)) from error
        if frozenset(pair) not in seen_edges:
            raise SystemExit(
                f"--language-pair {pair} is not configured. Available pairs: "
                f"{tuple(canonical_pairs)}"
            )
        return pair
    if len(canonical_pairs) == 1:
        return canonical_pairs[0]
    raise SystemExit(
        "Multilingual configurations require --language-pair LANG_A LANG_B. "
        f"Available pairs: {tuple(canonical_pairs)}"
    )


def parse_flores_code_overrides(specs: Sequence[str]) -> dict[str, str]:
    """Parse canonical language keys without silently overwriting aliases."""

    overrides: dict[str, str] = {}
    for spec in specs:
        raw_language, separator, code = spec.partition("=")
        if not separator or not raw_language or not code or code != code.strip():
            raise SystemExit(f"--flores-code must use LANG=CODE syntax: {spec}")
        try:
            language = canonicalize_language_tag(raw_language, field="--flores-code language")
        except LanguageTagError as error:
            raise SystemExit(str(error)) from error
        try:
            code = validate_flores_path_component(
                code,
                field=f"FLORES code for {language!r}",
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if language in overrides:
            raise SystemExit(
                "--flores-code contains duplicate language keys after BCP 47 "
                f"canonicalization: {language!r}"
            )
        overrides[language] = code
    return overrides


def resolve_flores_codes(
    language_pair: tuple[str, str],
    overrides: dict[str, str],
) -> dict[str, str]:
    """Resolve two distinct, path-safe FLORES identities for one benchmark pair."""

    unused = set(overrides).difference(language_pair)
    if unused:
        raise SystemExit(
            "--flores-code contains languages outside the selected --language-pair: "
            + ", ".join(sorted(unused))
        )
    codes = {language: flores_code(language, overrides.get(language)) for language in language_pair}
    for language, code in codes.items():
        try:
            validate_flores_path_component(
                code,
                field=f"FLORES code for {language!r}",
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
    if codes[language_pair[0]].casefold() == codes[language_pair[1]].casefold():
        raise SystemExit(
            "Both languages resolve to the same FLORES code on a case-insensitive "
            "filesystem: "
            f"{language_pair[0]}={codes[language_pair[0]]}, "
            f"{language_pair[1]}={codes[language_pair[1]]}"
        )
    return codes


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    config_path = args.config or (
        DEFAULT_CONFIG_FILE if Path(DEFAULT_CONFIG_FILE).exists() else None
    )
    config = config_from_raw(load_raw_config(config_path) if config_path else {})
    configured_pairs = config.data.language_pairs or [config.data.language_pair]
    pair = resolve_benchmark_language_pair(args.language_pair, configured_pairs)

    # Parse repeated overrides such as --flores-code ko=kor_Hang.
    overrides = parse_flores_code_overrides(args.flores_code)
    # Validate every code before doing filesystem or network work.
    codes = resolve_flores_codes(pair, overrides)

    if args.hf:
        log(f"Loading FLORES {args.split} from Hugging Face datasets ({pair[0]}↔{pair[1]})")
        pairs = pairs_from_hf_datasets(pair, split=args.split, code_overrides=overrides)
    else:
        if not args.flores_dir:
            raise SystemExit(
                "Provide --flores-dir <directory> or --hf. For offline use, download and "
                "extract FLORES-200, then pass the extracted directory to --flores-dir."
            )
        log(
            f"Loading local FLORES {args.split}: {args.flores_dir} "
            f"({codes[pair[0]]}, {codes[pair[1]]})"
        )
        pairs = pairs_from_local_flores(
            args.flores_dir, pair, split=args.split, code_overrides=overrides
        )

    output = args.output or f"benchmarks/flores_{pair[0]}_{pair[1]}.{args.split}.jsonl"
    count = write_jsonl(pairs, output)
    log(f"Wrote {count:,} parallel pairs: {output}")
    log(f"Evaluate with: sion-evaluate --benchmark {output}")


if __name__ == "__main__":
    main()
