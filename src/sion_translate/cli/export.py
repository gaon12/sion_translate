"""Convert a stable FP32/EMA export into deployment/storage formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sion_translate.console import configure_stdio
from sion_translate.training.export import (
    DEFAULT_CONVERSION_FORMATS,
    SUPPORTED_FORMATS,
    convert_export,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a Sion FP32/EMA state-dict export")
    parser.add_argument("source", help="model.pt or model_ema.pt")
    parser.add_argument(
        "--output",
        required=True,
        help="directory that will receive converted files and export_manifest.json",
    )
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_CONVERSION_FORMATS),
        help=(
            "comma-separated output formats (default: " + ",".join(DEFAULT_CONVERSION_FORMATS) + ")"
        ),
    )
    parser.add_argument("--tokenizer", help="tokenizer model whose SHA-256 will be recorded")
    parser.add_argument(
        "--token-features",
        help="MorphoScript token_features.npz, copied into and verified with HF checkpoints",
    )
    parser.add_argument(
        "--language-pair",
        nargs=2,
        action="append",
        dest="language_pairs",
        metavar=("SOURCE", "TARGET"),
        help=(
            "language pair to export; repeat for additional pairs, for example "
            "--language-pair ko ja --language-pair en ru"
        ),
    )
    direction_policy = parser.add_mutually_exclusive_group()
    direction_policy.add_argument(
        "--translation-direction",
        nargs=2,
        action="append",
        metavar=("SOURCE", "TARGET"),
        help=(
            "exact direction seen during training; repeat for additional directions and "
            "match existing metadata exactly"
        ),
    )
    direction_policy.add_argument(
        "--bidirectional",
        action="store_true",
        help="explicitly attest that both directions of every language pair were trained",
    )
    direction_policy.add_argument(
        "--unidirectional",
        action="store_true",
        help="record only the SOURCE-to-TARGET direction of each language pair as trained",
    )
    parser.add_argument(
        "--release-name",
        help="release name for a legacy export without metadata, such as sion or sion_translate",
    )
    parser.add_argument(
        "--release-version",
        help="model generation for a legacy export without metadata, such as 1.0",
    )
    release_capability = parser.add_mutually_exclusive_group()
    release_capability.add_argument(
        "--translation-capable",
        dest="translation_capable",
        action="store_true",
        help="attest that a legacy export without metadata completed translation training",
    )
    release_capability.add_argument(
        "--foundation-only",
        dest="translation_capable",
        action="store_false",
        help="attest that a legacy export contains foundation-only, non-translation weights",
    )
    parser.set_defaults(translation_capable=None)
    capability = parser.add_mutually_exclusive_group()
    capability.add_argument(
        "--revision-direction",
        nargs=2,
        action="append",
        metavar=("SOURCE", "TARGET"),
        help=(
            "exact direction trained with revision examples; repeat as needed and use only "
            "authenticated translation directions"
        ),
    )
    capability.add_argument(
        "--revision-trained",
        action="store_true",
        help=(
            "compatibility flag for a legacy model with exactly one direction; use "
            "--revision-direction when more than one direction exists"
        ),
    )
    capability.add_argument(
        "--no-revision-trained",
        action="store_true",
        help="record that no revision-training examples were used",
    )
    parser.add_argument(
        "--int4-backend",
        choices=("auto", "torchao", "packed"),
        default="auto",
        help="auto falls back to portable packed INT4 when TorchAO is unavailable",
    )
    parser.add_argument(
        "--llama-quantize",
        help=(
            "deprecated compatibility argument (currently ignored); GGUF uses the built-in "
            "deterministic K-quant implementation"
        ),
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    formats = tuple(value.strip().lower() for value in args.formats.split(",") if value.strip())
    unknown = sorted(set(formats) - set(SUPPORTED_FORMATS))
    if unknown:
        raise SystemExit(
            f"unsupported formats: {unknown} (supported: {', '.join(SUPPORTED_FORMATS)})"
        )
    revision_trained = (
        True if args.revision_trained else False if args.no_revision_trained else None
    )
    manifest = convert_export(
        args.source,
        args.output,
        formats=formats,
        tokenizer_path=args.tokenizer,
        token_features_path=args.token_features,
        language_pairs=args.language_pairs,
        translation_directions=args.translation_direction,
        bidirectional=(True if args.bidirectional else False if args.unidirectional else None),
        revision_directions=args.revision_direction,
        revision_trained=revision_trained,
        release_name=args.release_name,
        release_version=args.release_version,
        translation_capable=args.translation_capable,
        int4_backend=args.int4_backend,
        llama_quantize=args.llama_quantize,
    )
    print(json.dumps(manifest["formats"], ensure_ascii=False, indent=2))
    failures = [
        name for name, result in manifest["formats"].items() if result.get("status") != "ok"
    ]
    print(f"[sion] manifest: {Path(args.output) / 'export_manifest.json'}")
    if failures:
        raise SystemExit(f"export failed for formats: {', '.join(failures)}")


if __name__ == "__main__":
    main()
