"""Translate immutable JSONL queues into audited synthetic training shards."""

# Existing queue manifests are heterogeneous JSON mappings.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import getpass
import json
from pathlib import Path
from typing import cast

import torch

from sion_translate.config import config_from_raw, load_raw_config
from sion_translate.console import configure_stdio
from sion_translate.inference import Translator, find_exported_model
from sion_translate.language_tags import canonicalize_language_tag
from sion_translate.queue_translation import (
    QueueTranslationOptions,
    load_queue_run_metadata,
    translate_queue,
)
from sion_translate.scripts_registry import known_scripts

DEFAULT_CONFIG_FILE = "sion_translate.yaml"
DEFAULT_TEACHER_PILOT_ROWS = 1_000
_PROVENANCE_PLACEHOLDERS = {"n/a", "na", "none", "tbd", "unknown", "unset"}


def _provenance_value(value: str) -> str:
    normalized = value.strip()
    if not normalized or normalized.casefold() in _PROVENANCE_PLACEHOLDERS:
        raise argparse.ArgumentTypeError(
            "provenance must be explicit and cannot use an unknown placeholder"
        )
    return normalized


def _target_script_requirement(value: str) -> tuple[str, str, int]:
    """Parse one ``LANGUAGE=SCRIPT:MINIMUM`` target-writing-system rule."""

    raw_language, language_separator, remainder = value.partition("=")
    raw_script, minimum_separator, raw_minimum = remainder.rpartition(":")
    if not language_separator or not minimum_separator:
        raise argparse.ArgumentTypeError(
            "target script rules must use LANGUAGE=SCRIPT:MINIMUM syntax"
        )
    try:
        language = canonicalize_language_tag(raw_language.strip(), field="target language")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    script = raw_script.strip().lower()
    if script not in known_scripts():
        choices = ", ".join(known_scripts())
        raise argparse.ArgumentTypeError(f"unknown script {script!r}; choose one of: {choices}")
    try:
        minimum = int(raw_minimum)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target script minimum must be an integer") from exc
    if minimum <= 0:
        raise argparse.ArgumentTypeError("target script minimum must be positive")
    return language, script, minimum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume-safe translation and round-trip filtering for JSONL queues"
    )
    parser.add_argument("--input", required=True, help="pending queue JSONL")
    parser.add_argument("--output-dir", help="audit result shard directory")
    parser.add_argument("--accepted-dir", help="private manifest-gated accepted-part directory")
    parser.add_argument("--model", help="exported model path")
    parser.add_argument(
        "--int8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="auto-discovery uses the CPU INT8 export by default",
    )
    parser.add_argument("--device", help="torch device (INT8 exports remain CPU-only)")
    parser.add_argument("--config", help=f"config file (default: {DEFAULT_CONFIG_FILE})")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=1_000)
    parser.add_argument("--max-rows", type=int, help="additional rows to process this invocation")
    parser.add_argument(
        "--teacher-pilot-rows",
        type=int,
        default=DEFAULT_TEACHER_PILOT_ROWS,
        help=(
            "without --approve-teacher, stop after this many generated translations so the "
            "teacher's semantic quality can be reviewed"
        ),
    )
    parser.add_argument(
        "--approve-teacher",
        action="store_true",
        help="continue beyond the pilot after manually reviewing its translations",
    )
    parser.add_argument(
        "--approval-actor",
        help="name recorded in the manifest when --approve-teacher is accepted",
    )
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-output-length-ratio", type=float, default=2.0)
    parser.add_argument("--max-output-length-margin", type=int, default=12)
    parser.add_argument(
        "--roundtrip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "require a reverse direction recorded in the model artifact for cycle filtering; "
            "use --no-roundtrip for an intentionally forward-only model"
        ),
    )
    parser.add_argument("--roundtrip-num-beams", type=int, default=1)
    parser.add_argument("--roundtrip-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--roundtrip-max-output-length-ratio",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--roundtrip-max-output-length-margin",
        type=int,
        default=12,
    )
    parser.add_argument("--min-roundtrip-score", type=float, default=0.65)
    parser.add_argument("--min-pair-score", type=int, default=80)
    parser.add_argument("--min-target-language-fraction", type=float, default=0.50)
    parser.add_argument(
        "--require-target-script",
        action="append",
        default=[],
        type=_target_script_requirement,
        metavar="LANGUAGE=SCRIPT:MINIMUM",
        help=(
            "require a minimum writing-system character count for an exact BCP 47 target tag; "
            "repeat this option for additional languages or scripts"
        ),
    )
    parser.add_argument("--min-structured-similarity", type=float, default=1.0)
    parser.add_argument(
        "--threads",
        type=int,
        help="CPU intra-op threads (leave unset for PyTorch default)",
    )
    parser.add_argument(
        "--source-dataset",
        type=_provenance_value,
        help=(
            "explicit dataset or corpus identity; required for a new run and inherited "
            "unchanged from the checked manifest when resuming"
        ),
    )
    parser.add_argument(
        "--source-revision",
        type=_provenance_value,
        help=(
            "explicit source commit, release, snapshot, or content revision; required for "
            "a new run and inherited unchanged when resuming"
        ),
    )
    parser.add_argument(
        "--source-license",
        type=_provenance_value,
        help=(
            "explicit source license or usage-rights identifier; required for a new run "
            "and inherited unchanged when resuming"
        ),
    )
    return parser


def _loaded_identity(
    translator: Translator,
    attribute: str,
    *,
    required: bool,
) -> dict[str, object] | None:
    value = getattr(translator, attribute, None)
    if value is None:
        if required:
            raise SystemExit(f"Translator did not capture required load identity: {attribute}")
        return None
    if not isinstance(value, Mapping):
        raise SystemExit(f"Translator load identity is invalid: {attribute}")
    typed_value = cast(Mapping[object, object], value)
    return {str(key): item for key, item in typed_value.items()}


def _existing_run_metadata(output_dir: Path) -> dict[str, object] | None:
    try:
        run_metadata = load_queue_run_metadata(output_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return run_metadata


def _resolve_run_metadata_seed(
    output_dir: Path,
    *,
    source_dataset: str | None,
    source_revision: str | None,
    source_license: str | None,
) -> tuple[dict[str, object], bool]:
    supplied = {
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_license": source_license,
    }
    existing = _existing_run_metadata(output_dir)
    if existing is None:
        missing = [field for field, value in supplied.items() if value is None]
        if missing:
            flags = ", ".join("--" + field.replace("_", "-") for field in missing)
            raise SystemExit(f"new queue runs require explicit provenance: {flags}")
        return ({field: str(value) for field, value in supplied.items()}, False)

    for field, value in supplied.items():
        if value is not None and existing.get(field) != value:
            raise SystemExit(
                f"cannot change recorded {field} while resuming; use a new output directory"
            )
    return existing, True


def _validate_resume_runtime_metadata(
    recorded: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    def same_identity(recorded_value: object, current_value: object) -> bool:
        if recorded_value is None or current_value is None:
            return recorded_value is current_value
        if not isinstance(recorded_value, Mapping) or not isinstance(current_value, Mapping):
            return recorded_value == current_value
        typed_recorded = cast(Mapping[object, object], recorded_value)
        typed_current = cast(Mapping[object, object], current_value)
        fields = ("path", "size", "sha256")
        if any(typed_recorded.get(field) != typed_current.get(field) for field in fields):
            return False
        stat_fields = ("device", "inode", "mtime_ns")
        return not any(field in typed_recorded for field in stat_fields) or all(
            typed_recorded.get(field) == typed_current.get(field) for field in stat_fields
        )

    for field in ("translation_model", "tokenizer", "token_features"):
        if field not in recorded:
            raise SystemExit(
                f"legacy queue manifest cannot bind the current {field} bytes; "
                "start a new output directory"
            )
        if not same_identity(recorded[field], current.get(field)):
            raise SystemExit(
                f"current {field} differs from the queue manifest; "
                "resume with the original artifacts or use a new output directory"
            )
    if "tokenizer_metadata" in recorded and not same_identity(
        recorded["tokenizer_metadata"],
        current.get("tokenizer_metadata"),
    ):
        raise SystemExit(
            "current tokenizer_metadata differs from the queue manifest; "
            "resume with the original sidecar or use a new output directory"
        )
    if "translation_directions" in recorded and recorded["translation_directions"] != current.get(
        "translation_directions"
    ):
        raise SystemExit("current translation direction graph differs from the queue manifest")
    recorded_graph_source = recorded.get("translation_graph_source")
    current_graph_source = current.get("translation_graph_source")
    if recorded_graph_source is not None and recorded_graph_source != current_graph_source:
        raise SystemExit("current translation graph source differs from the queue manifest")
    if recorded_graph_source is None and current_graph_source == "tokenizer_metadata":
        if "tokenizer_metadata" not in recorded:
            raise SystemExit(
                "legacy queue manifest cannot bind the tokenizer metadata bytes that supply "
                "the translation graph; start a new output directory"
            )
        if not same_identity(
            recorded["tokenizer_metadata"],
            current.get("tokenizer_metadata"),
        ):
            raise SystemExit(
                "current graph-bearing tokenizer metadata differs from the queue manifest"
            )


def log(message: str) -> None:
    print(f"[sion-queue] {message}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    configure_stdio()
    args = build_parser().parse_args(argv)
    if args.threads is not None:
        if args.threads <= 0:
            raise SystemExit("--threads must be positive")
        torch.set_num_threads(args.threads)

    config_path = args.config or (
        DEFAULT_CONFIG_FILE if Path(DEFAULT_CONFIG_FILE).is_file() else None
    )
    config = config_from_raw(load_raw_config(config_path) if config_path else {})
    input_path = Path(args.input)
    output_dir = Path(args.output_dir or f"translation_queue/results/{input_path.stem}")
    accepted_dir = Path(args.accepted_dir or f"translation_queue/accepted/{input_path.stem}")
    metadata_seed, resuming = _resolve_run_metadata_seed(
        output_dir,
        source_dataset=args.source_dataset,
        source_revision=args.source_revision,
        source_license=args.source_license,
    )
    model_path = Path(
        args.model
        or find_exported_model(
            config.training.output_dir,
            int8=args.int8,
        )
    )
    tokenizer_path = Path(config.data.tokenizer_model)
    token_features_path = Path(config.data.tokenizer_features)
    options = QueueTranslationOptions(
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        max_output_length_ratio=args.max_output_length_ratio,
        max_output_length_margin=args.max_output_length_margin,
        roundtrip_enabled=args.roundtrip,
        roundtrip_num_beams=args.roundtrip_num_beams,
        roundtrip_max_new_tokens=args.roundtrip_max_new_tokens,
        roundtrip_max_output_length_ratio=(args.roundtrip_max_output_length_ratio),
        roundtrip_max_output_length_margin=(args.roundtrip_max_output_length_margin),
        min_roundtrip_score=args.min_roundtrip_score,
        min_pair_score=args.min_pair_score,
        min_target_language_fraction=args.min_target_language_fraction,
        required_target_scripts=tuple(args.require_target_script),
        min_structured_similarity=args.min_structured_similarity,
    )
    options.validate()

    log(f"model: {model_path}")
    translator = Translator(
        model_path,
        tokenizer_path,
        device=args.device,
        token_features_path=token_features_path,
    )
    raw_export_directions = translator.export_metadata.get("translation_directions")
    graph_in_model = False
    if isinstance(raw_export_directions, Sequence) and not isinstance(
        raw_export_directions, (str, bytes)
    ):
        graph_in_model = len(cast(Sequence[object], raw_export_directions)) > 0
    runtime_metadata = {
        "translation_model": _loaded_identity(
            translator,
            "translation_model_identity",
            required=True,
        ),
        "tokenizer": _loaded_identity(
            translator,
            "tokenizer_model_identity",
            required=True,
        ),
        "tokenizer_metadata": _loaded_identity(
            translator,
            "tokenizer_metadata_identity",
            required=False,
        ),
        "token_features": _loaded_identity(
            translator,
            "token_features_identity",
            required=False,
        ),
        "translation_directions": [
            list(direction) for direction in translator.translation_directions
        ],
        "translation_graph_source": (
            "translation_model" if graph_in_model else "tokenizer_metadata"
        ),
    }
    if resuming:
        _validate_resume_runtime_metadata(metadata_seed, runtime_metadata)
        metadata = metadata_seed
    else:
        metadata = {**metadata_seed, **runtime_metadata}
    manifest = translate_queue(
        input_path,
        output_dir,
        translator,
        accepted_dir=accepted_dir,
        options=options,
        run_metadata=metadata,
        max_rows=args.max_rows,
        teacher_pilot_rows=args.teacher_pilot_rows,
        approve_teacher=args.approve_teacher,
        approval_actor=args.approval_actor or getpass.getuser(),
        log=log,
    )
    teacher_review = manifest.get("teacher_review")
    if isinstance(teacher_review, dict) and teacher_review.get("review_required") is True:
        log(
            f"teacher pilot is ready for semantic review in {output_dir.resolve()}; "
            "continue with --approve-teacher only if its translations are acceptable"
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
