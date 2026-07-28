"""Translate immutable JSONL queues into audited synthetic training shards."""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path

import torch

from sion_translate.config import config_from_raw, load_raw_config
from sion_translate.console import configure_stdio
from sion_translate.inference import Translator, find_exported_model
from sion_translate.queue_translation import (
    QueueTranslationOptions,
    sha256_file,
    translate_queue,
)

DEFAULT_CONFIG_FILE = "sion_translate.yaml"
DEFAULT_TEACHER_PILOT_ROWS = 1_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume-safe translation and round-trip filtering for JSONL queues"
    )
    parser.add_argument("--input", required=True, help="pending queue JSONL")
    parser.add_argument("--output-dir", help="audit result shard directory")
    parser.add_argument("--accepted-dir", help="accepted bt_* shard directory")
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
        "--min-japanese-kana-chars",
        type=int,
        default=1,
        help="reject Japanese candidates with fewer kana characters (0 disables)",
    )
    parser.add_argument("--min-structured-similarity", type=float, default=1.0)
    parser.add_argument(
        "--threads",
        type=int,
        help="CPU intra-op threads (leave unset for PyTorch default)",
    )
    parser.add_argument("--source-dataset", default="heegyu/namuwiki")
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--source-license", default="CC-BY-NC-SA-2.0")
    return parser


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def log(message: str) -> None:
    print(f"[sion-queue] {message}", flush=True)


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
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
        min_japanese_kana_chars=args.min_japanese_kana_chars,
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
    metadata = {
        "source_dataset": args.source_dataset,
        "source_revision": args.source_revision,
        "source_license": args.source_license,
        "translation_model": _file_identity(model_path),
        "tokenizer": _file_identity(tokenizer_path),
        "token_features": (
            _file_identity(token_features_path) if token_features_path.is_file() else None
        ),
    }
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
    if isinstance(teacher_review, dict) and teacher_review.get("review_required"):
        log(
            f"teacher pilot is ready for semantic review in {output_dir.resolve()}; "
            "continue with --approve-teacher only if its translations are acceptable"
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
