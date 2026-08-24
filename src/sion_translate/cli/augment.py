"""Authenticated backtranslation data augmentation CLI.

Monolingual files use ``<name>.<language>.txt``. The model artifact owns the
generation edge ``T→S``; the destination dataset configuration must own the
opposite training edge ``S→T``. Every published row is scoped to that one
training direction so a generated pseudo-target is never learned in reverse.
"""

from __future__ import annotations

import argparse
from collections.abc import Collection, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import cast

from sion_translate.augmentation import (
    AugmentationIdentity,
    AugmentationRegistry,
    FileSnapshot,
    JobProgress,
    build_job_identity,
    count_prepared_direction_pairs,
    load_augmentation_registry,
    reconcile_job_identity,
    run_augmentation_job,
    snapshot_file,
    synthetic_budget,
    validate_prepared_raw_contract,
)
from sion_translate.config import AppConfig, config_from_raw, load_raw_config
from sion_translate.console import configure_stdio
from sion_translate.data.quality import canonical_text
from sion_translate.inference import Translator, find_exported_model
from sion_translate.language_tags import (
    LanguageTagError,
    canonicalize_language_pair,
    canonicalize_language_tag,
)
from sion_translate.locking import artifact_locks

DEFAULT_CONFIG_FILE = "sion_translate.yaml"
_SHA256_LENGTH = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtranslation data augmentation")
    parser.add_argument(
        "--mono-dir",
        default="data_mono",
        help="Directory containing monolingual text (default: data_mono)",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.0,
        help=(
            "Maximum synthetic training rows per direction: real training rows "
            "in that direction multiplied by this value"
        ),
    )
    parser.add_argument(
        "--model", help="Exported generator model path (default: discover from exports)"
    )
    parser.add_argument(
        "--tokenizer",
        help=(
            "Tokenizer path for the generator model. When omitted, use the "
            "destination training configuration's tokenizer. Specify this option "
            "for a separate reverse-generator artifact."
        ),
    )
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--language-pair",
        nargs=2,
        metavar=("LANG_A", "LANG_B"),
        help="Physical language pair to augment in a multi-pair model",
    )
    parser.add_argument(
        "--config",
        help=f"Destination training configuration (default: {DEFAULT_CONFIG_FILE})",
    )
    return parser


def log(message: str) -> None:
    print(f"[sion] {message}", flush=True)


def resolve_augmentation_pair(
    requested: Sequence[str] | None,
    trained_pairs: Sequence[Sequence[str]],
) -> tuple[str, str]:
    """Choose a physical pair from the generator artifact, never local YAML."""

    pairs = [
        canonicalize_language_pair(pair, field="augmentation model language pair")
        for pair in trained_pairs
    ]
    if requested is not None:
        requested_pair = canonicalize_language_pair(
            requested,
            field="--language-pair",
        )
        edge = frozenset(requested_pair)
        matches = [pair for pair in pairs if frozenset(pair) == edge]
        if len(matches) != 1:
            raise SystemExit(
                f"--language-pair is not present in the model: {tuple(requested)} "
                f"(supported: {pairs})"
            )
        return matches[0]
    if len(pairs) == 1:
        return pairs[0]
    raise SystemExit(
        f"--language-pair LANG_A LANG_B is required for a multi-pair model (supported: {pairs})"
    )


def resolve_augmentation_destination(
    model_pair: Sequence[str],
    destination_pairs: Sequence[Sequence[str]],
) -> tuple[str, str]:
    """Bind the generator pair to exactly one destination physical pair."""

    normalized_model_pair = canonicalize_language_pair(
        model_pair,
        field="augmentation model language pair",
    )
    edge = frozenset(normalized_model_pair)
    normalized_destinations = [
        canonicalize_language_pair(pair, field="augmentation destination language pair")
        for pair in destination_pairs
    ]
    matches = [pair for pair in normalized_destinations if frozenset(pair) == edge]
    if len(matches) != 1:
        configured = list(normalized_destinations)
        raise SystemExit(
            "The augmentation model's language pair must appear exactly once in "
            "the current training configuration: "
            f"model={normalized_model_pair}, config={configured}"
        )
    return matches[0]


def preflight_backtranslation_directions(
    pair: tuple[str, str],
    jobs: Sequence[tuple[Path, str]],
    generation_directions: Sequence[Sequence[str]],
    training_directions: Sequence[Sequence[str]],
) -> None:
    """Validate every true-BT generator/destination edge before generation."""

    pair = canonicalize_language_pair(pair, field="augmentation language pair")
    generated = {
        canonicalize_language_pair(direction, field="model generation direction")
        for direction in generation_directions
    }
    trained = {
        canonicalize_language_pair(direction, field="destination training direction")
        for direction in training_directions
    }
    normalized_jobs = [
        (
            path,
            canonicalize_language_tag(
                mono_language,
                field="augmentation monolingual language",
            ),
        )
        for path, mono_language in jobs
    ]
    unknown_job_languages = sorted(
        {mono_language for _, mono_language in normalized_jobs} - set(pair)
    )
    if unknown_job_languages:
        raise SystemExit(
            "Monolingual input languages fall outside the augmentation pair: "
            f"{unknown_job_languages} not in {pair}"
        )
    required_generation = {
        (mono_language, pair[0] if mono_language == pair[1] else pair[1])
        for _, mono_language in normalized_jobs
    }
    required_training = {(target, source) for source, target in required_generation}
    missing_generation = sorted(required_generation - generated)
    missing_training = sorted(required_training - trained)
    if not missing_generation and not missing_training:
        return

    failures: list[str] = []
    if missing_generation:
        needed = ", ".join(f"{source}→{target}" for source, target in missing_generation)
        supported = ", ".join(f"{source}→{target}" for source, target in sorted(generated))
        failures.append(
            f"missing model generation directions: {needed} (model supports: {supported or 'none'})"
        )
    if missing_training:
        needed = ", ".join(f"{source}→{target}" for source, target in missing_training)
        supported = ", ".join(f"{source}→{target}" for source, target in sorted(trained))
        failures.append(
            f"missing destination training directions: {needed} "
            f"(configuration supports: {supported or 'none'})"
        )
    raise SystemExit(
        "The true backtranslation direction contract is not satisfied: " + "; ".join(failures)
    )


def _valid_sha256(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return None
    try:
        int(value, 16)
    except ValueError:
        return None
    return value.lower()


def generator_identity(
    translator: Translator,
    model_snapshot: FileSnapshot,
) -> tuple[str, str]:
    """Return strong model and tokenizer identities authenticated by the export."""

    metadata = translator.export_metadata
    raw_source: object = metadata.get("source")
    source: Mapping[object, object] = (
        cast(Mapping[object, object], raw_source)
        if isinstance(raw_source, Mapping)
        else cast(Mapping[object, object], {})
    )
    source_sha = _valid_sha256(source.get("sha256"))
    raw_tokenizer: object = metadata.get("tokenizer")
    tokenizer: Mapping[object, object] = (
        cast(Mapping[object, object], raw_tokenizer)
        if isinstance(raw_tokenizer, Mapping)
        else cast(Mapping[object, object], {})
    )
    tokenizer_sha = _valid_sha256(tokenizer.get("sha256"))
    if tokenizer_sha is None and translator.tokenizer_metadata is not None:
        tokenizer_sha = _valid_sha256(translator.tokenizer_metadata.get("model_sha256"))
    if source_sha is None or tokenizer_sha is None:
        raise ValueError(
            "The augmentation generator export has no source/tokenizer SHA-256 "
            "identity. Use a current 1.5 export."
        )
    identity_payload = {
        "loaded_artifact": {
            "size": model_snapshot.size,
            "sha256": model_snapshot.sha256,
        },
        "source_sha256": source_sha,
        "tokenizer_sha256": tokenizer_sha,
        "release_name": metadata.get("release_name"),
        "release_version": metadata.get("release_version"),
        "step": metadata.get("step"),
        "language_pairs": [
            list(canonicalize_language_pair(pair, field="generator language pair"))
            for pair in translator.language_pairs
        ],
        "translation_directions": [
            list(canonicalize_language_pair(edge, field="generator translation direction"))
            for edge in translator.translation_directions
        ],
        "pipeline": metadata.get("pipeline"),
        "feature_flags": metadata.get("feature_flags"),
        "capabilities": metadata.get("capabilities"),
        "quantization": metadata.get("quantization"),
        "generation_defaults": metadata.get("generation_defaults"),
    }
    serialized = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), tokenizer_sha


def _discover_mono_files(mono_dir: Path, pair: tuple[str, str]) -> list[tuple[Path, str]]:
    pair = canonicalize_language_pair(pair, field="augmentation language pair")
    jobs: list[tuple[Path, str]] = []
    for path in sorted(mono_dir.glob("*.txt")) if mono_dir.exists() else []:
        parts = path.name.split(".")
        if len(parts) < 3:
            continue
        try:
            language = canonicalize_language_tag(
                parts[-2],
                field=f"monolingual filename {path.name}",
            )
        except LanguageTagError:
            continue
        if language in pair:
            jobs.append((path, language))
    if not jobs:
        raise SystemExit(
            f"No monolingual files were found in {mono_dir}/. Add files named "
            f"'name.<language>.txt' for {'/'.join(pair)}; for example, "
            f"news.{pair[1]}.txt."
        )
    return jobs


def _source_has_remaining_text(
    path: Path,
    progress: JobProgress,
    seen_mono_hashes: Collection[str] = frozenset(),
) -> bool:
    total_lines = 0
    has_text = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            total_lines = line_number + 1
            text = canonical_text(line)
            if (
                line_number >= progress.cursor_line
                and text
                and hashlib.sha256(text.encode("utf-8")).hexdigest() not in seen_mono_hashes
            ):
                has_text = True
    if progress.cursor_line > total_lines:
        raise ValueError(
            "The augmentation ledger cursor exceeds the monolingual input line count: "
            f"{progress.cursor_line} > {total_lines}"
        )
    if progress.eof and progress.cursor_line != total_lines:
        raise ValueError(
            "The augmentation ledger EOF state disagrees with the monolingual "
            "input line count: "
            f"cursor={progress.cursor_line}, lines={total_lines}"
        )
    return has_text


def _build_jobs(
    mono_files: Sequence[tuple[Path, str]],
    *,
    pair: tuple[str, str],
    synthetic_prefix: str,
    model_identity: str,
    tokenizer_identity: str,
    num_beams: int,
    max_new_tokens: int,
    registry: AugmentationRegistry,
) -> list[tuple[Path, AugmentationIdentity, JobProgress]]:
    jobs: list[tuple[Path, AugmentationIdentity, JobProgress]] = []
    for path, mono_language in mono_files:
        identity = build_job_identity(
            synthetic_prefix=synthetic_prefix,
            pair=pair,
            mono_language=mono_language,
            input_snapshot=snapshot_file(path),
            model_identity=model_identity,
            generator_tokenizer_sha256=tokenizer_identity,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        jobs.append((path, identity, reconcile_job_identity(registry, identity)))
    return jobs


def _run_locked(args: argparse.Namespace, config: AppConfig) -> None:
    data_dir = Path(config.data.raw_dir)
    prefix = config.data.synthetic_prefix
    fingerprint = validate_prepared_raw_contract(config.data, augment_prefix=prefix)
    registry = load_augmentation_registry(
        data_dir,
        prefix,
        [item.name for item in fingerprint.files],
    )

    model_path = Path(args.model or find_exported_model(config.training.output_dir)).resolve()
    generator_tokenizer = args.tokenizer or config.data.tokenizer_model
    log(f"Loading generator model: {model_path}")
    model_stat_before = model_path.stat()
    translator = Translator(model_path, generator_tokenizer)
    model_snapshot = snapshot_file(model_path)
    model_stat_after = model_path.stat()
    model_file_identity_before = (
        model_stat_before.st_size,
        model_stat_before.st_mtime_ns,
        model_stat_before.st_ctime_ns,
        model_stat_before.st_dev,
        model_stat_before.st_ino,
    )
    model_file_identity_after = (
        model_stat_after.st_size,
        model_stat_after.st_mtime_ns,
        model_stat_after.st_ctime_ns,
        model_stat_after.st_dev,
        model_stat_after.st_ino,
    )
    if model_file_identity_before != model_file_identity_after:
        raise RuntimeError("The generator model file changed while it was loaded and hashed")
    model_pair = resolve_augmentation_pair(args.language_pair, translator.language_pairs)
    pair = resolve_augmentation_destination(
        model_pair,
        config.data.configured_language_pairs(),
    )
    model_max_seq_len = int(translator.model_config.max_seq_len)
    if args.max_new_tokens > model_max_seq_len:
        raise SystemExit(
            f"--max-new-tokens {args.max_new_tokens} exceeds the model maximum "
            f"length {model_max_seq_len}. Stopping before the ledger is created."
        )
    model_identity, tokenizer_identity = generator_identity(translator, model_snapshot)

    mono_files = _discover_mono_files(Path(args.mono_dir), pair)
    jobs = _build_jobs(
        mono_files,
        pair=pair,
        synthetic_prefix=prefix,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        registry=registry,
    )
    seen_by_direction = registry.mono_hashes_by_direction()
    actionable: list[tuple[Path, AugmentationIdentity, JobProgress]] = []
    finalized_empty = 0
    for path, identity, progress in jobs:
        direction_seen = seen_by_direction.setdefault(identity.training_direction, set())
        has_remaining_text = _source_has_remaining_text(path, progress, direction_seen)
        if progress.eof:
            continue
        if has_remaining_text:
            actionable.append((path, identity, progress))
            continue
        result = run_augmentation_job(
            translator,
            mono_path=path,
            data_dir=data_dir,
            synthetic_prefix=prefix,
            progress=progress,
            accepted_budget=1,
            batch_size=args.batch_size,
            seen_mono_hashes=direction_seen,
        )
        if result.written:
            raise RuntimeError("blank-only augmentation finalization unexpectedly wrote a row")
        finalized_empty += 1
    if not actionable:
        if finalized_empty:
            log(
                f"Finalized {finalized_empty:,} jobs containing empty or already "
                "published input at EOF."
            )
        else:
            log("No new monolingual sentences require processing.")
        return

    preflight_backtranslation_directions(
        pair,
        [(path, identity.mono_language) for path, identity, _ in actionable],
        translator.translation_directions,
        config.data.configured_translation_directions(),
    )
    directions = tuple(dict.fromkeys(identity.training_direction for _, identity, _ in actionable))
    prepared = count_prepared_direction_pairs(config.data.dataset_dir, directions)
    pending = registry.pending_direction_counts()
    total_written = 0

    max_source_tokens = model_max_seq_len - 2

    def source_fits(text: str) -> bool:
        return len(translator.tokenizer.encode(text)) <= max_source_tokens

    for path, identity, progress in actionable:
        direction = identity.training_direction
        counts = prepared[direction]
        existing_synthetic = counts.synthetic + pending.get(direction, 0)
        budget = synthetic_budget(counts.real, existing_synthetic, args.max_ratio)
        log(
            f"{direction[0]}→{direction[1]}: real train {counts.real:,} / "
            f"existing or pending synthetic {existing_synthetic:,} / ratio "
            f"{args.max_ratio:g} → maximum {budget:,} rows"
        )
        if budget <= 0:
            continue
        result = run_augmentation_job(
            translator,
            mono_path=path,
            data_dir=data_dir,
            synthetic_prefix=prefix,
            progress=progress,
            accepted_budget=budget,
            batch_size=args.batch_size,
            seen_mono_hashes=seen_by_direction.setdefault(direction, set()),
            source_fits=source_fits,
        )
        pending[direction] = pending.get(direction, 0) + result.written
        total_written += result.written
        log(
            f"{path.name}: published {result.written:,} rows / quality-filtered "
            f"{result.quality_filtered:,} / duplicates {result.duplicates:,} / too long "
            f"{result.too_long:,}"
        )

    if total_written == 0:
        log("No rows were published because direction limits were reached or input ended.")
        return
    log(
        f"Complete: {total_written:,} total rows. The next sion-train run will "
        "use them only for training with synthetic sampling weight "
        f"{config.data.synthetic_sampling_weight:g}."
    )


def main() -> None:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.max_ratio) or args.max_ratio < 0:
        parser.error("--max-ratio must be finite and non-negative")
    if args.num_beams < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--num-beams, --batch-size, and --max-new-tokens must be positive")

    config_path = args.config or (
        DEFAULT_CONFIG_FILE if Path(DEFAULT_CONFIG_FILE).exists() else None
    )
    config = config_from_raw(load_raw_config(config_path) if config_path else {})
    config.validate()
    if config.data.synthetic_sampling_weight == 0:
        raise SystemExit(
            "data.synthetic_sampling_weight is 0, so generated rows would not be "
            "trained. Augmentation will not run."
        )

    # Training/prepare holds the same dataset-parent lease. Acquire it before
    # constructing Translator so two augment processes cannot load large models
    # onto one GPU and only then discover that one must stop.
    lock_roots = (
        Path(config.data.raw_dir).resolve(),
        Path(config.data.dataset_dir).resolve().parent,
    )
    with artifact_locks(lock_roots):
        _run_locked(args, config)


if __name__ == "__main__":
    main()
