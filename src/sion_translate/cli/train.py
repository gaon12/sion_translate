"""Fully automated sion_translate training entry point.

    sion-train            # This is the only command required.

Workflow:
    1. Detect the execution environment, including GPU count, VRAM, bf16, and CPU.
    2. Load sion_translate.yaml from the project root when present. Missing values
       remain automatic; --config can select another file.
    3. Discover data/*.jsonl, train a tokenizer when needed, and rebuild prepared
       data when source files change.
    4. Select a smooth token-scaled architecture, step count, batch size, and
       related values from the prepared data and available hardware.
    5. Resume each stage automatically from checkpoints/latest when possible.
    6. Run SFT pretraining and store its artifacts under pretrain/.
    7. Run composite-reward MRT and multi-candidate preference training, storing
       those artifacts separately under posttrain/.

Every stage prints a "[sion] ..." message so progress is visible in the terminal.
"""

# CLI configuration, CUDA properties, DataLoader internals, and torch.compile
# are runtime-selected integration points with incomplete annotations.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import secrets
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, Callable, Iterator, Sequence, TypeVar, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from sion_translate.auto import (
    apply_auto_data_settings,
    apply_auto_settings,
    backup_stale_dataset,
    describe_environment,
    estimate_pair_count,
    pick_vocab_size,
    probe_environment,
    scan_raw_data,
    stored_fingerprint,
    synchronize_environment,
)
from sion_translate.config import AppConfig, config_from_raw, load_raw_config
from sion_translate.console import configure_stdio
from sion_translate.data import (
    DistributedBucketBatchSampler,
    IndexedParallelDataset,
    SionBatchCollator,
)
from sion_translate.artifacts import (
    FOUNDATION_STAGE_DIRECTORY,
    MODEL_RELEASE_VERSION,
    TRANSLATION_RELEASE_NAME,
)
from sion_translate.data.collate import load_morphoscript_token_features
from sion_translate.data.integrity import (
    dataset_artifact_problem,
    validate_dataset_artifact_inventory,
)
from sion_translate.data.prepare import prepare_preprocessing_options
from sion_translate.data.reasoning import is_reasoning_jsonl
from sion_translate.locking import artifact_lock, training_run_lock
from sion_translate.foundation import (
    FOUNDATION_LINEAGE_SCHEMA,
    FoundationOutcome,
    FoundationPlan,
    build_foundation_config,
    build_translation_pipeline_identity,
    foundation_run_directory,
    plan_foundation_stage,
)
from sion_translate.fingerprint import DatasetFingerprint, file_sha256
from sion_translate.data.prepare_foundation import foundation_dataset_problem
from sion_translate.language_tags import (
    canonicalize_language_pair,
    canonicalize_language_tags,
)
from sion_translate.model import SionForConditionalGeneration
from sion_translate.tokenizer import (
    SionTokenizer,
    load_tokenizer_metadata,
    tokenizer_split_digits_policy,
    write_tokenizer_metadata,
)
from sion_translate.training.distributed import (
    DistributedContext,
    barrier,
    broadcast_bool,
    broadcast_int,
    broadcast_text,
    cleanup_distributed,
    distributed_failure_scope,
    initialize_distributed,
    parallelize_model,
    resolve_parallel_strategy,
)
from sion_translate.training.checkpoint import (
    DCP_COMPLETION_FILENAME,
    DCP_COMPLETION_SCHEMA,
    checkpoint_generation_bindings,
    checkpoint_generation_candidates,
    checkpoint_path_exists,
    initialize_model_from_checkpoint,
    inspect_checkpoint_identity,
    inspect_checkpoint_training_state,
    preflight_checkpoint_identity,
    preflight_checkpoint_load_structure,
    resolve_checkpoint_source,
    verified_checkpoint_generation_lease,
)
from sion_translate.training.export import export_inference_models, validate_export_directory
from sion_translate.training.objectives import MinimumRiskObjective
from sion_translate.training.trainer import (
    announce,
    build_training_checkpoint_identity,
    train,
)
from sion_translate.performance import build_cpu_plan

DEFAULT_CONFIG_FILE = "sion_translate.yaml"
FINAL_EXPORT_DEPENDENCIES = {
    "int8": ("torchao", "torchao"),
    "gguf_q4_k_m": ("gguf", "gguf-python"),
}
FINAL_EXPORT_STATUS_SCHEMA = "sion-final-export-status-v1"
RANK_ZERO_ACTION_STATUS_SCHEMA = "sion-rank-zero-action-status-v1"
FINAL_EXPORT_STATUS_TIMEOUT_SECONDS = 24 * 60 * 60
RANK_ZERO_ACTION_STALE_TIMEOUT_SECONDS = 15 * 60
RANK_ZERO_ACTION_HEARTBEAT_SECONDS = 30.0
RANK_ZERO_STATUS_FILE_BYTES = 16 * 1024


@contextmanager  # pyright: ignore[reportDeprecated]
def coordinated_training_run_lock(
    output_dir: str | Path,
    context: DistributedContext,
) -> Iterator[Path | None]:
    """Let rank 0 own the run lock and make every peer follow its decision."""

    scope = ExitStack()
    lock_path: Path | None = None
    acquisition_error: Exception | None = None
    if context.is_main:
        try:
            lock_path = scope.enter_context(training_run_lock(output_dir))
        except Exception as error:
            acquisition_error = error

    try:
        acquisition_failed = broadcast_bool(acquisition_error is not None, context)
        if acquisition_failed:
            if acquisition_error is not None:
                raise acquisition_error
            raise RuntimeError(
                "Rank 0 could not acquire the training.output_dir run lock. "
                "Inspect rank 0's error for the current owner and output_dir."
            )
        yield lock_path
    finally:
        scope.close()


def missing_final_export_dependencies(formats: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Return requested final formats whose optional modules are unavailable."""

    missing: dict[str, str] = {}
    for format_name in formats:
        dependency = FINAL_EXPORT_DEPENDENCIES.get(format_name)
        if dependency is None:
            continue
        module_name, distribution_name = dependency
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing[format_name] = distribution_name
    return missing


def preflight_final_export_dependencies(formats: list[str] | tuple[str, ...]) -> None:
    """Fail before data preparation or training when strict export cannot finish."""

    missing = missing_final_export_dependencies(formats)
    if not missing:
        return
    details = ", ".join(f"{format_name} → {package}" for format_name, package in missing.items())
    raise RuntimeError(
        "Dependencies for the requested final export formats are missing: "
        f"{details}. Before starting a long training run, install them with "
        'python -m pip install -e ".[export]" or remove the affected formats '
        "from training.final_export_formats."
    )


def preflight_morphoscript_token_features(
    config: AppConfig,
    tokenizer: SionTokenizer,
) -> None:
    """Require a model-compatible MorphoScript sidecar before training starts."""

    experimental = config.model.experimental
    if not experimental.morphoscript_enabled:
        return
    load_morphoscript_token_features(
        config.data.tokenizer_features,
        vocab_size=len(tokenizer),
        script_classes=experimental.script_classes,
    )


def build_collator_args(
    config: AppConfig,
    tokenizer: SionTokenizer,
) -> dict[str, Any]:
    """Build the common train/validation/post-training collator contract."""

    return {
        "tokenizer": tokenizer,
        "max_source_length": config.data.max_source_length,
        "max_target_length": config.data.max_target_length,
        "pad_to_multiple_of": config.data.pad_to_multiple_of,
        "denoise_noise_density": config.data.denoise_noise_density,
        "denoise_mean_span": config.data.denoise_mean_span,
        "source_only_languages": config.data.configured_source_only_languages(),
        "augmentation_seed": config.training.seed,
        "token_features": (
            config.data.tokenizer_features
            if config.model.experimental.morphoscript_enabled
            else None
        ),
        "script_classes": config.model.experimental.script_classes,
    }


def scan_configured_raw_data(
    config: AppConfig,
    data_dir: Path,
    tokenizer_path: Path,
) -> DatasetFingerprint:
    """Fingerprint every input that can change the prepared dataset."""

    from sion_translate.augmentation import load_augmentation_registry

    load_augmentation_registry(data_dir, config.data.synthetic_prefix, ())
    language_pairs = config.data.configured_language_pairs()
    preprocessing_options = prepare_preprocessing_options(
        approximate_split=config.data.approximate_split,
        source_only_languages=config.data.configured_source_only_languages(),
        translation_directions=config.data.configured_translation_directions(),
        train_only_prefixes=config.data.configured_synthetic_prefixes(),
        managed_augmentation_prefix=config.data.synthetic_prefix,
        synthetic_sampling_weight=config.data.synthetic_sampling_weight,
        language_pair_count=len(language_pairs),
    )
    return scan_raw_data(
        data_dir,
        language_pairs=language_pairs,
        tokenizer_model=tokenizer_path,
        preprocessing_options=preprocessing_options,
    )


def preflight_dataset_direction_contract(
    config: AppConfig,
    *datasets: IndexedParallelDataset,
    require_all_pairs: bool = False,
    require_all_directions: bool = False,
) -> None:
    """Refuse shards whose materialized graph differs from the run config."""

    expected = config.data.configured_translation_directions()
    expected_pairs = config.data.configured_language_pairs()
    for dataset in datasets:
        if dataset.language_pairs != expected_pairs:
            raise ValueError(
                "prepared dataset language pairs differ from the training config: "
                f"dataset={dataset.language_pairs}, config={expected_pairs}"
            )
        if dataset.translation_directions != expected:
            raise ValueError(
                "prepared dataset translation directions differ from the training config: "
                f"dataset={dataset.translation_directions}, config={expected}"
            )
        if require_all_pairs:
            observed = set(dataset.observed_language_pairs)
            missing = [pair for pair in expected_pairs if pair not in observed]
            if missing:
                raise ValueError(
                    "prepared training split has no accepted rows for configured language "
                    f"pairs: missing={missing!r}"
                )
        if require_all_directions:
            observed_directions = set(
                dataset.observed_translation_directions_for_physical_mask(
                    np.ones(dataset.pair_count, dtype=np.bool_)
                )
            )
            missing_directions = [
                direction for direction in expected if direction not in observed_directions
            ]
            if missing_directions:
                raise ValueError(
                    "prepared validation split has no held-out rows for configured "
                    f"translation directions: missing={missing_directions!r}"
                )


def preflight_effective_translation_training(
    config: AppConfig,
    sampler: DistributedBucketBatchSampler,
    *,
    authenticated_revision_directions: Sequence[Sequence[str]] = (),
) -> None:
    """Refuse advertised edges that have zero effective translation probability.

    Exact revision rows retain a translation objective even at full denoising because
    the collator excludes their authenticated ``<draft>`` structure from denoising.
    """

    dataset = sampler.dataset
    positive_mask = sampler.positive_sampling_pair_mask()
    configured_directions = config.data.configured_translation_directions()
    observed = set(dataset.observed_language_pairs_for_physical_mask(positive_mask))
    missing_pairs = [
        pair for pair in config.data.configured_language_pairs() if pair not in observed
    ]
    if missing_pairs:
        raise ValueError(
            "sampling policy assigns zero probability to every training row for configured "
            f"language pairs: missing={missing_pairs!r}"
        )
    observed_directions = set(
        dataset.observed_translation_directions_for_physical_mask(positive_mask)
    )
    missing_directions = [
        direction for direction in configured_directions if direction not in observed_directions
    ]
    if missing_directions:
        raise ValueError(
            "sampling policy and row-scoped direction metadata provide zero effective "
            f"examples for configured translation directions: missing={missing_directions!r}"
        )
    revision_edges = tuple(
        canonicalize_language_pair(
            direction,
            field=f"authenticated revision direction[{index}]",
        )
        for index, direction in enumerate(authenticated_revision_directions)
    )
    if len(revision_edges) != len(set(revision_edges)):
        raise ValueError("authenticated revision directions contain duplicate edges")
    configured_edges = set(configured_directions)
    unexpected_revision_edges = [
        direction for direction in revision_edges if direction not in configured_edges
    ]
    if unexpected_revision_edges:
        raise ValueError(
            "authenticated revision directions are outside the configured translation graph: "
            f"unexpected={unexpected_revision_edges!r}"
        )
    if config.data.denoise_probability >= 1.0:
        source_only = set(config.data.configured_source_only_languages())
        revision_edge_set = set(revision_edges)
        zero_translation_edges = [
            direction
            for direction in configured_directions
            if direction[0] not in source_only and direction not in revision_edge_set
        ]
        if zero_translation_edges:
            raise ValueError(
                "denoise_probability=1.0 leaves no translation objective for configured "
                f"directions: {zero_translation_edges!r}"
            )


def _foundation_languages_with_positive_sampling_mass(
    dataset: IndexedParallelDataset,
    sampler: DistributedBucketBatchSampler,
    planned_languages: Sequence[str],
) -> tuple[str, ...]:
    """Return only self-pair languages the finalized foundation sampler can select."""

    if sampler.dataset is not dataset:
        raise ValueError("foundation sampler must describe the inspected training dataset")
    positive_mask = sampler.positive_sampling_pair_mask()
    observed_pairs = dataset.observed_language_pairs_for_physical_mask(positive_mask)
    non_self_pairs = [pair for pair in observed_pairs if pair[0] != pair[1]]
    if non_self_pairs:
        raise ValueError(
            f"foundation training rows must be self-pairs: observed={non_self_pairs!r}"
        )
    observed_languages = {source for source, _ in observed_pairs}
    planned = tuple(planned_languages)
    if len(planned) != len(set(planned)):
        raise ValueError("configured foundation language reservation contains duplicates")
    unexpected = sorted(observed_languages - set(planned))
    if unexpected:
        raise ValueError(
            "foundation training rows contain languages outside the configured reservation: "
            f"{unexpected!r}"
        )
    languages = tuple(language for language in planned if language in observed_languages)
    if not languages:
        raise ValueError(
            "foundation sampling policy assigns zero probability to every self-pair language"
        )
    return languages


def _foundation_plan_for_lineage(
    foundation_plan: Any,
    foundation_lineage: dict[str, Any],
) -> FoundationPlan:
    """Build the narrow plan view expected by the pipeline identity validator.

    The discovery plan remains the tokenizer/reservation contract. Published
    lineage is narrower: it may name only languages with positive train-split
    sampling mass.
    """

    raw_languages = foundation_lineage.get("languages")
    if not isinstance(raw_languages, list) or not all(
        isinstance(language, str) for language in raw_languages
    ):
        raise ValueError("foundation lineage has no valid effective language list")
    effective = tuple(cast(list[str], raw_languages))
    configured = tuple(cast(Sequence[str], foundation_plan.languages))
    effective_set = set(effective)
    if not effective or len(effective) != len(effective_set):
        raise ValueError("foundation lineage effective languages must be non-empty and unique")
    if effective != tuple(language for language in configured if language in effective_set):
        raise ValueError(
            "foundation lineage effective languages must be an ordered subset of the "
            "configured foundation reservation"
        )
    return cast(
        FoundationPlan,
        SimpleNamespace(enabled=True, languages=effective),
    )


def resolve_training_revision_directions(
    config: AppConfig,
    dataset: IndexedParallelDataset,
    *,
    draft_token_id: int | None,
    max_source_tokens: int,
    physical_mask: np.ndarray | None = None,
) -> tuple[tuple[str, str], ...]:
    """Resolve revision capability without widening it beyond authenticated rows."""

    explicit = config.data.configured_revision_directions()
    derived = dataset.detect_revision_directions(
        draft_token_id=draft_token_id,
        max_source_tokens=max_source_tokens,
        physical_mask=physical_mask,
    )
    if explicit:
        explicit_set = set(explicit)
        derived_set = set(derived)
        missing = [direction for direction in derived if direction not in explicit_set]
        unsupported = [direction for direction in explicit if direction not in derived_set]
        if missing or unsupported:
            raise ValueError(
                "data.revision_directions must exactly match revision-marked indexed rows; "
                f"missing={missing!r}; unsupported={unsupported!r}"
            )
        resolved = explicit
    else:
        resolved = derived
    if config.data.revision_examples and not resolved:
        raise ValueError(
            "data.revision_examples=true has no revision-marked indexed rows; "
            "ingest authenticated revision provenance before enabling it"
        )
    config.data.revision_directions = [list(direction) for direction in resolved]
    config.data.revision_examples = bool(resolved)
    return resolved


def dataloader_runtime_kwargs(
    num_workers: int,
    device: torch.device,
    *,
    training: bool,
) -> dict[str, Any]:
    """Build stage-specific loader settings without retaining idle worker pools."""

    workers = max(0, num_workers)
    options: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        options.update(
            {
                "persistent_workers": training,
                "prefetch_factor": 4 if training else 2,
            }
        )
    return options


def shutdown_dataloader(loader: DataLoader[Any] | None) -> None:
    """Stop a persistent DataLoader pool before constructing the next stage."""

    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)  # pyright: ignore[reportPrivateUsage]
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if iterator is not None:
        loader._iterator = None  # pyright: ignore[reportPrivateUsage]


def release_stage_resources(
    context: DistributedContext,
    *loaders: DataLoader[Any] | None,
) -> dict[str, float]:
    """Release CPU workers and CUDA cache at a pretrain/posttrain boundary."""

    for loader in loaders:
        shutdown_dataloader(loader)
    gc.collect()
    if context.device.type != "cuda":
        return {}
    torch.cuda.synchronize(context.device)
    before_allocated = torch.cuda.memory_allocated(context.device) / 2**30
    before_reserved = torch.cuda.memory_reserved(context.device) / 2**30
    torch.cuda.empty_cache()
    after_allocated = torch.cuda.memory_allocated(context.device) / 2**30
    after_reserved = torch.cuda.memory_reserved(context.device) / 2**30
    torch.cuda.reset_peak_memory_stats(context.device)
    return {
        "before_allocated_gib": before_allocated,
        "before_reserved_gib": before_reserved,
        "after_allocated_gib": after_allocated,
        "after_reserved_gib": after_reserved,
    }


def requires_ddp_unused_parameter_detection(config: AppConfig) -> bool:
    """Return whether one DDP wrapper spans changing parameter-use graphs."""

    experimental = config.model.experimental
    bats_unused_during_sft = experimental.bats_enabled and (
        experimental.bats_loss_weight == 0 and experimental.bats_coverage_weight == 0
    )
    # MRT interleaves label-free candidate forwards with a supervised reference
    # forward. BATS and semantic parity only run on the latter, so the same DDP
    # wrapper observes a changing parameter-use graph and cannot be static.
    supervised_only_during_mrt = config.posttraining.enabled and (
        experimental.bats_enabled or experimental.semantic_parity_enabled
    )
    return bats_unused_during_sft or supervised_only_during_mrt


def validate_training_capacity(
    parameter_count: int,
    context: DistributedContext,
    *,
    parallel_strategy: str,
    ema_enabled: bool,
    per_gpu_vram_gib: float | None = None,
) -> dict[str, float | int] | None:
    """Fail before allocation when persistent training state consumes H100 headroom."""

    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    if per_gpu_vram_gib is None:
        if context.device.type != "cuda":
            return None
        per_gpu_vram_gib = torch.cuda.get_device_properties(context.device).total_memory / 2**30
    assert per_gpu_vram_gib is not None
    if per_gpu_vram_gib <= 0:
        raise ValueError("per_gpu_vram_gib must be positive")

    # FP32 master parameter + gradient + AdamW first/second moments = 16 B.
    # EMA adds another FP32 shard. Reserve just over half of VRAM for BF16 layer
    # all-gathers, activations, temporary kernels, and the CUDA context.
    bytes_per_parameter = 16 + (4 if ema_enabled else 0)
    sharding_factor = context.world_size if parallel_strategy == "fsdp2" else 1
    total_state_gib = parameter_count * bytes_per_parameter / 2**30
    per_rank_state_gib = total_state_gib / sharding_factor
    state_budget_gib = per_gpu_vram_gib * 0.49
    minimum_world_size = (
        math.ceil(total_state_gib / state_budget_gib)
        if parallel_strategy == "fsdp2"
        else context.world_size
    )
    report: dict[str, float | int] = {
        "bytes_per_parameter": bytes_per_parameter,
        "total_state_gib": total_state_gib,
        "per_rank_state_gib": per_rank_state_gib,
        "state_budget_gib": state_budget_gib,
        "minimum_world_size": minimum_world_size,
    }
    if per_rank_state_gib > state_budget_gib:
        strategy_hint = (
            f"Use at least {minimum_world_size} GPUs with FSDP2"
            if parallel_strategy == "fsdp2"
            else "Switch to FSDP2 or use a smaller model"
        )
        ema_hint = ", disable EMA (training.ema_decay=0)" if ema_enabled else ""
        raise RuntimeError(
            "Estimated persistent training state leaves insufficient accelerator "
            f"headroom: {per_rank_state_gib:.1f} GiB/rank versus a "
            f"{state_budget_gib:.1f} GiB safety budget on {per_gpu_vram_gib:.1f} GiB GPUs. "
            f"{strategy_hint}{ema_hint}, or use an explicitly validated lower-memory "
            "optimizer/offload policy."
        )
    return report


def construct_training_model(
    config: AppConfig,
    context: DistributedContext,
    *,
    pad_id: int,
    parallel_strategy: str,
) -> tuple[
    SionForConditionalGeneration,
    int,
    dict[str, float | int] | None,
    bool,
]:
    """Count and capacity-check CUDA models before allocating parameter storage."""

    materialize_meta = context.device.type == "cuda"
    construction_device: torch.device | str = "meta" if materialize_meta else context.device
    with torch.device(construction_device):
        model = SionForConditionalGeneration(config.model, pad_id=pad_id)
    parameter_count = model.parameter_count()
    capacity = validate_training_capacity(
        parameter_count,
        context,
        parallel_strategy=parallel_strategy,
        ema_enabled=config.training.ema_decay > 0,
    )
    return model, parameter_count, capacity, materialize_meta


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a JSON control file without publishing partial bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _control_status_paths(status_path: Path) -> tuple[Path, Path]:
    return status_path, status_path.with_name(f"{status_path.name}.backup")


def _bounded_status_text(value: object, *, max_bytes: int = 6000) -> str:
    """Fit diagnostic text in a status envelope using a UTF-8 byte budget."""

    normalized = "".join(character if ord(character) >= 32 else " " for character in str(value))
    encoded = normalized.encode("utf-8")
    if len(encoded) <= max_bytes:
        return normalized
    marker = "…"
    tail_budget = max(0, max_bytes - len(marker.encode("utf-8")))
    tail = encoded[-tail_budget:].decode("utf-8", errors="ignore") if tail_budget else ""
    return marker + tail


def _encode_control_status(payload: dict[str, Any]) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) + 1 > RANK_ZERO_STATUS_FILE_BYTES:
        raise ValueError("rank-zero status payload exceeds its preallocated file size")
    return encoded + b" " * (RANK_ZERO_STATUS_FILE_BYTES - len(encoded) - 1) + b"\n"


def _overwrite_control_status(handle: BinaryIO, payload: dict[str, Any]) -> None:
    encoded = _encode_control_status(payload)
    handle.seek(0)
    written = handle.write(encoded)
    if written != len(encoded):
        raise OSError(f"short rank-zero status write: {written}/{len(encoded)} bytes")
    handle.flush()
    os.fsync(handle.fileno())


def _initialize_control_status(
    status_path: Path,
    payload: dict[str, Any],
) -> tuple[BinaryIO, BinaryIO]:
    """Preallocate two writable status channels before a long rank-zero action."""

    handles: list[BinaryIO] = []
    try:
        for path in _control_status_paths(status_path):
            _atomic_write_json(path, payload)
            handle = path.open("r+b", buffering=0)
            _overwrite_control_status(handle, payload)
            handles.append(handle)
    except BaseException:
        for handle in handles:
            handle.close()
        raise
    return handles[0], handles[1]


def _publish_control_status(
    handles: tuple[BinaryIO, BinaryIO],
    payload: dict[str, Any],
) -> None:
    failures: list[BaseException] = []
    successes = 0
    for handle in handles:
        try:
            _overwrite_control_status(handle, payload)
            successes += 1
        except BaseException as error:
            failures.append(error)
    if successes == 0:
        raise RuntimeError("both preallocated rank-zero status channels failed") from failures[0]


def _close_control_status(handles: tuple[BinaryIO, BinaryIO] | None) -> None:
    if handles is None:
        return
    for handle in handles:
        try:
            handle.close()
        except OSError:
            pass


def _read_control_status(status_path: Path) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for candidate in _control_status_paths(status_path):
        try:
            raw_status: object = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(raw_status, dict):
            statuses.append(cast(dict[str, Any], raw_status))
    return statuses


def _control_status_is_visible(
    status_path: Path,
    *,
    schema: str,
    invocation: str,
) -> bool:
    return any(
        status.get("schema") == schema
        and status.get("invocation") == invocation
        and status.get("state") == "running"
        for status in _read_control_status(status_path)
    )


def _wait_for_final_export_status(
    status_path: Path,
    *,
    step: int,
    release_name: str,
    invocation: str,
) -> None:
    """Wait without a process-group timeout while rank 0 performs strict conversion."""

    deadline = time.monotonic() + FINAL_EXPORT_STATUS_TIMEOUT_SECONDS
    while True:
        for status in _read_control_status(status_path):
            matches_run = (
                status.get("schema") == FINAL_EXPORT_STATUS_SCHEMA
                and status.get("step") == step
                and status.get("release_name") == release_name
                and status.get("invocation") == invocation
            )
            if matches_run and status.get("state") == "complete":
                return
            if matches_run and status.get("state") == "failed":
                error_type = status.get("error_type", "RuntimeError")
                message = status.get("message", "unknown rank-0 export failure")
                raise RuntimeError(f"rank 0 final export failed: {error_type}: {message}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for rank 0 final export after "
                f"{FINAL_EXPORT_STATUS_TIMEOUT_SECONDS:g} seconds: {status_path}"
            )
        time.sleep(0.25)


def export_final_model(
    model: torch.nn.Module,
    config: AppConfig,
    context: DistributedContext,
    run_root: Path,
    *,
    stage: str,
    step: int,
    formats: Sequence[str] | None = None,
    release_name: str = TRANSLATION_RELEASE_NAME,
    translation_capable: bool = True,
    languages: Sequence[str] | None = None,
    translation_directions: Sequence[Sequence[str]] | None = None,
    authenticated_revision_directions: Sequence[Sequence[str]] | None = None,
    pipeline_identity: dict[str, Any] | None = None,
) -> Path:
    """Create the required final format set from the restored best weights."""

    export_dir = run_root / stage / "exports" / "best"
    status_path = export_dir.parent / f".{export_dir.name}.strict-export-status.json"
    requested_formats = tuple(
        formats if formats is not None else config.training.final_export_formats
    )
    if translation_capable and translation_directions is None:
        translation_directions = config.data.configured_translation_directions()
    if not translation_capable:
        if authenticated_revision_directions:
            raise ValueError("foundation exports cannot contain revision directions")
        export_revision_directions: tuple[tuple[str, str], ...] = ()
    else:
        configured_revision_directions = config.data.configured_revision_directions()
        if authenticated_revision_directions is None:
            if configured_revision_directions or config.data.revision_examples:
                raise ValueError(
                    "translation final export requires authenticated_revision_directions "
                    "from the indexed training provenance"
                )
            export_revision_directions = ()
        else:
            export_revision_directions = tuple(
                canonicalize_language_pair(
                    direction,
                    field=f"authenticated revision direction[{index}]",
                )
                for index, direction in enumerate(authenticated_revision_directions)
            )
            if len(set(export_revision_directions)) != len(export_revision_directions):
                raise ValueError("authenticated_revision_directions contains duplicate edges")
            if export_revision_directions != configured_revision_directions:
                raise ValueError(
                    "authenticated_revision_directions must exactly match the resolved data "
                    "revision graph"
                )
            if config.data.revision_examples is not bool(export_revision_directions):
                raise ValueError(
                    "data.revision_examples must exactly reflect authenticated_revision_directions"
                )
    status_handles: tuple[BinaryIO, BinaryIO] | None = None
    invocation = "single-process"
    if context.distributed:
        invocation = broadcast_text(
            secrets.token_hex(16) if context.is_main else None,
            context,
        )
        initialized_handles = _run_rank_zero_action(
            context,
            lambda: _initialize_control_status(
                status_path,
                {
                    "schema": FINAL_EXPORT_STATUS_SCHEMA,
                    "state": "running",
                    "invocation": invocation,
                    "step": step,
                    "release_name": release_name,
                    "formats": list(requested_formats),
                },
            ),
            description="publishing final export start state",
        )
        if context.is_main:
            assert initialized_handles is not None
            status_handles = initialized_handles
        visibility_scope = distributed_failure_scope(
            not _control_status_is_visible(
                status_path,
                schema=FINAL_EXPORT_STATUS_SCHEMA,
                invocation=invocation,
            ),
            context,
        )
        if visibility_scope != "none":
            _close_control_status(status_handles)
            raise RuntimeError(
                "final export status is not coherently visible to every distributed rank"
            )
    try:
        export_inference_models(
            export_dir,
            model,
            config.model,
            context,
            step=step,
            formats=requested_formats,
            release_name=release_name,
            translation_capable=translation_capable,
            pipeline_identity=pipeline_identity,
            tokenizer_path=config.data.tokenizer_model,
            token_features_path=(
                config.data.tokenizer_features
                if config.model.experimental.morphoscript_enabled
                else None
            ),
            language_pairs=(
                config.data.configured_language_pairs() if translation_capable else None
            ),
            languages=languages,
            translation_directions=translation_directions,
            bidirectional=config.data.bidirectional,
            revision_directions=export_revision_directions,
            revision_trained=bool(export_revision_directions),
            strict=True,
        )
    except BaseException as error:
        if context.distributed and context.is_main and status_handles is not None:
            try:
                _publish_control_status(
                    status_handles,
                    {
                        "schema": FINAL_EXPORT_STATUS_SCHEMA,
                        "state": "failed",
                        "invocation": invocation,
                        "step": step,
                        "release_name": release_name,
                        "error_type": type(error).__name__,
                        "message": _bounded_status_text(error),
                    },
                )
            finally:
                _close_control_status(status_handles)
        raise
    if context.distributed:
        if context.is_main:
            assert status_handles is not None
            try:
                _publish_control_status(
                    status_handles,
                    {
                        "schema": FINAL_EXPORT_STATUS_SCHEMA,
                        "state": "complete",
                        "invocation": invocation,
                        "step": step,
                        "release_name": release_name,
                        "formats": list(requested_formats),
                    },
                )
            finally:
                _close_control_status(status_handles)
        else:
            _wait_for_final_export_status(
                status_path,
                step=step,
                release_name=release_name,
                invocation=invocation,
            )
    return export_dir


def find_existing_checkpoint(config: AppConfig) -> Path | None:
    """Find any checkpoint that constrains the tokenizer vocabulary identity."""

    if config.training.resume_from:
        explicit = Path(config.training.resume_from)
        if explicit.exists():
            return explicit
    run_root = Path(config.training.output_dir)
    for stage_root in (
        run_root,
        run_root / "foundation",
        run_root / "pretrain",
        run_root / "posttrain",
    ):
        checkpoint_root = stage_root / "checkpoints"
        if not checkpoint_root.is_dir():
            continue
        for candidate in sorted(checkpoint_root.iterdir()):
            if checkpoint_path_exists(candidate):
                return candidate
    return None


def tokenizer_policy_problem(
    tokenizer_path: str | Path,
    language_pairs: tuple[tuple[str, str], ...],
    foundation_languages: tuple[str, ...] | None = None,
    reasoning_languages: tuple[str, ...] = (),
    *,
    translation_directions: tuple[tuple[str, str], ...] | None = None,
    require_recorded_directions: bool = False,
) -> str | None:
    """Return a concrete compatibility problem for a tokenizer, if any."""

    tokenizer_path = Path(tokenizer_path)
    try:
        tokenizer = SionTokenizer(tokenizer_path)
        metadata = load_tokenizer_metadata(tokenizer_path)
        recorded_policy = tokenizer_split_digits_policy(tokenizer_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return f"Could not read tokenizer policy/metadata: {exc}"

    if not tokenizer.splits_digits:
        return (
            "The tokenizer does not split multi-digit numbers into individual "
            "digits (split_digits=False behavior)"
        )
    if recorded_policy is False:
        return "tokenizer_metadata.json records split_digits=false"
    if recorded_policy is None or metadata is None:
        return "tokenizer_metadata.json version 2 or newer is missing"

    recorded_hash = metadata.get("model_sha256")
    if recorded_hash != file_sha256(tokenizer_path):
        return "tokenizer_metadata.json model_sha256 does not match the model file"
    vocab_path = tokenizer_path.with_suffix(".vocab")
    if not vocab_path.is_file() or metadata.get("vocab_sha256") != file_sha256(vocab_path):
        return "tokenizer_metadata.json vocab_sha256 does not match the vocabulary file"
    raw_pairs = metadata.get("language_pairs")
    recorded_pairs = (
        tuple((str(pair[0]), str(pair[1])) for pair in raw_pairs)
        if isinstance(raw_pairs, list)
        and all(isinstance(pair, list) and len(pair) == 2 for pair in raw_pairs)
        else ()
    )
    if recorded_pairs != language_pairs:
        return (
            "tokenizer_metadata.json language_pairs differs from the current configuration "
            f"(metadata={recorded_pairs}, config={language_pairs})"
        )
    raw_directions = metadata.get("translation_directions")
    if raw_directions is not None and (
        not isinstance(raw_directions, list)
        or not raw_directions
        or not all(
            isinstance(direction, list) and len(direction) == 2 for direction in raw_directions
        )
    ):
        return "tokenizer_metadata.json translation_directions has an invalid format"
    recorded_directions = (
        tuple((str(direction[0]), str(direction[1])) for direction in raw_directions)
        if isinstance(raw_directions, list)
        and all(isinstance(direction, list) and len(direction) == 2 for direction in raw_directions)
        else ()
    )
    expected_directions = translation_directions or ()
    if require_recorded_directions and not recorded_directions:
        return (
            "Tokenizer metadata cannot authenticate the explicit "
            "translation_directions. Retrain with the same direction policy "
            "instead of adding provenance to an existing tokenizer afterward."
        )
    if recorded_directions and expected_directions and recorded_directions != expected_directions:
        return (
            "tokenizer_metadata.json translation_directions differs from the "
            "current configuration "
            f"(metadata={recorded_directions}, config={expected_directions})"
        )
    expected_languages = {language for pair in language_pairs for language in pair}
    if set(tokenizer.languages) != expected_languages:
        return (
            "The tokenizer control-token language set differs from the current configuration "
            f"(tokenizer={sorted(tokenizer.languages)}, config={sorted(expected_languages)})"
        )
    # Tokenizer training creates denoising controls for every translation
    # language, then adds any foundation-only languages. A configured
    # foundation subset must therefore never remove translation controls.
    expected_denoise_languages = expected_languages | set(foundation_languages or ())
    if set(tokenizer.denoise_tags) != expected_denoise_languages:
        return (
            "The tokenizer denoising-tag language set differs from the foundation configuration "
            f"(tokenizer={sorted(tokenizer.denoise_tags)}, "
            f"config={sorted(expected_denoise_languages)})"
        )
    tokenizer_reasoning_languages = set(getattr(tokenizer, "reasoning_tags", {}))
    if tokenizer_reasoning_languages != set(reasoning_languages):
        return (
            "The tokenizer reasoning-tag language set differs from the structured corpus "
            f"(tokenizer={sorted(tokenizer_reasoning_languages)}, "
            f"corpus={sorted(reasoning_languages)})"
        )
    return None


def _prepared_foundation_manifest_policy(
    config: AppConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read language controls from an offline prepared foundation generation."""

    manifest_path = Path(config.foundation.dataset_dir) / "manifest.json"
    try:
        raw_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "The configured foundation source corpus is offline and the prepared "
            f"foundation manifest cannot be read: {manifest_path}"
        ) from error
    if not isinstance(raw_manifest, dict):
        raise RuntimeError("The prepared foundation manifest must be a JSON object")
    manifest = cast(dict[str, Any], raw_manifest)
    raw_languages = manifest.get("languages")
    if not isinstance(raw_languages, list):
        raise RuntimeError("The prepared foundation manifest has no language list")
    try:
        languages = canonicalize_language_tags(
            cast(list[object], raw_languages),
            field="prepared foundation languages",
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if list(languages) != raw_languages or not languages:
        raise RuntimeError("The prepared foundation language list must be non-empty and canonical")
    configured = config.foundation_languages()
    language_set = set(languages)
    if languages != tuple(language for language in configured if language in language_set):
        raise RuntimeError(
            "The prepared foundation languages are not an ordered subset of the "
            "configured foundation reservation"
        )
    if config.foundation.require_all_languages and languages != configured:
        raise RuntimeError(
            "foundation.require_all_languages=true, but the prepared foundation "
            "dataset does not cover every configured language"
        )

    raw_reasoning = manifest.get("reasoning")
    if not isinstance(raw_reasoning, dict):
        raise RuntimeError("The prepared foundation manifest has no reasoning policy")
    raw_reasoning_languages = cast(dict[str, Any], raw_reasoning).get("languages")
    if not isinstance(raw_reasoning_languages, list):
        raise RuntimeError("The prepared foundation reasoning language list is invalid")
    try:
        reasoning_languages = canonicalize_language_tags(
            cast(list[object], raw_reasoning_languages),
            field="prepared foundation reasoning languages",
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if list(reasoning_languages) != raw_reasoning_languages or not set(
        reasoning_languages
    ).issubset(language_set):
        raise RuntimeError(
            "The prepared foundation reasoning languages must be canonical members "
            "of its language list"
        )
    return languages, reasoning_languages


def _preflight_offline_foundation_dataset(
    config: AppConfig,
    foundation_plan: Any,
) -> tuple[str, ...]:
    """Authenticate prepared foundation shards when their raw corpus is omitted."""

    languages, reasoning_languages = _prepared_foundation_manifest_policy(config)
    if languages != tuple(
        language for language in foundation_plan.languages if language in set(languages)
    ):
        raise RuntimeError(
            "The offline foundation plan and prepared dataset reserve different languages"
        )
    from sion_translate.data.prepare_foundation import foundation_dataset_problem

    problem = foundation_dataset_problem(
        config.foundation.dataset_dir,
        foundation_plan.discovery,
        config.data.tokenizer_model,
        minimum_characters=config.foundation.minimum_characters,
        maximum_characters=config.foundation.maximum_characters,
        max_tokens=config.data.max_source_length - 2,
        max_target_tokens=config.data.max_target_length - 1,
        deduplicate=config.foundation.deduplicate,
        shard_size=config.foundation.shard_size,
        validation_fraction=config.foundation.validation_fraction,
        reasoning_sample_share=config.foundation.reasoning_sample_share,
        language_sampling_alpha=config.foundation.language_sampling_alpha,
        minimum_language_share=config.foundation.minimum_language_share,
        release_name=config.foundation.release_name,
        allow_offline_sources=True,
    )
    if problem is not None:
        raise RuntimeError(
            "The prepared foundation dataset is incompatible or corrupt, and its raw "
            f"source corpus is unavailable for a rebuild: {problem}"
        )
    return reasoning_languages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train sion_translate. With no arguments, detect the environment and "
            "data automatically."
        )
    )
    parser.add_argument(
        "--config",
        help=(
            f"Configuration file (default: {DEFAULT_CONFIG_FILE} in the project "
            "root; fully automatic when absent)"
        ),
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of complete SFT passes over the training dataset"
    )
    parser.add_argument(
        "--max-steps", type=int, help="Manual maximum step count; overrides automation"
    )
    parser.add_argument(
        "--posttrain-epochs",
        type=int,
        help="Number of complete MRT passes over the training dataset",
    )
    parser.add_argument("--posttrain-steps", type=int, help="Manual MRT posttraining step count")
    parser.add_argument(
        "--skip-posttraining",
        action="store_true",
        help="Stop after SFT pretraining",
    )
    parser.add_argument(
        "--resume-from",
        help="Checkpoint to resume manually (default: discover latest automatically)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare only the tokenizer and dataset shards, then exit before training",
    )
    return parser


def seed_everything(seed: int, rank: int) -> None:
    """Seed every random-number generator for reproducible training.

    The rank is added because runtime randomness such as dropout must differ
    between ranks during distributed training.
    """
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32))
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def preflight_embedded_bundle_config(
    requested_config: str | None,
    *,
    override_flags: tuple[str, ...] = (),
    root: Path | None = None,
) -> None:
    """Require an extracted GPU bundle to use its authenticated effective config."""

    bundle_root = (root or Path.cwd()).resolve()
    manifest_path = bundle_root / "PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        return
    try:
        raw_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read the embedded GPU bundle manifest: {manifest_path}"
        ) from error
    if not isinstance(raw_manifest, dict):
        raise RuntimeError("The embedded GPU bundle manifest must be a JSON object")
    raw_contract = cast(dict[str, Any], raw_manifest).get("training_contract")
    if not isinstance(raw_contract, dict):
        raise RuntimeError("The embedded GPU bundle has no authenticated training contract")
    contract = cast(dict[str, Any], raw_contract)
    config_path = contract.get("config_path")
    config_sha256 = contract.get("config_sha256")
    if config_path != DEFAULT_CONFIG_FILE or not isinstance(config_sha256, str):
        raise RuntimeError(
            "The embedded GPU bundle does not authenticate the default sion_translate.yaml"
        )
    expected_path = (bundle_root / DEFAULT_CONFIG_FILE).resolve()
    selected_path = (
        Path(requested_config).resolve() if requested_config is not None else expected_path
    )
    if selected_path != expected_path:
        raise RuntimeError(
            "This GPU bundle must be trained with its authenticated sion_translate.yaml; "
            f"refusing alternate config {selected_path}"
        )
    if not expected_path.is_file() or file_sha256(expected_path) != config_sha256:
        raise RuntimeError(
            "The extracted sion_translate.yaml differs from the GPU bundle training contract. "
            "Re-extract and verify the bundle before training."
        )
    if override_flags:
        raise RuntimeError(
            "This GPU bundle does not permit config-mutating command-line overrides: "
            f"{', '.join(override_flags)}. Update the tracked sion_translate.yaml, "
            "prepare matching artifacts, and build a new bundle instead."
        )


def bundle_config_override_flags(args: argparse.Namespace) -> tuple[str, ...]:
    """Return CLI flags that mutate the authenticated YAML training contract."""

    candidates = (
        ("--epochs", args.epochs is not None),
        ("--max-steps", args.max_steps is not None),
        ("--posttrain-epochs", args.posttrain_epochs is not None),
        ("--posttrain-steps", args.posttrain_steps is not None),
        ("--skip-posttraining", bool(args.skip_posttraining)),
        ("--resume-from", args.resume_from is not None),
    )
    return tuple(flag for flag, active in candidates if active)


def resolve_config(args: argparse.Namespace) -> tuple[AppConfig, dict[str, Any], str]:
    """Resolve and return the configuration, raw dictionary, and source label.

    The raw dictionary records which keys the user supplied explicitly.
    Automatic settings fill only keys that the user did not provide.
    """
    preflight_embedded_bundle_config(
        args.config,
        override_flags=bundle_config_override_flags(args),
    )
    if args.config:
        raw = load_raw_config(args.config)
        source = args.config
    elif Path(DEFAULT_CONFIG_FILE).exists():
        raw = load_raw_config(DEFAULT_CONFIG_FILE)
        source = DEFAULT_CONFIG_FILE
    else:
        raw = {}
        source = "built-in defaults (fully automatic)"

    # Command-line arguments override the file and count as explicit user settings.
    if args.epochs is not None:
        raw.setdefault("training", {})["num_train_epochs"] = args.epochs
    if args.max_steps is not None:
        raw.setdefault("training", {})["max_steps"] = args.max_steps
    if args.resume_from is not None:
        raw.setdefault("training", {})["resume_from"] = args.resume_from
    if args.posttrain_epochs is not None:
        raw.setdefault("posttraining", {})["num_train_epochs"] = args.posttrain_epochs
    if args.posttrain_steps is not None:
        post = raw.setdefault("posttraining", {})
        post["max_steps"] = args.posttrain_steps
        post["warmup_steps"] = min(int(post.get("warmup_steps", 200)), args.posttrain_steps)
    if args.skip_posttraining:
        raw.setdefault("posttraining", {})["enabled"] = False
    return config_from_raw(raw), raw, source


def _artifact_mutation_roots(
    config: AppConfig,
    foundation_plan: Any,
    *,
    prepare_foundation: bool,
) -> tuple[Path, ...]:
    """Return every sibling namespace that artifact preparation may mutate."""

    mutation_roots = {
        Path(config.data.raw_dir).resolve(),
        Path(config.data.tokenizer_model).resolve().parent,
        Path(config.data.dataset_dir).resolve().parent,
    }
    if config.foundation.enabled and prepare_foundation:
        mutation_roots.add(Path(config.foundation.dataset_dir).resolve().parent)
    return tuple(
        sorted(
            mutation_roots,
            key=lambda path: os.path.normcase(str(path)),
        )
    )


@contextmanager  # pyright: ignore[reportDeprecated]
def coordinated_artifact_run_locks(
    config: AppConfig,
    foundation_plan: Any,
    context: DistributedContext,
) -> Iterator[tuple[Path, ...]]:
    """Hold an exclusive read/write lease on every artifact root for the run."""

    roots = _artifact_mutation_roots(
        config,
        foundation_plan,
        # A foundation-enabled run may open this dataset well after initial
        # preparation, so its lease is needed even when SFT resume initially
        # defers foundation preparation.
        prepare_foundation=bool(config.foundation.enabled),
    )
    scope = ExitStack()
    acquisition_error: Exception | None = None
    if context.is_main:
        try:
            for root in roots:
                scope.enter_context(artifact_lock(root))
        except Exception as error:
            acquisition_error = error
    try:
        acquisition_failed = broadcast_bool(acquisition_error is not None, context)
        if acquisition_failed:
            if acquisition_error is not None:
                raise acquisition_error
            raise RuntimeError("rank 0 could not acquire the artifact run leases")
        yield roots
    finally:
        scope.close()


def _prepared_artifact_identity(
    config: AppConfig,
    foundation_plan: Any,
    *,
    prepare_foundation: bool,
) -> dict[str, Any]:
    """Hash the small files that bind every rank to one prepared generation."""

    def required_file(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"required prepared artifact is missing: {path}")
        return {
            "filename": path.name,
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }

    def optional_file(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"filename": path.name, "status": "missing"}
        return required_file(path)

    tokenizer_path = Path(config.data.tokenizer_model)
    dataset_dir = Path(config.data.dataset_dir)
    translation_inventory_sha256 = validate_dataset_artifact_inventory(dataset_dir)
    identity: dict[str, Any] = {
        "tokenizer": required_file(tokenizer_path),
        "tokenizer_metadata": optional_file(tokenizer_path.parent / "tokenizer_metadata.json"),
        "translation_manifest": required_file(dataset_dir / "manifest.json"),
        "translation_raw_fingerprint": required_file(dataset_dir / "raw_fingerprint.json"),
        "translation_artifact_inventory_sha256": translation_inventory_sha256,
    }
    if config.model.experimental.morphoscript_enabled:
        identity["token_features"] = required_file(Path(config.data.tokenizer_features))
    else:
        identity["token_features"] = {"status": "not-configured"}
    if foundation_plan.enabled and prepare_foundation:
        foundation_inventory_sha256 = validate_dataset_artifact_inventory(
            config.foundation.dataset_dir
        )
        identity["foundation_manifest"] = required_file(
            Path(config.foundation.dataset_dir) / "manifest.json"
        )
        identity["foundation_artifact_inventory_sha256"] = foundation_inventory_sha256
    else:
        identity["foundation_manifest"] = {"status": "not-prepared"}
        identity["foundation_artifact_inventory_sha256"] = {"status": "not-prepared"}
    return identity


def _verify_prepared_artifact_consensus(
    config: AppConfig,
    foundation_plan: Any,
    context: DistributedContext,
    *,
    prepare_foundation: bool,
) -> dict[str, Any]:
    """Require every rank to observe byte-identical prepared control artifacts."""

    identity: dict[str, Any] | None = None
    identity_error: Exception | None = None
    try:
        identity = _prepared_artifact_identity(
            config,
            foundation_plan,
            prepare_foundation=prepare_foundation,
        )
    except Exception as error:
        identity_error = error
    failure_scope = distributed_failure_scope(identity_error is not None, context)
    if failure_scope != "none":
        failure = RuntimeError(
            "prepared artifact visibility or readability differs across distributed ranks"
            if failure_scope == "partial"
            else "prepared artifacts are incomplete or unreadable"
        )
        if identity_error is not None:
            raise failure from identity_error
        raise failure
    assert identity is not None
    if not context.distributed:
        return identity
    local_digest = _mapping_sha256(identity)
    expected_digest = broadcast_text(local_digest if context.is_main else None, context)
    mismatch_scope = distributed_failure_scope(local_digest != expected_digest, context)
    if mismatch_scope != "none":
        raise RuntimeError("prepared tokenizer/dataset generation differs across distributed ranks")
    return identity


def _ensure_artifacts_on_main(
    config: AppConfig,
    context: DistributedContext,
    foundation_plan: Any | None = None,
    *,
    prepare_foundation: bool = True,
    require_offline_foundation: bool = False,
    locks_held: bool = False,
) -> None:
    """Create the tokenizer and prepared datasets when absent or stale.

    - Tokenizer: train only when it is missing. An incompatible artifact is
      neither moved nor overwritten automatically because another run may depend
      on its vocabulary. Instead, report a concrete error so the operator can
      inspect related checkpoints.
    - Dataset: record a filename-and-size fingerprint for data/. When files are
      added or changed, preserve the old dataset beside the active path and
      prepare a new one.

    This internal implementation runs only on rank 0. Distributed peers wait
    through ensure_artifacts' durable status channel rather than a process-group
    collective.
    """

    if foundation_plan is None:
        foundation_plan = plan_foundation_stage(config)
    reasoning_languages = (
        tuple(
            dict.fromkeys(
                source.language
                for source in foundation_plan.discovery.sources
                if is_reasoning_jsonl(source.path)
            )
        )
        if foundation_plan.enabled
        else ()
    )
    offline_foundation_required = bool(
        foundation_plan.enabled
        and not foundation_plan.discovery.sources
        and (prepare_foundation or require_offline_foundation)
    )
    if offline_foundation_required:
        reasoning_languages = _preflight_offline_foundation_dataset(
            config,
            foundation_plan,
        )
    # If two jobs sharing artifacts/ both decide that outputs are absent, tokenizer
    # and dataset generations can be mixed under one path. This is not an obvious
    # failure: fingerprinting would merely see the mixed combination as new.
    with ExitStack() as artifact_scope:
        if context.is_main and not locks_held:
            # Every run acquires the same target-root locks in the same order.
            # A tokenizer or foundation dataset may live outside the translation
            # dataset parent, and therefore cannot be protected by that one lock.
            for mutation_root in _artifact_mutation_roots(
                config,
                foundation_plan,
                prepare_foundation=prepare_foundation,
            ):
                artifact_scope.enter_context(artifact_lock(mutation_root))
        if context.is_main:
            data_dir = Path(config.data.raw_dir)
            tokenizer_path = Path(config.data.tokenizer_model)
            dataset_dir = Path(config.data.dataset_dir)
            files = scan_configured_raw_data(config, data_dir, tokenizer_path)
            dataset_ready = (dataset_dir / "manifest.json").exists()
            existing_checkpoint = find_existing_checkpoint(config)

            if not files and not dataset_ready:
                raise FileNotFoundError(
                    f"Neither source data ({data_dir}/*.jsonl) nor a prepared "
                    f"dataset ({dataset_dir}) exists."
                )
            if not tokenizer_path.is_file() and dataset_ready and not files:
                raise FileNotFoundError(
                    "A prepared dataset exists, but its tokenizer is missing: "
                    f"{tokenizer_path}. Start a new run with source data and new "
                    "output paths."
                )
            if not files and dataset_ready:
                integrity_problem = dataset_artifact_problem(dataset_dir)
                if integrity_problem is not None:
                    raise RuntimeError(
                        "The prepared translation dataset payload is corrupt, and "
                        "no source data is available to rebuild it: "
                        f"{integrity_problem}"
                    )
                if not reasoning_languages:
                    metadata = load_tokenizer_metadata(tokenizer_path)
                    raw_reasoning_languages = (
                        metadata.get("reasoning_languages") if isinstance(metadata, dict) else None
                    )
                    if isinstance(raw_reasoning_languages, list) and all(
                        isinstance(language, str) for language in raw_reasoning_languages
                    ):
                        reasoning_languages = tuple(cast(list[str], raw_reasoning_languages))
                policy_problem = tokenizer_policy_problem(
                    tokenizer_path,
                    config.data.configured_language_pairs(),
                    config.foundation_languages(),
                    reasoning_languages,
                    translation_directions=config.data.configured_translation_directions(),
                    require_recorded_directions=bool(config.data.translation_directions),
                )
                if policy_problem is not None:
                    raise RuntimeError(
                        "The prepared tokenizer is incompatible, and no source data is "
                        f"available to rebuild it: {policy_problem}"
                    )

            if files:
                cpu_plan = build_cpu_plan(input_files=len(files))
                announce(
                    f"Source data discovered: {len(files)} files, "
                    f"{sum(files.values()) / 2**30:.2f} GiB total ({data_dir}/)",
                    context,
                )
                announce(
                    f"Automatic CPU allocation: {cpu_plan.available} available → "
                    f"{cpu_plan.preprocess_workers} input-cleaning workers + "
                    f"{cpu_plan.sentencepiece_threads} SentencePiece threads; "
                    f"{cpu_plan.dataset_workers} dataset-preparation workers",
                    context,
                )
                # ── Tokenizer ─────────────────────────────────────────────
                if not tokenizer_path.exists():
                    if existing_checkpoint is not None:
                        raise RuntimeError(
                            "An existing checkpoint has no corresponding tokenizer: "
                            f"checkpoint={existing_checkpoint}. The previous vocabulary "
                            "cannot be guessed and overwritten with a new tokenizer. "
                            "Start a new run with new tokenizer_model, dataset_dir, "
                            "and training.output_dir paths."
                        )
                    from sion_translate.tokenizer import train_tokenizer

                    pair_estimate = estimate_pair_count(files, data_dir)
                    vocab_size = pick_vocab_size(pair_estimate)
                    announce(
                        "The tokenizer is missing, so a new one will be trained "
                        f"(approximately {pair_estimate:,} rows → vocab "
                        f"{vocab_size:,}). This may take time.",
                        context,
                    )
                    train_tokenizer(
                        [str(data_dir / "*.jsonl")],
                        tokenizer_path.parent,
                        vocab_size=vocab_size,
                        language_pairs=config.data.configured_language_pairs(),
                        translation_directions=config.data.configured_translation_directions(),
                        # Include monolingual corpora so foundation training does not
                        # encounter only out-of-vocabulary terms. Per-language caps
                        # prevent the largest corpus from dominating the vocabulary.
                        monolingual=foundation_plan.discovery,
                        monolingual_sample_ratio=config.foundation.tokenizer_sample_ratio,
                        foundation_languages=foundation_plan.languages,
                        reasoning_languages=reasoning_languages,
                        approximate_split=config.data.approximate_split,
                        source_only_languages=config.data.configured_source_only_languages(),
                        train_only_prefixes=config.data.configured_synthetic_prefixes(),
                        num_workers=cpu_plan.preprocess_workers,
                        num_threads=cpu_plan.sentencepiece_threads,
                    )
                    announce("Tokenizer training complete.", context)
                    # The tokenizer file's SHA-256 is part of the dataset fingerprint.
                    files = scan_configured_raw_data(config, data_dir, tokenizer_path)

                # ── Dataset with fingerprint-based change detection ───────
                policy_problem = tokenizer_policy_problem(
                    tokenizer_path,
                    config.data.configured_language_pairs(),
                    config.foundation_languages(),
                    reasoning_languages,
                    translation_directions=config.data.configured_translation_directions(),
                    require_recorded_directions=bool(config.data.translation_directions),
                )
                if policy_problem is not None:
                    existing_tokenizer = SionTokenizer(tokenizer_path)
                    if (
                        existing_checkpoint is None
                        and existing_tokenizer.splits_digits
                        and load_tokenizer_metadata(tokenizer_path) is None
                        and not config.data.translation_directions
                    ):
                        write_tokenizer_metadata(
                            tokenizer_path,
                            split_digits=True,
                            language_pairs=config.data.configured_language_pairs(),
                            translation_directions=config.data.configured_translation_directions(),
                        )
                        files = scan_configured_raw_data(config, data_dir, tokenizer_path)
                        policy_problem = tokenizer_policy_problem(
                            tokenizer_path,
                            config.data.configured_language_pairs(),
                            config.foundation_languages(),
                            reasoning_languages,
                            translation_directions=config.data.configured_translation_directions(),
                            require_recorded_directions=bool(config.data.translation_directions),
                        )
                    if policy_problem is not None:
                        checkpoint_detail = (
                            f" Automatic retraining will not break vocabulary "
                            f"compatibility with checkpoint={existing_checkpoint}."
                            if existing_checkpoint is not None
                            else ""
                        )
                        raise RuntimeError(
                            f"{policy_problem}.{checkpoint_detail} tokenizer_model, dataset_dir, "
                            "Review training.output_dir as well. For a new training "
                            "run, first confirm that no related run uses this vocabulary. "
                            "Then move the existing tokenizer/dataset to a separate "
                            "backup and prepare them again with split_digits=True."
                        )
                stored = stored_fingerprint(dataset_dir) if dataset_ready else None
                integrity_problem = (
                    dataset_artifact_problem(dataset_dir)
                    if dataset_ready and stored == files
                    else None
                )
                if not dataset_ready or stored != files or integrity_problem is not None:
                    from sion_translate.data.prepare import prepare_dataset

                    if dataset_ready:
                        backup = backup_stale_dataset(dataset_dir)
                        reason = (
                            f"indexed payload integrity failure ({integrity_problem})"
                            if integrity_problem is not None
                            else (
                                "no compatible fingerprint"
                                if stored is None
                                else "source/tokenizer/preprocessing changed"
                            )
                        )
                        announce(
                            f"Detected {reason}; preserving the existing dataset "
                            f"under {backup.name}/.",
                            context,
                        )
                    announce(
                        "Preparing the dataset (quality filtering, deduplication, "
                        "and tokenization). This may take time.",
                        context,
                    )
                    prepare_dataset(
                        [str(data_dir / "*.jsonl")],
                        tokenizer_path,
                        dataset_dir,
                        language_pairs=config.data.configured_language_pairs(),
                        translation_directions=config.data.configured_translation_directions(),
                        source_only_languages=config.data.configured_source_only_languages(),
                        approximate_split=config.data.approximate_split,
                        train_only_prefixes=config.data.configured_synthetic_prefixes(),
                        managed_augmentation_prefix=config.data.synthetic_prefix,
                        synthetic_sampling_weight=config.data.synthetic_sampling_weight,
                        num_workers=cpu_plan.dataset_workers,
                        expected_fingerprint=files,
                    )
                    announce("Dataset preparation complete.", context)
                else:
                    announce("Dataset is current; source data has not changed.", context)

                # ── Foundation monolingual dataset ────────────────────────
                for line in foundation_plan.report:
                    announce(f"  {line}", context)
                if not foundation_plan.enabled:
                    announce(f"Foundation stage: {foundation_plan.reason}", context)
                elif not prepare_foundation:
                    announce(
                        "Deferring foundation dataset preparation while SFT resume "
                        "candidates are validated first.",
                        context,
                    )
                else:
                    for warning in foundation_plan.warnings:
                        announce(f"[warning] foundation: {warning}", context)
                    foundation_dataset = Path(config.foundation.dataset_dir)
                    from sion_translate.data.prepare_foundation import (
                        foundation_dataset_problem,
                        prepare_foundation_dataset,
                        render_prepare_report,
                    )

                    foundation_problem = foundation_dataset_problem(
                        foundation_dataset,
                        foundation_plan.discovery,
                        tokenizer_path,
                        minimum_characters=config.foundation.minimum_characters,
                        maximum_characters=config.foundation.maximum_characters,
                        max_tokens=config.data.max_source_length - 2,
                        max_target_tokens=config.data.max_target_length - 1,
                        deduplicate=config.foundation.deduplicate,
                        shard_size=config.foundation.shard_size,
                        validation_fraction=config.foundation.validation_fraction,
                        reasoning_sample_share=config.foundation.reasoning_sample_share,
                        language_sampling_alpha=config.foundation.language_sampling_alpha,
                        minimum_language_share=config.foundation.minimum_language_share,
                        release_name=config.foundation.release_name,
                    )
                    if foundation_problem is None:
                        announce("Foundation dataset is current.", context)
                    else:
                        if foundation_dataset.exists() or foundation_dataset.is_symlink():
                            backup = backup_stale_dataset(foundation_dataset)
                            announce(
                                f"{foundation_problem}; preserving the existing "
                                f"foundation dataset under {backup.name}/.",
                                context,
                            )

                        announce(
                            "Preparing the foundation dataset (denoising and "
                            "reasoning tokenization). This may take time.",
                            context,
                        )
                        foundation_stats = prepare_foundation_dataset(
                            foundation_plan.discovery,
                            tokenizer_path,
                            foundation_dataset,
                            minimum_characters=config.foundation.minimum_characters,
                            maximum_characters=config.foundation.maximum_characters,
                            max_tokens=config.data.max_source_length - 2,
                            max_target_tokens=config.data.max_target_length - 1,
                            deduplicate=config.foundation.deduplicate,
                            shard_size=config.foundation.shard_size,
                            validation_fraction=config.foundation.validation_fraction,
                            language_sampling_alpha=config.foundation.language_sampling_alpha,
                            minimum_language_share=config.foundation.minimum_language_share,
                            reasoning_sample_share=config.foundation.reasoning_sample_share,
                            release_name=config.foundation.release_name,
                        )
                        for line in render_prepare_report(foundation_stats):
                            announce(f"  {line}", context)


def ensure_artifacts(
    config: AppConfig,
    context: DistributedContext,
    foundation_plan: Any | None = None,
    *,
    prepare_foundation: bool = True,
    require_offline_foundation: bool = False,
    locks_held: bool = False,
) -> None:
    """Prepare artifacts on rank 0 without timing out the training process group."""

    if foundation_plan is None:
        foundation_plan = plan_foundation_stage(config)
    if not locks_held:
        with coordinated_artifact_run_locks(config, foundation_plan, context):
            ensure_artifacts(
                config,
                context,
                foundation_plan,
                prepare_foundation=prepare_foundation,
                require_offline_foundation=require_offline_foundation,
                locks_held=True,
            )
        return
    _run_long_rank_zero_action(
        context,
        Path(config.training.output_dir) / ".artifact-preparation-status.json",
        operation="artifact preparation",
        action=lambda: _ensure_artifacts_on_main(
            config,
            context,
            foundation_plan,
            prepare_foundation=prepare_foundation,
            require_offline_foundation=require_offline_foundation,
            locks_held=locks_held,
        ),
    )
    _verify_prepared_artifact_consensus(
        config,
        foundation_plan,
        context,
        prepare_foundation=prepare_foundation or require_offline_foundation,
    )


FOUNDATION_COMPLETION_FILENAME = "stage_complete.json"
FOUNDATION_COMPLETION_SCHEMA = "sion-foundation-completion-v2"
_RankZeroActionT = TypeVar("_RankZeroActionT")


def _read_foundation_completion(completion: Path) -> dict[str, Any] | None:
    try:
        raw_marker: object = json.loads(completion.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw_marker, dict):
        return None
    return cast(dict[str, Any], raw_marker)


def _atomic_write_foundation_completion(
    completion: Path,
    marker: dict[str, Any],
) -> None:
    """Durably publish a completion marker without exposing partial JSON."""

    _atomic_write_json(completion, marker)


def _foundation_untrusted_derivatives(
    run_root: Path,
    *,
    trusted_best_artifact_sha256: str | None,
) -> tuple[Path, ...]:
    """Return completion/export and best generations not bound by latest."""

    untrusted = [
        run_root / FOUNDATION_COMPLETION_FILENAME,
        run_root / "exports" / "best",
    ]
    for best_generation in (
        run_root / "checkpoints" / "best",
        run_root / "checkpoints" / ".best.previous",
    ):
        if not (best_generation.exists() or best_generation.is_symlink()):
            continue
        try:
            matches_latest = (
                trusted_best_artifact_sha256 is not None
                and _checkpoint_artifact_sha256(best_generation) == trusted_best_artifact_sha256
            )
        except (OSError, RuntimeError, ValueError):
            matches_latest = False
        if not matches_latest:
            untrusted.append(best_generation)
    return tuple(untrusted)


def _quarantine_untrusted_foundation_derivatives(
    run_root: Path,
    *,
    trusted_best_artifact_sha256: str | None,
) -> Path | None:
    """Move only derivatives not content-bound by the resumable latest state."""

    sources = tuple(
        path
        for path in _foundation_untrusted_derivatives(
            run_root,
            trusted_best_artifact_sha256=trusted_best_artifact_sha256,
        )
        if path.exists() or path.is_symlink()
    )
    if not sources:
        return None
    quarantine = run_root.with_name(
        f"{run_root.name}.untrusted-derived-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
    )
    quarantine.mkdir(parents=False, exist_ok=False)
    moved: list[tuple[Path, Path]] = []
    try:
        for source in sources:
            destination = quarantine / source.relative_to(run_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved.append((source, destination))
    except BaseException as quarantine_error:
        rollback_errors: list[BaseException] = []
        for source, destination in reversed(moved):
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, source)
            except BaseException as error:
                rollback_errors.append(error)
        if rollback_errors:
            failure = RuntimeError(
                "failed to restore foundation derivatives after quarantine failed; "
                f"recoverable artifacts remain at {quarantine}"
            )
            for error in rollback_errors:
                failure.add_note(f"rollback error: {type(error).__name__}: {error}")
            raise failure from quarantine_error
        raise
    return quarantine


def _run_rank_zero_action(
    context: DistributedContext,
    action: Callable[[], _RankZeroActionT],
    *,
    description: str,
) -> _RankZeroActionT | None:
    """Run a rank-zero action and fail every rank at the same collective."""

    result: _RankZeroActionT | None = None
    failure: BaseException | None = None
    if context.is_main:
        try:
            result = action()
        except BaseException as error:
            failure = error
    failed = broadcast_bool(failure is not None, context)
    if failed:
        if failure is not None:
            raise failure
        raise RuntimeError(f"rank 0 failed while {description}")
    return result


def _wait_for_rank_zero_action(
    status_path: Path,
    *,
    operation: str,
    invocation: str,
    stale_timeout_seconds: float,
) -> Any:
    """Wait for and return the exact terminal status for one invocation."""

    last_progress = time.monotonic()
    heartbeat_sequence: int | None = None
    while True:
        for status in _read_control_status(status_path):
            matches_operation = (
                status.get("schema") == RANK_ZERO_ACTION_STATUS_SCHEMA
                and status.get("operation") == operation
                and status.get("invocation") == invocation
            )
            if matches_operation and status.get("state") == "complete":
                return status
            if matches_operation and status.get("state") == "failed":
                return status
            if matches_operation and status.get("state") == "running":
                raw_sequence = status.get("heartbeat_sequence")
                if isinstance(raw_sequence, int) and not isinstance(raw_sequence, bool):
                    if heartbeat_sequence != raw_sequence:
                        heartbeat_sequence = raw_sequence
                        last_progress = time.monotonic()
        if time.monotonic() - last_progress >= stale_timeout_seconds:
            raise TimeoutError(
                f"rank 0 {operation} stopped publishing heartbeats for "
                f"{stale_timeout_seconds:g} seconds: {status_path}"
            )
        time.sleep(0.25)


def _rank_zero_action_ack_path(
    status_path: Path,
    *,
    invocation: str,
    rank: int,
) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", invocation):
        raise ValueError("rank-zero action invocation must be a 128-bit lowercase hex nonce")
    if rank < 0:
        raise ValueError("rank-zero action acknowledgement rank must be non-negative")
    return status_path.with_name(f".{status_path.name}.{invocation}.ack-{rank:05d}.json")


def _wait_for_rank_zero_action_acknowledgements(
    status_path: Path,
    *,
    operation: str,
    invocation: str,
    expected_terminal_state: str | None,
    world_size: int,
    timeout_seconds: float,
) -> None:
    """Wait until every peer confirms it no longer depends on terminal status."""

    pending = set(range(1, world_size))
    deadline = time.monotonic() + timeout_seconds
    peer_errors: list[str] = []
    while pending:
        for rank in tuple(pending):
            ack_path = _rank_zero_action_ack_path(
                status_path,
                invocation=invocation,
                rank=rank,
            )
            try:
                raw_ack: object = json.loads(ack_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(raw_ack, dict):
                continue
            ack = cast(dict[str, Any], raw_ack)
            if (
                ack.get("schema") != RANK_ZERO_ACTION_STATUS_SCHEMA
                or ack.get("operation") != operation
                or ack.get("invocation") != invocation
                or ack.get("rank") != rank
                or ack.get("state") not in {"observed", "observer_error"}
            ):
                continue
            if ack["state"] == "observer_error":
                peer_errors.append(
                    f"rank {rank}: {ack.get('message', 'unknown terminal observation error')}"
                )
            elif ack.get("terminal_state") != expected_terminal_state:
                peer_errors.append(
                    f"rank {rank}: observed terminal state {ack.get('terminal_state')!r}, "
                    f"expected {expected_terminal_state!r}"
                )
            pending.remove(rank)
        if pending and time.monotonic() >= deadline:
            raise TimeoutError(
                f"rank-zero {operation} terminal acknowledgement timed out for ranks "
                f"{sorted(pending)}"
            )
        if pending:
            time.sleep(0.05)
    if peer_errors:
        raise RuntimeError(
            f"rank-zero {operation} terminal status was not observed cleanly: "
            + "; ".join(peer_errors)
        )


def _run_long_rank_zero_action(
    context: DistributedContext,
    status_path: Path,
    *,
    operation: str,
    action: Callable[[], _RankZeroActionT],
    stale_timeout_seconds: float = RANK_ZERO_ACTION_STALE_TIMEOUT_SECONDS,
    heartbeat_interval_seconds: float = RANK_ZERO_ACTION_HEARTBEAT_SECONDS,
) -> _RankZeroActionT:
    """Run long rank-zero I/O without parking peers in a timed collective."""

    if stale_timeout_seconds <= 0.0:
        raise ValueError("rank-zero action stale timeout must be positive")
    if heartbeat_interval_seconds <= 0.0:
        raise ValueError("rank-zero action heartbeat interval must be positive")
    if heartbeat_interval_seconds >= stale_timeout_seconds:
        raise ValueError("rank-zero action heartbeat interval must be shorter than stale timeout")
    if not context.distributed:
        return action()
    invocation = broadcast_text(
        secrets.token_hex(16) if context.is_main else None,
        context,
    )
    running = {
        "schema": RANK_ZERO_ACTION_STATUS_SCHEMA,
        "operation": operation,
        "state": "running",
        "invocation": invocation,
        "heartbeat_sequence": 0,
    }
    initialized_handles = _run_rank_zero_action(
        context,
        lambda: _initialize_control_status(status_path, running),
        description=f"publishing {operation} start state",
    )
    visibility_scope = distributed_failure_scope(
        not _control_status_is_visible(
            status_path,
            schema=RANK_ZERO_ACTION_STATUS_SCHEMA,
            invocation=invocation,
        ),
        context,
    )
    if visibility_scope != "none":
        if context.is_main:
            _close_control_status(initialized_handles)
        raise RuntimeError(f"rank-zero {operation} status is not visible to every rank")
    if not context.is_main:
        ack_path = _rank_zero_action_ack_path(
            status_path,
            invocation=invocation,
            rank=context.rank,
        )
        try:
            terminal = _wait_for_rank_zero_action(
                status_path,
                operation=operation,
                invocation=invocation,
                stale_timeout_seconds=stale_timeout_seconds,
            )
        except BaseException as error:
            _atomic_write_json(
                ack_path,
                {
                    "schema": RANK_ZERO_ACTION_STATUS_SCHEMA,
                    "operation": operation,
                    "invocation": invocation,
                    "rank": context.rank,
                    "state": "observer_error",
                    "message": _bounded_status_text(error),
                },
            )
            raise
        _atomic_write_json(
            ack_path,
            {
                "schema": RANK_ZERO_ACTION_STATUS_SCHEMA,
                "operation": operation,
                "invocation": invocation,
                "rank": context.rank,
                "state": "observed",
                "terminal_state": terminal.get("state"),
            },
        )
        if terminal.get("state") == "failed":
            error_type = terminal.get("error_type", "RuntimeError")
            message = terminal.get("message", "unknown rank-0 action failure")
            raise RuntimeError(f"rank 0 {operation} failed: {error_type}: {message}")
        return cast(_RankZeroActionT, terminal.get("result"))

    assert initialized_handles is not None
    status_lock = threading.Lock()
    heartbeat_stop = threading.Event()
    heartbeat_errors: list[BaseException] = []

    def publish_status(status: dict[str, Any]) -> None:
        with status_lock:
            _publish_control_status(initialized_handles, status)

    def publish_heartbeats() -> None:
        sequence = 0
        while not heartbeat_stop.wait(heartbeat_interval_seconds):
            sequence += 1
            try:
                publish_status(
                    {
                        "schema": RANK_ZERO_ACTION_STATUS_SCHEMA,
                        "operation": operation,
                        "state": "running",
                        "invocation": invocation,
                        "heartbeat_sequence": sequence,
                    }
                )
            except BaseException as error:
                heartbeat_errors.append(error)
                return

    heartbeat_thread = threading.Thread(
        target=publish_heartbeats,
        name=f"sion-{operation}-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()

    def stop_heartbeat() -> None:
        heartbeat_stop.set()
        heartbeat_thread.join()

    result: Any = None
    terminal_error: BaseException | None = None
    published_terminal_state: str | None = None
    try:
        try:
            result = action()
            # The terminal state must be the final write. Otherwise a heartbeat
            # that wakes immediately after ``complete`` can overwrite it with
            # ``running`` and strand every peer until the stale timeout.
            stop_heartbeat()
            if heartbeat_errors:
                raise RuntimeError(
                    f"failed to publish rank-zero {operation} heartbeat"
                ) from heartbeat_errors[0]
        except BaseException as error:
            stop_heartbeat()
            try:
                publish_status(
                    {
                        "schema": RANK_ZERO_ACTION_STATUS_SCHEMA,
                        "operation": operation,
                        "state": "failed",
                        "invocation": invocation,
                        "error_type": type(error).__name__,
                        "message": _bounded_status_text(error),
                    },
                )
            except BaseException as publication_error:
                terminal_error = publication_error
            else:
                terminal_error = error
                published_terminal_state = "failed"
        else:
            try:
                publish_status(
                    {
                        "schema": RANK_ZERO_ACTION_STATUS_SCHEMA,
                        "operation": operation,
                        "state": "complete",
                        "invocation": invocation,
                        "result": result,
                    },
                )
            except BaseException as error:
                terminal_error = error
            else:
                published_terminal_state = "complete"
        try:
            _wait_for_rank_zero_action_acknowledgements(
                status_path,
                operation=operation,
                invocation=invocation,
                expected_terminal_state=published_terminal_state,
                world_size=context.world_size,
                timeout_seconds=stale_timeout_seconds,
            )
        except BaseException as acknowledgement_error:
            if terminal_error is None:
                terminal_error = acknowledgement_error
            else:
                terminal_error.add_note(
                    "terminal acknowledgement error: "
                    f"{type(acknowledgement_error).__name__}: {acknowledgement_error}"
                )
    finally:
        stop_heartbeat()
        _close_control_status(initialized_handles)
        for rank in range(1, context.world_size):
            _rank_zero_action_ack_path(
                status_path,
                invocation=invocation,
                rank=rank,
            ).unlink(missing_ok=True)
    if terminal_error is not None:
        raise terminal_error
    return cast(_RankZeroActionT, result)


def _mapping_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _foundation_checkpoint_identity(
    foundation_config: AppConfig,
    context: DistributedContext,
) -> dict[str, Any]:
    """Rebuild the exact identity used by the foundation trainer."""

    effective_bucket_size = max(
        foundation_config.data.bucket_size,
        foundation_config.training.batch_size_per_gpu * context.world_size,
    )
    sampler_contract = SimpleNamespace(
        seed=foundation_config.training.seed,
        batch_size=foundation_config.training.batch_size_per_gpu,
        drop_last=True,
        bucket_size=effective_bucket_size,
    )
    return build_training_checkpoint_identity(
        foundation_config,
        batch_sampler=sampler_contract,
        context=context,
        stage_name="foundation/denoising",
        include_posttraining=False,
    )


def _checkpoint_artifact_sha256(checkpoint: Path) -> str:
    """Digest the local payload, DCP inventory marker, or legacy metadata binding."""

    local_payload = checkpoint / "checkpoint.pt"
    dcp_marker = checkpoint / DCP_COMPLETION_FILENAME
    dcp_metadata = checkpoint / ".metadata"
    has_distributed_payload = dcp_marker.is_file() or dcp_metadata.is_file()
    if local_payload.is_file() and has_distributed_payload:
        raise ValueError(f"checkpoint mixes local and distributed payload formats: {checkpoint}")
    if local_payload.is_file():
        return file_sha256(local_payload)
    if dcp_marker.is_file():
        return file_sha256(dcp_marker)
    if dcp_metadata.is_file():
        # This digest is diagnostic only. Pre-v2 DCP has no historical shard
        # inventory and coordinated resume deliberately refuses to trust it.
        return file_sha256(dcp_metadata)
    raise FileNotFoundError(f"checkpoint has no authenticated payload: {checkpoint}")


def _foundation_checkpoint_artifact_sha256(checkpoint: Path) -> str:
    """Backward-compatible foundation name for the generic artifact binding."""

    return _checkpoint_artifact_sha256(checkpoint)


def _resolve_bound_foundation_checkpoint(
    checkpoint: Path,
    marker: dict[str, Any],
    context: DistributedContext,
) -> Path:
    """Find the current/previous generation named by the completion marker.

    Distributed shard bytes are authenticated later by a generation lease (or,
    for lineage-only rank-zero inspection, by checkpoint preflight). This helper
    intentionally reads only the small v2 marker while choosing a candidate.
    """

    expected_digest = marker.get("checkpoint_artifact_sha256")
    if not isinstance(expected_digest, str):
        raise ValueError("foundation completion marker has no checkpoint artifact digest")
    previous = checkpoint.with_name(f".{checkpoint.name}.previous")
    failures: list[str] = []
    candidates = (
        checkpoint_generation_candidates(checkpoint, context)
        if context.distributed
        else (checkpoint, previous)
    )
    for candidate in candidates:
        try:
            if not context.distributed:
                resolved = resolve_checkpoint_source(candidate, context)
                if resolved != candidate:
                    failures.append(f"{candidate.name}: resolved to {resolved.name}")
                    continue
            actual_digest = _foundation_checkpoint_artifact_sha256(candidate)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            failures.append(f"{candidate.name}: {error}")
            continue
        if actual_digest == expected_digest:
            return candidate
        failures.append(f"{candidate.name}: digest mismatch")
    raise ValueError(
        "no authenticated foundation checkpoint generation matches the completion marker "
        f"({'; '.join(failures)})"
    )


def _has_unverifiable_legacy_dcp_generation(checkpoint: Path) -> bool:
    """Whether current/previous contains DCP bytes without historical hashes."""

    previous = checkpoint.with_name(f".{checkpoint.name}.previous")
    for generation in (checkpoint, previous):
        if not (generation / ".metadata").is_file():
            continue
        marker_path = generation / DCP_COMPLETION_FILENAME
        try:
            raw_marker: object = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return True
        if not isinstance(raw_marker, dict) or raw_marker.get("schema") != DCP_COMPLETION_SCHEMA:
            return True
    return False


def _inspect_foundation_resume(
    foundation_config: AppConfig,
    context: DistributedContext,
) -> dict[str, Any]:
    candidate = find_auto_resume(foundation_config)
    if candidate is None:
        return {"state": "absent"}
    latest = Path(candidate)
    previous = latest.with_name(f".{latest.name}.previous")
    try:
        if context.distributed:
            candidates = checkpoint_generation_candidates(latest, context)
            if not candidates:
                if _has_unverifiable_legacy_dcp_generation(latest):
                    return {
                        "state": "untrusted_legacy",
                        "reason": (
                            "foundation resume has only markerless/legacy DCP shards with "
                            "no historical content digests"
                        ),
                    }
                raise FileNotFoundError(
                    "foundation resume has no structurally complete v2 generation"
                )
            source = candidates[0]
        else:
            local_errors: list[str] = []
            source = latest
            for local_candidate in (latest, previous):
                try:
                    source = resolve_checkpoint_source(local_candidate, context)
                except Exception as error:
                    local_errors.append(f"{local_candidate.name}: {error}")
                    continue
                break
            else:
                raise FileNotFoundError(
                    "foundation resume has no local current/previous generation: "
                    + "; ".join(local_errors)
                )
        digest = _foundation_checkpoint_artifact_sha256(source)
    except Exception as error:
        return {"state": "invalid", "reason": _bounded_status_text(error)}
    return {
        "state": "available",
        "generation": "previous" if source == previous else "current",
        "checkpoint_artifact_sha256": digest,
    }


def _foundation_export_is_complete(
    export_dir: Path,
    *,
    required_formats: Sequence[str],
    release_name: str,
) -> bool:
    """Verify the complete base-model export generation before reusing it."""

    manifest_path = export_dir / "export_manifest.json"
    try:
        raw_manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw_manifest, dict):
            return False
        manifest = cast(dict[str, Any], raw_manifest)
        formats = manifest.get("formats")
        if not isinstance(formats, dict):
            return False
        for format_name in required_formats:
            entry = formats.get(format_name)
            if not isinstance(entry, dict) or entry.get("status") != "ok":
                return False
        report = validate_export_directory(
            export_dir,
            expected_release_name=release_name,
            expected_release_version=MODEL_RELEASE_VERSION,
            expected_translation_capable=False,
        )
        return bool(report.get("valid"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False


def _foundation_completion_marker(
    *,
    config: AppConfig,
    foundation_languages: Sequence[str],
    foundation_config: AppConfig,
    checkpoint_identity: dict[str, Any],
    checkpoint_source: Path,
    selected_step: int,
    best_validation_loss: float | None,
    export_dir: Path,
) -> dict[str, Any]:
    manifest_path = export_dir / "export_manifest.json"
    marker: dict[str, Any] = {
        "schema": FOUNDATION_COMPLETION_SCHEMA,
        "stage": "foundation",
        "release_name": config.foundation.release_name,
        "release_version": MODEL_RELEASE_VERSION,
        "languages": list(foundation_languages),
        "selected_step": selected_step,
        "foundation_manifest_sha256": file_sha256(
            Path(foundation_config.data.dataset_dir) / "manifest.json"
        ),
        "tokenizer_sha256": file_sha256(config.data.tokenizer_model),
        "checkpoint_identity_sha256": _mapping_sha256(checkpoint_identity),
        "checkpoint_artifact_sha256": _foundation_checkpoint_artifact_sha256(checkpoint_source),
        "export_manifest_sha256": file_sha256(manifest_path),
    }
    if best_validation_loss is not None and math.isfinite(best_validation_loss):
        marker["best_validation_loss"] = best_validation_loss
    return marker


def _update_foundation_export_binding(
    completion: Path,
    marker: dict[str, Any],
    *,
    checkpoint_source: Path,
    export_dir: Path,
) -> None:
    """Update only the export digest while preserving the trusted checkpoint binding."""

    current = _read_foundation_completion(completion)
    if current != marker:
        raise RuntimeError("foundation completion marker changed during export repair")
    if marker.get("checkpoint_artifact_sha256") != _foundation_checkpoint_artifact_sha256(
        checkpoint_source
    ):
        raise RuntimeError("foundation checkpoint changed during export repair")
    updated = dict(marker)
    updated["export_manifest_sha256"] = file_sha256(export_dir / "export_manifest.json")
    _atomic_write_foundation_completion(completion, updated)


def _foundation_completion_matches_inputs(
    completion: Path,
    *,
    dataset_dir: str | Path,
    tokenizer_path: str | Path,
) -> bool:
    """Whether a completed run was trained on the currently prepared inputs."""

    manifest_path = Path(dataset_dir) / "manifest.json"
    try:
        marker = _read_foundation_completion(completion)
        if marker is None:
            return False
        return marker.get("foundation_manifest_sha256") == file_sha256(
            manifest_path
        ) and marker.get("tokenizer_sha256") == file_sha256(tokenizer_path)
    except OSError:
        return False


def _foundation_marker_contract_matches(
    marker: dict[str, Any],
    *,
    config: AppConfig,
    foundation_languages: Sequence[str],
    checkpoint_identity: dict[str, Any],
) -> bool:
    return bool(
        marker.get("schema") == FOUNDATION_COMPLETION_SCHEMA
        and marker.get("stage") == "foundation"
        and marker.get("release_name") == config.foundation.release_name
        and marker.get("release_version") == MODEL_RELEASE_VERSION
        and marker.get("languages") == list(foundation_languages)
        and marker.get("checkpoint_identity_sha256") == _mapping_sha256(checkpoint_identity)
        and all(
            isinstance(marker.get(field), str) and len(cast(str, marker.get(field))) == 64
            for field in (
                "foundation_manifest_sha256",
                "tokenizer_sha256",
                "checkpoint_identity_sha256",
                "checkpoint_artifact_sha256",
                "export_manifest_sha256",
            )
        )
        and not isinstance(marker.get("selected_step"), bool)
        and isinstance(marker.get("selected_step"), int)
    )


def _foundation_dataset_problem_for_lineage(
    config: AppConfig,
    foundation_plan: Any,
) -> str | None:
    return foundation_dataset_problem(
        config.foundation.dataset_dir,
        foundation_plan.discovery,
        config.data.tokenizer_model,
        minimum_characters=config.foundation.minimum_characters,
        maximum_characters=config.foundation.maximum_characters,
        max_tokens=config.data.max_source_length - 2,
        max_target_tokens=config.data.max_target_length - 1,
        deduplicate=config.foundation.deduplicate,
        shard_size=config.foundation.shard_size,
        validation_fraction=config.foundation.validation_fraction,
        reasoning_sample_share=config.foundation.reasoning_sample_share,
        language_sampling_alpha=config.foundation.language_sampling_alpha,
        minimum_language_share=config.foundation.minimum_language_share,
        release_name=config.foundation.release_name,
    )


def _inspect_foundation_lineage(
    config: AppConfig,
    foundation_plan: Any,
    context: DistributedContext,
) -> dict[str, Any]:
    """Return one verified base generation suitable for translation ancestry."""

    if not foundation_plan.enabled:
        raise ValueError("translation-only pipeline has no foundation lineage")
    if foundation_plan.discovery.sources:
        foundation_problem = _foundation_dataset_problem_for_lineage(config, foundation_plan)
        if foundation_problem is not None:
            raise RuntimeError(f"foundation prepared inputs are stale: {foundation_problem}")
    else:
        # A completed/in-progress base remains reusable while the raw corpus is
        # temporarily offline. Authenticate the published prepared generation;
        # do not reinterpret missing raw files as proof that its lineage changed.
        validate_dataset_artifact_inventory(config.foundation.dataset_dir)
    foundation_config = build_foundation_config(config)
    _, _, foundation_languages = _foundation_training_contract(
        foundation_config,
        context,
        planned_languages=foundation_plan.languages,
        verify_integrity=False,
    )
    checkpoint_identity = _foundation_checkpoint_identity(foundation_config, context)
    run_root = foundation_run_directory(config)
    completion = run_root / FOUNDATION_COMPLETION_FILENAME
    marker = _read_foundation_completion(completion)
    if marker is None or not _foundation_marker_contract_matches(
        marker,
        config=config,
        foundation_languages=foundation_languages,
        checkpoint_identity=checkpoint_identity,
    ):
        raise RuntimeError("foundation completion marker does not match the current base contract")
    if not _foundation_completion_matches_inputs(
        completion,
        dataset_dir=foundation_config.data.dataset_dir,
        tokenizer_path=config.data.tokenizer_model,
    ):
        raise RuntimeError(
            "foundation completion marker does not match the current prepared inputs"
        )
    source = _resolve_bound_foundation_checkpoint(
        run_root / "checkpoints" / "best",
        marker,
        context,
    )
    checkpoint_step = preflight_checkpoint_identity(
        source,
        context,
        checkpoint_identity,
    )
    if checkpoint_step != marker["selected_step"]:
        raise RuntimeError(
            "foundation completion marker selected_step does not match its checkpoint"
        )
    return {
        "schema": FOUNDATION_LINEAGE_SCHEMA,
        "release_name": marker["release_name"],
        "release_version": marker["release_version"],
        "languages": list(marker["languages"]),
        "selected_step": int(marker["selected_step"]),
        "foundation_manifest_sha256": marker["foundation_manifest_sha256"],
        "tokenizer_sha256": marker["tokenizer_sha256"],
        "checkpoint_identity_sha256": marker["checkpoint_identity_sha256"],
        "checkpoint_artifact_sha256": marker["checkpoint_artifact_sha256"],
    }


def resolve_foundation_lineage(
    config: AppConfig,
    foundation_plan: Any,
    context: DistributedContext,
) -> dict[str, Any]:
    """Resolve base ancestry on rank 0 and publish the exact mapping to peers."""

    raw_lineage = _run_long_rank_zero_action(
        context,
        foundation_run_directory(config) / ".foundation-lineage-status.json",
        operation="foundation lineage validation",
        action=lambda: _inspect_foundation_lineage(config, foundation_plan, context),
    )
    return raw_lineage


def _foundation_source_sampling_weights(
    dataset: IndexedParallelDataset,
) -> dict[int, float]:
    """Convert the prepared language distribution into per-source multipliers.

    ``DistributedBucketBatchSampler`` balances source ids, while foundation
    preparation records the desired distribution by language.  Giving every
    source in a language ``target_share / language_count`` makes the summed
    source mass equal that language's target share without treating a language
    with more shard files as intrinsically more important.
    """

    manifest_path = dataset.dataset_root / "manifest.json"
    try:
        manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
        language_sampling = cast(dict[str, Any], manifest["language_sampling"])
        language_counts = cast(dict[str, Any], language_sampling["counts"])
        language_weights = cast(dict[str, Any], language_sampling["weights"])
        raw_sources = cast(list[Any], manifest["sources"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"foundation manifest has no usable language sampling policy: {manifest_path}"
        ) from error

    source_entries: list[dict[str, Any]] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            raise ValueError(f"foundation manifest has an invalid source entry: {raw_source!r}")
        source = cast(dict[str, Any], raw_source)
        try:
            source_id = int(source["id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"foundation manifest has an invalid source id: {source!r}") from error
        name = str(source.get("name", ""))
        language = str(source.get("language", ""))
        if language not in language_counts or language not in language_weights:
            raise ValueError(
                f"foundation source {name!r} has no language sampling weight for {language!r}"
            )
        if not name:
            raise ValueError(f"foundation source has no usable name: {source!r}")
        source_entries.append(source)

    task_aware = bool(source_entries) and all(
        source.get("task") in {"denoising", "reasoning"} for source in source_entries
    )
    multipliers: dict[int, float] = {}
    if task_aware:
        if dataset.pair_source_ids is None:
            raise ValueError("foundation task sampling requires source_id metadata")
        counts_by_source = np.bincount(
            dataset.pair_source_ids.astype(np.int64, copy=False),
            minlength=len(dataset.source_names),
        ).astype(np.float64)
        task_language_counts: dict[tuple[str, str], float] = {}
        for source in source_entries:
            source_id = int(source["id"])
            task = str(source["task"])
            language = str(source["language"])
            task_language_counts[(task, language)] = (
                task_language_counts.get((task, language), 0.0) + counts_by_source[source_id]
            )

        reasoning_manifest = manifest.get("reasoning")
        reasoning_policy = (
            cast(dict[str, Any], reasoning_manifest) if isinstance(reasoning_manifest, dict) else {}
        )
        reasoning_share = float(reasoning_policy.get("sample_share", 0.05))
        if not 0.0 <= reasoning_share <= 0.10:
            raise ValueError("foundation reasoning sample_share must be in [0, 0.10]")
        configured_task_shares = {
            "denoising": 1.0 - reasoning_share,
            "reasoning": reasoning_share,
        }
        active_tasks = {
            task
            for task, _ in task_language_counts
            if any(
                count > 0.0
                for (candidate_task, _), count in task_language_counts.items()
                if candidate_task == task
            )
        }
        task_mass = sum(configured_task_shares[task] for task in active_tasks)
        if task_mass <= 0.0:
            raise ValueError("foundation task sampling excludes every available source")

        language_mass_by_task = {
            task: sum(
                float(language_weights[language])
                for candidate_task, language in task_language_counts
                if candidate_task == task and task_language_counts[(candidate_task, language)] > 0
            )
            for task in active_tasks
        }
        for source in source_entries:
            source_id = int(source["id"])
            task = str(source["task"])
            language = str(source["language"])
            count = counts_by_source[source_id]
            task_language_count = task_language_counts[(task, language)]
            language_mass = language_mass_by_task.get(task, 0.0)
            if count <= 0.0 or task_language_count <= 0.0:
                multipliers[source_id] = 0.0
                continue
            if language_mass <= 0.0:
                raise ValueError(f"foundation task {task!r} has no language sampling mass")
            normalized_task_share = configured_task_shares[task] / task_mass
            normalized_language_share = float(language_weights[language]) / language_mass
            multipliers[source_id] = (
                normalized_task_share * normalized_language_share / task_language_count
            )
    else:
        # Legacy foundation manifests predate explicit task metadata.  Preserve
        # their language-only distribution so old denoising runs remain resumable.
        for source in source_entries:
            source_id = int(source["id"])
            name = str(source.get("name", ""))
            language = str(source.get("language", ""))
            try:
                count = float(language_counts[language])
                target_share = float(language_weights[language])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"foundation source {name!r} has no language sampling weight for {language!r}"
                ) from error
            if count < 0.0 or target_share < 0.0:
                raise ValueError(
                    f"foundation source {name!r} has invalid language sampling values: "
                    f"count={count}, weight={target_share}"
                )
            if count == 0.0:
                if not math.isclose(target_share, 0.0, abs_tol=1e-12):
                    raise ValueError(
                        f"foundation source {name!r} has a positive sampling weight "
                        f"for empty language {language!r}"
                    )
                multiplier = 0.0
            else:
                multiplier = target_share / count
            previous = multipliers.get(source_id)
            if previous is not None and not math.isclose(previous, multiplier):
                raise ValueError(
                    f"foundation source id {source_id} has conflicting language weights"
                )
            multipliers[source_id] = multiplier

    missing = sorted(set(range(len(dataset.source_names))) - set(multipliers))
    if missing:
        raise ValueError(f"foundation manifest has no language for source ids: {missing}")
    maximum = max(multipliers.values(), default=0.0)
    if maximum <= 0.0:
        raise ValueError("foundation language sampling excludes every source")
    return {source_id: value / maximum for source_id, value in multipliers.items()}


def _foundation_training_contract(
    foundation_config: AppConfig,
    context: DistributedContext,
    *,
    planned_languages: Sequence[str],
    verify_integrity: bool,
) -> tuple[
    IndexedParallelDataset,
    DistributedBucketBatchSampler,
    tuple[str, ...],
]:
    """Open the train split and resolve its effective sampled language contract."""

    train_dataset = IndexedParallelDataset(
        foundation_config.data.dataset_dir,
        foundation_config.data.train_split,
        bidirectional=foundation_config.data.bidirectional,
        legacy_language_pairs=foundation_config.data.configured_language_pairs(),
        verify_integrity=verify_integrity,
    )
    train_sampler = DistributedBucketBatchSampler(
        train_dataset,
        foundation_config.training.batch_size_per_gpu,
        rank=context.rank,
        world_size=context.world_size,
        bucket_size=foundation_config.data.bucket_size,
        seed=foundation_config.training.seed,
        source_sampling_weights_by_id=_foundation_source_sampling_weights(train_dataset),
        # The prepared manifest already defines the intended alpha-tempered
        # language distribution. The generic sampler's 3x safety cap would
        # silently alter that policy for sufficiently imbalanced corpora.
        max_source_upsampling=math.inf,
    )
    languages = _foundation_languages_with_positive_sampling_mass(
        train_dataset,
        train_sampler,
        planned_languages,
    )
    return train_dataset, train_sampler, languages


def run_foundation_stage(
    config: AppConfig,
    foundation_plan: Any,
    model: torch.nn.Module,
    tokenizer: SionTokenizer,
    context: DistributedContext,
    *,
    artifacts_verified: bool = False,
    _resume_lease_scope: ExitStack | None = None,
) -> FoundationOutcome:
    """Build the encoder-decoder through denoising and optional reasoning.

    Completed work must not run again. This is the longest pipeline stage and can
    take days, so a later translation-training failure must not repeat it. When a
    valid completion marker exists, skip training and inherit only the best weights.
    """

    if not foundation_plan.enabled:
        return FoundationOutcome(ran=False, reason=foundation_plan.reason)
    if _resume_lease_scope is None:
        with ExitStack() as resume_lease_scope:
            return run_foundation_stage(
                config,
                foundation_plan,
                model,
                tokenizer,
                context,
                artifacts_verified=artifacts_verified,
                _resume_lease_scope=resume_lease_scope,
            )

    foundation_config = build_foundation_config(config)
    train_dataset, train_sampler, foundation_languages = _foundation_training_contract(
        foundation_config,
        context,
        planned_languages=foundation_plan.languages,
        verify_integrity=not artifacts_verified,
    )
    run_root = foundation_run_directory(config)
    completion = run_root / FOUNDATION_COMPLETION_FILENAME
    best_checkpoint = run_root / "checkpoints" / "best"
    final_export_dir = run_root / "exports" / "best"
    checkpoint_identity = _foundation_checkpoint_identity(foundation_config, context)
    marker = _read_foundation_completion(completion) if context.is_main else None
    marker_is_current_here = bool(
        context.is_main
        and marker is not None
        and marker.get("schema") == FOUNDATION_COMPLETION_SCHEMA
    )
    completed_inputs_match_here = bool(
        context.is_main
        and marker is not None
        and _foundation_completion_matches_inputs(
            completion,
            dataset_dir=foundation_config.data.dataset_dir,
            tokenizer_path=config.data.tokenizer_model,
        )
    )
    marker_contract_matches_here = bool(
        marker_is_current_here
        and marker is not None
        and _foundation_marker_contract_matches(
            marker,
            config=config,
            foundation_languages=foundation_languages,
            checkpoint_identity=checkpoint_identity,
        )
    )
    stale_completed_run_here = bool(
        marker_is_current_here
        and (not marker_contract_matches_here or not completed_inputs_match_here)
    )
    stale_completed_run = broadcast_bool(stale_completed_run_here, context)

    def archive_completed_run(reason: str) -> None:
        backup = _run_rank_zero_action(
            context,
            lambda: backup_stale_dataset(run_root),
            description="archiving an invalid foundation run",
        )
        if context.is_main:
            assert isinstance(backup, Path)
            announce(f"{reason}; preserving the previous run under {backup.name}/.", context)
        archive_visibility_scope = distributed_failure_scope(run_root.exists(), context)
        if archive_visibility_scope != "none":
            raise RuntimeError(
                "archived foundation run remains visible on at least one distributed rank"
            )

    if stale_completed_run:
        archive_completed_run(
            "The completed foundation generation's input, configuration, or "
            "checkpoint layout differs from the current run"
        )
        marker = None
        marker_is_current_here = False
        completed_inputs_match_here = False

    reusable_completion_here = bool(
        context.is_main and marker_is_current_here and completed_inputs_match_here
    )
    reusable_completion = broadcast_bool(reusable_completion_here, context)
    export_complete = False
    if reusable_completion:
        export_complete = bool(
            _run_long_rank_zero_action(
                context,
                run_root / ".foundation-export-validation-status.json",
                operation="foundation export validation",
                action=lambda: _foundation_export_is_complete(
                    final_export_dir,
                    required_formats=config.foundation.final_export_formats,
                    release_name=config.foundation.release_name,
                ),
            )
        )
    checkpoint_source = best_checkpoint
    expected_checkpoint_step = 0

    if reusable_completion:
        expected_checkpoint_digest = broadcast_text(
            cast(str, marker.get("checkpoint_artifact_sha256"))
            if context.is_main and marker is not None
            else None,
            context,
        )
        expected_checkpoint_step = broadcast_int(
            cast(int, marker.get("selected_step"))
            if context.is_main and marker is not None
            else None,
            context,
        )
        try:
            checkpoint_source = Path(
                _coordinated_resume_preflight(
                    best_checkpoint,
                    checkpoint_identity,
                    context,
                    stage="completed foundation",
                    lease_scope=_resume_lease_scope,
                    expected_artifact_sha256=expected_checkpoint_digest,
                    expected_step=expected_checkpoint_step,
                )
            )
        except Exception as error:
            if context.distributed and _has_unverifiable_legacy_dcp_generation(best_checkpoint):
                raise RuntimeError(
                    "The completed foundation checkpoint is a legacy DCP without "
                    "historical shard digests. Automatic archive/retraining and "
                    "promotion of the current bytes are refused. Inspect the original "
                    f"backup and perform explicit offline recovery: {error}"
                ) from error
            archive_completed_run(
                "Could not authenticate an exact checkpoint generation matching "
                "the foundation completion marker on every rank; training a new "
                f"generation: {error}"
            )
            reusable_completion = False

    if reusable_completion:
        provenance: dict[str, Any] | None = None
        load_error: Exception | None = None
        try:
            provenance = initialize_model_from_checkpoint(
                checkpoint_source,
                model,
                context,
                expected_identity=checkpoint_identity,
            )
        except Exception as error:
            load_error = error
        load_failure_scope = distributed_failure_scope(
            load_error is not None,
            context,
        )
        if load_failure_scope != "none":
            detail = load_error or RuntimeError(
                "at least one distributed rank could not load foundation weights"
            )
            raise RuntimeError(
                "foundation checkpoint load failed after successful immutable preflight; "
                f"refusing to continue with partially mutated ranks: {detail}"
            ) from load_error
        assert provenance is not None
        export_binding_matches_here = False
        if (
            context.is_main
            and export_complete
            and completed_inputs_match_here
            and marker is not None
        ):
            try:
                export_binding_matches_here = all(
                    (
                        marker.get("schema") == FOUNDATION_COMPLETION_SCHEMA,
                        marker.get("selected_step") == int(provenance["step"]),
                        marker.get("checkpoint_identity_sha256")
                        == _mapping_sha256(checkpoint_identity),
                        marker.get("export_manifest_sha256")
                        == file_sha256(final_export_dir / "export_manifest.json"),
                    )
                )
            except OSError:
                export_binding_matches_here = False
        export_binding_matches = broadcast_bool(export_binding_matches_here, context)
        repaired_export = False
        if not export_binding_matches:
            preflight_final_export_dependencies(config.foundation.final_export_formats)
            announce(
                "The completed foundation checkpoint is valid, but the final base "
                "export is missing, corrupt, or not proven to use the same weight "
                "generation. Re-exporting without retraining.",
                context,
            )
            final_export_dir = export_final_model(
                model,
                foundation_config,
                context,
                Path(config.training.output_dir),
                stage=FOUNDATION_STAGE_DIRECTORY,
                step=int(provenance["step"]),
                formats=config.foundation.final_export_formats,
                release_name=config.foundation.release_name,
                translation_capable=False,
                languages=foundation_languages,
                authenticated_revision_directions=(),
            )
            repaired_export = True
        if repaired_export:
            if context.is_main:
                assert marker is not None
            marker_for_update = marker if marker is not None else {}
            _run_rank_zero_action(
                context,
                lambda: _update_foundation_export_binding(
                    completion,
                    marker_for_update,
                    checkpoint_source=Path(str(provenance["source"])),
                    export_dir=final_export_dir,
                ),
                description="publishing the repaired foundation export binding",
            )
        announce(
            "The foundation stage is already complete; inheriting weights from "
            f"{checkpoint_source} (step {provenance['step']:,}).",
            context,
        )
        return FoundationOutcome(
            ran=False,
            reason="Reused weights from the completed foundation stage.",
            best_checkpoint=str(checkpoint_source),
            selected_step=provenance["step"],
            languages=foundation_languages,
            warnings=foundation_plan.warnings,
        )

    resume_plan = _run_long_rank_zero_action(
        context,
        run_root / ".foundation-resume-resolution-status.json",
        operation="foundation resume resolution",
        action=lambda: _inspect_foundation_resume(foundation_config, context),
    )
    resume_state = resume_plan.get("state")
    if resume_state not in {"absent", "available", "invalid", "untrusted_legacy"}:
        raise RuntimeError(f"foundation resume resolution returned invalid state: {resume_state!r}")
    if resume_state == "untrusted_legacy":
        raise RuntimeError(
            "The foundation resume checkpoint is a legacy DCP without historical "
            "shard digests, so automatic archive/retraining and promotion of the "
            "current bytes are refused. Inspect the original backup and perform "
            "explicit offline recovery: "
            f"{resume_plan.get('reason', 'unverifiable legacy generation')}"
        )
    has_resume = resume_state == "available"
    if resume_state == "invalid":
        archive_completed_run(
            "Could not authenticate the foundation resume checkpoint; training "
            "a new generation: "
            f"{resume_plan.get('reason', 'unknown resolution failure')}"
        )

    resume_source: Path | None = None
    trusted_best_artifact_sha256: str | None = None
    if has_resume:
        latest_checkpoint = run_root / "checkpoints" / "latest"
        previous_latest = latest_checkpoint.with_name(f".{latest_checkpoint.name}.previous")
        resume_generation = resume_plan.get("generation")
        if resume_generation not in {"current", "previous"}:
            raise RuntimeError(
                f"foundation resume resolution returned invalid generation: {resume_generation!r}"
            )
        resume_source = previous_latest if resume_generation == "previous" else latest_checkpoint
        try:
            resume_artifact_sha256 = resume_plan.get("checkpoint_artifact_sha256")
            if not isinstance(resume_artifact_sha256, str):
                raise ValueError("foundation resume resolution omitted its artifact digest")
            resume_source = Path(
                _coordinated_resume_preflight(
                    resume_source,
                    checkpoint_identity,
                    context,
                    stage="foundation",
                    lease_scope=_resume_lease_scope,
                    expected_artifact_sha256=resume_artifact_sha256,
                )
            )
            resume_training_state: dict[str, Any] | None = None
            training_state_error: Exception | None = None
            try:
                resume_training_state = inspect_checkpoint_training_state(
                    resume_source,
                    context,
                )
            except Exception as error:
                training_state_error = error
            training_state_failure_scope = distributed_failure_scope(
                training_state_error is not None,
                context,
            )
            if training_state_failure_scope != "none":
                failure = RuntimeError(
                    "foundation resume training state could not be inspected on every rank"
                )
                if training_state_error is not None:
                    raise failure from training_state_error
                raise failure
            assert resume_training_state is not None
            raw_best_digest = resume_training_state.get("best_checkpoint_artifact_sha256")
            if (
                isinstance(raw_best_digest, str)
                and len(raw_best_digest) == 64
                and all(character in "0123456789abcdef" for character in raw_best_digest)
            ):
                trusted_best_artifact_sha256 = raw_best_digest
            consensus_best_digest = broadcast_text(
                (trusted_best_artifact_sha256 or "") if context.is_main else None,
                context,
            )
            mismatch_scope = distributed_failure_scope(
                (trusted_best_artifact_sha256 or "") != consensus_best_digest,
                context,
            )
            if mismatch_scope != "none":
                raise RuntimeError("foundation latest best-artifact binding differs across ranks")
            trusted_best_artifact_sha256 = consensus_best_digest or None
        except Exception as error:
            archive_completed_run(
                "Could not authenticate the generation, v2 inventory, and identity "
                f"of the foundation resume checkpoint; training a new generation: {error}"
            )
            has_resume = False
            resume_source = None

    if has_resume:
        assert resume_source is not None
        quarantine = _run_rank_zero_action(
            context,
            lambda: _quarantine_untrusted_foundation_derivatives(
                run_root,
                trusted_best_artifact_sha256=trusted_best_artifact_sha256,
            ),
            description="quarantining unbound foundation best and export artifacts",
        )
        if context.is_main and quarantine is not None:
            announce(
                "Foundation latest does not content-bind its completion/best/export. "
                "Separating those artifacts from the resume checkpoint and "
                f"preserving them under {quarantine.name}/.",
                context,
            )
        derivative_visibility_scope = distributed_failure_scope(
            any(
                path.exists() or path.is_symlink()
                for path in _foundation_untrusted_derivatives(
                    run_root,
                    trusted_best_artifact_sha256=trusted_best_artifact_sha256,
                )
            ),
            context,
        )
        if derivative_visibility_scope != "none":
            raise RuntimeError(
                "untrusted foundation best/export artifacts remain visible on at least one "
                "distributed rank"
            )
        foundation_config.training.resume_from = str(resume_source)
        announce(f"Foundation: previous run found; resuming from {resume_source}.", context)
    else:
        untrusted_partial_run_here = bool(
            context.is_main
            and run_root.exists()
            and any(
                (
                    completion.exists(),
                    best_checkpoint.exists(),
                    final_export_dir.exists(),
                    (run_root / "checkpoints" / "latest").exists(),
                )
            )
        )
        untrusted_partial_run = broadcast_bool(untrusted_partial_run_here, context)
        if untrusted_partial_run:
            archive_completed_run(
                "Separating a partial foundation run with no authenticatable "
                "completion marker or resume checkpoint from new training"
            )

    preflight_final_export_dependencies(config.foundation.final_export_formats)
    validation_dataset = IndexedParallelDataset(
        foundation_config.data.dataset_dir,
        foundation_config.data.validation_split,
        bidirectional=foundation_config.data.bidirectional,
        legacy_language_pairs=foundation_config.data.configured_language_pairs(),
        verify_integrity=not artifacts_verified,
    )
    announce(
        f"Foundation data: {len(train_dataset):,} training examples / "
        f"{len(validation_dataset):,} validation examples "
        f"(languages: {', '.join(foundation_languages)})",
        context,
    )

    collator_args = build_collator_args(foundation_config, tokenizer)
    train_collator = SionBatchCollator(
        **collator_args,
        denoise_probability=foundation_config.data.denoise_probability,
        source_token_dropout=0.0,
        decoder_input_noise=0.0,
    )
    validation_collator = SionBatchCollator(
        **collator_args,
        denoise_probability=foundation_config.data.validation_denoise_probability,
        source_token_dropout=0.0,
        decoder_input_noise=0.0,
    )
    validation_sampler = DistributedBucketBatchSampler(
        validation_dataset,
        foundation_config.training.batch_size_per_gpu,
        rank=context.rank,
        world_size=context.world_size,
        bucket_size=foundation_config.data.bucket_size,
        seed=foundation_config.training.seed + 1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_collator,
        **dataloader_runtime_kwargs(
            foundation_config.data.num_workers,
            context.device,
            training=True,
        ),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        collate_fn=validation_collator,
        **dataloader_runtime_kwargs(
            0 if foundation_config.data.num_workers == 0 else 1,
            context.device,
            training=False,
        ),
    )

    announce(
        "Starting stage 0 foundation pretraining (denoising and optional reasoning).",
        context,
    )
    result = train(
        model,
        train_loader,
        validation_loader,
        foundation_config,
        context,
        stage_name="foundation/denoising",
        export_release_name=config.foundation.release_name,
        export_translation_capable=False,
        export_languages=foundation_languages,
        authenticated_revision_directions=(),
    )
    selected_checkpoint = Path(str(result["selected_checkpoint_source"]))
    selected_checkpoint_digest = str(result["selected_checkpoint_artifact_sha256"])
    selected_checkpoint_error: Exception | None = None
    try:
        actual_selected_digest = _foundation_checkpoint_artifact_sha256(selected_checkpoint)
        if actual_selected_digest != selected_checkpoint_digest:
            raise RuntimeError(
                "trainer selected checkpoint digest changed before foundation completion "
                f"({actual_selected_digest} != {selected_checkpoint_digest})"
            )
    except Exception as error:
        selected_checkpoint_error = error
    selected_checkpoint_failure_scope = distributed_failure_scope(
        selected_checkpoint_error is not None,
        context,
    )
    if selected_checkpoint_failure_scope != "none":
        failure = RuntimeError(
            "foundation selected checkpoint could not be rebound on every distributed rank"
        )
        if selected_checkpoint_error is not None:
            raise failure from selected_checkpoint_error
        raise failure
    barrier(context)
    release_stage_resources(context, train_loader, validation_loader)
    del train_loader, validation_loader, train_sampler, validation_sampler
    del train_collator, validation_collator, train_dataset, validation_dataset

    final_export_dir = export_final_model(
        model,
        foundation_config,
        context,
        Path(config.training.output_dir),
        stage=FOUNDATION_STAGE_DIRECTORY,
        step=int(result["selected_step"]),
        formats=config.foundation.final_export_formats,
        # This is the foundation, not the translation model. A distinct name alone
        # is insufficient: the architectures match, so publishing it without this
        # capability boundary would accept direction tags and emit plausible junk.
        release_name=config.foundation.release_name,
        translation_capable=False,
        languages=foundation_languages,
        authenticated_revision_directions=(),
    )
    _run_rank_zero_action(
        context,
        lambda: _atomic_write_foundation_completion(
            completion,
            _foundation_completion_marker(
                config=config,
                foundation_languages=foundation_languages,
                foundation_config=foundation_config,
                checkpoint_identity=checkpoint_identity,
                checkpoint_source=selected_checkpoint,
                selected_step=int(result["selected_step"]),
                best_validation_loss=float(result["best_validation_loss"]),
                export_dir=final_export_dir,
            ),
        ),
        description="publishing the foundation completion marker",
    )
    return FoundationOutcome(
        ran=True,
        reason=foundation_plan.reason,
        best_checkpoint=str(selected_checkpoint),
        selected_step=int(result["selected_step"]),
        languages=foundation_languages,
        warnings=foundation_plan.warnings,
    )


def find_auto_resume(config: AppConfig) -> str | None:
    """Return the latest checkpoint path from a previous run when one exists."""
    latest = Path(config.training.output_dir) / "checkpoints" / "latest"
    if checkpoint_path_exists(latest):
        return str(latest)
    return None


def _find_distributed_auto_resume(
    *,
    explicit: str | None,
    automatic: Path,
    context: DistributedContext,
    stage: str,
) -> str | None:
    """Choose one resume branch without allowing rank-local filesystem control flow."""

    selected_explicit = broadcast_text(
        str(explicit or "") if context.is_main else None,
        context,
    )
    mismatch_scope = distributed_failure_scope(str(explicit or "") != selected_explicit, context)
    if mismatch_scope != "none":
        raise RuntimeError(f"{stage} explicit resume path differs across distributed ranks")
    if selected_explicit:
        return selected_explicit

    present_here = checkpoint_path_exists(automatic)
    absence_scope = distributed_failure_scope(not present_here, context)
    if absence_scope == "partial":
        raise RuntimeError(
            f"{stage} auto-resume checkpoint visibility differs across distributed ranks: "
            f"{automatic}"
        )
    if absence_scope == "all":
        return None
    return str(automatic)


def _archive_invalid_automatic_resume(
    stage_root: Path,
    context: DistributedContext,
    *,
    stage: str,
    error: BaseException,
) -> None:
    """Move an invalid automatically discovered stage aside before a fresh run."""

    backup = _run_rank_zero_action(
        context,
        lambda: backup_stale_dataset(stage_root),
        description=f"archiving invalid automatic {stage} resume",
    )
    if context.is_main:
        assert isinstance(backup, Path)
        announce(
            f"Could not authenticate the automatic {stage} resume candidate. "
            f"Preserving it under {backup.name}/ and returning to a fresh path: "
            f"{_bounded_status_text(error)}",
            context,
        )
    visibility_scope = distributed_failure_scope(
        stage_root.exists() or stage_root.is_symlink(),
        context,
    )
    if visibility_scope != "none":
        raise RuntimeError(
            f"archived automatic {stage} resume remains visible on at least one rank"
        )


def _checkpoint_pipeline_identity(
    source: str | Path,
    config: AppConfig,
    foundation_plan: Any,
    context: DistributedContext,
) -> dict[str, Any]:
    """Recover and validate ancestry already authenticated by an SFT checkpoint."""

    recorded_identity = inspect_checkpoint_identity(source, context)
    raw_pipeline = recorded_identity.get("pipeline")
    if not isinstance(raw_pipeline, dict):
        raise ValueError("SFT checkpoint has no recorded pipeline identity")
    branch = raw_pipeline.get("branch")
    if branch == "foundation-then-translation":
        if not config.foundation.enabled:
            raise ValueError(
                "checkpoint has foundation ancestry but foundation is explicitly disabled"
            )
        raw_foundation = raw_pipeline.get("foundation")
        if not isinstance(raw_foundation, dict):
            raise ValueError("foundation checkpoint branch has no recorded lineage")
        # A self-contained translation/posttraining checkpoint remains valid
        # when the raw monolingual corpus is temporarily offline. Reconstruct
        # the configured branch from its authenticated lineage instead of from
        # whether this machine can launch a fresh foundation run today.
        configured_lineage_plan = cast(
            FoundationPlan,
            SimpleNamespace(enabled=True, languages=config.foundation_languages()),
        )
        lineage_plan = _foundation_plan_for_lineage(
            configured_lineage_plan,
            cast(dict[str, Any], raw_foundation),
        )
        pipeline = build_translation_pipeline_identity(
            lineage_plan,
            foundation_lineage=raw_foundation,
        )
        if raw_foundation.get("release_name") != config.foundation.release_name:
            raise ValueError(
                "checkpoint foundation release does not match the current configured base"
            )
        tokenizer = recorded_identity.get("tokenizer")
        tokenizer_model = tokenizer.get("model") if isinstance(tokenizer, dict) else None
        tokenizer_sha256 = (
            tokenizer_model.get("sha256") if isinstance(tokenizer_model, dict) else None
        )
        if tokenizer_sha256 != raw_foundation.get("tokenizer_sha256"):
            raise ValueError(
                "checkpoint foundation lineage tokenizer does not match its tokenizer identity"
            )
    elif branch == "translation-only":
        if config.foundation.enabled:
            raise ValueError(
                "translation-only checkpoint conflicts with the configured foundation-first branch"
            )
        pipeline = build_translation_pipeline_identity(foundation_plan)
    else:
        raise ValueError("checkpoint pipeline identity has an unsupported branch")
    if raw_pipeline != pipeline:
        raise ValueError("checkpoint pipeline identity does not match the current branch")
    return pipeline


def _coordinated_checkpoint_pipeline_identity(
    source: str | Path,
    config: AppConfig,
    foundation_plan: Any,
    context: DistributedContext,
) -> dict[str, Any]:
    pipeline: dict[str, Any] | None = None
    pipeline_error: Exception | None = None
    try:
        pipeline = _checkpoint_pipeline_identity(
            source,
            config,
            foundation_plan,
            context,
        )
    except Exception as error:
        pipeline_error = error
    failure_scope = distributed_failure_scope(pipeline_error is not None, context)
    if failure_scope != "none":
        failure = RuntimeError(
            "SFT checkpoint pipeline identity could not be validated on every rank"
        )
        if pipeline_error is not None:
            raise failure from pipeline_error
        raise failure
    assert pipeline is not None
    return pipeline


def _coordinated_exact_checkpoint_identity_preflight(
    source: str | Path,
    expected_identity: dict[str, Any],
    context: DistributedContext,
    *,
    stage: str,
) -> int:
    checkpoint_step: int | None = None
    identity_error: Exception | None = None
    try:
        checkpoint_step = preflight_checkpoint_identity(
            source,
            context,
            expected_identity,
        )
        if checkpoint_step is None:
            raise ValueError(f"{stage} checkpoint has no recorded step")
    except Exception as error:
        identity_error = error
    failure_scope = distributed_failure_scope(identity_error is not None, context)
    if failure_scope != "none":
        failure = RuntimeError(f"{stage} checkpoint identity could not be validated on every rank")
        if identity_error is not None:
            raise failure from identity_error
        raise failure
    assert checkpoint_step is not None
    consensus_step = broadcast_int(checkpoint_step if context.is_main else None, context)
    mismatch_scope = distributed_failure_scope(checkpoint_step != consensus_step, context)
    if mismatch_scope != "none":
        raise RuntimeError(f"{stage} checkpoint step differs across distributed ranks")
    return checkpoint_step


def _coordinated_resume_preflight(
    candidate: str | Path,
    expected_identity: dict[str, Any] | None,
    context: DistributedContext,
    *,
    stage: str,
    lease_scope: ExitStack | None = None,
    expected_artifact_sha256: str | None = None,
    expected_step: int | None = None,
) -> str:
    """Authenticate one exact generation and retain its one-shot load lease."""

    if context.distributed and lease_scope is None:
        raise ValueError(
            f"{stage} distributed resume requires a lease scope that survives until load"
        )
    generation_context = verified_checkpoint_generation_lease(
        candidate,
        context,
        expected_identity,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_step=expected_step,
    )
    if lease_scope is not None:
        generation = lease_scope.enter_context(generation_context)
        return str(generation.source)
    with generation_context as generation:
        return str(generation.source)


def _configured_foundation_branch_plan(
    config: AppConfig,
    discovered_plan: FoundationPlan,
) -> FoundationPlan:
    """Keep configured ancestry separate from today's raw-corpus availability."""

    if not config.foundation.enabled or discovered_plan.enabled:
        return discovered_plan
    return FoundationPlan(
        enabled=True,
        reason=(
            "Foundation is enabled in the configuration, but its source corpus "
            "is currently unavailable. Reuse only an authenticated downstream "
            "checkpoint or prepared base generation."
        ),
        discovery=discovered_plan.discovery,
        languages=discovered_plan.languages,
        report=discovered_plan.report,
        warnings=discovered_plan.warnings,
    )


def _build_posttraining_config(config: AppConfig, run_root: Path) -> AppConfig:
    post = config.posttraining
    post_config = copy.deepcopy(config)
    post_config.training.output_dir = str(run_root / "posttrain")
    post_config.training.num_train_epochs = post.num_train_epochs
    post_config.training.max_steps = post.max_steps
    post_config.training.batch_size_per_gpu = post.batch_size_per_gpu
    post_config.training.gradient_accumulation_steps = post.gradient_accumulation_steps
    post_config.training.learning_rate = post.learning_rate
    post_config.training.warmup_steps = post.warmup_steps
    post_config.training.eval_every = post.eval_every
    post_config.training.save_every = post.save_every
    post_config.training.early_stopping_min_epochs = post.early_stopping_min_epochs
    post_config.training.early_stopping_patience = post.early_stopping_patience
    post_config.training.resume_from = None
    post_config.training.tensorboard_dir = None
    return post_config


def _coordinated_checkpoint_load_structure(
    source: str | Path,
    model: torch.nn.Module,
    config: AppConfig,
    context: DistributedContext,
    *,
    stage: str,
) -> int:
    checkpoint_step: int | None = None
    structure_error: Exception | None = None
    try:
        checkpoint_step = preflight_checkpoint_load_structure(
            source,
            model,
            context,
            require_scaler=(
                context.device.type == "cuda" and config.training.precision.lower() == "fp16"
            ),
            require_ema=config.training.ema_decay > 0,
        )
    except Exception as error:
        structure_error = error
    failure_scope = distributed_failure_scope(structure_error is not None, context)
    if failure_scope != "none":
        failure = RuntimeError(
            f"{stage} checkpoint load structure could not be validated on every rank"
        )
        if structure_error is not None:
            raise failure from structure_error
        raise failure
    assert checkpoint_step is not None
    consensus_step = broadcast_int(checkpoint_step if context.is_main else None, context)
    mismatch_scope = distributed_failure_scope(checkpoint_step != consensus_step, context)
    if mismatch_scope != "none":
        raise RuntimeError(f"{stage} checkpoint structural step differs across ranks")
    return checkpoint_step


def _select_translation_resume_candidate(
    candidate: str | Path,
    config: AppConfig,
    foundation_plan: FoundationPlan,
    model: torch.nn.Module,
    batch_sampler: Any,
    context: DistributedContext,
    *,
    stage: str,
    stage_name: str,
    include_posttraining: bool,
    lease_scope: ExitStack,
) -> tuple[str, dict[str, Any]]:
    """Select current/previous transactionally and retain only the winner's lease."""

    bindings = checkpoint_generation_bindings(candidate, context)
    if not bindings:
        raise FileNotFoundError(f"{stage} checkpoint has no discoverable generation: {candidate}")
    last_error: BaseException | None = None
    rejected_generations: list[str] = []
    for generation_index, binding in enumerate(bindings, start=1):
        attempt_scope = ExitStack()
        try:
            source = _coordinated_resume_preflight(
                candidate,
                None,
                context,
                stage=stage,
                lease_scope=attempt_scope,
                expected_artifact_sha256=binding.artifact_sha256,
            )
            candidate_pipeline_identity = _coordinated_checkpoint_pipeline_identity(
                source,
                config,
                foundation_plan,
                context,
            )
            expected_identity = build_training_checkpoint_identity(
                config,
                batch_sampler=batch_sampler,
                context=context,
                stage_name=stage_name,
                include_posttraining=include_posttraining,
                pipeline_identity=candidate_pipeline_identity,
            )
            identity_step = _coordinated_exact_checkpoint_identity_preflight(
                source,
                expected_identity,
                context,
                stage=stage,
            )
            structure_step = _coordinated_checkpoint_load_structure(
                source,
                model,
                config,
                context,
                stage=stage,
            )
            if structure_step != identity_step:
                raise RuntimeError(
                    f"{stage} checkpoint identity/structure step mismatch "
                    f"({identity_step} != {structure_step})"
                )
        except BaseException as error:
            last_error = error
            rejected_generations.append(
                f"generation {generation_index} "
                f"({binding.artifact_sha256[:12]}...): "
                f"{_bounded_status_text(error, max_bytes=512)}"
            )
            attempt_scope.close()
            continue
        lease_scope.enter_context(attempt_scope.pop_all())
        if rejected_generations:
            announce(
                f"[warning] {stage}: rejected a newer checkpoint before selecting "
                f"authenticated generation {generation_index} at {source}. "
                f"Rejected candidate details: {'; '.join(rejected_generations)}",
                context,
            )
        return source, candidate_pipeline_identity
    failure = RuntimeError(
        f"no authenticated, compatible, structurally loadable {stage} generation matched "
        f"{candidate}"
    )
    if last_error is not None:
        raise failure from last_error
    raise failure


def run_foundation_before_translation(
    config: AppConfig,
    foundation_plan: Any,
    model: torch.nn.Module,
    tokenizer: SionTokenizer,
    context: DistributedContext,
    *,
    validated_pretrain_resume: str | None,
    artifacts_verified: bool = False,
) -> FoundationOutcome:
    """Run foundation only when no validated SFT state will supersede it."""

    if validated_pretrain_resume:
        return FoundationOutcome(
            ran=False,
            reason="A validated SFT resume takes precedence over foundation execution/loading.",
        )
    return run_foundation_stage(
        config,
        foundation_plan,
        model,
        tokenizer,
        context,
        artifacts_verified=artifacts_verified,
    )


def translation_initialization_message(
    foundation_outcome: FoundationOutcome,
    *,
    resume_from: str | None,
    foundation_release_name: str = "sion",
    pipeline_branch: str | None = None,
) -> str:
    """Describe the ancestry that the translation stages will actually use."""

    if resume_from:
        branch = pipeline_branch or "validated"
        return (
            f"Resuming the {branch} SFT checkpoint {resume_from} first. The "
            "foundation stage will not be trained or loaded in this run, and the "
            "final release is sion_translate."
        )
    if foundation_outcome.best_checkpoint:
        return (
            "Starting translation training from foundation weights "
            f"({foundation_release_name} step {foundation_outcome.selected_step:,})."
        )
    return (
        "The foundation model (sion) will not be trained or exported. Starting "
        "from fresh initialization and proceeding directly through translation "
        "SFT/MRT to produce only sion_translate."
    )


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    context = initialize_distributed()
    run_scope = ExitStack()
    try:
        # ── Stage 1: detect the execution environment ────────────────────
        env = probe_environment()
        env = synchronize_environment(env, context)
        announce(f"Preparation 1: execution environment — {describe_environment(env)}", context)

        # ── Stage 2: load configuration ──────────────────────────────────
        config, raw, source = resolve_config(args)
        run_scope.enter_context(coordinated_training_run_lock(config.training.output_dir, context))
        checkpoint_lease_scope = ExitStack()
        run_scope.enter_context(checkpoint_lease_scope)
        announce(f"Preparation 2: configuration loaded — {source}", context)
        # The built-in collator has no dense alignment-label provider. Reject a
        # permanently-zero BATS alignment objective even for preparation-only
        # runs so an upload-ready artifact set cannot hide a training failure.
        config.validate_training_supervision(alignment_targets_available=False)

        # ── Stage 3: discover source data and prepare artifacts ──────────
        announce("Preparation 3: checking source data.", context)
        discovered_foundation_plan = plan_foundation_stage(config)
        foundation_plan = _configured_foundation_branch_plan(
            config,
            discovered_foundation_plan,
        )
        # Translation formats are unavoidable. Check them locally during a
        # preparation-only run as well; otherwise the same configuration can
        # spend upload and GPU setup time before reporting a missing converter.
        preflight_final_export_dependencies(config.training.final_export_formats)
        run_scope.enter_context(
            coordinated_artifact_run_locks(
                config,
                foundation_plan,
                context,
            )
        )
        pipeline_identity: dict[str, Any] | None = None
        initial_run_root = Path(config.training.output_dir)
        explicit_pretrain_resume = bool(config.training.resume_from)
        pretrain_resume_candidate = _find_distributed_auto_resume(
            explicit=config.training.resume_from,
            automatic=initial_run_root / "pretrain" / "checkpoints" / "latest",
            context=context,
            stage="SFT",
        )
        posttrain_resume_candidate = None
        if config.posttraining.enabled and not explicit_pretrain_resume:
            posttrain_resume_candidate = _find_distributed_auto_resume(
                explicit=None,
                automatic=initial_run_root / "posttrain" / "checkpoints" / "latest",
                context=context,
                stage="posttraining",
            )
        require_offline_foundation = bool(
            args.prepare_only
            or (
                foundation_plan.enabled
                and not discovered_foundation_plan.discovery.sources
                # Automatic checkpoints are only candidates until their full
                # lineage and state authentication succeeds. If one is stale,
                # execution falls back to foundation training, so raw-free base
                # shards must pass the complete preflight before model placement.
                # An explicit SFT resume fails closed instead of falling back.
                and not explicit_pretrain_resume
            )
        )
        ensure_artifacts(
            config,
            context,
            foundation_plan,
            # Effective auto-sized training identity and the exact SFT resume
            # generation must be known before deciding whether base preparation
            # can safely be skipped. A coarse `.metadata` presence check is not
            # authority to bypass the foundation path.
            prepare_foundation=args.prepare_only,
            # A raw-free fresh run must authenticate its prepared base shards
            # before constructing or placing a model on paid GPU memory.
            require_offline_foundation=require_offline_foundation,
            locks_held=True,
        )
        tokenizer = SionTokenizer(config.data.tokenizer_model)
        config.model.vocab_size = len(tokenizer)
        preflight_morphoscript_token_features(config, tokenizer)

        train_dataset = IndexedParallelDataset(
            config.data.dataset_dir,
            config.data.train_split,
            bidirectional=True,
            legacy_bidirectional=config.data.bidirectional,
            legacy_language_pairs=config.data.configured_language_pairs(),
            verify_integrity=False,
        )
        validation_dataset = IndexedParallelDataset(
            config.data.dataset_dir,
            config.data.validation_split,
            bidirectional=True,
            legacy_bidirectional=config.data.bidirectional,
            legacy_language_pairs=config.data.configured_language_pairs(),
            verify_integrity=False,
        )
        preflight_dataset_direction_contract(config, train_dataset, require_all_pairs=True)
        preflight_dataset_direction_contract(
            config,
            validation_dataset,
            require_all_directions=True,
        )
        announce(
            f"Data size: {len(train_dataset):,} training examples / "
            f"{len(validation_dataset):,} validation examples "
            "(including configured translation directions)",
            context,
        )

        # ── Stage 4: derive settings from data and hardware ──────────────
        if args.prepare_only:
            decisions = apply_auto_data_settings(
                config,
                raw,
                train_examples=len(train_dataset),
                physical_train_pairs=train_dataset.pair_count,
                # ensure_artifacts authenticated the complete prepared inventory
                # under the held artifact lease before these indexes were opened.
                physical_train_tokens=train_dataset.physical_token_count,
                source_names=train_dataset.source_names,
            )
        else:
            decisions = apply_auto_settings(
                config,
                raw,
                env,
                train_examples=len(train_dataset),
                validation_examples=len(validation_dataset),
                physical_train_pairs=train_dataset.pair_count,
                physical_train_tokens=train_dataset.physical_token_count,
                source_names=train_dataset.source_names,
            )
        if decisions:
            announce("Preparation 4: automatically selected settings —", context)
            for line in decisions:
                announce(f"  · {line}", context)
        config.validate()

        # Exported capabilities must reflect rows that the finalized sampling
        # policy can actually select. Resolve this before copying stage configs.
        train_sampler = DistributedBucketBatchSampler(
            train_dataset,
            config.training.batch_size_per_gpu,
            rank=context.rank,
            world_size=context.world_size,
            bucket_size=config.data.bucket_size,
            seed=config.training.seed,
            source_sampling_alpha=config.data.source_sampling_alpha,
            source_sampling_weights=config.data.source_sampling_weights,
            max_source_upsampling=config.data.max_source_upsampling,
        )
        positive_sampling_mask = train_sampler.positive_sampling_pair_mask()
        revision_directions = resolve_training_revision_directions(
            config,
            train_dataset,
            draft_token_id=tokenizer.draft_id,
            max_source_tokens=config.data.max_source_length - 2,
            physical_mask=positive_sampling_mask,
        )
        preflight_effective_translation_training(
            config,
            train_sampler,
            authenticated_revision_directions=revision_directions,
        )
        if args.prepare_only:
            announce(
                "Preparation-only run complete: tokenizer, indexed shards, exact "
                "language graph, sampling coverage, revision capabilities, automatic "
                "model sizing, and export dependencies passed local preflight.",
                context,
            )
            return

        # Initialize model parameters with the same rank-0 seed regardless of
        # world size. Reseed runtime randomness per rank after model construction.
        seed_everything(config.training.seed, 0)
        if revision_directions:
            announce(
                "Revision training directions authenticated with positive "
                "sampling probability: "
                + ", ".join(f"{source}→{target}" for source, target in revision_directions),
                context,
            )

        if not foundation_plan.enabled:
            pipeline_identity = build_translation_pipeline_identity(foundation_plan)

        # Keep pretraining and posttraining artifacts separate under the run root.
        run_root = Path(config.training.output_dir)
        pretrain_config = copy.deepcopy(config)
        pretrain_config.training.output_dir = str(run_root / "pretrain")
        post_config = (
            _build_posttraining_config(config, run_root) if config.posttraining.enabled else None
        )

        # ── Stage 5: discover previous stage checkpoints ─────────────────
        if not pretrain_config.training.resume_from and pretrain_resume_candidate:
            pretrain_config.training.resume_from = pretrain_resume_candidate
        if pretrain_config.training.resume_from:
            announce(
                "Preparation 5: previous SFT checkpoint candidate found → "
                f"{pretrain_config.training.resume_from}",
                context,
            )
        if posttrain_resume_candidate:
            announce(
                "Preparation 5: previous MRT checkpoint candidate found → "
                f"{posttrain_resume_candidate}",
                context,
            )

        # ── Build DataLoaders ─────────────────────────────────────────────
        # The collator tokenizes and pads sources/translations into tensor batches.
        collator_args = build_collator_args(config, tokenizer)
        train_collator = SionBatchCollator(
            **collator_args,
            denoise_probability=config.data.denoise_probability,
            # Apply online source-token dropout only during training.
            source_token_dropout=config.data.source_token_dropout,
            decoder_input_noise=config.data.decoder_input_noise,
        )
        validation_collator = SionBatchCollator(
            **collator_args,
            denoise_probability=config.data.validation_denoise_probability,
            source_token_dropout=0.0,  # Validation always uses clean input.
            decoder_input_noise=0.0,
        )
        # The sampler buckets similar lengths to reduce padding and partitions
        # batches across ranks without overlap during distributed training.
        post_sampler: DistributedBucketBatchSampler | None = None
        post_validation_sampler: DistributedBucketBatchSampler | None = None
        if post_config is not None:
            post = config.posttraining
            post_sampler = DistributedBucketBatchSampler(
                train_dataset,
                post.batch_size_per_gpu,
                rank=context.rank,
                world_size=context.world_size,
                bucket_size=config.data.bucket_size,
                seed=config.training.seed + 2,
                source_sampling_alpha=config.data.source_sampling_alpha,
                source_sampling_weights=config.data.source_sampling_weights,
                max_source_upsampling=config.data.max_source_upsampling,
            )
            post_validation_sampler = DistributedBucketBatchSampler(
                validation_dataset,
                post.eval_batch_size_per_gpu,
                rank=context.rank,
                world_size=context.world_size,
                bucket_size=config.data.bucket_size,
                seed=config.training.seed + 3,
            )
        validation_sampler = DistributedBucketBatchSampler(
            validation_dataset,
            config.training.batch_size_per_gpu,
            rank=context.rank,
            world_size=context.world_size,
            bucket_size=config.data.bucket_size,
            seed=config.training.seed + 1,
        )
        train_loader_args = dataloader_runtime_kwargs(
            config.data.num_workers,
            context.device,
            training=True,
        )
        validation_workers = (
            0 if config.data.num_workers == 0 else min(4, max(1, config.data.num_workers // 4))
        )
        validation_loader_args = dataloader_runtime_kwargs(
            validation_workers,
            context.device,
            training=False,
        )
        # ── Construct and distribute the model ───────────────────────────
        announce("Constructing the model and placing it on devices.", context)
        # Every CUDA strategy checks parameter count and persistent-state capacity
        # on the meta device first. Only after this passes do single/DDP allocate
        # the full model and FSDP2 allocate shards on actual GPUs. Oversized
        # configurations therefore fail clearly before a constructor OOM.
        parallel_strategy = resolve_parallel_strategy(
            config.training.parallel_strategy,
            context,
            legacy_fsdp2=config.training.fsdp2,
        )
        model, parameter_count, capacity, materialize_meta = construct_training_model(
            config,
            context,
            pad_id=tokenizer.pad_id,
            parallel_strategy=parallel_strategy,
        )
        # SFT and MRT share one DDP wrapper, so unused-parameter detection accounts
        # for the parameter-use set after the stage transition as well.
        detect_unused_parameters = requires_ddp_unused_parameter_detection(config)
        model = parallelize_model(
            model,
            context,
            strategy=config.training.parallel_strategy,
            use_fsdp2=config.training.fsdp2,
            precision=config.training.precision,
            reduce_dtype=config.training.fsdp_reduce_dtype,
            reshard_after_forward=config.training.reshard_after_forward,
            materialize_meta=materialize_meta,
            find_unused_parameters=detect_unused_parameters,
        )
        if config.training.compile:
            model = cast(torch.nn.Module, torch.compile(model))
        # Runtime randomness such as dropout and denoising must differ by rank.
        seed_everything(config.training.seed, context.rank)
        announce(
            f"Model parameters: {parameter_count:,}; parallel strategy: {parallel_strategy}",
            context,
        )
        if capacity is not None:
            announce(
                "Estimated persistent training state: "
                f"{capacity['per_rank_state_gib']:.1f} GiB per rank / "
                f"{capacity['state_budget_gib']:.1f} GiB safety budget",
                context,
            )

        # ── Stage 5: select checkpoints downstream-first ─────────────────
        # Recover the most advanced stage first so completed foundation/SFT work
        # does not run again. Validate each logical checkpoint's current and
        # previous generations through independent leases, retaining only one
        # fully validated lease through the actual load.
        validated_posttrain_resume = False
        if posttrain_resume_candidate:
            assert post_config is not None
            assert post_sampler is not None
            try:
                selected_posttrain, selected_pipeline = _select_translation_resume_candidate(
                    posttrain_resume_candidate,
                    post_config,
                    foundation_plan,
                    model,
                    post_sampler,
                    context,
                    stage="posttraining",
                    stage_name="posttrain/composite-MRT+preference",
                    include_posttraining=True,
                    lease_scope=checkpoint_lease_scope,
                )
            except BaseException as error:
                _archive_invalid_automatic_resume(
                    run_root / "posttrain",
                    context,
                    stage="posttraining",
                    error=error,
                )
                posttrain_resume_candidate = None
            else:
                post_config.training.resume_from = selected_posttrain
                pipeline_identity = selected_pipeline
                validated_posttrain_resume = True
                announce(
                    "Preparation 5: MRT resume fully validated; skipping "
                    "foundation and SFT execution/loading.",
                    context,
                )

        validated_pretrain_resume = False
        if not validated_posttrain_resume and pretrain_config.training.resume_from:
            try:
                selected_pretrain, selected_pipeline = _select_translation_resume_candidate(
                    pretrain_config.training.resume_from,
                    pretrain_config,
                    foundation_plan,
                    model,
                    train_sampler,
                    context,
                    stage="SFT",
                    stage_name="pretrain/SFT",
                    include_posttraining=False,
                    lease_scope=checkpoint_lease_scope,
                )
            except BaseException as error:
                if explicit_pretrain_resume:
                    raise RuntimeError("explicit SFT resume authentication failed") from error
                _archive_invalid_automatic_resume(
                    run_root / "pretrain",
                    context,
                    stage="SFT",
                    error=error,
                )
                pretrain_config.training.resume_from = None
                pretrain_resume_candidate = None
            else:
                pretrain_config.training.resume_from = selected_pretrain
                pipeline_identity = selected_pipeline
                validated_pretrain_resume = True
                announce(
                    "Preparation 5: SFT resume fully validated; skipping "
                    "foundation execution/loading.",
                    context,
                )

        # ── Stage 5b: foundation pretraining (denoising + reasoning) ─────
        # Build the encoder-decoder before exposing it to translation pairs.
        # This stage produces a foundation rather than a translation model, so it
        # is released under a separate name.
        if validated_posttrain_resume or validated_pretrain_resume:
            foundation_outcome = FoundationOutcome(
                ran=False,
                reason=(
                    "A validated downstream resume takes precedence over foundation "
                    "execution/loading."
                ),
            )
        else:
            if discovered_foundation_plan.enabled:
                ensure_artifacts(
                    config,
                    context,
                    discovered_foundation_plan,
                    prepare_foundation=True,
                    locks_held=True,
                )
            elif foundation_plan.enabled:
                _verify_prepared_artifact_consensus(
                    config,
                    foundation_plan,
                    context,
                    prepare_foundation=True,
                )
                announce(
                    "The foundation source corpus is offline, so only authenticated "
                    "prepared data and checkpoints will be used.",
                    context,
                )
            foundation_outcome = run_foundation_before_translation(
                config,
                foundation_plan,
                model,
                tokenizer,
                context,
                validated_pretrain_resume=None,
                artifacts_verified=True,
            )
        if pipeline_identity is None:
            if not foundation_plan.enabled:
                raise RuntimeError("translation pipeline identity was not initialized")
            foundation_lineage = resolve_foundation_lineage(
                config,
                foundation_plan,
                context,
            )
            effective_foundation_plan = _foundation_plan_for_lineage(
                foundation_plan,
                foundation_lineage,
            )
            pipeline_identity = build_translation_pipeline_identity(
                effective_foundation_plan,
                foundation_lineage=foundation_lineage,
            )
        if validated_posttrain_resume:
            assert post_config is not None
            announce(
                f"Resuming validated MRT checkpoint {post_config.training.resume_from} "
                "directly. Foundation and SFT will not be trained or loaded in this run.",
                context,
            )
        else:
            announce(
                translation_initialization_message(
                    foundation_outcome,
                    resume_from=pretrain_config.training.resume_from,
                    foundation_release_name=config.foundation.release_name,
                    pipeline_branch=pipeline_identity["branch"],
                ),
                context,
            )

        # ── Stage 6: SFT pretraining ─────────────────────────────────────
        pretrain_result: dict[str, float | int | bool | str] | None = None
        if validated_posttrain_resume:
            announce(
                "A validated MRT resume exists; skipping SFT DataLoader creation and training.",
                context,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                collate_fn=train_collator,
                **train_loader_args,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_sampler=validation_sampler,
                collate_fn=validation_collator,
                **validation_loader_args,
            )
            announce("Starting stage 1 SFT pretraining.", context)
            pretrain_result = train(
                model,
                train_loader,
                validation_loader,
                pretrain_config,
                context,
                stage_name="pretrain/SFT",
                language_tags=tokenizer.language_tags,
                authenticated_revision_directions=revision_directions,
                pipeline_identity=pipeline_identity,
            )
            barrier(context)
            memory = release_stage_resources(context, train_loader, validation_loader)
            del train_loader, validation_loader
            if memory:
                announce(
                    "Pretraining memory cleanup: "
                    f"allocated {memory['before_allocated_gib']:.2f}→"
                    f"{memory['after_allocated_gib']:.2f} GiB, "
                    f"reserved {memory['before_reserved_gib']:.2f}→"
                    f"{memory['after_reserved_gib']:.2f} GiB",
                    context,
                )
        del train_sampler, validation_sampler
        del train_collator, validation_collator

        # ── Stage 7: MRT posttraining ────────────────────────────────────
        if post_config is not None:
            post = config.posttraining
            assert post_sampler is not None
            assert post_validation_sampler is not None

            # Disable augmentation because rewards require clean sources/references.
            post_collator = SionBatchCollator(
                **collator_args,
                denoise_probability=0.0,
                source_token_dropout=0.0,
                decoder_input_noise=0.0,
            )
            post_loader = DataLoader(
                train_dataset,
                batch_sampler=post_sampler,
                collate_fn=post_collator,
                **train_loader_args,
            )
            post_validation_loader = DataLoader(
                validation_dataset,
                batch_sampler=post_validation_sampler,
                collate_fn=post_collator,
                **validation_loader_args,
            )
            objective = MinimumRiskObjective(tokenizer, post)
            announce(
                "Starting stage 2 composite MRT/preference posttraining: "
                f"{post.samples_per_source} candidates, risk {post.risk_weight:.2f}, "
                f"preference {post.preference_weight:.2f}, "
                f"validation beam {post.validation_num_beams}",
                context,
            )
            posttrain_result = train(
                model,
                post_loader,
                post_validation_loader,
                post_config,
                context,
                objective=objective,
                stage_name="posttrain/composite-MRT+preference",
                language_tags=tokenizer.language_tags,
                authenticated_revision_directions=revision_directions,
                pipeline_identity=pipeline_identity,
            )
            barrier(context)
            memory = release_stage_resources(
                context,
                post_loader,
                post_validation_loader,
            )
            if memory:
                announce(
                    "Posttraining memory cleanup: "
                    f"allocated {memory['before_allocated_gib']:.2f}→"
                    f"{memory['after_allocated_gib']:.2f} GiB, "
                    f"reserved {memory['before_reserved_gib']:.2f}→"
                    f"{memory['after_reserved_gib']:.2f} GiB",
                    context,
                )
            final_step = int(posttrain_result["selected_step"])
        else:
            announce("posttraining.enabled=false; skipping posttraining.", context)
            assert pretrain_result is not None
            final_step = int(pretrain_result["selected_step"])

        # Intermediate best/latest checkpoints store only lightweight formats
        # needed for resume and quick validation. After every training stage, emit
        # all requested formats once from the selected best weights so expensive
        # CPU quantization and I/O do not leave an H100 idle at every evaluation.
        final_stage = "posttrain" if config.posttraining.enabled else "pretrain"
        announce(
            "Final export from selected best weights: "
            + ", ".join(config.training.final_export_formats),
            context,
        )
        final_export_dir = export_final_model(
            model,
            config,
            context,
            run_root,
            stage=final_stage,
            step=final_step,
            authenticated_revision_directions=revision_directions,
            pipeline_identity=pipeline_identity,
        )
        announce(f"Final model export validation complete: {final_export_dir}", context)
    finally:
        try:
            cleanup_distributed(context)
        finally:
            run_scope.close()


if __name__ == "__main__":
    main()
