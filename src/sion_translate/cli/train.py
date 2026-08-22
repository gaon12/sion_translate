"""sion_translate 학습 실행 진입점(CLI) — 인자 없이 실행하는 완전 자동 파이프라인.

    sion-train            ← 이것만 실행하면 됩니다.

동작 순서:
    ① 실행 환경(GPU 수·VRAM·bf16·CPU) 자동 인식
    ② 설정 로드 — 프로젝트 루트의 ``sion_translate.yaml`` 이 있으면 그 값을 우선 적용
       (없거나 비워 두면 전부 자동). ``--config`` 로 다른 파일도 지정 가능.
    ③ ``data/*.jsonl`` 자동 인식 — 토크나이저가 없으면 학습하고,
       파일이 추가/변경되었으면 데이터셋을 자동으로 다시 준비
    ④ 데이터 규모에 맞춰 모델 크기·step 수·배치 등 수치 자동 결정
    ⑤ 이전 학습이 있으면 단계별 checkpoints/latest 에서 자동 재개
    ⑥ SFT 사전학습 후 pretrain/에 저장
    ⑦ 복합 보상 MRT + 다중 후보 선호학습 후 posttrain/에 별도 저장

각 단계가 시작될 때마다 "[sion] …" 텍스트가 출력되므로 현재 어디까지
진행됐는지 터미널에서 바로 확인할 수 있습니다.
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
                "rank 0이 training.output_dir run lock을 획득하지 못했습니다. "
                "rank 0의 오류에서 현재 보유자와 output_dir를 확인하십시오."
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
        "요청한 최종 export 포맷의 의존성이 없습니다: "
        f"{details}. 장기 학습을 시작하기 전에 "
        'python -m pip install -e ".[export]" 로 설치하거나 '
        "training.final_export_formats에서 해당 포맷을 제거하세요."
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

    language_pairs = config.data.configured_language_pairs()
    preprocessing_options = prepare_preprocessing_options(
        approximate_split=config.data.approximate_split,
        source_only_languages=config.data.configured_source_only_languages(),
        train_only_prefixes=config.data.configured_synthetic_prefixes(),
        synthetic_sampling_weight=config.data.synthetic_sampling_weight,
        language_pair_count=len(language_pairs),
    )
    return scan_raw_data(
        data_dir,
        language_pairs=language_pairs,
        tokenizer_model=tokenizer_path,
        preprocessing_options=preprocessing_options,
    )


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
            revision_trained=config.data.revision_examples,
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
) -> str | None:
    """Return a concrete compatibility problem for a tokenizer, if any."""

    tokenizer_path = Path(tokenizer_path)
    try:
        tokenizer = SionTokenizer(tokenizer_path)
        metadata = load_tokenizer_metadata(tokenizer_path)
        recorded_policy = tokenizer_split_digits_policy(tokenizer_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return f"토크나이저 정책/메타데이터를 읽을 수 없습니다: {exc}"

    if not tokenizer.splits_digits:
        return "토크나이저가 여러 자리 숫자를 한 자리씩 분리하지 않습니다 (split_digits=False 동작)"
    if recorded_policy is False:
        return "tokenizer_metadata.json에 split_digits=false가 기록되어 있습니다"
    if recorded_policy is None or metadata is None:
        return "버전 2 이상의 tokenizer_metadata.json이 없습니다"

    recorded_hash = metadata.get("model_sha256")
    if recorded_hash != file_sha256(tokenizer_path):
        return "tokenizer_metadata.json의 model_sha256이 실제 모델과 다릅니다"
    vocab_path = tokenizer_path.with_suffix(".vocab")
    if not vocab_path.is_file() or metadata.get("vocab_sha256") != file_sha256(vocab_path):
        return "tokenizer_metadata.json의 vocab_sha256이 실제 vocabulary와 다릅니다"
    raw_pairs = metadata.get("language_pairs")
    recorded_pairs = (
        tuple((str(pair[0]), str(pair[1])) for pair in raw_pairs)
        if isinstance(raw_pairs, list)
        and all(isinstance(pair, list) and len(pair) == 2 for pair in raw_pairs)
        else ()
    )
    if recorded_pairs != language_pairs:
        return (
            "tokenizer_metadata.json의 language_pairs가 현재 설정과 다릅니다 "
            f"(metadata={recorded_pairs}, config={language_pairs})"
        )
    expected_languages = {language for pair in language_pairs for language in pair}
    if set(tokenizer.languages) != expected_languages:
        return (
            "토크나이저 제어 토큰의 언어 집합이 현재 설정과 다릅니다 "
            f"(tokenizer={sorted(tokenizer.languages)}, config={sorted(expected_languages)})"
        )
    expected_denoise_languages = set(foundation_languages or expected_languages)
    if set(tokenizer.denoise_tags) != expected_denoise_languages:
        return (
            "토크나이저 복원 태그의 언어 집합이 foundation 설정과 다릅니다 "
            f"(tokenizer={sorted(tokenizer.denoise_tags)}, "
            f"config={sorted(expected_denoise_languages)})"
        )
    tokenizer_reasoning_languages = set(getattr(tokenizer, "reasoning_tags", {}))
    if tokenizer_reasoning_languages != set(reasoning_languages):
        return (
            "토크나이저 reasoning 태그의 언어 집합이 구조화 코퍼스와 다릅니다 "
            f"(tokenizer={sorted(tokenizer_reasoning_languages)}, "
            f"corpus={sorted(reasoning_languages)})"
        )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train sion_translate. 인자 없이 실행하면 환경/데이터를 자동 인식합니다."
    )
    parser.add_argument(
        "--config", help=f"설정 파일 (기본: 루트의 {DEFAULT_CONFIG_FILE}, 없으면 전부 자동)"
    )
    parser.add_argument("--epochs", type=int, help="SFT가 전체 학습 dataset을 완주할 횟수")
    parser.add_argument("--max-steps", type=int, help="최대 step 수동 지정 (자동값 무시)")
    parser.add_argument(
        "--posttrain-epochs", type=int, help="MRT가 전체 학습 dataset을 완주할 횟수"
    )
    parser.add_argument("--posttrain-steps", type=int, help="MRT 사후학습 step 수동 지정")
    parser.add_argument("--skip-posttraining", action="store_true", help="SFT 사전학습까지만 실행")
    parser.add_argument(
        "--resume-from", help="재개할 체크포인트 수동 지정 (기본: latest 자동 감지)"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="토크나이저와 dataset shard만 준비하고 학습 전에 종료",
    )
    return parser


def seed_everything(seed: int, rank: int) -> None:
    """재현 가능한 학습을 위해 모든 난수 생성기에 시드를 고정합니다.

    rank 를 더해 주는 이유: 분산 학습에서 rank 마다 dropout 등
    실행 시점 난수가 서로 달라야 하기 때문입니다.
    """
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32))
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def resolve_config(args: argparse.Namespace) -> tuple[AppConfig, dict[str, Any], str]:
    """설정 파일을 찾아 (config, raw dict, 출처 설명) 을 돌려줍니다.

    raw dict 는 '사용자가 어떤 키를 직접 적었는지'를 기억하는 용도입니다.
    자동 설정은 사용자가 적지 않은 키만 채웁니다.
    """
    if args.config:
        raw = load_raw_config(args.config)
        source = args.config
    elif Path(DEFAULT_CONFIG_FILE).exists():
        raw = load_raw_config(DEFAULT_CONFIG_FILE)
        source = DEFAULT_CONFIG_FILE
    else:
        raw = {}
        source = "내장 기본값 (전부 자동)"

    # 커맨드라인 인자는 파일보다 우선하며, '사용자 지정'으로 취급합니다.
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
    locks_held: bool = False,
) -> None:
    """토크나이저와 준비된 데이터셋이 없거나 낡았으면 자동으로 만듭니다.

    - 토크나이저: 없을 때만 학습합니다. 기존 vocabulary를 사용하는 다른 run을
      깨뜨릴 수 있으므로 내용 계약에 맞지 않는 산출물은 자동 이동하거나 덮어쓰지
      않고, 운영자가 관련 checkpoint를 확인할 수 있도록 구체적인 오류를 냅니다.
    - 데이터셋: ``data/`` 의 파일 이름+크기 지문을 기록해 두고, 지문이
      달라지면(파일 추가/변경) 기존 데이터셋을 옆으로 보관한 뒤 다시 만듭니다.

    이 내부 구현은 rank 0에서만 호출됩니다. 분산 peer 대기는 process-group
    collective가 아니라 ``ensure_artifacts``의 durable status channel이 맡습니다.
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
    # 같은 artifacts/ 를 쓰는 작업 두 개가 동시에 "없으니 만들자"고 판단하면
    # 서로 다른 세대의 토크나이저와 데이터셋이 같은 경로에 섞입니다. 실패가
    # 아니라 **섞인 상태**라 지문 검사는 그 조합을 처음 보는 것으로만 인식합니다.
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
                    f"원천 데이터({data_dir}/*.jsonl)도 준비된 데이터셋({dataset_dir})도 없습니다."
                )
            if not tokenizer_path.is_file() and dataset_ready and not files:
                raise FileNotFoundError(
                    f"준비된 데이터셋은 있지만 대응하는 토크나이저가 없습니다: {tokenizer_path}. "
                    "원천 데이터와 새 출력 경로를 지정해 새 run을 시작하세요."
                )
            if not files and dataset_ready:
                integrity_problem = dataset_artifact_problem(dataset_dir)
                if integrity_problem is not None:
                    raise RuntimeError(
                        "준비된 번역 데이터셋 payload가 손상됐지만 다시 만들 원천 "
                        f"데이터가 없습니다: {integrity_problem}"
                    )

            if files:
                cpu_plan = build_cpu_plan(input_files=len(files))
                announce(
                    f"원천 데이터 인식: {len(files)}개 파일, "
                    f"총 {sum(files.values()) / 2**30:.2f} GiB ({data_dir}/)",
                    context,
                )
                announce(
                    f"CPU 자동 배분: 할당 {cpu_plan.available}개 → "
                    f"입력 정제 {cpu_plan.preprocess_workers}개 + "
                    f"SentencePiece {cpu_plan.sentencepiece_threads}개; "
                    f"dataset 준비 {cpu_plan.dataset_workers}개",
                    context,
                )
                # ── 토크나이저 ────────────────────────────────────────────
                if not tokenizer_path.exists():
                    if existing_checkpoint is not None:
                        raise RuntimeError(
                            "기존 체크포인트가 있지만 대응하는 토크나이저가 없습니다. "
                            f"checkpoint={existing_checkpoint}. 기존 vocabulary를 추측해 "
                            "새 토크나이저로 덮어쓸 수 없습니다. tokenizer_model, dataset_dir, "
                            "training.output_dir을 새 경로로 지정해 새 run을 시작하세요."
                        )
                    from sion_translate.tokenizer import train_tokenizer

                    pair_estimate = estimate_pair_count(files, data_dir)
                    vocab_size = pick_vocab_size(pair_estimate)
                    announce(
                        f"토크나이저가 없어 새로 학습합니다 "
                        f"(약 {pair_estimate:,}행 → vocab {vocab_size:,}) — 시간이 걸립니다.",
                        context,
                    )
                    train_tokenizer(
                        [str(data_dir / "*.jsonl")],
                        tokenizer_path.parent,
                        vocab_size=vocab_size,
                        language_pairs=config.data.configured_language_pairs(),
                        # foundation 단계가 자기 코퍼스에 없는 어휘로 학습하지 않도록
                        # 단일어 코퍼스도 넣습니다. 언어별 상한이 없으면 분량이 큰
                        # 언어가 vocab 을 독식합니다.
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
                    announce("토크나이저 학습 완료.", context)
                    # 토크나이저 파일의 SHA-256도 데이터셋 지문에 포함됩니다.
                    files = scan_configured_raw_data(config, data_dir, tokenizer_path)

                # ── 데이터셋 (지문 기반 변경 감지) ─────────────────────────
                policy_problem = tokenizer_policy_problem(
                    tokenizer_path,
                    config.data.configured_language_pairs(),
                    config.foundation_languages(),
                    reasoning_languages,
                )
                if policy_problem is not None:
                    existing_tokenizer = SionTokenizer(tokenizer_path)
                    if (
                        existing_checkpoint is None
                        and existing_tokenizer.splits_digits
                        and load_tokenizer_metadata(tokenizer_path) is None
                    ):
                        write_tokenizer_metadata(
                            tokenizer_path,
                            split_digits=True,
                            language_pairs=config.data.configured_language_pairs(),
                        )
                        files = scan_configured_raw_data(config, data_dir, tokenizer_path)
                        policy_problem = tokenizer_policy_problem(
                            tokenizer_path,
                            config.data.configured_language_pairs(),
                            config.foundation_languages(),
                            reasoning_languages,
                        )
                    if policy_problem is not None:
                        checkpoint_detail = (
                            f" 기존 checkpoint={existing_checkpoint}와 vocabulary 호환성을 "
                            "깨뜨리는 자동 재학습은 수행하지 않습니다."
                            if existing_checkpoint is not None
                            else ""
                        )
                        raise RuntimeError(
                            f"{policy_problem}.{checkpoint_detail} tokenizer_model, dataset_dir, "
                            "training.output_dir을 함께 검토하세요. 새 학습이라면 관련 run이 "
                            "이 vocabulary를 쓰지 않는지 확인한 뒤 기존 tokenizer/dataset을 "
                            "별도 백업으로 옮기고 split_digits=True로 다시 준비하십시오."
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
                            f"indexed payload 무결성 실패 ({integrity_problem})"
                            if integrity_problem is not None
                            else (
                                "호환 가능한 지문 없음"
                                if stored is None
                                else "원천/토크나이저/전처리 변경"
                            )
                        )
                        announce(
                            f"{reason} 감지 → 기존 데이터셋을 {backup.name}/ 으로 보관합니다.",
                            context,
                        )
                    announce(
                        "데이터셋 준비 시작 (품질 필터 + 중복 제거 + 토큰화) — 시간이 걸립니다.",
                        context,
                    )
                    prepare_dataset(
                        [str(data_dir / "*.jsonl")],
                        tokenizer_path,
                        dataset_dir,
                        language_pairs=config.data.configured_language_pairs(),
                        source_only_languages=config.data.configured_source_only_languages(),
                        approximate_split=config.data.approximate_split,
                        train_only_prefixes=config.data.configured_synthetic_prefixes(),
                        synthetic_sampling_weight=config.data.synthetic_sampling_weight,
                        num_workers=cpu_plan.dataset_workers,
                        expected_fingerprint=files,
                    )
                    announce("데이터셋 준비 완료.", context)
                else:
                    announce("데이터셋 최신 상태 확인 (원천 데이터 변경 없음).", context)

                # ── foundation(단일어) 데이터셋 ──────────────────────────
                for line in foundation_plan.report:
                    announce(f"  {line}", context)
                if not foundation_plan.enabled:
                    announce(f"foundation 단계: {foundation_plan.reason}", context)
                elif not prepare_foundation:
                    announce(
                        "SFT resume 후보를 먼저 검증하므로 foundation 데이터셋 준비를 보류합니다.",
                        context,
                    )
                else:
                    for warning in foundation_plan.warnings:
                        announce(f"[경고] foundation: {warning}", context)
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
                        announce("foundation 데이터셋 최신 상태 확인.", context)
                    else:
                        if foundation_dataset.exists() or foundation_dataset.is_symlink():
                            backup = backup_stale_dataset(foundation_dataset)
                            announce(
                                f"{foundation_problem} → 기존 foundation 데이터셋을 "
                                f"{backup.name}/ 으로 보관합니다.",
                                context,
                            )

                        announce(
                            "foundation 데이터셋 준비 시작 (복원 + reasoning 토큰화) — "
                            "시간이 걸립니다.",
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
            locks_held=locks_held,
        ),
    )
    _verify_prepared_artifact_consensus(
        config,
        foundation_plan,
        context,
        prepare_foundation=prepare_foundation,
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
    foundation_plan: Any,
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
        "languages": list(foundation_plan.languages),
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
    foundation_plan: Any,
    checkpoint_identity: dict[str, Any],
) -> bool:
    return bool(
        marker.get("schema") == FOUNDATION_COMPLETION_SCHEMA
        and marker.get("stage") == "foundation"
        and marker.get("release_name") == config.foundation.release_name
        and marker.get("release_version") == MODEL_RELEASE_VERSION
        and marker.get("languages") == list(foundation_plan.languages)
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
    checkpoint_identity = _foundation_checkpoint_identity(foundation_config, context)
    run_root = foundation_run_directory(config)
    completion = run_root / FOUNDATION_COMPLETION_FILENAME
    marker = _read_foundation_completion(completion)
    if marker is None or not _foundation_marker_contract_matches(
        marker,
        config=config,
        foundation_plan=foundation_plan,
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
    """번역 학습 전에 복원과 선택적 reasoning으로 encoder-decoder를 만든다.

    끝난 단계를 다시 돌리지 않는 것이 중요합니다. 이 단계는 파이프라인에서
    가장 오래 걸리는 구간이라, 번역 학습이 실패해 다시 실행할 때마다 며칠짜리
    사전학습을 반복하면 안 됩니다. 완료 표시가 있으면 학습을 건너뛰고 best
    가중치만 물려받습니다.
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
            foundation_plan=foundation_plan,
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
            announce(f"{reason} → 이전 실행을 {backup.name}/ 으로 보관합니다.", context)
        archive_visibility_scope = distributed_failure_scope(run_root.exists(), context)
        if archive_visibility_scope != "none":
            raise RuntimeError(
                "archived foundation run remains visible on at least one distributed rank"
            )

    if stale_completed_run:
        archive_completed_run(
            "foundation 완료 세대의 입력·설정·checkpoint 구성이 현재 실행과 다릅니다"
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
                    "foundation 완료 checkpoint가 historical shard digest 없는 legacy "
                    "DCP입니다. 자동 archive/retrain 또는 현재 bytes의 자동 승격을 "
                    "거부합니다. 원본 백업을 확인한 뒤 명시적 offline recovery를 "
                    f"수행하세요: {error}"
                ) from error
            archive_completed_run(
                "foundation 완료 marker와 일치하는 exact checkpoint 세대를 모든 "
                f"rank에서 인증하지 못해 새로 학습합니다: {error}"
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
                "foundation 완료 checkpoint는 유효하지만 최종 base export가 없거나 "
                "손상됐거나 같은 가중치 세대로 증명되지 않아 재학습 없이 다시 내보냅니다.",
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
                languages=foundation_plan.languages,
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
            f"foundation 단계는 이미 완료됐습니다 → {checkpoint_source} 의 가중치를 "
            f"물려받습니다 (step {provenance['step']:,}).",
            context,
        )
        return FoundationOutcome(
            ran=False,
            reason="이미 완료된 foundation 단계의 가중치를 재사용했습니다.",
            best_checkpoint=str(checkpoint_source),
            selected_step=provenance["step"],
            languages=foundation_plan.languages,
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
            "foundation resume checkpoint는 historical shard digest가 없는 legacy DCP라 "
            "자동 archive/retrain 또는 현재 bytes의 자동 승격을 거부합니다. 원본 "
            "백업을 확인하고 명시적 offline recovery를 수행하세요: "
            f"{resume_plan.get('reason', 'unverifiable legacy generation')}"
        )
    has_resume = resume_state == "available"
    if resume_state == "invalid":
        archive_completed_run(
            "foundation 재개 checkpoint를 인증하지 못해 새로 학습합니다: "
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
                "foundation 재개 checkpoint의 세대·v2 inventory·identity를 인증하지 "
                f"못해 새로 학습합니다: {error}"
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
                "foundation latest가 content-bind하지 않은 completion/best/export를 "
                "재개 checkpoint와 분리해 "
                f"{quarantine.name}/ 에 보관합니다.",
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
        announce(f"foundation: 이전 실행 발견 → {resume_source} 에서 재개합니다.", context)
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
                "인증할 완료 marker나 재개 checkpoint가 없는 foundation 부분 실행을 "
                "새 학습과 분리합니다"
            )

    preflight_final_export_dependencies(config.foundation.final_export_formats)
    train_dataset = IndexedParallelDataset(
        foundation_config.data.dataset_dir,
        foundation_config.data.train_split,
        bidirectional=foundation_config.data.bidirectional,
        verify_integrity=not artifacts_verified,
    )
    validation_dataset = IndexedParallelDataset(
        foundation_config.data.dataset_dir,
        foundation_config.data.validation_split,
        bidirectional=foundation_config.data.bidirectional,
        verify_integrity=not artifacts_verified,
    )
    announce(
        f"foundation 데이터 규모: 학습 {len(train_dataset):,}개 / "
        f"검증 {len(validation_dataset):,}개 (언어: {', '.join(foundation_plan.languages)})",
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

    announce("0단계 foundation 사전학습(복원 + 선택적 reasoning)을 시작합니다.", context)
    result = train(
        model,
        train_loader,
        validation_loader,
        foundation_config,
        context,
        stage_name="foundation/denoising",
        export_release_name=config.foundation.release_name,
        export_translation_capable=False,
        export_languages=foundation_plan.languages,
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
        # 번역 모델이 아니라 그 파운데이션입니다. 이름을 나누는 것만으로는
        # 부족합니다 — 구조가 번역 모델과 같아서 그대로 실으면 방향 태그를
        # 받아들이고 그럴듯한 쓰레기를 냅니다.
        release_name=config.foundation.release_name,
        translation_capable=False,
        languages=foundation_plan.languages,
    )
    _run_rank_zero_action(
        context,
        lambda: _atomic_write_foundation_completion(
            completion,
            _foundation_completion_marker(
                config=config,
                foundation_plan=foundation_plan,
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
        languages=foundation_plan.languages,
        warnings=foundation_plan.warnings,
    )


def find_auto_resume(config: AppConfig) -> str | None:
    """이전 학습의 latest 체크포인트가 있으면 그 경로를 돌려줍니다."""
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
            f"{stage} 자동 재개 후보를 인증하지 못해 {backup.name}/ 으로 보관하고 "
            f"fresh 경로로 돌아갑니다: {_bounded_status_text(error)}",
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
        lineage_plan = cast(
            FoundationPlan,
            SimpleNamespace(
                enabled=True,
                languages=config.foundation_languages(),
            ),
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
            "foundation은 설정상 활성화되어 있으나 원천 코퍼스가 현재 보이지 않습니다. "
            "인증된 downstream checkpoint 또는 준비된 base 세대만 재사용합니다."
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
    for binding in bindings:
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
            attempt_scope.close()
            continue
        lease_scope.enter_context(attempt_scope.pop_all())
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
            reason="검증된 SFT resume가 foundation 실행/로드보다 우선합니다.",
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
        branch = pipeline_branch or "검증된"
        return (
            f"{branch} SFT 체크포인트 {resume_from} 를 우선 재개합니다. foundation 단계는 "
            "이번 실행에서 학습하거나 로드하지 않으며 최종 release는 sion_translate입니다."
        )
    if foundation_outcome.best_checkpoint:
        return (
            "번역 학습을 foundation 가중치에서 시작합니다 "
            f"({foundation_release_name} step {foundation_outcome.selected_step:,})."
        )
    return (
        "foundation 모델(sion)은 학습·내보내지 않습니다. fresh initialization에서 "
        "번역 SFT/MRT로 바로 진행해 sion_translate만 만듭니다."
    )


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    context = initialize_distributed()
    run_scope = ExitStack()
    try:
        # ── 단계 ①: 환경 자동 인식 ──────────────────────────────────────
        env = probe_environment()
        env = synchronize_environment(env, context)
        announce(f"준비 ①: 실행 환경 — {describe_environment(env)}", context)

        # ── 단계 ②: 설정 로드 ───────────────────────────────────────────
        config, raw, source = resolve_config(args)
        run_scope.enter_context(coordinated_training_run_lock(config.training.output_dir, context))
        checkpoint_lease_scope = ExitStack()
        run_scope.enter_context(checkpoint_lease_scope)
        announce(f"준비 ②: 설정 로드 — {source}", context)
        if not args.prepare_only:
            # The built-in collator has no dense alignment-label provider.
            # Reject a permanently-zero BATS alignment objective before doing
            # artifact preparation or allocating model parameters.
            config.validate_training_supervision(alignment_targets_available=False)

        # ── 단계 ③: 원천 데이터 인식 + 토크나이저/데이터셋 자동 준비 ──
        announce("준비 ③: 원천 데이터를 확인합니다.", context)
        discovered_foundation_plan = plan_foundation_stage(config)
        foundation_plan = _configured_foundation_branch_plan(
            config,
            discovered_foundation_plan,
        )
        if not args.prepare_only:
            # Translation formats are unavoidable. Foundation-only converters
            # are checked later, after downstream resume candidates have had a
            # chance to supersede every base-stage action.
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
        ensure_artifacts(
            config,
            context,
            discovered_foundation_plan,
            # Effective auto-sized training identity and the exact SFT resume
            # generation must be known before deciding whether base preparation
            # can safely be skipped. A coarse `.metadata` presence check is not
            # authority to bypass the foundation path.
            prepare_foundation=args.prepare_only,
            locks_held=True,
        )
        tokenizer = SionTokenizer(config.data.tokenizer_model)
        config.model.vocab_size = len(tokenizer)
        preflight_morphoscript_token_features(config, tokenizer)
        if args.prepare_only:
            announce("전처리 전용 실행 완료.", context)
            return

        # 모델 파라미터 초기화는 world size 와 무관하게 같은 시드(rank 0 기준)로
        # 수행합니다. 실행 시점 난수는 모델 생성 후에 rank 별로 다시 시드합니다.
        seed_everything(config.training.seed, 0)

        train_dataset = IndexedParallelDataset(
            config.data.dataset_dir,
            config.data.train_split,
            bidirectional=config.data.bidirectional,
            verify_integrity=False,
        )
        validation_dataset = IndexedParallelDataset(
            config.data.dataset_dir,
            config.data.validation_split,
            bidirectional=config.data.bidirectional,
            verify_integrity=False,
        )
        revision_sources = [
            source
            for source in train_dataset.source_names
            if Path(source).name.startswith("revise_")
        ]
        if revision_sources and not config.data.revision_examples:
            config.data.revision_examples = True
            announce(
                "revision 예제 원천을 자동 감지했습니다: "
                + ", ".join(revision_sources[:3])
                + (" …" if len(revision_sources) > 3 else ""),
                context,
            )
        announce(
            f"데이터 규모: 학습 {len(train_dataset):,}개 / 검증 {len(validation_dataset):,}개 "
            f"(양방향 포함)",
            context,
        )

        # ── 단계 ④: 데이터 규모·환경 기반 자동 수치 결정 ────────────────
        decisions = apply_auto_settings(
            config,
            raw,
            env,
            train_examples=len(train_dataset),
            validation_examples=len(validation_dataset),
            source_names=train_dataset.source_names,
        )
        if decisions:
            announce("준비 ④: 자동 결정된 설정 —", context)
            for line in decisions:
                announce(f"  · {line}", context)
        config.validate()

        if not foundation_plan.enabled:
            pipeline_identity = build_translation_pipeline_identity(foundation_plan)

        # 실행 루트 아래에 사전학습/사후학습 산출물을 서로 분리합니다.
        run_root = Path(config.training.output_dir)
        pretrain_config = copy.deepcopy(config)
        pretrain_config.training.output_dir = str(run_root / "pretrain")
        post_config = (
            _build_posttraining_config(config, run_root) if config.posttraining.enabled else None
        )

        # ── 단계 ⑤: 이전 사전학습 자동 재개 ────────────────────────────
        if not pretrain_config.training.resume_from and pretrain_resume_candidate:
            pretrain_config.training.resume_from = pretrain_resume_candidate
        if pretrain_config.training.resume_from:
            announce(
                f"준비 ⑤: 이전 SFT 체크포인트 후보 발견 → {pretrain_config.training.resume_from}",
                context,
            )
        if posttrain_resume_candidate:
            announce(
                f"준비 ⑤: 이전 MRT 체크포인트 후보 발견 → {posttrain_resume_candidate}",
                context,
            )

        # ── DataLoader 구성 ──────────────────────────────────────────────
        # collator: 원문/번역문을 토큰화하고 패딩해 텐서 배치로 만듭니다.
        collator_args = build_collator_args(config, tokenizer)
        train_collator = SionBatchCollator(
            **collator_args,
            denoise_probability=config.data.denoise_probability,
            # 온라인 증강(원문 토큰 dropout)은 학습에만 적용합니다.
            source_token_dropout=config.data.source_token_dropout,
            decoder_input_noise=config.data.decoder_input_noise,
        )
        validation_collator = SionBatchCollator(
            **collator_args,
            denoise_probability=config.data.validation_denoise_probability,
            source_token_dropout=0.0,  # 검증은 항상 깨끗한 입력으로
            decoder_input_noise=0.0,
        )
        # sampler: 비슷한 길이끼리 묶어(bucket) 패딩 낭비를 줄이고,
        # 분산 학습에서 rank 별로 겹치지 않게 배치를 나눕니다.
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
        # ── 모델 생성과 분산 배치 ────────────────────────────────────────
        announce("모델을 생성하고 장치에 배치합니다.", context)
        # 모든 CUDA 전략은 meta device에서 파라미터 수와 영구 상태 용량을 먼저
        # 검사합니다. 통과한 뒤에만 single/DDP는 전체 모델을, FSDP2는 shard를
        # 실제 GPU에 할당하므로 과대 구성도 constructor OOM보다 명확히 실패합니다.
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
        # SFT와 MRT가 같은 DDP wrapper를 공유하므로 단계 전환 뒤의 파라미터
        # 사용 집합까지 고려해 unused-parameter 탐지 여부를 정합니다.
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
        # 여기서부터의 난수(dropout, denoising 등)는 rank 별로 달라야 합니다.
        seed_everything(config.training.seed, context.rank)
        announce(
            f"모델 파라미터 수: {parameter_count:,}; 병렬 전략: {parallel_strategy}",
            context,
        )
        if capacity is not None:
            announce(
                "영구 학습 상태 추정: "
                f"rank당 {capacity['per_rank_state_gib']:.1f} GiB / "
                f"안전 예산 {capacity['state_budget_gib']:.1f} GiB",
                context,
            )

        # ── 단계 ⑤: downstream-first 체크포인트 선택 ──────────────────
        # 가장 진행된 단계를 먼저 복구해야 이미 끝난 foundation/SFT를 다시
        # 실행하지 않습니다. 각 논리 체크포인트의 current와 previous는 서로
        # 독립된 lease에서 검증하고, 완전히 검증된 하나만 실제 load까지 유지합니다.
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
                    "준비 ⑤: MRT resume 전체 검증 완료 — foundation과 SFT 실행/로드를 건너뜁니다.",
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
                    "준비 ⑤: SFT resume 전체 검증 완료 — foundation 실행/로드를 건너뜁니다.",
                    context,
                )

        # ── 단계 ⑤-b: foundation 사전학습 (복원 + reasoning) ───────────
        # 번역쌍을 보기 전에 encoder-decoder 를 먼저 만듭니다. 이 단계의
        # 산출물은 번역 모델이 아니라 그 파운데이션이라 별도 이름으로 나갑니다.
        if validated_posttrain_resume or validated_pretrain_resume:
            foundation_outcome = FoundationOutcome(
                ran=False,
                reason="검증된 downstream resume가 foundation 실행/로드보다 우선합니다.",
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
                    "foundation 원천 코퍼스가 오프라인이므로 인증된 준비 데이터와 "
                    "checkpoint만 사용합니다.",
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
            pipeline_identity = build_translation_pipeline_identity(
                foundation_plan,
                foundation_lineage=foundation_lineage,
            )
        if validated_posttrain_resume:
            assert post_config is not None
            announce(
                f"검증된 MRT 체크포인트 {post_config.training.resume_from} 를 직접 재개합니다. "
                "foundation과 SFT는 이번 실행에서 학습하거나 로드하지 않습니다.",
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

        # ── 단계 ⑥: SFT 사전학습 ───────────────────────────────────────
        pretrain_result: dict[str, float | int | bool | str] | None = None
        if validated_posttrain_resume:
            announce(
                "검증된 MRT resume가 있으므로 SFT DataLoader 생성과 학습을 건너뜁니다.",
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
            announce("1단계 SFT 사전학습을 시작합니다.", context)
            pretrain_result = train(
                model,
                train_loader,
                validation_loader,
                pretrain_config,
                context,
                stage_name="pretrain/SFT",
                language_tags=tokenizer.language_tags,
                pipeline_identity=pipeline_identity,
            )
            barrier(context)
            memory = release_stage_resources(context, train_loader, validation_loader)
            del train_loader, validation_loader
            if memory:
                announce(
                    "사전학습 메모리 정리: "
                    f"allocated {memory['before_allocated_gib']:.2f}→"
                    f"{memory['after_allocated_gib']:.2f} GiB, "
                    f"reserved {memory['before_reserved_gib']:.2f}→"
                    f"{memory['after_reserved_gib']:.2f} GiB",
                    context,
                )
        del train_sampler, validation_sampler
        del train_collator, validation_collator

        # ── 단계 ⑦: MRT 사후학습 ───────────────────────────────────────
        if post_config is not None:
            post = config.posttraining
            assert post_sampler is not None
            assert post_validation_sampler is not None

            # 보상 계산은 깨끗한 원문/정답을 기준으로 해야 하므로 증강을 끕니다.
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
                f"2단계 복합 MRT/선호 사후학습을 시작합니다: "
                f"후보 {post.samples_per_source}개, risk {post.risk_weight:.2f}, "
                f"preference {post.preference_weight:.2f}, "
                f"검증 beam {post.validation_num_beams}",
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
                    "사후학습 메모리 정리: "
                    f"allocated {memory['before_allocated_gib']:.2f}→"
                    f"{memory['after_allocated_gib']:.2f} GiB, "
                    f"reserved {memory['before_reserved_gib']:.2f}→"
                    f"{memory['after_reserved_gib']:.2f} GiB",
                    context,
                )
            final_step = int(posttrain_result["selected_step"])
        else:
            announce("posttraining.enabled=false — 사후학습을 건너뜁니다.", context)
            assert pretrain_result is not None
            final_step = int(pretrain_result["selected_step"])

        # 중간 best/latest에서는 학습 재개와 빠른 확인에 필요한 경량 포맷만
        # 저장합니다. 모든 학습 단계가 끝난 지금 선택된 best 가중치에서 7종을
        # 한 번만 생성해, 매 평가 때 대형 CPU 양자화/I/O로 H100을 세우지 않습니다.
        final_stage = "posttrain" if config.posttraining.enabled else "pretrain"
        announce(
            "선택된 best 가중치 최종 내보내기: " + ", ".join(config.training.final_export_formats),
            context,
        )
        final_export_dir = export_final_model(
            model,
            config,
            context,
            run_root,
            stage=final_stage,
            step=final_step,
            pipeline_identity=pipeline_identity,
        )
        announce(f"최종 모델 내보내기 검증 완료: {final_export_dir}", context)
    finally:
        try:
            cleanup_distributed(context)
        finally:
            run_scope.close()


if __name__ == "__main__":
    main()
