"""Environment-aware defaults for zero-argument training.

The automatic pipeline detects available accelerators, memory, BF16 support,
and CPU capacity; fingerprints raw JSONL inputs; prepares the tokenizer and
indexed datasets when inputs change; and chooses model, batch, and schedule
settings from the resulting data scale.

Explicit values in ``sion_translate.yaml`` always take precedence. The project
configuration can therefore remain a small, intentional override file.
"""

# CUDA device properties and YAML payloads are dynamically typed boundaries.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
import math
import os
import platform as platform_module
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from sion_translate.config import AppConfig
from sion_translate.fingerprint import (
    PREPROCESSING_SCHEMA,
    DatasetFingerprint,
    build_dataset_fingerprint,
)
from sion_translate.performance import available_cpu_count

# Environment discovery


@dataclass(frozen=True)
class EnvironmentInfo:
    """Hardware and process properties used to choose training settings."""

    cuda: bool  # Whether a CUDA accelerator is available.
    world_size: int  # Number of processes launched by torchrun; one otherwise.
    device_count: int  # Number of GPUs visible on this machine.
    device_name: str  # Representative GPU name, or "CPU".
    min_vram_gib: float  # Memory of the smallest GPU, used for safe batch sizing.
    bf16: bool  # Whether every visible accelerator supports native BF16.
    cpu_count: int  # Available logical CPU count.
    os_name: str  # "Windows", "Linux", or "Darwin".


def _all_devices_support_native_bf16(properties: Sequence[Any]) -> bool:
    """Use BF16 only when every visible accelerator supports it natively."""

    if not properties:
        return False
    if torch.version.hip is not None:
        return True
    return all(int(getattr(device, "major", 0)) >= 8 for device in properties)


def probe_environment() -> EnvironmentInfo:
    """Inspect the local hardware and distributed process environment."""
    cuda = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if cuda:
        properties = [torch.cuda.get_device_properties(index) for index in range(device_count)]
        min_vram_gib = min(p.total_memory for p in properties) / (1024**3)
        device_names = tuple(dict.fromkeys(str(device.name) for device in properties))
        device_name = (
            device_names[0] if len(device_names) == 1 else "mixed: " + " / ".join(device_names)
        )
        bf16 = _all_devices_support_native_bf16(properties)
    else:
        min_vram_gib = 0.0
        device_name = "CPU"
        bf16 = False
    return EnvironmentInfo(
        cuda=cuda,
        world_size=world_size,
        device_count=device_count,
        device_name=device_name,
        min_vram_gib=min_vram_gib,
        bf16=bf16,
        cpu_count=available_cpu_count(),
        os_name=platform_module.system(),
    )


def synchronize_environment(
    env: EnvironmentInfo,
    context: Any,
) -> EnvironmentInfo:
    """Use the least-capable rank for settings shared by a distributed job."""

    if not context.distributed or not env.cuda:
        return env
    minimums = torch.tensor(
        [env.min_vram_gib, float(env.bf16)],
        device=context.device,
        dtype=torch.float64,
    )
    torch.distributed.all_reduce(minimums, op=torch.distributed.ReduceOp.MIN)
    return replace(
        env,
        min_vram_gib=float(minimums[0].item()),
        bf16=bool(minimums[1].item()),
    )


def describe_environment(env: EnvironmentInfo) -> str:
    """Return a concise, human-readable environment summary."""
    if env.cuda:
        return (
            f"{env.device_count} GPU(s) ({env.device_name}, minimum "
            f"{env.min_vram_gib:.0f} GiB), {env.world_size} process(es), "
            f"BF16 {'supported' if env.bf16 else 'unsupported'}, {env.cpu_count} CPU(s)"
        )
    return f"No GPU ({env.cpu_count} CPU(s), {env.os_name})"


# Raw-data discovery and change detection

FINGERPRINT_FILENAME = "raw_fingerprint.json"


def scan_raw_data(
    data_dir: str | Path,
    *,
    language_pairs: Sequence[Sequence[str]] = (),
    tokenizer_model: str | Path | None = None,
    preprocessing_schema: str = PREPROCESSING_SCHEMA,
    preprocessing_options: Mapping[str, Any] | None = None,
) -> DatasetFingerprint:
    """Build a content-addressed fingerprint for every raw JSONL input.

    The return value still behaves as ``Mapping[str, int]`` for legacy callers:
    iteration yields file names and values are byte sizes. Equality additionally
    covers file SHA-256, language pairs, tokenizer SHA-256, preprocessing schema,
    and normalized preprocessing options.
    """
    data_dir = Path(data_dir)
    return build_dataset_fingerprint(
        sorted(data_dir.glob("*.jsonl")),
        language_pairs=language_pairs,
        tokenizer_model=tokenizer_model,
        preprocessing_schema=preprocessing_schema,
        preprocessing_options=preprocessing_options,
    )


def stored_fingerprint(
    dataset_dir: str | Path,
) -> DatasetFingerprint | dict[str, int] | None:
    """Read the fingerprint from a previous preparation, if one exists."""
    path = Path(dataset_dir) / FINGERPRINT_FILENAME
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict) and value.get("schema"):
        try:
            return DatasetFingerprint.from_dict(value)
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, dict):
        # v1 fingerprints only tracked byte sizes. Returning the legacy mapping
        # makes it compare unequal to a v2 DatasetFingerprint and forces one
        # safe rebuild.
        try:
            return {str(key): int(size) for key, size in value.items()}
        except (TypeError, ValueError):
            return None
    return None


def write_fingerprint(
    dataset_dir: str | Path,
    fingerprint: DatasetFingerprint | Mapping[str, int],
) -> None:
    """Record a completed dataset generation's raw-data fingerprint."""
    payload = (
        fingerprint.to_dict() if isinstance(fingerprint, DatasetFingerprint) else dict(fingerprint)
    )
    path = Path(dataset_dir) / FINGERPRINT_FILENAME
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def backup_stale_dataset(dataset_dir: str | Path) -> Path:
    """Move an incompatible prepared dataset to a recoverable sibling path.

    The operation retains the old generation instead of deleting it, so an
    operator can restore it manually when needed.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists() and not dataset_dir.is_symlink():
        raise FileNotFoundError(dataset_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    while True:
        backup = dataset_dir.with_name(
            f"{dataset_dir.name}.stale-{timestamp}-{uuid.uuid4().hex[:12]}"
        )
        if not backup.exists() and not backup.is_symlink():
            break
    # The UUID makes the destination unoccupied, so shutil.move cannot interpret
    # it as an existing directory and nest the source inside it. Keep shutil's
    # copy/remove fallback for Windows paths that temporarily reject rename while
    # a just-closed training artifact is still releasing an OS handle.
    shutil.move(str(dataset_dir), str(backup))
    return backup


def estimate_pair_count(files: Mapping[str, int], data_dir: str | Path) -> int:
    """Count raw JSONL rows as a fast approximation of parallel pair count.

    This runs once when selecting the tokenizer vocabulary size. Counting
    newline bytes in large blocks avoids parsing multi-gigabyte files.
    """
    data_dir = Path(data_dir)
    total = 0
    for name in files:
        with (data_dir / name).open("rb") as handle:
            while chunk := handle.read(1 << 22):  # Read 4 MiB per iteration.
                total += chunk.count(b"\n")
    return total


# Data- and environment-aware defaults

# Nominal boundaries describe the original data-fit policy. A promotion buffer
# keeps small corpus changes from doubling model capacity at a sharp boundary.
# The final preset is intentionally unbounded.
MODEL_PROMOTION_BUFFER_PERCENT = 5
MODEL_PROMOTION_BUFFER_RATIO = MODEL_PROMOTION_BUFFER_PERCENT / 100
MODEL_SIZE_KEYS = (
    "d_model",
    "encoder_layers",
    "decoder_layers",
    "num_heads",
    "num_kv_heads",
    "d_ff",
)
MODEL_PRESETS: tuple[tuple[int | None, str, dict[str, int]], ...] = (
    # (nominal upper boundary, stable preset name, model settings)
    (
        200_000,
        "small",
        dict(
            d_model=512,
            encoder_layers=8,
            decoder_layers=4,
            num_heads=8,
            num_kv_heads=2,
            d_ff=1536,
        ),
    ),
    (
        3_000_000,
        "medium",
        dict(
            d_model=640,
            encoder_layers=12,
            decoder_layers=6,
            num_heads=10,
            num_kv_heads=2,
            d_ff=1792,
        ),
    ),
    (
        30_000_000,
        "base",
        dict(
            d_model=768,
            encoder_layers=16,
            decoder_layers=8,
            num_heads=12,
            num_kv_heads=4,
            d_ff=2048,
        ),
    ),
    (
        100_000_000,
        "large",
        dict(
            d_model=1024,
            encoder_layers=20,
            decoder_layers=10,
            num_heads=16,
            num_kv_heads=4,
            d_ff=2816,
        ),
    ),
    (
        None,
        "xlarge",
        dict(
            d_model=1280,
            encoder_layers=24,
            decoder_layers=12,
            num_heads=20,
            num_kv_heads=4,
            d_ff=3584,
        ),
    ),
)

# Target sequences per optimizer update: batch times ranks times accumulation.
TARGET_EFFECTIVE_BATCH = 256


def target_epochs(pair_count: int) -> int:
    """Choose complete corpus passes without allowing the step budget to explode.

    Larger datasets expose new examples on nearly every update and therefore
    need fewer passes. Early stopping can still end an over-provisioned budget.
    """
    if pair_count < 500_000:
        return 8
    if pair_count < 5_000_000:
        return 5
    if pair_count < 30_000_000:
        return 3
    if pair_count < 100_000_000:
        return 2
    # Complete at least two shuffled passes, even for very large corpora, so the
    # second pass can counter initialization and first-order ordering effects.
    return 2


def pick_vocab_size(pair_estimate: int) -> int:
    """Choose a SentencePiece vocabulary size from the corpus scale."""
    if pair_estimate < 200_000:
        return 16_000
    if pair_estimate < 3_000_000:
        return 32_000
    if pair_estimate < 100_000_000:
        return 48_000
    return 64_000


def buffered_model_promotion_threshold(nominal_threshold: int) -> int:
    """Return the first pair count that promotes beyond a nominal boundary."""

    if type(nominal_threshold) is not int:
        raise TypeError("nominal_threshold must be an integer")
    if nominal_threshold <= 0:
        raise ValueError("nominal_threshold must be positive")
    return (nominal_threshold * (100 + MODEL_PROMOTION_BUFFER_PERCENT) + 99) // 100


def _validated_pair_count(pair_count: int) -> int:
    if type(pair_count) is not int:
        raise TypeError("pair_count must be a non-negative integer")
    if pair_count < 0:
        raise ValueError("pair_count must be non-negative")
    return pair_count


def _selected_model_preset_index(pair_count: int) -> int:
    count = _validated_pair_count(pair_count)
    for index, (nominal_threshold, _name, _preset) in enumerate(MODEL_PRESETS):
        if nominal_threshold is None:
            return index
        if count < buffered_model_promotion_threshold(nominal_threshold):
            return index
    raise AssertionError("MODEL_PRESETS must end with an unbounded preset")


def pick_model_preset(pair_count: int) -> tuple[str, dict[str, int]]:
    """Select a deterministic preset after applying the promotion buffer."""

    _threshold, name, preset = MODEL_PRESETS[_selected_model_preset_index(pair_count)]
    return name, dict(preset)


def pick_batch_size(env: EnvironmentInfo, d_model: int) -> int:
    """Choose a per-GPU batch size from memory and model width.

    Values are conservative for 512-token sequences with activation
    checkpointing. Explicit ``training.batch_size_per_gpu`` remains available
    when a workload needs more headroom.
    """
    if not env.cuda:
        return 2  # CPU execution is intended for smoke tests.
    vram = env.min_vram_gib
    if vram >= 70:
        # 80 GiB-class cards run the baseline without checkpointing by default.
        # Keep headroom for rare 512-token buckets instead of selecting 64 from
        # short-sentence averages and failing late in the run.
        base = 32
    elif vram >= 40:
        base = 16
    elif vram >= 22:
        base = 8
    elif vram >= 14:
        base = 4
    elif vram >= 10:
        base = 2
    else:
        base = 1
    # Reduce the batch above the 768-wide base preset.
    if d_model > 1024:
        base = max(1, base // 4)
    elif d_model > 768:
        base = max(1, base // 2)
    return base


def pick_parallel_strategy(env: EnvironmentInfo, d_model: int) -> str:
    """Prefer lower-overhead DDP whenever one GPU has enough training memory."""

    if env.world_size <= 1:
        return "auto"
    if env.min_vram_gib >= 70 and d_model <= 1024:
        return "ddp"
    if env.min_vram_gib >= 40 and d_model <= 768:
        return "ddp"
    return "fsdp2"


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def apply_auto_settings(
    config: AppConfig,
    raw: dict[str, Any],
    env: EnvironmentInfo,
    *,
    train_examples: int,
    validation_examples: int,
    physical_train_pairs: int | None = None,
    source_names: list[str] | None = None,
) -> list[str]:
    """Fill settings that the user did not specify in ``sion_translate.yaml``.

    ``raw`` retains key presence from the source configuration so explicit
    values are never overwritten. The return value contains readable summaries
    of every automatic decision.
    """
    raw_model = dict(raw.get("model") or {})
    raw_training = dict(raw.get("training") or {})
    raw_data = dict(raw.get("data") or {})
    decisions: list[str] = []

    def auto(section: dict[str, Any], key: str) -> bool:
        """Return whether a key is eligible for automatic configuration."""
        return key not in section

    pair_count = _validated_pair_count(
        physical_train_pairs
        if physical_train_pairs is not None
        else train_examples // (2 if config.data.bidirectional else 1)
    )

    # Model capacity follows physical pair count, not expanded virtual directions.
    explicit_size_keys = tuple(key for key in MODEL_SIZE_KEYS if key in raw_model)
    if explicit_size_keys and len(explicit_size_keys) != len(MODEL_SIZE_KEYS):
        missing = tuple(key for key in MODEL_SIZE_KEYS if key not in raw_model)
        raise ValueError(
            "Manual model architecture overrides must provide every preset-defining key. "
            f"Provided: {explicit_size_keys}; missing: {missing}."
        )
    if not explicit_size_keys:
        name, preset = pick_model_preset(pair_count)
        for key, value in preset.items():
            setattr(config.model, key, value)
        preset_index = _selected_model_preset_index(pair_count)
        nominal_next = MODEL_PRESETS[preset_index][0]
        next_note = (
            f"; next promotion at {buffered_model_promotion_threshold(nominal_next):,}"
            if nominal_next is not None
            else "; final unbounded tier"
        )
        decisions.append(
            f"Model preset: {name} for {pair_count:,} physical training pairs "
            f"({MODEL_PROMOTION_BUFFER_RATIO:.0%} promotion buffer{next_note})"
        )
    if auto(raw_model, "gradient_checkpointing"):
        config.model.gradient_checkpointing = env.cuda and env.min_vram_gib < 70
        if config.model.gradient_checkpointing:
            decisions.append("Activation checkpointing: enabled to protect GPUs below 70 GiB")

    # Precision preference: BF16, then FP16, then FP32.
    if auto(raw_training, "precision"):
        config.training.precision = "bf16" if env.bf16 else ("fp16" if env.cuda else "fp32")
        decisions.append(f"Precision: {config.training.precision}")

    # Batch and accumulation target a stable effective batch across hardware.
    if auto(raw_training, "batch_size_per_gpu"):
        config.training.batch_size_per_gpu = pick_batch_size(env, config.model.d_model)
        basis = f"{env.min_vram_gib:.0f} GiB VRAM" if env.cuda else "CPU smoke-test policy"
        decisions.append(f"Per-device batch: {config.training.batch_size_per_gpu} ({basis})")
    if auto(raw_training, "gradient_accumulation_steps"):
        per_update = config.training.batch_size_per_gpu * env.world_size
        config.training.gradient_accumulation_steps = max(
            1, round(TARGET_EFFECTIVE_BATCH / per_update)
        )
        effective = per_update * config.training.gradient_accumulation_steps
        decisions.append(
            f"accumulation: {config.training.gradient_accumulation_steps} "
            f"(effective batch {effective} sequences/update)"
        )

    # Use complete epochs instead of truncating the corpus at an arbitrary step.
    batches_per_epoch = math.ceil(
        train_examples / max(1, config.training.batch_size_per_gpu * env.world_size)
    )
    steps_per_epoch = math.ceil(batches_per_epoch / config.training.gradient_accumulation_steps)
    epochs = target_epochs(pair_count)
    if auto(raw_training, "num_train_epochs") and auto(raw_training, "max_steps"):
        config.training.num_train_epochs = epochs
        config.training.max_steps = None
        decisions.append(
            f"num_train_epochs: {epochs} "
            f"(about {steps_per_epoch:,} optimizer steps per complete corpus pass)"
        )
    planned_steps = config.training.max_steps or (
        config.training.num_train_epochs * steps_per_epoch
    )
    if auto(raw_training, "warmup_steps"):
        config.training.warmup_steps = _clamp(int(0.025 * planned_steps), 10, 4000)
        config.training.warmup_steps = min(config.training.warmup_steps, planned_steps)
        decisions.append(f"warmup_steps: {config.training.warmup_steps:,}")

    # Scale validation and checkpoint cadence with epoch length.
    if auto(raw_training, "eval_every"):
        config.training.eval_every = _clamp(steps_per_epoch // 8, 50, 2500)
        decisions.append(f"eval_every: {config.training.eval_every:,}")
    if auto(raw_training, "save_every"):
        config.training.save_every = config.training.eval_every * 2
        decisions.append(f"save_every: {config.training.save_every:,}")
    if auto(raw_training, "eval_batches"):
        per_rank = config.training.batch_size_per_gpu * env.world_size
        needed = math.ceil(validation_examples / max(1, per_rank))
        config.training.eval_batches = _clamp(needed, 8, 200)
        decisions.append(f"eval_batches: {config.training.eval_batches}")

    # Execution strategy.
    # ``parallel_strategy: auto`` is an explicit request for the environment
    # picker, not a request to leave the generic DDP fallback unresolved.
    if config.training.parallel_strategy.lower() == "auto" and auto(raw_training, "fsdp2"):
        config.training.parallel_strategy = pick_parallel_strategy(
            env,
            config.model.d_model,
        )
        if env.world_size > 1:
            decisions.append(f"Multi-GPU strategy: {config.training.parallel_strategy.upper()}")
    if auto(raw_training, "fsdp_reduce_dtype"):
        config.training.fsdp_reduce_dtype = "bf16" if env.bf16 else "fp32"
    if auto(raw_training, "reshard_after_forward"):
        # Keep FSDP2's memory-bounded default. Disabling resharding retains
        # full parameters after every forward and can erase sharding's VRAM
        # benefit even on an 80 GiB H100; users may still opt out explicitly.
        config.training.reshard_after_forward = True
    if auto(raw_training, "compile"):
        # Compiler/backend support varies across CUDA architectures and container
        # builds. Reliability-first automatic runs stay eager; measured profiles
        # can still opt in with ``training.compile: true``.
        config.training.compile = False
    if auto(raw_data, "num_workers"):
        per_rank = max(1, env.cpu_count // max(1, env.world_size))
        config.data.num_workers = min(16, max(0, per_rank - 1))
        decisions.append(f"DataLoader workers: {config.data.num_workers}")

    # Downweight synthetic sources unless the user supplies an explicit policy.
    # Equal weighting can amplify systematic errors from a model's own output.
    if auto(raw_data, "source_sampling_weights") and source_names:
        prefixes = config.data.configured_synthetic_prefixes()
        synthetic = [name for name in source_names if name.startswith(prefixes)]
        if synthetic:
            weight = config.data.synthetic_sampling_weight
            config.data.source_sampling_weights = {name: weight for name in synthetic}
            decisions.append(
                f"Synthetic sampling: {len(synthetic)} source(s) matching "
                f"{', '.join(prefixes)}* weighted by {weight:g}"
            )

    return decisions
