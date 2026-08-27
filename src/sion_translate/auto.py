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
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from sion_translate.config import AppConfig, ExperimentalConfig
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

# Model capacity follows the unique token mass in the authenticated prepared
# training split. Virtual reverse directions and repeated epochs reuse those
# tokens and therefore must not make the model appear to have more source data.
# Thirty-two tokens per physical pair is an explicit compatibility reference
# used only when a legacy caller cannot provide prepared token lengths. It is
# not a measurement of the current corpus and never replaces prepared lengths
# in the production training path.
MODEL_REFERENCE_TOKENS_PER_PAIR = 32
DEFAULT_MODEL_SIZING_VOCAB_SIZE = 48_000
MODEL_SIZE_KEYS = (
    "d_model",
    "encoder_layers",
    "decoder_layers",
    "num_heads",
    "num_kv_heads",
    "d_ff",
)
MODEL_PRESETS: tuple[tuple[int, str, dict[str, int]], ...] = (
    # (physical-pair-equivalent anchor, stable anchor name, model settings)
    (
        200_000,
        "small",
        dict(
            d_model=512,
            encoder_layers=8,
            decoder_layers=4,
            num_heads=8,
            num_kv_heads=4,
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
            num_kv_heads=5,
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
            num_kv_heads=6,
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
            num_kv_heads=8,
            d_ff=2816,
        ),
    ),
    (
        300_000_000,
        "xlarge",
        dict(
            d_model=1280,
            encoder_layers=24,
            decoder_layers=12,
            num_heads=20,
            num_kv_heads=10,
            d_ff=3584,
        ),
    ),
)

_MODEL_LADDER_QUANTA = {
    "d_model": 32,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "d_ff": 128,
}


@dataclass(frozen=True)
class _ModelArchitectureCandidate:
    """One reproducible, GPU-compatible point on the capacity ladder."""

    name: str
    settings: dict[str, int]


def _head_layout_for_width(d_model: int) -> tuple[int, int]:
    """Choose a tensor-friendly GQA layout without reducing KV capacity.

    Automatic candidates use two query heads per KV head. This makes the total
    KV projection width exactly half of d_model, so increasing model width can
    never silently reduce the rank of the key/value representation. Among
    valid layouts, prefer a head dimension near 64 while requiring an 8-wide
    tensor-core alignment and keeping the dimension in the practical 64--160
    range used by the ladder.
    """

    candidates: list[tuple[int, int, int]] = []
    for num_heads in range(2, d_model + 1, 2):
        if d_model % num_heads:
            continue
        head_dim = d_model // num_heads
        if 64 <= head_dim <= 160 and head_dim % 8 == 0:
            candidates.append((abs(head_dim - 64), -num_heads, num_heads))
    if not candidates:
        raise AssertionError(f"No supported automatic attention layout for width {d_model}")
    num_heads = min(candidates)[2]
    return num_heads, num_heads // 2


def _interpolate_architecture_segment(
    lower: Mapping[str, int],
    upper: Mapping[str, int],
) -> tuple[dict[str, int], ...]:
    """Walk between anchors while changing one primary architecture knob per step."""

    current = dict(lower)
    architectures = [dict(current)]
    events: list[tuple[Fraction, int, str, int]] = []
    for order, (key, quantum) in enumerate(_MODEL_LADDER_QUANTA.items()):
        distance = upper[key] - lower[key]
        if distance < 0 or distance % quantum:
            raise AssertionError(f"Invalid model anchor progression for {key}")
        increments = [quantum] * (distance // quantum)
        if key == "d_model":
            # Thirty-two-wide candidates keep early-model jumps below 12%.
            # Some widths have no group-size-two, 8-aligned head layout in the
            # supported head-dimension range; merge those increments into the
            # next valid width instead of emitting an inefficient shape.
            increments = []
            previous_width = lower[key]
            for width in range(lower[key] + quantum, upper[key] + 1, quantum):
                try:
                    _head_layout_for_width(width)
                except AssertionError:
                    continue
                increments.append(width - previous_width)
                previous_width = width
            if previous_width != upper[key]:
                raise AssertionError(f"Upper model width {upper[key]} has no supported layout")
        step_count = len(increments)
        for step, increment in enumerate(increments, start=1):
            # Midpoint event positions interleave width, depth, and FFN growth.
            # The secondary key keeps simultaneous events reproducible while
            # retaining the one-knob-per-candidate maximum-jump guarantee.
            events.append((Fraction(2 * step - 1, 2 * step_count), order, key, increment))
    for _position, _order, key, quantum in sorted(events):
        current = dict(current)
        current[key] += quantum
        if key == "d_model":
            current["num_heads"], current["num_kv_heads"] = _head_layout_for_width(current[key])
        architectures.append(current)
    if current != dict(upper):
        raise AssertionError("Model architecture anchors cannot be connected by the ladder")
    return tuple(architectures)


def _build_model_architecture_ladder() -> tuple[_ModelArchitectureCandidate, ...]:
    candidates: list[_ModelArchitectureCandidate] = []
    for anchor_index in range(len(MODEL_PRESETS) - 1):
        _lower_tokens, lower_name, lower = MODEL_PRESETS[anchor_index]
        _upper_tokens, upper_name, upper = MODEL_PRESETS[anchor_index + 1]
        segment = _interpolate_architecture_segment(lower, upper)
        for step, settings in enumerate(segment[:-1]):
            name = lower_name if step == 0 else f"{lower_name}-{upper_name}-{step:02d}"
            candidates.append(_ModelArchitectureCandidate(name, settings))
    _tokens, final_name, final_settings = MODEL_PRESETS[-1]
    candidates.append(_ModelArchitectureCandidate(final_name, dict(final_settings)))
    return tuple(candidates)


MODEL_ARCHITECTURE_LADDER = _build_model_architecture_ladder()

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


def _validated_pair_count(pair_count: int) -> int:
    if type(pair_count) is not int:
        raise TypeError("pair_count must be a non-negative integer")
    if pair_count < 0:
        raise ValueError("pair_count must be non-negative")
    return pair_count


def _validated_training_token_count(training_token_count: int) -> int:
    if type(training_token_count) is not int:
        raise TypeError("training_token_count must be a non-negative integer")
    if training_token_count < 0:
        raise ValueError("training_token_count must be non-negative")
    return training_token_count


def _validated_architecture(architecture: Mapping[str, int]) -> dict[str, int]:
    missing = tuple(key for key in MODEL_SIZE_KEYS if key not in architecture)
    if missing:
        raise ValueError(f"Model architecture is missing required keys: {missing}")
    settings: dict[str, int] = {}
    for key in MODEL_SIZE_KEYS:
        value = architecture[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"Model architecture {key} must be a positive integer")
        settings[key] = value
    if settings["d_model"] % settings["num_heads"]:
        raise ValueError("Model architecture d_model must be divisible by num_heads")
    if settings["num_heads"] % settings["num_kv_heads"]:
        raise ValueError("Model architecture num_heads must be divisible by num_kv_heads")
    return settings


def estimate_model_parameter_count(
    architecture: Mapping[str, int],
    *,
    vocab_size: int,
    tie_embeddings: bool = True,
    experimental: ExperimentalConfig | None = None,
) -> int:
    """Return the exact trainable parameter count for one supported architecture.

    The calculation mirrors ``SionForConditionalGeneration`` without allocating
    tensors. It includes the configured vocabulary, tied or untied output
    embeddings, and every optional module that can add trainable parameters.
    """

    settings = _validated_architecture(architecture)
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        tie_embeddings, bool
    ):
        raise TypeError("tie_embeddings must be a boolean")
    exp = experimental if experimental is not None else ExperimentalConfig()

    d_model = settings["d_model"]
    num_heads = settings["num_heads"]
    num_kv_heads = settings["num_kv_heads"]
    d_ff = settings["d_ff"]
    encoder_layers = settings["encoder_layers"]
    decoder_layers = settings["decoder_layers"]
    head_dim = d_model // num_heads

    # One grouped-query attention block has Q and output projections of d*d,
    # plus K and V projections of d*(kv_heads*head_dim).
    attention = 2 * d_model * d_model + 2 * d_model * num_kv_heads * head_dim
    feed_forward = 3 * d_model * d_ff
    encoder_layer = attention + feed_forward + 2 * d_model
    decoder_layer = 2 * attention + feed_forward + 3 * d_model
    parameters = (
        vocab_size * d_model
        + encoder_layers * encoder_layer
        + decoder_layers * decoder_layer
        + 2 * d_model
    )
    if not tie_embeddings:
        parameters += vocab_size * d_model

    if exp.morphoscript_enabled:
        feature_dim = max(32, d_model // 16)
        feature_rows = exp.script_classes + 20 + 22 + 29
        parameters += (
            feature_rows * feature_dim
            + 4 * feature_dim * d_model
            + d_model * d_model
            + encoder_layers
        )
    if exp.core_enabled:
        parameters += d_model + 2 * d_model * exp.register_classes + exp.register_classes + 1
    if exp.tetm_enabled:
        parameters += (exp.tetm_types + exp.tetm_modes + 2) * d_model + attention + 1
    if exp.bats_enabled:
        parameters += 2 * d_model * exp.bats_dim + exp.bats_dim
    if exp.evidence_repair_enabled:
        parameters += attention + 3 * d_model * d_model + 3 * d_model + 2
    if exp.candidate_refinement_enabled:
        parameters += 5 * d_model * d_model + 2 * d_model + 1
    if exp.semantic_parity_enabled:
        parameters += 2 * d_model + 2 * d_model * exp.semantic_parity_dim
    return parameters


def smooth_model_parameter_target(training_token_count: int) -> int:
    """Map unique prepared tokens to a continuous total-parameter budget.

    The log-log interpolation follows the established model anchors but uses a
    fixed 48k tied-embedding reference. Consequently, changing the tokenizer
    vocabulary consumes a different share of one stable total budget instead
    of adding a second capacity cliff on top of the data scaling rule.
    """

    token_count = _validated_training_token_count(training_token_count)
    anchors = tuple(
        (
            pair_anchor * MODEL_REFERENCE_TOKENS_PER_PAIR,
            estimate_model_parameter_count(
                settings,
                vocab_size=DEFAULT_MODEL_SIZING_VOCAB_SIZE,
            ),
        )
        for pair_anchor, _name, settings in MODEL_PRESETS
    )
    if token_count <= anchors[0][0]:
        return anchors[0][1]
    if token_count >= anchors[-1][0]:
        return anchors[-1][1]
    for (lower_tokens, lower_parameters), (upper_tokens, upper_parameters) in zip(
        anchors[:-1],
        anchors[1:],
        strict=True,
    ):
        if token_count <= upper_tokens:
            fraction = math.log(token_count / lower_tokens) / math.log(upper_tokens / lower_tokens)
            log_target = math.log(lower_parameters) + fraction * math.log(
                upper_parameters / lower_parameters
            )
            return round(math.exp(log_target))
    raise AssertionError("Model scaling anchors must cover every validated token count")


def _select_model_architecture(
    training_token_count: int,
    *,
    vocab_size: int,
    tie_embeddings: bool,
    experimental: ExperimentalConfig | None,
) -> tuple[str, dict[str, int], int, int]:
    token_count = _validated_training_token_count(training_token_count)
    target = smooth_model_parameter_target(token_count)
    scored = tuple(
        (
            estimate_model_parameter_count(
                candidate.settings,
                vocab_size=vocab_size,
                tie_embeddings=tie_embeddings,
                experimental=experimental,
            ),
            candidate,
        )
        for candidate in MODEL_ARCHITECTURE_LADDER
    )
    estimated_parameters, selected = min(
        scored,
        # Relative log distance treats equal percentage under/over-shoots
        # symmetrically. A tie chooses the smaller model for memory safety.
        key=lambda item: (
            abs(math.log(item[0] / target)),
            item[0],
            item[1].name,
        ),
    )
    return selected.name, dict(selected.settings), target, estimated_parameters


def pick_model_architecture(
    training_token_count: int,
    *,
    vocab_size: int = DEFAULT_MODEL_SIZING_VOCAB_SIZE,
    tie_embeddings: bool = True,
    experimental: ExperimentalConfig | None = None,
) -> tuple[str, dict[str, int]]:
    """Select the nearest valid architecture to a smooth token-scaled target."""

    name, settings, _target, _estimated = _select_model_architecture(
        training_token_count,
        vocab_size=vocab_size,
        tie_embeddings=tie_embeddings,
        experimental=experimental,
    )
    return name, settings


def pick_model_preset(
    pair_count: int,
    *,
    vocab_size: int = DEFAULT_MODEL_SIZING_VOCAB_SIZE,
    tie_embeddings: bool = True,
    experimental: ExperimentalConfig | None = None,
) -> tuple[str, dict[str, int]]:
    """Compatibility wrapper using the documented tokens-per-pair reference."""

    count = _validated_pair_count(pair_count)
    return pick_model_architecture(
        count * MODEL_REFERENCE_TOKENS_PER_PAIR,
        vocab_size=vocab_size,
        tie_embeddings=tie_embeddings,
        experimental=experimental,
    )


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


def apply_auto_data_settings(
    config: AppConfig,
    raw: dict[str, Any],
    *,
    train_examples: int,
    physical_train_pairs: int | None = None,
    physical_train_tokens: int | None = None,
    source_names: list[str] | None = None,
) -> list[str]:
    """Apply only decisions derived from authenticated prepared data.

    This phase deliberately has no execution-environment argument. It is safe
    to run on a local CPU before a GPU upload because it cannot choose precision,
    per-device batch size, accumulation, parallelism, or DataLoader workers.
    """

    raw_model = dict(raw.get("model") or {})
    raw_training = dict(raw.get("training") or {})
    raw_data = dict(raw.get("data") or {})
    decisions: list[str] = []

    def auto(section: dict[str, Any], key: str) -> bool:
        return key not in section

    pair_count = _validated_pair_count(
        physical_train_pairs
        if physical_train_pairs is not None
        else train_examples // (2 if config.data.bidirectional else 1)
    )
    if physical_train_tokens is None:
        training_token_count = pair_count * MODEL_REFERENCE_TOKENS_PER_PAIR
        token_basis = (
            f"legacy {MODEL_REFERENCE_TOKENS_PER_PAIR}-tokens-per-pair reference; "
            "prepared token lengths unavailable"
        )
    else:
        training_token_count = _validated_training_token_count(physical_train_tokens)
        token_basis = "authenticated prepared src+tgt lengths"

    # Model capacity follows unique physical token mass. Expanding one row into
    # additional graph directions or repeating it for more epochs does not
    # create new source data and therefore cannot increase this budget.
    explicit_size_keys = tuple(key for key in MODEL_SIZE_KEYS if key in raw_model)
    if explicit_size_keys and len(explicit_size_keys) != len(MODEL_SIZE_KEYS):
        missing = tuple(key for key in MODEL_SIZE_KEYS if key not in raw_model)
        raise ValueError(
            "Manual model architecture overrides must provide every preset-defining key. "
            f"Provided: {explicit_size_keys}; missing: {missing}."
        )
    if not explicit_size_keys:
        sizing_vocab = config.model.vocab_size or DEFAULT_MODEL_SIZING_VOCAB_SIZE
        name, preset, target_parameters, estimated_parameters = _select_model_architecture(
            training_token_count,
            vocab_size=sizing_vocab,
            tie_embeddings=config.model.tie_embeddings,
            experimental=config.model.experimental,
        )
        for key, value in preset.items():
            setattr(config.model, key, value)
        decisions.append(
            f"Model architecture: {name} for {training_token_count:,} unique physical "
            f"training tokens ({token_basis}; {pair_count:,} physical pairs; virtual "
            f"directions and epochs excluded). Estimated {estimated_parameters:,} "
            f"parameters against a smooth {target_parameters:,}-parameter target "
            f"with vocabulary size {sizing_vocab:,}."
        )

    if auto(raw_training, "num_train_epochs") and auto(raw_training, "max_steps"):
        epochs = target_epochs(pair_count)
        config.training.num_train_epochs = epochs
        config.training.max_steps = None
        decisions.append(
            f"num_train_epochs: {epochs} complete corpus passes for {pair_count:,} physical pairs"
        )

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


def apply_auto_settings(
    config: AppConfig,
    raw: dict[str, Any],
    env: EnvironmentInfo,
    *,
    train_examples: int,
    validation_examples: int,
    physical_train_pairs: int | None = None,
    physical_train_tokens: int | None = None,
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
    decisions = apply_auto_data_settings(
        config,
        raw,
        train_examples=train_examples,
        physical_train_pairs=physical_train_pairs,
        physical_train_tokens=physical_train_tokens,
        source_names=source_names,
    )

    def auto(section: dict[str, Any], key: str) -> bool:
        """Return whether a key is eligible for automatic configuration."""
        return key not in section

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

    # Convert the data-derived complete-epoch budget into this runtime's step
    # count only after per-device batch and accumulation have been selected.
    batches_per_epoch = math.ceil(
        train_examples / max(1, config.training.batch_size_per_gpu * env.world_size)
    )
    steps_per_epoch = math.ceil(batches_per_epoch / config.training.gradient_accumulation_steps)
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

    return decisions
