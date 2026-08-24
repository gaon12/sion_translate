"""Training loop for sion_translate.

The main sequence is:

1. Prepare AdamW, the warmup-plus-cosine scheduler, and the AMP scaler.
2. Restore training state from a checkpoint when requested.
3. Run micro-batches, gradient accumulation, and optimizer updates.
   - Show loss, learning rate, and gradient norm in a tqdm progress bar.
   - Write one JSON record and TensorBoard values every ``log_every`` steps.
4. Validate every ``eval_every`` steps, update the best checkpoint, and apply
   early stopping.
5. Save two artifact classes at each relevant point:

   - checkpoints/best, checkpoints/latest, checkpoints/final
     for resume, including the large optimizer state.
   - exports/best, exports/latest
     for inference, including regular FP32 ``model.pt`` and quantized INT8
     ``model_int8.pt`` artifacts.
"""

# DataLoader samplers, AMP scalers, tqdm, and SummaryWriter expose dynamic hooks.
# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Sized
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from sion_translate.artifacts import TRANSLATION_RELEASE_NAME
from sion_translate.config import AppConfig, TrainingConfig
from sion_translate.language_tags import canonicalize_language_pair

from .checkpoint import (
    build_checkpoint_identity,
    build_objective_identity,
    load_checkpoint,
    preflight_checkpoint_identity,
    save_checkpoint,
    verified_checkpoint_generation_lease,
)
from .distributed import (
    DistributedContext,
    broadcast_bool,
    distributed_failure_scope,
    maybe_no_sync,
    reduce_max,
    reduce_sum,
)
from .ema import EMAWeights
from .export import build_export_metadata, export_inference_models
from .objectives import ObjectiveOutput


@dataclass(frozen=True)
class TrainingBudget:
    """Resolved stopping budget based on either complete epochs or a step override."""

    max_optimizer_steps: int
    target_epochs: int | None
    batches_per_epoch: int | None

    @property
    def epoch_limited(self) -> bool:
        return self.target_epochs is not None

    def should_continue(self, *, step: int, epoch: int) -> bool:
        if self.target_epochs is not None:
            return epoch < self.target_epochs
        return step < self.max_optimizer_steps


def _atomic_write_resolved_config(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably publish one complete resolved configuration generation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
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


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _publish_resolved_config(
    config: AppConfig,
    output_dir: Path,
    context: DistributedContext,
) -> None:
    """Publish on rank 0 and make every peer observe a write failure."""

    write_error: Exception | None = None
    if context.is_main:
        try:
            _atomic_write_resolved_config(output_dir / "resolved_config.json", config.to_dict())
        except Exception as error:
            write_error = error
    write_failed = broadcast_bool(write_error is not None, context)
    if not write_failed:
        return
    if write_error is not None:
        raise write_error
    raise RuntimeError("rank 0 failed to publish resolved_config.json")


def resolve_training_budget(
    loader: Iterable[dict[str, torch.Tensor]],
    training: TrainingConfig,
) -> TrainingBudget:
    """Turn the public epoch contract into an exact optimizer-step horizon.

    ``max_steps`` remains an explicit legacy/debug override. Normal training
    requires a sized loader so it can finish every batch in every requested
    epoch and build a scheduler with the exact number of optimizer updates.
    """

    batches_per_epoch = len(loader) if isinstance(loader, Sized) else None
    if batches_per_epoch is not None and batches_per_epoch <= 0:
        raise ValueError("training loader produced no batches")
    if training.max_steps is not None:
        return TrainingBudget(
            max_optimizer_steps=training.max_steps,
            target_epochs=None,
            batches_per_epoch=batches_per_epoch,
        )
    if batches_per_epoch is None:
        raise TypeError("epoch-based training requires a sized training loader")
    updates_per_epoch = math.ceil(batches_per_epoch / training.gradient_accumulation_steps)
    return TrainingBudget(
        max_optimizer_steps=updates_per_epoch * training.num_train_epochs,
        target_epochs=training.num_train_epochs,
        batches_per_epoch=batches_per_epoch,
    )


def announce(message: str, context: DistributedContext) -> None:
    """Print a human-readable progress message from rank 0.

    ``tqdm.write`` places the message above the progress bar without corrupting
    its display.
    """
    if context.is_main:
        tqdm.write(f"[sion] {message}")


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a collated tensor batch to its CPU or GPU training device."""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    max_steps: int,
    min_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a linear-warmup and cosine-decay learning-rate schedule.

    The multiplier rises linearly from zero to its maximum over
    ``warmup_steps``, then follows a cosine curve down to ``min_ratio`` of that
    maximum.
    """

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def build_optimizer_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split parameters so AdamW decay applies only to matrix-like weights.

    Norm weights, biases, and one-dimensional gates are conventionally excluded
    because decaying them can destabilize training.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized_name = name.lower()
        should_decay = (
            parameter.ndim >= 2
            and not normalized_name.endswith(".bias")
            and "norm" not in normalized_name
        )
        (decay if should_decay else no_decay).append(parameter)
    if not decay and weight_decay > 0:
        raise ValueError("No decay-eligible model parameters were found")
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _autocast_context(precision: str, device: torch.device):
    """Return an AMP context enabled only for CUDA bf16 or fp16 compute."""
    if device.type != "cuda" or precision.lower() == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision.lower() == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_grad_scaler(training: TrainingConfig, context: DistributedContext):
    """Create the GradScaler that prevents gradient underflow in fp16 training.

    bf16 and fp32 receive a disabled scaler that behaves as a no-op.
    """
    enabled = context.device.type == "cuda" and training.precision.lower() == "fp16"
    fsdp_enabled = training.parallel_strategy.lower() == "fsdp2" or (
        training.parallel_strategy.lower() == "auto" and training.fsdp2 is True
    )
    if enabled and context.distributed and fsdp_enabled:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        return ShardedGradScaler(device="cuda", enabled=True)
    return torch.amp.GradScaler("cuda", enabled=enabled)


def _make_summary_writer(
    training: TrainingConfig,
    output_dir: Path,
    context: DistributedContext,
    start_step: int,
):
    """Create a rank-0 TensorBoard writer and purge superseded resumed steps."""
    if not training.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging is enabled but the 'tensorboard' package is unavailable. "
            "Install the project dependencies or set training.tensorboard=false."
        ) from exc
    if not context.is_main:
        return None
    log_dir = (
        Path(training.tensorboard_dir) if training.tensorboard_dir else output_dir / "tensorboard"
    )
    return SummaryWriter(
        log_dir=str(log_dir),
        purge_step=start_step if start_step > 0 else None,
    )


def _fail_if_known_empty(loader: Iterable[dict[str, torch.Tensor]], name: str) -> None:
    """Reject an empty sized loader before training starts."""
    try:
        length = len(loader)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return
    if length == 0:
        raise ValueError(f"{name} loader is empty")


def _language_metric_layout(
    language_tags: Mapping[str, int] | None,
) -> tuple[tuple[str, int], ...]:
    """Return a rank-stable language order for packed distributed statistics."""

    if not language_tags:
        return ()
    layout = tuple(
        sorted((str(language), int(tag_id)) for language, tag_id in language_tags.items())
    )
    tag_ids = [tag_id for _, tag_id in layout]
    if len(tag_ids) != len(set(tag_ids)):
        raise ValueError("language_tags must assign a unique token id to every language")
    return layout


def _objective_metric_layout(
    local_names: Iterable[str],
    context: DistributedContext,
    language_layout: Sequence[tuple[str, int]],
) -> tuple[str, ...]:
    """Build one rank-stable layout for optional validation objective metrics.

    Direction reward accumulators have a layout known from ``language_tags``.
    Other objectives may expose arbitrary metric names, so distributed ranks
    exchange only those small string tuples and reduce the numeric values later
    in one packed tensor. A metric absent locally is consequently represented by
    zero instead of changing the collective count or order.
    """

    direction_names: set[str] = set()
    for source_language, _ in language_layout:
        for target_language, _ in language_layout:
            if source_language == target_language:
                continue
            prefix = f"direction_{source_language}_to_{target_language}"
            direction_names.update((f"{prefix}_reward_sum", f"{prefix}_rows"))

    custom_names = tuple(sorted(set(local_names) - direction_names))
    if not context.distributed:
        return tuple(sorted(direction_names.union(custom_names)))

    gathered_names: list[tuple[str, ...] | None] = [None] * context.world_size
    dist.all_gather_object(gathered_names, custom_names)
    for rank_names in gathered_names:
        if rank_names is None:
            raise RuntimeError("distributed objective metric name gathering was incomplete")
        direction_names.update(rank_names)
    return tuple(sorted(direction_names))


def _perplexity(nll: float) -> float:
    """Exponentiate token NLL, representing an unreportably large value as infinity."""

    try:
        return math.exp(nll)
    except OverflowError:
        return float("inf")


def _validation_metric_suffix(key: str) -> str:
    prefix = "validation_ema_" if key.startswith("validation_ema_") else "validation_"
    return key.removeprefix(prefix)


def _select_sft_validation_metric(
    metrics: Mapping[str, float],
    configured_metric: str,
    *,
    prefer_ema: bool,
) -> tuple[float, str, bool]:
    """Choose a finite SFT metric, falling back without making training unselectable.

    Direction metrics can legitimately be absent when a bounded validation slice
    contains only denoising rows or when a custom caller did not provide language
    tag metadata. The fallback order keeps the requested direction metric first,
    then true global NLL, and only then the legacy label-smoothed loss.
    """

    suffix_by_setting = {
        "global_nll": "nll",
        "macro_direction_nll": "macro_direction_nll",
        "worst_direction_nll": "worst_direction_nll",
    }
    configured_suffix = suffix_by_setting[configured_metric.lower()]
    prefixes = ["validation_ema_"] if prefer_ema else ["validation_"]
    candidate_keys: list[str] = []
    for suffix in (configured_suffix, "nll", "loss"):
        for prefix in prefixes:
            key = f"{prefix}{suffix}"
            if key not in candidate_keys:
                candidate_keys.append(key)
    for key in candidate_keys:
        if key not in metrics:
            continue
        value = float(metrics[key])
        if math.isfinite(value):
            return value, key, _validation_metric_suffix(key) != configured_suffix
    requested_prefix = prefixes[0]
    return float("inf"), f"{requested_prefix}{configured_suffix}", True


def _select_posttraining_validation_metric(
    metrics: Mapping[str, float],
    configured_metric: str,
    *,
    prefer_ema: bool,
) -> tuple[float, str, bool]:
    """Select the same raw/EMA model family that will be restored for deployment."""

    prefixes = ["validation_ema_"] if prefer_ema else ["validation_"]
    candidate_keys: list[str] = []
    for suffix in (configured_metric, "reward"):
        for prefix in prefixes:
            key = f"{prefix}{suffix}"
            if key not in candidate_keys:
                candidate_keys.append(key)
    for key in candidate_keys:
        if key not in metrics:
            continue
        value = float(metrics[key])
        if math.isfinite(value):
            return value, key, _validation_metric_suffix(key) != configured_metric
    requested_prefix = prefixes[0]
    return float("-inf"), f"{requested_prefix}{configured_metric}", True


def _selection_metric_label(key: str) -> str:
    ema = key.startswith("validation_ema_")
    labels = {
        "macro_direction_nll": "direction-balanced macro NLL",
        "worst_direction_nll": "worst-direction NLL",
        "reward": "composite generation reward",
        "macro_direction_reward": "direction-balanced macro reward",
        "worst_direction_reward": "worst-direction reward",
        "nll": "overall token NLL",
        "loss": "validation loss",
    }
    label = labels.get(_validation_metric_suffix(key), key)
    return f"EMA {label}" if ema else label


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    context: DistributedContext,
    max_batches: int,
    *,
    precision: str = "fp32",
    show_progress: bool = False,
    objective: Callable[[nn.Module, dict[str, torch.Tensor]], ObjectiveOutput] | None = None,
    language_tags: Mapping[str, int] | None = None,
) -> dict[str, float]:
    """Calculate CE/NLL and optional generation-quality rewards without gradients.

    - ``validation_loss`` retains the same label-smoothed CE used for training.
    - Perplexity and language/direction metrics use true unsmoothed token NLL.
    - Every rank reduces the same fixed-layout direction tensor, so collective
      order remains stable even when one rank has no rows for a direction.
    - Distributed totals are all-reduced before global means are calculated.
    """
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=context.device, dtype=torch.float64)
    token_count = torch.zeros((), device=context.device, dtype=torch.float64)
    nll_sum = torch.zeros((), device=context.device, dtype=torch.float64)
    nll_token_count = torch.zeros((), device=context.device, dtype=torch.float64)
    aux_sum = torch.zeros((), device=context.device, dtype=torch.float64)
    objective_sums: dict[str, torch.Tensor] = {}
    objective_count = torch.zeros((), device=context.device, dtype=torch.float64)
    language_layout = _language_metric_layout(language_tags)
    language_count = len(language_layout)
    # First N rows are target-language stats; the remaining N*N rows are
    # source-major explicit direction stats. Each row is [NLL sum, token count].
    language_stats = torch.zeros(
        (language_count + language_count * language_count, 2),
        device=context.device,
        dtype=torch.float64,
    )
    batches = 0

    # Show a short rank-0 validation progress bar and remove it when complete.
    progress = None
    if show_progress and context.is_main:
        try:
            total = min(len(loader), max_batches)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            total = max_batches
        progress = tqdm(
            total=total, desc="validation", unit="batch", leave=False, dynamic_ncols=True
        )

    try:
        for batch in loader:
            batch = move_to_device(batch, context.device)
            validation_metrics = getattr(objective, "validation_metrics", None)
            with _autocast_context(precision, context.device):
                output = model(**batch)
                generated_metrics = (
                    validation_metrics(model, batch) if validation_metrics is not None else None
                )
            labels = batch["labels"]
            token_nll = F.cross_entropy(
                output.logits.detach().float().reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(labels)
            valid_tokens = labels.ne(-100)
            row_nll = token_nll.sum(dim=1).double()
            row_token_count = valid_tokens.sum(dim=1).double()
            loss_sum += output.lm_loss_sum.detach().double()
            token_count += output.token_count.detach().double()
            nll_sum += row_nll.sum()
            nll_token_count += row_token_count.sum()
            aux_sum += output.auxiliary_loss.detach().double()
            if language_layout:
                target_tag_ids = batch["input_ids"][:, 0]
                source_tag_ids = batch.get("source_language_tag_ids")
                for target_index, (_, target_tag_id) in enumerate(language_layout):
                    target_rows = target_tag_ids.eq(target_tag_id)
                    language_stats[target_index, 0] += row_nll[target_rows].sum()
                    language_stats[target_index, 1] += row_token_count[target_rows].sum()
                    if source_tag_ids is None:
                        continue
                    for source_index, (_, source_tag_id) in enumerate(language_layout):
                        direction_rows = target_rows & source_tag_ids.eq(source_tag_id)
                        direction_index = (
                            language_count + source_index * language_count + target_index
                        )
                        language_stats[direction_index, 0] += row_nll[direction_rows].sum()
                        language_stats[direction_index, 1] += row_token_count[direction_rows].sum()
            if generated_metrics is not None:
                source_count = float(batch["input_ids"].shape[0])
                objective_count += source_count
                for name, value in generated_metrics.items():
                    if name not in objective_sums:
                        objective_sums[name] = torch.zeros(
                            (), device=context.device, dtype=torch.float64
                        )
                    objective_sums[name] += value.detach().double() * source_count
            batches += 1
            if progress is not None:
                progress.update(1)
            if batches >= max_batches:
                break
    finally:
        if progress is not None:
            progress.close()

    if batches == 0:
        model.train(was_training)
        raise ValueError("validation loader produced no batches")
    reduce_sum(loss_sum, context)
    reduce_sum(token_count, context)
    reduce_sum(nll_sum, context)
    reduce_sum(nll_token_count, context)
    reduce_sum(aux_sum, context)
    reduce_sum(objective_count, context)
    # One fixed-size collective covers every language/direction, including rows
    # that are locally zero but observed on another DDP rank.
    if language_layout:
        reduce_sum(language_stats, context)
    objective_layout = (
        _objective_metric_layout(objective_sums, context, language_layout)
        if objective is not None
        else ()
    )
    if objective_layout:
        packed_objective_sums = torch.zeros(
            len(objective_layout),
            device=context.device,
            dtype=torch.float64,
        )
        for index, name in enumerate(objective_layout):
            local_value = objective_sums.get(name)
            if local_value is not None:
                packed_objective_sums[index] = local_value
        reduce_sum(packed_objective_sums, context)
        objective_sums = {
            name: packed_objective_sums[index] for index, name in enumerate(objective_layout)
        }
    batch_tensor = torch.tensor(float(batches), device=context.device, dtype=torch.float64)
    reduce_sum(batch_tensor, context)
    model.train(was_training)
    mean_loss = (loss_sum / token_count.clamp_min(1)).item()
    mean_nll = (nll_sum / nll_token_count.clamp_min(1)).item()
    metrics = {
        "validation_loss": mean_loss,
        "validation_nll": mean_nll,
        "validation_perplexity": _perplexity(mean_nll),
        "validation_auxiliary_loss": (aux_sum / batch_tensor.clamp_min(1)).item(),
        "validation_tokens": nll_token_count.item(),
    }
    direction_nlls: list[float] = []
    for target_index, (target_language, _) in enumerate(language_layout):
        target_tokens = language_stats[target_index, 1].item()
        if target_tokens > 0:
            target_nll = (language_stats[target_index, 0] / target_tokens).item()
            prefix = f"validation_target_{target_language}"
            metrics[f"{prefix}_nll"] = target_nll
            metrics[f"{prefix}_perplexity"] = _perplexity(target_nll)
            metrics[f"{prefix}_tokens"] = target_tokens
        for source_index, (source_language, _) in enumerate(language_layout):
            direction_index = language_count + source_index * language_count + target_index
            direction_tokens = language_stats[direction_index, 1].item()
            if direction_tokens <= 0:
                continue
            direction_nll = (language_stats[direction_index, 0] / direction_tokens).item()
            direction_nlls.append(direction_nll)
            prefix = f"validation_direction_{source_language}_to_{target_language}"
            metrics[f"{prefix}_nll"] = direction_nll
            metrics[f"{prefix}_perplexity"] = _perplexity(direction_nll)
            metrics[f"{prefix}_tokens"] = direction_tokens
    if direction_nlls:
        macro_direction_nll = sum(direction_nlls) / len(direction_nlls)
        worst_direction_nll = max(direction_nlls)
        metrics.update(
            {
                "validation_macro_direction_nll": macro_direction_nll,
                "validation_macro_direction_perplexity": _perplexity(macro_direction_nll),
                "validation_worst_direction_nll": worst_direction_nll,
                "validation_worst_direction_perplexity": _perplexity(worst_direction_nll),
                "validation_direction_count": float(len(direction_nlls)),
            }
        )
    if objective_sums:
        denominator = objective_count.clamp_min(1)
        # Sums and row counts receive the same aggregation weight, which cancels
        # when divided. Worst-direction and macro rewards prevent a higher global
        # mean from hiding a regression in one translation direction.
        direction_rewards: dict[str, float] = {}
        for name in list(objective_sums):
            if not name.endswith("_reward_sum"):
                continue
            direction = name.removesuffix("_reward_sum")
            rows = objective_sums.get(f"{direction}_rows")
            if rows is None or float(rows.item()) <= 0:
                continue
            direction_rewards[direction] = float((objective_sums[name] / rows).item())
        if direction_rewards:
            for direction, value in direction_rewards.items():
                metrics[f"validation_{direction}_reward"] = value
            metrics["validation_worst_direction_reward"] = min(direction_rewards.values())
            metrics["validation_macro_direction_reward"] = sum(direction_rewards.values()) / len(
                direction_rewards
            )
            metrics["validation_reward_direction_count"] = float(len(direction_rewards))
        metrics.update(
            {
                f"validation_{name}": (value / denominator).item()
                for name, value in objective_sums.items()
                if not (name.endswith("_reward_sum") or name.endswith("_rows"))
            }
        )
    return metrics


def build_training_checkpoint_identity(
    config: AppConfig,
    *,
    batch_sampler: Any,
    context: DistributedContext,
    stage_name: str,
    include_posttraining: bool,
    pipeline_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact identity shared by resume preflight and ``train``."""

    return build_checkpoint_identity(
        model_config=config.model,
        tokenizer_path=config.data.tokenizer_model,
        token_features_path=config.data.tokenizer_features,
        dataset_dir=config.data.dataset_dir,
        data_config=config.data,
        sampling_seed=getattr(
            batch_sampler,
            "seed",
            config.training.seed,
        ),
        stage_name=stage_name,
        # Objective and optimization settings are part of checkpoint identity.
        # Include the reward definition only for MRT so unrelated MRT changes do
        # not invalidate an SFT resume.
        objective_identity=build_objective_identity(
            config.training,
            config.posttraining,
            include_posttraining=include_posttraining,
        ),
        pipeline_identity=pipeline_identity,
        loader_config={
            "batch_size_per_rank": getattr(
                batch_sampler,
                "batch_size",
                config.training.batch_size_per_gpu,
            ),
            "world_size": context.world_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "drop_last": getattr(batch_sampler, "drop_last", None),
            "bucket_size": getattr(batch_sampler, "bucket_size", None),
        },
    )


def train(
    model: nn.Module,
    train_loader: Iterable[dict[str, torch.Tensor]],
    validation_loader: Iterable[dict[str, torch.Tensor]],
    config: AppConfig,
    context: DistributedContext,
    *,
    start_step: int = 0,
    objective: Callable[[nn.Module, dict[str, torch.Tensor]], ObjectiveOutput] | None = None,
    stage_name: str = "pretrain",
    language_tags: Mapping[str, int] | None = None,
    export_release_name: str = TRANSLATION_RELEASE_NAME,
    export_translation_capable: bool = True,
    export_languages: Sequence[str] | None = None,
    authenticated_revision_directions: Sequence[Sequence[str]] | None = None,
    pipeline_identity: Mapping[str, Any] | None = None,
) -> dict[str, float | int | bool | str]:
    """Train sion_translate and return progress plus best-selection metadata."""
    config.validate()
    configured_revision_directions = config.data.configured_revision_directions()
    if not export_translation_capable:
        if authenticated_revision_directions:
            raise ValueError(
                "foundation training cannot authenticate translation revision directions"
            )
        export_revision_directions: tuple[tuple[str, str], ...] = ()
    elif authenticated_revision_directions is None:
        if configured_revision_directions or config.data.revision_examples:
            raise ValueError(
                "revision-capable training requires authenticated_revision_directions from "
                "indexed provenance and the positive sampling graph"
            )
        export_revision_directions = ()
    else:
        normalized_revision_directions: list[tuple[str, str]] = []
        seen_revision_directions: set[tuple[str, str]] = set()
        for index, raw_direction in enumerate(authenticated_revision_directions):
            direction = canonicalize_language_pair(
                raw_direction,
                field=f"authenticated revision direction[{index}]",
            )
            if direction in seen_revision_directions:
                raise ValueError(
                    "authenticated_revision_directions contains a duplicate canonical edge: "
                    f"{direction!r}"
                )
            seen_revision_directions.add(direction)
            normalized_revision_directions.append(direction)
        export_revision_directions = tuple(normalized_revision_directions)
        if export_revision_directions != configured_revision_directions:
            raise ValueError(
                "authenticated_revision_directions must exactly match the resolved data "
                "revision graph"
            )
        if config.data.revision_examples is not bool(export_revision_directions):
            raise ValueError(
                "data.revision_examples must exactly reflect authenticated_revision_directions"
            )
    # Export role/ancestry is part of the training contract, not a late
    # serialization detail. Reject a missing or malformed 1.5 pipeline before
    # optimizer allocation or the first expensive training step.
    build_export_metadata(
        config.model,
        tokenizer_path=config.data.tokenizer_model,
        token_features_path=(
            config.data.tokenizer_features
            if config.model.experimental.morphoscript_enabled
            else None
        ),
        language_pairs=(
            config.data.configured_language_pairs() if export_translation_capable else None
        ),
        languages=export_languages,
        translation_directions=(
            config.data.configured_translation_directions() if export_translation_capable else None
        ),
        bidirectional=config.data.bidirectional,
        revision_directions=export_revision_directions,
        revision_trained=bool(export_revision_directions),
        release_name=export_release_name,
        translation_capable=export_translation_capable,
        pipeline_identity=pipeline_identity,
    )
    _fail_if_known_empty(train_loader, "training")
    _fail_if_known_empty(validation_loader, "validation")

    training = config.training
    budget = resolve_training_budget(train_loader, training)
    if training.warmup_steps > budget.max_optimizer_steps:
        raise ValueError(
            f"warmup_steps ({training.warmup_steps}) cannot exceed the resolved training "
            f"budget ({budget.max_optimizer_steps} optimizer steps)"
        )
    output_dir = Path(training.output_dir)
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    checkpoint_identity = build_training_checkpoint_identity(
        config,
        batch_sampler=batch_sampler,
        context=context,
        stage_name=stage_name,
        include_posttraining=objective is not None,
        pipeline_identity=pipeline_identity,
    )
    if training.resume_from:
        # Incompatible or corrupt retries must fail before mutating the live
        # model/optimizer or replacing the prior run's configuration evidence.
        preflight_error: Exception | None = None
        try:
            preflight_checkpoint_identity(
                training.resume_from,
                context,
                checkpoint_identity,
            )
        except Exception as error:
            preflight_error = error
        preflight_failure_scope = distributed_failure_scope(
            preflight_error is not None,
            context,
        )
        if preflight_failure_scope != "none":
            failure = RuntimeError(
                "checkpoint resume preflight failed on at least one distributed rank"
            )
            if preflight_error is not None:
                raise failure from preflight_error
            raise failure

    # Stage 1/4: prepare the optimizer, scheduler, and AMP scaler.
    announce("Stage 1/4: preparing AdamW and the learning-rate scheduler.", context)
    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(model, training.weight_decay),
        lr=training.learning_rate,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_eps,
        weight_decay=0.0,  # Each parameter group already specifies its decay.
        fused=context.device.type == "cuda",
    )
    scheduler = cosine_scheduler(
        optimizer,
        warmup_steps=training.warmup_steps,
        max_steps=budget.max_optimizer_steps,
        min_ratio=training.min_learning_rate_ratio,
    )
    scaler = _make_grad_scaler(training, context)
    # Update EMA shadows after each optimizer step and use them for evaluation
    # and export to stabilize translation quality.
    ema = EMAWeights(model, training.ema_decay) if training.ema_decay > 0 else None
    if ema is not None:
        announce(f"EMA weight averaging enabled (decay={training.ema_decay}).", context)
    configured_selection_metric = (
        "validation_reward" if objective is not None else training.sft_selection_metric.lower()
    )
    training_state: dict[str, Any] = {
        "best_validation_loss": float("inf"),
        "best_step": -1,
        "early_stopping_bad_evals": 0,
        "epoch": 0,
        "batch_in_epoch": 0,
    }

    # Stage 2/4: optionally resume from a checkpoint.
    if training.resume_from:
        announce(f"Stage 2/4: resuming from checkpoint {training.resume_from}.", context)
        start_step = load_checkpoint(
            training.resume_from,
            model,
            optimizer,
            scheduler,
            context,
            scaler=scaler if scaler.is_enabled() else None,
            training_state=training_state,
            ema=ema,
            expected_identity=checkpoint_identity,
        )
        announce(f"Resume complete; continuing from step {start_step}.", context)
        loaded_metric = training_state.get("configured_selection_metric")
        loaded_best_metric = training_state.get("best_selection_metric")
        loaded_best_step = int(training_state.get("best_step", -1))
        loaded_best_value = float(training_state.get("best_validation_loss", float("inf")))
        has_recorded_best = loaded_best_step >= 0 or math.isfinite(loaded_best_value)
        selection_metadata_matches = loaded_metric == configured_selection_metric and (
            not has_recorded_best or isinstance(loaded_best_metric, str)
        )
        if not selection_metadata_matches:
            previous = loaded_metric if loaded_metric is not None else "legacy/unknown"
            announce(
                "The checkpoint's best-selection metric is incompatible with the "
                f"current configuration. Resetting best and early-stopping history ({previous!r} -> "
                f"{configured_selection_metric!r}).",
                context,
            )
            training_state.update(
                {
                    "best_validation_loss": float("inf"),
                    "best_step": -1,
                    "early_stopping_bad_evals": 0,
                    "best_selection_metric": None,
                    "best_checkpoint_artifact_sha256": None,
                }
            )
        recorded_best_step = int(training_state.get("best_step", -1))
        recorded_best_value = float(training_state.get("best_validation_loss", float("inf")))
        recorded_best_artifact_sha256 = training_state.get("best_checkpoint_artifact_sha256")
        has_compatible_recorded_best = recorded_best_step >= 0 or math.isfinite(recorded_best_value)
        recorded_best_invalid = has_compatible_recorded_best and (
            recorded_best_step < 0 or not _is_sha256_digest(recorded_best_artifact_sha256)
        )
        if has_compatible_recorded_best and not recorded_best_invalid:
            try:
                # DCP candidate order and its small completion-marker binding are
                # chosen by rank 0. Every rank then authenticates the exact visible
                # inventory and preflights identity + recorded step inside one lease.
                with verified_checkpoint_generation_lease(
                    output_dir / "checkpoints" / "best",
                    context,
                    checkpoint_identity,
                    expected_artifact_sha256=recorded_best_artifact_sha256,
                    expected_step=recorded_best_step,
                ):
                    pass
            except Exception:
                recorded_best_invalid = True
        if recorded_best_invalid:
            announce(
                "The exact best artifact referenced by the resumed checkpoint could "
                "not be authenticated on every rank. The next validation will safely "
                "select a new best checkpoint.",
                context,
            )
            training_state.update(
                {
                    "best_validation_loss": float("inf"),
                    "best_step": -1,
                    "early_stopping_bad_evals": 0,
                    "best_selection_metric": None,
                    "best_checkpoint_artifact_sha256": None,
                }
            )
    else:
        announce("Stage 2/4: no resume checkpoint; starting a new run.", context)
    # A retry becomes this run generation only after its checkpoint has loaded
    # successfully. Until then, retain the previous resolved configuration.
    _publish_resolved_config(config, output_dir, context)
    training_state["configured_selection_metric"] = configured_selection_metric

    writer = _make_summary_writer(training, output_dir, context, start_step)
    step = start_step
    epoch = int(training_state.get("epoch", 0))
    batch_in_epoch = int(training_state.get("batch_in_epoch", 0))
    if batch_in_epoch < 0:
        raise ValueError("checkpoint batch_in_epoch must be non-negative")
    best_validation_loss = float(training_state.get("best_validation_loss", float("inf")))
    best_step = int(training_state.get("best_step", -1))
    raw_best_checkpoint_artifact_sha256 = training_state.get("best_checkpoint_artifact_sha256")
    best_checkpoint_artifact_sha256 = (
        raw_best_checkpoint_artifact_sha256
        if _is_sha256_digest(raw_best_checkpoint_artifact_sha256)
        else None
    )
    selected_checkpoint_source: str | None = None
    selected_checkpoint_artifact_sha256: str | None = None
    bad_evals = int(training_state.get("early_stopping_bad_evals", 0))
    loaded_best_selection_metric = training_state.get("best_selection_metric")
    best_selection_metric = (
        loaded_best_selection_metric if isinstance(loaded_best_selection_metric, str) else None
    )
    stopped_early = False
    # A list permits the nested validation function to record this condition
    # without rebinding a local boolean.
    reward_fallback_reported: list[bool] = []
    last_eval_step = -1
    last_train_loss: float | None = None
    micro_step = 0
    # SFT normalizes gradients by token count; MRT uses source-sentence count.
    accumulated_local_normalizer = torch.zeros((), device=context.device, dtype=torch.float64)
    # [loss sum, normalizer, auxiliary-loss sum, auxiliary normalizer, processed tokens]
    window = torch.zeros(5, device=context.device, dtype=torch.float64)
    objective_window: dict[str, torch.Tensor] = {}
    log_start = time.perf_counter()
    data_wait_seconds = 0.0
    steps_since_log = 0
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)

    def current_training_state() -> dict[str, Any]:
        return {
            "best_validation_loss": best_validation_loss,
            "best_step": best_step,
            "early_stopping_bad_evals": bad_evals,
            "configured_selection_metric": configured_selection_metric,
            "best_selection_metric": best_selection_metric,
            "best_checkpoint_artifact_sha256": best_checkpoint_artifact_sha256,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
        }

    def save(path: Path) -> None:
        """Save model, optimizer, scheduler, and progress state for resume."""
        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler,
            step,
            context,
            scaler=scaler if scaler.is_enabled() else None,
            training_state=current_training_state(),
            ema=ema,
            identity=checkpoint_identity,
        )

    def export_models(name: str) -> None:
        """Save an inference model under ``exports/<name>/``.

        During training, save only ``model_ema.pt`` when EMA is enabled, or
        ``model.pt`` otherwise. Resume checkpoints already contain raw weights,
        so another full-state export would be redundant. Slow quantization and
        Hugging Face conversion run once from the selected best checkpoint after
        all training stages finish.
        """
        token_features_path = (
            config.data.tokenizer_features
            if config.model.experimental.morphoscript_enabled
            else None
        )
        manifest = export_inference_models(
            output_dir / "exports" / name,
            model,
            config.model,
            context,
            step,
            ema=ema,
            tokenizer_path=config.data.tokenizer_model,
            token_features_path=token_features_path,
            language_pairs=(
                config.data.configured_language_pairs() if export_translation_capable else None
            ),
            languages=export_languages,
            translation_directions=(
                config.data.configured_translation_directions()
                if export_translation_capable
                else None
            ),
            bidirectional=config.data.bidirectional,
            revision_directions=export_revision_directions,
            revision_trained=bool(export_revision_directions),
            release_name=export_release_name,
            translation_capable=export_translation_capable,
            pipeline_identity=pipeline_identity,
        )
        if context.is_main and manifest is not None:
            successful = [
                format_name
                for format_name, entry in manifest["formats"].items()
                if entry.get("status") == "ok"
            ]
            failed = [
                f"{format_name}({entry.get('error_type', 'error')}: "
                f"{entry.get('message', 'unknown')})"
                for format_name, entry in manifest["formats"].items()
                if entry.get("status") != "ok"
            ]
            announce(
                f"Inference export complete: exports/{name} [{', '.join(successful)}]",
                context,
            )
            if failed:
                announce(
                    "Some intermediate formats failed to export; checkpointed training "
                    "continues: " + ", ".join(failed),
                    context,
                )

    def validate_and_update_early_stopping() -> bool:
        """Validate, update the best checkpoint, and decide whether to stop early.

        ``True`` means training should stop because improvement has stalled.
        """
        nonlocal best_validation_loss, best_step, bad_evals, last_eval_step
        nonlocal best_checkpoint_artifact_sha256, best_selection_metric
        announce(f"Validation starting at step {step}.", context)
        metrics = evaluate(
            model,
            validation_loader,
            context,
            training.eval_batches,
            precision=training.precision,
            show_progress=True,
            objective=objective,
            language_tags=language_tags,
        )
        if ema is not None:
            # Validate EMA weights separately. Best selection and early stopping
            # use their usually lower and more stable metric.
            with ema.swap(model):
                ema_metrics = evaluate(
                    model,
                    validation_loader,
                    context,
                    training.eval_batches,
                    precision=training.precision,
                    show_progress=True,
                    objective=objective,
                    language_tags=language_tags,
                )
            for name, value in ema_metrics.items():
                if name.startswith("validation_"):
                    metrics[f"validation_ema_{name.removeprefix('validation_')}"] = value
        if last_train_loss is not None and objective is None:
            # Validation loss minus training loss; larger values indicate overfitting.
            metrics["generalization_gap"] = float(metrics["validation_loss"]) - last_train_loss
        last_eval_step = step
        if context.is_main:
            summary = "Validation: loss={:.4f}, NLL={:.4f}, perplexity={:.2f}".format(
                metrics["validation_loss"],
                metrics.get("validation_nll", metrics["validation_loss"]),
                metrics["validation_perplexity"],
            )
            if "validation_macro_direction_nll" in metrics:
                summary += ", direction macro NLL={:.4f}".format(
                    metrics["validation_macro_direction_nll"]
                )
            if "validation_worst_direction_nll" in metrics:
                summary += ", worst-direction NLL={:.4f}".format(
                    metrics["validation_worst_direction_nll"]
                )
            if "validation_ema_loss" in metrics:
                summary += ", EMA loss={:.4f}".format(metrics["validation_ema_loss"])
            if "validation_reward" in metrics:
                summary += ", reward={:.4f}".format(metrics["validation_reward"])
            if "validation_ema_reward" in metrics:
                summary += ", EMA reward={:.4f}".format(metrics["validation_ema_reward"])
            announce(summary, context)
            tqdm.write(json.dumps({"step": step, **metrics}))
            if writer is not None:
                for name, value in metrics.items():
                    writer.add_scalar(f"validation/{name.removeprefix('validation_')}", value, step)

        # Rank 0 decides whether the metric improved, then broadcasts the result
        # so every rank saves and stops together. Post-training maximizes actual
        # generation reward, while SFT minimizes the configured NLL metric. Store
        # maximization metrics negated for compatibility with existing state.
        if objective is not None and "validation_reward" in metrics:
            configured = config.posttraining.selection_metric
            selection_value, selection_key, used_fallback = _select_posttraining_validation_metric(
                metrics,
                configured,
                prefer_ema=ema is not None,
            )
            if used_fallback and configured != "reward" and not reward_fallback_reported:
                announce(
                    f"Direction metadata required for posttraining.selection_metric={configured} "
                    "is unavailable. Falling back to mean reward.",
                    context,
                )
                reward_fallback_reported.append(True)
            candidate = -selection_value
            selection_name = _selection_metric_label(selection_key)
        else:
            candidate, selection_key, used_fallback = _select_sft_validation_metric(
                metrics,
                training.sft_selection_metric,
                prefer_ema=ema is not None,
            )
            selection_value = candidate
            selection_name = _selection_metric_label(selection_key)
            if used_fallback:
                announce(
                    f"SFT selection metric {training.sft_selection_metric!r} is unavailable; "
                    f"falling back to {selection_name}.",
                    context,
                )
        if best_selection_metric is not None and selection_key != best_selection_metric:
            announce(
                "The validation selection metric differs from the previous best record. "
                "Resetting the comparison baseline "
                f"({best_selection_metric!r} → {selection_key!r}).",
                context,
            )
            best_validation_loss = float("inf")
            best_step = -1
            bad_evals = 0
            best_checkpoint_artifact_sha256 = None
            best_selection_metric = None
        improved_here = candidate < best_validation_loss - training.early_stopping_min_delta
        improved = broadcast_bool(improved_here if context.is_main else False, context)
        if improved:
            best_validation_loss = candidate
            best_step = step
            bad_evals = 0
            best_selection_metric = selection_key
            announce(
                f"New best {selection_name} ({selection_value:.4f}); saving best checkpoint.",
                context,
            )
            best_checkpoint = output_dir / "checkpoints" / "best"
            best_checkpoint_artifact_sha256 = None
            save(best_checkpoint)
            with verified_checkpoint_generation_lease(
                best_checkpoint,
                context,
                checkpoint_identity,
                expected_step=best_step,
            ) as authenticated_best:
                if authenticated_best.source.resolve() != best_checkpoint.resolve():
                    raise RuntimeError(
                        "newly saved best checkpoint did not publish as the current generation"
                    )
                best_checkpoint_artifact_sha256 = authenticated_best.artifact_sha256
            # Persist the exact best-generation binding before any fallible
            # inference export. A failed export can then restart from latest
            # without repeating completed optimizer/evaluation work.
            save(output_dir / "checkpoints" / "latest")
            export_models("best")
        else:
            bad_evals += 1
            announce(
                f"No improvement ({bad_evals} consecutive evaluations; "
                f"patience {training.early_stopping_patience}).",
                context,
            )
        should_stop_here = (
            training.early_stopping_patience > 0 and bad_evals >= training.early_stopping_patience
        )
        should_stop = broadcast_bool(should_stop_here if context.is_main else False, context)
        if context.is_main and writer is not None:
            best_selection_value = (
                -best_validation_loss if objective is not None else best_validation_loss
            )
            writer.add_scalar("validation/best_selection_value", best_selection_value, step)
            writer.add_scalar("validation/early_stopping_bad_evals", bad_evals, step)
            writer.flush()
        return should_stop

    # Stage 3/4: training loop.
    target_description = (
        f"complete {budget.target_epochs} full dataset epochs"
        if budget.target_epochs is not None
        else f"explicit limit of {budget.max_optimizer_steps} optimizer steps"
    )
    announce(
        f"Stage 3/4: starting {stage_name} training (target: {target_description}; "
        f"current step: {start_step}; completed epochs: {epoch}).",
        context,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # The main progress bar advances once per optimizer update.
    progress = tqdm(
        total=budget.max_optimizer_steps,
        initial=start_step,
        desc="training",
        unit="step",
        dynamic_ncols=True,
        disable=not context.is_main,
    )
    try:
        while budget.should_continue(step=step, epoch=epoch) and not stopped_early:
            # Give the sampler a new deterministic shuffle order for every epoch.
            if hasattr(train_loader, "batch_sampler") and hasattr(
                train_loader.batch_sampler, "set_epoch"
            ):
                train_loader.batch_sampler.set_epoch(epoch)
            sampler_has_cursor = hasattr(train_loader, "batch_sampler") and hasattr(
                train_loader.batch_sampler,
                "set_start_batch",
            )
            if sampler_has_cursor:
                train_loader.batch_sampler.set_start_batch(batch_in_epoch)
            if hasattr(train_loader, "collate_fn") and hasattr(
                train_loader.collate_fn, "set_epoch"
            ):
                train_loader.collate_fn.set_epoch(epoch)
            batches_this_epoch = batch_in_epoch
            data_wait_started = time.perf_counter()
            # DataLoader iterator creation draws a worker seed from torch's global
            # RNG. Preserve the model RNG so a mid-epoch restart does not inject
            # one extra random draw into dropout or other stochastic layers.
            torch_rng_before_iterator = torch.get_rng_state()
            train_iterator = iter(train_loader)
            torch.set_rng_state(torch_rng_before_iterator)
            if batch_in_epoch and not sampler_has_cursor:
                # Generic Iterable compatibility. The project's sampler uses the
                # cursor above and therefore avoids fetching/collating skipped data.
                for _ in range(batch_in_epoch):
                    try:
                        next(train_iterator)
                    except StopIteration as error:
                        raise ValueError(
                            "checkpoint batch_in_epoch exceeds the training loader length"
                        ) from error
                # Skipping a generic iterator may use torch RNG in its collator.
                # The skipped batches happened before the checkpoint, so discard
                # those duplicate RNG draws.
                torch.set_rng_state(torch_rng_before_iterator)
            epoch_completed = True
            for batch in train_iterator:
                data_wait_seconds += time.perf_counter() - data_wait_started
                batches_this_epoch += 1
                batch_in_epoch += 1
                batch = move_to_device(batch, context.device)
                # Apply the final incomplete accumulation window in each epoch;
                # otherwise the last batches are read but their gradients are lost.
                is_epoch_last_batch = (
                    budget.batches_per_epoch is not None
                    and batches_this_epoch >= budget.batches_per_epoch
                )
                is_last_micro = (
                    micro_step + 1 >= training.gradient_accumulation_steps or is_epoch_last_batch
                )
                with maybe_no_sync(model, enabled=context.distributed and not is_last_micro):
                    with _autocast_context(training.precision, context.device):
                        if objective is None:
                            output = model(**batch)
                            loss_sum = (
                                output.lm_loss_sum
                                + output.auxiliary_loss * output.token_count.detach()
                            )
                            normalizer = output.token_count.detach()
                            processed_tokens = output.token_count.detach()
                            auxiliary_loss = output.auxiliary_loss.detach()
                            # Native optional modules expose their own diagnostics
                            # in addition to the combined auxiliary loss. Recording
                            # only non-None values keeps older/custom model outputs
                            # compatible while making A/B behavior observable.
                            objective_metrics = {
                                name: value
                                for name in (
                                    "register_loss",
                                    "register_unsupervised_rate",
                                    "alignment_loss",
                                    "coverage_loss",
                                    "uncertainty_loss",
                                    "evidence_budget_loss",
                                    "evidence_request_rate",
                                    "evidence_repair_gain_loss",
                                    "evidence_repair_gain",
                                    "candidate_refinement_loss",
                                    "candidate_refinement_gain",
                                    "candidate_refinement_steps",
                                    "semantic_parity_loss",
                                    "semantic_parity_score",
                                )
                                if (value := getattr(output, name, None)) is not None
                            }
                        else:
                            objective_output = objective(model, batch)
                            loss_sum = objective_output.loss_sum
                            normalizer = objective_output.normalizer.detach()
                            processed_tokens = objective_output.processed_tokens.detach()
                            auxiliary_loss = objective_output.auxiliary_loss.detach()
                            objective_metrics = objective_output.metrics
                        backward_loss = loss_sum * context.world_size
                    if scaler.is_enabled():
                        scaler.scale(backward_loss).backward()
                    else:
                        backward_loss.backward()

                micro_step += 1
                accumulated_local_normalizer += normalizer.double()
                window[0] += loss_sum.detach().double()
                window[1] += normalizer.double()
                window[2] += auxiliary_loss.double() * normalizer.double()
                window[3] += normalizer.double()
                window[4] += processed_tokens.double()
                for name, value in objective_metrics.items():
                    if name not in objective_window:
                        objective_window[name] = torch.zeros(
                            2, device=context.device, dtype=torch.float64
                        )
                    objective_window[name][0] += value.detach().double() * normalizer.double()
                    objective_window[name][1] += normalizer.double()
                if not is_last_micro:
                    data_wait_started = time.perf_counter()
                    continue  # Continue until the accumulation window is full.

                # Optimizer step: normalize gradients, clip, and update parameters.
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                # Aggregate the normalizer for the complete window across all ranks once.
                global_normalizer = accumulated_local_normalizer.clone()
                reduce_sum(global_normalizer, context)
                gradient_denominator = global_normalizer.clamp_min(1.0)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(gradient_denominator)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
                optimizer_updated = True
                if scaler.is_enabled():
                    # The scaler skips this update when fp16 overflow occurs.
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_updated = scaler.get_scale() >= old_scale
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_local_normalizer.zero_()
                micro_step = 0
                if not optimizer_updated:
                    data_wait_started = time.perf_counter()
                    continue  # Do not count an update skipped due to overflow.

                if ema is not None:
                    # Update EMA shadows only after a successful optimizer step.
                    ema.update(model)
                scheduler.step()
                step += 1
                steps_since_log += 1

                # Advance progress without synchronization. Converting loss and
                # gradient norm to Python scalars would force a CUDA sync, so the
                # postfix is refreshed only in the log_every block below.
                if context.is_main:
                    progress.update(1)

                # Aggregate all ranks and write JSON plus TensorBoard every log_every steps.
                if step % training.log_every == 0:
                    elapsed = max(time.perf_counter() - log_start, 1e-6)
                    reduced_window = window.clone()
                    reduce_sum(reduced_window, context)
                    timing = torch.tensor(
                        [data_wait_seconds, elapsed],
                        device=context.device,
                        dtype=torch.float64,
                    )
                    reduce_sum(timing, context)
                    mean_data_wait = timing[0] / context.world_size
                    mean_elapsed = timing[1] / context.world_size
                    records = {
                        "step": step,
                        "epoch": epoch,
                        "loss": (reduced_window[0] / reduced_window[1].clamp_min(1)).item(),
                        "auxiliary_loss": (
                            reduced_window[2] / reduced_window[3].clamp_min(1)
                        ).item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm),
                        "global_tokens_per_second": reduced_window[4].item() / elapsed,
                        "seconds_per_step": (mean_elapsed / max(steps_since_log, 1)).item(),
                        "data_wait_fraction": (
                            mean_data_wait / mean_elapsed.clamp_min(1e-6)
                        ).item(),
                    }
                    if context.device.type == "cuda":
                        memory = torch.tensor(
                            [
                                torch.cuda.memory_allocated(context.device),
                                torch.cuda.memory_reserved(context.device),
                                torch.cuda.max_memory_allocated(context.device),
                                torch.cuda.max_memory_reserved(context.device),
                            ],
                            device=context.device,
                            dtype=torch.float64,
                        )
                        reduce_max(memory, context)
                        (
                            records["cuda_allocated_gib"],
                            records["cuda_reserved_gib"],
                            records["cuda_peak_allocated_gib"],
                            records["cuda_peak_reserved_gib"],
                        ) = (value.item() / 2**30 for value in memory)
                    for name, values in objective_window.items():
                        reduced_values = values.clone()
                        reduce_sum(reduced_values, context)
                        records[name] = (reduced_values[0] / reduced_values[1].clamp_min(1)).item()
                    last_train_loss = float(records["loss"])
                    if context.is_main:
                        progress.set_postfix(
                            {
                                "loss": f"{records['loss']:.4f}",
                                "lr": f"{records['learning_rate']:.2e}",
                                "grad_norm": f"{records['grad_norm']:.2f}",
                                "epoch": epoch,
                            },
                            refresh=False,
                        )
                        tqdm.write(json.dumps(records))
                        if writer is not None:
                            for name, value in records.items():
                                if name not in {"step", "epoch"}:
                                    writer.add_scalar(f"train/{name}", value, step)
                    window.zero_()
                    for values in objective_window.values():
                        values.zero_()
                    log_start = time.perf_counter()
                    data_wait_seconds = 0.0
                    steps_since_log = 0
                    if context.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(context.device)

                # Explicit step-limited runs validate every eval_every steps. Formal
                # epoch-based runs validate only at completed-epoch boundaries so an
                # intermediate evaluation cannot consume patience within the same epoch.
                if not budget.epoch_limited and step % training.eval_every == 0:
                    stopped_early = validate_and_update_early_stopping()
                    if stopped_early:
                        announce(
                            f"Early stopping after {bad_evals} consecutive evaluations "
                            "without improvement.",
                            context,
                        )
                        epoch_completed = False
                        break

                # Save the latest resume checkpoint and inference export periodically.
                if step % training.save_every == 0:
                    announce(
                        f"Saving latest checkpoint at step {step}: checkpoints/latest.", context
                    )
                    save(output_dir / "checkpoints" / "latest")
                    export_models("latest")
                if not budget.epoch_limited and step >= budget.max_optimizer_steps:
                    epoch_completed = False
                    break
                data_wait_started = time.perf_counter()
            if batches_this_epoch == 0:
                raise ValueError("training loader produced no batches")
            if not stopped_early and epoch_completed:
                epoch += 1
                batch_in_epoch = 0
                if budget.epoch_limited:
                    should_stop = validate_and_update_early_stopping()
                    stopped_early = bool(
                        should_stop
                        and epoch >= training.early_stopping_min_epochs
                        and budget.target_epochs is not None
                        and epoch < budget.target_epochs
                    )
                    if stopped_early:
                        announce(
                            f"Early stopping after {epoch} completed epochs and "
                            f"{bad_evals} consecutive evaluations without improvement.",
                            context,
                        )

        # Stage 4/4: final validation and persistence.
        # Validate once more if the final step has not yet been evaluated.
        if last_eval_step != step:
            should_stop = validate_and_update_early_stopping()
            stopped_early = stopped_early or (
                should_stop
                and (
                    epoch < budget.target_epochs
                    if budget.target_epochs is not None
                    else step < budget.max_optimizer_steps
                )
            )
        announce(
            "Stage 4/4: saving final model state "
            "(checkpoints/final + checkpoints/latest + exports/latest)",
            context,
        )
        save(output_dir / "checkpoints" / "final")
        save(output_dir / "checkpoints" / "latest")
        export_models("latest")
        best_checkpoint = output_dir / "checkpoints" / "best"
        if (
            best_step < 0
            or best_selection_metric is None
            or best_checkpoint_artifact_sha256 is None
        ):
            raise RuntimeError(
                "training produced no finite validation selection metric or authenticated "
                "best checkpoint; refusing to publish unbound final weights"
            )
        # Do not infer safety from rank 0's `.metadata` existence. Rank 0 binds
        # an exact current/previous generation, every rank verifies its own bytes,
        # and the full load remains inside that one-shot lease.
        with verified_checkpoint_generation_lease(
            best_checkpoint,
            context,
            checkpoint_identity,
            expected_artifact_sha256=best_checkpoint_artifact_sha256,
            expected_step=best_step,
        ) as authenticated_best:
            best_step = load_checkpoint(
                authenticated_best.source,
                model,
                optimizer,
                scheduler,
                context,
                scaler=scaler if scaler.is_enabled() else None,
                ema=ema,
                expected_identity=checkpoint_identity,
            )
            if best_step != authenticated_best.step:
                raise RuntimeError("best checkpoint step changed after authenticated preflight")
            selected_checkpoint_source = str(authenticated_best.source)
            selected_checkpoint_artifact_sha256 = authenticated_best.artifact_sha256
        selected_weights = "EMA" if ema is not None else "raw"
        if ema is not None:
            ema.copy_to(model)
        announce(
            f"Restored best checkpoint for the next stage (step {best_step}, "
            f"{selected_weights} weights).",
            context,
        )
        if objective is not None:
            final_selection_name = "composite generation reward"
            final_selection_value = -best_validation_loss
        else:
            final_metric_key = best_selection_metric or f"validation_{configured_selection_metric}"
            final_selection_name = _selection_metric_label(final_metric_key)
            final_selection_value = best_validation_loss
        announce(
            f"Training complete at step {step}; best {final_selection_name}: "
            f"{final_selection_value:.4f}" + (" (early stopping)" if stopped_early else ""),
            context,
        )
    finally:
        progress.close()
        if writer is not None:
            writer.close()

    result: dict[str, float | int | bool | str] = {
        "step": step,
        "best_step": best_step,
        "selected_step": best_step,
        "epoch": epoch,
        # Legacy key retained for checkpoint/API compatibility. For SFT this is
        # the selected minimizing metric; for reward objectives it is negated.
        "best_validation_loss": best_validation_loss,
        "configured_selection_metric": configured_selection_metric,
        "best_selection_metric": best_selection_metric or configured_selection_metric,
        "best_selection_value": (
            -best_validation_loss if objective is not None else best_validation_loss
        ),
        "early_stopping_bad_evals": bad_evals,
        "stopped_early": stopped_early,
        "selected_checkpoint_source": selected_checkpoint_source,
        "selected_checkpoint_artifact_sha256": selected_checkpoint_artifact_sha256,
    }
    if objective is not None:
        result["best_validation_reward"] = -best_validation_loss
    return result
