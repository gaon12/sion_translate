"""sion_translate 학습 루프.

이 파일의 큰 흐름은 다음과 같습니다.

1. optimizer(AdamW) / scheduler(warmup + cosine) / AMP scaler 준비
2. (요청 시) 체크포인트에서 학습 상태 복원
3. 학습 루프: micro-batch → gradient accumulation → optimizer step
   - 매 step 마다 tqdm progress bar 에 loss, LR, grad_norm 등을 표시
   - ``log_every`` step 마다 JSON 한 줄 로그 + TensorBoard 기록
4. ``eval_every`` step 마다 검증 → best 갱신 판단 + early stopping
5. 저장 정책 (각 시점마다 두 종류를 남깁니다)
   - checkpoints/best, checkpoints/latest, checkpoints/final
     : 학습 재개용 (optimizer 상태 포함, 용량 큼)
   - exports/best, exports/latest
     : 추론용. 일반(FP32) ``model.pt`` 와 INT8 양자화 ``model_int8.pt`` 둘 다 저장
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn
from tqdm.auto import tqdm

from sion_translate.config import AppConfig, TrainingConfig

from .checkpoint import load_checkpoint, save_checkpoint
from .distributed import (
    DistributedContext,
    broadcast_bool,
    maybe_no_sync,
    reduce_max,
    reduce_sum,
)
from .ema import EMAWeights
from .export import export_inference_models
from .objectives import ObjectiveOutput


def announce(message: str, context: DistributedContext) -> None:
    """현재 진행 단계를 사람이 읽기 좋은 텍스트로 출력합니다.

    ``tqdm.write`` 를 쓰면 progress bar 를 깨뜨리지 않고 그 위에
    한 줄을 출력할 수 있습니다. rank 0(main)에서만 출력합니다.
    """
    if context.is_main:
        tqdm.write(f"[sion] {message}")


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """collator 가 만든 batch(텐서 dict)를 학습 장치(GPU/CPU)로 옮깁니다."""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    max_steps: int,
    min_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """학습률 스케줄: 처음 ``warmup_steps`` 동안 0→최대로 선형 증가(warmup),
    그 뒤에는 cosine 곡선을 따라 최대치의 ``min_ratio`` 배까지 서서히 감소합니다."""

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def build_optimizer_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW weight decay 를 '행렬 형태의 가중치'에만 적용하도록 파라미터를 두 그룹으로 나눕니다.

    norm 가중치, bias, 1차원 게이트 같은 파라미터에 decay 를 걸면
    학습이 불안정해질 수 있어 관례적으로 제외합니다.
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
    """AMP(자동 혼합 정밀도) 컨텍스트. CUDA + bf16/fp16 일 때만 활성화됩니다."""
    if device.type != "cuda" or precision.lower() == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision.lower() == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_grad_scaler(training: TrainingConfig, context: DistributedContext):
    """fp16 학습에서 gradient underflow 를 막아 주는 GradScaler 를 만듭니다.

    bf16/fp32 에서는 비활성 상태의 scaler 가 반환되어 아무 일도 하지 않습니다.
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
    """TensorBoard 기록기. rank 0 에서만 만들고, 재개 시 이후 step 기록을 정리(purge)합니다."""
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
    """길이를 알 수 있는 loader 가 비어 있으면 학습 시작 전에 바로 실패시킵니다."""
    try:
        length = len(loader)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return
    if length == 0:
        raise ValueError(f"{name} loader is empty")


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
) -> dict[str, float]:
    """검증 데이터로 CE와 선택적 생성 품질 보상을 계산합니다 (gradient 없음).

    - loss 는 '토큰당 평균'으로 계산합니다: 배치별 평균을 다시 평균하면
      짧은 배치가 과대평가되므로, 합계를 모았다가 마지막에 한 번 나눕니다.
    - 분산 학습에서는 모든 rank 의 합계를 all-reduce 로 모아 전체 평균을 냅니다.
    """
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=context.device, dtype=torch.float64)
    token_count = torch.zeros((), device=context.device, dtype=torch.float64)
    aux_sum = torch.zeros((), device=context.device, dtype=torch.float64)
    objective_sums: dict[str, torch.Tensor] = {}
    objective_count = torch.zeros((), device=context.device, dtype=torch.float64)
    batches = 0

    # 검증에도 작은 progress bar 를 표시합니다 (rank 0 전용, 끝나면 지워짐).
    progress = None
    if show_progress and context.is_main:
        try:
            total = min(len(loader), max_batches)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            total = max_batches
        progress = tqdm(total=total, desc="검증", unit="batch", leave=False, dynamic_ncols=True)

    try:
        for batch in loader:
            batch = move_to_device(batch, context.device)
            with _autocast_context(precision, context.device):
                output = model(**batch)
            loss_sum += output.lm_loss_sum.detach().double()
            token_count += output.token_count.detach().double()
            aux_sum += output.auxiliary_loss.detach().double()
            validation_metrics = getattr(objective, "validation_metrics", None)
            if validation_metrics is not None:
                generated_metrics = validation_metrics(model, batch)
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
    reduce_sum(aux_sum, context)
    reduce_sum(objective_count, context)
    for value in objective_sums.values():
        reduce_sum(value, context)
    batch_tensor = torch.tensor(float(batches), device=context.device, dtype=torch.float64)
    reduce_sum(batch_tensor, context)
    model.train(was_training)
    mean_loss = (loss_sum / token_count.clamp_min(1)).item()
    metrics = {
        "validation_loss": mean_loss,
        # exp(loss) 가 너무 커져 overflow 하지 않도록 loss 를 20으로 제한합니다.
        "validation_perplexity": math.exp(min(mean_loss, 20.0)),
        "validation_auxiliary_loss": (aux_sum / batch_tensor.clamp_min(1)).item(),
        "validation_tokens": token_count.item(),
    }
    if objective_sums:
        denominator = objective_count.clamp_min(1)
        metrics.update(
            {
                f"validation_{name}": (value / denominator).item()
                for name, value in objective_sums.items()
            }
        )
    return metrics


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
) -> dict[str, float | int | bool]:
    """sion_translate 학습의 본체. 반환값은 마지막 step/epoch 과 best 검증 loss 요약입니다."""
    config.validate()
    _fail_if_known_empty(train_loader, "training")
    _fail_if_known_empty(validation_loader, "validation")

    training = config.training
    output_dir = Path(training.output_dir)
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        # 이 run 이 정확히 어떤 설정으로 돌았는지 나중에 확인할 수 있도록 저장합니다.
        with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)

    # ── 단계 1/4: optimizer · scheduler · AMP scaler 준비 ─────────────────
    announce("단계 1/4: optimizer(AdamW)와 학습률 스케줄러를 준비합니다.", context)
    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(model, training.weight_decay),
        lr=training.learning_rate,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_eps,
        weight_decay=0.0,  # decay 는 param group 별로 이미 지정했으므로 여기서는 0
        fused=context.device.type == "cuda",
    )
    scheduler = cosine_scheduler(
        optimizer,
        warmup_steps=training.warmup_steps,
        max_steps=training.max_steps,
        min_ratio=training.min_learning_rate_ratio,
    )
    scaler = _make_grad_scaler(training, context)
    # EMA(가중치 지수이동평균): 매 step 뒤 shadow 가중치를 갱신해 두었다가
    # 평가·내보내기에 사용합니다. 번역 품질을 안정적으로 올려 주는 기법입니다.
    ema = EMAWeights(model, training.ema_decay) if training.ema_decay > 0 else None
    if ema is not None:
        announce(f"EMA 가중치 평균 활성화 (decay={training.ema_decay})", context)
    training_state: dict[str, Any] = {
        "best_validation_loss": float("inf"),
        "early_stopping_bad_evals": 0,
        "epoch": 0,
    }

    # ── 단계 2/4: (선택) 체크포인트에서 재개 ──────────────────────────────
    if training.resume_from:
        announce(f"단계 2/4: 체크포인트에서 학습을 재개합니다 → {training.resume_from}", context)
        start_step = load_checkpoint(
            training.resume_from,
            model,
            optimizer,
            scheduler,
            context,
            scaler=scaler if scaler.is_enabled() else None,
            training_state=training_state,
            ema=ema,
        )
        announce(f"재개 완료: step {start_step} 부터 다시 시작합니다.", context)
    else:
        announce("단계 2/4: 재개할 체크포인트가 없어 처음부터 학습합니다.", context)

    writer = _make_summary_writer(training, output_dir, context, start_step)
    step = start_step
    epoch = int(training_state.get("epoch", 0))
    best_validation_loss = float(training_state.get("best_validation_loss", float("inf")))
    bad_evals = int(training_state.get("early_stopping_bad_evals", 0))
    stopped_early = False
    last_eval_step = -1
    last_train_loss: float | None = None
    micro_step = 0
    # SFT는 token 수, MRT는 source 문장 수를 gradient 정규화 분모로 씁니다.
    accumulated_local_normalizer = torch.zeros((), device=context.device, dtype=torch.float64)
    # [loss 합, 정규화 분모, 보조 loss 합, 보조 분모, 실제 처리 token 수]
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
            "early_stopping_bad_evals": bad_evals,
            "epoch": epoch,
        }

    def save(path: Path) -> None:
        """학습 재개용 체크포인트(모델 + optimizer + scheduler + 진행 상태)를 저장합니다."""
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
        )

    def export_models(name: str) -> None:
        """추론용 모델을 exports/<name>/ 에 저장합니다.

        일반(FP32) model.pt, (EMA 활성 시) model_ema.pt, INT8 양자화
        model_int8.pt 세 가지가 함께 저장됩니다. 분산 학습에서는 가중치
        수집이 집단 통신이므로 모든 rank 가 함께 호출합니다.
        """
        export_inference_models(
            output_dir / "exports" / name,
            model,
            config.model,
            context,
            step,
            ema=ema,
        )
        saved = (
            "model.pt + model_int8.pt" if ema is None else "model.pt + model_ema.pt + model_int8.pt"
        )
        announce(f"추론용 모델 저장 완료: exports/{name}/{saved}", context)

    def validate_and_update_early_stopping() -> bool:
        """검증을 수행하고 best 갱신 여부와 early stopping 여부를 결정합니다.

        반환값이 True 면 '더 이상 개선이 없어 학습을 멈춰야 한다'는 뜻입니다.
        """
        nonlocal best_validation_loss, bad_evals, last_eval_step
        announce(f"검증 시작 (step {step})", context)
        metrics = evaluate(
            model,
            validation_loader,
            context,
            training.eval_batches,
            precision=training.precision,
            show_progress=True,
            objective=objective,
        )
        if ema is not None:
            # EMA 가중치로도 한 번 더 검증합니다. best 선택과 early stopping 은
            # EMA 손실을 기준으로 합니다 (보통 원본보다 낮고 안정적입니다).
            with ema.swap(model):
                ema_metrics = evaluate(
                    model,
                    validation_loader,
                    context,
                    training.eval_batches,
                    precision=training.precision,
                    show_progress=True,
                    objective=objective,
                )
            for name, value in ema_metrics.items():
                if name.startswith("validation_"):
                    metrics[f"validation_ema_{name.removeprefix('validation_')}"] = value
        if last_train_loss is not None and objective is None:
            # 검증 loss - 학습 loss. 값이 커질수록 과적합 신호입니다.
            metrics["generalization_gap"] = float(metrics["validation_loss"]) - last_train_loss
        last_eval_step = step
        if context.is_main:
            summary = "검증 결과: loss={:.4f}, perplexity={:.2f}".format(
                metrics["validation_loss"], metrics["validation_perplexity"]
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

        # best 갱신 판단은 rank 0 에서만 하고, 그 결과를 모든 rank 에 방송해
        # 전 rank 가 같은 시점에 저장/종료하도록 맞춥니다.
        # 사후학습은 실제 생성 reward를 최대화하고, SFT는 CE loss를 최소화합니다.
        # 기존 체크포인트 상태와 호환하기 위해 최대화 지표는 음수로 저장합니다.
        if objective is not None and "validation_reward" in metrics:
            selection_value = float(
                metrics.get("validation_ema_reward", metrics["validation_reward"])
            )
            candidate = -selection_value
            selection_name = "생성 복합 reward"
        else:
            candidate = float(metrics.get("validation_ema_loss", metrics["validation_loss"]))
            selection_value = candidate
            selection_name = "검증 loss"
        improved_here = candidate < best_validation_loss - training.early_stopping_min_delta
        improved = broadcast_bool(improved_here if context.is_main else False, context)
        if improved:
            best_validation_loss = candidate
            bad_evals = 0
            announce(
                f"{selection_name} 최고 기록 갱신 ({selection_value:.4f}) → best 저장",
                context,
            )
            save(output_dir / "checkpoints" / "best")
            export_models("best")
        else:
            bad_evals += 1
            announce(
                f"개선 없음 (연속 {bad_evals}회 / 허용 {training.early_stopping_patience}회)",
                context,
            )
        should_stop_here = (
            training.early_stopping_patience > 0 and bad_evals >= training.early_stopping_patience
        )
        should_stop = broadcast_bool(should_stop_here if context.is_main else False, context)
        if context.is_main and writer is not None:
            if objective is not None:
                writer.add_scalar("validation/best_reward", -best_validation_loss, step)
            else:
                writer.add_scalar("validation/best_loss", best_validation_loss, step)
            writer.add_scalar("validation/early_stopping_bad_evals", bad_evals, step)
            writer.flush()
        return should_stop

    # ── 단계 3/4: 학습 루프 ───────────────────────────────────────────────
    announce(
        f"단계 3/4: {stage_name} 학습 시작 (목표 {training.max_steps} step, "
        f"현재 step {start_step}, epoch {epoch})",
        context,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # 전체 학습 진행률 bar. optimizer step 단위로 1씩 증가합니다.
    progress = tqdm(
        total=training.max_steps,
        initial=start_step,
        desc="학습",
        unit="step",
        dynamic_ncols=True,
        disable=not context.is_main,
    )
    try:
        while step < training.max_steps and not stopped_early:
            # epoch 마다 sampler 의 셔플 순서를 바꿔 같은 배치 순서가 반복되지 않게 합니다.
            if hasattr(train_loader, "batch_sampler") and hasattr(
                train_loader.batch_sampler, "set_epoch"
            ):
                train_loader.batch_sampler.set_epoch(epoch)
            if hasattr(train_loader, "collate_fn") and hasattr(
                train_loader.collate_fn, "set_epoch"
            ):
                train_loader.collate_fn.set_epoch(epoch)
            batches_this_epoch = 0
            data_wait_started = time.perf_counter()
            for batch in train_loader:
                data_wait_seconds += time.perf_counter() - data_wait_started
                batches_this_epoch += 1
                batch = move_to_device(batch, context.device)
                # accumulation 창의 마지막 micro-batch 에서만 gradient 를 동기화합니다.
                is_last_micro = (micro_step + 1) % training.gradient_accumulation_steps == 0
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
                            objective_metrics: dict[str, torch.Tensor] = {}
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
                    continue  # accumulation 창이 아직 안 찼으면 다음 micro-batch 로

                # ── optimizer step: gradient 정규화 → clip → 파라미터 갱신 ──
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                # 창 전체(모든 rank)의 정규화 분모를 여기서 한 번만 모읍니다.
                global_normalizer = accumulated_local_normalizer.clone()
                reduce_sum(global_normalizer, context)
                gradient_denominator = global_normalizer.clamp_min(1.0)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(gradient_denominator)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
                optimizer_updated = True
                if scaler.is_enabled():
                    # fp16 overflow 가 발생하면 scaler 가 이 step 을 건너뜁니다.
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_updated = scaler.get_scale() >= old_scale
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_local_normalizer.zero_()
                if not optimizer_updated:
                    data_wait_started = time.perf_counter()
                    continue  # overflow 로 건너뛴 step 은 세지 않습니다

                if ema is not None:
                    # 성공한 optimizer step 뒤에만 EMA shadow 를 갱신합니다.
                    ema.update(model)
                scheduler.step()
                step += 1
                steps_since_log += 1

                # 진행률 자체는 동기화 없이 매 step 갱신합니다. loss/grad_norm을
                # 매번 Python scalar로 바꾸면 CUDA가 강제 동기화되므로, postfix는
                # 아래 log_every 구간에서만 갱신합니다.
                if context.is_main:
                    progress.update(1)

                # log_every step 마다: 전 rank 합산 통계를 JSON + TensorBoard 로 기록
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

                # eval_every step 마다: 검증 + best 저장 + early stopping 판단
                if step % training.eval_every == 0:
                    stopped_early = validate_and_update_early_stopping()
                    if stopped_early:
                        announce(
                            f"Early stopping: {bad_evals}회 연속 개선이 없어 학습을 종료합니다.",
                            context,
                        )
                        break

                # save_every step 마다: 최신(latest) 체크포인트 + 추론용 모델 저장
                if step % training.save_every == 0:
                    announce(f"최신 체크포인트 저장: checkpoints/latest (step {step})", context)
                    save(output_dir / "checkpoints" / "latest")
                    export_models("latest")
                if step >= training.max_steps:
                    break
                data_wait_started = time.perf_counter()
            if batches_this_epoch == 0:
                raise ValueError("training loader produced no batches")
            if not stopped_early:
                epoch += 1

        # ── 단계 4/4: 마무리 저장 ─────────────────────────────────────────
        # 마지막 step 에서 검증을 아직 안 했다면 한 번 더 수행합니다.
        if last_eval_step != step:
            should_stop = validate_and_update_early_stopping()
            stopped_early = stopped_early or (should_stop and step < training.max_steps)
        announce(
            "단계 4/4: 종료 시점 모델을 저장합니다 "
            "(checkpoints/final + checkpoints/latest + exports/latest)",
            context,
        )
        save(output_dir / "checkpoints" / "final")
        save(output_dir / "checkpoints" / "latest")
        export_models("latest")
        announce(
            f"학습 종료: step {step}, best "
            + (
                f"생성 복합 reward {-best_validation_loss:.4f}"
                if objective is not None
                else f"검증 loss {best_validation_loss:.4f}"
            )
            + (" (early stopping)" if stopped_early else ""),
            context,
        )
    finally:
        progress.close()
        if writer is not None:
            writer.close()

    result: dict[str, float | int | bool] = {
        "step": step,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "early_stopping_bad_evals": bad_evals,
        "stopped_early": stopped_early,
    }
    if objective is not None:
        result["best_validation_reward"] = -best_validation_loss
    return result
