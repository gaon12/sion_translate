"""학습 재개용 체크포인트 저장/복원.

여기서 저장하는 체크포인트는 '학습을 이어서 하기 위한' 것으로,
모델 가중치 외에 optimizer / scheduler / (fp16이면) scaler / 진행 상태
(best loss, early-stopping 카운터, epoch)까지 전부 포함합니다.

번역(추론)에만 쓸 가벼운 저장본은 ``sion_translate.training.export`` 가 따로 만듭니다.

저장 형식은 학습 방식에 따라 둘로 나뉩니다.
- 단일 프로세스: ``checkpoint.pt`` 파일 하나에 torch.save 로 저장
- 분산 학습(FSDP2/DDP): torch.distributed.checkpoint(DCP) 형식의 디렉터리.
  가중치가 rank 별로 조각나 있어도 그대로 저장/복원할 수 있습니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .distributed import DistributedContext, barrier


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    context: DistributedContext,
    *,
    scaler: Any | None = None,
    training_state: dict[str, Any] | None = None,
    ema: Any | None = None,
) -> None:
    """현재 학습 상태 전체를 ``path`` 디렉터리에 저장합니다.

    분산 학습에서는 모든 rank 가 함께 호출해야 합니다(집단 통신 발생).
    """
    path = Path(path)
    if context.is_main:
        path.mkdir(parents=True, exist_ok=True)
    # 디렉터리 생성이 끝나기 전에 다른 rank 가 쓰기 시작하지 않도록 동기화합니다.
    barrier(context)
    state: dict[str, Any] = {
        "scheduler": scheduler.state_dict(),
        "step": step,
        "training_state": dict(training_state or {}),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if ema is not None:
        # EMA shadow 가중치도 함께 저장해 재개 시 평균 이력이 끊기지 않게 합니다.
        state["ema"] = ema.state_dict()

    if context.distributed:
        # 분산 학습: DCP 가 rank 별 조각(shard)을 병렬로 저장합니다.
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_state, optimizer_state = get_state_dict(model, optimizer)
        state["model"] = model_state
        state["optimizer"] = optimizer_state
        dcp.save(state, checkpoint_id=path)
    elif context.is_main:
        # 단일 프로세스: 파일 하나로 충분합니다.
        state["model"] = model.state_dict()
        state["optimizer"] = optimizer.state_dict()
        torch.save(
            state,
            path / "checkpoint.pt",
        )
    # 저장이 완전히 끝난 뒤에만 모든 rank 가 다음 단계로 진행합니다.
    barrier(context)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    context: DistributedContext,
    *,
    scaler: Any | None = None,
    training_state: dict[str, Any] | None = None,
    ema: Any | None = None,
) -> int:
    """``path`` 의 체크포인트를 읽어 학습 상태를 복원하고, 재개할 step 을 반환합니다.

    ``training_state`` dict 를 넘기면 best loss / early-stopping 카운터 /
    epoch 같은 진행 상태가 그 안에 채워집니다.
    """
    path = Path(path)
    if context.distributed:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        model_state, optimizer_state = get_state_dict(model, optimizer)
        state: dict[str, Any] = {
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": scheduler.state_dict(),
            "step": 0,
            "training_state": dict(training_state or {}),
        }
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        if ema is not None:
            # DCP 는 여기 넣어 둔 텐서 '안으로' 값을 읽어들이므로,
            # 현재 shadow 텐서를 로드 대상으로 미리 등록합니다.
            state["ema"] = ema.state_dict()
        # 체크포인트가 불완전하거나 구조가 맞지 않으면 여기서 바로 실패합니다.
        # 일부 파라미터가 초기값인 채로 조용히 재개되는 것이 훨씬 위험하기 때문입니다.
        dcp.load(state, checkpoint_id=path)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema"):
            ema.load_state_dict(state["ema"])
        if training_state is not None:
            training_state.update(state.get("training_state", {}))
        return int(state["step"])

    # 단일 프로세스: torch.load 로 한 번에 복원합니다.
    state = torch.load(path / "checkpoint.pt", map_location=context.device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if ema is not None and state.get("ema"):
        ema.load_state_dict(state["ema"])
    if training_state is not None:
        training_state.update(state.get("training_state", {}))
    return int(state["step"])
