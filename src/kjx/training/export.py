"""추론(번역) 전용 모델 내보내기 도구.

체크포인트(checkpoint)와 내보내기(export)는 목적이 다릅니다.

- 체크포인트(`checkpoints/…`): 학습을 이어서 재개하기 위한 저장본.
  optimizer, scheduler, scaler 상태까지 전부 들어 있어 용량이 큽니다.
- 내보내기(`exports/…`): 번역(추론)에만 쓰는 가벼운 저장본.
  같은 시점의 모델을 두 가지 형태로 저장합니다.

    exports/<이름>/model.pt       ← 일반(FP32) 가중치. GPU/CPU 어디서나 로드 가능.
    exports/<이름>/model_int8.pt  ← INT8 동적 양자화 모델. CPU 추론용, 용량 약 1/3.

<이름>은 "best"(검증 손실이 가장 좋았던 시점) 또는 "latest"(가장 최근 저장 시점)입니다.

불러오는 방법 (예시):

    # 일반 모델
    payload = torch.load("exports/best/model.pt", map_location="cpu", weights_only=False)
    config = ModelConfig(**{k: v for k, v in payload["model_config"].items() if k != "experimental"},
                         experimental=ExperimentalConfig(**payload["model_config"]["experimental"]))
    model = KJXForConditionalGeneration(config, pad_id=payload["pad_id"])
    model.load_state_dict(payload["model"])

    # 양자화 모델 (모듈 전체가 저장되어 있어 바로 사용 가능)
    payload = torch.load("exports/best/model_int8.pt", map_location="cpu", weights_only=False)
    model = payload["model"]
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kjx.config import ExperimentalConfig, ModelConfig
from kjx.model import KJXForConditionalGeneration

from .distributed import DistributedContext, barrier


def load_exported_model(path: str | Path) -> tuple[nn.Module, ModelConfig, int]:
    """내보낸 모델 파일(model.pt / model_ema.pt / model_int8.pt)을 불러옵니다.

    반환값: (추론 준비가 끝난 모델(eval 모드), 모델 설정, pad_id)
    양자화본은 모듈 전체가 저장되어 있어 그대로 쓰고, 일반/EMA 저장본은
    함께 저장된 설정으로 모델을 재조립한 뒤 가중치를 넣습니다.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_config = dict(payload["model_config"])
    config = ModelConfig(
        **{key: value for key, value in raw_config.items() if key != "experimental"},
        experimental=ExperimentalConfig(**raw_config["experimental"]),
    )
    pad_id = int(payload["pad_id"])
    stored = payload["model"]
    if isinstance(stored, nn.Module):
        model = stored  # INT8 양자화본: 모듈 전체가 저장되어 있음
    else:
        config.gradient_checkpointing = False  # 추론 전용
        with torch.random.fork_rng(devices=[]):
            model = KJXForConditionalGeneration(config, pad_id=pad_id)
        model.load_state_dict(stored)
    model.eval()
    return model, config, pad_id


def unwrap_model(model: nn.Module) -> nn.Module:
    """torch.compile·DDP 등 학습용 래퍼(wrapper)를 벗겨 원본 모델을 돌려줍니다.

    - torch.compile 로 감싼 모델은 원본이 ``_orig_mod`` 속성에 들어 있습니다.
    - DistributedDataParallel 로 감싼 모델은 원본이 ``module`` 속성에 들어 있습니다.
    - FSDP2(fully_shard)는 모델을 감싸지 않으므로 그대로 반환됩니다.
    """
    unwrapped = model
    while True:
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
            continue
        if isinstance(getattr(unwrapped, "module", None), nn.Module):
            unwrapped = unwrapped.module
            continue
        return unwrapped


def gather_full_state_dict(
    model: nn.Module, context: DistributedContext
) -> dict[str, torch.Tensor]:
    """어떤 학습 방식(단일 GPU/DDP/FSDP2)이든 '완전한 CPU 가중치 사전'을 만듭니다.

    주의: 분산 학습 중에는 가중치가 여러 GPU에 조각(shard)나 있으므로,
    이 함수는 **모든 rank가 함께 호출**해야 합니다(집단 통신이 발생합니다).
    """
    if context.distributed:
        # FSDP2/DDP 어느 쪽이든 PyTorch 공식 API가 조각난 가중치를
        # 완전한 형태로 모아 CPU로 내려줍니다.
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        return get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    # 단일 프로세스: 래퍼만 벗기고 CPU로 복사하면 됩니다.
    base = unwrap_model(model)
    return {
        name: tensor.detach().to("cpu", copy=True)
        for name, tensor in base.state_dict().items()
    }


def _quantize_int8(
    model_config: ModelConfig,
    state_dict: dict[str, torch.Tensor],
    pad_id: int,
) -> nn.Module:
    """FP32 가중치로 CPU 모델을 다시 만든 뒤 INT8 동적 양자화를 적용합니다.

    '동적 양자화'는 nn.Linear 의 가중치를 8비트 정수로 저장해 두었다가
    추론 시점에만 되돌려 계산하는 방식입니다. 별도의 보정(calibration)
    데이터가 필요 없어서 학습 중에도 안전하게 만들 수 있습니다.

    참고: 임베딩(token_embedding)은 FP32 그대로 남습니다. 이 모델은
    임베딩을 출력층과 공유(tie)하므로, 마지막 로짓 계산은 양자화 이득을
    받지 않습니다. 그래도 encoder/decoder 내부의 모든 Linear 가
    양자화되므로 용량과 CPU 추론 속도에서 큰 이득이 있습니다.
    """
    # 학습 중인 모델을 건드리지 않도록 설정을 복사해서 새 CPU 모델을 만듭니다.
    config_copy = copy.deepcopy(model_config)
    config_copy.gradient_checkpointing = False  # 추론 전용이므로 불필요
    # 모델 생성 시 가중치 초기화가 난수를 소모하는데, 그대로 두면 '저장 시점'에
    # 따라 이후 학습의 dropout/denoising 난수 흐름이 달라져 재현성이 깨집니다.
    # fork_rng 로 전역 난수 상태를 격리해 내보내기가 학습에 영향을 주지 않게 합니다.
    with torch.random.fork_rng(devices=[]):
        cpu_model = KJXForConditionalGeneration(config_copy, pad_id=pad_id)
    cpu_model.load_state_dict(state_dict)
    cpu_model.eval()
    return torch.ao.quantization.quantize_dynamic(
        cpu_model, {nn.Linear}, dtype=torch.qint8
    )


def export_inference_models(
    directory: str | Path,
    model: nn.Module,
    model_config: ModelConfig,
    context: DistributedContext,
    step: int,
    *,
    ema: Any | None = None,
) -> None:
    """현재 모델을 추론용으로 저장합니다.

    저장 파일:
    - ``model.pt``      : 일반(FP32) 원본 가중치
    - ``model_ema.pt``  : EMA(지수이동평균) 가중치 — 보통 원본보다 번역 품질이
      좋아 실제 배포/평가에는 이쪽을 권장합니다 (EMA 활성 시에만 저장)
    - ``model_int8.pt`` : INT8 동적 양자화 (EMA 가 있으면 EMA 기준으로 양자화)

    분산 학습에서는 모든 rank가 함께 호출해야 하며(가중치 수집 때문),
    실제 파일 쓰기는 rank 0(main)만 수행합니다.
    """
    state_dict = gather_full_state_dict(model, context)
    ema_state: dict[str, torch.Tensor] | None = None
    if ema is not None:
        # swap 블록 안에서 모델 가중치가 잠시 EMA 값으로 바뀝니다.
        # (모든 rank 가 같은 블록을 지나므로 분산 수집과도 안전하게 맞물립니다.)
        with ema.swap(model):
            ema_state = gather_full_state_dict(model, context)

    if context.is_main:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        pad_id = int(getattr(unwrap_model(model), "pad_id", 0))
        # 로드할 때 모델을 그대로 재조립할 수 있도록 설정과 함께 저장합니다.
        common: dict[str, Any] = {
            "step": step,
            "model_config": asdict(model_config),
            "pad_id": pad_id,
        }

        # ① 일반(FP32) 원본 가중치
        torch.save({**common, "model": state_dict}, directory / "model.pt")

        # ② EMA 가중치 (활성 시) — 배포/평가 권장 저장본
        if ema_state is not None:
            torch.save({**common, "model": ema_state}, directory / "model_ema.pt")

        # ③ INT8 동적 양자화 — CPU 추론용 경량 저장본.
        #    양자화된 모듈은 구조 자체가 바뀌므로 state_dict 대신
        #    모듈 전체를 저장해 로드가 한 줄로 끝나게 합니다.
        #    품질이 더 좋은 EMA 가중치가 있으면 그쪽을 양자화합니다.
        quantized = _quantize_int8(
            model_config, ema_state if ema_state is not None else state_dict, pad_id
        )
        torch.save({**common, "model": quantized}, directory / "model_int8.pt")

    # 다른 rank 들이 파일 쓰기가 끝나기 전에 먼저 진행하지 않도록 맞춥니다.
    barrier(context)
