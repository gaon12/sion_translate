"""가중치 지수이동평균(EMA, Exponential Moving Average).

학습 중의 가중치는 step 마다 이리저리 흔들립니다. EMA 는 그 흔들림을
평균 내어 '가중치의 궤적을 부드럽게 따라가는 그림자(shadow) 복사본'을
유지합니다.

    shadow = decay × shadow + (1 - decay) × 현재 가중치   (매 step)

번역 모델에서는 EMA(또는 checkpoint averaging) 가중치로 평가/배포하는 것이
원본 가중치보다 검증 loss 와 BLEU 를 안정적으로 개선하는 검증된 기법이라
기본으로 켜 둡니다 (``training.ema_decay``, 0 이면 비활성).

분산 학습 참고: FSDP2 에서는 파라미터가 DTensor(조각난 텐서)인데,
shadow 도 같은 방식으로 조각난 복사본으로 만들어지므로 rank 별 추가
메모리는 자기 조각만큼만 듭니다. lerp_/copy_ 같은 elementwise 연산은
DTensor 에서도 그대로 동작합니다.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


class EMAWeights:
    """모델 파라미터의 EMA shadow 복사본을 유지합니다.

    사용 흐름:
        ema = EMAWeights(model, decay=0.999)
        ...  # optimizer.step() 성공 후마다
        ema.update(model)
        ...  # EMA 가중치로 평가/내보내기 할 때
        with ema.swap(model):
            evaluate(model, ...)   # 이 블록 안에서 model 은 EMA 가중치
    """

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        # 학습 대상 파라미터만 따라갑니다 (버퍼는 이 모델에 학습 상태가 없음).
        self.shadow: dict[str, torch.Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """shadow ← decay·shadow + (1-decay)·param. optimizer step 뒤에 호출합니다."""
        one_minus_decay = 1.0 - self.decay
        for name, parameter in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is not None:
                shadow.lerp_(parameter.detach(), one_minus_decay)

    @contextmanager
    def swap(self, model: nn.Module) -> Iterator[None]:
        """블록 안에서만 모델 가중치를 EMA 값으로 바꿉니다 (나가면 원상복구).

        평가나 내보내기 도중 예외가 나도 원본 가중치가 반드시 복원되도록
        try/finally 로 감쌉니다.
        """
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                shadow = self.shadow.get(name)
                if shadow is not None:
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(shadow)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    saved = backup.get(name)
                    if saved is not None:
                        parameter.copy_(saved)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """체크포인트에 함께 저장해 재개 시 EMA 이력이 끊기지 않게 합니다."""
        return dict(self.shadow)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, tensor in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)
