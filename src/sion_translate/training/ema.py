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
from collections.abc import Mapping
from typing import Iterator, cast

import torch
from torch import nn


def _named_parameters_below_compile(model: nn.Module):
    """Yield stable names from below any torch.compile wrapper."""

    unwrapped = model
    while True:
        original = getattr(unwrapped, "_orig_mod", None)
        if not isinstance(original, nn.Module):
            break
        unwrapped = original
    return unwrapped.named_parameters()


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
            for name, parameter in _named_parameters_below_compile(model)
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """shadow ← decay·shadow + (1-decay)·param. optimizer step 뒤에 호출합니다."""
        one_minus_decay = 1.0 - self.decay
        for name, parameter in _named_parameters_below_compile(model):
            shadow = self.shadow.get(name)
            if shadow is not None:
                shadow.lerp_(parameter.detach(), one_minus_decay)

    @staticmethod
    @torch.no_grad()
    def _exchange(parameter: torch.Tensor, shadow: torch.Tensor) -> None:
        """Exchange one parameter with its shadow using one-tensor scratch space."""

        current = parameter.detach().clone()
        parameter.copy_(shadow)
        shadow.copy_(current)

    @contextmanager  # pyright: ignore[reportDeprecated]
    def swap(self, model: nn.Module) -> Iterator[None]:
        """블록 안에서만 모델 가중치를 EMA 값으로 바꿉니다 (나가면 원상복구).

        전체 모델 크기의 backup dict를 만들지 않고 파라미터와 shadow를
        하나씩 교환합니다. 따라서 추가 peak 메모리는 가장 큰 파라미터
        하나뿐이며, 평가나 내보내기 도중 예외가 나도 원래 상태로 복원됩니다.
        """
        swapped: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            for name, parameter in _named_parameters_below_compile(model):
                shadow = self.shadow.get(name)
                if shadow is not None:
                    self._exchange(parameter, shadow)
                    swapped.append((parameter, shadow))
            yield
        finally:
            for parameter, shadow in reversed(swapped):
                self._exchange(parameter, shadow)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Permanently copy EMA weights into a model without allocating a backup."""

        for name, parameter in _named_parameters_below_compile(model):
            shadow = self.shadow.get(name)
            if shadow is not None:
                parameter.copy_(shadow)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """체크포인트에 함께 저장해 재개 시 EMA 이력이 끊기지 않게 합니다."""
        return dict(self.shadow)

    def _validated_state_dict(self, state: object) -> dict[str, torch.Tensor]:
        if not isinstance(state, Mapping):
            raise ValueError("EMA checkpoint state must be an object")
        normalized: dict[str, torch.Tensor] = {}
        for raw_name, raw_tensor in cast(Mapping[object, object], state).items():
            if not isinstance(raw_name, str) or not isinstance(raw_tensor, torch.Tensor):
                raise ValueError("EMA checkpoint state is malformed")
            canonical_name = raw_name
            while canonical_name.startswith("_orig_mod."):
                canonical_name = canonical_name.removeprefix("_orig_mod.")
            if canonical_name in normalized:
                raise ValueError(f"EMA checkpoint contains duplicate key: {canonical_name}")
            normalized[canonical_name] = raw_tensor
        missing = sorted(set(self.shadow) - set(normalized))
        unexpected = sorted(set(normalized) - set(self.shadow))
        if missing or unexpected:
            raise ValueError(
                "EMA checkpoint does not match the model parameters: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        for name, target in self.shadow.items():
            source = normalized[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(
                    "EMA checkpoint tensor metadata mismatch for "
                    f"{name}: checkpoint=(shape={tuple(source.shape)}, dtype={source.dtype}), "
                    f"model=(shape={tuple(target.shape)}, dtype={target.dtype})"
                )
        return normalized

    def validate_state_dict(self, state: object) -> None:
        """Validate a complete EMA snapshot without changing any shadow tensor."""

        self._validated_state_dict(state)

    @torch.no_grad()
    def load_state_dict(self, state: object) -> None:
        normalized = self._validated_state_dict(state)
        for name, tensor in normalized.items():
            canonical_name = name
            self.shadow[canonical_name].copy_(tensor)
