"""FP8 가중치를 FP8 인 채로 상주시켜 추론하는 경로.

export 가 저장한 FP8 가중치와 블록 스케일은 모델 버퍼에서도 FP8 로 유지되므로
상주 메모리는 줄어듭니다. 현재 ``Fp8Linear`` 의 계산 경로는 하나뿐입니다.
매 ``forward`` 에서 전체 가중치를 ``compute_dtype`` (기본값 bf16)으로
역양자화한 뒤 평범한 ``torch.nn.functional.linear`` 를 호출합니다.

따라서 이 구현은 ``torch._scaled_mm`` 을 호출하지 않고 네이티브 FP8
텐서코어 GEMM 을 사용하지 않습니다. 역양자화된 임시 dense 가중치를 만들기
때문에 실행 중 메모리 대역폭이나 연산량이 줄어든다고도 보장할 수 없습니다.
장치 성능 이득은 별도 벤치마크로 확인해야 합니다.

활성값은 FP8 로 내리지 않습니다. 가중치만 내리는 편이 더 정확하고(출력 오차
2.57% 대 3.63%) 활성값 이상치에 영향받지 않습니다. 근거 수치는
``sion_translate.fp8`` 문서에 있습니다.
"""

# Torch FP8 primitives and optional packed-module hooks are incompletely typed.
# pyright: reportArgumentType=false, reportCallIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import torch
from torch import nn

from sion_translate.fp8 import DEFAULT_BLOCK, FORWARD_DTYPE


class Fp8Linear(nn.Module):
    """``nn.Linear`` 대체품. 가중치를 E4M3 + 블록 스케일로 들고 있습니다.

    ``bias`` 는 받지 않습니다 — 이 모델의 projection 은 전부 bias 가 없고,
    쓰지 않을 분기를 두지 않기 위해서입니다.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
        *,
        block: int = DEFAULT_BLOCK,
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if weight.dtype is not FORWARD_DTYPE:
            raise ValueError(f"Fp8Linear expects {FORWARD_DTYPE} weights, got {weight.dtype}")
        if weight.ndim != 2:
            raise ValueError("Fp8Linear expects a 2-D weight")
        out_features, in_features = weight.shape
        if in_features % block:
            raise ValueError(
                f"in_features {in_features} is not a multiple of the FP8 block size {block}"
            )
        if scales.shape != (out_features, in_features // block):
            raise ValueError(
                f"scale shape {tuple(scales.shape)} does not match "
                f"{(out_features, in_features // block)}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.block = block
        self.compute_dtype = compute_dtype
        # 버퍼로 둡니다. 학습 대상이 아니고, state_dict 에는 남아야 합니다.
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)

    def dequantized_weight(self) -> torch.Tensor:
        grouped = self.weight.to(self.compute_dtype).reshape(self.out_features, -1, self.block)
        restored = grouped * self.scales.to(self.compute_dtype).unsqueeze(-1)
        return restored.reshape(self.out_features, self.in_features)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight()
        return torch.nn.functional.linear(hidden.to(weight.dtype), weight)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block={self.block}, weight_dtype={FORWARD_DTYPE}"
        )


def _replace_child(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, attribute, replacement)


def apply_fp8_weights(
    model: nn.Module,
    packed_state: dict[str, dict[str, object]],
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """FP8 로 저장된 항목을 ``Fp8Linear`` 로 바꿔 끼운다.

    ``packed_state`` 는 ``training.export._pack_fp8_state`` 가 낸 것입니다.
    ``block_fp8`` 항목만 교체하고 나머지는 건드리지 않으므로, 어떤 텐서가
    고정밀도로 남을지는 export 시점의 정책이 이미 정해 둔 그대로입니다.

    반환값은 교체한 모듈 수입니다.
    """

    replaced = 0
    for name, entry in packed_state.items():
        if entry.get("kind") != "block_fp8":
            continue
        if not name.endswith(".weight"):
            raise ValueError(f"FP8 packed entry is not a linear weight: {name}")
        module_name = name.removesuffix(".weight")
        target = model.get_submodule(module_name)
        if not isinstance(target, nn.Linear):
            raise ValueError(
                f"{module_name} is a {type(target).__name__}, not nn.Linear; "
                "the FP8 export and this model do not agree"
            )
        if target.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(f"{module_name} has a bias, which Fp8Linear does not carry")
        _replace_child(
            model,
            module_name,
            Fp8Linear(
                entry["values"],
                entry["scales"],
                block=int(entry.get("block", DEFAULT_BLOCK)),
                compute_dtype=compute_dtype,
            ),
        )
        replaced += 1
    return replaced


def describe_runtime(device: torch.device) -> str:
    """실제로 사용하는 계산 경로를 배포 로그용 한 줄로 설명합니다.

    ``device`` 는 호출 API 호환성을 위해 받습니다. 현재 구현은 장치 capability와
    무관하게 같은 역양자화 경로를 사용합니다.
    """

    del device
    return (
        "FP8 상주 가중치 + BF16 즉시 역양자화 후 dense GEMM "
        "(상주 메모리 절감; 네이티브 FP8 텐서코어 미사용)"
    )


__all__ = ["Fp8Linear", "apply_fp8_weights", "describe_runtime"]
