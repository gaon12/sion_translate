"""FP8 가중치를 **FP8 인 채로** 들고 추론하는 경로.

export 는 가중치를 FP8 로 저장합니다. 그것을 불러올 때 고정밀도로 되돌리면
디스크만 줄고 정작 노리던 것 — 디코딩 대역폭 — 은 그대로입니다. 이 모듈은
가중치를 FP8 로 둔 채 GEMM 마다 필요한 만큼만 쓰게 합니다.

두 가지 경로가 있고, 이득의 출처가 다릅니다.

- **``_scaled_mm`` (Hopper 이상)**: 가중치도 활성값도 FP8 로 텐서코어에
  들어갑니다. 대역폭과 연산량을 모두 줄입니다.
- **on-the-fly 역양자화 (그 외 전부)**: 가중치를 FP8 로 읽어 와 bf16 으로
  풀어서 평범한 GEMM 을 합니다. 연산량은 그대로지만 **HBM 에서 읽는 바이트가
  절반**이고, 이 모델의 디코딩은 가중치 대역폭 바운드이므로(KV cache 는 옮기는
  바이트의 1.5%) 이득의 대부분이 여기서 나옵니다. FP8 텐서코어가 없는 GPU
  에서도 동작합니다.

활성값은 어느 경로에서도 FP8 로 내리지 않습니다. 가중치만 내리는 편이 더
정확하고(출력 오차 2.57% 대 3.63%) 활성값 이상치에 영향받지 않습니다. 근거
수치는 ``sion_translate.fp8`` 문서에 있습니다.
"""

from __future__ import annotations

import torch
from torch import nn

from sion_translate.fp8 import DEFAULT_BLOCK, FORWARD_DTYPE, fp8_gemm_supported


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
        if target.bias is not None:
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
    """어느 경로로 도는지 한 줄로. 배포 로그에 남길 용도입니다."""

    if fp8_gemm_supported(device):
        return "FP8 텐서코어 GEMM (대역폭 + 연산량 절감)"
    return "FP8 가중치 + 즉시 역양자화 (대역폭만 절감; FP8 텐서코어 없음)"


__all__ = ["Fp8Linear", "apply_fp8_weights", "describe_runtime"]
