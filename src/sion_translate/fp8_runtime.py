"""FP8 가중치를 FP8 인 채로 상주시켜 추론하는 경로.

export 가 저장한 FP8 가중치와 블록 스케일은 모델 버퍼에서도 FP8 로 유지되므로
상주 메모리는 줄어듭니다. 현재 ``Fp8Linear`` 의 계산 경로는 하나뿐입니다.
매 ``forward`` 에서 전체 가중치를 장치가 지원하는 ``compute_dtype``
(기본 BF16, BF16 미지원 CUDA에서는 FP16)으로 역양자화한 뒤 평범한
``torch.nn.functional.linear`` 를 호출합니다.

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

from sion_translate.fp8 import DEFAULT_BLOCK, FORWARD_DTYPE, fp8_gemm_supported


def _cuda_bf16_supported(device: torch.device) -> bool:
    """Return whether ``device`` has native BF16 CUDA arithmetic.

    CUDA BF16 tensor-core arithmetic starts with Ampere (SM 8.x).  Looking up
    the requested device directly also avoids accidentally inspecting CUDA's
    current device in a heterogeneous process.
    """

    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(device)
    return major >= 8


def resolve_fp8_compute_dtype(device: torch.device) -> torch.dtype:
    """Choose the dense dtype used after restoring resident FP8 weights.

    FP8 is only the resident weight format in this runtime.  CUDA devices use
    BF16 when the architecture supports it (including A100) and otherwise use
    FP16, so loading an FP8 export never requires native FP8 tensor cores.  The
    existing CPU path remains BF16; other accelerators conservatively use FP16.
    """

    if device.type == "cpu":
        return torch.bfloat16
    if device.type == "cuda" and _cuda_bf16_supported(device):
        return torch.bfloat16
    return torch.float16


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
        compute_dtype: torch.dtype | None = None,
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
        compute_dtype = self.compute_dtype or resolve_fp8_compute_dtype(self.weight.device)
        grouped = self.weight.to(compute_dtype).reshape(self.out_features, -1, self.block)
        restored = grouped * self.scales.to(compute_dtype).unsqueeze(-1)
        return restored.reshape(self.out_features, self.in_features)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight()
        output = torch.nn.functional.linear(hidden.to(weight.dtype), weight)
        # Fp8Linear is embedded between residual branches.  Returning FP16 into
        # a BF16 residual promotes the sum to FP32 and makes the next BF16
        # Linear fail with a mixed-dtype matmul.  Keep the module boundary in
        # the caller's dtype while using the selected fallback dtype internally.
        return output.to(hidden.dtype)

    def extra_repr(self) -> str:
        compute_dtype = (
            str(self.compute_dtype).removeprefix("torch.")
            if self.compute_dtype is not None
            else "auto(bfloat16->float16)"
        )
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block={self.block}, weight_dtype={FORWARD_DTYPE}, "
            f"compute_dtype={compute_dtype}"
        )


def _replace_child(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, attribute, replacement)


def apply_fp8_weights(
    model: nn.Module,
    packed_state: dict[str, dict[str, object]],
    *,
    compute_dtype: torch.dtype | None = None,
) -> int:
    """FP8 로 저장된 항목을 ``Fp8Linear`` 로 바꿔 끼운다.

    ``packed_state`` 는 ``training.export._pack_fp8_state`` 가 낸 것입니다.
    ``block_fp8`` 항목만 교체하고 나머지는 건드리지 않으므로, 어떤 텐서가
    고정밀도로 남을지는 export 시점의 정책이 이미 정해 둔 그대로입니다.

    ``compute_dtype`` 을 지정하지 않으면 forward 시점의 장치에서 BF16 지원을
    확인하고, 지원하지 않는 CUDA 장치에서는 FP16 으로 자동 fallback 합니다.
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


def prepare_fp8_model_for_device(model: nn.Module, device: torch.device) -> torch.dtype:
    """Move an FP8-resident model and align its dense state with ``device``.

    ``nn.Module.to(dtype=...)`` would also cast ``Fp8Linear.weight`` and
    ``Fp8Linear.scales``, destroying the resident FP8 representation and its
    full-precision block scales.  Cast only ordinary parameters and non-FP8
    floating buffers on their current device first, then move every tensor
    without changing dtype.  In the common CPU-load path this also avoids ever
    placing BF16 dense tensors on a CUDA device that cannot execute them.

    The returned dtype is the dense compute dtype selected for telemetry and
    tests.  This helper is intended for inference models, before any gradients
    have been accumulated.
    """

    compute_dtype = resolve_fp8_compute_dtype(device)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype is not compute_dtype:
                parameter.data = parameter.data.to(dtype=compute_dtype)
            if parameter.grad is not None and parameter.grad.is_floating_point():
                parameter.grad.data = parameter.grad.data.to(dtype=compute_dtype)
        for module in model.modules():
            if isinstance(module, Fp8Linear):
                module.compute_dtype = compute_dtype
                continue
            for name, buffer in module.named_buffers(recurse=False):
                if buffer.is_floating_point() and buffer.dtype is not compute_dtype:
                    setattr(module, name, buffer.to(dtype=compute_dtype))
    model.to(device=device)
    return compute_dtype


def describe_runtime(device: torch.device) -> str:
    """실제로 사용하는 계산 경로를 배포 로그용 한 줄로 설명합니다.

    상주 형식, 실제 dense 계산 dtype, 네이티브 FP8 지원 여부를 구분해서
    기록합니다. 네이티브 FP8 지원 장치에서도 현재 커널은 이를 사용하지 않습니다.
    """

    compute_dtype = resolve_fp8_compute_dtype(device)
    compute_name = "BF16" if compute_dtype is torch.bfloat16 else "FP16"
    if device.type == "cuda":
        if fp8_gemm_supported(device):
            hardware = "네이티브 FP8 텐서코어 지원 장치이지만 현재 경로에서는 미사용"
        else:
            hardware = "네이티브 FP8 텐서코어 미지원 장치 fallback"
    else:
        hardware = f"{device.type.upper()} dense fallback"
    return (
        f"FP8 상주 가중치 + {compute_name} 즉시 역양자화 후 dense GEMM "
        f"(상주 메모리 절감; {hardware})"
    )


__all__ = [
    "Fp8Linear",
    "apply_fp8_weights",
    "describe_runtime",
    "prepare_fp8_model_for_device",
    "resolve_fp8_compute_dtype",
]
