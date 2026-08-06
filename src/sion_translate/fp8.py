"""FP8 수치와 "어디까지 내려도 되는가" 정책.

이 모듈은 FP8 **연산**을 하지 않습니다. FP8 텐서코어는 Hopper 이상에서만
돌고, 이 저장소의 CI 는 CPU 입니다. 여기 있는 것은 두 가지입니다.

1. 스케일링·양자화 수치. CPU 에서 그대로 검증됩니다(캐스팅은 CPU 에서
   동작합니다). 실제 GEMM 은 ``torch._scaled_mm`` 이 하고, 그 경로는
   하드웨어가 있을 때만 켜집니다.
2. **정책**: 어떤 텐서를 FP8 로 내려도 되는가. 이쪽이 더 중요합니다 —
   FP8 학습이 실패하는 방식은 대부분 "느려짐"이 아니라 "내리면 안 되는 것을
   내림"이기 때문입니다.

## 이 모델에서 실측한 것 (sion_data_fit, vocab 48,000)

GEMM 출력 상대오차 (M=2048, K=768, N=2048):

    입력 분포              per-tensor   block128   block32   bf16
    정규분포                  3.74%       3.64%     3.39%    0.23%
    이상치 0.1% x30배         3.75%       3.19%     2.97%    0.23%

두 가지를 읽을 수 있습니다. 첫째, FP8 GEMM 오차는 bf16 의 **약 15배**입니다 —
"거의 같다"가 아닙니다. 둘째, block 단위 스케일링의 이득은 **이상치가 있을
때만** 나옵니다(정규분포에서는 3.74%→3.64%, 이상치에서는 3.75%→3.19%).
학습된 활성값에는 이상치가 생기므로 기본값을 block 으로 둡니다.

기울기용 E5M2 는 같은 텐서에서 E4M3 의 두 배 오차입니다(2.66% 대 5.25%).
mantissa 가 한 비트 적으니 당연하고, 그럼에도 기울기에 E5M2 를 쓰는 이유는
정밀도가 아니라 **동적 범위** 때문입니다.

## 절대 내리면 안 되는 것

어휘 projection 입니다. 48,000 어휘에서 hidden 과 가중치를 모두 E4M3 로
내리면 **argmax 가 6.45% 바뀝니다** — 여섯 토큰 중 하나 꼴로 다른 단어를
고른다는 뜻이고, greedy 디코딩에서는 그대로 오역입니다.

이 모델에서는 그 금지가 임베딩까지 번집니다. ``tie_embeddings=True`` 라
출력 projection 이 곧 ``token_embedding.weight`` 이기 때문입니다. 가중치를
FP8 로 **저장**해 버리면 입력 임베딩 조회까지 같이 망가집니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# 순전파(가중치·활성값)는 E4M3, 역전파(기울기)는 E5M2. 활성값은 정밀도가,
# 기울기는 동적 범위가 아쉬운 쪽이라 표준적으로 이렇게 나눕니다.
FORWARD_DTYPE = torch.float8_e4m3fn
GRADIENT_DTYPE = torch.float8_e5m2

# 블록 단위 스케일링의 기본 폭. 좁힐수록 이상치에 강하지만 스케일 계수가
# 늘고 커널이 느려집니다. 128 은 이상치 실험에서 이득의 대부분을 가져오는
# 지점입니다(3.75% → 3.19%; block32 로 더 좁혀도 2.97% 로 조금 더 갈 뿐).
DEFAULT_BLOCK = 128

# FP8 로 내려도 되는 projection. 모델 전체 파라미터의 81.6% 입니다.
QUANTIZABLE_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# 이름에 이것이 들어가면 어떤 설정에서도 FP8 로 내리지 않습니다.
PROTECTED_SUBSTRINGS = (
    "token_embedding",  # tie_embeddings 면 이것이 곧 출력 projection
    "lm_head",
    "norm",  # RMSNorm 은 이미 fp32 로 계산합니다
    "register_embeddings",
    "type_embedding",
    "mode_embedding",
    "uncertainty_head",  # 스칼라 게이트, 이득 없음
)


@dataclass(frozen=True)
class Fp8Policy:
    """무엇을 FP8 로 내릴지. 기본값은 측정에서 나온 것입니다."""

    enabled: bool = False
    block: int = DEFAULT_BLOCK
    # 어휘 projection 을 내리는 것은 실측에서 argmax 6.45% 변경이라
    # 기본은 False 입니다. 연구 목적으로 켜려면 명시해야 합니다.
    quantize_vocabulary_projection: bool = False

    def validate(self) -> None:
        if self.block < 1:
            raise ValueError("fp8 block size must be positive")
        if self.block & (self.block - 1):
            raise ValueError("fp8 block size must be a power of two")

    def allows(self, parameter_name: str) -> bool:
        """``named_parameters()`` 이름 하나가 FP8 대상인지."""

        if not self.enabled:
            return False
        if not self.quantize_vocabulary_projection and any(
            token in parameter_name for token in PROTECTED_SUBSTRINGS
        ):
            return False
        return any(token in parameter_name for token in QUANTIZABLE_PROJECTIONS)


def scale_for(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype = FORWARD_DTYPE,
    block: int | None = DEFAULT_BLOCK,
) -> torch.Tensor:
    """양자화 스케일. ``block`` 이 ``None`` 이면 텐서 전체가 한 블록입니다.

    스케일은 ``amax / dtype_max`` 입니다. 0 인 블록에서 0 으로 나누지 않도록
    아주 작은 값으로 하한을 둡니다 — 그 블록은 어차피 전부 0 이라 어떤
    스케일을 써도 결과가 같습니다.
    """

    maximum = torch.finfo(dtype).max
    if block is None:
        return tensor.abs().amax().clamp_min(torch.finfo(torch.float32).tiny) / maximum
    if tensor.shape[-1] % block:
        raise ValueError(
            f"last dimension {tensor.shape[-1]} is not a multiple of the FP8 block size {block}; "
            "pad the tensor or choose a divisor block size"
        )
    grouped = tensor.reshape(*tensor.shape[:-1], -1, block)
    amax = grouped.abs().amax(-1, keepdim=True)
    return amax.clamp_min(torch.finfo(torch.float32).tiny) / maximum


def quantize_dequantize(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype = FORWARD_DTYPE,
    block: int | None = DEFAULT_BLOCK,
) -> torch.Tensor:
    """FP8 왕복. 실제 FP8 GEMM 이 보는 값을 고정밀도로 재현합니다.

    학습 경로에서 쓰는 것이 아니라, 하드웨어 없이 **오차를 측정**하고
    회귀를 잡기 위한 것입니다. 실제 커널은 ``torch._scaled_mm`` 입니다.
    """

    original = tensor.dtype
    work = tensor.float()
    scale = scale_for(work, dtype=dtype, block=block)
    if block is None:
        return ((work / scale).to(dtype).float() * scale).to(original)
    grouped = work.reshape(*work.shape[:-1], -1, block)
    restored = (grouped / scale).to(dtype).float() * scale
    return restored.reshape(work.shape).to(original)


def relative_error(approximate: torch.Tensor, exact: torch.Tensor) -> float:
    """Frobenius 상대 오차. 0 텐서에서도 정의됩니다."""

    denominator = exact.float().norm()
    if denominator == 0:
        return float(approximate.float().norm())
    return float((approximate.float() - exact.float()).norm() / denominator)


def gemm_error(
    activations: torch.Tensor,
    weights: torch.Tensor,
    *,
    block: int | None = DEFAULT_BLOCK,
) -> float:
    """``activations @ weights.T`` 을 FP8 로 했을 때의 출력 상대 오차.

    개별 텐서의 양자화 오차가 아니라 **출력** 오차를 봅니다. 실제로 학습에
    영향을 주는 것은 이쪽이고, 축소 차원을 따라 오차가 일부 상쇄되므로 두
    수치는 같지 않습니다.
    """

    exact = activations.float() @ weights.float().T
    approximate = (
        quantize_dequantize(activations, block=block).float()
        @ quantize_dequantize(weights, block=block).float().T
    )
    return relative_error(approximate, exact)


def fp8_gemm_supported(device: torch.device | None = None) -> bool:
    """이 장치에서 FP8 텐서코어 GEMM 이 실제로 도는지.

    dtype 이 존재하는 것과 커널이 있는 것은 다릅니다. CPU 에서도 FP8 로
    캐스팅은 되지만 ``_scaled_mm`` 은 없습니다.
    """

    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    # 8.9 = Ada, 9.0 = Hopper. 그 아래는 FP8 텐서코어가 없습니다.
    return (major, minor) >= (8, 9)


__all__ = [
    "DEFAULT_BLOCK",
    "FORWARD_DTYPE",
    "GRADIENT_DTYPE",
    "PROTECTED_SUBSTRINGS",
    "QUANTIZABLE_PROJECTIONS",
    "Fp8Policy",
    "fp8_gemm_supported",
    "gemm_error",
    "quantize_dequantize",
    "relative_error",
    "scale_for",
]
