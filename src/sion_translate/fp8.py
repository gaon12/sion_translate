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

## 가중치만 FP8 (활성값은 bf16)

디코딩은 이 모델에서 가중치 대역폭 바운드입니다(KV cache 는 전체 바이트의
1.5%). 그래서 대역폭을 줄이는 데는 가중치만 내려도 충분하고, 그편이 더
정확합니다:

    가중치 FP8 + 활성값 bf16    출력오차 2.57%   (활성값 이상치와 무관)
    가중치·활성값 모두 FP8       출력오차 3.63%
    bf16                       출력오차 0.23%

## 오차는 깊이를 따라 누적된다

이것이 이 파일에서 가장 중요한 수치입니다. GEMM 하나의 2.57% 는 그대로
남지 않습니다. sion_data_fit(encoder 16 / decoder 8) 전체를 통과시키면:

    GEMM 1회        2.57%
    encoder 16층    11.7%      (약 sqrt(층수) 배)
    최종 logits     13.1%

층을 지날수록 독립적인 양자화 잡음이 잔차 스트림에 쌓입니다. 단일 GEMM
수치만 보고 "3% 정도"라고 판단하면 안 됩니다.

## 무엇이 누적을 끊는가 (실측)

    구성                          logits오차  argmax불일치   KL      FP8비율
    전 층 FP8                       13.11%      18.75%   0.0628    100%
    앞뒤 1층씩 bf16                   11.37%      18.75%   0.0471     83%
    앞뒤 2층씩 bf16                    9.42%      13.54%   0.0325     65%
    앞뒤 3층씩 bf16                    7.61%      13.54%   0.0213     48%
    FFN 만 FP8 (attention bf16)      6.39%       8.33%   0.0150     69%

**FFN 만 내리는 것이 가장 좋은 거래입니다.** 오차는 절반인데 양자화 대상의
69% 를 덮습니다. attention projection 은 softmax 를 거치며 오차가 증폭되고,
FFN 의 오차는 잔차에 더해질 뿐이라 그렇습니다. "가장자리 층을 빼는" 흔한
처방보다 이 모델에서는 이쪽이 낫습니다.

## 시도했으나 효과가 없던 보정 (다시 하지 마십시오)

- **양자화 잔차의 저계수 보정** (LQER 계열): rank-32 가 대역폭을 11.5% 더
  쓰면서 가중치 오차를 2.57% → 2.44% 로 줄일 뿐입니다. 반올림 잔차는
  백색잡음에 가까워 저계수 구조가 없습니다.
- **활성값 인지 스케일링** (AWQ 계열): 2.571% → 2.544%. 채널 크기가 100 배
  차이 나는 활성값에서도 그렇습니다.

둘 다 실패하는 이유는 같습니다. 여기서 남은 오차는 **범위**가 아니라 E4M3
의 mantissa 3 비트, 즉 **해상도** 문제입니다. 블록 스케일링으로 범위를 이미
맞춘 뒤에는 스케일을 아무리 영리하게 골라도 mantissa 를 늘릴 수 없습니다.
블록 폭을 128→32 로 좁히면 2.571% → 2.400% 로 조금 더 가지만, 스케일 계수가
4 배가 됩니다.
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

# FFN projection. 실측에서 가장 좋은 거래를 주는 범위입니다 — 오차는 전 층
# FP8 의 절반인데 양자화 대상의 69% 를 덮습니다.
FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# attention projection. 여기까지 내리면 대역폭은 더 줄지만 softmax 를 거치며
# 오차가 증폭됩니다 (logits 오차 6.39% → 13.11%).
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")

# 두 범위를 합친 것. 모델 전체 파라미터의 81.6%.
QUANTIZABLE_PROJECTIONS = FFN_PROJECTIONS + ATTENTION_PROJECTIONS

# 기본 범위. 이름이 아니라 측정에서 나온 값입니다.
SCOPE_FFN = "ffn"
SCOPE_ALL = "all"
SCOPES = (SCOPE_FFN, SCOPE_ALL)

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
    # 기본은 FFN 만. attention 까지 내리면 대역폭은 더 줄지만 최종 logits
    # 오차가 두 배가 됩니다 (6.39% → 13.11%).
    scope: str = SCOPE_FFN
    # 어휘 projection 을 내리는 것은 실측에서 argmax 6.45% 변경이라
    # 기본은 False 입니다. 연구 목적으로 켜려면 명시해야 합니다.
    quantize_vocabulary_projection: bool = False

    def validate(self) -> None:
        if self.block < 1:
            raise ValueError("fp8 block size must be positive")
        if self.block & (self.block - 1):
            raise ValueError("fp8 block size must be a power of two")
        if self.scope not in SCOPES:
            raise ValueError(f"fp8 scope must be one of {SCOPES}, got {self.scope!r}")

    def targets(self) -> tuple[str, ...]:
        return FFN_PROJECTIONS if self.scope == SCOPE_FFN else QUANTIZABLE_PROJECTIONS

    def allows(self, parameter_name: str) -> bool:
        """``named_parameters()`` 이름 하나가 FP8 대상인지."""

        if not self.enabled:
            return False
        if self.scope not in SCOPES:
            raise ValueError(f"fp8 scope must be one of {SCOPES}, got {self.scope!r}")
        if not self.quantize_vocabulary_projection and any(
            token in parameter_name for token in PROTECTED_SUBSTRINGS
        ):
            return False
        return any(token in parameter_name for token in self.targets())


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
    "ATTENTION_PROJECTIONS",
    "DEFAULT_BLOCK",
    "FFN_PROJECTIONS",
    "FORWARD_DTYPE",
    "GRADIENT_DTYPE",
    "PROTECTED_SUBSTRINGS",
    "QUANTIZABLE_PROJECTIONS",
    "SCOPES",
    "SCOPE_ALL",
    "SCOPE_FFN",
    "Fp8Policy",
    "fp8_gemm_supported",
    "scale_for",
]
