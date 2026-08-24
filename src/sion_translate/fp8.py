"""FP8 measurements and the policy that defines safe quantization scope.

This module does not perform native FP8 computation. The current inference
runtime dequantizes resident FP8 weights to BF16, or FP16 on CUDA devices that
do not support BF16, and then runs a dense GEMM. It does not use a native FP8
Tensor Core GEMM. This module provides two things:

1. Scaling and quantization utilities that can be verified on a CPU. The error
   measurements below restore FP8-rounded values to a higher precision before
   running GEMM; they do not describe the kernel used in production.
2. The policy that decides which tensors may be quantized. This is the more
   important part because an unsafe FP8 configuration usually fails by
   quantizing a sensitive tensor, not merely by running slowly.

Measurements for ``sion_data_fit`` with a 48,000-token vocabulary follow.
Relative GEMM output error was measured at M=2048, K=768, and N=2048:

    input distribution       per-tensor   block128   block32   bf16
    normal                       3.74%       3.64%     3.39%    0.23%
    0.1% outliers at 30x         3.75%       3.19%     2.97%    0.23%

FP8 GEMM error is about 15 times the BF16 error, so it is not "nearly the
same." Block scaling helps mainly when outliers are present: 3.74% to 3.64%
for a normal distribution, but 3.75% to 3.19% with outliers. Trained
activations contain outliers, so block scaling is the default.

E5M2 produced twice the error of E4M3 on the same tensor (5.25% versus 2.66%).
It has one fewer mantissa bit. E5M2 is nevertheless useful for gradients
because of its dynamic range, not because of its precision.

The vocabulary projection must not be quantized by default. Quantizing both
hidden states and weights to E4M3 changed the argmax for 6.45% of positions in
a 48,000-token vocabulary. In greedy decoding, each change directly selects a
different token. With ``tie_embeddings=True``, this restriction also protects
``token_embedding.weight`` because it is the output projection. Storing that
weight in FP8 would also degrade input embedding lookup.

The export and runtime format stores weights, but not activations, in FP8.
Dense computation uses BF16, or FP16 where CUDA BF16 is unavailable. This
reduces stored and resident weight bytes while avoiding activation
quantization error. The default runtime still dequantizes each weight to the
selected compute dtype on every forward pass and uses dense GEMM, so this
format does not promise execution-bandwidth or operation-count savings:

    FP8 weights + BF16 activations     output error 2.57%
    FP8 weights + FP8 activations      output error 3.63%
    BF16                               output error 0.23%

Quantization error accumulates with depth. A single GEMM's 2.57% error became
11.7% after the 16-layer encoder and 13.1% at the final logits in the measured
16-encoder/8-decoder model. Independent quantization noise accumulates in the
residual stream, so a single-GEMM measurement must not be treated as the
end-to-end model error.

Measured scope trade-offs were:

    configuration              logit error   argmax mismatch   KL      FP8 share
    every layer FP8                13.11%          18.75%      0.0628      100%
    first/last 1 layer BF16        11.37%          18.75%      0.0471       83%
    first/last 2 layers BF16        9.42%          13.54%      0.0325       65%
    first/last 3 layers BF16        7.61%          13.54%      0.0213       48%
    FFN only FP8                     6.39%           8.33%      0.0150       69%

FFN-only quantization is the best measured trade-off: it halves error while
covering 69% of quantizable weights. Attention projection error is amplified
through softmax, while FFN error is added to the residual stream. This was
better for this model than merely exempting edge layers.

Two attempted corrections were not useful and should not be repeated without
new evidence:

- A rank-32 low-rank quantization-residual correction (LQER family) consumed
  11.5% more bandwidth and reduced weight error only from 2.57% to 2.44%. The
  rounding residual was close to white noise and had little low-rank structure.
- Activation-aware scaling (AWQ family) reduced error only from 2.571% to
  2.544%, even when activation channel magnitudes differed by 100 times.

Both approaches fail for the same reason: after block scaling controls range,
the remaining limit is the three-bit E4M3 mantissa, or resolution. Smarter
scales cannot add mantissa bits. Reducing the block width from 128 to 32 lowers
error from 2.571% to 2.400%, but requires four times as many scale values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# E4M3 favors forward precision; E5M2 gives gradients more dynamic range.
FORWARD_DTYPE = torch.float8_e4m3fn
GRADIENT_DTYPE = torch.float8_e5m2

# Narrow blocks handle outliers better, but add scale values and kernel cost.
# Width 128 captured most of the measured gain (3.75% to 3.19%); width 32
# improved it only to 2.97%.
DEFAULT_BLOCK = 128

# FFN projections gave the best measured trade-off: half the all-layer error
# while covering 69% of quantizable weights.
FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# Adding attention projections increases resident FP8 coverage, but softmax
# amplifies their error (6.39% to 13.11% logit error).
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")

# Together these scopes cover 81.6% of all model parameters.
QUANTIZABLE_PROJECTIONS = FFN_PROJECTIONS + ATTENTION_PROJECTIONS

# Scope defaults are based on the measurements above.
SCOPE_FFN = "ffn"
SCOPE_ALL = "all"
SCOPES = (SCOPE_FFN, SCOPE_ALL)

# Parameters containing these names remain high precision under every scope.
PROTECTED_SUBSTRINGS = (
    "token_embedding",  # This is also the output projection when embeddings are tied.
    "lm_head",
    "norm",  # RMSNorm already computes in FP32.
    "register_embeddings",
    "type_embedding",
    "mode_embedding",
    "uncertainty_head",  # A scalar gate offers no useful memory saving.
)


@dataclass(frozen=True)
class Fp8Policy:
    """Select which weights use FP8; defaults follow measured quality trade-offs."""

    enabled: bool = False
    block: int = DEFAULT_BLOCK
    # FFN-only is the default. Adding attention doubles measured final-logit
    # error from 6.39% to 13.11% despite increasing resident FP8 coverage.
    scope: str = SCOPE_FFN
    # Vocabulary projection quantization changed 6.45% of measured argmax
    # choices, so experimental use requires an explicit opt-in.
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
        """Return whether a ``named_parameters()`` entry is eligible for FP8."""

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
    """Return quantization scales, treating the full tensor as one block if needed.

    Each scale is ``amax / dtype_max``. A tiny lower bound avoids division by
    zero for an all-zero block; any nonzero scale produces the same zeros there.
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
    """Return whether the device generation can support FP8 Tensor Core GEMM.

    A dtype can exist even when no corresponding kernel exists; a CPU can cast
    to FP8 but cannot run ``_scaled_mm``. This checks hardware capability only.
    It does not imply that :class:`Fp8Linear` uses a native FP8 kernel.
    """

    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    # Ada is capability 8.9 and Hopper is 9.0; earlier devices lack FP8 Tensor Cores.
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
