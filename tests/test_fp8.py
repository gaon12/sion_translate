"""FP8 measurements and policy tests.

Many tests here pin **how inaccurate FP8 is**, rather than claiming that it is
accurate. FP8 training usually fails by quantizing tensors that must remain
precise, not merely by becoming slower. Numeric bounds preserve that distinction.
"""

from __future__ import annotations

import pytest
import torch

from sion_translate.fp8 import (
    DEFAULT_BLOCK,
    FORWARD_DTYPE,
    GRADIENT_DTYPE,
    SCOPE_ALL,
    SCOPE_FFN,
    Fp8Policy,
    fp8_gemm_supported,
    scale_for,
)

# The next three functions are **measurement tools**. The production path does
# not use them, so they stay out of production code. They measure FP8 error, and
# the tests below pin the measured bounds.


def quantize_dequantize(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype = FORWARD_DTYPE,
    block: int | None = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Reproduce an FP8 round trip in high precision.

    This is not part of training. It measures error without requiring FP8
    hardware and catches regressions. The result is the quantized-value reference
    for a native FP8 GEMM; the current production runtime does not use
    ``torch._scaled_mm``.
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
    """Return Frobenius relative error, defined even for an all-zero tensor."""

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
    """Measure output error when ``activations @ weights.T`` uses FP8.

    This measures **output** error, not the quantization error of an individual
    tensor. Output error affects training directly, and some errors cancel along
    the reduction dimension, so the two measurements differ.
    """

    exact = activations.float() @ weights.float().T
    approximate = (
        quantize_dequantize(activations, block=block).float()
        @ quantize_dequantize(weights, block=block).float().T
    )
    return relative_error(approximate, exact)


def test_round_trip_preserves_the_scale_of_the_tensor() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(64, 256) * 3.0
    restored = quantize_dequantize(tensor)
    assert restored.shape == tensor.shape
    assert relative_error(restored, tensor) < 0.05
    assert restored.abs().max() == pytest.approx(tensor.abs().max(), rel=0.05)


def test_round_trip_keeps_the_input_dtype() -> None:
    tensor = torch.randn(8, 128, dtype=torch.bfloat16)
    assert quantize_dequantize(tensor).dtype is torch.bfloat16


def test_an_all_zero_block_does_not_divide_by_zero() -> None:
    tensor = torch.zeros(4, 256)
    restored = quantize_dequantize(tensor)
    assert torch.isfinite(restored).all()
    assert float(restored.abs().sum()) == 0.0


def test_a_ragged_last_dimension_is_refused_rather_than_silently_padded() -> None:
    with pytest.raises(ValueError, match="multiple of the FP8 block size"):
        quantize_dequantize(torch.randn(4, 130), block=128)


def test_per_tensor_scaling_is_available_for_comparison() -> None:
    tensor = torch.randn(4, 130)
    assert quantize_dequantize(tensor, block=None).shape == tensor.shape
    assert scale_for(tensor, block=None).ndim == 0


# ── Pin measured behavior ───────────────────────────────────────────────


def test_fp8_gemm_error_is_an_order_of_magnitude_worse_than_bf16() -> None:
    """FP8 is not "almost the same" as BF16.

    Measured with normal inputs at M=2048, K=768, N=2048: FP8 block-128 error
    was 3.64%, versus 0.23% for BF16. Losing this ratio can send an accuracy
    investigation toward the wrong cause.
    """
    torch.manual_seed(0)
    activations = torch.randn(512, 768)
    weights = torch.randn(1024, 768) * 0.02

    fp8 = gemm_error(activations, weights)
    exact = activations.float() @ weights.float().T
    bf16 = relative_error(
        activations.bfloat16().float() @ weights.bfloat16().float().T,
        exact,
    )

    assert 0.02 < fp8 < 0.06
    assert bf16 < 0.005
    assert fp8 > bf16 * 5


def test_block_scaling_earns_its_cost_only_when_there_are_outliers() -> None:
    """Block scaling addresses outliers; it is not a general accuracy improvement.

    Measured error changed from 3.74% to 3.64% on normal inputs, but from 3.75%
    to 3.19% with outliers.
    """
    torch.manual_seed(0)
    weights = torch.randn(1024, 768) * 0.02

    plain = torch.randn(512, 768)
    spiked = plain.clone()
    index = torch.randperm(spiked.numel())[: spiked.numel() // 1000]
    spiked.view(-1)[index] *= 30.0

    plain_gain = gemm_error(plain, weights, block=None) - gemm_error(plain, weights)
    spiked_gain = gemm_error(spiked, weights, block=None) - gemm_error(spiked, weights)

    assert spiked_gain > plain_gain
    assert spiked_gain > 0.002


def test_e5m2_trades_precision_for_range() -> None:
    """E5M2 is used for gradients because of dynamic range, not precision."""
    torch.manual_seed(0)
    tensor = torch.randn(256, 768)

    precise = relative_error(quantize_dequantize(tensor, dtype=FORWARD_DTYPE), tensor)
    wide = relative_error(quantize_dequantize(tensor, dtype=GRADIENT_DTYPE), tensor)

    assert wide > precise
    assert torch.finfo(GRADIENT_DTYPE).max > torch.finfo(FORWARD_DTYPE).max * 100


def test_the_vocabulary_projection_changes_the_predicted_token(monkeypatch) -> None:
    """Demonstrate why this project must not use FP8 for vocabulary projection.

    Quantizing hidden states and weights to E4M3 changed measured argmax results
    by 6.45% with a 48,000-piece vocabulary. In greedy decoding, every changed
    argmax is a different output token.
    """
    torch.manual_seed(0)
    hidden = torch.randn(256, 768)
    projection = torch.randn(8192, 768) * 0.02

    exact = (hidden.float() @ projection.float().T).argmax(-1)
    quantized = (
        quantize_dequantize(hidden).float() @ quantize_dequantize(projection).float().T
    ).argmax(-1)

    mismatch = float((quantized != exact).float().mean())
    assert mismatch > 0.01, "이 경고가 무의미해졌다면 측정을 다시 하십시오"


# ── Policy ──────────────────────────────────────────────────────────────


def test_the_policy_is_off_by_default() -> None:
    policy = Fp8Policy()
    assert not policy.enabled
    assert not policy.allows("encoder_layers.0.self_attn.q_proj.weight")


@pytest.mark.parametrize(
    "name",
    [
        "encoder_layers.0.self_attn.q_proj.weight",
        "encoder_layers.3.self_attn.k_proj.weight",
        "decoder_layers.1.cross_attn.v_proj.weight",
        "decoder_layers.2.self_attn.out_proj.weight",
        "encoder_layers.0.ffn.gate_proj.weight",
        "encoder_layers.0.ffn.up_proj.weight",
        "decoder_layers.0.ffn.down_proj.weight",
    ],
)
def test_the_big_projections_are_quantizable(name: str) -> None:
    assert Fp8Policy(enabled=True, scope=SCOPE_ALL).allows(name)


@pytest.mark.parametrize(
    "name",
    [
        "token_embedding.weight",
        "lm_head.weight",
        "encoder_norm.weight",
        "encoder_layers.0.attn_norm.weight",
        "register_state.register_embeddings.weight",
        "typed_memory.type_embedding.weight",
        "evidence_repair.uncertainty_head.weight",
    ],
)
def test_the_protected_tensors_are_never_quantized(name: str) -> None:
    assert not Fp8Policy(enabled=True, scope=SCOPE_ALL).allows(name)


def test_the_tied_embedding_is_protected_because_it_is_also_the_output_head() -> None:
    """With ``tie_embeddings=True``, output projection is the embedding matrix.

    Storing this weight in FP8 damages both output projection and input embedding
    lookup.
    """
    assert not Fp8Policy(enabled=True, scope=SCOPE_ALL).allows("token_embedding.weight")
    assert (
        Fp8Policy(enabled=True, scope=SCOPE_ALL, quantize_vocabulary_projection=True).allows(
            "token_embedding.weight"
        )
        is False
    )  # It remains excluded because it is not in QUANTIZABLE_PROJECTIONS.


def test_the_policy_rejects_a_non_power_of_two_block() -> None:
    with pytest.raises(ValueError, match="power of two"):
        Fp8Policy(enabled=True, block=100).validate()
    with pytest.raises(ValueError, match="positive"):
        Fp8Policy(enabled=True, block=0).validate()
    Fp8Policy(enabled=True, block=DEFAULT_BLOCK).validate()


def test_hardware_support_is_reported_separately_from_dtype_availability() -> None:
    """Dtype availability and kernel availability are separate capabilities.

    A CPU can cast to FP8 but cannot run ``_scaled_mm``.
    """
    assert torch.zeros(1).to(FORWARD_DTYPE).dtype is FORWARD_DTYPE
    assert not fp8_gemm_supported(torch.device("cpu"))


# ── Scope: defaults derived from measurements ──────────────────────────


def test_the_default_scope_is_ffn_only() -> None:
    """Measured logit error is 6.39% for FFN-only and 13.11% for all layers.

    Attention-projection error is amplified through softmax, while FFN error is
    only added to the residual stream.
    """
    policy = Fp8Policy(enabled=True)
    assert policy.scope == SCOPE_FFN
    assert policy.allows("encoder_layers.0.ffn.gate_proj.weight")
    assert not policy.allows("encoder_layers.0.self_attn.q_proj.weight")
    assert not policy.allows("decoder_layers.0.cross_attn.out_proj.weight")


def test_the_all_scope_adds_the_attention_projections() -> None:
    policy = Fp8Policy(enabled=True, scope=SCOPE_ALL)
    assert policy.allows("encoder_layers.0.ffn.gate_proj.weight")
    assert policy.allows("encoder_layers.0.self_attn.q_proj.weight")
    assert not policy.allows("token_embedding.weight")


def test_an_unknown_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="fp8 scope"):
        Fp8Policy(enabled=True, scope="everything").validate()
    with pytest.raises(ValueError, match="fp8 scope"):
        Fp8Policy(enabled=True, scope="everything").allows("x.ffn.gate_proj.weight")


def test_weight_only_quantization_beats_quantizing_both_operands() -> None:
    """Weight-only quantization has less error than quantizing both operands.

    Measured error was 2.57% for weights only and 3.63% for both operands.
    Weight-only quantization is also unaffected by activation outliers. This test
    makes no claim about runtime speed.
    """
    torch.manual_seed(0)
    activations = torch.randn(256, 768)
    weights = torch.randn(1024, 768) * 0.02
    exact = activations.float() @ weights.float().T

    weight_only = relative_error(
        activations.bfloat16().float() @ quantize_dequantize(weights).float().T,
        exact,
    )
    both = relative_error(
        quantize_dequantize(activations).float() @ quantize_dequantize(weights).float().T,
        exact,
    )
    assert weight_only < both


def test_quantization_error_compounds_with_depth() -> None:
    """A single GEMM error does not characterize a complete model.

    Measured error grew from 2.57% for one GEMM to 11.7% after 16 encoder layers
    and 13.1% at final logits. Independent quantization noise accumulates in the
    residual stream.
    """
    import copy

    from sion_translate.config import ExperimentalConfig, ModelConfig
    from sion_translate.model import SionForConditionalGeneration

    config = ModelConfig(
        vocab_size=512,
        d_model=128,
        encoder_layers=8,
        decoder_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=256,
        max_seq_len=64,
        dropout=0.0,
        experimental=ExperimentalConfig(),
    )
    torch.manual_seed(0)
    reference = SionForConditionalGeneration(config).eval()
    quantized = copy.deepcopy(reference)
    policy = Fp8Policy(enabled=True, scope=SCOPE_ALL)
    with torch.no_grad():
        for name, parameter in quantized.named_parameters():
            if policy.allows(name):
                parameter.copy_(quantize_dequantize(parameter))

    ids = torch.randint(4, 500, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.bool)
    with torch.no_grad():
        deep = relative_error(quantized.encode(ids, mask), reference.encode(ids, mask))

    # Use a comparable baseline: the model path quantizes only weights, so compare
    # it with the quantization error of one individual weight tensor.
    with torch.no_grad():
        per_tensor = [
            relative_error(quantize_dequantize(parameter), parameter)
            for name, parameter in reference.named_parameters()
            if policy.allows(name)
        ]
    single = sum(per_tensor) / len(per_tensor)

    assert deep > single, (deep, single)
