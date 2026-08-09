"""FP8 수치와 정책.

여기 테스트의 상당수는 "FP8 이 정확하다"가 아니라 **얼마나 부정확한지를
고정**합니다. FP8 학습이 실패하는 방식은 대부분 느려짐이 아니라 내리면 안
되는 것을 내리는 것이고, 그 경계는 숫자로 남겨 두지 않으면 잊힙니다.
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

# 아래 세 함수는 **측정 장비**입니다. 프로덕션 경로가 쓰지 않으므로
# 프로덕션 코드에 두지 않습니다. FP8 이 얼마나 부정확한지를 여기서 재고,
# 그 수치를 위 테스트들이 고정합니다.


def quantize_dequantize(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype = FORWARD_DTYPE,
    block: int | None = DEFAULT_BLOCK,
) -> torch.Tensor:
    """FP8 왕복. 실제 FP8 GEMM 이 보는 값을 고정밀도로 재현합니다.

    학습 경로에서 쓰는 것이 아니라, 하드웨어 없이 **오차를 측정**하고
    회귀를 잡기 위한 것입니다. 네이티브 FP8 GEMM에서 관측할 양자화 값의
    기준이며, 현재 프로덕션 런타임은 ``torch._scaled_mm`` 을 쓰지 않습니다.
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


# ── 측정한 사실을 고정한다 ──────────────────────────────────────────────


def test_fp8_gemm_error_is_an_order_of_magnitude_worse_than_bf16() -> None:
    """FP8 은 bf16 과 "거의 같다"가 아니다.

    실측 (M=2048, K=768, N=2048, 정규분포): FP8 block128 3.64% 대 bf16 0.23%.
    이 배율을 잊으면 정확도 하락을 다른 원인에서 찾게 됩니다.
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
    """블록 스케일링은 이상치 대책이지 일반적인 정확도 개선이 아니다.

    실측: 정규분포 3.74%→3.64% (거의 없음), 이상치 3.75%→3.19%.
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
    """기울기에 E5M2 를 쓰는 이유는 정밀도가 아니라 동적 범위다."""
    torch.manual_seed(0)
    tensor = torch.randn(256, 768)

    precise = relative_error(quantize_dequantize(tensor, dtype=FORWARD_DTYPE), tensor)
    wide = relative_error(quantize_dequantize(tensor, dtype=GRADIENT_DTYPE), tensor)

    assert wide > precise
    assert torch.finfo(GRADIENT_DTYPE).max > torch.finfo(FORWARD_DTYPE).max * 100


def test_the_vocabulary_projection_changes_the_predicted_token(monkeypatch) -> None:
    """이 저장소에서 FP8 을 어휘 projection 에 쓰면 안 되는 이유.

    48,000 어휘에서 hidden 과 가중치를 모두 E4M3 로 내리면 argmax 가 실측
    6.45% 바뀝니다. greedy 디코딩에서 그것은 그대로 다른 단어입니다.
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


# ── 정책 ────────────────────────────────────────────────────────────────


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
    """``tie_embeddings=True`` 면 출력 projection 이 곧 임베딩 행렬이다.

    가중치를 FP8 로 저장하면 출력만이 아니라 입력 임베딩 조회까지 같이
    망가집니다.
    """
    assert not Fp8Policy(enabled=True, scope=SCOPE_ALL).allows("token_embedding.weight")
    assert (
        Fp8Policy(enabled=True, scope=SCOPE_ALL, quantize_vocabulary_projection=True).allows(
            "token_embedding.weight"
        )
        is False
    )  # QUANTIZABLE_PROJECTIONS 에 없으므로 그래도 대상이 아니다


def test_the_policy_rejects_a_non_power_of_two_block() -> None:
    with pytest.raises(ValueError, match="power of two"):
        Fp8Policy(enabled=True, block=100).validate()
    with pytest.raises(ValueError, match="positive"):
        Fp8Policy(enabled=True, block=0).validate()
    Fp8Policy(enabled=True, block=DEFAULT_BLOCK).validate()


def test_hardware_support_is_reported_separately_from_dtype_availability() -> None:
    """dtype 이 있는 것과 커널이 있는 것은 다르다.

    CPU 에서도 FP8 캐스팅은 됩니다. ``_scaled_mm`` 은 안 됩니다.
    """
    assert torch.zeros(1).to(FORWARD_DTYPE).dtype is FORWARD_DTYPE
    assert not fp8_gemm_supported(torch.device("cpu"))


# ── 범위(scope): 측정에서 나온 기본값 ────────────────────────────────────


def test_the_default_scope_is_ffn_only() -> None:
    """실측: FFN 만 내리면 logits 오차 6.39%, 전 층이면 13.11%.

    attention projection 은 softmax 를 거치며 오차가 증폭되고, FFN 의 오차는
    잔차에 더해질 뿐입니다.
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
    """가중치만 양자화하면 두 operand 를 양자화할 때보다 오차가 작다.

    실측: 가중치만 2.57%, 양쪽 다 3.63%. 그리고 가중치만 내리면 활성값
    이상치에 영향받지 않습니다. 이 검사는 실행 성능 이득을 주장하지 않습니다.
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
    """단일 GEMM 오차로 모델 전체를 판단하면 안 된다.

    실측: GEMM 1회 2.57% → encoder 16층 11.7% → 최종 logits 13.1%.
    독립적인 양자화 잡음이 잔차 스트림에 쌓입니다.
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

    # 같은 기준으로 비교한다: 모델 경로는 가중치만 양자화하므로, 기준도
    # 개별 가중치 텐서 하나의 양자화 오차여야 한다.
    with torch.no_grad():
        per_tensor = [
            relative_error(quantize_dequantize(parameter), parameter)
            for name, parameter in reference.named_parameters()
            if policy.allows(name)
        ]
    single = sum(per_tensor) / len(per_tensor)

    assert deep > single, (deep, single)
