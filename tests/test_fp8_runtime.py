"""FP8 가중치를 FP8 인 채로 들고 추론하는 경로.

불러올 때 고정밀도로 되돌려 상주시켜 버리면 디스크만 줄어듭니다. 여기서
확인하는 것은 **상주 가중치가 FP8 인가** 와 실제 계산 경로를 정확히 보고하는가
입니다.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import sion_translate.fp8_runtime as fp8_runtime
from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.fp8 import FORWARD_DTYPE, Fp8Policy
from sion_translate.fp8_runtime import (
    Fp8Linear,
    apply_fp8_weights,
    describe_runtime,
    prepare_fp8_model_for_device,
    resolve_fp8_compute_dtype,
)
from sion_translate.model import SionForConditionalGeneration
from sion_translate.training.export import _pack_fp8_state, _unpack_fp8_state


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=512,
        d_model=128,
        encoder_layers=2,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=256,
        max_seq_len=64,
        dropout=0.0,
        experimental=ExperimentalConfig(),
    )


def _packed():
    torch.manual_seed(0)
    model = SionForConditionalGeneration(_config()).eval()
    packed, _ = _pack_fp8_state(model.state_dict(), Fp8Policy(enabled=True))
    return packed


def _loaded(packed):
    model = SionForConditionalGeneration(_config()).eval()
    model.load_state_dict(_unpack_fp8_state(packed))
    return model


def test_the_ffn_projections_become_fp8_modules() -> None:
    packed = _packed()
    model = _loaded(packed)
    replaced = apply_fp8_weights(model, packed)

    assert replaced > 0
    fp8_modules = [m for m in model.modules() if isinstance(m, Fp8Linear)]
    assert len(fp8_modules) == replaced
    for module in fp8_modules:
        assert module.weight.dtype is FORWARD_DTYPE


def test_the_resident_weights_are_actually_fp8_not_restored() -> None:
    """이 테스트가 이 모듈의 존재 이유다.

    되돌려 실은 채 두면 export 는 성공하고 대역폭은 그대로입니다.
    """
    packed = _packed()
    model = _loaded(packed)
    apply_fp8_weights(model, packed)

    bytes_per_element = {}
    for module in model.modules():
        if isinstance(module, Fp8Linear):
            bytes_per_element[module] = (
                module.weight.element_size() * module.weight.numel()
                + module.scales.element_size() * module.scales.numel()
            ) / module.weight.numel()
    assert bytes_per_element
    # bf16 이면 2.0. 스케일 오버헤드를 포함해도 1.1 을 넘지 않아야 한다.
    assert max(bytes_per_element.values()) < 1.1


def test_the_attention_projections_stay_dense_under_the_default_scope() -> None:
    packed = _packed()
    model = _loaded(packed)
    apply_fp8_weights(model, packed)

    for name, module in model.named_modules():
        if any(token in name for token in ("q_proj", "k_proj", "v_proj", "out_proj")):
            assert isinstance(module, nn.Linear), name
            assert not isinstance(module, Fp8Linear), name


def test_the_model_still_runs_and_matches_the_restored_weights() -> None:
    """FP8 상주 모델과 되돌려 실은 모델은 같은 가중치를 본다.

    남는 차이는 계산 정밀도(bf16 대 fp32)뿐이라 작아야 합니다.
    """
    packed = _packed()
    restored = _loaded(packed)
    resident = _loaded(packed)
    apply_fp8_weights(resident, packed)

    ids = torch.randint(4, 500, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.bool)
    decoder_ids = torch.randint(4, 500, (2, 8))
    with torch.no_grad():
        a = restored(ids, mask, decoder_ids).logits.float()
        b = resident(ids, mask, decoder_ids).logits.float()

    assert torch.isfinite(b).all()
    assert (a - b).abs().max() < 0.05


def test_fp16_fallback_prepares_a_bf16_model_end_to_end_without_casting_fp8(
    monkeypatch,
) -> None:
    packed = _packed()
    resident = _loaded(packed).to(torch.bfloat16)
    apply_fp8_weights(resident, packed)
    fp8_before = [
        (module.weight.clone(), module.scales.clone())
        for module in resident.modules()
        if isinstance(module, Fp8Linear)
    ]
    monkeypatch.setattr(
        fp8_runtime,
        "resolve_fp8_compute_dtype",
        lambda _device: torch.float16,
    )

    selected = prepare_fp8_model_for_device(resident, torch.device("cpu"))

    ids = torch.randint(4, 500, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.bool)
    decoder_ids = torch.randint(4, 500, (2, 8))
    with torch.no_grad():
        logits = resident(ids, mask, decoder_ids).logits

    assert selected is torch.float16
    assert torch.isfinite(logits).all()
    assert {parameter.dtype for parameter in resident.parameters()} == {torch.float16}
    fp8_after = [
        (module.weight, module.scales)
        for module in resident.modules()
        if isinstance(module, Fp8Linear)
    ]
    assert len(fp8_after) == len(fp8_before) > 0
    for (weight_before, scales_before), (weight_after, scales_after) in zip(
        fp8_before,
        fp8_after,
        strict=True,
    ):
        assert weight_after.dtype is FORWARD_DTYPE
        assert scales_after.dtype is scales_before.dtype
        assert torch.equal(weight_after, weight_before)
        assert torch.equal(scales_after, scales_before)


def test_a_shape_mismatch_between_weight_and_scales_is_refused() -> None:
    weight = torch.zeros(8, 256).to(FORWARD_DTYPE)
    with pytest.raises(ValueError, match="scale shape"):
        Fp8Linear(weight, torch.ones(8, 3))


def test_a_non_fp8_weight_is_refused() -> None:
    with pytest.raises(ValueError, match="expects"):
        Fp8Linear(torch.zeros(8, 256), torch.ones(8, 2))


def test_a_width_that_is_not_a_multiple_of_the_block_is_refused() -> None:
    weight = torch.zeros(8, 130).to(FORWARD_DTYPE)
    with pytest.raises(ValueError, match="multiple of the FP8 block size"):
        Fp8Linear(weight, torch.ones(8, 2))


def test_replacing_a_module_that_is_not_linear_is_refused() -> None:
    packed = _packed()
    model = _loaded(packed)
    name = next(k for k, v in packed.items() if v["kind"] == "block_fp8")
    renamed = dict(packed)
    renamed["encoder_norm.weight"] = renamed.pop(name)
    with pytest.raises(ValueError, match="not nn.Linear"):
        apply_fp8_weights(model, renamed)


def test_auto_compute_dtype_is_resolved_at_forward_time(monkeypatch) -> None:
    weight = torch.zeros(8, 256).to(FORWARD_DTYPE)
    module = Fp8Linear(weight, torch.ones(8, 2))
    monkeypatch.setattr(
        fp8_runtime,
        "resolve_fp8_compute_dtype",
        lambda _device: torch.float16,
    )

    assert module.compute_dtype is None
    assert module.dequantized_weight().dtype is torch.float16


def test_an_explicit_compute_dtype_overrides_auto_selection(monkeypatch) -> None:
    weight = torch.zeros(8, 256).to(FORWARD_DTYPE)
    module = Fp8Linear(weight, torch.ones(8, 2), compute_dtype=torch.float32)
    monkeypatch.setattr(
        fp8_runtime,
        "resolve_fp8_compute_dtype",
        lambda _device: torch.float16,
    )

    assert module.dequantized_weight().dtype is torch.float32


def test_fp8_linear_returns_the_residual_input_dtype() -> None:
    weight = torch.zeros(8, 256).to(FORWARD_DTYPE)
    module = Fp8Linear(weight, torch.ones(8, 2), compute_dtype=torch.float16)
    hidden = torch.randn(2, 256, dtype=torch.bfloat16)

    output = module(hidden)

    assert output.dtype is hidden.dtype


def test_cpu_fp8_runtime_keeps_the_existing_bf16_dense_path() -> None:
    device = torch.device("cpu")

    assert resolve_fp8_compute_dtype(device) is torch.bfloat16
    description = describe_runtime(device)

    assert "FP8 상주 가중치" in description
    assert "BF16 즉시 역양자화" in description
    assert "CPU dense fallback" in description


@pytest.mark.parametrize(
    ("capability", "expected_dtype", "compute_name", "hardware_description"),
    [
        ((8, 0), torch.bfloat16, "BF16", "네이티브 FP8 텐서코어 미지원 장치 fallback"),
        ((7, 0), torch.float16, "FP16", "네이티브 FP8 텐서코어 미지원 장치 fallback"),
        (
            (9, 0),
            torch.bfloat16,
            "BF16",
            "네이티브 FP8 텐서코어 지원 장치이지만 현재 경로에서는 미사용",
        ),
    ],
    ids=("a100", "bf16-unsupported", "h100"),
)
def test_cuda_fp8_runtime_falls_back_without_requiring_native_fp8(
    monkeypatch,
    capability: tuple[int, int],
    expected_dtype: torch.dtype,
    compute_name: str,
    hardware_description: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: capability)
    device = torch.device("cuda", 0)

    assert resolve_fp8_compute_dtype(device) is expected_dtype
    description = describe_runtime(device)

    assert "FP8 상주 가중치" in description
    assert f"{compute_name} 즉시 역양자화" in description
    assert "dense GEMM" in description
    assert "상주 메모리 절감" in description
    assert hardware_description in description
    assert "대역폭" not in description
