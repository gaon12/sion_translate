"""FP8 가중치를 FP8 인 채로 들고 추론하는 경로.

불러올 때 고정밀도로 되돌려 상주시켜 버리면 디스크만 줄어듭니다. 여기서
확인하는 것은 **상주 가중치가 FP8 인가** 와 실제 계산 경로를 정확히 보고하는가
입니다.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.fp8 import FORWARD_DTYPE, Fp8Policy
from sion_translate.fp8_runtime import Fp8Linear, apply_fp8_weights, describe_runtime
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


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_the_runtime_description_reports_the_actual_path_on_every_device(
    device: torch.device,
) -> None:
    description = describe_runtime(device)

    assert "BF16 즉시 역양자화" in description
    assert "dense GEMM" in description
    assert "상주 메모리 절감" in description
    assert "네이티브 FP8 텐서코어 미사용" in description
    assert "대역폭" not in description


def test_the_runtime_description_does_not_change_with_cuda_capability() -> None:
    assert describe_runtime(torch.device("cpu")) == describe_runtime(torch.device("cuda", 7))
