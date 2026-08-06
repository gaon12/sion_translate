"""FP8 export 왕복.

여기서 확인하는 것은 "FP8 로 저장된다"가 아니라 **무엇이 FP8 로 저장되지
않는가** 입니다. 어휘 projection 이 섞여 들어가면 export 는 성공하고 번역만
망가집니다.
"""

from __future__ import annotations

import torch

from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.fp8 import Fp8Policy
from sion_translate.model import SionForConditionalGeneration
from sion_translate.training.export import (
    SUPPORTED_FORMATS,
    _pack_fp8_state,
    _unpack_fp8_state,
)


def relative_error(approximate: torch.Tensor, exact: torch.Tensor) -> float:
    denominator = exact.float().norm()
    if denominator == 0:
        return float(approximate.float().norm())
    return float((approximate.float() - exact.float()).norm() / denominator)


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


def _state():
    torch.manual_seed(0)
    return SionForConditionalGeneration(_config()).state_dict()


def test_fp8_is_an_export_format() -> None:
    assert "fp8" in SUPPORTED_FORMATS


def test_the_round_trip_restores_every_key_and_shape() -> None:
    state = _state()
    packed, _ = _pack_fp8_state(state, Fp8Policy(enabled=True))
    restored = _unpack_fp8_state(packed)

    assert set(restored) == set(state)
    for name, tensor in state.items():
        assert restored[name].shape == tensor.shape, name
        assert restored[name].dtype == tensor.dtype, name


def test_only_the_ffn_projections_are_quantized_by_default() -> None:
    state = _state()
    packed, quantization = _pack_fp8_state(state, Fp8Policy(enabled=True))

    quantized = {name for name, entry in packed.items() if entry["kind"] == "block_fp8"}
    assert quantized
    assert all(
        any(token in name for token in ("gate_proj", "up_proj", "down_proj")) for name in quantized
    )
    assert quantization["scope"] == "ffn"
    assert 0.0 < quantization["quantized_fraction"] < 1.0


def test_the_tied_vocabulary_matrix_is_never_quantized() -> None:
    """tie_embeddings 라 이 행렬은 임베딩이자 출력 헤드다.

    FP8 로 저장하면 export 는 성공하고 번역만 조용히 망가집니다.
    """
    state = _state()
    for scope in ("ffn", "all"):
        packed, _ = _pack_fp8_state(state, Fp8Policy(enabled=True, scope=scope))
        assert packed["token_embedding.weight"]["kind"] == "tensor", scope


def test_norms_are_never_quantized() -> None:
    state = _state()
    packed, _ = _pack_fp8_state(state, Fp8Policy(enabled=True, scope="all"))
    for name, entry in packed.items():
        if "norm" in name:
            assert entry["kind"] == "tensor", name


def test_quantized_weights_survive_the_round_trip_within_the_measured_bound() -> None:
    state = _state()
    packed, _ = _pack_fp8_state(state, Fp8Policy(enabled=True))
    restored = _unpack_fp8_state(packed)

    for name, entry in packed.items():
        if entry["kind"] != "block_fp8":
            # 양자화하지 않은 것은 비트까지 같아야 한다.
            assert torch.equal(restored[name], state[name]), name
        else:
            assert relative_error(restored[name], state[name]) < 0.05, name


def test_the_all_scope_covers_more_parameters_than_ffn() -> None:
    state = _state()
    _, ffn = _pack_fp8_state(state, Fp8Policy(enabled=True, scope="ffn"))
    _, everything = _pack_fp8_state(state, Fp8Policy(enabled=True, scope="all"))
    assert everything["quantized_fraction"] > ffn["quantized_fraction"]


def test_the_manifest_records_what_was_done() -> None:
    _, quantization = _pack_fp8_state(_state(), Fp8Policy(enabled=True))
    assert quantization["algorithm"] == "weight-only-fp8-e4m3-blockwise"
    assert quantization["activation_dtype"] == "bfloat16"
    assert quantization["weight_dtype"] == "float8_e4m3fn"
    assert quantization["block"] == 128


def test_a_tensor_whose_width_is_not_a_multiple_of_the_block_is_preserved() -> None:
    """조용히 자르거나 채우지 않고 고정밀도로 남긴다."""
    state = {"encoder_layers.0.ffn.gate_proj.weight": torch.randn(8, 130)}
    packed, quantization = _pack_fp8_state(state, Fp8Policy(enabled=True))
    assert packed["encoder_layers.0.ffn.gate_proj.weight"]["kind"] == "tensor"
    assert quantization["quantized_parameters"] == 0
