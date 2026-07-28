"""공유 블록 hidden-state 반복 검증."""

from __future__ import annotations

import warnings

import pytest
import torch

from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.model import SionForConditionalGeneration


def _config(**experimental) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        encoder_layers=4,
        decoder_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
        experimental=ExperimentalConfig(**experimental),
    )


def _inputs(length: int = 6):
    input_ids = torch.randint(4, 60, (2, length))
    attention_mask = torch.ones(2, length, dtype=torch.bool)
    return input_ids, attention_mask


def test_disabled_by_default_so_existing_checkpoints_load() -> None:
    plain = SionForConditionalGeneration(_config())
    recurrent = SionForConditionalGeneration(_config(recurrent_block_layers=2, recurrent_steps=3))
    # 가중치를 새로 만들지 않으므로 state_dict 모양이 같아야 한다.
    assert plain.state_dict().keys() == recurrent.state_dict().keys()
    recurrent.load_state_dict(plain.state_dict())


def test_recurrence_changes_the_encoder_output() -> None:
    torch.manual_seed(0)
    config = _config()
    plain = SionForConditionalGeneration(config).eval()
    recurrent = SionForConditionalGeneration(
        _config(recurrent_block_layers=2, recurrent_steps=3)
    ).eval()
    recurrent.load_state_dict(plain.state_dict())

    input_ids, attention_mask = _inputs()
    with torch.no_grad():
        base = plain.encode(input_ids, attention_mask)
        looped = recurrent.encode(input_ids, attention_mask)
    assert base.shape == looped.shape
    # 같은 가중치인데도 계산이 달라야 반복이 실제로 일어난 것이다.
    assert not torch.allclose(base, looped, atol=1e-5)


def test_one_step_is_identical_to_no_recurrence() -> None:
    torch.manual_seed(1)
    plain = SionForConditionalGeneration(_config()).eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        once = SionForConditionalGeneration(
            _config(recurrent_block_layers=3, recurrent_steps=1)
        ).eval()
    once.load_state_dict(plain.state_dict())

    input_ids, attention_mask = _inputs()
    with torch.no_grad():
        assert torch.allclose(
            plain.encode(input_ids, attention_mask),
            once.encode(input_ids, attention_mask),
            atol=1e-6,
        )


def test_block_larger_than_the_encoder_is_clamped() -> None:
    model = SionForConditionalGeneration(_config(recurrent_block_layers=99, recurrent_steps=2))
    assert model.recurrent_block_layers == 4  # encoder_layers
    input_ids, attention_mask = _inputs()
    with torch.no_grad():
        assert model.encode(input_ids, attention_mask).shape == (2, 6, 32)


def test_gradients_flow_through_every_repeat() -> None:
    torch.manual_seed(2)
    model = SionForConditionalGeneration(_config(recurrent_block_layers=2, recurrent_steps=3))
    input_ids, attention_mask = _inputs()
    model.encode(input_ids, attention_mask).sum().backward()
    # 반복된 블록의 가중치는 여러 경로에서 기울기를 받아야 한다.
    last_layer = model.encoder_layers[-1]
    grads = [p.grad for p in last_layer.parameters() if p.requires_grad]
    assert grads and all(grad is not None for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)


def test_forward_and_generate_still_work_with_recurrence() -> None:
    torch.manual_seed(3)
    model = SionForConditionalGeneration(
        _config(recurrent_block_layers=2, recurrent_steps=2)
    ).eval()
    input_ids, attention_mask = _inputs()
    with torch.no_grad():
        generated = model.generate(
            input_ids, attention_mask, bos_id=2, eos_id=3, max_new_tokens=4, num_beams=2
        )
    assert generated.shape[0] == input_ids.shape[0]


def test_config_rejects_invalid_recurrence_settings() -> None:
    with pytest.raises(ValueError, match="recurrent_block_layers"):
        ExperimentalConfig(recurrent_block_layers=-1).validate()
    with pytest.raises(ValueError, match="recurrent_steps"):
        ExperimentalConfig(recurrent_steps=0).validate()


def test_config_warns_when_recurrence_would_do_nothing() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ExperimentalConfig(recurrent_block_layers=2, recurrent_steps=1).validate()
    assert len(caught) == 1
    assert "recurrent_steps" in str(caught[0].message)
