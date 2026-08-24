"""Test hidden-state recurrence through a shared block."""

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
    # Recurrence adds no weights, so the state-dictionary structure must match.
    assert plain.state_dict().keys() == recurrent.state_dict().keys()
    recurrent.load_state_dict(plain.state_dict())


def test_repeats_the_block_as_a_unit_not_each_layer() -> None:
    """Repeat the final N layers together as one block.

    Per-layer recurrence (L2 three times, then L3 three times) differs from block
    recurrence ((L2, L3) three times). The configuration promises the latter, so
    the test checks it directly.
    """
    torch.manual_seed(4)
    steps, block = 3, 2
    model = SionForConditionalGeneration(
        _config(recurrent_block_layers=block, recurrent_steps=steps)
    ).eval()
    input_ids, attention_mask = _inputs()
    boundary = len(model.encoder_layers) - block

    with torch.no_grad():
        actual = model.encode(input_ids, attention_mask)

        hidden = model._embed(input_ids)
        for layer in model.encoder_layers[:boundary]:
            hidden = layer(hidden, attention_mask)
        for _ in range(steps):
            for layer in model.encoder_layers[boundary:]:
                hidden = layer(hidden, attention_mask)
        per_block = model.encoder_norm(hidden)

        hidden = model._embed(input_ids)
        for index, layer in enumerate(model.encoder_layers):
            for _ in range(steps if index >= boundary else 1):
                hidden = layer(hidden, attention_mask)
        per_layer = model.encoder_norm(hidden)

    assert torch.allclose(actual, per_block, atol=1e-6)
    # The fixture must make both interpretations differ for the assertion to matter.
    assert not torch.allclose(per_block, per_layer, atol=1e-5)


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
    # Identical weights must compute a different result when recurrence is active.
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
    # Reused block weights must receive gradients through every recurrent path.
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
