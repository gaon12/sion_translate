from __future__ import annotations

import pytest
import torch

from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.hf import SionConfig
from sion_translate.hf import SionForConditionalGeneration as HFSionForConditionalGeneration
from sion_translate.model import SionForConditionalGeneration


def _config(*, evidence_repair_enabled: bool = True) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
        gradient_checkpointing=False,
        experimental=ExperimentalConfig(
            evidence_repair_enabled=evidence_repair_enabled,
        ),
    )


def _batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[4, 5, 3, 0], [6, 7, 8, 3]])
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0),
        "decoder_input_ids": torch.tensor([[2, 9, 0], [2, 10, 11]]),
        "labels": torch.tensor([[9, 3, -100], [10, 11, 3]]),
    }


@pytest.mark.parametrize("reasoning_level", [-1, 10])
def test_reasoning_level_must_be_in_the_public_zero_to_nine_range(
    reasoning_level: int,
) -> None:
    model = SionForConditionalGeneration(_config())

    with pytest.raises(ValueError, match="between 0 and 9"):
        model(**_batch(), reasoning_level=reasoning_level)


@pytest.mark.parametrize("reasoning_level", [True, 1.0, "1"])
def test_reasoning_level_rejects_non_integer_values(reasoning_level: object) -> None:
    model = SionForConditionalGeneration(_config())

    with pytest.raises(TypeError, match="integer from 0 to 9"):
        model(**_batch(), reasoning_level=reasoning_level)  # type: ignore[arg-type]


def test_level_zero_matches_the_direct_model_and_bypasses_all_repair_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(31)  # pyright: ignore[reportUnknownMemberType]
    direct_model = SionForConditionalGeneration(_config(evidence_repair_enabled=False))
    torch.manual_seed(31)  # pyright: ignore[reportUnknownMemberType]
    ssrt_model = SionForConditionalGeneration(_config())
    assert ssrt_model.evidence_repair is not None

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reasoning level 0 must not enter the auditor/reasoner")

    monkeypatch.setattr(ssrt_model.evidence_repair, "forward", fail)
    monkeypatch.setattr(ssrt_model.evidence_repair, "project_key_value", fail)

    direct = direct_model(**_batch())
    level_zero = ssrt_model(**_batch(), reasoning_level=0)
    torch.testing.assert_close(level_zero.logits, direct.logits)
    assert level_zero.reasoning_level is not None
    assert level_zero.reasoning_level.item() == 0
    assert level_zero.reasoning_budget is not None
    assert level_zero.reasoning_budget.item() == 0.0
    assert level_zero.reasoning_active is not None
    assert not level_zero.reasoning_active.item()
    assert level_zero.evidence_request_rate is not None
    assert level_zero.evidence_request_rate.item() == 0.0

    context = ssrt_model.prepare_generation(
        _batch()["input_ids"],
        _batch()["attention_mask"],
        reasoning_level=0,
    )
    assert context.reasoning_level == 0
    assert context.evidence_key_value is None
    generated = ssrt_model.generate(
        _batch()["input_ids"],
        _batch()["attention_mask"],
        bos_id=2,
        eos_id=3,
        max_new_tokens=2,
        generation_context=context,
        reasoning_level=0,
    )
    assert generated.shape[0] == _batch()["input_ids"].shape[0]


def test_levels_condition_the_request_budget_without_changing_checkpoint_keys() -> None:
    model = SionForConditionalGeneration(_config())
    assert model.evidence_repair is not None
    with torch.no_grad():
        model.evidence_repair.uncertainty_head.weight.zero_()
        model.evidence_repair.uncertainty_head.bias.zero_()
    original_keys = tuple(model.state_dict())

    legacy = model(**_batch())
    level_one = model(**_batch(), reasoning_level=1)
    level_nine = model(**_batch(), reasoning_level=9)

    assert legacy.reasoning_level is None
    assert legacy.evidence_request_rate is not None
    assert level_one.evidence_request_rate is not None
    assert level_nine.evidence_request_rate is not None
    assert level_one.reasoning_budget is not None
    assert level_nine.reasoning_budget is not None
    assert level_one.reasoning_budget.item() == pytest.approx(1.0 / 9.0)
    assert level_nine.reasoning_budget.item() == 1.0
    assert level_one.evidence_request_rate.item() == pytest.approx(0.5 / 9.0)
    assert level_nine.evidence_request_rate.item() == pytest.approx(0.5)
    torch.testing.assert_close(legacy.logits, level_nine.logits)
    assert tuple(model.state_dict()) == original_keys


def test_generation_context_keeps_the_level_and_rejects_an_explicit_mismatch() -> None:
    model = SionForConditionalGeneration(_config())
    batch = _batch()
    context = model.prepare_generation(
        batch["input_ids"],
        batch["attention_mask"],
        reasoning_level=3,
    )

    assert context.reasoning_level == 3
    assert context.evidence_key_value is not None
    with pytest.raises(ValueError, match="reasoning_level does not match"):
        model.generate(
            batch["input_ids"],
            batch["attention_mask"],
            bos_id=2,
            eos_id=3,
            max_new_tokens=2,
            generation_context=context,
            reasoning_level=4,
        )


def test_transformers_forward_and_generation_preserve_the_reasoning_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_config = _config()
    config = SionConfig.from_model_config(  # pyright: ignore[reportUnknownMemberType]
        native_config,
        languages=["ko", "ja"],
        language_pairs=[["ko", "ja"]],
    )
    model = HFSionForConditionalGeneration(config).eval()
    assert model.model.evidence_repair is not None

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the Transformers level-0 path must bypass reasoning")

    monkeypatch.setattr(model.model.evidence_repair, "forward", fail)
    monkeypatch.setattr(model.model.evidence_repair, "project_key_value", fail)
    batch = _batch()
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=batch["decoder_input_ids"],
        reasoning_level=0,
    )
    assert output.logits.shape[:2] == batch["decoder_input_ids"].shape

    generated = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=2,
        num_beams=1,
        reasoning_level=0,
    )
    assert isinstance(generated, torch.Tensor)
    assert generated.shape[0] == batch["input_ids"].shape[0]
