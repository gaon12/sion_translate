from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from sion_translate.cli.translate import build_parser
from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.hf import SionConfig
from sion_translate.hf import SionForConditionalGeneration as HFSionForConditionalGeneration
from sion_translate.model import SionForConditionalGeneration
from sion_translate.training.export import (
    build_export_metadata,
    export_state_dict_formats,
    load_exported_model,
)


def _config(
    *,
    enabled: bool = True,
    steps: int = 1,
    vocab_size: int = 24,
    d_model: int = 16,
) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=d_model * 2,
        max_seq_len=16,
        dropout=0.0,
        label_smoothing=0.1,
        gradient_checkpointing=False,
        experimental=ExperimentalConfig(
            candidate_refinement_enabled=enabled,
            candidate_refinement_steps=steps,
            candidate_refinement_temperature=0.8,
            candidate_refinement_loss_weight=0.25,
            candidate_refinement_vocab_chunk_size=5,
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


def test_full_vocabulary_expectation_and_draft_ce_match_dense_reference() -> None:
    torch.manual_seed(3)  # pyright: ignore[reportUnknownMemberType]
    model = SionForConditionalGeneration(_config()).eval()
    hidden = torch.randn(2, 3, model.config.d_model)
    labels = torch.tensor([[4, 5, -100], [6, 7, 8]])

    expectation, draft_loss_sum, token_nll = model._candidate_distribution_statistics(
        hidden,
        labels,
    )
    dense_logits = model._logits(hidden).float()
    dense_expectation = (
        torch.softmax(
            dense_logits / model.config.experimental.candidate_refinement_temperature,
            dim=-1,
        )
        @ model.token_embedding.weight.float()
    ).to(hidden.dtype)
    dense_loss_sum = F.cross_entropy(
        dense_logits.reshape(-1, dense_logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
        label_smoothing=model.config.label_smoothing,
    )
    dense_nll = F.cross_entropy(
        dense_logits.transpose(1, 2),
        labels,
        ignore_index=-100,
        reduction="none",
    )

    torch.testing.assert_close(expectation, dense_expectation, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(draft_loss_sum, dense_loss_sum, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(token_nll, dense_nll, rtol=1e-5, atol=1e-6)


def test_training_supervises_draft_and_refinement_without_detaching_gradients() -> None:
    torch.manual_seed(5)  # pyright: ignore[reportUnknownMemberType]
    model = SionForConditionalGeneration(_config()).train()
    assert model.candidate_refinement is not None
    with torch.no_grad():
        model.candidate_refinement.refinement_scale.fill_(0.2)

    output = model(**_batch())
    assert output.candidate_refinement_steps is not None
    assert output.candidate_refinement_steps.item() == 1
    assert output.candidate_refinement_loss is not None
    assert output.candidate_refinement_loss.item() > 0
    assert output.candidate_refinement_gain is not None
    assert torch.isfinite(output.candidate_refinement_gain)
    assert output.candidate_refinement_token_nll_gain is not None
    assert output.candidate_refinement_token_nll_gain.shape == _batch()["labels"].shape
    assert torch.isfinite(output.candidate_refinement_token_nll_gain).all()
    assert output.loss is not None
    output.loss.backward()

    assert model.candidate_refinement.refinement_scale.grad is not None
    assert model.candidate_refinement.gate.weight.grad is not None
    assert model.candidate_refinement.gate.weight.grad.abs().sum().item() > 0
    assert model.token_embedding.weight.grad is not None
    assert model.token_embedding.weight.grad.abs().sum().item() > 0


def test_reported_gain_matches_provisional_to_final_token_nll() -> None:
    torch.manual_seed(13)  # pyright: ignore[reportUnknownMemberType]
    model = SionForConditionalGeneration(_config(steps=3)).eval()
    assert model.candidate_refinement is not None
    with torch.no_grad():
        model.candidate_refinement.refinement_scale.fill_(0.2)

    provisional_states: list[torch.Tensor] = []

    def capture_provisional_states(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        if not provisional_states:
            provisional_states.append(args[0].detach().clone())

    handle = model.candidate_refinement.register_forward_pre_hook(capture_provisional_states)
    batch = _batch()
    output = model(**batch)
    handle.remove()

    assert len(provisional_states) == 1
    provisional_logits = model._logits(provisional_states[0]).float()
    provisional_nll = F.cross_entropy(
        provisional_logits.transpose(1, 2),
        batch["labels"],
        ignore_index=-100,
        reduction="none",
    )
    final_nll = F.cross_entropy(
        output.logits.float().transpose(1, 2),
        batch["labels"],
        ignore_index=-100,
        reduction="none",
    )
    expected_gain = (provisional_nll - final_nll) * batch["labels"].ne(-100)

    assert output.candidate_refinement_token_nll_gain is not None
    torch.testing.assert_close(output.candidate_refinement_token_nll_gain, expected_gain)


def test_zero_initialized_refiner_wakes_up_across_two_optimizer_steps() -> None:
    torch.manual_seed(6)  # pyright: ignore[reportUnknownMemberType]
    model = SionForConditionalGeneration(_config()).train()
    assert model.candidate_refinement is not None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    first = model(**_batch())
    assert first.loss is not None
    first.loss.backward()
    scale_gradient = model.candidate_refinement.refinement_scale.grad
    gate_gradient = model.candidate_refinement.gate.weight.grad
    assert scale_gradient is not None and scale_gradient.abs().sum().item() > 0
    assert gate_gradient is not None and gate_gradient.count_nonzero().item() == 0
    optimizer.step()
    assert model.candidate_refinement.refinement_scale.abs().sum().item() > 0

    optimizer.zero_grad(set_to_none=True)
    second = model(**_batch())
    assert second.loss is not None
    second.loss.backward()
    gate_gradient = model.candidate_refinement.gate.weight.grad
    proposal_gradient = model.candidate_refinement.proposal[0].weight.grad
    assert gate_gradient is not None and gate_gradient.abs().sum().item() > 0
    assert proposal_gradient is not None and proposal_gradient.abs().sum().item() > 0


def test_labels_never_condition_logits_and_future_tokens_remain_causal() -> None:
    torch.manual_seed(7)  # pyright: ignore[reportUnknownMemberType]
    model = SionForConditionalGeneration(_config()).eval()
    assert model.candidate_refinement is not None
    with torch.no_grad():
        model.candidate_refinement.refinement_scale.fill_(0.3)
    batch = _batch()
    alternate_labels = batch["labels"].roll(1, dims=0)

    original = model(**batch)
    relabeled = model(**{**batch, "labels": alternate_labels})
    torch.testing.assert_close(original.logits, relabeled.logits)

    changed_decoder = batch["decoder_input_ids"].clone()
    changed_decoder[:, -1] = 15
    changed = model(**{**batch, "decoder_input_ids": changed_decoder})
    torch.testing.assert_close(original.logits[:, :-1], changed.logits[:, :-1])


def test_final_prediction_can_leave_the_preliminary_top_candidates() -> None:
    config = _config(vocab_size=8, d_model=8)
    config.num_heads = 2
    config.num_kv_heads = 1
    model = SionForConditionalGeneration(config).eval()
    assert model.candidate_refinement is not None
    with torch.no_grad():
        model.token_embedding.weight.zero_()
        model.token_embedding.weight[4, 0] = 1.0
        model.token_embedding.weight[5, 1] = 1.0
    hidden = torch.zeros(1, 1, config.d_model)
    hidden[..., 0] = 4.0
    assert model._logits(hidden).argmax(-1).item() == 4

    def choose_unseen_token(
        _module: torch.nn.Module,
        states: torch.Tensor,
        _expectation: torch.Tensor,
    ) -> torch.Tensor:
        revised = torch.zeros_like(states)
        revised[..., 1] = 4.0
        return revised

    model.candidate_refinement.forward = types.MethodType(  # pyright: ignore[reportAttributeAccessIssue]
        choose_unseen_token,
        model.candidate_refinement,
    )
    refined, _, _, _ = model._apply_candidate_refinement(
        hidden,
        None,
        reasoning_level=None,
    )

    assert model._logits(refined).argmax(-1).item() == 5


def test_reasoning_levels_select_trained_endpoints_and_zero_is_exact_bypass() -> None:
    torch.manual_seed(11)  # pyright: ignore[reportUnknownMemberType]
    direct = SionForConditionalGeneration(_config(enabled=False, steps=3)).eval()
    torch.manual_seed(11)  # pyright: ignore[reportUnknownMemberType]
    refined = SionForConditionalGeneration(_config(steps=3)).eval()
    batch = _batch()

    direct_output = direct(**batch)
    level_zero = refined(**batch, reasoning_level=0)
    torch.testing.assert_close(level_zero.logits, direct_output.logits)
    assert direct_output.candidate_refinement_token_nll_gain is None
    assert level_zero.candidate_refinement_token_nll_gain is None
    assert level_zero.candidate_refinement_steps is not None
    assert level_zero.candidate_refinement_steps.item() == 0
    assert refined(**batch, reasoning_level=1).candidate_refinement_steps.item() == 1
    assert refined(**batch, reasoning_level=5).candidate_refinement_steps.item() == 2
    assert refined(**batch, reasoning_level=9).candidate_refinement_steps.item() == 3
    assert refined(**batch).candidate_refinement_steps.item() == 3


def test_decoder_cache_advances_once_while_refinement_iterates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SionForConditionalGeneration(_config(steps=3)).eval()
    assert model.candidate_refinement is not None
    batch = _batch()
    context = model.prepare_generation(batch["input_ids"][:1], batch["attention_mask"][:1])
    caches = model._fresh_caches(len(model.decoder_layers), context.cross_key_values)
    layer = model.decoder_layers[0]
    original_forward_step = layer.forward_step
    decoder_calls = 0
    refinement_calls = 0
    original_refinement = model.candidate_refinement.forward

    def counted_decoder(*args: object, **kwargs: object):
        nonlocal decoder_calls
        decoder_calls += 1
        return original_forward_step(*args, **kwargs)

    def counted_refinement(*args: object, **kwargs: object):
        nonlocal refinement_calls
        refinement_calls += 1
        return original_refinement(*args, **kwargs)

    monkeypatch.setattr(layer, "forward_step", counted_decoder)
    monkeypatch.setattr(model.candidate_refinement, "forward", counted_refinement)
    model._decoder_step(
        torch.tensor([[2]]),
        context.encoder_states,
        context.source_mask,
        caches,
        0,
        context.register_context,
        reasoning_level=None,
    )

    assert decoder_calls == 1
    assert refinement_calls == 3
    assert caches[0]["self"] is not None
    assert caches[0]["self"][0].shape[-2] == 1


def test_greedy_sampling_beam_and_hf_beam_all_use_refined_decoder_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SionForConditionalGeneration(_config()).eval()
    assert model.candidate_refinement is not None
    batch = _batch()
    calls = 0
    original = model.candidate_refinement.forward

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model.candidate_refinement, "forward", counted)
    model.generate(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        max_new_tokens=2,
        min_new_tokens=1,
        num_beams=1,
    )
    greedy_calls = calls
    assert greedy_calls > 0
    model.sample(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        max_new_tokens=2,
        min_new_tokens=1,
        num_samples=2,
        generator=torch.Generator().manual_seed(1),
    )
    sampling_calls = calls - greedy_calls
    assert sampling_calls > 0
    model.generate(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        max_new_tokens=2,
        min_new_tokens=1,
        num_beams=2,
    )
    assert calls - greedy_calls - sampling_calls > 0

    hf_config = SionConfig.from_model_config(_config())
    hf_model = HFSionForConditionalGeneration(hf_config).eval()
    assert hf_model.model.candidate_refinement is not None
    hf_calls = 0
    hf_original = hf_model.model.candidate_refinement.forward

    def counted_hf(*args: object, **kwargs: object):
        nonlocal hf_calls
        hf_calls += 1
        return hf_original(*args, **kwargs)

    monkeypatch.setattr(hf_model.model.candidate_refinement, "forward", counted_hf)
    generated = hf_model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=2,
        num_beams=2,
        num_return_sequences=2,
    )
    assert generated.shape[0] == 4
    assert hf_calls > 0


def test_hf_default_forward_and_generation_use_the_final_refinement_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_config = _config(steps=3)
    with pytest.raises(ValueError, match="default_reasoning_level=9"):
        SionConfig.from_model_config(
            native_config,
            default_reasoning_level=0,
        )

    hf_model = HFSionForConditionalGeneration(
        SionConfig.from_model_config(
            native_config,
            default_reasoning_level=9,
        )
    ).eval()
    assert hf_model.model.candidate_refinement is not None
    refinement_calls = 0
    original = hf_model.model.candidate_refinement.forward

    def counted(*args: object, **kwargs: object):
        nonlocal refinement_calls
        refinement_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(hf_model.model.candidate_refinement, "forward", counted)
    batch = _batch()
    hf_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=batch["decoder_input_ids"],
    )
    assert refinement_calls == native_config.experimental.candidate_refinement_steps

    refinement_calls = 0
    hf_model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=1,
    )
    assert refinement_calls == native_config.experimental.candidate_refinement_steps


def test_disabled_state_is_strictly_compatible_but_cannot_fake_trained_refinement() -> None:
    torch.manual_seed(17)  # pyright: ignore[reportUnknownMemberType]
    old_model = SionForConditionalGeneration(_config(enabled=False))
    torch.manual_seed(17)  # pyright: ignore[reportUnknownMemberType]
    disabled = SionForConditionalGeneration(_config(enabled=False))
    disabled.load_state_dict(old_model.state_dict(), strict=True)

    enabled = SionForConditionalGeneration(_config(enabled=True))
    with pytest.raises(RuntimeError, match="candidate_refinement"):
        enabled.load_state_dict(old_model.state_dict(), strict=True)


def test_native_export_strictly_round_trips_trained_refinement(tmp_path) -> None:
    model = SionForConditionalGeneration(_config()).eval()
    metadata = build_export_metadata(
        model.config,
        language_pair=("en", "de"),
        translation_directions=(("en", "de"),),
        pipeline_identity={
            "schema": "sion-translation-pipeline-v2",
            "branch": "translation-only",
        },
    )
    export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        model.config,
        0,
        formats=("fp32",),
        metadata=metadata,
    )

    restored, restored_config, _ = load_exported_model(tmp_path / "model.pt")
    assert restored_config.experimental.candidate_refinement_enabled is True
    assert isinstance(restored, SionForConditionalGeneration)
    assert restored.candidate_refinement is not None
    restored_output = restored(**_batch(), reasoning_level=9)
    original_output = model(**_batch(), reasoning_level=9)
    torch.testing.assert_close(restored_output.logits, original_output.logits)


def test_empty_targets_remain_finite_and_cli_uses_checkpoint_default() -> None:
    model = SionForConditionalGeneration(_config()).eval()
    batch = _batch()
    batch["labels"] = torch.full_like(batch["labels"], -100)
    output = model(**batch)

    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.candidate_refinement_loss is not None
    assert output.candidate_refinement_loss.item() == 0.0
    assert build_parser().parse_args([]).reasoning_level is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_refinement_enabled", 1, "must be a boolean"),
        ("candidate_refinement_steps", 1.5, "must be an integer"),
        ("candidate_refinement_steps", True, "must be an integer"),
        ("candidate_refinement_vocab_chunk_size", 4.0, "must be an integer"),
        ("candidate_refinement_temperature", float("nan"), "finite real number"),
        ("candidate_refinement_loss_weight", float("inf"), "finite real number"),
    ],
)
def test_candidate_refinement_config_rejects_runtime_type_traps(
    field: str,
    value: object,
    message: str,
) -> None:
    config = ExperimentalConfig()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=message):
        config.validate()
