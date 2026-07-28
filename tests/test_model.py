from __future__ import annotations

import pytest
import torch

from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.model import SionForConditionalGeneration


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=160,
        max_seq_len=32,
        dropout=0.0,
        gradient_checkpointing=False,
        experimental=ExperimentalConfig(
            bats_enabled=True,
            bats_dim=16,
            bats_loss_weight=0.1,
            bats_coverage_weight=0.01,
            bats_stride=1,
            bats_max_positions=32,
            core_enabled=True,
            tetm_enabled=True,
            morphoscript_enabled=True,
            morphoscript_interval=1,
        ),
    )


def make_batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[10, 11, 12, 3, 0], [10, 20, 21, 22, 3]])
    attention_mask = input_ids.ne(0)
    decoder_input_ids = torch.tensor([[2, 30, 31, 0], [2, 40, 41, 42]])
    labels = torch.tensor([[30, 31, 3, -100], [40, 41, 42, 3]])
    alignment = torch.zeros(2, 4, 5)
    alignment[0, 0, 0] = 1
    alignment[0, 1, 1] = 1
    alignment[1, 0, 0] = 1
    alignment[1, 1, 1] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
        "register_labels": torch.tensor([2, 1]),
        "memory_token_ids": torch.tensor([[[50]], [[0]]]),
        "memory_mask": torch.tensor([[True], [False]]),
        "memory_type_ids": torch.tensor([[1], [0]]),
        "memory_mode_ids": torch.tensor([[2], [0]]),
        "src_script_ids": torch.zeros_like(input_ids),
        "src_onset_ids": torch.zeros_like(input_ids),
        "src_vowel_ids": torch.zeros_like(input_ids),
        "src_coda_ids": torch.zeros_like(input_ids),
        "alignment_targets": alignment,
    }


def test_forward_backward_all_experimental_modules() -> None:
    model = SionForConditionalGeneration(tiny_config())
    output = model(**make_batch())
    assert output.logits.shape == (2, 4, 128)
    assert output.token_count.item() == 7
    assert torch.isfinite(output.loss)
    assert output.register_loss.item() > 0
    assert output.alignment_loss.item() >= 0
    output.loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_greedy_generation() -> None:
    model = SionForConditionalGeneration(tiny_config())
    batch = make_batch()
    generated = model.generate(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        max_new_tokens=3,
        src_script_ids=batch["src_script_ids"],
        src_onset_ids=batch["src_onset_ids"],
        src_vowel_ids=batch["src_vowel_ids"],
        src_coda_ids=batch["src_coda_ids"],
    )
    assert generated.shape[0] == 2
    assert 2 <= generated.shape[1] <= 4


def test_stochastic_sampling_returns_multiple_candidates() -> None:
    torch.manual_seed(7)
    model = SionForConditionalGeneration(tiny_config())
    batch = make_batch()
    sampled = model.sample(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        num_samples=3,
        max_new_tokens=5,
        temperature=0.9,
        top_k=16,
        forbidden_token_ids=(0, 2),
    )
    assert sampled.shape[:2] == (2, 3)
    assert sampled.shape[-1] <= 6
    assert sampled[:, :, 0].eq(2).all()


@pytest.mark.parametrize("max_new_tokens", (0, -1, 33))
def test_native_generation_rejects_invalid_lengths(max_new_tokens: int) -> None:
    model = SionForConditionalGeneration(tiny_config())
    batch = make_batch()
    common = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "bos_id": 2,
        "eos_id": 3,
        "max_new_tokens": max_new_tokens,
    }
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(**common)
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.sample(**common, num_samples=1)


def test_tied_embedding_output() -> None:
    model = SionForConditionalGeneration(tiny_config())
    assert model.lm_head is None
    assert model.parameter_count() > 0


def test_bf16_autocast_keeps_encoder_decoder_residuals_in_bf16() -> None:
    config = tiny_config()
    model = SionForConditionalGeneration(config)
    batch = make_batch()
    observed: dict[str, tuple[torch.dtype, torch.dtype]] = {}

    def capture(name: str):
        def hook(_module, inputs, output) -> None:
            observed[name] = (inputs[0].dtype, output.dtype)

        return hook

    encoder_hook = model.encoder_layers[0].register_forward_hook(capture("encoder"))
    decoder_hook = model.decoder_layers[0].register_forward_hook(capture("decoder"))
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(**batch)
    finally:
        encoder_hook.remove()
        decoder_hook.remove()

    assert observed == {
        "encoder": (torch.bfloat16, torch.bfloat16),
        "decoder": (torch.bfloat16, torch.bfloat16),
    }
    assert output.logits.dtype == torch.bfloat16


def test_meta_materialization_rebuilds_rope_buffers() -> None:
    config = tiny_config()
    config.experimental = ExperimentalConfig()
    reference = SionForConditionalGeneration(config)

    with torch.device("meta"):
        materialized = SionForConditionalGeneration(config)
    assert materialized.encoder_rope.cos.is_meta
    assert materialized.decoder_rope.sin.is_meta

    materialized.to_empty(device="cpu")
    materialized.init_weights()

    assert not materialized.encoder_rope.cos.is_meta
    assert not materialized.decoder_rope.sin.is_meta
    torch.testing.assert_close(materialized.encoder_rope.cos, reference.encoder_rope.cos)
    torch.testing.assert_close(materialized.encoder_rope.sin, reference.encoder_rope.sin)
    torch.testing.assert_close(materialized.decoder_rope.cos, reference.decoder_rope.cos)
    torch.testing.assert_close(materialized.decoder_rope.sin, reference.decoder_rope.sin)

    batch = make_batch()
    output = materialized(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=batch["decoder_input_ids"],
    )
    assert torch.isfinite(output.logits).all()


def test_kv_cache_greedy_matches_full_redecode() -> None:
    """KV cache 디코딩이 '매번 prefix 전체를 다시 계산'하는 방식과
    토큰 단위로 완전히 같은 결과를 내야 한다."""
    torch.manual_seed(7)
    model = SionForConditionalGeneration(tiny_config())
    model.eval()
    batch = make_batch()
    features = {
        key: batch[key]
        for key in ("src_script_ids", "src_onset_ids", "src_vowel_ids", "src_coda_ids")
    }
    cached = model.generate(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        max_new_tokens=6,
        **features,
    )
    with torch.no_grad():
        encoder_states = model.encode(batch["input_ids"], batch["attention_mask"], **features)
        register_context = None
        if model.register_state is not None:
            _, register_context, _ = model.register_state(
                encoder_states, batch["attention_mask"], register_labels=None
            )
        generated = torch.full((2, 1), 2, dtype=torch.long)
        finished = torch.zeros(2, dtype=torch.bool)
        for _ in range(6):
            hidden = model.decode(
                generated,
                encoder_states,
                batch["attention_mask"],
                register_context=register_context,
            )
            next_token = model._logits(hidden[:, -1:]).argmax(-1)
            next_token = torch.where(finished[:, None], 3, next_token)
            generated = torch.cat((generated, next_token), dim=1)
            finished |= next_token.squeeze(1).eq(3)
            if finished.all():
                break
    assert cached.tolist() == generated.tolist()


def test_beam_search_generation() -> None:
    torch.manual_seed(7)
    model = SionForConditionalGeneration(tiny_config())
    batch = make_batch()
    features = {
        key: batch[key]
        for key in ("src_script_ids", "src_onset_ids", "src_vowel_ids", "src_coda_ids")
    }
    output = model.generate(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        max_new_tokens=6,
        num_beams=3,
        length_penalty=1.0,
        **features,
    )
    assert output.shape[0] == 2
    assert output[:, 0].eq(2).all()  # BOS 로 시작
    assert output[:, -1].eq(3).all() or output.shape[1] == 7  # EOS 로 끝나거나 길이 제한


def test_morph_gates_only_when_morphoscript_enabled() -> None:
    from sion_translate.config import ExperimentalConfig, ModelConfig

    disabled = ModelConfig(
        vocab_size=64,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        max_seq_len=16,
        gradient_checkpointing=False,
        experimental=ExperimentalConfig(morphoscript_enabled=False),
    )
    model = SionForConditionalGeneration(disabled)
    assert model.morph_gates is None
    assert all("morph_gates" not in name for name, _ in model.named_parameters())
    enabled_model = SionForConditionalGeneration(tiny_config())
    assert enabled_model.morph_gates is not None
