from __future__ import annotations

import types

import pytest
import torch

import sion_translate.model.transformer as transformer_module
from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.model import SionForConditionalGeneration
from sion_translate.model.layers import SwiGLU


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


def test_situglu_bounds_activations_and_keeps_swiglu_state_compatible() -> None:
    torch.manual_seed(11)
    swiglu = SwiGLU(8, 16, 0.0)
    situglu = SwiGLU(8, 16, 0.0, gate_beta=4.0, up_beta=25.0)
    situglu.load_state_dict(swiglu.state_dict(), strict=True)
    assert swiglu.state_dict().keys() == situglu.state_dict().keys()

    small_input = torch.randn(2, 3, 8) * 1e-4
    torch.testing.assert_close(
        situglu.gated_activations(small_input),
        swiglu.gated_activations(small_input),
        rtol=1e-5,
        atol=1e-10,
    )
    huge_input = torch.full((2, 3, 8), 1e6)
    bounded = situglu.gated_activations(huge_input)
    assert torch.isfinite(bounded).all()
    assert bounded.abs().max() <= 100.0


def test_model_wires_situglu_into_encoder_and_decoder_without_extra_parameters() -> None:
    baseline_config = tiny_config()
    situ_config = tiny_config()
    situ_config.experimental.situglu_enabled = True
    baseline = SionForConditionalGeneration(baseline_config)
    situ = SionForConditionalGeneration(situ_config)

    assert baseline.parameter_count() == situ.parameter_count()
    assert baseline.encoder_layers[0].ffn.gate_beta is None
    assert situ.encoder_layers[0].ffn.gate_beta == 4.0
    assert situ.decoder_layers[0].ffn.up_beta == 25.0


def test_auxiliary_loss_masks_are_compile_safe_and_keep_empty_batches_finite() -> None:
    config = ModelConfig(
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
        experimental=ExperimentalConfig(core_enabled=True),
    )
    model = SionForConditionalGeneration(config)
    batch = {
        "input_ids": torch.tensor([[4, 5, 3], [4, 6, 3]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.bool),
        "decoder_input_ids": torch.tensor([[2, 8, 0], [2, 9, 10]]),
        "labels": torch.tensor([[8, 3, -100], [9, 10, 3]]),
        "register_labels": torch.tensor([2, 0]),
    }

    eager = model(**batch)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    compiled_output = compiled(**batch)
    torch.testing.assert_close(compiled_output.loss, eager.loss)
    torch.testing.assert_close(compiled_output.register_loss, eager.register_loss)

    empty_batch = {
        **batch,
        "labels": torch.full_like(batch["labels"], -100),
        "register_labels": torch.zeros_like(batch["register_labels"]),
    }
    empty_output = model(**empty_batch)
    assert empty_output.register_loss.item() == 0.0
    assert empty_output.auxiliary_loss.item() == 0.0
    assert torch.isfinite(empty_output.loss)
    empty_output.loss.backward()
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


def test_generation_context_reuses_encoder_and_cross_attention_state() -> None:
    torch.manual_seed(19)
    model = SionForConditionalGeneration(tiny_config())
    model.eval()
    batch = make_batch()
    features = {
        key: batch[key]
        for key in ("src_script_ids", "src_onset_ids", "src_vowel_ids", "src_coda_ids")
    }
    baseline_beam = model.generate(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        max_new_tokens=4,
        num_beams=2,
        **features,
    )
    baseline_samples = model.sample(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        num_samples=2,
        max_new_tokens=4,
        temperature=0.8,
        top_k=16,
        generator=torch.Generator().manual_seed(41),
        **features,
    )

    encode_calls = 0
    real_encode = model.encode

    def count_encode(*args, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        return real_encode(*args, **kwargs)

    model.encode = count_encode
    context = model.prepare_generation(
        batch["input_ids"],
        batch["attention_mask"],
        **features,
    )
    shared_beam = model.generate(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        max_new_tokens=4,
        num_beams=2,
        generation_context=context,
    )
    shared_samples = model.sample(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        num_samples=2,
        max_new_tokens=4,
        temperature=0.8,
        top_k=16,
        generator=torch.Generator().manual_seed(41),
        generation_context=context,
    )

    assert encode_calls == 1
    torch.testing.assert_close(shared_beam, baseline_beam)
    torch.testing.assert_close(shared_samples, baseline_samples)


def test_decode_constraints_block_control_tokens_and_repeated_ngrams() -> None:
    logits = torch.zeros(2, 16)
    logits[:, 3] = 9.0
    logits[:, 5] = 10.0
    sequences = torch.tensor(
        [
            [2, 5, 6, 7, 5, 6, 7],
            [2, 8, 9, 10, 11, 12, 13],
        ]
    )

    constrained = SionForConditionalGeneration._apply_decode_constraints(
        logits,
        sequences,
        eos_id=3,
        position=0,
        min_new_tokens=1,
        forbidden_token_ids=(0, 2),
        no_repeat_ngram_size=4,
    )

    assert constrained[:, 0].isneginf().all()
    assert constrained[:, 2].isneginf().all()
    assert constrained[:, 3].isneginf().all()
    # 첫 행의 마지막 5,6,7 뒤에는 과거에 5가 왔으므로 그 4-gram을 차단합니다.
    assert constrained[0, 5].isneginf()
    assert torch.isfinite(constrained[1, 5])


def test_sampling_waits_until_every_distributed_rank_is_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SionForConditionalGeneration(tiny_config())
    batch = make_batch()

    def eos_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (*hidden.shape[:-1], self.config.vocab_size),
            -1_000.0,
            device=hidden.device,
        )
        logits[..., 3] = 0.0
        return logits

    synchronized_results = iter((False, True))
    model._logits = types.MethodType(eos_logits, model)
    model._synchronize_generation_across_ranks = True
    monkeypatch.setattr(
        transformer_module,
        "_all_ranks_finished",
        lambda _local, _device: next(synchronized_results),
    )
    sampled = model.sample(
        batch["input_ids"],
        batch["attention_mask"],
        bos_id=2,
        eos_id=3,
        num_samples=1,
        max_new_tokens=3,
    )
    assert sampled.shape == (2, 1, 3)
    assert sampled[:, :, 1:].eq(3).all()


def test_distributed_finished_consensus_uses_all_rank_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transformer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(transformer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(transformer_module.dist, "get_world_size", lambda: 2)
    reductions: list[object] = []

    def one_remote_rank_is_running(flag: torch.Tensor, *, op: object) -> None:
        reductions.append(op)
        flag.zero_()

    monkeypatch.setattr(transformer_module.dist, "all_reduce", one_remote_rank_is_running)
    assert not transformer_module._all_ranks_finished(True, torch.device("cpu"))
    assert reductions == [transformer_module.dist.ReduceOp.MIN]


def test_distributed_generation_uses_largest_rank_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transformer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(transformer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(transformer_module.dist, "get_world_size", lambda: 2)
    reductions: list[object] = []

    def remote_rank_has_longer_batch(limit: torch.Tensor, *, op: object) -> None:
        reductions.append(op)
        limit.fill_(17)

    monkeypatch.setattr(transformer_module.dist, "all_reduce", remote_rank_has_longer_batch)
    assert transformer_module._all_ranks_max_new_tokens(9, torch.device("cpu")) == 17
    assert reductions == [transformer_module.dist.ReduceOp.MAX]


def test_beam_finalization_compares_live_and_completed_hypotheses() -> None:
    config = ModelConfig(
        vocab_size=8,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=32,
        max_seq_len=8,
        dropout=0.0,
        gradient_checkpointing=False,
    )
    model = SionForConditionalGeneration(config)

    def fixed_step(
        self,
        tokens,
        encoder_states,
        source_mask,
        caches,
        position,
        register_context,
        **kwargs,
    ):
        del encoder_states, source_mask, position, register_context, kwargs
        total = tokens.shape[0]
        for cache in caches:
            zeros = torch.zeros(total, 1, 1, 1)
            cache["self"] = (zeros, zeros)
            cache["cross"] = (zeros, zeros)
        return torch.zeros(total, 1, config.d_model)

    def fixed_logits(self, hidden):
        del self
        result = torch.full((*hidden.shape[:-1], config.vocab_size), -20.0)
        result[..., 4] = 5.0
        result[..., 3] = 0.0
        return result

    model._decoder_step = types.MethodType(fixed_step, model)
    model._logits = types.MethodType(fixed_logits, model)
    output = model._beam_decode(
        torch.zeros(1, 2, config.d_model),
        torch.ones(1, 2, dtype=torch.bool),
        None,
        bos_id=2,
        eos_id=3,
        max_new_tokens=2,
        num_beams=2,
        length_penalty=1.0,
    )
    assert output.tolist() == [[2, 4, 4, 3]]


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
