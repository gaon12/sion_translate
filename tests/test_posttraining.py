from __future__ import annotations

from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sion_translate.config import PostTrainingConfig
from sion_translate.training.objectives import CompositeTranslationReward, MinimumRiskObjective


class TextTokenizer:
    pad_id = 0
    bos_id = 2
    eos_id = 3
    mask_id = 6
    language_tags = {"ja": 4, "ko": 5}
    denoise_tags = {"ja": 7, "ko": 8}
    draft_id = 9
    slot_ids = [30]
    pieces = {
        10: "가격 ",
        11: "100",
        20: "価格 ",
        21: "100円",
        22: "200円",
        23: "あ" * 20,
        30: "<TERM_0>",
    }

    @classmethod
    def decode(cls, ids) -> str:
        return "".join(cls.pieces.get(int(token_id), "") for token_id in ids)


def test_composite_reward_penalizes_number_corruption_and_slot_omission() -> None:
    reward = CompositeTranslationReward(TextTokenizer(), PostTrainingConfig())
    source = torch.tensor([[4, 10, 11, 30, 3]])
    reference = torch.tensor([[20, 21, 30, 3]])
    candidates = torch.tensor([[[2, 20, 21, 30, 3], [2, 20, 22, 0, 3]]])

    result = reward(candidates, source, reference)

    assert result.reward[0, 0] > result.reward[0, 1]
    assert result.components["number"][0].tolist() == [1.0, 0.0]
    assert result.components["slot"][0].tolist() == [1.0, 0.0]


def test_a_value_changing_candidate_cannot_win_on_chrf() -> None:
    """가중 성분만으로는 숫자를 틀린 후보가 이긴다. 그래서 하드 페널티를 쓴다.

    `reward_number_weight` 는 0.10 이라 값 변조 하나의 손실이 chrF 우위로
    덮입니다. 배포 홀드아웃 10문장 중 8문장의 숫자가 바뀐 것이 그 결과입니다.
    """

    weighted_only = PostTrainingConfig(reward_number_corruption_penalty=0.0)
    gated = PostTrainingConfig()  # 기본값 0.35

    source = torch.tensor([[4, 10, 11, 3]])  # "가격 100"
    reference = torch.tensor([[20, 21, 3]])  # "価格 100円"
    # 값을 지킨 후보와 100 을 200 으로 바꾼 후보.
    candidates = torch.tensor([[[2, 20, 21, 3], [2, 20, 22, 3]]])

    weighted = CompositeTranslationReward(TextTokenizer(), weighted_only)(
        candidates, source, reference
    ).reward[0]
    gated_score = CompositeTranslationReward(TextTokenizer(), gated)(
        candidates, source, reference
    ).reward[0]

    # 값을 지킨 후보는 두 설정에서 똑같이 온전하다.
    assert weighted[0] == pytest.approx(gated_score[0])
    # 값을 바꾼 후보만 더 낮아지고, 격차가 벌어진다.
    assert gated_score[1] < weighted[1]
    assert (gated_score[0] - gated_score[1]) > (weighted[0] - weighted[1])


def test_a_number_the_reference_licenses_is_not_a_corruption() -> None:
    """한국어는 수를 한글로 자주 적는다. 정답이 뒷받침하는 값은 발명이 아니다."""

    gated = PostTrainingConfig()
    source = torch.tensor([[4, 10, 3]])  # "가격 " — 숫자가 없다
    reference = torch.tensor([[20, 21, 3]])  # "価格 100円"
    candidates = torch.tensor([[[2, 20, 21, 3]]])  # 정답과 같은 값

    score = CompositeTranslationReward(TextTokenizer(), gated)(
        candidates, source, reference
    ).reward.item()
    unpenalized = CompositeTranslationReward(
        TextTokenizer(), PostTrainingConfig(reward_number_corruption_penalty=0.0)
    )(candidates, source, reference).reward.item()

    assert score == pytest.approx(unpenalized)


def test_composite_reward_applies_repetition_and_source_copy_penalties() -> None:
    # 이 테스트는 반복·복사 페널티만 격리합니다. 값 변조 페널티는 양쪽에서 끕니다 —
    # 켜 두면 반복 후보가 원문의 100 을 빠뜨린 것으로 함께 감점되어, 두 설정
    # 모두에서 0 으로 눌리고 비교가 성립하지 않습니다.
    base = PostTrainingConfig(
        reward_repetition_penalty=0.0,
        reward_copy_penalty=0.0,
        reward_number_corruption_penalty=0.0,
    )
    penalized = PostTrainingConfig(
        reward_repetition_penalty=0.4,
        reward_copy_penalty=0.4,
        reward_number_corruption_penalty=0.0,
    )
    source = torch.tensor([[4, 10, 11, 3]])
    reference = torch.tensor([[20, 21, 3]])
    candidates = torch.tensor([[[2, 23, 3, 3], [2, 10, 11, 3]]])

    base_score = CompositeTranslationReward(TextTokenizer(), base)(
        candidates, source, reference
    ).reward[0]
    penalized_score = CompositeTranslationReward(TextTokenizer(), penalized)(
        candidates, source, reference
    ).reward[0]

    assert penalized_score[0] < base_score[0]
    assert penalized_score[1] < base_score[1]


def test_composite_reward_preserves_reference_backed_expressive_repetition_and_copy() -> None:
    base = PostTrainingConfig(
        reward_repetition_penalty=0.0,
        reward_copy_penalty=0.0,
    )
    penalized = PostTrainingConfig(
        reward_repetition_penalty=0.4,
        reward_copy_penalty=0.4,
    )
    # Some laughter/cry forms and onomatopoeia are intentionally identical on
    # both sides. A reference-backed match is not generation collapse or lazy copy.
    source = torch.tensor([[4, 23, 3]])
    reference = torch.tensor([[23, 3]])
    candidates = torch.tensor([[[2, 23, 3]]])

    base_score = CompositeTranslationReward(TextTokenizer(), base)(
        candidates, source, reference
    ).reward.item()
    penalized_score = CompositeTranslationReward(TextTokenizer(), penalized)(
        candidates, source, reference
    ).reward.item()

    assert base_score == 1.0
    assert penalized_score == base_score


def test_roundtrip_reward_requires_recovering_the_source() -> None:
    config = PostTrainingConfig(
        roundtrip_enabled=True,
        roundtrip_reward_weight=0.4,
        roundtrip_failure_penalty=0.3,
        roundtrip_min_score=0.6,
    )
    reward = CompositeTranslationReward(TextTokenizer(), config)
    source = torch.tensor([[4, 10, 11, 3]])
    reference = torch.tensor([[20, 21, 3]])
    # 정방향 후보는 같고, 역번역 결과만 하나는 원문을 복원하고 하나는 실패합니다.
    candidates = torch.tensor([[[2, 20, 21, 3], [2, 20, 21, 3]]])
    roundtrips = torch.tensor([[[2, 10, 11, 3], [2, 20, 22, 3]]])

    result = reward(
        candidates,
        source,
        reference,
        roundtrip_candidates=roundtrips,
    )

    assert result.components["roundtrip"][0, 0] > result.components["roundtrip"][0, 1]
    assert result.reward[0, 0] > result.reward[0, 1] + 0.2


def test_backtranslation_skips_rows_without_a_trained_reverse_edge() -> None:
    class RecordingGenerator:
        config = SimpleNamespace(max_seq_len=16, vocab_size=64)

        def __init__(self) -> None:
            self.inputs: torch.Tensor | None = None

        def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
            del attention_mask, kwargs
            self.inputs = input_ids
            return torch.tensor(
                [[2, 10, 11, 3]] * input_ids.shape[0],
                device=input_ids.device,
            )

    objective = MinimumRiskObjective(
        TextTokenizer(),
        PostTrainingConfig(roundtrip_enabled=True),
    )
    model = RecordingGenerator()
    batch = {
        "attention_mask": torch.ones(3, 4, dtype=torch.bool),
        "source_language_tag_ids": torch.tensor([5, 5, -1]),
        # The middle row is a source-only/unidirectional example. The last is
        # denoising and would be excluded by its -1 tag independently.
        "reverse_direction_trained": torch.tensor([True, False, True]),
    }
    candidates = torch.tensor(
        [
            [[2, 20, 21, 3], [2, 20, 22, 3]],
            [[2, 20, 21, 3], [2, 20, 22, 3]],
            [[2, 20, 21, 3], [2, 20, 22, 3]],
        ]
    )

    roundtrips, mask = objective._backtranslate_candidates(model, batch, candidates)

    assert model.inputs is not None
    assert model.inputs.shape[0] == 2
    assert model.inputs[:, 0].tolist() == [5, 5]
    assert mask is not None
    assert mask.tolist() == [
        [True, True],
        [False, False],
        [False, False],
    ]
    assert roundtrips is not None
    assert roundtrips[0, 0].tolist() == [2, 10, 11, 3]


def test_backtranslation_is_conservative_without_reverse_graph_metadata() -> None:
    objective = MinimumRiskObjective(
        TextTokenizer(),
        PostTrainingConfig(roundtrip_enabled=True),
    )
    batch = {
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "source_language_tag_ids": torch.tensor([5]),
    }
    candidates = torch.tensor([[[2, 20, 21, 3]]])

    roundtrips, mask = objective._backtranslate_candidates(
        SimpleNamespace(config=SimpleNamespace(max_seq_len=16)),
        batch,
        candidates,
    )

    assert roundtrips is None
    assert mask is not None
    assert not mask.any()


def test_backtranslation_keeps_fsdp_rank_in_collectives_without_local_reverse_rows() -> None:
    class RecordingGenerator:
        config = SimpleNamespace(max_seq_len=16, vocab_size=64)
        _synchronize_generation_across_ranks = True

        def __init__(self) -> None:
            self.inputs: torch.Tensor | None = None

        def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
            del attention_mask, kwargs
            self.inputs = input_ids
            return torch.tensor([[2, 10, 11, 3]], device=input_ids.device)

    objective = MinimumRiskObjective(
        TextTokenizer(),
        PostTrainingConfig(roundtrip_enabled=True),
    )
    model = RecordingGenerator()
    batch = {
        "attention_mask": torch.ones(2, 4, dtype=torch.bool),
        "source_language_tag_ids": torch.tensor([5, -1]),
        "reverse_direction_trained": torch.tensor([False, True]),
    }
    candidates = torch.tensor(
        [
            [[2, 20, 21, 3], [2, 20, 22, 3]],
            [[2, 20, 21, 3], [2, 20, 22, 3]],
        ]
    )

    roundtrips, mask = objective._backtranslate_candidates(model, batch, candidates)

    assert model.inputs is not None
    assert model.inputs.shape[0] == 1
    assert model.inputs[0, 0].item() in TextTokenizer.language_tags.values()
    assert roundtrips is not None
    assert mask is not None
    assert not mask.any()


def test_multi_pair_preference_loss_uses_reward_ordering() -> None:
    objective = MinimumRiskObjective(TextTokenizer(), PostTrainingConfig())
    rewards = torch.tensor([[0.9, 0.6, 0.2]])
    aligned_scores = torch.tensor([[0.0, -1.0, -2.0]], requires_grad=True)
    reversed_scores = torch.tensor([[-2.0, -1.0, 0.0]], requires_grad=True)

    aligned_loss, aligned_weight = objective._pairwise_preference_loss(aligned_scores, rewards)
    reversed_loss, reversed_weight = objective._pairwise_preference_loss(reversed_scores, rewards)

    assert aligned_weight.item() == reversed_weight.item()
    assert aligned_loss < reversed_loss
    reversed_loss.backward()
    assert reversed_scores.grad is not None


def test_reference_preference_preserves_sequence_gradient_when_rewards_tie() -> None:
    objective = MinimumRiskObjective(TextTokenizer(), PostTrainingConfig())
    generated_scores = torch.tensor([[0.0, 0.0]], requires_grad=True)
    reference_scores = torch.tensor([0.0], requires_grad=True)
    candidate_labels = torch.tensor([[[20, 22, 3], [23, 21, 3]]])
    reference_labels = torch.tensor([[20, 21, 3]])
    tied_rewards = torch.tensor([[0.5, 0.5]])

    candidate_loss, pair_weight = objective._pairwise_preference_loss(
        generated_scores,
        tied_rewards,
    )
    anchor_loss, margin, violation_rate, comparisons = objective._reference_preference_loss(
        generated_scores,
        reference_scores,
        candidate_labels,
        reference_labels,
    )

    assert candidate_loss.item() == 0.0
    assert pair_weight.item() == 0.0
    assert anchor_loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert margin.item() == 0.0
    assert violation_rate.item() == 1.0
    assert comparisons.item() == 2.0
    anchor_loss.backward()
    assert reference_scores.grad is not None
    assert reference_scores.grad.item() < 0.0
    assert generated_scores.grad is not None
    assert torch.all(generated_scores.grad > 0.0)


def test_reference_preference_excludes_exact_gold_duplicates() -> None:
    objective = MinimumRiskObjective(TextTokenizer(), PostTrainingConfig())
    generated_scores = torch.tensor([[0.0, -0.5]], requires_grad=True)
    reference_scores = torch.tensor([0.0], requires_grad=True)
    reference_labels = torch.tensor([[20, 21, 3]])
    candidate_labels = torch.tensor([[[20, 21, 3], [20, 22, 3]]])

    loss, margin, violation_rate, comparisons = objective._reference_preference_loss(
        generated_scores,
        reference_scores,
        candidate_labels,
        reference_labels,
    )

    assert loss.item() == pytest.approx(F.softplus(torch.tensor(-0.5)).item())
    assert margin.item() == pytest.approx(0.5)
    assert violation_rate.item() == 0.0
    assert comparisons.item() == 1.0
    loss.backward()
    assert generated_scores.grad is not None
    assert generated_scores.grad[0, 0].item() == 0.0
    assert generated_scores.grad[0, 1].item() > 0.0


def test_reference_preference_rejects_non_finite_sequence_scores() -> None:
    objective = MinimumRiskObjective(TextTokenizer(), PostTrainingConfig())
    candidate_labels = torch.tensor([[[20, 22, 3]]])
    reference_labels = torch.tensor([[20, 21, 3]])

    with pytest.raises(FloatingPointError, match="generated sequence scores"):
        objective._reference_preference_loss(
            torch.tensor([[float("nan")]]),
            torch.tensor([0.0]),
            candidate_labels,
            reference_labels,
        )


class TinyCandidateScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 12)
        self.projection = nn.Linear(12, 64, bias=False)
        self.config = SimpleNamespace(label_smoothing=0.1)
        self.forward_batch_sizes: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        del labels
        self.forward_batch_sizes.append(input_ids.shape[0])
        source_mask = attention_mask.unsqueeze(-1).to(self.embedding.weight.dtype)
        source = (self.embedding(input_ids) * source_mask).sum(1)
        source = source / source_mask.sum(1).clamp_min(1)
        hidden = torch.tanh(self.embedding(decoder_input_ids) + source[:, None])
        return SimpleNamespace(logits=self.projection(hidden))


class ReferenceAuxiliaryScorer(TinyCandidateScorer):
    def __init__(self) -> None:
        super().__init__()
        self.aux_scale = nn.Parameter(torch.tensor(0.5))
        self.reference_register_labels: torch.Tensor | None = None
        self.reference_alignment_targets: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        register_labels: torch.Tensor | None = None,
        alignment_targets: torch.Tensor | None = None,
    ):
        self.reference_register_labels = register_labels
        self.reference_alignment_targets = alignment_targets
        output = super().forward(input_ids, attention_mask, decoder_input_ids, labels)
        if labels is not None:
            output.lm_loss_sum = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            output.auxiliary_loss = self.aux_scale.square()
            output.evidence_request_rate = self.aux_scale.sigmoid()
        return output


def test_mrt_reference_pass_keeps_native_auxiliary_supervision() -> None:
    objective = MinimumRiskObjective(TextTokenizer(), PostTrainingConfig())
    model = ReferenceAuxiliaryScorer()
    batch = posttraining_batch()
    batch["register_labels"] = torch.tensor([2])
    batch["alignment_targets"] = torch.tensor([[[0, 1], [1, 2]]])

    ce, tokens, auxiliary, reference_scores, diagnostics = objective._reference_cross_entropy(
        model,
        batch,
        batch["decoder_input_ids"],
        batch["labels"],
        label_smoothing=0.1,
    )
    (ce + auxiliary).backward()

    assert tokens.item() == 3
    assert auxiliary.item() == pytest.approx(0.25)
    assert reference_scores.shape == (1,)
    assert torch.isfinite(reference_scores).all()
    assert diagnostics["evidence_request_rate"].item() == pytest.approx(
        torch.sigmoid(torch.tensor(0.5)).item()
    )
    assert model.aux_scale.grad is not None
    torch.testing.assert_close(model.reference_register_labels, batch["register_labels"])
    torch.testing.assert_close(model.reference_alignment_targets, batch["alignment_targets"])


class ConcurrentCandidateScorer(TinyCandidateScorer):
    def __init__(
        self,
        reward_started: threading.Event,
        candidate_scoring_started: threading.Event,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(label_smoothing=0.1, max_seq_len=16, vocab_size=64)
        self.reward_started = reward_started
        self.candidate_scoring_started = candidate_scoring_started

    def sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        num_samples: int,
        **kwargs,
    ) -> torch.Tensor:
        del attention_mask, kwargs
        candidates = torch.tensor(
            [[2, 20, 21, 3], [2, 20, 22, 3]],
            device=input_ids.device,
        )
        return candidates[:num_samples].unsqueeze(0).expand(input_ids.shape[0], -1, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        self.candidate_scoring_started.set()
        if not self.reward_started.wait(timeout=2):
            raise AssertionError("reward worker did not start before candidate scoring")
        return super().forward(
            input_ids,
            attention_mask,
            decoder_input_ids,
            labels,
        )


class AutocastCacheRecordingScorer(TinyCandidateScorer):
    """Record cache state across no-grad generation and checkpoint replay."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(label_smoothing=0.1, max_seq_len=16, vocab_size=64)
        self.sample_cache_states: list[bool] = []
        self.forward_cache_states: list[tuple[bool, bool]] = []

    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        num_samples: int,
        **kwargs,
    ) -> torch.Tensor:
        del attention_mask, kwargs
        self.sample_cache_states.append(torch.is_autocast_cache_enabled())
        candidates = torch.tensor(
            [[2, 20, 21, 3], [2, 20, 22, 3]],
            device=input_ids.device,
        )
        return candidates[:num_samples].unsqueeze(0).expand(input_ids.shape[0], -1, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        self.forward_cache_states.append((labels is None, torch.is_autocast_cache_enabled()))
        return super().forward(input_ids, attention_mask, decoder_input_ids, labels)


class MemoryAwareCandidateScorer(TinyCandidateScorer):
    """Small objective double that records source-side generation context."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(label_smoothing=0.1, max_seq_len=16, vocab_size=64)
        self.sample_memory: dict[str, torch.Tensor] = {}
        self.generate_memory: dict[str, torch.Tensor] = {}
        self.sample_options: dict[str, object] = {}
        self.generate_options: dict[str, object] = {}
        self.scoring_memory: list[torch.Tensor] = []

    @staticmethod
    def _memory_from(kwargs: dict) -> dict[str, torch.Tensor]:
        names = (
            "memory_token_ids",
            "memory_mask",
            "memory_type_ids",
            "memory_mode_ids",
        )
        return {name: kwargs[name].detach().clone() for name in names if name in kwargs}

    def sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        num_samples: int,
        **kwargs,
    ) -> torch.Tensor:
        del attention_mask
        self.sample_memory = self._memory_from(kwargs)
        self.sample_options = dict(kwargs)
        candidates = torch.tensor(
            [[2, 20, 21, 3], [2, 20, 22, 3]],
            device=input_ids.device,
        )
        return candidates[:num_samples].unsqueeze(0).expand(input_ids.shape[0], -1, -1)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del attention_mask
        self.generate_memory = self._memory_from(kwargs)
        self.generate_options = dict(kwargs)
        return torch.tensor(
            [[2, 20, 21, 3]],
            device=input_ids.device,
        ).expand(input_ids.shape[0], -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        memory_token_ids = kwargs.get("memory_token_ids")
        if memory_token_ids is not None:
            self.scoring_memory.append(memory_token_ids.detach().clone())
        return super().forward(input_ids, attention_mask, decoder_input_ids, labels)


def posttraining_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[4, 10, 11, 3]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "decoder_input_ids": torch.tensor([[2, 20, 21]]),
        "labels": torch.tensor([[20, 21, 3]]),
    }


def test_tetm_memory_is_shared_by_sampling_scoring_and_validation_generation() -> None:
    config = PostTrainingConfig(
        samples_per_source=2,
        candidate_micro_batch=1,
        candidate_gradient_checkpointing=False,
        max_new_tokens=4,
        validation_num_beams=1,
    )
    objective = MinimumRiskObjective(TextTokenizer(), config)
    model = MemoryAwareCandidateScorer()
    batch = posttraining_batch()
    batch.update(
        memory_token_ids=torch.tensor([[[30]]]),
        memory_mask=torch.tensor([[True]]),
        memory_type_ids=torch.tensor([[8]]),
        memory_mode_ids=torch.tensor([[4]]),
    )

    objective(model, batch)
    expected_names = {
        "memory_token_ids",
        "memory_mask",
        "memory_type_ids",
        "memory_mode_ids",
    }
    assert set(model.sample_memory) == expected_names
    for name in expected_names:
        torch.testing.assert_close(model.sample_memory[name], batch[name])
    assert model.scoring_memory
    assert all(memory.shape[-2:] == (1, 1) for memory in model.scoring_memory)
    assert model.sample_options["min_new_tokens"] == 1
    assert model.sample_options["no_repeat_ngram_size"] == 4
    assert model.sample_options["forbidden_token_ids"] == (0, 2, 4, 5, 6, 7, 8, 9)
    torch.testing.assert_close(
        model.sample_options["max_new_tokens_per_row"],  # type: ignore[arg-type]
        torch.tensor([4]),
    )

    objective.validation_metrics(model, batch)
    assert set(model.generate_memory) == expected_names
    for name in expected_names:
        torch.testing.assert_close(model.generate_memory[name], batch[name])
    assert model.generate_options["length_penalty"] == 1.0
    assert model.generate_options["min_new_tokens"] == 1
    assert model.generate_options["no_repeat_ngram_size"] == 4
    assert model.generate_options["forbidden_token_ids"] == (0, 2, 4, 5, 6, 7, 8, 9)
    torch.testing.assert_close(
        model.generate_options["max_new_tokens_per_row"],  # type: ignore[arg-type]
        torch.tensor([4]),
    )


def test_posttraining_generation_limits_ignore_reference_length() -> None:
    config = PostTrainingConfig(
        max_new_tokens=20,
        decode_max_output_length_ratio=2.0,
        decode_max_output_length_margin=3,
    )
    objective = MinimumRiskObjective(TextTokenizer(), config)
    model = MemoryAwareCandidateScorer()
    attention_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ]
    )

    max_new_tokens, row_limits = objective._deployment_generation_limits(
        model,  # type: ignore[arg-type]
        attention_mask,
    )

    # Remove the language tag and source EOS before applying the ratio:
    # (4 - 2) * 2 + 3 = 7 and (6 - 2) * 2 + 3 = 11.
    assert max_new_tokens == 11
    torch.testing.assert_close(row_limits, torch.tensor([7, 11]))


def test_reward_cpu_work_overlaps_candidate_scoring_and_reports_wait_telemetry() -> None:
    config = PostTrainingConfig(
        samples_per_source=2,
        candidate_micro_batch=1,
        candidate_gradient_checkpointing=False,
        max_new_tokens=4,
    )
    objective = MinimumRiskObjective(TextTokenizer(), config)
    reward_started = threading.Event()
    candidate_scoring_started = threading.Event()
    model = ConcurrentCandidateScorer(reward_started, candidate_scoring_started)
    real_score_cpu = objective.reward_model.score_cpu
    worker_names: list[str] = []

    def synchronized_score(*args, **kwargs):
        worker_names.append(threading.current_thread().name)
        reward_started.set()
        if not candidate_scoring_started.wait(timeout=2):
            raise AssertionError("candidate scoring did not overlap CPU reward")
        return real_score_cpu(*args, **kwargs)

    objective.reward_model.score_cpu = synchronized_score
    output = objective(model, posttraining_batch())

    assert reward_started.is_set()
    assert candidate_scoring_started.is_set()
    assert worker_names and worker_names[0].startswith("sion-reward")
    for metric_name in (
        "reward_cpu_seconds",
        "reward_wait_seconds",
        "reward_overlap_seconds",
        "reward_overlap_fraction",
        "reward_input_transfer_seconds",
        "candidate_scoring_seconds",
        "candidate_preference_loss",
        "reference_preference_loss",
        "reference_preference_margin",
        "reference_preference_violation_rate",
        "reference_preference_comparison_fraction",
        "reference_sequence_score",
    ):
        assert torch.isfinite(output.metrics[metric_name])
    for metric_name in (
        "reward_cpu_seconds",
        "reward_wait_seconds",
        "reward_overlap_seconds",
        "reward_overlap_fraction",
        "reward_input_transfer_seconds",
        "candidate_scoring_seconds",
        "candidate_preference_loss",
        "reference_preference_loss",
    ):
        assert output.metrics[metric_name].item() >= 0.0
    assert 0.0 <= output.metrics["reference_preference_violation_rate"].item() <= 1.0
    assert 0.0 <= output.metrics["reference_preference_comparison_fraction"].item() <= 1.0
    assert output.metrics["reward_overlap_fraction"].item() <= 1.0
    assert (
        output.metrics["reward_overlap_seconds"].item()
        <= output.metrics["candidate_scoring_seconds"].item()
    )
    assert (
        output.metrics["reward_overlap_seconds"].item()
        <= output.metrics["reward_cpu_seconds"].item()
    )
    output.loss_sum.backward()
    assert model.embedding.weight.grad is not None


def test_mrt_autocast_cache_is_disabled_for_generation_and_checkpoint_replay() -> None:
    """No-grad casts and cross-chunk casts must not leak into MRT scoring.

    CUDA autocast in PyTorch 2.8 otherwise gives later candidate chunks a
    different saved-tensor graph from their fresh-cache backward replay.  CPU
    autocast exposes the cache state directly, so this regression test does not
    need CUDA to protect the required context boundaries.
    """

    config = PostTrainingConfig(
        samples_per_source=2,
        candidate_micro_batch=1,
        candidate_gradient_checkpointing=True,
        max_new_tokens=4,
    )
    objective = MinimumRiskObjective(TextTokenizer(), config)
    model = AutocastCacheRecordingScorer()

    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=True):
        output = objective(model, posttraining_batch())
    output.loss_sum.backward()

    assert model.sample_cache_states == [False]
    candidate_cache_states = [
        cache_enabled
        for labels_are_absent, cache_enabled in model.forward_cache_states
        if labels_are_absent
    ]
    # Two original chunks and their two backward recomputations all use the
    # same cache-disabled mixed-precision context.
    assert candidate_cache_states == [False, False, False, False]
    reference_cache_states = [
        cache_enabled
        for labels_are_absent, cache_enabled in model.forward_cache_states
        if not labels_are_absent
    ]
    # The ordinary reference pass may keep the outer cache: unlike no-grad
    # generation and checkpoint replay, it owns all of its parameter edges.
    assert reference_cache_states == [True]
    assert model.projection.weight.grad is not None


def test_reward_worker_exception_propagates_without_leaking_executor_thread() -> None:
    config = PostTrainingConfig(
        samples_per_source=2,
        candidate_micro_batch=1,
        candidate_gradient_checkpointing=False,
        max_new_tokens=4,
    )
    objective = MinimumRiskObjective(TextTokenizer(), config)
    reward_started = threading.Event()
    candidate_scoring_started = threading.Event()
    model = ConcurrentCandidateScorer(reward_started, candidate_scoring_started)

    def fail_reward(*args, **kwargs):
        del args, kwargs
        reward_started.set()
        raise RuntimeError("intentional reward failure")

    objective.reward_model.score_cpu = fail_reward
    with pytest.raises(RuntimeError, match="intentional reward failure"):
        objective(model, posttraining_batch())
    assert not any(
        thread.name.startswith("sion-reward") and thread.is_alive()
        for thread in threading.enumerate()
    )


def _legacy_full_candidate_loss(
    objective: MinimumRiskObjective,
    model: TinyCandidateScorer,
    batch: dict[str, torch.Tensor],
    decoder_inputs: torch.Tensor,
    labels: torch.Tensor,
    rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, repeats, target_length = decoder_inputs.shape
    repeated = objective._repeated_model_inputs(batch, repeats)
    output = model(
        **repeated,
        decoder_input_ids=decoder_inputs.reshape(batch_size * repeats, target_length),
        labels=None,
    )
    logits = output.logits.float().view(batch_size, repeats, target_length, output.logits.shape[-1])
    token_log_probs = (
        F.log_softmax(logits, dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    )
    valid = labels.ne(-100)
    samples = repeats - 1
    generated_valid = valid[:, :samples]
    generated_scores = token_log_probs[:, :samples].masked_fill(~generated_valid, 0.0).sum(
        -1
    ) / generated_valid.sum(-1).clamp_min(1)
    candidate_distribution = torch.softmax(objective.config.mrt_alpha * generated_scores, dim=-1)
    risk = (candidate_distribution * (1.0 - rewards)).sum(-1).mean()
    preference_loss, _ = objective._pairwise_preference_loss(generated_scores, rewards)
    reference_logits = logits[:, samples]
    reference_labels = labels[:, samples]
    reference_valid = reference_labels.ne(-100)
    reference_token_nll = F.cross_entropy(
        reference_logits.transpose(1, 2),
        reference_labels,
        ignore_index=-100,
        reduction="none",
    )
    reference_scores = -reference_token_nll.masked_fill(~reference_valid, 0.0).sum(
        -1
    ) / reference_valid.sum(-1).clamp_min(1)
    reference_preference_loss, _, _, _ = objective._reference_preference_loss(
        generated_scores,
        reference_scores,
        labels[:, :samples],
        reference_labels,
    )
    ce_sum = F.cross_entropy(
        reference_logits.reshape(-1, reference_logits.shape[-1]),
        reference_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
        label_smoothing=model.config.label_smoothing,
    )
    ce_loss = ce_sum / reference_labels.ne(-100).sum().clamp_min(1)
    return (
        ce_loss
        + objective.config.risk_weight * risk
        + objective.config.preference_weight * (preference_loss + reference_preference_loss),
        generated_scores,
    )


def test_candidate_micro_batches_match_legacy_loss_and_gradients() -> None:
    torch.manual_seed(17)
    config = PostTrainingConfig(
        samples_per_source=3,
        candidate_micro_batch=1,
        risk_weight=0.2,
        preference_weight=0.1,
    )
    objective = MinimumRiskObjective(TextTokenizer(), config)
    legacy_model = TinyCandidateScorer()
    chunked_model = deepcopy(legacy_model)
    batch = {
        "input_ids": torch.tensor([[4, 10, 3], [4, 11, 3]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.bool),
        "decoder_input_ids": torch.tensor([[2, 20, 21, 0], [2, 22, 23, 0]]),
        "labels": torch.tensor([[20, 21, 3, -100], [22, 23, 3, -100]]),
    }
    candidate_inputs = torch.tensor(
        [
            [[2, 20, 21, 0], [2, 22, 21, 0], [2, 23, 21, 0]],
            [[2, 22, 23, 0], [2, 20, 23, 0], [2, 21, 23, 0]],
        ]
    )
    candidate_labels = torch.tensor(
        [
            [[20, 21, 3, -100], [22, 21, 3, -100], [23, 21, 3, -100]],
            [[22, 23, 3, -100], [20, 23, 3, -100], [21, 23, 3, -100]],
        ]
    )
    decoder_inputs = torch.cat((candidate_inputs, batch["decoder_input_ids"][:, None]), dim=1)
    labels = torch.cat((candidate_labels, batch["labels"][:, None]), dim=1)
    rewards = torch.tensor([[0.9, 0.5, 0.2], [0.8, 0.4, 0.1]])

    legacy_loss, legacy_scores = _legacy_full_candidate_loss(
        objective,
        legacy_model,
        batch,
        decoder_inputs,
        labels,
        rewards,
    )
    score_chunks = [
        objective._sequence_log_probabilities(
            chunked_model,
            batch,
            decoder_inputs[:, start : start + config.candidate_micro_batch],
            labels[:, start : start + config.candidate_micro_batch],
        )
        for start in range(0, config.samples_per_source, config.candidate_micro_batch)
    ]
    chunked_scores = torch.cat(score_chunks, dim=1)
    ce_loss, _, reference_auxiliary_loss, reference_scores, _ = objective._reference_cross_entropy(
        chunked_model,
        batch,
        decoder_inputs[:, config.samples_per_source],
        labels[:, config.samples_per_source],
        label_smoothing=chunked_model.config.label_smoothing,
    )
    distribution = torch.softmax(config.mrt_alpha * chunked_scores, dim=-1)
    risk = (distribution * (1.0 - rewards)).sum(-1).mean()
    preference_loss, _ = objective._pairwise_preference_loss(chunked_scores, rewards)
    reference_preference_loss, _, _, _ = objective._reference_preference_loss(
        chunked_scores,
        reference_scores,
        labels[:, : config.samples_per_source],
        labels[:, config.samples_per_source],
    )
    chunked_loss = (
        ce_loss
        + reference_auxiliary_loss
        + config.risk_weight * risk
        + config.preference_weight * (preference_loss + reference_preference_loss)
    )

    torch.testing.assert_close(chunked_scores, legacy_scores)
    torch.testing.assert_close(chunked_loss, legacy_loss)
    assert legacy_model.forward_batch_sizes == [8]
    assert chunked_model.forward_batch_sizes == [2, 2, 2, 2]
    legacy_loss.backward()
    chunked_loss.backward()
    for legacy_parameter, chunked_parameter in zip(
        legacy_model.parameters(), chunked_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            chunked_parameter.grad,
            legacy_parameter.grad,
            rtol=1e-5,
            atol=1e-6,
        )
    assert chunked_model.forward_batch_sizes == [2, 2, 2, 2, 2, 2, 2]
