from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

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


def test_composite_reward_applies_repetition_and_source_copy_penalties() -> None:
    base = PostTrainingConfig(
        reward_repetition_penalty=0.0,
        reward_copy_penalty=0.0,
    )
    penalized = PostTrainingConfig(
        reward_repetition_penalty=0.4,
        reward_copy_penalty=0.4,
    )
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
    assert penalized_score < base_score - 0.7


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
        + objective.config.preference_weight * preference_loss,
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
    ce_loss, _ = objective._reference_cross_entropy(
        chunked_model,
        batch,
        decoder_inputs[:, config.samples_per_source],
        labels[:, config.samples_per_source],
        label_smoothing=chunked_model.config.label_smoothing,
    )
    distribution = torch.softmax(config.mrt_alpha * chunked_scores, dim=-1)
    risk = (distribution * (1.0 - rewards)).sum(-1).mean()
    preference_loss, _ = objective._pairwise_preference_loss(chunked_scores, rewards)
    chunked_loss = ce_loss + config.risk_weight * risk + config.preference_weight * preference_loss

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
