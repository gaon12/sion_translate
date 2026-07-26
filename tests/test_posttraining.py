from __future__ import annotations

import torch

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
    candidates = torch.tensor(
        [[[2, 20, 21, 30, 3], [2, 20, 22, 0, 3]]]
    )

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

    aligned_loss, aligned_weight = objective._pairwise_preference_loss(
        aligned_scores, rewards
    )
    reversed_loss, reversed_weight = objective._pairwise_preference_loss(
        reversed_scores, rewards
    )

    assert aligned_weight.item() == reversed_weight.item()
    assert aligned_loss < reversed_loss
    reversed_loss.backward()
    assert reversed_scores.grad is not None
