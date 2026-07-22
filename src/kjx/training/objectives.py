from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata

import torch
import torch.nn.functional as F
from sacrebleu.metrics import CHRF
from torch import nn

from kjx.config import PostTrainingConfig
from kjx.data.quality import canonical_text, language_fraction
from kjx.tokenizer import KJTokenizer

from .export import unwrap_model


_NUMBER = re.compile(r"(?<![\w])[-+]?\d[\d,.:/%+\-]*\d|(?<![\w])[-+]?\d(?![\w])")
_STRUCTURED = re.compile(
    r"https?://[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Z0-9_-]*|[A-Za-z]+[-_][A-Za-z0-9_-]+)"
    r"(?![A-Za-z0-9])"
)


@dataclass
class ObjectiveOutput:
    """Trainer가 목적함수 종류와 무관하게 누적·정규화할 수 있는 출력."""

    loss_sum: torch.Tensor
    normalizer: torch.Tensor
    processed_tokens: torch.Tensor
    auxiliary_loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass
class RewardOutput:
    """문장별 최종 보상과 진단 가능한 세부 항목."""

    reward: torch.Tensor
    components: dict[str, torch.Tensor]


def _multiset_f1(expected: list[object], actual: list[object]) -> float:
    """중복을 보존하는 F1. 둘 다 비었으면 위반이 없으므로 1입니다."""
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    if not expected_counts and not actual_counts:
        return 1.0
    if not expected_counts or not actual_counts:
        return 0.0
    overlap = sum((expected_counts & actual_counts).values())
    precision = overlap / sum(actual_counts.values())
    recall = overlap / sum(expected_counts.values())
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def _normalized_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text)
    return [match.group(0).casefold().rstrip(".,;:!?") for match in pattern.finditer(normalized)]


def _has_excessive_repetition(text: str) -> bool:
    surface = [char for char in text if not char.isspace()]
    if len(surface) < 12:
        return False
    if Counter(surface).most_common(1)[0][1] / len(surface) >= 0.70:
        return True
    return re.search(r"(.{1,8})\1{4,}", "".join(surface)) is not None


class CompositeTranslationReward:
    """참조 유사도와 번역 보존성/건전성을 결합한 로컬 보상.

    단일 metric 최적화의 reward hacking을 줄이기 위해 chrF 외에 token F1,
    숫자·URL·이메일·슬롯 보존, 목표 언어 문자, 길이를 함께 사용합니다.
    반복 문자열과 원문 복사에는 별도의 감점을 적용합니다.
    """

    def __init__(self, tokenizer: KJTokenizer, config: PostTrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.chrf = CHRF(word_order=0)
        self.language_by_tag = {
            token_id: language for language, token_id in tokenizer.language_tags.items()
        }
        self.slot_ids = set(getattr(tokenizer, "slot_ids", ()))
        self.special_ids = {
            tokenizer.pad_id,
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.mask_id,
            *tokenizer.language_tags.values(),
            *tokenizer.denoise_tags.values(),
        }
        self.weights = {
            "chrf": config.reward_chrf_weight,
            "token_f1": config.reward_token_f1_weight,
            "number": config.reward_number_weight,
            "structured": config.reward_structured_weight,
            "slot": config.reward_slot_weight,
            "language": config.reward_language_weight,
            "length": config.reward_length_weight,
        }
        self.weight_sum = sum(self.weights.values())

    def decode(self, token_ids: list[int]) -> str:
        kept: list[int] = []
        for token_id in token_ids:
            if token_id == self.tokenizer.eos_id:
                break
            if token_id >= 0 and token_id not in self.special_ids:
                kept.append(token_id)
        return self.tokenizer.decode(kept).strip()

    def _content_ids(self, token_ids: list[int]) -> list[int]:
        content: list[int] = []
        for token_id in token_ids:
            if token_id == self.tokenizer.eos_id:
                break
            if token_id >= 0 and token_id not in self.special_ids:
                content.append(token_id)
        return content

    def _score_one(
        self,
        source_ids: list[int],
        candidate_ids: list[int],
        reference_ids: list[int],
        target_language: str | None,
    ) -> tuple[float, dict[str, float]]:
        source_text = self.decode(source_ids)
        hypothesis = self.decode(candidate_ids)
        reference_text = self.decode(reference_ids)
        if not hypothesis:
            return 0.0, {name: 0.0 for name in self.weights}

        candidate_content = self._content_ids(candidate_ids)
        reference_content = self._content_ids(reference_ids)
        chrf = self.chrf.sentence_score(hypothesis, [reference_text]).score / 100.0
        token_f1 = _multiset_f1(reference_content, candidate_content)
        number = _multiset_f1(
            _normalized_matches(_NUMBER, reference_text),
            _normalized_matches(_NUMBER, hypothesis),
        )
        structured = _multiset_f1(
            _normalized_matches(_STRUCTURED, reference_text),
            _normalized_matches(_STRUCTURED, hypothesis),
        )
        source_slots = [token_id for token_id in source_ids if token_id in self.slot_ids]
        candidate_slots = [token_id for token_id in candidate_ids if token_id in self.slot_ids]
        slot = _multiset_f1(source_slots, candidate_slots)
        letter_count = sum(char.isalpha() for char in hypothesis)
        language = (
            language_fraction(hypothesis, target_language)
            if target_language is not None and letter_count >= 4
            else 1.0
        )
        reference_length = sum(not char.isspace() for char in reference_text)
        hypothesis_length = sum(not char.isspace() for char in hypothesis)
        length = math.exp(
            -abs(math.log((hypothesis_length + 1.0) / (reference_length + 1.0)))
        )
        components = {
            "chrf": chrf,
            "token_f1": token_f1,
            "number": number,
            "structured": structured,
            "slot": slot,
            "language": language,
            "length": length,
        }
        reward = sum(self.weights[name] * value for name, value in components.items())
        reward /= self.weight_sum
        if _has_excessive_repetition(hypothesis):
            reward -= self.config.reward_repetition_penalty
        if canonical_text(source_text).casefold() == canonical_text(hypothesis).casefold():
            reward -= self.config.reward_copy_penalty
        return max(0.0, min(1.0, reward)), components

    def __call__(
        self,
        candidates: torch.Tensor,
        input_ids: torch.Tensor,
        reference_labels: torch.Tensor,
        *,
        candidates_include_bos: bool = True,
    ) -> RewardOutput:
        """candidates ``(batch, samples, length)``의 복합 보상을 계산합니다."""
        candidates_cpu = candidates.detach().to("cpu")
        inputs_cpu = input_ids.detach().to("cpu")
        references_cpu = reference_labels.detach().to("cpu")
        reward_rows: list[list[float]] = []
        component_rows: dict[str, list[list[float]]] = {
            name: [] for name in self.weights
        }
        for source, candidate_row, reference in zip(
            inputs_cpu, candidates_cpu, references_cpu, strict=True
        ):
            source_ids = source.tolist()
            reference_ids = reference.tolist()
            target_language = self.language_by_tag.get(source_ids[0]) if source_ids else None
            row_rewards: list[float] = []
            row_components = {name: [] for name in self.weights}
            for candidate in candidate_row:
                candidate_ids = candidate.tolist()
                if candidates_include_bos:
                    candidate_ids = candidate_ids[1:]
                reward, components = self._score_one(
                    source_ids, candidate_ids, reference_ids, target_language
                )
                row_rewards.append(reward)
                for name, value in components.items():
                    row_components[name].append(value)
            reward_rows.append(row_rewards)
            for name, values in row_components.items():
                component_rows[name].append(values)
        device = candidates.device
        return RewardOutput(
            reward=torch.tensor(reward_rows, device=device, dtype=torch.float32),
            components={
                name: torch.tensor(rows, device=device, dtype=torch.float32)
                for name, rows in component_rows.items()
            },
        )


class MinimumRiskObjective:
    """Reference CE + 복합 MRT + 다중 후보쌍 선호학습 목적함수."""

    def __init__(self, tokenizer: KJTokenizer, config: PostTrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.reward_model = CompositeTranslationReward(tokenizer, config)
        self.special_ids = self.reward_model.special_ids

    @staticmethod
    def _candidate_labels(sampled: torch.Tensor, eos_id: int) -> torch.Tensor:
        labels = sampled[..., 1:].clone()
        # sample()은 종료 뒤 EOS를 반복해 길이를 맞춥니다. 첫 EOS만 loss에 남깁니다.
        labels.masked_fill_(labels.eq(eos_id).cumsum(dim=-1) > 1, -100)
        return labels

    def _max_new_tokens(self, base: nn.Module, reference_labels: torch.Tensor) -> int:
        return min(
            self.config.max_new_tokens,
            base.config.max_seq_len - 1,
            reference_labels.shape[1] + 32,
        )

    @staticmethod
    def _encoder_features(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        names = ("src_script_ids", "src_onset_ids", "src_vowel_ids", "src_coda_ids")
        return {name: batch[name] for name in names if name in batch}

    def _pairwise_preference_loss(
        self, generated_scores: torch.Tensor, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """reward 차이가 충분한 모든 ordered candidate pair를 학습합니다."""
        reward_gap = rewards[:, :, None] - rewards[:, None, :]
        score_gap = generated_scores[:, :, None] - generated_scores[:, None, :]
        mask = reward_gap > self.config.preference_min_gap
        weights = reward_gap.masked_fill(~mask, 0.0)
        losses = F.softplus(-score_gap / self.config.preference_temperature)
        pair_weight = weights.sum()
        if not bool(mask.any()):
            return generated_scores.sum() * 0.0, pair_weight
        return (losses * weights).sum() / pair_weight.clamp_min(1e-8), pair_weight

    @torch.no_grad()
    def validation_metrics(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """실제 추론과 가까운 beam 출력으로 사후학습 품질을 검증합니다."""
        base = unwrap_model(model)
        generated = base.generate(
            batch["input_ids"],
            batch["attention_mask"],
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            max_new_tokens=self._max_new_tokens(base, batch["labels"]),
            num_beams=self.config.validation_num_beams,
            **self._encoder_features(batch),
        )
        reward_output = self.reward_model(
            generated[:, None, :], batch["input_ids"], batch["labels"]
        )
        metrics = {"reward": reward_output.reward.mean()}
        metrics.update(
            {f"reward_{name}": values.mean() for name, values in reward_output.components.items()}
        )
        return metrics

    def __call__(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> ObjectiveOutput:
        base = unwrap_model(model)
        reference_labels = batch["labels"]
        batch_size = reference_labels.shape[0]
        encoder_features = self._encoder_features(batch)
        forbidden = tuple(
            sorted(
                token_id
                for token_id in self.special_ids - {self.tokenizer.eos_id}
                if 0 <= token_id < base.config.vocab_size
            )
        )
        sampled = base.sample(
            batch["input_ids"],
            batch["attention_mask"],
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            num_samples=self.config.samples_per_source,
            max_new_tokens=self._max_new_tokens(base, reference_labels),
            temperature=self.config.sampling_temperature,
            top_k=self.config.top_k,
            forbidden_token_ids=forbidden,
            **encoder_features,
        )
        reward_output = self.reward_model(sampled, batch["input_ids"], reference_labels)
        rewards = reward_output.reward

        samples = self.config.samples_per_source
        candidate_inputs = sampled[..., :-1]
        candidate_labels = self._candidate_labels(sampled, self.tokenizer.eos_id)
        reference_inputs = batch["decoder_input_ids"][:, None, :]
        reference_targets = reference_labels[:, None, :]
        target_length = max(candidate_inputs.shape[-1], reference_inputs.shape[-1])

        decoder_inputs = torch.full(
            (batch_size, samples + 1, target_length),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=sampled.device,
        )
        labels = torch.full_like(decoder_inputs, -100)
        decoder_inputs[:, :samples, : candidate_inputs.shape[-1]] = candidate_inputs
        labels[:, :samples, : candidate_labels.shape[-1]] = candidate_labels
        decoder_inputs[:, samples:, : reference_inputs.shape[-1]] = reference_inputs
        labels[:, samples:, : reference_targets.shape[-1]] = reference_targets

        repeats = samples + 1
        repeated: dict[str, torch.Tensor] = {}
        for name, value in batch.items():
            if name in {"decoder_input_ids", "labels", "register_labels", "alignment_targets"}:
                continue
            repeated[name] = value.repeat_interleave(repeats, dim=0)
        output = model(
            **repeated,
            decoder_input_ids=decoder_inputs.reshape(batch_size * repeats, target_length),
            labels=None,
        )
        logits = output.logits.float().view(
            batch_size, repeats, target_length, output.logits.shape[-1]
        )
        log_probs = F.log_softmax(logits, dim=-1)
        safe_labels = labels.clamp_min(0)
        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        valid = labels.ne(-100)

        generated_valid = valid[:, :samples]
        generated_scores = (
            token_log_probs[:, :samples].masked_fill(~generated_valid, 0.0).sum(-1)
            / generated_valid.sum(-1).clamp_min(1)
        )
        candidate_distribution = torch.softmax(
            self.config.mrt_alpha * generated_scores, dim=-1
        )
        risk = (candidate_distribution * (1.0 - rewards)).sum(-1).mean()
        preference_loss, pair_weight = self._pairwise_preference_loss(
            generated_scores, rewards
        )

        reference_logits = logits[:, samples]
        reference_flat = labels[:, samples]
        ce_sum = F.cross_entropy(
            reference_logits.reshape(-1, reference_logits.shape[-1]),
            reference_flat.reshape(-1),
            ignore_index=-100,
            reduction="sum",
            label_smoothing=base.config.label_smoothing,
        )
        reference_tokens = reference_flat.ne(-100).sum()
        ce_loss = ce_sum / reference_tokens.clamp_min(1)
        total_loss = (
            ce_loss
            + self.config.risk_weight * risk
            + self.config.preference_weight * preference_loss
        )
        normalizer = torch.tensor(float(batch_size), device=sampled.device)
        metrics = {
            "ce_loss": ce_loss.detach(),
            "risk": risk.detach(),
            "preference_loss": preference_loss.detach(),
            "preference_pair_weight": pair_weight.detach(),
            "reward": rewards.mean().detach(),
        }
        metrics.update(
            {
                f"reward_{name}": values.mean().detach()
                for name, values in reward_output.components.items()
            }
        )
        return ObjectiveOutput(
            loss_sum=total_loss * normalizer,
            normalizer=normalizer,
            processed_tokens=reference_tokens.detach(),
            auxiliary_loss=(
                self.config.risk_weight * risk
                + self.config.preference_weight * preference_loss
            ).detach(),
            metrics=metrics,
        )


__all__ = [
    "CompositeTranslationReward",
    "MinimumRiskObjective",
    "ObjectiveOutput",
    "RewardOutput",
]
