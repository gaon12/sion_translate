from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
from sacrebleu.metrics.chrf import CHRF
from torch import nn
from torch.utils.checkpoint import checkpoint  # pyright: ignore[reportUnknownVariableType]

from sion_translate.config import PostTrainingConfig
from sion_translate.data.quality import canonical_text, language_fraction
from sion_translate.evaluation import (
    has_excessive_repetition,
    multiset_f1,
    numeric_corruption,
    numeric_tokens,
    structured_tokens,
)
from sion_translate.model import SionForConditionalGeneration
from sion_translate.tokenizer import SionTokenizer

from .export import unwrap_model


_activation_checkpoint = cast(Callable[..., torch.Tensor], checkpoint)


def _autocast_without_weight_cache(device_type: str) -> AbstractContextManager[Any]:
    """Preserve mixed precision without reusing casts across grad modes.

    MRT generates candidates under ``no_grad`` before scoring them with
    gradients.  PyTorch's autocast weight cache is shared by nested contexts,
    so a cast first created by generation can be reused by the training pass
    without its parameter edge.  Checkpointed candidate chunks have a second
    problem: each backward recomputation starts with a fresh autocast cache,
    while later original chunks can reuse casts from an earlier chunk.  On
    PyTorch 2.8/CUDA that difference raises ``CheckpointError`` because the
    saved-tensor sequence no longer matches.

    A nested cache-disabled context keeps the selected BF16/FP16 dtype while
    making both sides build the same autograd graph.  Outside autocast it is a
    no-op.
    """

    if not torch.is_autocast_enabled(device_type):
        return nullcontext()
    return torch.autocast(
        device_type=device_type,
        dtype=torch.get_autocast_dtype(device_type),
        cache_enabled=False,
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


class CompositeTranslationReward:
    """참조 유사도와 번역 보존성/건전성을 결합한 로컬 보상.

    단일 metric 최적화의 reward hacking을 줄이기 위해 chrF 외에 token F1,
    숫자·URL·이메일·슬롯 보존, 목표 언어 문자, 길이를 함께 사용합니다.
    반복 문자열과 원문 복사에는 별도의 감점을 적용합니다.
    """

    def __init__(self, tokenizer: SionTokenizer, config: PostTrainingConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.chrf = CHRF(word_order=0)
        self.language_by_tag = {
            token_id: language for language, token_id in tokenizer.language_tags.items()
        }
        self.slot_ids: set[int] = set(cast(Iterable[int], getattr(tokenizer, "slot_ids", ())))
        self.special_ids: set[int] = {
            tokenizer.pad_id,
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.mask_id,
            *tokenizer.language_tags.values(),
            *tokenizer.denoise_tags.values(),
            *getattr(tokenizer, "reasoning_tags", {}).values(),
            *getattr(tokenizer, "reasoning_trace_ids", {}).values(),
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
        if config.roundtrip_enabled:
            self.weights["roundtrip"] = config.roundtrip_reward_weight

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
        roundtrip_ids: list[int] | None,
    ) -> tuple[float, dict[str, float]]:
        source_text = self.decode(source_ids)
        hypothesis = self.decode(candidate_ids)
        reference_text = self.decode(reference_ids)
        if not hypothesis:
            return 0.0, {name: 0.0 for name in self.weights}

        candidate_content = self._content_ids(candidate_ids)
        reference_content = self._content_ids(reference_ids)
        chrf = self.chrf.sentence_score(hypothesis, [reference_text]).score / 100.0
        token_f1 = multiset_f1(reference_content, candidate_content)
        # 평가(sion-evaluate / sion-compare)와 같은 정의를 씁니다. 보상과 지표가
        # 다른 것을 재면 사후학습이 개선했다는 항목이 리포트에 나타나지 않습니다.
        number = multiset_f1(numeric_tokens(reference_text), numeric_tokens(hypothesis))
        structured = multiset_f1(structured_tokens(reference_text), structured_tokens(hypothesis))
        source_slots = [token_id for token_id in source_ids if token_id in self.slot_ids]
        candidate_slots = [token_id for token_id in candidate_ids if token_id in self.slot_ids]
        slot = multiset_f1(source_slots, candidate_slots)
        letter_count = sum(char.isalpha() for char in hypothesis)
        language = (
            language_fraction(hypothesis, target_language)
            if target_language is not None and letter_count >= 4
            else None
        )
        reference_length = sum(not char.isspace() for char in reference_text)
        hypothesis_length = sum(not char.isspace() for char in hypothesis)
        length = math.exp(-abs(math.log((hypothesis_length + 1.0) / (reference_length + 1.0))))
        components: dict[str, float] = {
            "chrf": chrf,
            "token_f1": token_f1,
            "number": number,
            "structured": structured,
            "slot": slot,
            "length": length,
        }
        active_weights = dict(self.weights)
        if language is None:
            active_weights.pop("language", None)
        else:
            components["language"] = language
        if "roundtrip" in self.weights:
            roundtrip = 0.0
            if roundtrip_ids is None:
                # 단일언어 denoise처럼 역방향이 정의되지 않은 행은 기존
                # 정방향 보상만 사용합니다.
                active_weights.pop("roundtrip")
            else:
                roundtrip_text = self.decode(roundtrip_ids)
                roundtrip_content = self._content_ids(roundtrip_ids)
                cycle_chrf = (
                    self.chrf.sentence_score(roundtrip_text, [source_text]).score / 100.0
                    if roundtrip_text
                    else 0.0
                )
                cycle_token_f1 = multiset_f1(
                    self._content_ids(source_ids),
                    roundtrip_content,
                )
                cycle_number = multiset_f1(
                    numeric_tokens(source_text),
                    numeric_tokens(roundtrip_text),
                )
                cycle_structured = multiset_f1(
                    structured_tokens(source_text),
                    structured_tokens(roundtrip_text),
                )
                roundtrip = (
                    0.50 * cycle_chrf
                    + 0.25 * cycle_token_f1
                    + 0.15 * cycle_number
                    + 0.10 * cycle_structured
                )
            components["roundtrip"] = roundtrip

        weight_sum = sum(active_weights.values())
        reward = sum(active_weights[name] * components[name] for name in active_weights)
        reward /= weight_sum
        # Repeated laughter, cries and prolonged vowels are legitimate when the
        # reference contains the same expressive device. Penalize degeneration,
        # not the very phenomenon the model is being taught to preserve.
        if has_excessive_repetition(hypothesis) and not has_excessive_repetition(reference_text):
            reward -= self.config.reward_repetition_penalty
        # 값 변조는 개수로 셉니다. `number` 성분은 비율이라 값 하나를 지어낸
        # 후보도 chrF 가 조금 높으면 이기고, 그것이 배포 홀드아웃에서 10문장 중
        # 8문장의 숫자가 바뀐 이유입니다. 여기서는 변조 하나당 고정액을 빼서
        # 숫자를 틀린 후보가 이기지 못하게 합니다.
        if self.config.reward_number_corruption_penalty > 0:
            invented, dropped = numeric_corruption(source_text, reference_text, hypothesis)
            if invented or dropped:
                reward -= self.config.reward_number_corruption_penalty * (invented + dropped)
        source_key = canonical_text(source_text).casefold()
        hypothesis_key = canonical_text(hypothesis).casefold()
        reference_key = canonical_text(reference_text).casefold()
        if source_key == hypothesis_key and source_key != reference_key:
            reward -= self.config.reward_copy_penalty
        if roundtrip_ids is not None and "roundtrip" in components:
            threshold = self.config.roundtrip_min_score
            if threshold > 0 and components["roundtrip"] < threshold:
                shortfall = (threshold - components["roundtrip"]) / threshold
                reward -= self.config.roundtrip_failure_penalty * shortfall
        return max(0.0, min(1.0, reward)), components

    def __call__(
        self,
        candidates: torch.Tensor,
        input_ids: torch.Tensor,
        reference_labels: torch.Tensor,
        *,
        candidates_include_bos: bool = True,
        roundtrip_candidates: torch.Tensor | None = None,
        roundtrip_mask: torch.Tensor | None = None,
    ) -> RewardOutput:
        """candidates ``(batch, samples, length)``의 복합 보상을 계산합니다."""

        cpu_output = self.score_cpu(
            candidates,
            input_ids,
            reference_labels,
            candidates_include_bos=candidates_include_bos,
            roundtrip_candidates=roundtrip_candidates,
            roundtrip_mask=roundtrip_mask,
        )
        device = candidates.device
        return RewardOutput(
            reward=cpu_output.reward.to(device=device),
            components={
                name: values.to(device=device) for name, values in cpu_output.components.items()
            },
        )

    def score_cpu(
        self,
        candidates: torch.Tensor,
        input_ids: torch.Tensor,
        reference_labels: torch.Tensor,
        *,
        candidates_include_bos: bool = True,
        roundtrip_candidates: torch.Tensor | None = None,
        roundtrip_mask: torch.Tensor | None = None,
    ) -> RewardOutput:
        """Detach inputs and calculate the Python/string reward entirely on CPU."""

        candidates_cpu = candidates.detach().to("cpu")
        inputs_cpu = input_ids.detach().to("cpu")
        references_cpu = reference_labels.detach().to("cpu")
        roundtrips_cpu = (
            roundtrip_candidates.detach().to("cpu") if roundtrip_candidates is not None else None
        )
        roundtrip_mask_cpu = (
            roundtrip_mask.detach().to(device="cpu", dtype=torch.bool)
            if roundtrip_mask is not None
            else None
        )
        reward_rows: list[list[float]] = []
        component_rows: dict[str, list[list[float]]] = {name: [] for name in self.weights}
        for row_index, (source, candidate_row, reference) in enumerate(
            zip(inputs_cpu, candidates_cpu, references_cpu, strict=True)
        ):
            source_ids = cast(
                list[int],
                source.tolist(),  # pyright: ignore[reportUnknownMemberType]
            )
            reference_ids = cast(
                list[int],
                reference.tolist(),  # pyright: ignore[reportUnknownMemberType]
            )
            target_language = self.language_by_tag.get(source_ids[0]) if source_ids else None
            row_rewards: list[float] = []
            row_components: dict[str, list[float]] = {name: [] for name in self.weights}
            for candidate_index, candidate in enumerate(candidate_row):
                candidate_ids = cast(
                    list[int],
                    candidate.tolist(),  # pyright: ignore[reportUnknownMemberType]
                )
                if candidates_include_bos:
                    candidate_ids = candidate_ids[1:]
                roundtrip_ids = None
                if roundtrips_cpu is not None and (
                    roundtrip_mask_cpu is None
                    or bool(roundtrip_mask_cpu[row_index, candidate_index])
                ):
                    roundtrip_ids = cast(
                        list[int],
                        roundtrips_cpu[row_index, candidate_index].tolist(),  # pyright: ignore[reportUnknownMemberType]
                    )
                    if candidates_include_bos:
                        roundtrip_ids = roundtrip_ids[1:]
                reward, components = self._score_one(
                    source_ids,
                    candidate_ids,
                    reference_ids,
                    target_language,
                    roundtrip_ids,
                )
                row_rewards.append(reward)
                for name, value in components.items():
                    row_components[name].append(value)
            reward_rows.append(row_rewards)
            for name, values in row_components.items():
                component_rows[name].append(values)
        return RewardOutput(
            reward=torch.tensor(reward_rows, dtype=torch.float32),
            components={
                name: torch.tensor(rows, dtype=torch.float32)
                for name, rows in component_rows.items()
            },
        )


class MinimumRiskObjective:
    """Reference CE + 복합 MRT + 다중 후보쌍 선호학습 목적함수."""

    def __init__(self, tokenizer: SionTokenizer, config: PostTrainingConfig):
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

    def _max_new_tokens(
        self, base: SionForConditionalGeneration, reference_labels: torch.Tensor
    ) -> int:
        return min(
            self.config.max_new_tokens,
            base.config.max_seq_len - 1,
            reference_labels.shape[1] + 32,
        )

    @staticmethod
    def _generation_features(batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        # Candidate scoring receives every non-target batch tensor through
        # ``_repeated_model_inputs``. Candidate generation must use the same
        # source-side context, otherwise a TETM-enabled model samples without
        # protected memory and then scores those candidates with it.
        names = (
            "src_script_ids",
            "src_onset_ids",
            "src_vowel_ids",
            "src_coda_ids",
            "memory_token_ids",
            "memory_mask",
            "memory_type_ids",
            "memory_mode_ids",
        )
        return {name: batch[name] for name in names if name in batch}

    @torch.no_grad()
    def _backtranslate_candidates(
        self,
        base: SionForConditionalGeneration,
        batch: dict[str, torch.Tensor],
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Translate candidates back to each row's source language in one batch.

        The collator marks denoising rows with ``-1`` and separately carries
        whether the dataset actually trained the reverse graph edge. All valid
        ``batch × candidates`` rows share one rollout to keep cycle checking
        substantially cheaper than invoking ``generate`` candidate by candidate.
        """
        source_tags = batch.get("source_language_tag_ids")
        if not self.config.roundtrip_enabled or source_tags is None:
            return None, None

        batch_size, samples, _ = candidates.shape
        reverse_direction_trained = batch.get("reverse_direction_trained")
        if reverse_direction_trained is None:
            eligible_rows = torch.zeros_like(source_tags, dtype=torch.bool)
        else:
            if reverse_direction_trained.shape != source_tags.shape:
                raise ValueError("reverse_direction_trained must match source_language_tag_ids")
            eligible_rows = source_tags.ge(0) & reverse_direction_trained.to(
                device=source_tags.device,
                dtype=torch.bool,
            )
        valid_mask = eligible_rows[:, None].expand(batch_size, samples)
        flat_valid_mask = valid_mask.reshape(-1)
        has_local_candidates = bool(flat_valid_mask.any())
        synchronize_generation = bool(getattr(base, "_synchronize_generation_across_ranks", False))
        if not has_local_candidates and not synchronize_generation:
            return None, valid_mask

        if has_local_candidates:
            flat_candidates = candidates.reshape(batch_size * samples, -1)[flat_valid_mask]
            flat_tags = (
                source_tags[:, None].expand(batch_size, samples).reshape(-1)[flat_valid_mask]
            )
            source_lengths = batch["attention_mask"].sum(dim=-1)[eligible_rows]
        else:
            # FSDP2 generation all-gathers parameters and synchronizes decode
            # termination on every rank. A rank whose local batch contains only
            # denoising or source-only rows must therefore enter generate() too,
            # or peers with eligible rows will wait forever in a collective.
            # This dummy rollout is masked out of the reward below.
            try:
                fallback_tag = next(iter(self.tokenizer.language_tags.values()))
            except StopIteration as error:
                raise RuntimeError(
                    "round-trip generation requires at least one language tag"
                ) from error
            flat_candidates = candidates.reshape(batch_size * samples, -1)[:1]
            flat_tags = torch.full(
                (1,),
                fallback_tag,
                dtype=source_tags.dtype,
                device=source_tags.device,
            )
            source_lengths = batch["attention_mask"].sum(dim=-1)[:1]

        # BOS와 기존 EOS를 제거한 후보 본문 뒤에 EOS를 정확히 한 번 붙입니다.
        max_content = max(0, base.config.max_seq_len - 2)
        raw_content = flat_candidates[:, 1 : 1 + max_content]
        before_eos = raw_content.eq(self.tokenizer.eos_id).cumsum(dim=-1).eq(0)
        content_mask = before_eos & raw_content.ne(self.tokenizer.pad_id)
        content_lengths = content_mask.sum(dim=-1)
        reverse_length = int(content_lengths.max().item()) + 2
        reverse_inputs = torch.full(
            (flat_candidates.shape[0], reverse_length),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=candidates.device,
        )
        reverse_inputs[:, 0] = flat_tags
        copied_length = reverse_length - 2
        if copied_length:
            reverse_inputs[:, 1 : 1 + copied_length] = raw_content[:, :copied_length].masked_fill(
                ~content_mask[:, :copied_length],
                self.tokenizer.pad_id,
            )
        reverse_inputs.scatter_(
            1,
            (content_lengths + 1).unsqueeze(1),
            self.tokenizer.eos_id,
        )
        positions = torch.arange(reverse_length, device=candidates.device)
        reverse_mask = positions[None, :] < (content_lengths + 2)[:, None]

        max_new_tokens = min(
            self.config.roundtrip_max_new_tokens,
            base.config.max_seq_len - 1,
            int(source_lengths.max().item()) + 32,
        )
        generated = base.generate(
            reverse_inputs,
            reverse_mask,
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            max_new_tokens=max_new_tokens,
            num_beams=self.config.roundtrip_num_beams,
        )
        roundtrips = torch.full(
            (batch_size * samples, generated.shape[-1]),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=candidates.device,
        )
        if has_local_candidates:
            roundtrips[flat_valid_mask] = generated
        return roundtrips.view(batch_size, samples, -1), valid_mask

    def _pairwise_preference_loss(
        self, generated_scores: torch.Tensor, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Train every ordered candidate pair with a sufficient reward gap."""
        if generated_scores.shape != rewards.shape:
            raise ValueError("generated scores and rewards must have the same shape")
        if not bool(torch.isfinite(generated_scores).all()):
            raise FloatingPointError("generated sequence scores must be finite")
        if not bool(torch.isfinite(rewards).all()):
            raise FloatingPointError("candidate rewards must be finite")
        reward_gap = rewards[:, :, None] - rewards[:, None, :]
        score_gap = generated_scores[:, :, None] - generated_scores[:, None, :]
        mask = reward_gap > self.config.preference_min_gap
        weights = reward_gap.masked_fill(~mask, 0.0)
        temperature = max(
            float(self.config.preference_temperature),
            torch.finfo(generated_scores.dtype).eps,
        )
        losses = F.softplus(-score_gap / temperature)
        pair_weight = weights.sum()
        if not bool(mask.any()):
            return generated_scores.sum() * 0.0, pair_weight
        return (losses * weights).sum() / pair_weight.clamp_min(1e-8), pair_weight

    def _reference_preference_loss(
        self,
        generated_scores: torch.Tensor,
        reference_scores: torch.Tensor,
        candidate_labels: torch.Tensor,
        reference_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefer the gold sequence over every distinct sampled candidate.

        Candidate-only MRT has no sequence-ordering gradient when every sample
        receives the same reward. The supervised cross-entropy term still
        teaches gold tokens independently, but it does not explicitly lower
        competing sequence probabilities. This anchor compares each distinct
        candidate with the gold sequence, independent of sampled reward ties.

        Exact gold duplicates are excluded because both sides represent the
        same token sequence and their pairwise gradients cancel. Empty rows are
        also excluded defensively. The returned diagnostics are the mean score
        margin, violation rate, and number of active comparisons.
        """
        if generated_scores.ndim != 2:
            raise ValueError("generated scores must have shape (batch, candidates)")
        batch_size, candidates = generated_scores.shape
        if reference_scores.shape != (batch_size,):
            raise ValueError("reference scores must have shape (batch,)")
        if candidate_labels.ndim != 3 or candidate_labels.shape[:2] != (
            batch_size,
            candidates,
        ):
            raise ValueError("candidate labels must have shape (batch, candidates, target length)")
        if reference_labels.shape != (batch_size, candidate_labels.shape[-1]):
            raise ValueError("reference labels must match candidate target length")
        if not bool(torch.isfinite(generated_scores).all()):
            raise FloatingPointError("generated sequence scores must be finite")
        if not bool(torch.isfinite(reference_scores).all()):
            raise FloatingPointError("reference sequence scores must be finite")

        candidate_valid = candidate_labels.ne(-100).any(dim=-1)
        reference_valid = reference_labels.ne(-100).any(dim=-1)
        exact_reference = candidate_labels.eq(reference_labels[:, None, :]).all(dim=-1)
        active = candidate_valid & reference_valid[:, None] & ~exact_reference
        active_count = active.sum().to(dtype=generated_scores.dtype)
        differentiable_zero = generated_scores.sum() * 0.0 + reference_scores.sum() * 0.0
        if not bool(active.any()):
            detached_zero = differentiable_zero.detach()
            return differentiable_zero, detached_zero, detached_zero, active_count

        score_gaps = reference_scores[:, None] - generated_scores
        selected_gaps = score_gaps[active]
        if not bool(torch.isfinite(selected_gaps).all()):
            raise FloatingPointError("reference preference score gaps must be finite")
        temperature = max(
            float(self.config.preference_temperature),
            torch.finfo(generated_scores.dtype).eps,
        )
        scaled_gaps = selected_gaps / temperature
        if not bool(torch.isfinite(scaled_gaps).all()):
            raise FloatingPointError("scaled reference preference gaps must be finite")
        loss = F.softplus(-scaled_gaps).mean()
        mean_margin = selected_gaps.detach().mean()
        violation_rate = selected_gaps.detach().le(0).to(torch.float32).mean()
        return loss, mean_margin, violation_rate, active_count

    @staticmethod
    def _repeated_model_inputs(
        batch: dict[str, torch.Tensor], repeats: int
    ) -> dict[str, torch.Tensor]:
        excluded = {
            "decoder_input_ids",
            "labels",
            "register_labels",
            "alignment_targets",
            "source_language_tag_ids",
            "reverse_direction_trained",
        }
        return {
            name: value.repeat_interleave(repeats, dim=0)
            for name, value in batch.items()
            if name not in excluded
        }

    def _sequence_log_probabilities(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean token log-probability for a small candidate chunk.

        ``decoder_input_ids`` and ``labels`` have shape
        ``(batch, candidates, length)``. Cross entropy computes only the target
        token NLL instead of materializing a second, explicit log-softmax tensor.
        Candidate activation checkpointing additionally keeps the large model and
        vocabulary activations out of the retained forward graph.
        """

        def score_chunk(
            chunk_decoder_input_ids: torch.Tensor,
            chunk_labels: torch.Tensor,
        ) -> torch.Tensor:
            # Original chunks share the trainer's outer autocast context, but
            # checkpoint recomputation creates a fresh one.  Do not let cached
            # parameter casts make those two graphs differ on PyTorch 2.8.
            with _autocast_without_weight_cache(chunk_decoder_input_ids.device.type):
                batch_size, candidates, target_length = chunk_decoder_input_ids.shape
                output = model(
                    **self._repeated_model_inputs(batch, candidates),
                    decoder_input_ids=chunk_decoder_input_ids.reshape(
                        batch_size * candidates, target_length
                    ),
                    labels=None,
                )
                flat_labels = chunk_labels.reshape(batch_size * candidates, target_length)
                token_nll = F.cross_entropy(
                    output.logits.float().transpose(1, 2),
                    flat_labels,
                    ignore_index=-100,
                    reduction="none",
                )
                valid = flat_labels.ne(-100)
                sequence_log_probs = -token_nll.masked_fill(~valid, 0.0).sum(-1) / valid.sum(
                    -1
                ).clamp_min(1)
                return sequence_log_probs.view(batch_size, candidates)

        if self.config.candidate_gradient_checkpointing and torch.is_grad_enabled():
            return _activation_checkpoint(
                score_chunk,
                decoder_input_ids,
                labels,
                use_reentrant=False,
            )
        return score_chunk(decoder_input_ids, labels)

    def _reference_cross_entropy(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        label_smoothing: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        reference_inputs = self._repeated_model_inputs(batch, 1)
        # Candidate scoring deliberately excludes target-side annotations. The
        # single reference pass can safely retain them, preserving CoRe/BATS
        # supervision alongside evidence and semantic-parity objectives.
        for name in ("register_labels", "alignment_targets"):
            if name in batch:
                reference_inputs[name] = batch[name]
        output = model(
            **reference_inputs,
            decoder_input_ids=decoder_input_ids,
            # Supplying labels lets native auxiliary heads (semantic parity and
            # uncertainty/evidence budgeting) receive their supervised signal
            # during MRT instead of becoming unused post-training parameters.
            labels=labels,
        )
        reference_tokens = labels.ne(-100).sum()
        token_nll = F.cross_entropy(
            output.logits.float().transpose(1, 2),
            labels,
            ignore_index=-100,
            reduction="none",
        )
        valid = labels.ne(-100)
        reference_scores = -token_nll.masked_fill(~valid, 0.0).sum(-1) / valid.sum(-1).clamp_min(1)
        lm_loss_sum = getattr(output, "lm_loss_sum", None)
        if lm_loss_sum is None:
            lm_loss_sum = F.cross_entropy(
                output.logits.float().reshape(-1, output.logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
                label_smoothing=label_smoothing,
            )
        auxiliary_loss = getattr(output, "auxiliary_loss", None)
        if auxiliary_loss is None:
            auxiliary_loss = output.logits.new_zeros(())
        diagnostics = {
            name: value.detach()
            for name in (
                "register_loss",
                "alignment_loss",
                "coverage_loss",
                "uncertainty_loss",
                "evidence_budget_loss",
                "evidence_request_rate",
                "evidence_repair_gain_loss",
                "evidence_repair_gain",
                "candidate_refinement_loss",
                "candidate_refinement_gain",
                "candidate_refinement_steps",
                "semantic_parity_loss",
                "semantic_parity_score",
            )
            if (value := getattr(output, name, None)) is not None
        }
        return (
            lm_loss_sum / reference_tokens.clamp_min(1),
            reference_tokens,
            auxiliary_loss,
            reference_scores,
            diagnostics,
        )

    @torch.no_grad()
    def validation_metrics(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """실제 추론과 가까운 beam 출력으로 사후학습 품질을 검증합니다."""
        base = cast(SionForConditionalGeneration, unwrap_model(model))
        generated = base.generate(
            batch["input_ids"],
            batch["attention_mask"],
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            max_new_tokens=self._max_new_tokens(base, batch["labels"]),
            num_beams=self.config.validation_num_beams,
            **self._generation_features(batch),
        )
        roundtrip_candidates, roundtrip_mask = self._backtranslate_candidates(
            base,
            batch,
            generated[:, None, :],
        )
        reward_output = self.reward_model(
            generated[:, None, :],
            batch["input_ids"],
            batch["labels"],
            roundtrip_candidates=roundtrip_candidates,
            roundtrip_mask=roundtrip_mask,
        )
        metrics = {"reward": reward_output.reward.mean()}
        metrics.update(
            {f"reward_{name}": values.mean() for name, values in reward_output.components.items()}
        )
        metrics.update(self._direction_reward_metrics(batch, reward_output.reward))
        return metrics

    def _direction_reward_metrics(
        self,
        batch: dict[str, torch.Tensor],
        rewards: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """방향별 reward 합계와 행 수. 평균은 호출자가 나눠서 만듭니다.

        평균을 여기서 내면 안 됩니다. 검증 aggregation 이 각 지표를 **배치
        크기**로 가중하므로, 한 배치에 ko→ja 가 두 행뿐이어도 그 평균이 배치
        전체 무게로 들어갑니다. 합계와 행 수를 따로 내보내면 두 값이 같은
        가중을 받아 나눌 때 정확히 상쇄됩니다.
        """

        source_tags = batch.get("source_language_tag_ids")
        if source_tags is None:
            return {}
        target_tags = batch["input_ids"][:, 0]
        rows = float(target_tags.shape[0])
        if rows <= 0:
            return {}
        per_row = rewards.reshape(rewards.shape[0], -1).mean(dim=-1).detach()
        metrics: dict[str, torch.Tensor] = {}
        identifier_to_language = {
            int(token_id): language for language, token_id in self.tokenizer.language_tags.items()
        }
        for source_id, source_language in identifier_to_language.items():
            for target_id, target_language in identifier_to_language.items():
                if source_language == target_language:
                    continue
                selected = source_tags.eq(source_id) & target_tags.eq(target_id)
                if not bool(selected.any()):
                    continue
                name = f"direction_{source_language}_to_{target_language}"
                metrics[f"{name}_reward_sum"] = per_row[selected].sum() / rows
                metrics[f"{name}_rows"] = selected.sum().to(per_row.dtype) / rows
        return metrics

    def __call__(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> ObjectiveOutput:
        base = cast(SionForConditionalGeneration, unwrap_model(model))
        reference_labels = batch["labels"]
        batch_size = reference_labels.shape[0]
        generation_features = self._generation_features(batch)
        forbidden = tuple(
            sorted(
                token_id
                for token_id in self.special_ids - {self.tokenizer.eos_id}
                if 0 <= token_id < base.config.vocab_size
            )
        )
        # Sampling and round-trip generation are no-grad operations inside the
        # trainer's autocast scope.  Prevent their detached parameter casts from
        # being reused by the subsequent gradient-bearing scoring passes.
        with _autocast_without_weight_cache(batch["input_ids"].device.type):
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
                **generation_features,
            )
            roundtrip_started = time.perf_counter()
            roundtrip_candidates, roundtrip_mask = self._backtranslate_candidates(
                base,
                batch,
                sampled,
            )
        roundtrip_generation_seconds = time.perf_counter() - roundtrip_started
        reward_transfer_started = time.perf_counter()
        reward_candidates_cpu = sampled.detach().to("cpu")
        reward_inputs_cpu = batch["input_ids"].detach().to("cpu")
        reward_references_cpu = reference_labels.detach().to("cpu")
        reward_roundtrips_cpu = (
            roundtrip_candidates.detach().to("cpu") if roundtrip_candidates is not None else None
        )
        reward_roundtrip_mask_cpu = (
            roundtrip_mask.detach().to("cpu") if roundtrip_mask is not None else None
        )
        reward_input_transfer_seconds = time.perf_counter() - reward_transfer_started

        def calculate_reward() -> tuple[RewardOutput, float, float]:
            started = time.perf_counter()
            output = self.reward_model.score_cpu(
                reward_candidates_cpu,
                reward_inputs_cpu,
                reward_references_cpu,
                roundtrip_candidates=reward_roundtrips_cpu,
                roundtrip_mask=reward_roundtrip_mask_cpu,
            )
            return output, started, time.perf_counter()

        # String decoding and chrF/structure metrics are CPU/Python-heavy. Run
        # them while the main thread submits candidate-scoring work to the GPU.
        # A per-call executor has a small cost relative to generation, and its
        # context guarantees that exceptions and cancellation never leak a
        # worker beyond this optimizer micro-step.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sion-reward") as executor:
            reward_future = executor.submit(calculate_reward)

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

            scoring_start_event = None
            scoring_end_event = None
            if sampled.device.type == "cuda":
                scoring_start_event = torch.cuda.Event(enable_timing=True)
                scoring_end_event = torch.cuda.Event(enable_timing=True)
                scoring_start_event.record()
                scoring_start_event.synchronize()
            candidate_scoring_started = time.perf_counter()
            score_chunks = [
                self._sequence_log_probabilities(
                    model,
                    batch,
                    decoder_inputs[:, start : start + self.config.candidate_micro_batch],
                    labels[:, start : start + self.config.candidate_micro_batch],
                )
                for start in range(0, samples, self.config.candidate_micro_batch)
            ]
            generated_scores = torch.cat(score_chunks, dim=1)
            (
                ce_loss,
                reference_tokens,
                reference_auxiliary_loss,
                reference_scores,
                reference_diagnostics,
            ) = self._reference_cross_entropy(
                model,
                batch,
                decoder_inputs[:, samples],
                labels[:, samples],
                label_smoothing=base.config.label_smoothing,
            )
            if scoring_end_event is not None:
                scoring_end_event.record()
                scoring_end_event.synchronize()
                assert scoring_start_event is not None
                candidate_scoring_seconds = (
                    scoring_start_event.elapsed_time(scoring_end_event) / 1_000.0
                )
                candidate_scoring_finished = candidate_scoring_started + candidate_scoring_seconds
            else:
                candidate_scoring_finished = time.perf_counter()
                candidate_scoring_seconds = candidate_scoring_finished - candidate_scoring_started
            reward_wait_started = time.perf_counter()
            reward_output_cpu, reward_cpu_started, reward_cpu_finished = reward_future.result()
            reward_wait_seconds = time.perf_counter() - reward_wait_started

        reward_cpu_seconds = max(0.0, reward_cpu_finished - reward_cpu_started)
        reward_overlap_seconds = max(
            0.0,
            min(reward_cpu_finished, candidate_scoring_finished)
            - max(reward_cpu_started, candidate_scoring_started),
        )
        reward_output = RewardOutput(
            reward=reward_output_cpu.reward.to(device=sampled.device),
            components={
                name: values.to(device=sampled.device)
                for name, values in reward_output_cpu.components.items()
            },
        )
        rewards = reward_output.reward
        candidate_distribution = torch.softmax(self.config.mrt_alpha * generated_scores, dim=-1)
        risk = (candidate_distribution * (1.0 - rewards)).sum(-1).mean()
        candidate_preference_loss, pair_weight = self._pairwise_preference_loss(
            generated_scores,
            rewards,
        )
        (
            reference_preference_loss,
            reference_preference_margin,
            reference_preference_violation_rate,
            reference_preference_comparisons,
        ) = self._reference_preference_loss(
            generated_scores,
            reference_scores,
            labels[:, :samples],
            labels[:, samples],
        )
        preference_loss = candidate_preference_loss + reference_preference_loss

        total_loss = (
            ce_loss
            + reference_auxiliary_loss
            + self.config.risk_weight * risk
            + self.config.preference_weight * preference_loss
        )
        normalizer = torch.tensor(float(batch_size), device=sampled.device)
        metrics = {
            "ce_loss": ce_loss.detach(),
            "reference_auxiliary_loss": reference_auxiliary_loss.detach(),
            "risk": risk.detach(),
            "preference_loss": preference_loss.detach(),
            "candidate_preference_loss": candidate_preference_loss.detach(),
            "reference_preference_loss": reference_preference_loss.detach(),
            "preference_pair_weight": pair_weight.detach(),
            "reference_preference_margin": reference_preference_margin,
            "reference_preference_violation_rate": reference_preference_violation_rate,
            "reference_preference_comparison_fraction": (
                reference_preference_comparisons / max(1, generated_scores.numel())
            ).detach(),
            "reference_sequence_score": reference_scores.mean().detach(),
            "reward": rewards.mean().detach(),
            "reward_cpu_seconds": torch.tensor(
                reward_cpu_seconds,
                device=sampled.device,
                dtype=torch.float32,
            ),
            "reward_wait_seconds": torch.tensor(
                reward_wait_seconds,
                device=sampled.device,
                dtype=torch.float32,
            ),
            "reward_overlap_seconds": torch.tensor(
                reward_overlap_seconds,
                device=sampled.device,
                dtype=torch.float32,
            ),
            "reward_overlap_fraction": torch.tensor(
                reward_overlap_seconds / max(reward_cpu_seconds, 1e-9),
                device=sampled.device,
                dtype=torch.float32,
            ),
            "candidate_scoring_seconds": torch.tensor(
                candidate_scoring_seconds,
                device=sampled.device,
                dtype=torch.float32,
            ),
            "reward_input_transfer_seconds": torch.tensor(
                reward_input_transfer_seconds,
                device=sampled.device,
                dtype=torch.float32,
            ),
            "roundtrip_generation_seconds": torch.tensor(
                roundtrip_generation_seconds,
                device=sampled.device,
                dtype=torch.float32,
            ),
        }
        metrics.update(
            {
                f"reward_{name}": values.mean().detach()
                for name, values in reward_output.components.items()
            }
        )
        metrics.update(reference_diagnostics)
        return ObjectiveOutput(
            loss_sum=total_loss * normalizer,
            normalizer=normalizer,
            processed_tokens=reference_tokens.detach(),
            auxiliary_loss=(
                reference_auxiliary_loss
                + self.config.risk_weight * risk
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
