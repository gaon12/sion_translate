from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from sion_translate.config import ModelConfig

from .experimental import (
    ActiveEvidenceRepair,
    BilingualAlignmentTransport,
    ContentRegisterState,
    MorphoScriptFusion,
    SemanticParityHead,
    TypedEntityMemory,
)
from .layers import DecoderLayer, EncoderLayer, GQAAttention, RMSNorm, RotaryEmbedding, SwiGLU


def _all_ranks_finished(local_finished: bool, device: torch.device) -> bool:
    """Return true only when every distributed rank can leave its decode loop."""

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return local_finished
    flag = torch.tensor(1 if local_finished else 0, dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _all_ranks_max_new_tokens(local_max_new_tokens: int, device: torch.device) -> int:
    """Use one decode-loop bound even when rank-local batches have different lengths."""

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return local_max_new_tokens
    limit = torch.tensor(local_max_new_tokens, dtype=torch.int32, device=device)
    dist.all_reduce(limit, op=dist.ReduceOp.MAX)
    return int(limit.item())


def _validate_row_generation_limits(
    limits: torch.Tensor | None,
    *,
    batch_size: int,
    max_new_tokens: int,
    min_new_tokens: int,
    device: torch.device,
) -> torch.Tensor | None:
    if limits is None:
        return None
    if not isinstance(limits, torch.Tensor):
        raise TypeError("max_new_tokens_per_row must be a tensor or None")
    if limits.ndim != 1 or limits.shape[0] != batch_size:
        raise ValueError(
            f"max_new_tokens_per_row must have shape ({batch_size},), got {tuple(limits.shape)}"
        )
    limits = limits.to(device=device, dtype=torch.long)
    if bool((limits <= min_new_tokens).any()) or bool((limits > max_new_tokens).any()):
        raise ValueError(
            "max_new_tokens_per_row values must be greater than min_new_tokens "
            "and no larger than max_new_tokens"
        )
    return limits


def _force_eos_at_row_limits(
    logits: torch.Tensor,
    limits: torch.Tensor | None,
    *,
    position: int,
    eos_id: int,
) -> torch.Tensor:
    if limits is None:
        return logits
    force_eos = limits <= position + 1
    if not bool(force_eos.any()):
        return logits
    logits = logits.masked_fill(force_eos[:, None], float("-inf"))
    logits[force_eos, eos_id] = 0.0
    return logits


@dataclass
class SionOutput:
    """forward 결과 묶음.

    - logits: 각 위치의 다음 토큰 예측 점수 (batch, seq, vocab)
    - loss: 토큰당 평균 LM loss + 보조 loss (backward 용 요약값)
    - lm_loss_sum / token_count: loss 의 분자/분모.
      trainer 가 gradient accumulation 시 '토큰 수 기준'으로 정확히
      정규화할 수 있도록 합계 형태로 따로 내보냅니다.
    - auxiliary_loss: z-loss + 실험 기능(register/alignment/evidence/parity) 보조 loss 합
    - evidence_*: 불확실성 오차 위치와 원문 재참조 budget 진단
    - semantic_parity_*: 원문/정답 표현의 의미 checksum 보조 목적과 cosine 점수
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    lm_loss_sum: torch.Tensor | None = None
    token_count: torch.Tensor | None = None
    auxiliary_loss: torch.Tensor | None = None
    register_loss: torch.Tensor | None = None
    alignment_loss: torch.Tensor | None = None
    coverage_loss: torch.Tensor | None = None
    uncertainty_loss: torch.Tensor | None = None
    evidence_budget_loss: torch.Tensor | None = None
    evidence_request_rate: torch.Tensor | None = None
    evidence_repair_gain_loss: torch.Tensor | None = None
    evidence_repair_gain: torch.Tensor | None = None
    semantic_parity_loss: torch.Tensor | None = None
    semantic_parity_score: torch.Tensor | None = None


@dataclass(frozen=True)
class GenerationContext:
    """Encoder and cross-attention state reusable across decode strategies."""

    encoder_states: torch.Tensor
    source_mask: torch.Tensor
    register_context: torch.Tensor | None
    cross_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    evidence_key_value: tuple[torch.Tensor, torch.Tensor] | None = None
    memory_token_ids: torch.Tensor | None = None
    memory_mask: torch.Tensor | None = None
    memory_type_ids: torch.Tensor | None = None
    memory_mode_ids: torch.Tensor | None = None


class SionForConditionalGeneration(nn.Module):
    """sion_translate 번역 모델 본체 (encoder-decoder Transformer).

    구성 요약:
    - 임베딩: 한/일 공용(joint) vocab 하나를 encoder·decoder·출력층이
      모두 공유(tie)해 파라미터를 절약합니다.
    - encoder: ``encoder_layers`` 층 (깊은 encoder / 얕은 decoder 구성 —
      번역 품질은 encoder 깊이의 영향이 크고, decoder 가 얕으면 추론이 빠릅니다)
    - decoder: causal self-attention + cross-attention
    - 위치 정보: RoPE (encoder/decoder self-attention 에만 적용)
    - 실험 기능(BATS/CoRe/TETM/MorphoScript)은 설정으로 켤 때만 생성됩니다.
    """

    def __init__(self, config: ModelConfig, *, pad_id: int = 0):
        super().__init__()
        config.validate()
        if config.vocab_size <= 0:
            raise ValueError("ModelConfig.vocab_size must be set before model construction")
        self.config = config
        self.pad_id = pad_id
        # FSDP2 enables this after sharding. Replicated single/DDP inference
        # keeps its low-latency local exit without a collective per token.
        self._synchronize_generation_across_ranks = False
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(config.dropout)
        head_dim = config.d_model // config.num_heads
        self.encoder_rope = RotaryEmbedding(head_dim, config.max_seq_len, config.rope_base)
        self.decoder_rope = RotaryEmbedding(head_dim, config.max_seq_len, config.rope_base)
        layer_args = dict(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            qk_norm=config.qk_norm,
            norm_eps=config.rms_norm_eps,
            ffn_gate_beta=(
                config.experimental.situglu_gate_beta
                if config.experimental.situglu_enabled
                else None
            ),
            ffn_up_beta=(
                config.experimental.situglu_up_beta if config.experimental.situglu_enabled else None
            ),
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(**layer_args, rope=self.encoder_rope)
                for _ in range(config.encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(**layer_args, rope=self.decoder_rope)
                for _ in range(config.decoder_layers)
            ]
        )
        # 공유 블록 반복 설정. 가중치를 새로 만들지 않으므로 state_dict 가 바뀌지
        # 않습니다 — 기존 체크포인트를 그대로 불러올 수 있습니다.
        self.recurrent_block_layers = min(
            config.experimental.recurrent_block_layers, config.encoder_layers
        )
        self.recurrent_steps = max(1, config.experimental.recurrent_steps)
        self.encoder_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.decoder_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = (
            None
            if config.tie_embeddings
            else nn.Linear(config.d_model, config.vocab_size, bias=False)
        )

        # Optional heads are constructed after the shared backbone. Preserve
        # the CPU RNG position across their framework-default initialization so
        # ``init_weights`` starts common parameters from the same state in a
        # same-seed baseline/ablation pair. Every optional parameter is then
        # deliberately initialized by ``init_weights`` below.
        initialization_rng_state = torch.get_rng_state()
        initialization_cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        )
        exp = config.experimental
        self.morphoscript = (
            MorphoScriptFusion(config.d_model, exp.script_classes)
            if exp.morphoscript_enabled
            else None
        )
        # MorphoScript 를 켰을 때만 게이트 파라미터를 만듭니다. 꺼져 있는데
        # 파라미터로 등록해 두면 DDP 가 find_unused_parameters=True 를
        # 요구하게 되어 매 step 불필요한 통신 비용이 생깁니다.
        self.morph_gates = (
            nn.Parameter(torch.zeros(config.encoder_layers)) if exp.morphoscript_enabled else None
        )
        self.register_state = (
            ContentRegisterState(config.d_model, exp.register_classes) if exp.core_enabled else None
        )
        self.typed_memory = (
            TypedEntityMemory(
                config.d_model,
                config.num_heads,
                config.num_kv_heads,
                exp.tetm_types,
                exp.tetm_modes,
                dropout=config.dropout,
                qk_norm=config.qk_norm,
                norm_eps=config.rms_norm_eps,
            )
            if exp.tetm_enabled
            else None
        )
        self.alignment_head = (
            BilingualAlignmentTransport(config.d_model, exp.bats_dim) if exp.bats_enabled else None
        )
        self.evidence_repair = (
            ActiveEvidenceRepair(
                config.d_model,
                config.num_heads,
                config.num_kv_heads,
                dropout=config.dropout,
                qk_norm=config.qk_norm,
                norm_eps=config.rms_norm_eps,
            )
            if exp.evidence_repair_enabled
            else None
        )
        self.semantic_parity = (
            SemanticParityHead(
                config.d_model,
                exp.semantic_parity_dim,
                exp.semantic_parity_temperature,
                norm_eps=config.rms_norm_eps,
            )
            if exp.semantic_parity_enabled
            else None
        )
        torch.set_rng_state(initialization_rng_state)
        if initialization_cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(initialization_cuda_rng_states)
        self.init_weights()

    def init_weights(self) -> None:
        """가중치 초기화. 기본은 N(0, init_std²) 정규분포이고,
        잔차 경로로 합쳐지는 출력 projection(out_proj/down_proj)은
        층 수에 비례해 더 작게 초기화해 깊은 모델의 초기 발산을 막습니다."""
        std = self.config.init_std
        residual_std = (
            std / (2 * max(self.config.encoder_layers, self.config.decoder_layers)) ** 0.5
        )

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, GQAAttention):
                nn.init.normal_(module.out_proj.weight, mean=0.0, std=residual_std)
            elif isinstance(module, SwiGLU):
                nn.init.normal_(module.down_proj.weight, mean=0.0, std=residual_std)
            elif isinstance(module, RotaryEmbedding):
                # Meta-device construction followed by ``to_empty`` leaves
                # derived, non-persistent RoPE buffers uninitialized.
                module.reset_parameters()

        self.apply(initialize)
        with torch.no_grad():
            if self.morph_gates is not None:
                self.morph_gates.zero_()
            if self.evidence_repair is not None:
                self.evidence_repair.repair_scale.zero_()
            if self.register_state is not None:
                self.register_state.inject_gate.zero_()
            if self.typed_memory is not None:
                self.typed_memory.gate.zero_()
            if self.alignment_head is not None:
                self.alignment_head.null_source.zero_()

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(token_ids)
        device_type = hidden.device.type
        if torch.is_autocast_enabled(device_type):
            # Embedding is not an autocast op, so its FP32 output would promote
            # every residual addition back to FP32 unless converted explicitly.
            hidden = hidden.to(dtype=torch.get_autocast_dtype(device_type))
        return self.embedding_dropout(hidden)

    def _checkpoint(self, layer: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        src_script_ids: torch.Tensor | None = None,
        src_onset_ids: torch.Tensor | None = None,
        src_vowel_ids: torch.Tensor | None = None,
        src_coda_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self._embed(input_ids)
        side_states = None
        if self.morphoscript is not None and src_script_ids is not None:
            zero = torch.zeros_like(src_script_ids)
            side_states = self.morphoscript(
                src_script_ids,
                src_onset_ids if src_onset_ids is not None else zero,
                src_vowel_ids if src_vowel_ids is not None else zero,
                src_coda_ids if src_coda_ids is not None else zero,
            )
        interval = max(1, self.config.experimental.morphoscript_interval)

        def run_layer(index: int, hidden: torch.Tensor) -> torch.Tensor:
            hidden = self._checkpoint(self.encoder_layers[index], hidden, attention_mask)
            if side_states is not None and (index + 1) % interval == 0:
                hidden = hidden + torch.tanh(self.morph_gates[index]) * side_states
            return hidden

        # 공유 블록 반복: 마지막 recurrent_block_layers 개 층을 **하나의 블록으로
        # 묶어** recurrent_steps 번 통과시킵니다. 층마다 따로 반복하는 것과는 다른
        # 계산입니다 — (L2, L3) 를 3번 도는 것과 L2 를 3번 돈 뒤 L3 를 3번 도는 것은
        # 서로 다른 함수이고, 여기서 의도한 것은 전자(블록 단위 재귀)입니다.
        # 같은 가중치를 재사용하므로 파라미터는 늘지 않고 유효 깊이만 늘어납니다.
        # 0 이면 블록이 비어 아래 루프가 원래대로 각 층을 한 번씩만 돕니다.
        block_size = min(self.recurrent_block_layers, len(self.encoder_layers))
        boundary = len(self.encoder_layers) - block_size
        for index in range(boundary):
            hidden = run_layer(index, hidden)
        for _ in range(self.recurrent_steps if block_size else 1):
            for index in range(boundary, len(self.encoder_layers)):
                hidden = run_layer(index, hidden)
        return self.encoder_norm(hidden)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        register_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self._embed(decoder_input_ids)
        if register_context is not None:
            hidden = hidden + register_context[:, None, :].to(dtype=hidden.dtype)
        for layer in self.decoder_layers:
            hidden = self._checkpoint(layer, hidden, encoder_states, source_mask)
        return self.decoder_norm(hidden)

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # tie_embeddings=True 면 별도 출력층 없이 입력 임베딩 행렬을
        # 그대로 출력 projection 으로 재사용합니다 (파라미터 절약 + 일반화 도움).
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return F.linear(hidden, self.token_embedding.weight)

    def _apply_typed_memory(
        self,
        decoder_states: torch.Tensor,
        *,
        memory_token_ids: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_type_ids: torch.Tensor | None = None,
        memory_mode_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self.typed_memory is None
            or memory_token_ids is None
            or memory_mask is None
            or memory_type_ids is None
            or memory_mode_ids is None
        ):
            return decoder_states
        return self.typed_memory(
            decoder_states,
            self.token_embedding,
            memory_token_ids,
            memory_type_ids,
            memory_mode_ids,
            memory_mask,
            self.pad_id,
        )

    def _apply_evidence_repair(
        self,
        decoder_states: torch.Tensor,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        evidence_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.evidence_repair is None:
            return decoder_states, None, None
        return self.evidence_repair(
            decoder_states,
            encoder_states,
            source_mask,
            evidence_key_value=evidence_key_value,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        register_labels: torch.Tensor | None = None,
        memory_token_ids: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_type_ids: torch.Tensor | None = None,
        memory_mode_ids: torch.Tensor | None = None,
        src_script_ids: torch.Tensor | None = None,
        src_onset_ids: torch.Tensor | None = None,
        src_vowel_ids: torch.Tensor | None = None,
        src_coda_ids: torch.Tensor | None = None,
        alignment_targets: torch.Tensor | None = None,
        source_language_tag_ids: torch.Tensor | None = None,
        reverse_direction_trained: torch.Tensor | None = None,
    ) -> SionOutput:
        # Collator가 제공하는 사후학습용 방향 메타데이터입니다. 일반 SFT
        # forward에서는 의미가 없지만 batch를 그대로 전달할 수 있게 받습니다.
        del source_language_tag_ids, reverse_direction_trained
        encoder_states = self.encode(
            input_ids,
            attention_mask,
            src_script_ids=src_script_ids,
            src_onset_ids=src_onset_ids,
            src_vowel_ids=src_vowel_ids,
            src_coda_ids=src_coda_ids,
        )
        register_context = None
        register_logits = None
        if self.register_state is not None:
            _, register_context, register_logits = self.register_state(
                encoder_states, attention_mask, register_labels
            )
        decoder_states = self.decode(
            decoder_input_ids,
            encoder_states,
            attention_mask,
            register_context=register_context,
        )
        decoder_states = self._apply_typed_memory(
            decoder_states,
            memory_token_ids=memory_token_ids,
            memory_mask=memory_mask,
            memory_type_ids=memory_type_ids,
            memory_mode_ids=memory_mode_ids,
        )
        pre_repair_error_targets = None
        pre_repair_token_nll = None
        if labels is not None and self.evidence_repair is not None:
            # Supervise "should I re-read?" from the base decoder, before the
            # request itself can change the answer. A post-repair target is
            # self-negating: once a useful repair fixes a token it would teach
            # the gate that the successful request should not have happened.
            with torch.no_grad():
                pre_repair_logits = self._logits(decoder_states).float()
                pre_repair_error_targets = pre_repair_logits.argmax(-1).ne(labels)
                pre_repair_token_nll = F.cross_entropy(
                    pre_repair_logits.transpose(1, 2),
                    labels,
                    ignore_index=-100,
                    reduction="none",
                )
        decoder_states, uncertainty_logits, evidence_requests = self._apply_evidence_repair(
            decoder_states,
            encoder_states,
            attention_mask,
        )
        logits = self._logits(decoder_states)

        if labels is None:
            request_rate = evidence_requests.mean() if evidence_requests is not None else None
            return SionOutput(logits=logits, evidence_request_rate=request_rate)
        # label 이 -100 인 위치(패딩)는 loss 계산에서 제외합니다.
        token_count = labels.ne(-100).sum()
        lm_loss_sum = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
            label_smoothing=self.config.label_smoothing,
        )
        auxiliary_loss = logits.new_zeros(())
        # z-loss: logsumexp(logits)² 에 작은 벌점을 줘 logit 크기가
        # 무한정 커지는 것을 막습니다 (혼합 정밀도 학습 안정화).
        if self.config.z_loss_weight > 0:
            log_normalizer = logits.float().logsumexp(-1)
            valid_targets = labels.ne(-100).to(dtype=log_normalizer.dtype)
            z_loss = (log_normalizer.square() * valid_targets).sum()
            z_loss = z_loss / valid_targets.sum().clamp_min(1.0)
            auxiliary_loss = auxiliary_loss + self.config.z_loss_weight * z_loss
        register_loss = logits.new_zeros(())
        if register_logits is not None and register_labels is not None:
            known = register_labels > 0
            safe_labels = torch.where(known, register_labels, torch.zeros_like(register_labels))
            register_losses = F.cross_entropy(
                register_logits.float(),
                safe_labels,
                reduction="none",
            )
            known_weights = known.to(dtype=register_losses.dtype)
            register_loss = (register_losses * known_weights).sum()
            register_loss = register_loss / known_weights.sum().clamp_min(1.0)
            auxiliary_loss = auxiliary_loss + (
                self.config.experimental.register_loss_weight * register_loss
            )

        alignment_loss = logits.new_zeros(())
        coverage_loss = logits.new_zeros(())
        uncertainty_loss = logits.new_zeros(())
        evidence_budget_loss = logits.new_zeros(())
        evidence_request_rate = logits.new_zeros(())
        evidence_repair_gain_loss = logits.new_zeros(())
        evidence_repair_gain = logits.new_zeros(())
        semantic_parity_loss = logits.new_zeros(())
        semantic_parity_score = logits.new_zeros(())
        exp = self.config.experimental
        if self.alignment_head is not None and (
            alignment_targets is not None
            or exp.bats_loss_weight > 0
            or exp.bats_coverage_weight > 0
        ):
            target_mask = labels.ne(-100)
            _, alignment_loss, coverage_loss = self.alignment_head(
                encoder_states,
                decoder_states,
                attention_mask,
                target_mask,
                stride=exp.bats_stride,
                max_positions=exp.bats_max_positions,
                alignment_targets=alignment_targets,
            )
            auxiliary_loss = auxiliary_loss + exp.bats_loss_weight * alignment_loss
            auxiliary_loss = auxiliary_loss + exp.bats_coverage_weight * coverage_loss

        target_mask = labels.ne(-100)
        if uncertainty_logits is not None and evidence_requests is not None:
            assert pre_repair_error_targets is not None
            assert pre_repair_token_nll is not None
            # The detached pre-repair argmax is a stable error map; the repair
            # proposal itself remains trained by the final LM loss.
            error_targets = pre_repair_error_targets.to(logits.dtype)
            valid = target_mask.to(logits.dtype)
            uncertainty_values = F.binary_cross_entropy_with_logits(
                uncertainty_logits.float(),
                error_targets.float(),
                reduction="none",
            )
            uncertainty_loss = (uncertainty_values * valid).sum() / valid.sum().clamp_min(1.0)
            evidence_request_rate = (evidence_requests * valid).sum() / valid.sum().clamp_min(1.0)
            has_target = target_mask.any().to(logits.dtype)
            evidence_budget_loss = (
                F.relu(evidence_request_rate - exp.evidence_budget_target).square() * has_target
            )
            post_repair_token_nll = F.cross_entropy(
                logits.float().transpose(1, 2),
                labels,
                ignore_index=-100,
                reduction="none",
            )
            token_gain = (pre_repair_token_nll - post_repair_token_nll).detach()
            evidence_repair_gain = (token_gain * valid).sum() / valid.sum().clamp_min(1.0)
            unproductive_request_cost = F.relu(exp.evidence_minimum_gain - token_gain)
            evidence_repair_gain_loss = (
                evidence_requests * unproductive_request_cost * valid
            ).sum() / valid.sum().clamp_min(1.0)
            auxiliary_loss = auxiliary_loss + (
                exp.evidence_uncertainty_loss_weight * uncertainty_loss
            )
            auxiliary_loss = auxiliary_loss + (
                exp.evidence_budget_loss_weight * evidence_budget_loss
            )
            auxiliary_loss = auxiliary_loss + (
                exp.evidence_repair_gain_loss_weight * evidence_repair_gain_loss
            )

        if self.semantic_parity is not None:
            semantic_parity_loss, semantic_parity_score = self.semantic_parity(
                encoder_states,
                decoder_states,
                attention_mask,
                target_mask,
            )
            auxiliary_loss = auxiliary_loss + (
                exp.semantic_parity_loss_weight * semantic_parity_loss
            )

        mean_lm_loss = lm_loss_sum / token_count.clamp_min(1)
        return SionOutput(
            logits=logits,
            loss=mean_lm_loss + auxiliary_loss,
            lm_loss_sum=lm_loss_sum,
            token_count=token_count,
            auxiliary_loss=auxiliary_loss,
            register_loss=register_loss,
            alignment_loss=alignment_loss,
            coverage_loss=coverage_loss,
            uncertainty_loss=uncertainty_loss,
            evidence_budget_loss=evidence_budget_loss,
            evidence_request_rate=evidence_request_rate,
            evidence_repair_gain_loss=evidence_repair_gain_loss,
            evidence_repair_gain=evidence_repair_gain,
            semantic_parity_loss=semantic_parity_loss,
            semantic_parity_score=semantic_parity_score,
        )

    def _decoder_step(
        self,
        tokens: torch.Tensor,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
        caches: list[dict[str, tuple[torch.Tensor, torch.Tensor] | None]],
        position: int,
        register_context: torch.Tensor | None,
        *,
        evidence_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        memory_token_ids: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_type_ids: torch.Tensor | None = None,
        memory_mode_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """KV cache 를 사용해 새 토큰 1개를 디코딩합니다 (추론 전용).

        ``caches`` 는 layer 별 {"self": (k, v), "cross": (k, v)} 목록이며
        이 함수가 제자리에서(in-place) 갱신합니다.
        """
        hidden = self._embed(tokens)
        if register_context is not None:
            hidden = hidden + register_context[:, None, :].to(dtype=hidden.dtype)
        for layer, cache in zip(self.decoder_layers, caches, strict=True):
            hidden, cache["self"], cache["cross"] = layer.forward_step(
                hidden,
                encoder_states,
                source_mask,
                self_kv=cache["self"],
                cross_kv=cache["cross"],
                position_offset=position,
            )
        hidden = self.decoder_norm(hidden)
        hidden = self._apply_typed_memory(
            hidden,
            memory_token_ids=memory_token_ids,
            memory_mask=memory_mask,
            memory_type_ids=memory_type_ids,
            memory_mode_ids=memory_mode_ids,
        )
        hidden, _, _ = self._apply_evidence_repair(
            hidden,
            encoder_states,
            source_mask,
            evidence_key_value=evidence_key_value,
        )
        return hidden

    @staticmethod
    def _fresh_caches(
        layer_count: int,
        cross_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        *,
        repeats: int = 1,
    ) -> list[dict[str, Any]]:
        if cross_key_values is None:
            return [{"self": None, "cross": None} for _ in range(layer_count)]
        if len(cross_key_values) != layer_count:
            raise ValueError("cross_key_values must have one entry per decoder layer")
        return [
            {
                "self": None,
                "cross": (
                    tuple(value.repeat_interleave(repeats, dim=0) for value in key_value)
                    if repeats > 1
                    else key_value
                ),
            }
            for key_value in cross_key_values
        ]

    @torch.no_grad()
    def prepare_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        memory_token_ids: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_type_ids: torch.Tensor | None = None,
        memory_mode_ids: torch.Tensor | None = None,
        **encoder_features: torch.Tensor,
    ) -> GenerationContext:
        """Encode once and pre-project decoder cross-attention key/value states."""
        was_training = self.training
        self.eval()
        try:
            encoder_states = self.encode(input_ids, attention_mask, **encoder_features)
            register_context = None
            if self.register_state is not None:
                _, register_context, _ = self.register_state(
                    encoder_states,
                    attention_mask,
                    register_labels=None,
                )
            cross_key_values = tuple(
                layer.project_cross_key_value(encoder_states) for layer in self.decoder_layers
            )
            evidence_key_value = (
                self.evidence_repair.project_key_value(encoder_states)
                if self.evidence_repair is not None
                else None
            )
            return GenerationContext(
                encoder_states=encoder_states,
                source_mask=attention_mask,
                register_context=register_context,
                cross_key_values=cross_key_values,
                evidence_key_value=evidence_key_value,
                memory_token_ids=memory_token_ids,
                memory_mask=memory_mask,
                memory_type_ids=memory_type_ids,
                memory_mode_ids=memory_mode_ids,
            )
        finally:
            self.train(was_training)

    @staticmethod
    def _validate_generation_context(
        context: GenerationContext,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        if context.encoder_states.shape[0] != input_ids.shape[0]:
            raise ValueError("generation_context batch size does not match input_ids")
        if context.source_mask.shape != attention_mask.shape:
            raise ValueError("generation_context source mask does not match attention_mask")
        if context.encoder_states.device != input_ids.device:
            raise ValueError("generation_context and input_ids must be on the same device")

    @staticmethod
    def _apply_decode_constraints(
        logits: torch.Tensor,
        sequences: torch.Tensor,
        *,
        eos_id: int,
        position: int,
        min_new_tokens: int,
        forbidden_token_ids: tuple[int, ...],
        no_repeat_ngram_size: int,
    ) -> torch.Tensor:
        """Mask invalid next tokens without moving beam state to Python."""
        if not forbidden_token_ids and min_new_tokens == 0 and no_repeat_ngram_size == 0:
            return logits
        if forbidden_token_ids:
            valid_forbidden = [
                token_id for token_id in forbidden_token_ids if 0 <= token_id < logits.shape[-1]
            ]
            if valid_forbidden:
                logits[:, valid_forbidden] = float("-inf")
        if position < min_new_tokens and 0 <= eos_id < logits.shape[-1]:
            logits[:, eos_id] = float("-inf")

        ngram_size = no_repeat_ngram_size
        if ngram_size == 1 and sequences.shape[1] > 0:
            logits.scatter_(1, sequences, float("-inf"))
        elif ngram_size > 1 and sequences.shape[1] >= ngram_size:
            prefix_size = ngram_size - 1
            prefix = sequences[:, -prefix_size:]
            previous_prefixes = sequences[:, :-1].unfold(1, prefix_size, 1)
            matches = previous_prefixes.eq(prefix[:, None, :]).all(dim=-1)
            previous_next_tokens = sequences[:, prefix_size:]
            rows, columns = matches.nonzero(as_tuple=True)
            logits[rows, previous_next_tokens[rows, columns]] = float("-inf")

        # 극단적으로 작은 vocab이나 과도한 사용자 제약에서도 multinomial/
        # argmax가 NaN으로 무너지지 않게 EOS를 최후의 탈출구로 둡니다.
        no_valid_token = ~torch.isfinite(logits).any(dim=-1)
        logits[no_valid_token, eos_id] = 0.0
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        memory_token_ids: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_type_ids: torch.Tensor | None = None,
        memory_mode_ids: torch.Tensor | None = None,
        generation_context: GenerationContext | None = None,
        forbidden_token_ids: tuple[int, ...] = (),
        min_new_tokens: int = 0,
        no_repeat_ngram_size: int = 0,
        max_new_tokens_per_row: torch.Tensor | None = None,
        **encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """번역문 생성.

        - ``num_beams=1``: greedy 디코딩 (가장 빠름)
        - ``num_beams>=2``: beam search + GNMT length penalty.
          번역 품질 평가나 실사용에는 beam 4 + length_penalty 1.0 을 권장합니다.

        두 경로 모두 KV cache 를 사용해 토큰당 비용이 문장 길이에 선형입니다
        (이전 구현은 매 토큰마다 prefix 전체를 다시 계산했습니다).
        """
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise TypeError("max_new_tokens must be an integer")
        if not 1 <= max_new_tokens <= self.config.max_seq_len:
            raise ValueError(
                "max_new_tokens must be between 1 and "
                f"model max_seq_len ({self.config.max_seq_len})"
            )
        if min_new_tokens < 0 or min_new_tokens >= max_new_tokens:
            raise ValueError("min_new_tokens must be in [0, max_new_tokens)")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative")
        synchronize_ranks = self._synchronize_generation_across_ranks
        if synchronize_ranks:
            max_new_tokens = _all_ranks_max_new_tokens(max_new_tokens, input_ids.device)
        max_new_tokens_per_row = _validate_row_generation_limits(
            max_new_tokens_per_row,
            batch_size=input_ids.shape[0],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            device=input_ids.device,
        )
        was_training = self.training
        self.eval()
        try:
            if generation_context is None:
                generation_context = self.prepare_generation(
                    input_ids,
                    attention_mask,
                    memory_token_ids=memory_token_ids,
                    memory_mask=memory_mask,
                    memory_type_ids=memory_type_ids,
                    memory_mode_ids=memory_mode_ids,
                    **encoder_features,
                )
            else:
                self._validate_generation_context(
                    generation_context,
                    input_ids,
                    attention_mask,
                )
            encoder_states = generation_context.encoder_states
            source_mask = generation_context.source_mask
            register_context = generation_context.register_context
            if num_beams <= 1:
                return self._greedy_decode(
                    encoder_states,
                    source_mask,
                    register_context,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    max_new_tokens=max_new_tokens,
                    cross_key_values=generation_context.cross_key_values,
                    evidence_key_value=generation_context.evidence_key_value,
                    forbidden_token_ids=forbidden_token_ids,
                    min_new_tokens=min_new_tokens,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    max_new_tokens_per_row=max_new_tokens_per_row,
                    memory_token_ids=generation_context.memory_token_ids,
                    memory_mask=generation_context.memory_mask,
                    memory_type_ids=generation_context.memory_type_ids,
                    memory_mode_ids=generation_context.memory_mode_ids,
                )
            return self._beam_decode(
                encoder_states,
                source_mask,
                register_context,
                bos_id=bos_id,
                eos_id=eos_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                cross_key_values=generation_context.cross_key_values,
                evidence_key_value=generation_context.evidence_key_value,
                forbidden_token_ids=forbidden_token_ids,
                min_new_tokens=min_new_tokens,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens_per_row=max_new_tokens_per_row,
                memory_token_ids=generation_context.memory_token_ids,
                memory_mask=generation_context.memory_mask,
                memory_type_ids=generation_context.memory_type_ids,
                memory_mode_ids=generation_context.memory_mode_ids,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        num_samples: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        forbidden_token_ids: tuple[int, ...] = (),
        generator: torch.Generator | None = None,
        memory_token_ids: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_type_ids: torch.Tensor | None = None,
        memory_mode_ids: torch.Tensor | None = None,
        generation_context: GenerationContext | None = None,
        min_new_tokens: int = 0,
        no_repeat_ngram_size: int = 0,
        max_new_tokens_per_row: torch.Tensor | None = None,
        **encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """MRT용 확률적 후보를 ``(batch, samples, length)``로 생성합니다."""
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise TypeError("max_new_tokens must be an integer")
        if not 1 <= max_new_tokens <= self.config.max_seq_len:
            raise ValueError(
                "max_new_tokens must be between 1 and "
                f"model max_seq_len ({self.config.max_seq_len})"
            )
        synchronize_ranks = self._synchronize_generation_across_ranks
        if synchronize_ranks:
            max_new_tokens = _all_ranks_max_new_tokens(max_new_tokens, input_ids.device)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if min_new_tokens < 0 or min_new_tokens >= max_new_tokens:
            raise ValueError("min_new_tokens must be in [0, max_new_tokens)")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative")
        max_new_tokens_per_row = _validate_row_generation_limits(
            max_new_tokens_per_row,
            batch_size=input_ids.shape[0],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            device=input_ids.device,
        )
        was_training = self.training
        self.eval()
        try:
            if generation_context is None:
                generation_context = self.prepare_generation(
                    input_ids,
                    attention_mask,
                    memory_token_ids=memory_token_ids,
                    memory_mask=memory_mask,
                    memory_type_ids=memory_type_ids,
                    memory_mode_ids=memory_mode_ids,
                    **encoder_features,
                )
            else:
                self._validate_generation_context(
                    generation_context,
                    input_ids,
                    attention_mask,
                )
            encoder_states = generation_context.encoder_states
            source_mask = generation_context.source_mask
            register_context = generation_context.register_context
            evidence_key_value = generation_context.evidence_key_value
            memory_token_ids = generation_context.memory_token_ids
            memory_mask = generation_context.memory_mask
            memory_type_ids = generation_context.memory_type_ids
            memory_mode_ids = generation_context.memory_mode_ids
            encoder_states = encoder_states.repeat_interleave(num_samples, dim=0)
            source_mask = source_mask.repeat_interleave(num_samples, dim=0)
            if register_context is not None:
                register_context = register_context.repeat_interleave(num_samples, dim=0)
            if evidence_key_value is not None:
                evidence_key_value = tuple(
                    value.repeat_interleave(num_samples, dim=0) for value in evidence_key_value
                )
            if memory_token_ids is not None:
                memory_token_ids = memory_token_ids.repeat_interleave(num_samples, dim=0)
            if memory_mask is not None:
                memory_mask = memory_mask.repeat_interleave(num_samples, dim=0)
            if memory_type_ids is not None:
                memory_type_ids = memory_type_ids.repeat_interleave(num_samples, dim=0)
            if memory_mode_ids is not None:
                memory_mode_ids = memory_mode_ids.repeat_interleave(num_samples, dim=0)

            total = input_ids.shape[0] * num_samples
            if max_new_tokens_per_row is not None:
                max_new_tokens_per_row = max_new_tokens_per_row.repeat_interleave(
                    num_samples,
                    dim=0,
                )
            caches = self._fresh_caches(
                len(self.decoder_layers),
                generation_context.cross_key_values,
                repeats=num_samples,
            )
            current = torch.full((total, 1), bos_id, dtype=torch.long, device=input_ids.device)
            sequences = current
            finished = torch.zeros(total, dtype=torch.bool, device=input_ids.device)
            for position in range(max_new_tokens):
                hidden = self._decoder_step(
                    current,
                    encoder_states,
                    source_mask,
                    caches,
                    position,
                    register_context,
                    evidence_key_value=evidence_key_value,
                    memory_token_ids=memory_token_ids,
                    memory_mask=memory_mask,
                    memory_type_ids=memory_type_ids,
                    memory_mode_ids=memory_mode_ids,
                )
                logits = self._logits(hidden[:, -1]).float() / temperature
                logits = self._apply_decode_constraints(
                    logits,
                    sequences,
                    eos_id=eos_id,
                    position=position,
                    min_new_tokens=min_new_tokens,
                    forbidden_token_ids=forbidden_token_ids,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
                logits = _force_eos_at_row_limits(
                    logits,
                    max_new_tokens_per_row,
                    position=position,
                    eos_id=eos_id,
                )
                if 0 < top_k < logits.shape[-1]:
                    threshold = logits.topk(top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1),
                    1,
                    generator=generator,
                )
                next_token = torch.where(finished[:, None], eos_id, next_token)
                sequences = torch.cat((sequences, next_token), dim=1)
                finished |= next_token.squeeze(1).eq(eos_id)
                local_finished = bool(finished.all())
                if (
                    _all_ranks_finished(local_finished, input_ids.device)
                    if synchronize_ranks
                    else local_finished
                ):
                    break
                current = next_token
            return sequences.view(input_ids.shape[0], num_samples, -1)
        finally:
            self.train(was_training)

    def _greedy_decode(
        self,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
        register_context: torch.Tensor | None,
        *,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
        cross_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        evidence_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        forbidden_token_ids: tuple[int, ...] = (),
        min_new_tokens: int = 0,
        no_repeat_ngram_size: int = 0,
        max_new_tokens_per_row: torch.Tensor | None = None,
        memory_token_ids: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_type_ids: torch.Tensor | None = None,
        memory_mode_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = encoder_states.shape[0]
        device = encoder_states.device
        caches = self._fresh_caches(
            len(self.decoder_layers),
            cross_key_values,
        )
        current = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
        sequences = current
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for position in range(max_new_tokens):
            hidden = self._decoder_step(
                current,
                encoder_states,
                source_mask,
                caches,
                position,
                register_context,
                evidence_key_value=evidence_key_value,
                memory_token_ids=memory_token_ids,
                memory_mask=memory_mask,
                memory_type_ids=memory_type_ids,
                memory_mode_ids=memory_mode_ids,
            )
            logits = self._apply_decode_constraints(
                self._logits(hidden[:, -1]).float(),
                sequences,
                eos_id=eos_id,
                position=position,
                min_new_tokens=min_new_tokens,
                forbidden_token_ids=forbidden_token_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            logits = _force_eos_at_row_limits(
                logits,
                max_new_tokens_per_row,
                position=position,
                eos_id=eos_id,
            )
            next_token = logits.argmax(-1, keepdim=True)
            # 이미 끝난 문장은 EOS 를 반복해 길이만 맞춥니다.
            next_token = torch.where(finished[:, None], eos_id, next_token)
            sequences = torch.cat((sequences, next_token), dim=1)
            finished |= next_token.squeeze(1).eq(eos_id)
            local_finished = bool(finished.all())
            if (
                _all_ranks_finished(local_finished, device)
                if self._synchronize_generation_across_ranks
                else local_finished
            ):
                break
            current = next_token
        return sequences

    def _beam_decode(
        self,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
        register_context: torch.Tensor | None,
        *,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
        num_beams: int,
        length_penalty: float,
        cross_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        evidence_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        forbidden_token_ids: tuple[int, ...] = (),
        min_new_tokens: int = 0,
        no_repeat_ngram_size: int = 0,
        max_new_tokens_per_row: torch.Tensor | None = None,
        memory_token_ids: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_type_ids: torch.Tensor | None = None,
        memory_mode_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """배치 beam search.

        각 문장을 ``num_beams`` 개로 복제해 (batch × beams) 를 하나의 큰
        배치처럼 디코딩하고, 매 스텝 상위 2×beams 후보 중에서
        - EOS 로 끝난 후보는 완성 가설로 저장하고 (GNMT length penalty 적용)
        - 나머지 중 상위 beams 개를 살아있는 beam 으로 유지합니다.
        """
        batch = encoder_states.shape[0]
        device = encoder_states.device
        total = batch * num_beams

        # 문장마다 beam 수만큼 encoder 출력을 복제합니다.
        encoder_states = encoder_states.repeat_interleave(num_beams, dim=0)
        source_mask = source_mask.repeat_interleave(num_beams, dim=0)
        if register_context is not None:
            register_context = register_context.repeat_interleave(num_beams, dim=0)
        if evidence_key_value is not None:
            evidence_key_value = tuple(
                value.repeat_interleave(num_beams, dim=0) for value in evidence_key_value
            )
        if memory_token_ids is not None:
            memory_token_ids = memory_token_ids.repeat_interleave(num_beams, dim=0)
        if memory_mask is not None:
            memory_mask = memory_mask.repeat_interleave(num_beams, dim=0)
        if memory_type_ids is not None:
            memory_type_ids = memory_type_ids.repeat_interleave(num_beams, dim=0)
        if memory_mode_ids is not None:
            memory_mode_ids = memory_mode_ids.repeat_interleave(num_beams, dim=0)
        beam_row_limits = (
            max_new_tokens_per_row.repeat_interleave(num_beams, dim=0)
            if max_new_tokens_per_row is not None
            else None
        )

        caches = self._fresh_caches(
            len(self.decoder_layers),
            cross_key_values,
            repeats=num_beams,
        )
        sequences = torch.full((total, 1), bos_id, dtype=torch.long, device=device)
        # 첫 스텝에서 모든 beam 이 같은 BOS 에서 출발하므로, beam 0 만 점수 0 으로
        # 두고 나머지는 -inf 로 시작해 중복 후보를 걸러냅니다.
        beam_scores = torch.full((batch, num_beams), float("-inf"), device=device)
        beam_scores[:, 0] = 0.0
        # 완성된 가설: batch 별 (length-penalty 적용 점수, 토큰 텐서) 목록
        done: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(batch)]
        maximum_completion_lengths = (
            max_new_tokens_per_row.tolist()
            if max_new_tokens_per_row is not None
            else [max_new_tokens] * batch
        )

        def penalized(raw_score: float, length: int) -> float:
            # GNMT length penalty: 길이가 짧은 번역이 유리해지는 것을 보정합니다.
            return raw_score / (((5.0 + length) / 6.0) ** length_penalty)

        for position in range(max_new_tokens):
            hidden = self._decoder_step(
                sequences[:, -1:],
                encoder_states,
                source_mask,
                caches,
                position,
                register_context,
                evidence_key_value=evidence_key_value,
                memory_token_ids=memory_token_ids,
                memory_mask=memory_mask,
                memory_type_ids=memory_type_ids,
                memory_mode_ids=memory_mode_ids,
            )
            logits = self._apply_decode_constraints(
                self._logits(hidden[:, -1]).float(),
                sequences,
                eos_id=eos_id,
                position=position,
                min_new_tokens=min_new_tokens,
                forbidden_token_ids=forbidden_token_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            logits = _force_eos_at_row_limits(
                logits,
                beam_row_limits,
                position=position,
                eos_id=eos_id,
            )
            log_probs = F.log_softmax(logits, dim=-1)
            vocab = log_probs.shape[-1]
            # 누적 점수 = 지금까지의 beam 점수 + 새 토큰 log 확률
            candidate_scores = (beam_scores.view(-1, 1) + log_probs).view(batch, num_beams * vocab)
            # EOS 로 빠지는 후보가 있어도 살아있는 beam 을 채울 수 있도록 2배수 선택
            top_scores, top_indices = candidate_scores.topk(2 * num_beams, dim=-1)
            source_beams = top_indices // vocab  # 어느 beam 에서 나온 후보인지
            new_tokens = top_indices % vocab

            # 후보 선별을 텐서 연산으로 처리합니다. 이전 구현은 step 마다
            # batch × 2·beams × 3 번 스칼라를 host 로 읽어 왔고, GPU 에서는
            # 그 하나하나가 device 동기화입니다.
            #
            # topk 결과는 점수 내림차순이므로 "살아있는 후보 중 앞쪽 num_beams
            # 개"가 곧 "점수가 가장 높은 num_beams 개"입니다. cumsum 으로 각
            # 후보의 생존 순위를 구해 그 순위를 목적지 슬롯으로 씁니다.
            finite = top_scores.ne(float("-inf"))
            ends_here = new_tokens.eq(eos_id)
            alive = finite & ~ends_here
            alive_rank = alive.cumsum(dim=1) - 1
            keep = alive & alive_rank.lt(num_beams)
            flat_sources = (
                torch.arange(batch, device=device).unsqueeze(1) * num_beams + source_beams
            )
            # 채택되지 않은 후보는 여분의 마지막 열로 흘려보내고 잘라 버립니다.
            # scatter 의 목적지가 겹치지 않아야 하는데, 채택된 후보의 순위는
            # 서로 다르므로 충돌이 없습니다.
            spill = torch.where(keep, alive_rank, torch.full_like(alive_rank, num_beams))
            width = num_beams + 1
            score_buffer = torch.full((batch, width), float("-inf"), device=device)
            source_buffer = torch.zeros((batch, width), dtype=torch.long, device=device)
            token_buffer = torch.full((batch, width), eos_id, dtype=torch.long, device=device)
            score_buffer.scatter_(1, spill, top_scores)
            source_buffer.scatter_(1, spill, flat_sources)
            token_buffer.scatter_(1, spill, new_tokens)
            # (batch, num_beams) each, so making them contiguous costs the same
            # allocation the previous implementation did up front, and keeps
            # later view() calls valid.
            next_scores = score_buffer[:, :num_beams].contiguous()
            gather_flat = source_buffer[:, :num_beams].contiguous()
            step_tokens = token_buffer[:, :num_beams].contiguous()

            # 완성 가설만 host 로 내립니다. 보통 step 당 0~2개이므로 한 번의
            # 동기화로 끝나고, nonzero 는 row-major 라 (batch, 후보) 순서가
            # 기존 이중 루프와 같습니다 — max() 의 동점 처리 순서가 유지됩니다.
            finished = (finite & ends_here).nonzero(as_tuple=False)
            if finished.numel():
                finished_rows = finished[:, 0]
                finished_scores = top_scores[finished_rows, finished[:, 1]].tolist()
                finished_sources = flat_sources[finished_rows, finished[:, 1]].tolist()
                eos_column = torch.tensor([eos_id], device=device)
                for row, score, source in zip(
                    finished_rows.tolist(),
                    finished_scores,
                    finished_sources,
                    strict=True,
                ):
                    # 완성 가설 (BOS 제외한 생성 길이 = position+1)
                    done[row].append(
                        (
                            penalized(score, position + 1),
                            torch.cat((sequences[source], eos_column)),
                        )
                    )

            flat_index = gather_flat.reshape(-1)
            # 살아남은 beam 의 순서에 맞게 문장 기록과 self KV cache 를 재배열합니다.
            # cross KV 는 encoder 출력에서 나온 것이라 같은 문장의 beam 끼리
            # 내용이 완전히 같습니다. beam 순서가 바뀌어도 값이 그대로이므로
            # 재배열하지 않습니다 (step 마다 수 MiB 를 복사하던 낭비였습니다).
            sequences = torch.cat(
                (sequences.index_select(0, flat_index), step_tokens.reshape(-1, 1)), dim=1
            )
            for cache in caches:
                cache["self"] = tuple(t.index_select(0, flat_index) for t in cache["self"])
            beam_scores = next_scores

            # 모든 문장이 '완성 가설이 충분하고, 살아있는 beam 이 더 나은 점수를
            # 낼 가능성이 없는' 상태면 일찍 종료합니다.
            all_done = all(len(hypotheses) >= num_beams for hypotheses in done)
            if all_done:
                # One reduction and one host transfer instead of one per row.
                best_alive = beam_scores.amax(dim=1).tolist()
                for row, hypotheses in enumerate(done):
                    # Log-probability can never increase. With a positive
                    # length penalty, however, dividing that negative score by
                    # the larger maximum-length penalty can improve its final
                    # normalized value. Use that optimistic reachable length
                    # so early stopping never discards a future winner.
                    optimistic_length = (
                        maximum_completion_lengths[row] if length_penalty > 0 else position + 1
                    )
                    best_possible = penalized(best_alive[row], optimistic_length)
                    if best_possible > min(score for score, _ in hypotheses):
                        all_done = False
                        break
            if (
                _all_ranks_finished(all_done, device)
                if self._synchronize_generation_across_ranks
                else all_done
            ):
                break

        # 길이 제한에 걸린 live beam도 이미 끝난 가설과 함께 비교합니다.
        # 낮은 확률의 EOS가 일찍 한 번 나왔다는 이유만으로 더 높은 점수의
        # 진행 중 번역을 버리면 max_new_tokens 경계에서 품질이 역전됩니다.
        outputs: list[torch.Tensor] = []
        generated_length = max(1, sequences.shape[1] - 1)
        for b in range(batch):
            hypotheses = list(done[b])
            for beam_index in range(num_beams):
                raw_score = float(beam_scores[b, beam_index])
                if raw_score == float("-inf"):
                    continue
                sequence = sequences[b * num_beams + beam_index]
                hypotheses.append(
                    (
                        penalized(raw_score, generated_length),
                        torch.cat((sequence, torch.tensor([eos_id], device=device))),
                    )
                )
            if not hypotheses:
                raise RuntimeError("beam search did not produce a finite hypothesis")
            outputs.append(max(hypotheses, key=lambda item: item[0])[1])
        longest = max(len(seq) for seq in outputs)
        padded = torch.full((batch, longest), eos_id, dtype=torch.long, device=device)
        for b, seq in enumerate(outputs):
            padded[b, : len(seq)] = seq
        return padded

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
