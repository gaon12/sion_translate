from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from sion_translate.config import ModelConfig

from .experimental import (
    BilingualAlignmentTransport,
    ContentRegisterState,
    MorphoScriptFusion,
    TypedEntityMemory,
)
from .layers import DecoderLayer, EncoderLayer, GQAAttention, RMSNorm, RotaryEmbedding, SwiGLU


@dataclass
class SionOutput:
    """forward 결과 묶음.

    - logits: 각 위치의 다음 토큰 예측 점수 (batch, seq, vocab)
    - loss: 토큰당 평균 LM loss + 보조 loss (backward 용 요약값)
    - lm_loss_sum / token_count: loss 의 분자/분모.
      trainer 가 gradient accumulation 시 '토큰 수 기준'으로 정확히
      정규화할 수 있도록 합계 형태로 따로 내보냅니다.
    - auxiliary_loss: z-loss + 실험 기능(register/alignment 등) 보조 loss 합
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    lm_loss_sum: torch.Tensor | None = None
    token_count: torch.Tensor | None = None
    auxiliary_loss: torch.Tensor | None = None
    register_loss: torch.Tensor | None = None
    alignment_loss: torch.Tensor | None = None
    coverage_loss: torch.Tensor | None = None


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
        )
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(**layer_args, rope=self.encoder_rope) for _ in range(config.encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(**layer_args, rope=self.decoder_rope) for _ in range(config.decoder_layers)]
        )
        # 공유 블록 반복 설정. 가중치를 새로 만들지 않으므로 state_dict 가 바뀌지
        # 않습니다 — 기존 체크포인트를 그대로 불러올 수 있습니다.
        self.recurrent_block_layers = min(
            config.experimental.recurrent_block_layers, config.encoder_layers
        )
        self.recurrent_steps = max(1, config.experimental.recurrent_steps)
        self.encoder_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.decoder_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = None if config.tie_embeddings else nn.Linear(
            config.d_model, config.vocab_size, bias=False
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
            nn.Parameter(torch.zeros(config.encoder_layers))
            if exp.morphoscript_enabled
            else None
        )
        self.register_state = (
            ContentRegisterState(config.d_model, exp.register_classes)
            if exp.core_enabled
            else None
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
            BilingualAlignmentTransport(config.d_model, exp.bats_dim)
            if exp.bats_enabled
            else None
        )
        self.init_weights()

    def init_weights(self) -> None:
        """가중치 초기화. 기본은 N(0, init_std²) 정규분포이고,
        잔차 경로로 합쳐지는 출력 projection(out_proj/down_proj)은
        층 수에 비례해 더 작게 초기화해 깊은 모델의 초기 발산을 막습니다."""
        std = self.config.init_std

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

        self.apply(initialize)
        residual_std = std / (2 * max(self.config.encoder_layers, self.config.decoder_layers)) ** 0.5
        for module in self.modules():
            if isinstance(module, GQAAttention):
                nn.init.normal_(module.out_proj.weight, mean=0.0, std=residual_std)
            elif isinstance(module, SwiGLU):
                nn.init.normal_(module.down_proj.weight, mean=0.0, std=residual_std)
        if self.morph_gates is not None:
            with torch.no_grad():
                self.morph_gates.zero_()

    def _checkpoint(self, layer: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training:
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
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
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
        # 공유 블록 반복: 마지막 recurrent_block_layers 개 층을 recurrent_steps 번
        # 통과시킵니다. 같은 가중치를 재사용하므로 파라미터는 늘지 않고 유효 깊이만
        # 늘어납니다. 0 이면 아래 루프가 원래대로 각 층을 한 번씩만 돕니다.
        block_size = min(self.recurrent_block_layers, len(self.encoder_layers))
        boundary = len(self.encoder_layers) - block_size
        for index, layer in enumerate(self.encoder_layers):
            repeats = self.recurrent_steps if index >= boundary and block_size else 1
            for _ in range(repeats):
                hidden = self._checkpoint(layer, hidden, attention_mask)
            if side_states is not None and (index + 1) % interval == 0:
                hidden = hidden + torch.tanh(self.morph_gates[index]) * side_states
        return self.encoder_norm(hidden)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        register_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embedding_dropout(self.token_embedding(decoder_input_ids))
        if register_context is not None:
            hidden = hidden + register_context[:, None, :]
        for layer in self.decoder_layers:
            hidden = self._checkpoint(layer, hidden, encoder_states, source_mask)
        return self.decoder_norm(hidden)

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # tie_embeddings=True 면 별도 출력층 없이 입력 임베딩 행렬을
        # 그대로 출력 projection 으로 재사용합니다 (파라미터 절약 + 일반화 도움).
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return F.linear(hidden, self.token_embedding.weight)

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
    ) -> SionOutput:
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
        if (
            self.typed_memory is not None
            and memory_token_ids is not None
            and memory_mask is not None
            and memory_type_ids is not None
            and memory_mode_ids is not None
        ):
            decoder_states = self.typed_memory(
                decoder_states,
                self.token_embedding,
                memory_token_ids,
                memory_type_ids,
                memory_mode_ids,
                memory_mask,
                self.pad_id,
            )
        logits = self._logits(decoder_states)

        if labels is None:
            return SionOutput(logits=logits)
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
            valid_logits = logits.float().logsumexp(-1)[labels.ne(-100)]
            if valid_logits.numel() > 0:
                auxiliary_loss = auxiliary_loss + self.config.z_loss_weight * valid_logits.square().mean()
        register_loss = logits.new_zeros(())
        if register_logits is not None and register_labels is not None:
            known = register_labels > 0
            if known.any():
                register_loss = F.cross_entropy(register_logits[known].float(), register_labels[known])
                auxiliary_loss = auxiliary_loss + (
                    self.config.experimental.register_loss_weight * register_loss
                )

        alignment_loss = logits.new_zeros(())
        coverage_loss = logits.new_zeros(())
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
        )

    def _decoder_step(
        self,
        tokens: torch.Tensor,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
        caches: list[dict[str, tuple[torch.Tensor, torch.Tensor] | None]],
        position: int,
        register_context: torch.Tensor | None,
    ) -> torch.Tensor:
        """KV cache 를 사용해 새 토큰 1개를 디코딩합니다 (추론 전용).

        ``caches`` 는 layer 별 {"self": (k, v), "cross": (k, v)} 목록이며
        이 함수가 제자리에서(in-place) 갱신합니다.
        """
        hidden = self.embedding_dropout(self.token_embedding(tokens))
        if register_context is not None:
            hidden = hidden + register_context[:, None, :]
        for layer, cache in zip(self.decoder_layers, caches, strict=True):
            hidden, cache["self"], cache["cross"] = layer.forward_step(
                hidden,
                encoder_states,
                source_mask,
                self_kv=cache["self"],
                cross_kv=cache["cross"],
                position_offset=position,
            )
        return self.decoder_norm(hidden)

    @staticmethod
    def _fresh_caches(layer_count: int) -> list[dict[str, Any]]:
        return [{"self": None, "cross": None} for _ in range(layer_count)]

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
        **encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """번역문 생성.

        - ``num_beams=1``: greedy 디코딩 (가장 빠름)
        - ``num_beams>=2``: beam search + GNMT length penalty.
          번역 품질 평가나 실사용에는 beam 4 + length_penalty 1.0 을 권장합니다.

        두 경로 모두 KV cache 를 사용해 토큰당 비용이 문장 길이에 선형입니다
        (이전 구현은 매 토큰마다 prefix 전체를 다시 계산했습니다).
        """
        was_training = self.training
        self.eval()
        try:
            encoder_states = self.encode(input_ids, attention_mask, **encoder_features)
            register_context = None
            if self.register_state is not None:
                _, register_context, _ = self.register_state(
                    encoder_states, attention_mask, register_labels=None
                )
            if num_beams <= 1:
                return self._greedy_decode(
                    encoder_states,
                    attention_mask,
                    register_context,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    max_new_tokens=max_new_tokens,
                )
            return self._beam_decode(
                encoder_states,
                attention_mask,
                register_context,
                bos_id=bos_id,
                eos_id=eos_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
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
        **encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """MRT용 확률적 후보를 ``(batch, samples, length)``로 생성합니다."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        was_training = self.training
        self.eval()
        try:
            encoder_states = self.encode(input_ids, attention_mask, **encoder_features)
            register_context = None
            if self.register_state is not None:
                _, register_context, _ = self.register_state(
                    encoder_states, attention_mask, register_labels=None
                )
            encoder_states = encoder_states.repeat_interleave(num_samples, dim=0)
            source_mask = attention_mask.repeat_interleave(num_samples, dim=0)
            if register_context is not None:
                register_context = register_context.repeat_interleave(num_samples, dim=0)

            total = input_ids.shape[0] * num_samples
            caches = self._fresh_caches(len(self.decoder_layers))
            current = torch.full((total, 1), bos_id, dtype=torch.long, device=input_ids.device)
            pieces = [current]
            finished = torch.zeros(total, dtype=torch.bool, device=input_ids.device)
            for position in range(max_new_tokens):
                hidden = self._decoder_step(
                    current, encoder_states, source_mask, caches, position, register_context
                )
                logits = self._logits(hidden[:, -1]).float() / temperature
                if forbidden_token_ids:
                    logits[:, list(forbidden_token_ids)] = float("-inf")
                if 0 < top_k < logits.shape[-1]:
                    threshold = logits.topk(top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                next_token = torch.where(finished[:, None], eos_id, next_token)
                pieces.append(next_token)
                finished |= next_token.squeeze(1).eq(eos_id)
                if finished.all():
                    break
                current = next_token
            sequences = torch.cat(pieces, dim=1)
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
    ) -> torch.Tensor:
        batch = encoder_states.shape[0]
        device = encoder_states.device
        caches = self._fresh_caches(len(self.decoder_layers))
        current = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
        pieces = [current]
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for position in range(max_new_tokens):
            hidden = self._decoder_step(
                current, encoder_states, source_mask, caches, position, register_context
            )
            next_token = self._logits(hidden[:, -1:]).argmax(-1)
            # 이미 끝난 문장은 EOS 를 반복해 길이만 맞춥니다.
            next_token = torch.where(finished[:, None], eos_id, next_token)
            pieces.append(next_token)
            finished |= next_token.squeeze(1).eq(eos_id)
            if finished.all():
                break
            current = next_token
        return torch.cat(pieces, dim=1)

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

        caches = self._fresh_caches(len(self.decoder_layers))
        sequences = torch.full((total, 1), bos_id, dtype=torch.long, device=device)
        # 첫 스텝에서 모든 beam 이 같은 BOS 에서 출발하므로, beam 0 만 점수 0 으로
        # 두고 나머지는 -inf 로 시작해 중복 후보를 걸러냅니다.
        beam_scores = torch.full((batch, num_beams), float("-inf"), device=device)
        beam_scores[:, 0] = 0.0
        # 완성된 가설: batch 별 (length-penalty 적용 점수, 토큰 텐서) 목록
        done: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(batch)]

        def penalized(raw_score: float, length: int) -> float:
            # GNMT length penalty: 길이가 짧은 번역이 유리해지는 것을 보정합니다.
            return raw_score / (((5.0 + length) / 6.0) ** length_penalty)

        for position in range(max_new_tokens):
            hidden = self._decoder_step(
                sequences[:, -1:], encoder_states, source_mask, caches, position, register_context
            )
            log_probs = F.log_softmax(self._logits(hidden[:, -1]).float(), dim=-1)
            vocab = log_probs.shape[-1]
            # 누적 점수 = 지금까지의 beam 점수 + 새 토큰 log 확률
            candidate_scores = (beam_scores.view(-1, 1) + log_probs).view(
                batch, num_beams * vocab
            )
            # EOS 로 빠지는 후보가 있어도 살아있는 beam 을 채울 수 있도록 2배수 선택
            top_scores, top_indices = candidate_scores.topk(2 * num_beams, dim=-1)
            source_beams = top_indices // vocab  # 어느 beam 에서 나온 후보인지
            new_tokens = top_indices % vocab

            next_scores = torch.full_like(beam_scores, float("-inf"))
            gather_flat = torch.zeros(batch, num_beams, dtype=torch.long, device=device)
            step_tokens = torch.full(
                (batch, num_beams), eos_id, dtype=torch.long, device=device
            )
            for b in range(batch):
                slot = 0
                for cand in range(2 * num_beams):
                    score = float(top_scores[b, cand])
                    if score == float("-inf"):
                        continue
                    token = int(new_tokens[b, cand])
                    flat_source = b * num_beams + int(source_beams[b, cand])
                    if token == eos_id:
                        # 완성 가설로 저장 (BOS 제외한 생성 길이 = position+1)
                        finished_seq = torch.cat(
                            (
                                sequences[flat_source],
                                torch.tensor([eos_id], device=device),
                            )
                        )
                        done[b].append((penalized(score, position + 1), finished_seq))
                        continue
                    if slot < num_beams:
                        next_scores[b, slot] = score
                        gather_flat[b, slot] = flat_source
                        step_tokens[b, slot] = token
                        slot += 1

            flat_index = gather_flat.view(-1)
            # 살아남은 beam 의 순서에 맞게 문장 기록과 KV cache 를 재배열합니다.
            sequences = torch.cat(
                (sequences.index_select(0, flat_index), step_tokens.view(-1, 1)), dim=1
            )
            for cache in caches:
                cache["self"] = tuple(t.index_select(0, flat_index) for t in cache["self"])
                cache["cross"] = tuple(t.index_select(0, flat_index) for t in cache["cross"])
            beam_scores = next_scores

            # 모든 문장이 '완성 가설이 충분하고, 살아있는 beam 이 더 나은 점수를
            # 낼 가능성이 없는' 상태면 일찍 종료합니다.
            all_done = True
            for b in range(batch):
                if len(done[b]) < num_beams:
                    all_done = False
                    break
                best_alive = float(beam_scores[b].max())
                best_possible = penalized(best_alive, position + 1)
                worst_kept = min(score for score, _ in done[b])
                if best_possible > worst_kept:
                    all_done = False
                    break
            if all_done:
                break

        # batch 별 최고 가설을 고릅니다. 완성 가설이 없으면(길이 제한에 걸림)
        # 살아있는 beam 중 최고 점수에 EOS 를 붙여 사용합니다.
        outputs: list[torch.Tensor] = []
        for b in range(batch):
            if done[b]:
                outputs.append(max(done[b], key=lambda item: item[0])[1])
            else:
                best = int(beam_scores[b].argmax())
                seq = sequences[b * num_beams + best]
                outputs.append(
                    torch.cat((seq, torch.tensor([eos_id], device=device)))
                )
        longest = max(len(seq) for seq in outputs)
        padded = torch.full(
            (batch, longest), eos_id, dtype=torch.long, device=device
        )
        for b, seq in enumerate(outputs):
            padded[b, : len(seq)] = seq
        return padded

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
