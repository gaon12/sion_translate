from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint  # pyright: ignore[reportUnknownVariableType]

from sion_translate.config import ModelConfig

from .experimental import (
    ActiveEvidenceRepair,
    BilingualAlignmentTransport,
    CandidateDistributionRefinement,
    ContentRegisterState,
    MorphoScriptFusion,
    SemanticParityHead,
    TypedEntityMemory,
)
from .layers import DecoderLayer, EncoderLayer, GQAAttention, RMSNorm, RotaryEmbedding, SwiGLU


_activation_checkpoint = cast(Callable[..., torch.Tensor], checkpoint)
_KeyValue = tuple[torch.Tensor, torch.Tensor]
_MAX_REASONING_LEVEL = 9


class _LayerCache(TypedDict):
    self: _KeyValue | None
    cross: _KeyValue | None


def _all_ranks_finished(local_finished: bool, device: torch.device) -> bool:
    """Return true only when every distributed rank can leave its decode loop."""

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return local_finished
    flag = torch.tensor(1 if local_finished else 0, dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)  # pyright: ignore[reportUnknownMemberType]
    return bool(flag.item())


def _all_ranks_max_new_tokens(local_max_new_tokens: int, device: torch.device) -> int:
    """Use one decode-loop bound even when rank-local batches have different lengths."""

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return local_max_new_tokens
    limit = torch.tensor(local_max_new_tokens, dtype=torch.int32, device=device)
    dist.all_reduce(limit, op=dist.ReduceOp.MAX)  # pyright: ignore[reportUnknownMemberType]
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


def _validate_reasoning_level(reasoning_level: object) -> int | None:
    """Validate the public SSRT compute-control contract.

    ``None`` retains the pre-SSRT behavior for checkpoint and caller
    compatibility. Explicit levels use the stable integer range 0..9, with 0
    meaning that all optional auditing and local repair computation is skipped.
    """

    if reasoning_level is None:
        return None
    if isinstance(reasoning_level, bool) or not isinstance(reasoning_level, int):
        raise TypeError("reasoning_level must be an integer from 0 to 9 or None")
    if not 0 <= reasoning_level <= _MAX_REASONING_LEVEL:
        raise ValueError("reasoning_level must be between 0 and 9")
    return reasoning_level


def _reasoning_budget(reasoning_level: int | None) -> float:
    """Map an SSRT level to its local-reasoning request budget.

    The legacy/unspecified path retains the original full budget. Explicit
    levels divide the available request intensity into nine monotonic bands.
    """

    return 1.0 if reasoning_level is None else reasoning_level / _MAX_REASONING_LEVEL


def _candidate_refinement_steps(reasoning_level: int | None, configured_steps: int) -> int:
    """Map the public compute level to a trained refinement endpoint."""

    reasoning_level = _validate_reasoning_level(reasoning_level)
    if reasoning_level == 0:
        return 0
    if reasoning_level is None:
        return configured_steps
    return max(
        1,
        (configured_steps * reasoning_level + _MAX_REASONING_LEVEL - 1) // _MAX_REASONING_LEVEL,
    )


@dataclass
class SionOutput:
    """Collect model outputs and token-exact loss accounting.

    - ``logits`` contains next-token scores with shape ``(batch, seq, vocab)``.
    - ``loss`` is mean language-model loss per token plus auxiliary losses.
    - ``lm_loss_sum`` and ``token_count`` expose the numerator and denominator
      separately so the trainer can normalize gradient accumulation by the
      exact number of target tokens.
    - ``auxiliary_loss`` combines z-loss and enabled register, alignment,
      evidence, and semantic-parity objectives.
    - ``evidence_*`` reports uncertain error positions and source reread budget.
    - ``semantic_parity_*`` reports the source/target representation checksum
      objective and cosine score.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    lm_loss_sum: torch.Tensor | None = None
    token_count: torch.Tensor | None = None
    auxiliary_loss: torch.Tensor | None = None
    register_loss: torch.Tensor | None = None
    # Fraction of rows not supervised because no register rule matched. A high
    # value means the CoRe auxiliary objective sees only part of the batch.
    register_unsupervised_rate: torch.Tensor | None = None
    alignment_loss: torch.Tensor | None = None
    coverage_loss: torch.Tensor | None = None
    uncertainty_loss: torch.Tensor | None = None
    evidence_budget_loss: torch.Tensor | None = None
    evidence_request_rate: torch.Tensor | None = None
    evidence_repair_gain_loss: torch.Tensor | None = None
    evidence_repair_gain: torch.Tensor | None = None
    candidate_refinement_loss: torch.Tensor | None = None
    candidate_refinement_gain: torch.Tensor | None = None
    candidate_refinement_steps: torch.Tensor | None = None
    # Explicit SSRT request diagnostics. ``reasoning_level`` remains None for
    # legacy calls that did not opt into the new input contract.
    reasoning_level: torch.Tensor | None = None
    reasoning_budget: torch.Tensor | None = None
    reasoning_active: torch.Tensor | None = None
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
    reasoning_level: int | None = None
    memory_token_ids: torch.Tensor | None = None
    memory_mask: torch.Tensor | None = None
    memory_type_ids: torch.Tensor | None = None
    memory_mode_ids: torch.Tensor | None = None


class SionForConditionalGeneration(nn.Module):
    """Implement the Sion encoder-decoder translation model.

    Architecture summary:

    - One joint vocabulary can be tied across encoder input, decoder input, and
      output projection to reduce parameters and improve sharing across the
      configured language graph.
    - The encoder uses ``encoder_layers`` layers. A deeper encoder with a
      shallower decoder can improve source understanding while keeping
      autoregressive inference faster.
    - The decoder combines causal self-attention with source cross-attention.
    - RoPE is applied to encoder and decoder self-attention only.
    - Experimental BATS, CoRe, TETM, MorphoScript, evidence, and refinement
      modules are constructed only when their configuration enables them.
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
        ffn_gate_beta = (
            config.experimental.situglu_gate_beta if config.experimental.situglu_enabled else None
        )
        ffn_up_beta = (
            config.experimental.situglu_up_beta if config.experimental.situglu_enabled else None
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    config.d_model,
                    config.num_heads,
                    config.num_kv_heads,
                    config.d_ff,
                    dropout=config.dropout,
                    rope=self.encoder_rope,
                    qk_norm=config.qk_norm,
                    norm_eps=config.rms_norm_eps,
                    ffn_gate_beta=ffn_gate_beta,
                    ffn_up_beta=ffn_up_beta,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    config.d_model,
                    config.num_heads,
                    config.num_kv_heads,
                    config.d_ff,
                    dropout=config.dropout,
                    rope=self.decoder_rope,
                    qk_norm=config.qk_norm,
                    norm_eps=config.rms_norm_eps,
                    ffn_gate_beta=ffn_gate_beta,
                    ffn_up_beta=ffn_up_beta,
                )
                for _ in range(config.decoder_layers)
            ]
        )
        # Recurrent blocks reuse existing weights and therefore do not add new
        # state-dict keys, preserving checkpoint shape compatibility.
        self.recurrent_block_layers = min(
            config.experimental.recurrent_block_layers, config.encoder_layers
        )
        self.recurrent_steps = max(1, config.experimental.recurrent_steps)
        self.encoder_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.decoder_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head: nn.Linear | None = (
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
        self.morphoscript: MorphoScriptFusion | None = (
            MorphoScriptFusion(config.d_model, exp.script_classes)
            if exp.morphoscript_enabled
            else None
        )
        # Create MorphoScript gate parameters only when enabled. Registering
        # unused parameters would force DDP to search for unused values and add
        # unnecessary communication to every step.
        self.morph_gates: nn.Parameter | None = (
            nn.Parameter(torch.zeros(config.encoder_layers)) if exp.morphoscript_enabled else None
        )
        self.register_state: ContentRegisterState | None = (
            ContentRegisterState(
                config.d_model,
                exp.register_classes,
                norm_eps=config.rms_norm_eps,
            )
            if exp.core_enabled
            else None
        )
        self.typed_memory: TypedEntityMemory | None = (
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
        self.alignment_head: BilingualAlignmentTransport | None = (
            BilingualAlignmentTransport(config.d_model, exp.bats_dim) if exp.bats_enabled else None
        )
        self.evidence_repair: ActiveEvidenceRepair | None = (
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
        self.candidate_refinement: CandidateDistributionRefinement | None = (
            CandidateDistributionRefinement(
                config.d_model,
                norm_eps=config.rms_norm_eps,
            )
            if exp.candidate_refinement_enabled
            else None
        )
        self.semantic_parity: SemanticParityHead | None = (
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

    def residual_write_count(self) -> int:
        """Count additions to the residual stream during one forward pass.

        Weight initialization scales by residual writes rather than layer count
        because residual-stream variance accumulates with every added branch.

        - Each encoder layer writes twice: self-attention and FFN.
        - Each decoder layer writes three times: self-attention, cross-attention,
          and FFN. Counting layers alone underestimates decoder writes by 1.5x.
        - Reusing a recurrent block adds the writes from every repeated layer.
        """

        encoder_writes = 2 * self.config.encoder_layers
        block_size = min(self.recurrent_block_layers, self.config.encoder_layers)
        encoder_writes += 2 * block_size * (self.recurrent_steps - 1)
        decoder_writes = 3 * self.config.decoder_layers
        return max(encoder_writes, decoder_writes)

    def init_weights(self) -> None:
        """Initialize weights while scaling residual output projections.

        Most weights use ``N(0, init_std**2)``. Output and down projections that
        write into the residual stream use a smaller scale based on
        ``residual_write_count`` to prevent early divergence in deep models.
        """
        std = self.config.init_std
        residual_std = std / self.residual_write_count() ** 0.5

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
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
            if self.candidate_refinement is not None:
                self.candidate_refinement.refinement_scale.zero_()
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
            return _activation_checkpoint(layer, *args, use_reentrant=False)
        return cast(torch.Tensor, layer(*args))

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
            layer = cast(EncoderLayer, self.encoder_layers[index])
            hidden = self._checkpoint(layer, hidden, attention_mask)
            if side_states is not None and (index + 1) % interval == 0:
                if self.morph_gates is None:
                    raise RuntimeError("MorphoScript gates are unavailable")
                hidden = hidden + torch.tanh(self.morph_gates[index]) * side_states
            return hidden

        # Treat the final ``recurrent_block_layers`` as one block and run that
        # whole block ``recurrent_steps`` times. Repeating (L2, L3) together is
        # not equivalent to running L2 repeatedly and then L3 repeatedly. The
        # shared weights increase effective depth without increasing parameter
        # count. A block size of zero leaves the ordinary one-pass layer loop.
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
        # Tied embeddings reuse the input embedding matrix as the output
        # projection, reducing parameters and encouraging shared structure.
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
        reasoning_level: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        reasoning_level = _validate_reasoning_level(reasoning_level)
        if self.evidence_repair is None or reasoning_level == 0:
            return decoder_states, None, None
        return self.evidence_repair(
            decoder_states,
            encoder_states,
            source_mask,
            evidence_key_value=evidence_key_value,
            reasoning_budget=_reasoning_budget(reasoning_level),
        )

    def _candidate_distribution_statistics(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return exact full-vocabulary expectation and optional draft CE.

        Vocabulary chunks bound the temporary probability/logit storage. This
        is mathematically identical to ``softmax(logits / temperature) @ E``;
        no top-k truncation or gradient detachment is used.
        """

        exp = self.config.experimental
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        output_weight = (
            self.lm_head.weight if self.lm_head is not None else self.token_embedding.weight
        )
        embedding_weight = self.token_embedding.weight
        vocabulary_size = output_weight.shape[0]
        chunk_size = min(exp.candidate_refinement_vocab_chunk_size, vocabulary_size)
        row_count = flat_hidden.shape[0]
        raw_log_normalizer = torch.full(
            (row_count,),
            float("-inf"),
            device=hidden.device,
            dtype=torch.float32,
        )
        logit_sum = torch.zeros_like(raw_log_normalizer)
        target_logits = torch.zeros_like(raw_log_normalizer)
        distribution_max = raw_log_normalizer.clone()
        distribution_denominator = torch.zeros_like(raw_log_normalizer)
        candidate_numerator = torch.zeros(
            (row_count, hidden.shape[-1]),
            device=hidden.device,
            dtype=torch.float32,
        )
        flat_labels = labels.reshape(-1) if labels is not None else None
        safe_labels = (
            flat_labels.masked_fill(flat_labels.eq(-100), 0) if flat_labels is not None else None
        )

        for start in range(0, vocabulary_size, chunk_size):
            stop = min(start + chunk_size, vocabulary_size)
            chunk_logits = F.linear(flat_hidden, output_weight[start:stop]).float()
            raw_log_normalizer = torch.logaddexp(
                raw_log_normalizer,
                chunk_logits.logsumexp(-1),
            )
            scaled_logits = chunk_logits / exp.candidate_refinement_temperature
            chunk_max = scaled_logits.max(-1).values
            next_distribution_max = torch.maximum(distribution_max, chunk_max)
            previous_scale = torch.exp(distribution_max - next_distribution_max)
            chunk_weights = torch.exp(scaled_logits - next_distribution_max[:, None])
            distribution_denominator = (
                distribution_denominator * previous_scale + chunk_weights.sum(-1)
            )
            candidate_numerator = (
                candidate_numerator * previous_scale[:, None]
                + chunk_weights @ embedding_weight[start:stop].float()
            )
            distribution_max = next_distribution_max
            logit_sum = logit_sum + chunk_logits.sum(-1)
            if safe_labels is not None:
                selected_by_chunk = safe_labels.ge(start) & safe_labels.lt(stop)
                local_labels = (safe_labels - start).clamp(0, stop - start - 1)
                selected_logits = chunk_logits.gather(1, local_labels[:, None]).squeeze(1)
                target_logits = torch.where(
                    selected_by_chunk,
                    selected_logits,
                    target_logits,
                )

        candidate_expectation = (
            (
                candidate_numerator
                / distribution_denominator.clamp_min(torch.finfo(torch.float32).tiny)[:, None]
            )
            .reshape_as(hidden)
            .to(hidden.dtype)
        )

        if flat_labels is None:
            token_nll = raw_log_normalizer.new_zeros(hidden.shape[:-1])
            return candidate_expectation, hidden.new_zeros(()), token_nll
        valid = flat_labels.ne(-100)
        token_nll_flat = raw_log_normalizer - target_logits
        smooth_loss = raw_log_normalizer - logit_sum / vocabulary_size
        draft_loss_values = (
            1.0 - self.config.label_smoothing
        ) * token_nll_flat + self.config.label_smoothing * smooth_loss
        draft_loss_sum = (draft_loss_values * valid.to(dtype=draft_loss_values.dtype)).sum()
        token_nll = torch.where(valid, token_nll_flat, torch.zeros_like(token_nll_flat))
        return (
            candidate_expectation,
            draft_loss_sum,
            token_nll.reshape(hidden.shape[:-1]).detach(),
        )

    def _candidate_refinement_step(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.candidate_refinement is not None
        candidate_expectation, draft_loss_sum, token_nll = self._candidate_distribution_statistics(
            hidden, labels
        )
        return (
            self.candidate_refinement(hidden, candidate_expectation),
            draft_loss_sum,
            token_nll,
        )

    def _apply_candidate_refinement(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
        *,
        reasoning_level: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        steps = (
            _candidate_refinement_steps(
                reasoning_level,
                self.config.experimental.candidate_refinement_steps,
            )
            if self.candidate_refinement is not None
            else 0
        )
        draft_loss_sum = hidden.new_zeros(())
        first_token_nll: torch.Tensor | None = None

        def refinement_step(
            states: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self._candidate_refinement_step(states, labels)

        for _ in range(steps):
            if self.training and torch.is_grad_enabled():
                refined = cast(
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    checkpoint(
                        refinement_step,
                        hidden,
                        use_reentrant=False,
                    ),
                )
            else:
                refined = self._candidate_refinement_step(hidden, labels)
            hidden, step_loss_sum, token_nll = refined
            draft_loss_sum = draft_loss_sum + step_loss_sum
            if first_token_nll is None and labels is not None:
                first_token_nll = token_nll
        return hidden, draft_loss_sum, first_token_nll, steps

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
        reasoning_level: int | None = None,
    ) -> SionOutput:
        # The collator supplies these direction fields for posttraining. Plain
        # SFT does not consume them, but accepting them keeps batch forwarding
        # uniform across objectives.
        del source_language_tag_ids, reverse_direction_trained
        reasoning_level = _validate_reasoning_level(reasoning_level)
        evidence_reasoning_is_active = self.evidence_repair is not None and reasoning_level != 0
        configured_candidate_steps = (
            _candidate_refinement_steps(
                reasoning_level,
                self.config.experimental.candidate_refinement_steps,
            )
            if self.candidate_refinement is not None
            else 0
        )
        reasoning_is_active = evidence_reasoning_is_active or configured_candidate_steps > 0
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
        if labels is not None and evidence_reasoning_is_active:
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
                # Holding one ``(batch, seq, vocab)`` tensor can cost gigabytes
                # for a large vocabulary. Retain only the two summaries.
                del pre_repair_logits
        decoder_states, uncertainty_logits, evidence_requests = self._apply_evidence_repair(
            decoder_states,
            encoder_states,
            attention_mask,
            reasoning_level=reasoning_level,
        )
        (
            decoder_states,
            candidate_draft_loss_sum,
            pre_refinement_token_nll,
            applied_candidate_steps,
        ) = self._apply_candidate_refinement(
            decoder_states,
            labels,
            reasoning_level=reasoning_level,
        )
        logits = self._logits(decoder_states)

        requested_level = (
            logits.new_tensor(reasoning_level, dtype=torch.long)
            if reasoning_level is not None
            else None
        )
        reasoning_budget = logits.new_tensor(_reasoning_budget(reasoning_level))
        reasoning_active = logits.new_tensor(reasoning_is_active)
        candidate_steps_tensor = logits.new_tensor(applied_candidate_steps, dtype=torch.long)

        if labels is None:
            request_rate = evidence_requests.mean() if evidence_requests is not None else None
            if reasoning_level is not None and request_rate is None:
                request_rate = logits.new_zeros(())
            return SionOutput(
                logits=logits,
                evidence_request_rate=request_rate,
                reasoning_level=requested_level,
                reasoning_budget=reasoning_budget,
                reasoning_active=reasoning_active,
                candidate_refinement_steps=candidate_steps_tensor,
            )
        # Labels equal to -100 mark padding and do not contribute to loss.
        token_count = labels.ne(-100).sum()
        # Loss must use FP32 under BF16 autocast. Each separate ``logits.float()``
        # call creates a ``(batch, seq, vocab)`` copy that survives until
        # backward. At batch 32, sequence 512, and vocabulary 48,000, one copy
        # is about 3.1 GB. Share one conversion across every FP32 consumer.
        float_logits = logits.float()
        lm_loss_sum = F.cross_entropy(
            float_logits.reshape(-1, float_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
            label_smoothing=self.config.label_smoothing,
        )
        auxiliary_loss = logits.new_zeros(())
        # Z-loss lightly penalizes squared logsumexp so logits cannot grow
        # without bound, improving mixed-precision training stability.
        if self.config.z_loss_weight > 0:
            log_normalizer = float_logits.logsumexp(-1)
            valid_targets = labels.ne(-100).to(dtype=log_normalizer.dtype)
            z_loss = (log_normalizer.square() * valid_targets).sum()
            z_loss = z_loss / valid_targets.sum().clamp_min(1.0)
            auxiliary_loss = auxiliary_loss + self.config.z_loss_weight * z_loss
        register_loss = logits.new_zeros(())
        register_unsupervised_rate = logits.new_zeros(())
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
            # Surface the unsupervised fraction. Fragments, interjections, and
            # other unmatched rows do not enter this loss; without the metric,
            # CoRe could see only half a batch without making that gap visible.
            register_unsupervised_rate = 1.0 - (known_weights.sum() / known_weights.numel())

        alignment_loss = logits.new_zeros(())
        coverage_loss = logits.new_zeros(())
        uncertainty_loss = logits.new_zeros(())
        evidence_budget_loss = logits.new_zeros(())
        evidence_request_rate = logits.new_zeros(())
        evidence_repair_gain_loss = logits.new_zeros(())
        evidence_repair_gain = logits.new_zeros(())
        candidate_refinement_loss = logits.new_zeros(())
        candidate_refinement_gain = logits.new_zeros(())
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
        final_token_nll = None
        if evidence_requests is not None or applied_candidate_steps:
            final_token_nll = F.cross_entropy(
                float_logits.transpose(1, 2),
                labels,
                ignore_index=-100,
                reduction="none",
            )
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
                F.relu(
                    evidence_request_rate
                    - exp.evidence_budget_target * _reasoning_budget(reasoning_level)
                ).square()
                * has_target
            )
            assert final_token_nll is not None
            post_evidence_token_nll = (
                pre_refinement_token_nll
                if pre_refinement_token_nll is not None
                else final_token_nll.detach()
            )
            token_gain = (pre_repair_token_nll - post_evidence_token_nll).detach()
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

        if applied_candidate_steps:
            assert pre_refinement_token_nll is not None
            assert final_token_nll is not None
            candidate_refinement_loss = candidate_draft_loss_sum / (
                token_count.clamp_min(1) * applied_candidate_steps
            )
            valid = target_mask.to(logits.dtype)
            refinement_gain = (pre_refinement_token_nll - final_token_nll.detach()) * valid
            candidate_refinement_gain = refinement_gain.sum() / valid.sum().clamp_min(1.0)
            auxiliary_loss = auxiliary_loss + (
                exp.candidate_refinement_loss_weight * candidate_refinement_loss
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
            register_unsupervised_rate=register_unsupervised_rate,
            alignment_loss=alignment_loss,
            coverage_loss=coverage_loss,
            uncertainty_loss=uncertainty_loss,
            evidence_budget_loss=evidence_budget_loss,
            evidence_request_rate=evidence_request_rate,
            evidence_repair_gain_loss=evidence_repair_gain_loss,
            evidence_repair_gain=evidence_repair_gain,
            candidate_refinement_loss=candidate_refinement_loss,
            candidate_refinement_gain=candidate_refinement_gain,
            candidate_refinement_steps=candidate_steps_tensor,
            reasoning_level=requested_level,
            reasoning_budget=reasoning_budget,
            reasoning_active=reasoning_active,
            semantic_parity_loss=semantic_parity_loss,
            semantic_parity_score=semantic_parity_score,
        )

    def _decoder_step(
        self,
        tokens: torch.Tensor,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
        caches: list[_LayerCache],
        position: int,
        register_context: torch.Tensor | None,
        *,
        evidence_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        memory_token_ids: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_type_ids: torch.Tensor | None = None,
        memory_mode_ids: torch.Tensor | None = None,
        reasoning_level: int | None = None,
    ) -> torch.Tensor:
        """Decode one new token with inference KV caches.

        ``caches`` contains one ``{"self": (k, v), "cross": (k, v)}`` mapping
        per decoder layer. This method updates those mappings in place.
        """
        hidden = self._embed(tokens)
        if register_context is not None:
            hidden = hidden + register_context[:, None, :].to(dtype=hidden.dtype)
        for raw_layer, cache in zip(self.decoder_layers, caches, strict=True):
            layer = cast(DecoderLayer, raw_layer)
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
            reasoning_level=reasoning_level,
        )
        hidden, _, _, _ = self._apply_candidate_refinement(
            hidden,
            None,
            reasoning_level=reasoning_level,
        )
        return hidden

    @staticmethod
    def _fresh_caches(
        layer_count: int,
        cross_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        *,
        repeats: int = 1,
    ) -> list[_LayerCache]:
        if cross_key_values is None:
            return [{"self": None, "cross": None} for _ in range(layer_count)]
        if len(cross_key_values) != layer_count:
            raise ValueError("cross_key_values must have one entry per decoder layer")
        caches: list[_LayerCache] = [
            {
                "self": None,
                "cross": (
                    (
                        key_value[0].repeat_interleave(repeats, dim=0),
                        key_value[1].repeat_interleave(repeats, dim=0),
                    )
                    if repeats > 1
                    else key_value
                ),
            }
            for key_value in cross_key_values
        ]
        return caches

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
        reasoning_level: int | None = None,
        **encoder_features: torch.Tensor,
    ) -> GenerationContext:
        """Encode once and pre-project decoder cross-attention key/value states."""
        reasoning_level = _validate_reasoning_level(reasoning_level)
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
                cast(DecoderLayer, layer).project_cross_key_value(encoder_states)
                for layer in self.decoder_layers
            )
            evidence_key_value = (
                self.evidence_repair.project_key_value(encoder_states)
                if self.evidence_repair is not None and reasoning_level != 0
                else None
            )
            return GenerationContext(
                encoder_states=encoder_states,
                source_mask=attention_mask,
                register_context=register_context,
                cross_key_values=cross_key_values,
                evidence_key_value=evidence_key_value,
                reasoning_level=reasoning_level,
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
        reasoning_level: int | None = None,
    ) -> None:
        if context.encoder_states.shape[0] != input_ids.shape[0]:
            raise ValueError("generation_context batch size does not match input_ids")
        if context.source_mask.shape != attention_mask.shape:
            raise ValueError("generation_context source mask does not match attention_mask")
        if context.encoder_states.device != input_ids.device:
            raise ValueError("generation_context and input_ids must be on the same device")
        if reasoning_level is not None and context.reasoning_level != reasoning_level:
            raise ValueError("generation_context reasoning_level does not match reasoning_level")

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

        # Keep EOS as a last-resort escape when a tiny vocabulary or excessive
        # constraints remove every other token, preventing NaN sampling.
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
        reasoning_level: int | None = None,
        **encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Generate translations with cached greedy or beam decoding.

        - ``num_beams=1`` uses the fastest greedy path.
        - ``num_beams>=2`` uses beam search with the GNMT length penalty.

        Both paths use KV caches, making each new token linear in current
        sequence length instead of recomputing the full prefix. Select beam and
        length-penalty values from evaluation of the actual release checkpoint.
        """
        reasoning_level = _validate_reasoning_level(reasoning_level)
        if isinstance(max_new_tokens, bool):
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
                    reasoning_level=reasoning_level,
                    **encoder_features,
                )
            else:
                self._validate_generation_context(
                    generation_context,
                    input_ids,
                    attention_mask,
                    reasoning_level,
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
                    reasoning_level=generation_context.reasoning_level,
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
                reasoning_level=generation_context.reasoning_level,
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
        reasoning_level: int | None = None,
        **encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Generate stochastic MRT candidates as ``(batch, samples, length)``."""
        reasoning_level = _validate_reasoning_level(reasoning_level)
        if isinstance(max_new_tokens, bool):
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
                    reasoning_level=reasoning_level,
                    **encoder_features,
                )
            else:
                self._validate_generation_context(
                    generation_context,
                    input_ids,
                    attention_mask,
                    reasoning_level,
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
                evidence_key_value = (
                    evidence_key_value[0].repeat_interleave(num_samples, dim=0),
                    evidence_key_value[1].repeat_interleave(num_samples, dim=0),
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
                    reasoning_level=generation_context.reasoning_level,
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
        reasoning_level: int | None = None,
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
                reasoning_level=reasoning_level,
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
            # Repeat EOS for completed rows only to keep tensor lengths aligned.
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
        reasoning_level: int | None = None,
    ) -> torch.Tensor:
        """Run batched beam search.

        Each source is replicated ``num_beams`` times and decoded as one
        ``batch * beams`` tensor. At every step, the top ``2 * beams`` options
        are split into completed EOS hypotheses, scored with the GNMT length
        penalty, and the best remaining live beams.
        """
        batch = encoder_states.shape[0]
        device = encoder_states.device
        total = batch * num_beams

        # Replicate encoder output once for every beam of each source row.
        encoder_states = encoder_states.repeat_interleave(num_beams, dim=0)
        source_mask = source_mask.repeat_interleave(num_beams, dim=0)
        if register_context is not None:
            register_context = register_context.repeat_interleave(num_beams, dim=0)
        if evidence_key_value is not None:
            evidence_key_value = (
                evidence_key_value[0].repeat_interleave(num_beams, dim=0),
                evidence_key_value[1].repeat_interleave(num_beams, dim=0),
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
        # Every beam begins at the same BOS. Give only beam zero a finite score
        # so the first step does not create duplicate hypotheses.
        beam_scores = torch.full((batch, num_beams), float("-inf"), device=device)
        beam_scores[:, 0] = 0.0
        # Completed hypotheses per row: (length-penalized score, token tensor).
        done: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(batch)]
        maximum_completion_lengths: list[int] = (
            cast(
                list[int],
                max_new_tokens_per_row.tolist(),  # pyright: ignore[reportUnknownMemberType]
            )
            if max_new_tokens_per_row is not None
            else [max_new_tokens] * batch
        )

        def penalized(raw_score: float, length: int) -> float:
            # GNMT length penalty offsets the raw-score preference for short text.
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
                reasoning_level=reasoning_level,
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
            # Accumulate the existing beam score and the new token log probability.
            candidate_scores = (beam_scores.view(-1, 1) + log_probs).view(batch, num_beams * vocab)
            # Select twice as many options so EOS completions do not empty live beams.
            top_scores, top_indices = candidate_scores.topk(2 * num_beams, dim=-1)
            source_beams = top_indices // vocab  # Beam that produced each candidate.
            new_tokens = top_indices % vocab

            # Select candidates with tensor operations. The previous loop read
            # ``batch * 2 * beams * 3`` scalars on the host per step, and every
            # GPU scalar read forced device synchronization.
            #
            # ``topk`` is already score-descending, so the first ``num_beams``
            # live candidates are the best live candidates. A cumulative sum
            # gives each survivor a unique destination slot.
            finite = top_scores.ne(float("-inf"))
            ends_here = new_tokens.eq(eos_id)
            alive = finite & ~ends_here
            alive_rank = alive.cumsum(dim=1) - 1
            keep = alive & alive_rank.lt(num_beams)
            flat_sources = (
                torch.arange(batch, device=device).unsqueeze(1) * num_beams + source_beams
            )
            # Scatter rejected options into one disposable final column. Every
            # accepted survivor has a distinct rank, so accepted writes cannot
            # collide.
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

            # Transfer only completed hypotheses to the host. There are usually
            # zero to two per step, and row-major ``nonzero`` preserves the old
            # ``(batch, candidate)`` tie-breaking order in a single sync.
            finished = (finite & ends_here).nonzero(as_tuple=False)
            if finished.numel():
                finished_rows = finished[:, 0]
                finished_scores = cast(
                    list[float],
                    top_scores[finished_rows, finished[:, 1]].tolist(),  # pyright: ignore[reportUnknownMemberType]
                )
                finished_sources = cast(
                    list[int],
                    flat_sources[finished_rows, finished[:, 1]].tolist(),  # pyright: ignore[reportUnknownMemberType]
                )
                finished_row_numbers = cast(
                    list[int],
                    finished_rows.tolist(),  # pyright: ignore[reportUnknownMemberType]
                )
                eos_column = torch.tensor([eos_id], device=device)
                for row, score, source in zip(
                    finished_row_numbers,
                    finished_scores,
                    finished_sources,
                    strict=True,
                ):
                    # Completed hypothesis; generated length excludes BOS.
                    done[row].append(
                        (
                            penalized(score, position + 1),
                            torch.cat((sequences[source], eos_column)),
                        )
                    )

            flat_index = gather_flat.reshape(-1)
            # Reorder sequences and self-attention caches to match surviving
            # beams. Cross-attention caches come from encoder output and are
            # identical across beams for one source, so reordering them would
            # only copy several unnecessary MiB per step.
            sequences = torch.cat(
                (sequences.index_select(0, flat_index), step_tokens.reshape(-1, 1)), dim=1
            )
            for cache in caches:
                self_key_value = cache["self"]
                if self_key_value is None:
                    raise RuntimeError("decoder cache was not initialized")
                cache["self"] = (
                    self_key_value[0].index_select(0, flat_index),
                    self_key_value[1].index_select(0, flat_index),
                )
            beam_scores = next_scores

            # Stop once every row has enough completions and no live beam can
            # improve its retained score.
            all_done = all(len(hypotheses) >= num_beams for hypotheses in done)
            if all_done:
                # One reduction and one host transfer instead of one per row.
                best_alive = cast(
                    list[float],
                    beam_scores.amax(dim=1).tolist(),  # pyright: ignore[reportUnknownMemberType]
                )
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

        # Compare live beams that reached the length limit with completed
        # hypotheses. Otherwise one early low-probability EOS could discard a
        # better unfinished translation exactly at ``max_new_tokens``.
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
