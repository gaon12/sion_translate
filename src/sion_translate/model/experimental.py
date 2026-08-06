from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import nn

from .layers import GQAAttention, RMSNorm


class ActiveEvidenceRepair(nn.Module):
    """Uncertainty-gated source re-reading and local decoder-state repair.

    The regular decoder cross-attention has already read the source. This module
    is an optional second evidence path after the decoder stack: each target
    position predicts whether it needs more evidence, attends to the encoder
    again, and applies a bounded residual repair. A learned request budget keeps
    the gate from degenerating into "always re-read" during supervised training.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        dropout: float,
        qk_norm: bool,
        norm_eps: float,
    ):
        super().__init__()
        self.query_norm = RMSNorm(d_model, norm_eps)
        self.evidence_norm = RMSNorm(d_model, norm_eps)
        self.attention = GQAAttention(
            d_model,
            num_heads,
            num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            rope=None,
        )
        self.uncertainty_head = nn.Linear(d_model, 1)
        self.repair = nn.Sequential(
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        # Start checkpoint-compatibly as an identity residual. Shape (1,) is
        # shardable by FSDP2 and broadcasts over (batch, length, d_model).
        self.repair_scale = nn.Parameter(torch.zeros(1))

    def project_key_value(
        self,
        encoder_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project encoder evidence once for cached autoregressive decoding."""

        return self.attention.project_key_value(self.evidence_norm(encoder_states))

    def forward(
        self,
        decoder_states: torch.Tensor,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        evidence_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query_norm(decoder_states)
        uncertainty_logits = self.uncertainty_head(query).squeeze(-1)
        request_probabilities = uncertainty_logits.sigmoid()
        # ``key_value_states`` keeps this on the cross-attention path. Its
        # contents are ignored when the projected cache is supplied, so avoid
        # normalizing the full encoder sequence on every generated token.
        evidence_states = (
            encoder_states if evidence_key_value is not None else self.evidence_norm(encoder_states)
        )
        evidence = self.attention(
            query,
            key_value_states=evidence_states,
            key_padding_mask=source_mask,
            past_key_value=evidence_key_value,
        )
        proposal = self.repair(torch.cat((decoder_states, evidence), dim=-1))
        repaired = decoder_states + (
            torch.tanh(self.repair_scale)
            * request_probabilities.unsqueeze(-1).to(proposal.dtype)
            * proposal
        )
        return repaired, uncertainty_logits, request_probabilities


class SemanticParityHead(nn.Module):
    """A train-time semantic checksum shared by source and target states."""

    def __init__(
        self,
        d_model: int,
        parity_dim: int,
        temperature: float,
        *,
        norm_eps: float,
    ):
        super().__init__()
        self.source_norm = RMSNorm(d_model, norm_eps)
        self.target_norm = RMSNorm(d_model, norm_eps)
        self.source_proj = nn.Linear(d_model, parity_dim, bias=False)
        self.target_proj = nn.Linear(d_model, parity_dim, bias=False)
        self.temperature = temperature

    @staticmethod
    def _pool(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(states.dtype)
        return (states * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def forward(
        self,
        source_states: torch.Tensor,
        target_states: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = F.normalize(
            self.source_proj(self._pool(self.source_norm(source_states), source_mask)),
            dim=-1,
        )
        target = F.normalize(
            self.target_proj(self._pool(self.target_norm(target_states), target_mask)),
            dim=-1,
        )
        similarity = source @ target.transpose(0, 1)
        labels = torch.arange(similarity.shape[0], device=similarity.device)
        valid_rows = target_mask.any(-1)
        valid_weights = valid_rows.to(similarity.dtype)
        denominator = valid_weights.sum().clamp_min(1.0)
        scaled = similarity.float() / self.temperature
        minimum = torch.finfo(scaled.dtype).min
        source_to_target = scaled.masked_fill(~valid_rows[None, :], minimum)
        target_to_source = scaled.transpose(0, 1).masked_fill(~valid_rows[None, :], minimum)
        # The sentinel is finite, so even a completely padded micro-batch has a
        # well-defined softmax. Invalid query rows are removed by the weights;
        # invalid paired examples are also masked as candidates in both
        # directions, rather than becoming accidental hard negatives.
        contrastive_values = 0.5 * (
            F.cross_entropy(source_to_target, labels, reduction="none")
            + F.cross_entropy(target_to_source, labels, reduction="none")
        )
        contrastive = (contrastive_values * valid_weights).sum() / denominator
        positive_similarity = (similarity.diagonal() * valid_weights).sum() / denominator
        # The positive cosine term gives a useful signal for batch size one,
        # where the in-batch contrastive objective is exactly zero.
        has_any_target = valid_rows.any().to(similarity.dtype)
        loss = (contrastive + (1.0 - positive_similarity)) * has_any_target
        return loss, positive_similarity


class MorphoScriptFusion(nn.Module):
    def __init__(self, d_model: int, script_classes: int):
        super().__init__()
        feature_dim = max(32, d_model // 16)
        self.script = nn.Embedding(script_classes, feature_dim)
        self.onset = nn.Embedding(20, feature_dim)
        self.vowel = nn.Embedding(22, feature_dim)
        self.coda = nn.Embedding(29, feature_dim)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim * 4, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

    def forward(
        self,
        script_ids: torch.Tensor,
        onset_ids: torch.Tensor,
        vowel_ids: torch.Tensor,
        coda_ids: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                self.script(script_ids),
                self.onset(onset_ids),
                self.vowel(vowel_ids),
                self.coda(coda_ids),
            ),
            dim=-1,
        )
        return self.proj(features)


class ContentRegisterState(nn.Module):
    def __init__(self, d_model: int, register_classes: int, *, norm_eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(d_model, norm_eps)
        self.classifier = nn.Linear(d_model, register_classes)
        self.register_embeddings = nn.Embedding(register_classes, d_model)
        # FSDP2 cannot shard scalar parameters. A one-element vector has the
        # same broadcast semantics against ``(batch, d_model)`` contexts while
        # remaining a shardable parameter.
        self.inject_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        encoder_states: torch.Tensor,
        source_mask: torch.Tensor,
        register_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = source_mask.unsqueeze(-1).to(encoder_states.dtype)
        pooled = (encoder_states * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        pooled = self.norm(pooled)
        logits = self.classifier(pooled)
        probabilities = logits.softmax(-1)
        # Decoder conditioning must be identical in training and generation.
        # Earlier versions injected the gold register embedding whenever a
        # label was available, but generation can only use the classifier's
        # prediction. That oracle path taught the decoder to depend on context
        # it never receives at inference time, which is especially damaging to
        # style-sensitive short expressions. Labels remain useful to supervise
        # ``logits`` in the parent model; they never select decoder features.
        del register_labels
        context = probabilities @ self.register_embeddings.weight
        return pooled, torch.tanh(self.inject_gate) * context, logits


class TypedEntityMemory(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        memory_types: int,
        memory_modes: int,
        *,
        dropout: float,
        qk_norm: bool,
        norm_eps: float,
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(memory_types, d_model)
        self.mode_embedding = nn.Embedding(memory_modes, d_model)
        self.slot_norm = RMSNorm(d_model, norm_eps)
        self.query_norm = RMSNorm(d_model, norm_eps)
        self.attention = GQAAttention(
            d_model,
            num_heads,
            num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            norm_eps=norm_eps,
            rope=None,
        )
        # Keep this gate one-dimensional for FSDP2 compatibility. Shape ``(1,)``
        # still broadcasts exactly like the former scalar over
        # ``(batch, target_length, d_model)`` attention outputs.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        decoder_states: torch.Tensor,
        token_embedding: nn.Embedding,
        memory_token_ids: torch.Tensor,
        memory_type_ids: torch.Tensor,
        memory_mode_ids: torch.Tensor,
        memory_mask: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        token_mask = memory_token_ids.ne(pad_id).unsqueeze(-1)
        token_states = token_embedding(memory_token_ids)
        slot_states = (token_states * token_mask).sum(-2) / token_mask.sum(-2).clamp_min(1)
        slot_states = (
            slot_states
            + self.type_embedding(memory_type_ids)
            + self.mode_embedding(memory_mode_ids)
        )
        has_memory = memory_mask.any(-1)
        safe_mask = memory_mask.clone()
        safe_mask[~has_memory, 0] = True
        attended = self.attention(
            self.query_norm(decoder_states),
            key_value_states=self.slot_norm(slot_states),
            key_padding_mask=safe_mask,
        )
        attended = attended * has_memory[:, None, None].to(attended.dtype)
        return decoder_states + torch.tanh(self.gate) * attended


class BilingualAlignmentTransport(nn.Module):
    def __init__(self, d_model: int, alignment_dim: int):
        super().__init__()
        self.source_proj = nn.Linear(d_model, alignment_dim, bias=False)
        self.target_proj = nn.Linear(d_model, alignment_dim, bias=False)
        self.null_source = nn.Parameter(torch.zeros(alignment_dim))
        self.scale = alignment_dim**-0.5

    @staticmethod
    def _positions(length: int, stride: int, maximum: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(0, length, max(1, stride), device=device)
        if len(positions) > maximum:
            select = torch.linspace(0, len(positions) - 1, maximum, device=device).round().long()
            positions = positions.index_select(0, select)
        return positions

    def forward(
        self,
        source_states: torch.Tensor,
        target_states: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        stride: int,
        max_positions: int,
        alignment_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_pos = self._positions(
            source_states.shape[1], stride, max_positions, source_states.device
        )
        tgt_pos = self._positions(
            target_states.shape[1], stride, max_positions, target_states.device
        )
        source = self.source_proj(source_states.index_select(1, src_pos))
        target = self.target_proj(target_states.index_select(1, tgt_pos))
        src_mask = source_mask.index_select(1, src_pos)
        tgt_mask = target_mask.index_select(1, tgt_pos)
        null = self.null_source[None, None].expand(source.shape[0], 1, -1)
        source = torch.cat((source, null), dim=1)
        src_mask = torch.cat(
            (src_mask, torch.ones(src_mask.shape[0], 1, device=src_mask.device, dtype=torch.bool)),
            dim=1,
        )
        logits = torch.einsum("btd,bsd->bts", target, source) * self.scale
        logits = logits.masked_fill(~src_mask[:, None, :], torch.finfo(logits.dtype).min)
        probabilities = logits.softmax(-1)

        alignment_loss = logits.new_zeros(())
        if alignment_targets is not None:
            targets = alignment_targets.index_select(1, tgt_pos).index_select(2, src_pos)
            targets = torch.cat((targets, torch.zeros_like(targets[:, :, :1])), dim=-1)
            valid = tgt_mask[:, :, None].to(targets.dtype)
            denominator = (targets * valid).sum().clamp_min(1.0)
            alignment_loss = -(targets * valid * logits.log_softmax(-1)).sum() / denominator

        # A deliberately weak under-coverage signal; it is disabled by default.
        non_null = probabilities[:, :, :-1] * tgt_mask[:, :, None].to(probabilities.dtype)
        coverage = non_null.sum(1)
        expected = tgt_mask.sum(1, keepdim=True) / src_mask[:, :-1].sum(1, keepdim=True).clamp_min(
            1
        )
        coverage_loss = (
            F.relu(0.5 * expected - coverage) * src_mask[:, :-1].to(coverage.dtype)
        ).sum() / src_mask[:, :-1].sum().clamp_min(1)
        return logits, alignment_loss, coverage_loss
