from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import nn

from .layers import GQAAttention, RMSNorm


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
    def __init__(self, d_model: int, register_classes: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
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
        predicted_context = probabilities @ self.register_embeddings.weight
        if register_labels is not None:
            safe_labels = register_labels.clamp_min(0)
            known = register_labels > 0
            gold_context = self.register_embeddings(safe_labels)
            context = torch.where(known[:, None], gold_context, predicted_context)
        else:
            context = predicted_context
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
