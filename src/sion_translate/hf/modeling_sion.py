"""Transformers model wrapper around the native Sion implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput

from sion_translate.model import (
    SionForConditionalGeneration as NativeSionForConditionalGeneration,
)

from .configuration_sion import SionConfig


def shift_tokens_right(
    labels: torch.Tensor,
    *,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> torch.Tensor:
    shifted = labels.new_full(labels.shape, pad_token_id)
    shifted[:, 0] = decoder_start_token_id
    shifted[:, 1:] = labels[:, :-1]
    return shifted.masked_fill(shifted.eq(-100), pad_token_id)


class SionForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """Hugging Face-compatible facade with a stable ``model.*`` state dict."""

    config_class = SionConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["EncoderLayer", "DecoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: SionConfig):
        super().__init__(config)
        self.model = NativeSionForConditionalGeneration(
            config.to_model_config(),
            pad_id=int(config.pad_token_id),
        )
        # Required by Transformers 5 for tied-weight metadata, sharded loading,
        # device maps and gradient-checkpointing compatibility. The native model
        # has already performed its architecture-specific initialization.
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        del module

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.token_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.token_embedding = value

    def get_output_embeddings(self) -> nn.Module | None:
        return self.model.lm_head

    def set_output_embeddings(self, value: nn.Module | None) -> None:
        self.model.lm_head = value

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Callable[..., Any] = checkpoint,
    ) -> None:
        del gradient_checkpointing_func
        self.model.config.gradient_checkpointing = enable
        self.config.gradient_checkpointing = enable

    def prepare_decoder_input_ids_from_labels(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return shift_tokens_right(
            labels,
            pad_token_id=int(self.config.pad_token_id),
            decoder_start_token_id=int(self.config.decoder_start_token_id),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput | tuple[torch.Tensor, ...]:
        if attention_mask is None:
            attention_mask = input_ids.ne(int(self.config.pad_token_id))
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("decoder_input_ids or labels must be provided")
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
        supported = {
            name: kwargs[name]
            for name in (
                "register_labels",
                "memory_token_ids",
                "memory_mask",
                "memory_type_ids",
                "memory_mode_ids",
                "src_script_ids",
                "src_onset_ids",
                "src_vowel_ids",
                "src_coda_ids",
                "alignment_targets",
            )
            if name in kwargs
        }
        native = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **supported,
        )
        if return_dict is False:
            return (native.loss, native.logits) if native.loss is not None else (native.logits,)
        return Seq2SeqLMOutput(loss=native.loss, logits=native.logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        if attention_mask is None:
            attention_mask = input_ids.ne(int(self.config.pad_token_id))
        encoder_features = {
            name: kwargs[name]
            for name in (
                "src_script_ids",
                "src_onset_ids",
                "src_vowel_ids",
                "src_coda_ids",
            )
            if name in kwargs
        }
        return self.model.generate(
            input_ids,
            attention_mask,
            bos_id=int(self.config.decoder_start_token_id),
            eos_id=int(self.config.eos_token_id),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            **encoder_features,
        )
