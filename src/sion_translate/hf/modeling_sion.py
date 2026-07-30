"""Transformers model wrapper around the native Sion implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.generation.utils import GenerateEncoderDecoderOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

try:
    from sion_translate.model import (
        SionForConditionalGeneration as NativeSionForConditionalGeneration,
    )
except ImportError:
    # Remote checkpoints contain the native runtime under relative module names,
    # so only torch/transformers are needed when sion-translate is not installed.
    from importlib import import_module

    NativeSionForConditionalGeneration = import_module(
        f"{__package__}.sion_native_transformer"
    ).SionForConditionalGeneration

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
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput | tuple[torch.Tensor, ...]:
        if attention_mask is None:
            attention_mask = input_ids.ne(int(self.config.pad_token_id))
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("decoder_input_ids or labels must be provided")
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
        del kwargs
        supported = {
            name: value
            for name, value in (
                ("register_labels", register_labels),
                ("memory_token_ids", memory_token_ids),
                ("memory_mask", memory_mask),
                ("memory_type_ids", memory_type_ids),
                ("memory_mode_ids", memory_mode_ids),
                ("src_script_ids", src_script_ids),
                ("src_onset_ids", src_onset_ids),
                ("src_vowel_ids", src_vowel_ids),
                ("src_coda_ids", src_coda_ids),
                ("alignment_targets", alignment_targets),
            )
            if value is not None
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
    def _beam_generate_multiple(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences: int,
        length_penalty: float,
        native_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return multiple ranked hypotheses using the native cached decoder."""

        model = self.model
        was_training = model.training
        model.eval()
        try:
            encoder_features = {
                name: value
                for name, value in native_inputs.items()
                if name
                in {
                    "src_script_ids",
                    "src_onset_ids",
                    "src_vowel_ids",
                    "src_coda_ids",
                }
            }
            memory = {
                name: value
                for name, value in native_inputs.items()
                if name
                in {
                    "memory_token_ids",
                    "memory_mask",
                    "memory_type_ids",
                    "memory_mode_ids",
                }
            }
            encoder_states = model.encode(input_ids, attention_mask, **encoder_features)
            register_context = None
            if model.register_state is not None:
                _, register_context, _ = model.register_state(
                    encoder_states,
                    attention_mask,
                    register_labels=None,
                )

            batch = encoder_states.shape[0]
            device = encoder_states.device
            total = batch * num_beams
            encoder_states = encoder_states.repeat_interleave(num_beams, dim=0)
            source_mask = attention_mask.repeat_interleave(num_beams, dim=0)
            if register_context is not None:
                register_context = register_context.repeat_interleave(num_beams, dim=0)
            memory = {
                name: value.repeat_interleave(num_beams, dim=0) for name, value in memory.items()
            }

            bos_id = int(self.config.decoder_start_token_id)
            eos_id = int(self.config.eos_token_id)
            caches = model._fresh_caches(len(model.decoder_layers))
            sequences = torch.full((total, 1), bos_id, dtype=torch.long, device=device)
            beam_scores = torch.full((batch, num_beams), float("-inf"), device=device)
            beam_scores[:, 0] = 0.0
            done: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(batch)]

            def penalized(raw_score: float, length: int) -> float:
                return raw_score / (((5.0 + length) / 6.0) ** length_penalty)

            for position in range(max_new_tokens):
                hidden = model._decoder_step(
                    sequences[:, -1:],
                    encoder_states,
                    source_mask,
                    caches,
                    position,
                    register_context,
                    **memory,
                )
                log_probs = torch.log_softmax(model._logits(hidden[:, -1]).float(), dim=-1)
                vocab = log_probs.shape[-1]
                candidates = (beam_scores.view(-1, 1) + log_probs).view(
                    batch,
                    num_beams * vocab,
                )
                top_scores, top_indices = candidates.topk(2 * num_beams, dim=-1)
                source_beams = top_indices // vocab
                new_tokens = top_indices % vocab
                next_scores = torch.full_like(beam_scores, float("-inf"))
                gather_flat = torch.zeros(
                    batch,
                    num_beams,
                    dtype=torch.long,
                    device=device,
                )
                step_tokens = torch.full(
                    (batch, num_beams),
                    eos_id,
                    dtype=torch.long,
                    device=device,
                )
                for batch_index in range(batch):
                    slot = 0
                    for candidate_index in range(2 * num_beams):
                        score = float(top_scores[batch_index, candidate_index])
                        if score == float("-inf"):
                            continue
                        token = int(new_tokens[batch_index, candidate_index])
                        flat_source = batch_index * num_beams + int(
                            source_beams[batch_index, candidate_index]
                        )
                        if token == eos_id:
                            finished = torch.cat(
                                (
                                    sequences[flat_source],
                                    torch.tensor([eos_id], device=device),
                                )
                            )
                            done[batch_index].append((penalized(score, position + 1), finished))
                        elif slot < num_beams:
                            next_scores[batch_index, slot] = score
                            gather_flat[batch_index, slot] = flat_source
                            step_tokens[batch_index, slot] = token
                            slot += 1

                flat_index = gather_flat.view(-1)
                sequences = torch.cat(
                    (
                        sequences.index_select(0, flat_index),
                        step_tokens.view(-1, 1),
                    ),
                    dim=1,
                )
                for cache in caches:
                    cache["self"] = tuple(
                        tensor.index_select(0, flat_index) for tensor in cache["self"]
                    )
                    cache["cross"] = tuple(
                        tensor.index_select(0, flat_index) for tensor in cache["cross"]
                    )
                beam_scores = next_scores

                all_done = True
                for batch_index in range(batch):
                    if len(done[batch_index]) < num_beams:
                        all_done = False
                        break
                    best_possible = penalized(
                        float(beam_scores[batch_index].max()),
                        max_new_tokens if length_penalty > 0 else position + 1,
                    )
                    worst_kept = min(score for score, _ in done[batch_index])
                    if best_possible > worst_kept:
                        all_done = False
                        break
                if all_done:
                    break

            outputs: list[torch.Tensor] = []
            generated_length = max(1, sequences.shape[1] - 1)
            for batch_index in range(batch):
                hypotheses = list(done[batch_index])
                for beam_index in range(num_beams):
                    raw_score = float(beam_scores[batch_index, beam_index])
                    if raw_score == float("-inf"):
                        continue
                    sequence = sequences[batch_index * num_beams + beam_index]
                    hypotheses.append(
                        (
                            penalized(raw_score, generated_length),
                            torch.cat((sequence, torch.tensor([eos_id], device=device))),
                        )
                    )
                hypotheses.sort(key=lambda item: item[0], reverse=True)
                outputs.extend(sequence for _, sequence in hypotheses[:num_return_sequences])
            if len(outputs) != batch * num_return_sequences:
                raise RuntimeError("beam search did not produce enough hypotheses")
            longest = max(map(len, outputs))
            padded = torch.full(
                (len(outputs), longest),
                eos_id,
                dtype=torch.long,
                device=device,
            )
            for index, sequence in enumerate(outputs):
                padded[index, : len(sequence)] = sequence
            return padded
        finally:
            model.train(was_training)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        max_new_tokens: int | None = None,
        max_length: int | None = None,
        num_beams: int | None = None,
        length_penalty: float | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        num_return_sequences: int | None = None,
        return_dict_in_generate: bool | None = None,
        output_scores: bool | None = None,
        generator: torch.Generator | None = None,
        generation_config: GenerationConfig | None = None,
        synced_gpus: bool | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | GenerateEncoderDecoderOutput:
        inputs = kwargs.pop("inputs", None)
        if inputs is not None:
            if input_ids is not None:
                raise ValueError("pass only one of inputs or input_ids")
            input_ids = inputs
        if input_ids is None:
            raise ValueError("inputs or input_ids must be provided")
        if attention_mask is None:
            attention_mask = input_ids.ne(int(self.config.pad_token_id))
        active_generation_config = generation_config or self.generation_config
        explicit_temperature = temperature is not None
        explicit_top_k = top_k is not None
        unsupported_non_neutral = {
            name: getattr(active_generation_config, name)
            for name, neutral in (
                ("top_p", 1.0),
                ("typical_p", 1.0),
                ("epsilon_cutoff", 0.0),
                ("eta_cutoff", 0.0),
                ("repetition_penalty", 1.0),
                ("encoder_repetition_penalty", 1.0),
                ("no_repeat_ngram_size", 0),
                ("diversity_penalty", 0.0),
                ("num_beam_groups", 1),
                ("min_length", 0),
                ("early_stopping", False),
                ("remove_invalid_values", False),
                ("renormalize_logits", False),
                ("token_healing", False),
                ("output_attentions", False),
                ("output_hidden_states", False),
                ("output_logits", False),
                ("return_legacy_cache", False),
            )
            if hasattr(active_generation_config, name)
            and getattr(active_generation_config, name) not in (None, neutral)
        }
        for name in (
            "min_p",
            "penalty_alpha",
            "bad_words_ids",
            "force_words_ids",
            "constraints",
            "min_new_tokens",
            "forced_bos_token_id",
            "forced_eos_token_id",
            "exponential_decay_length_penalty",
            "suppress_tokens",
            "begin_suppress_tokens",
            "sequence_bias",
            "guidance_scale",
            "watermarking_config",
            "stop_strings",
            "max_time",
            "cache_implementation",
            "cache_config",
        ):
            value = getattr(active_generation_config, name, None)
            if value not in (None, 0, False, (), [], {}):
                unsupported_non_neutral[name] = value
        if unsupported_non_neutral:
            raise NotImplementedError(
                "unsupported Sion generation_config options: "
                + ", ".join(sorted(unsupported_non_neutral))
            )

        def configured(value: Any, name: str, fallback: Any) -> Any:
            if value is not None:
                return value
            configured_value = getattr(active_generation_config, name, None)
            return fallback if configured_value is None else configured_value

        num_beams = int(configured(num_beams, "num_beams", 1))
        length_penalty = float(configured(length_penalty, "length_penalty", 1.0))
        do_sample = bool(configured(do_sample, "do_sample", False))
        temperature = float(configured(temperature, "temperature", 1.0))
        top_k = int(configured(top_k, "top_k", 0))
        num_return_sequences = int(configured(num_return_sequences, "num_return_sequences", 1))
        return_dict_in_generate = bool(
            configured(return_dict_in_generate, "return_dict_in_generate", False)
        )
        output_scores = bool(configured(output_scores, "output_scores", False))
        if output_scores:
            raise NotImplementedError(
                "Sion native generation does not expose per-step scores; "
                "output_scores=True is unsupported"
            )
        if synced_gpus:
            raise NotImplementedError(
                "synced_gpus=True is unsupported by Sion native generation; "
                "run prediction with synchronized input counts instead"
            )
        unsupported = {
            name
            for name in (
                "top_p",
                "min_p",
                "typical_p",
                "epsilon_cutoff",
                "eta_cutoff",
                "penalty_alpha",
                "repetition_penalty",
                "encoder_repetition_penalty",
                "no_repeat_ngram_size",
                "bad_words_ids",
                "force_words_ids",
                "constraints",
                "prefix_allowed_tokens_fn",
                "logits_processor",
                "stopping_criteria",
                "assistant_model",
                "streamer",
                "watermarking_config",
            )
            if name in kwargs
        }
        if unsupported:
            raise NotImplementedError(
                "unsupported Sion generation options: " + ", ".join(sorted(unsupported))
            )

        for name, expected in (
            ("bos_token_id", int(self.config.bos_token_id)),
            ("decoder_start_token_id", int(self.config.decoder_start_token_id)),
            ("eos_token_id", int(self.config.eos_token_id)),
            ("pad_token_id", int(self.config.pad_token_id)),
        ):
            value = (
                kwargs.pop(name)
                if name in kwargs
                else getattr(active_generation_config, name, None)
            )
            if value is not None:
                if isinstance(value, (list, tuple)) or int(value) != expected:
                    raise ValueError(
                        f"{name}={value!r} does not match the checkpoint value {expected}"
                    )
        use_cache = (
            kwargs.pop("use_cache")
            if "use_cache" in kwargs
            else getattr(active_generation_config, "use_cache", True)
        )
        if use_cache is False:
            raise NotImplementedError("use_cache=False is unsupported by Sion native generation")
        for no_op_name in ("output_logits", "return_legacy_cache"):
            if no_op_name in kwargs and bool(kwargs.pop(no_op_name)):
                raise NotImplementedError(f"{no_op_name}=True is unsupported")
        for no_op_name in ("output_attentions", "output_hidden_states"):
            if no_op_name in kwargs and bool(kwargs.pop(no_op_name)):
                raise NotImplementedError(f"{no_op_name}=True is unsupported")
        # Seq2SeqTrainer forwards labels to generate during prediction. They
        # constrain neither encoder input nor free-running decoder generation.
        kwargs.pop("labels", None)

        if max_new_tokens is not None and max_length is not None:
            raise ValueError("set only one of max_new_tokens or max_length")
        if max_new_tokens is None and max_length is None:
            configured_new_tokens = getattr(active_generation_config, "max_new_tokens", None)
            if configured_new_tokens is not None:
                max_new_tokens = int(configured_new_tokens)
            else:
                configured_max_length = getattr(active_generation_config, "max_length", None)
                max_length = 20 if configured_max_length is None else int(configured_max_length)
        if max_length is not None:
            if max_length < 2:
                raise ValueError("max_length must be at least 2 because output starts with BOS")
            max_new_tokens = max_length - 1
        assert max_new_tokens is not None
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if max_new_tokens > self.model.config.max_seq_len:
            raise ValueError(
                "max_new_tokens exceeds model max_seq_len: "
                f"{max_new_tokens} > {self.model.config.max_seq_len}"
            )
        if num_beams < 1:
            raise ValueError("num_beams must be positive")
        if num_return_sequences < 1:
            raise ValueError("num_return_sequences must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if do_sample and num_beams != 1:
            raise NotImplementedError("Sion does not implement beam sampling; set num_beams=1")
        if not do_sample and num_return_sequences > num_beams:
            raise ValueError(
                "num_return_sequences cannot exceed num_beams for deterministic generation"
            )
        if not do_sample and num_beams == 1 and num_return_sequences != 1:
            raise ValueError("greedy generation supports only num_return_sequences=1")
        if not do_sample and (
            (explicit_temperature and temperature != 1.0) or (explicit_top_k and top_k != 0)
        ):
            raise ValueError("temperature and top_k only apply when do_sample=True")
        if do_sample and length_penalty != 1.0:
            raise ValueError("length_penalty only applies to deterministic beam search")

        native_input_names = (
            "src_script_ids",
            "src_onset_ids",
            "src_vowel_ids",
            "src_coda_ids",
            "memory_token_ids",
            "memory_mask",
            "memory_type_ids",
            "memory_mode_ids",
        )
        native_inputs = {
            name: kwargs[name]
            for name in native_input_names
            if name in kwargs and kwargs[name] is not None
        }
        for name in native_input_names:
            kwargs.pop(name, None)
        if kwargs:
            raise TypeError("unexpected Sion generation options: " + ", ".join(sorted(kwargs)))

        if do_sample:
            sampled = self.model.sample(
                input_ids,
                attention_mask,
                bos_id=int(self.config.decoder_start_token_id),
                eos_id=int(self.config.eos_token_id),
                num_samples=num_return_sequences,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
                **native_inputs,
            )
            sequences = sampled.reshape(-1, sampled.shape[-1])
        elif num_return_sequences > 1:
            sequences = self._beam_generate_multiple(
                input_ids,
                attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                native_inputs=native_inputs,
            )
        else:
            sequences = self.model.generate(
                input_ids,
                attention_mask,
                bos_id=int(self.config.decoder_start_token_id),
                eos_id=int(self.config.eos_token_id),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                **native_inputs,
            )
        sequences = sequences[:, : max_new_tokens + 1]
        if return_dict_in_generate:
            return GenerateEncoderDecoderOutput(sequences=sequences)
        return sequences
