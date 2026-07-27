"""Transformers configuration for the Sion encoder-decoder architecture."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from transformers import PretrainedConfig

from sion_translate.config import ExperimentalConfig, ModelConfig


class SionConfig(PretrainedConfig):
    """Serializable Transformers counterpart of :class:`ModelConfig`."""

    model_type = "sion"
    keys_to_ignore_at_inference = [
        "lm_loss_sum",
        "token_count",
        "auxiliary_loss",
        "register_loss",
        "alignment_loss",
        "coverage_loss",
    ]

    def __init__(
        self,
        vocab_size: int = 48000,
        d_model: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        d_ff: int = 1536,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        dropout: float = 0.1,
        rms_norm_eps: float = 1e-6,
        qk_norm: bool = True,
        tie_embeddings: bool = True,
        label_smoothing: float = 0.10,
        z_loss_weight: float = 1e-4,
        gradient_checkpointing: bool = False,
        init_std: float = 0.02,
        experimental: ExperimentalConfig | dict[str, Any] | None = None,
        languages: list[str] | tuple[str, ...] | None = None,
        language_pairs: list[list[str]] | tuple[tuple[str, str], ...] | None = None,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        decoder_start_token_id: int | None = None,
        **kwargs: Any,
    ):
        is_encoder_decoder = bool(kwargs.pop("is_encoder_decoder", True))
        serialized_tie_embeddings = bool(kwargs.pop("tie_word_embeddings", tie_embeddings))
        if not is_encoder_decoder:
            raise ValueError("SionConfig only supports encoder-decoder models")
        if serialized_tie_embeddings != tie_embeddings:
            raise ValueError("tie_embeddings and tie_word_embeddings must have the same value")
        decoder_start_token_id = (
            bos_token_id if decoder_start_token_id is None else decoder_start_token_id
        )
        # Transformers 5 validates token IDs from inside ``PretrainedConfig.__init__``
        # and that validation calls our ``to_dict`` before ``super().__init__`` returns.
        # Populate every Sion-specific field first so this partially constructed state
        # is still serializable.
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        self.qk_norm = qk_norm
        self.tie_embeddings = tie_embeddings
        self.label_smoothing = label_smoothing
        self.z_loss_weight = z_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        self.init_std = init_std
        if isinstance(experimental, ExperimentalConfig):
            self.experimental = experimental
        else:
            self.experimental = ExperimentalConfig(**dict(experimental or {}))
        self.languages = list(languages or [])
        self.language_pairs = [list(pair) for pair in (language_pairs or [])]
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=serialized_tie_embeddings,
            **kwargs,
        )
        self.validate()

    def validate(self) -> None:
        self.to_model_config().validate()
        if self.pad_token_id is None or self.pad_token_id < 0:
            raise ValueError("pad_token_id must be a non-negative integer")
        for name in ("bos_token_id", "eos_token_id", "decoder_start_token_id"):
            value = getattr(self, name)
            if value is None or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            rope_base=self.rope_base,
            dropout=self.dropout,
            rms_norm_eps=self.rms_norm_eps,
            qk_norm=self.qk_norm,
            tie_embeddings=self.tie_embeddings,
            label_smoothing=self.label_smoothing,
            z_loss_weight=self.z_loss_weight,
            gradient_checkpointing=self.gradient_checkpointing,
            init_std=self.init_std,
            experimental=ExperimentalConfig(**asdict(self.experimental)),
        )

    @classmethod
    def from_model_config(
        cls,
        config: ModelConfig,
        *,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        languages: list[str] | tuple[str, ...] | None = None,
        language_pairs: list[list[str]] | tuple[tuple[str, str], ...] | None = None,
        **kwargs: Any,
    ) -> SionConfig:
        values = asdict(config)
        return cls(
            **values,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            languages=languages,
            language_pairs=language_pairs,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["experimental"] = asdict(self.experimental)
        output["architectures"] = ["SionForConditionalGeneration"]
        return output
