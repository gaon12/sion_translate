"""Transformers configuration for the Sion encoder-decoder architecture."""

# Transformers configuration dictionaries intentionally carry arbitrary JSON.
# pyright: reportArgumentType=false, reportCallIssue=false, reportInvalidTypeForm=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from transformers import PretrainedConfig

try:
    from sion_translate.config import ExperimentalConfig, ModelConfig
except ImportError:
    # ``save_transformers_checkpoint`` writes this small runtime module next to
    # the remote-code files.  Keeping the fallback in a relative import lets a
    # Hub checkpoint load without installing the Sion source package.
    from importlib import import_module

    _native_config = import_module(f"{__package__}.sion_native_config")
    ExperimentalConfig = _native_config.ExperimentalConfig
    ModelConfig = _native_config.ModelConfig


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
        translation_directions: list[list[str]] | tuple[tuple[str, str], ...] | None = None,
        revision_trained: bool | None = None,
        slot_token_ids: list[int] | tuple[int, ...] | None = None,
        tokenizer_sha256: str | None = None,
        token_features_sha256: str | None = None,
        token_features_shapes: dict[str, list[int] | tuple[int, ...]] | None = None,
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
        self.language_pairs = [list(pair) for pair in (language_pairs or [])]
        self.languages = list(
            languages
            or dict.fromkeys(language for pair in self.language_pairs for language in pair)
        )
        self.translation_directions = (
            [list(direction) for direction in translation_directions]
            if translation_directions is not None
            else [
                list(direction)
                for pair in self.language_pairs
                for direction in (pair, list(reversed(pair)))
            ]
        )
        if revision_trained is not None and not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            revision_trained, bool
        ):
            raise ValueError("revision_trained must be a boolean or null")
        self.revision_trained = revision_trained
        self.slot_token_ids = [int(token_id) for token_id in (slot_token_ids or [])]
        self.tokenizer_sha256 = tokenizer_sha256
        self.token_features_sha256 = token_features_sha256
        self.token_features_shapes = {
            str(name): [int(dimension) for dimension in shape]
            for name, shape in (token_features_shapes or {}).items()
        }
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
        if len(self.slot_token_ids) > 64:
            raise ValueError("slot_token_ids may contain at most 64 protected slot IDs")
        if len(set(self.slot_token_ids)) != len(self.slot_token_ids):
            raise ValueError("slot_token_ids must not contain duplicates")
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in self.slot_token_ids):
            raise ValueError("slot_token_ids must be valid vocabulary IDs")
        required_features = {"script", "onset", "vowel", "coda"}
        if self.token_features_shapes and set(self.token_features_shapes) != required_features:
            raise ValueError(
                "token_features_shapes must contain exactly script, onset, vowel, and coda"
            )
        for name, shape in self.token_features_shapes.items():
            if shape != [self.vocab_size]:
                raise ValueError(
                    f"token feature {name} shape must be [{self.vocab_size}], got {shape}"
                )
        allowed_edges: set[frozenset[str]] = set()
        for pair in self.language_pairs:
            if (
                len(pair) != 2
                or pair[0] == pair[1]
                or any(language not in self.languages for language in pair)
            ):
                raise ValueError(f"invalid language pair: {pair!r}")
            allowed_edges.add(frozenset(pair))
        seen_directions: set[tuple[str, str]] = set()
        if self.language_pairs and not self.translation_directions:
            raise ValueError(
                "translation_directions cannot be empty when language pairs are configured"
            )
        for direction in self.translation_directions:
            key = tuple(direction)
            if (
                len(direction) != 2
                or direction[0] == direction[1]
                or frozenset(direction) not in allowed_edges
            ):
                raise ValueError(f"invalid translation direction: {direction!r}")
            if key in seen_directions:
                raise ValueError(f"duplicate translation direction: {direction!r}")
            seen_directions.add(key)

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
        translation_directions: list[list[str]] | tuple[tuple[str, str], ...] | None = None,
        revision_trained: bool | None = None,
        slot_token_ids: list[int] | tuple[int, ...] | None = None,
        tokenizer_sha256: str | None = None,
        token_features_sha256: str | None = None,
        token_features_shapes: dict[str, list[int] | tuple[int, ...]] | None = None,
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
            translation_directions=translation_directions,
            revision_trained=revision_trained,
            slot_token_ids=slot_token_ids,
            tokenizer_sha256=tokenizer_sha256,
            token_features_sha256=token_features_sha256,
            token_features_shapes=token_features_shapes,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["experimental"] = asdict(self.experimental)
        output["architectures"] = ["SionForConditionalGeneration"]
        return output
