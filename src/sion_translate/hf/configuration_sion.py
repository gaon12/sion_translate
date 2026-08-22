"""Transformers configuration for the Sion encoder-decoder architecture."""

# Transformers configuration dictionaries intentionally carry arbitrary JSON.
# pyright: reportArgumentType=false, reportCallIssue=false, reportInvalidTypeForm=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import re
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

try:
    from sion_translate.language_tags import (
        canonicalize_language_pair,
        canonicalize_language_tags,
    )
except ImportError:
    from .sion_language_tags import (  # type: ignore[import-not-found]
        canonicalize_language_pair,
        canonicalize_language_tags,
    )


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
        release_name: str | None = None,
        release_version: str | None = None,
        translation_capable: bool = True,
        revision_trained: bool | None = None,
        default_reasoning_level: int | None = None,
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
        experimental_config: ExperimentalConfig = (
            experimental
            if isinstance(experimental, ExperimentalConfig)
            else ExperimentalConfig(**dict(experimental or {}))
        )
        self.experimental = experimental_config
        self.language_pairs: list[list[str]] = []
        for raw_pair in language_pairs or ():
            if isinstance(raw_pair, (str, bytes)) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                raw_pair, Sequence
            ):
                raise ValueError("each language pair must be a two-item language sequence")
            self.language_pairs.append(
                list(
                    canonicalize_language_pair(
                        raw_pair,
                        field="config language pair",
                    )
                )
            )
        configured_languages = (
            list(languages)
            if languages is not None
            else [language for pair in self.language_pairs for language in pair]
        )
        self.languages = list(
            canonicalize_language_tags(
                configured_languages,
                field="config languages",
                reject_duplicates=False,
            )
        )
        self.release_name = release_name
        self.release_version = release_version
        current_direction_contract = bool(
            kwargs.get("pipeline") is not None
            or (
                isinstance(release_version, str)
                and re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", release_version)
                and tuple(int(part) for part in release_version.split("."))[:2] >= (1, 5)
            )
        )
        if translation_directions is None and self.language_pairs and current_direction_contract:
            raise ValueError(
                "current translation configs with language pairs require explicit "
                "translation_directions"
            )
        if translation_directions is None:
            self.translation_directions = [
                list(direction)
                for pair in self.language_pairs
                for direction in (pair, list(reversed(pair)))
            ]
        else:
            self.translation_directions = []
            for raw_direction in translation_directions:
                if isinstance(raw_direction, (str, bytes)) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                    raw_direction, Sequence
                ):
                    raise ValueError(
                        "each translation direction must be a two-item language sequence"
                    )
                self.translation_directions.append(
                    list(
                        canonicalize_language_pair(
                            raw_direction,
                            field="config translation direction",
                        )
                    )
                )
        if not isinstance(translation_capable, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("translation_capable must be a boolean")
        self.translation_capable = translation_capable
        if revision_trained is not None and not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            revision_trained, bool
        ):
            raise ValueError("revision_trained must be a boolean or null")
        self.revision_trained = revision_trained
        if default_reasoning_level is None:
            default_reasoning_level = (
                9
                if self.experimental.evidence_repair_enabled
                or self.experimental.candidate_refinement_enabled
                else 0
            )
        self.default_reasoning_level = default_reasoning_level
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
        release_identity = (self.release_name, self.release_version)
        if (self.release_name is None) != (self.release_version is None):
            raise ValueError("release_name and release_version must be provided together")
        if any(
            value is not None
            and (
                not isinstance(value, str)  # pyright: ignore[reportUnnecessaryIsInstance]
                or not value.strip()
            )
            for value in release_identity
        ):
            raise ValueError("release_name and release_version must be non-empty strings")
        if (
            self.release_version is not None
            and re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", self.release_version) is None
        ):
            raise ValueError("release_version must use a numeric major.minor[.patch] value")
        if self.release_name == "sion" and self.translation_capable:
            raise ValueError("the sion foundation release cannot be translation-capable")
        if self.release_name == "sion_translate" and not self.translation_capable:
            raise ValueError("the sion_translate release must be translation-capable")
        if isinstance(self.default_reasoning_level, bool) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.default_reasoning_level,
            int,
        ):
            raise TypeError("default_reasoning_level must be an integer from 0 to 9")
        if not 0 <= self.default_reasoning_level <= 9:
            raise ValueError("default_reasoning_level must be between 0 and 9")
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
            edge = frozenset(pair)
            if edge in allowed_edges:
                raise ValueError(
                    f"duplicate or reversed language pair after BCP 47 canonicalization: {pair!r}"
                )
            allowed_edges.add(edge)
        seen_directions: set[tuple[str, str]] = set()
        if self.language_pairs and not self.translation_directions:
            raise ValueError(
                "translation_directions cannot be empty when language pairs are configured"
            )
        if not self.translation_capable and (self.language_pairs or self.translation_directions):
            raise ValueError(
                "translation-incapable configs cannot advertise language pairs or directions"
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
        covered_edges = {frozenset(direction) for direction in seen_directions}
        missing_pairs = [
            pair for pair in self.language_pairs if frozenset(pair) not in covered_edges
        ]
        if missing_pairs:
            raise ValueError(
                "every language pair must have at least one translation direction: "
                f"missing={missing_pairs!r}"
            )

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
        release_name: str | None = None,
        release_version: str | None = None,
        translation_capable: bool = True,
        revision_trained: bool | None = None,
        default_reasoning_level: int | None = None,
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
            release_name=release_name,
            release_version=release_version,
            translation_capable=translation_capable,
            revision_trained=revision_trained,
            default_reasoning_level=default_reasoning_level,
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
