"""Transformers configuration for the Sion encoder-decoder architecture."""

# Transformers configuration dictionaries intentionally carry arbitrary JSON.
# pyright: reportArgumentType=false, reportCallIssue=false, reportInvalidTypeForm=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import re
from typing import Any

from transformers import PretrainedConfig


_TRANSLATION_PIPELINE_SCHEMA = "sion-translation-pipeline-v2"
_FOUNDATION_LINEAGE_SCHEMA = "sion-foundation-lineage-v1"
_LOWERCASE_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

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
        revision_directions: list[list[str]] | tuple[tuple[str, str], ...] | None = None,
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
        if translation_directions is None and self.language_pairs:
            raise ValueError(
                "translation configs with language pairs require explicit "
                "translation_directions; legacy directionality is unknown"
            )
        if translation_directions is None:
            self.translation_directions = []
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
        self.revision_directions: list[list[str]] | None = (
            None if revision_directions is None else []
        )
        for raw_direction in revision_directions or ():
            if isinstance(raw_direction, (str, bytes)) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                raw_direction, Sequence
            ):
                raise ValueError("each revision direction must be a two-item language sequence")
            assert self.revision_directions is not None
            self.revision_directions.append(
                list(
                    canonicalize_language_pair(
                        raw_direction,
                        field="config revision direction",
                    )
                )
            )
        if revision_trained is not None and not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            revision_trained, bool
        ):
            raise ValueError("revision_trained must be a boolean or null")
        if self.revision_directions is not None:
            expected_revision_trained = bool(self.revision_directions)
            if revision_trained is not None and revision_trained is not expected_revision_trained:
                raise ValueError("revision_trained disagrees with revision_directions")
            self.revision_trained = expected_revision_trained
        else:
            # Bool-only configs are legacy. Keep the uncertainty visible so a
            # multi-edge checkpoint cannot silently authorize every edge.
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
        self._validate_pipeline_identity()
        if not self.translation_capable and (
            self.revision_trained is True or bool(self.revision_directions)
        ):
            raise ValueError("translation-incapable configs cannot advertise revision capability")
        current_capability_contract = bool(
            getattr(self, "pipeline", None) is not None
            or (
                isinstance(self.release_version, str)
                and re.fullmatch(
                    r"[0-9]+\.[0-9]+(?:\.[0-9]+)?",
                    self.release_version,
                )
                and tuple(int(part) for part in self.release_version.split("."))[:2] >= (1, 5)
            )
        )
        if current_capability_contract and self.revision_directions is None:
            raise ValueError(
                "current configs require explicit revision_directions; use an empty list "
                "when revision was not trained"
            )
        if current_capability_contract and self.translation_capable and not self.language_pairs:
            raise ValueError(
                "current translation-capable configs require a non-empty language-pair graph"
            )
        if current_capability_contract and not self.translation_capable and not self.languages:
            raise ValueError("current foundation configs require a non-empty trained language list")
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
        if current_capability_contract and self.translation_capable:
            pair_languages = {language for pair in self.language_pairs for language in pair}
            if set(self.languages) != pair_languages:
                raise ValueError(
                    "current translation config languages must exactly match the "
                    "language-pair graph"
                )
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
        if self.revision_directions is not None:
            seen_revision_directions: set[tuple[str, str]] = set()
            for direction in self.revision_directions:
                key = tuple(direction)
                if key in seen_revision_directions:
                    raise ValueError(f"duplicate revision direction: {direction!r}")
                if key not in seen_directions:
                    raise ValueError(
                        "revision_directions must be a subset of translation_directions: "
                        f"{direction!r}"
                    )
                seen_revision_directions.add(key)
        covered_edges = {frozenset(direction) for direction in seen_directions}
        missing_pairs = [
            pair for pair in self.language_pairs if frozenset(pair) not in covered_edges
        ]
        if missing_pairs:
            raise ValueError(
                "every language pair must have at least one translation direction: "
                f"missing={missing_pairs!r}"
            )

    def _validate_pipeline_identity(self) -> None:
        pipeline = getattr(self, "pipeline", None)
        if pipeline is None:
            if (
                self.translation_capable
                and isinstance(self.release_version, str)
                and re.fullmatch(
                    r"[0-9]+\.[0-9]+(?:\.[0-9]+)?",
                    self.release_version,
                )
                and tuple(int(part) for part in self.release_version.split("."))[:2] >= (1, 5)
            ):
                raise ValueError(
                    "translation-capable release 1.5 or newer requires pipeline identity"
                )
            return
        if self.release_name is None or self.release_version is None:
            raise ValueError("pipeline identity requires an explicit release identity")
        if not self.translation_capable:
            raise ValueError("foundation-only configs must not contain pipeline identity")
        if not isinstance(pipeline, Mapping):
            raise ValueError("pipeline identity must be a JSON object")
        if pipeline.get("schema") != _TRANSLATION_PIPELINE_SCHEMA:
            raise ValueError(f"pipeline.schema must be {_TRANSLATION_PIPELINE_SCHEMA!r}")
        branch = pipeline.get("branch")
        if branch == "translation-only":
            if set(pipeline) != {"schema", "branch"}:
                raise ValueError(
                    "translation-only pipeline identity must contain exactly schema and branch"
                )
            return
        if branch != "foundation-then-translation":
            raise ValueError("pipeline.branch is unsupported")
        if set(pipeline) != {"schema", "branch", "foundation"}:
            raise ValueError(
                "foundation pipeline identity must contain exactly schema, branch, and foundation"
            )
        foundation = pipeline.get("foundation")
        if not isinstance(foundation, Mapping):
            raise ValueError("pipeline.foundation must be a JSON object")
        expected_fields = {
            "schema",
            "release_name",
            "release_version",
            "languages",
            "selected_step",
            "foundation_manifest_sha256",
            "tokenizer_sha256",
            "checkpoint_identity_sha256",
            "checkpoint_artifact_sha256",
        }
        if set(foundation) != expected_fields:
            raise ValueError(
                "foundation lineage must contain exactly its schema, release identity, "
                "languages, selected step, and four SHA-256 digests"
            )
        if foundation.get("schema") != _FOUNDATION_LINEAGE_SCHEMA:
            raise ValueError(f"foundation lineage schema must be {_FOUNDATION_LINEAGE_SCHEMA!r}")
        foundation_release_name = foundation.get("release_name")
        if (
            not isinstance(foundation_release_name, str)
            or not foundation_release_name
            or foundation_release_name != foundation_release_name.strip()
            or not foundation_release_name.isascii()
        ):
            raise ValueError("foundation lineage release_name must be normalized non-empty ASCII")
        if foundation_release_name == self.release_name:
            raise ValueError(
                "foundation lineage release_name must differ from the translation release name"
            )
        if foundation.get("release_version") != self.release_version:
            raise ValueError(
                "foundation lineage release_version must match the translation release"
            )
        raw_languages = foundation.get("languages")
        if (
            not isinstance(raw_languages, list)
            or not raw_languages
            or any(not isinstance(language, str) for language in raw_languages)
        ):
            raise ValueError(
                "foundation lineage languages must be a non-empty list of unique "
                "normalized BCP 47 tags"
            )
        try:
            canonical_languages = list(
                canonicalize_language_tags(
                    raw_languages,
                    field="foundation lineage languages",
                    reject_duplicates=True,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "foundation lineage languages must be a non-empty list of unique "
                "normalized BCP 47 tags"
            ) from error
        if raw_languages != canonical_languages:
            raise ValueError(
                "foundation lineage languages must be a non-empty list of unique "
                "normalized BCP 47 tags"
            )
        selected_step = foundation.get("selected_step")
        if (
            isinstance(selected_step, bool)
            or not isinstance(selected_step, int)
            or selected_step < 0
        ):
            raise ValueError("foundation lineage selected_step must be a non-negative integer")
        for field_name in (
            "foundation_manifest_sha256",
            "tokenizer_sha256",
            "checkpoint_identity_sha256",
            "checkpoint_artifact_sha256",
        ):
            digest = foundation.get(field_name)
            if not isinstance(digest, str) or _LOWERCASE_SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError(
                    f"foundation lineage {field_name} must be a lowercase SHA-256 digest"
                )
        if self.tokenizer_sha256 != foundation.get("tokenizer_sha256"):
            raise ValueError(
                "foundation lineage tokenizer_sha256 must exactly match tokenizer_sha256"
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
        revision_directions: list[list[str]] | tuple[tuple[str, str], ...] | None = None,
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
            revision_directions=revision_directions,
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
