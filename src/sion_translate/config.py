# YAML constructors and raw configuration objects are dynamically typed.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, cast, Sequence

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from sion_translate.artifacts import (
    DEFAULT_DATASET_DIRECTORY,
    DEFAULT_FOUNDATION_DATASET_DIRECTORY,
    DEFAULT_MONOLINGUAL_CORPUS_DIRECTORY,
    DEFAULT_RUN_DIRECTORY,
    DEFAULT_TOKENIZER_FEATURES,
    DEFAULT_TOKENIZER_MODEL,
    FOUNDATION_RELEASE_NAME,
    TRANSLATION_RELEASE_NAME,
)
from sion_translate.data.monolingual import (
    DEFAULT_LANGUAGE_SAMPLING_ALPHA,
    foundation_languages,
)
from sion_translate.data.records import (
    normalize_language_pairs,
    normalize_translation_directions,
)
from sion_translate.generation import (
    DEFAULT_LENGTH_PENALTY,
    DEFAULT_MAX_OUTPUT_LENGTH_MARGIN,
    DEFAULT_MAX_OUTPUT_LENGTH_RATIO,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_NUM_BEAMS,
)
from sion_translate.language_tags import canonicalize_language_pair, canonicalize_language_tags
from sion_translate.synthetic import (
    DEFAULT_SYNTHETIC_PREFIXES,
    DEFAULT_SYNTHETIC_SAMPLING_WEIGHT,
    normalize_synthetic_prefixes,
)

DEFAULT_CANDIDATE_REFINEMENT_MIN_WORST_DIRECTION_NLL_GAIN = 1e-5


@dataclass
class ExperimentalConfig:
    bats_enabled: bool = False
    bats_dim: int = 256
    bats_loss_weight: float = 0.0
    bats_coverage_weight: float = 0.0
    bats_stride: int = 2
    bats_max_positions: int = 1024
    core_enabled: bool = False
    register_classes: int = 4
    register_loss_weight: float = 0.05
    tetm_enabled: bool = False
    tetm_types: int = 10
    tetm_modes: int = 5
    morphoscript_enabled: bool = False
    morphoscript_interval: int = 4
    script_classes: int = 9
    # Let uncertain decoder positions reread encoder evidence. Uncertainty and
    # budget losses prevent the request rate from growing without bound.
    evidence_repair_enabled: bool = False
    evidence_uncertainty_loss_weight: float = 0.02
    evidence_budget_loss_weight: float = 0.001
    evidence_budget_target: float = 0.25
    evidence_repair_gain_loss_weight: float = 0.005
    evidence_minimum_gain: float = 0.01
    # Compress the first full next-token distribution into its expected token
    # embedding, feed it back into decoder state, and recompute all vocabulary
    # logits. The disabled default preserves module-free legacy checkpoints.
    candidate_refinement_enabled: bool = False
    candidate_refinement_steps: int = 1
    candidate_refinement_temperature: float = 1.0
    candidate_refinement_loss_weight: float = 0.25
    candidate_refinement_vocab_chunk_size: int = 2048
    # Contrast pooled source and target representations so meaning can remain
    # aligned even when their literal surface forms differ.
    semantic_parity_enabled: bool = False
    semantic_parity_dim: int = 256
    semantic_parity_temperature: float = 0.07
    semantic_parity_loss_weight: float = 0.05
    # Recurrent latent computation repeats the final N encoder layers with
    # shared weights. It adds hidden-state computation depth without adding
    # parameters or explicit reasoning tokens. Zero preserves legacy behavior.
    recurrent_block_layers: int = 0
    recurrent_steps: int = 1
    # Kimi K3 SiTU-GLU softly bounds SwiGLU gate/up pre-activations to reduce
    # large activations and low-precision overflow. Projection shapes remain
    # compatible with checkpoints that leave the feature disabled.
    situglu_enabled: bool = False
    situglu_gate_beta: float = 4.0
    situglu_up_beta: float = 25.0

    def validate(self) -> None:
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.candidate_refinement_enabled, bool
        ):
            raise ValueError("experimental.candidate_refinement_enabled must be a boolean")
        if isinstance(self.candidate_refinement_steps, bool) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.candidate_refinement_steps, int
        ):
            raise ValueError("experimental.candidate_refinement_steps must be an integer")
        if isinstance(self.candidate_refinement_vocab_chunk_size, bool) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.candidate_refinement_vocab_chunk_size, int
        ):
            raise ValueError(
                "experimental.candidate_refinement_vocab_chunk_size must be an integer"
            )
        for name, value in (
            ("candidate_refinement_temperature", self.candidate_refinement_temperature),
            ("candidate_refinement_loss_weight", self.candidate_refinement_loss_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))  # pyright: ignore[reportUnnecessaryIsInstance]
                or not math.isfinite(value)
            ):
                raise ValueError(f"experimental.{name} must be a finite real number")
        if self.recurrent_block_layers < 0:
            raise ValueError("experimental.recurrent_block_layers must be non-negative")
        if self.recurrent_steps < 1:
            raise ValueError("experimental.recurrent_steps must be at least 1")
        if not 1 <= self.candidate_refinement_steps <= 4:
            raise ValueError("experimental.candidate_refinement_steps must be between 1 and 4")
        if self.candidate_refinement_temperature <= 0:
            raise ValueError("experimental.candidate_refinement_temperature must be positive")
        if self.candidate_refinement_vocab_chunk_size <= 0:
            raise ValueError("experimental.candidate_refinement_vocab_chunk_size must be positive")
        if self.situglu_gate_beta <= 0 or self.situglu_up_beta <= 0:
            raise ValueError("experimental SiTU-GLU beta values must be positive")
        if self.recurrent_block_layers and self.recurrent_steps == 1:
            warnings.warn(
                "experimental.recurrent_block_layers is set, but recurrent_steps is 1. "
                "A single pass adds no recurrent computation. Set recurrent_steps to at "
                "least 2 or set recurrent_block_layers to 0.",
                RuntimeWarning,
                stacklevel=2,
            )
        for name, value in (
            ("bats_dim", self.bats_dim),
            ("bats_stride", self.bats_stride),
            ("bats_max_positions", self.bats_max_positions),
            ("register_classes", self.register_classes),
            ("tetm_types", self.tetm_types),
            ("tetm_modes", self.tetm_modes),
            ("morphoscript_interval", self.morphoscript_interval),
            ("script_classes", self.script_classes),
            ("semantic_parity_dim", self.semantic_parity_dim),
        ):
            if value <= 0:
                raise ValueError(f"experimental.{name} must be positive")
        if self.semantic_parity_temperature <= 0:
            raise ValueError("experimental.semantic_parity_temperature must be positive")
        if self.tetm_enabled and self.tetm_types < 9:
            raise ValueError(
                "experimental.tetm_types must be at least 9 when TETM is enabled "
                "because protected slots use type ID 8"
            )
        if self.tetm_enabled and self.tetm_modes < 5:
            raise ValueError(
                "experimental.tetm_modes must be at least 5 when TETM is enabled "
                "because protected slots use mode ID 4"
            )
        for name, value in (
            ("bats_loss_weight", self.bats_loss_weight),
            ("bats_coverage_weight", self.bats_coverage_weight),
            ("register_loss_weight", self.register_loss_weight),
            ("evidence_uncertainty_loss_weight", self.evidence_uncertainty_loss_weight),
            ("evidence_budget_loss_weight", self.evidence_budget_loss_weight),
            ("evidence_repair_gain_loss_weight", self.evidence_repair_gain_loss_weight),
            ("candidate_refinement_loss_weight", self.candidate_refinement_loss_weight),
            ("semantic_parity_loss_weight", self.semantic_parity_loss_weight),
        ):
            if value < 0:
                raise ValueError(f"experimental.{name} must be non-negative")
        if not 0.0 <= self.evidence_budget_target <= 1.0:
            raise ValueError("experimental.evidence_budget_target must be in [0, 1]")
        if self.evidence_minimum_gain < 0:
            raise ValueError("experimental.evidence_minimum_gain must be non-negative")
        if self.candidate_refinement_enabled and self.candidate_refinement_loss_weight == 0:
            raise ValueError(
                "experimental.candidate_refinement_loss_weight must be positive when "
                "candidate refinement is enabled"
            )

        # An enabled module with zero auxiliary weights adds parameters and
        # forward cost without a training signal, so warn instead of wasting
        # compute silently.
        if self.bats_enabled and not (self.bats_loss_weight or self.bats_coverage_weight):
            warnings.warn(
                "experimental.bats_enabled is true, but bats_loss_weight and "
                "bats_coverage_weight are both zero. Assign a positive weight or "
                "disable BATS to avoid unused parameters and computation.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.core_enabled and not self.register_loss_weight:
            warnings.warn(
                "experimental.core_enabled is true, but register_loss_weight is zero. "
                "Assign a positive weight or disable CoRe because the register "
                "classifier otherwise receives no training signal.",
                RuntimeWarning,
                stacklevel=2,
            )


@dataclass
class ModelConfig:
    vocab_size: int = 0
    d_model: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 2
    d_ff: int = 1536
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_embeddings: bool = True
    label_smoothing: float = 0.10
    z_loss_weight: float = 1e-4
    # H100-class GPUs have enough memory for the baseline without activation
    # recomputation. Disable by default for throughput; users can re-enable it
    # for larger models or memory-constrained devices.
    gradient_checkpointing: bool = False
    init_std: float = 0.02
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    def validate(self) -> None:
        self.experimental.validate()
        if self.vocab_size < 0:
            raise ValueError("vocab_size must be non-negative")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.encoder_layers <= 0 or self.decoder_layers <= 0:
            raise ValueError("encoder_layers and decoder_layers must be positive")
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("num_heads and num_kv_heads must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.max_seq_len < 8:
            raise ValueError("max_seq_len is unexpectedly small")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.z_loss_weight < 0:
            raise ValueError("z_loss_weight must be non-negative")
        if self.rms_norm_eps <= 0 or self.rope_base <= 0 or self.init_std <= 0:
            raise ValueError("normalization, RoPE, and initialization values must be positive")


@dataclass
class DataConfig:
    # Directory containing source JSONL. The automatic pipeline scans it for
    # additions or changes and prepares compatible tokenizer and dataset artifacts.
    raw_dir: str = "data"
    # Physical language pair represented by JSONL field names, for example
    # ["de", "fr"]. Tokenizer controls, preprocessing, and direction tags are
    # derived from this value rather than from language-specific Python code.
    language_pair: list[str] = field(default_factory=list)
    # Use multiple physical pairs in one model. An empty list falls back to the
    # single pair above; YAML should provide only one of these two fields.
    language_pairs: list[list[str]] = field(default_factory=list)
    # Exact directed edges to train. For de<->fr plus sw->ar, use
    # [[de, fr], [fr, de], [sw, ar]]. An empty list derives edges from the
    # bidirectional and source-only policies for backward compatibility.
    translation_directions: list[list[str]] = field(default_factory=list)
    # Directed edges with real ``source <draft> draft -> gold`` revision
    # supervision. An empty list derives them from indexed row provenance. Keep
    # them separate from translation edges so exports cannot invent revision
    # capability for an edge that never trained it.
    revision_directions: list[list[str]] = field(default_factory=list)
    # Languages allowed only as sources, never as generated targets. Use this
    # for mixed-language varieties, dialect inputs, or any graph node intended
    # only for normalization into target languages. Pairs containing one of
    # these nodes train only edges away from it, even when bidirectional is true.
    source_only_languages: list[str] = field(default_factory=list)
    # Online source-token dropout. A small value can reduce overfitting, while a
    # large one removes necessary evidence. Validation never applies it; zero
    # disables it.
    source_token_dropout: float = 0.05
    # Optional decoder-input corruption for exposure bias. Teacher forcing
    # always supplies a gold prefix, so a first inference error can move the
    # decoder into an unseen state and compound later errors. MRT addresses this
    # only during a shorter posttraining stage.
    #
    # Labels remain unchanged; only the prefix used to predict the next token is
    # perturbed, so the supervised objective remains the same.
    #
    # The default is zero because this changes decoder conditioning. If tested,
    # start around 0.1 and compare validation loss in a controlled ablation.
    decoder_input_noise: float = 0.0
    # File prefix for synthetic data such as backtranslation. Matching files are
    # restricted to the training split and automatically down-weighted to keep
    # synthetic rows from dominating. This legacy singular option is merged
    # with ``synthetic_prefixes``.
    synthetic_prefix: str = "bt_"
    synthetic_prefixes: list[str] = field(default_factory=lambda: list(DEFAULT_SYNTHETIC_PREFIXES))
    synthetic_sampling_weight: float = DEFAULT_SYNTHETIC_SAMPLING_WEIGHT
    # Reserve this fraction in a dedicated candidate-refinement evidence split.
    # It is separate from validation/test because relative T1-to-T2 evidence
    # must never influence absolute quality selection or early stopping.
    refinement_evidence_fraction: float = 0.0
    # Exact synthetic source basenames allowed to contribute source-only rows to
    # the refinement evidence split. Prefixes are intentionally insufficient:
    # adding a future synthetic file must not silently change release evidence.
    # These rows remain synthetic and are not absolute-quality evidence.
    source_only_synthetic_evidence_files: list[str] = field(default_factory=list)
    # Record that true ``source <draft> draft -> gold`` examples are present.
    # The training CLI detects ``revise_`` inputs automatically; set this only
    # when an equivalent reviewed source uses another file name.
    revision_examples: bool = False
    # Optional glossary JSON path. Translation and evaluation load it by default
    # and enforce configured term mappings. An empty string disables it.
    glossary: str = ""
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    tokenizer_features: str = DEFAULT_TOKENIZER_FEATURES
    dataset_dir: str = DEFAULT_DATASET_DIRECTORY
    train_split: str = "train"
    validation_split: str = "validation"
    bidirectional: bool = True
    # Assign splits through character 5-gram MinHash buckets instead of exact
    # strings so near duplicates cannot cross from training into holdouts.
    # Disable only to reproduce a historical exact split; scores from the two
    # policies are not directly comparable.
    approximate_split: bool = True
    max_source_length: int = 512
    max_target_length: int = 512
    # Round dynamic padding to this multiple for Tensor Core-friendly sequence
    # shapes. A value of one disables multiple-based padding.
    pad_to_multiple_of: int = 8
    denoise_probability: float = 0.10
    validation_denoise_probability: float = 0.0
    denoise_noise_density: float = 0.15
    denoise_mean_span: float = 3.0
    num_workers: int = 4
    # Stop a paid GPU run when a worker cannot produce the next batch. PyTorch's
    # default is zero, which waits forever. This timeout is used only when
    # ``num_workers`` is positive because single-process DataLoaders require zero.
    dataloader_timeout_seconds: float = 300.0
    bucket_size: int = 4096
    source_sampling_alpha: float = 1.0
    source_sampling_weights: dict[str, float] = field(default_factory=dict)
    max_source_upsampling: float = 3.0
    # Temperature over language pairs, applied on top of source balancing and
    # derived from the rows the dataset actually holds, so an edge that falls
    # behind is compensated without hand-written per-source weights. 1.0 follows
    # the corpus as built; 0.0 samples every configured pair equally. The
    # ``max_source_upsampling`` cap still bounds what any one source may become,
    # so a tiny edge cannot take over the batch.
    language_pair_sampling_alpha: float = 1.0

    def configured_language_pairs(self) -> tuple[tuple[str, str], ...]:
        has_single = bool(self.language_pair)
        has_multiple = bool(self.language_pairs)
        if has_single == has_multiple:
            raise ValueError("configure exactly one of data.language_pair or data.language_pairs")
        raw_pairs = self.language_pairs if has_multiple else [self.language_pair]
        return normalize_language_pairs(language_pairs=raw_pairs)

    def configured_synthetic_prefixes(self) -> tuple[str, ...]:
        return normalize_synthetic_prefixes(
            self.synthetic_prefixes,
            legacy_prefix=self.synthetic_prefix,
        )

    def configured_source_only_languages(self) -> tuple[str, ...]:
        return canonicalize_language_tags(
            self.source_only_languages,
            field="data.source_only_languages",
            reject_duplicates=False,
        )

    def configured_source_only_synthetic_evidence_files(self) -> tuple[str, ...]:
        """Return exact, path-safe synthetic evidence source basenames."""

        raw_files = cast(object, self.source_only_synthetic_evidence_files)
        if not isinstance(raw_files, list):
            raise ValueError(
                "data.source_only_synthetic_evidence_files must be a list of basenames"
            )
        normalized: list[str] = []
        seen_casefolded: set[str] = set()
        for index, raw_name in enumerate(cast(list[object], raw_files)):
            if not isinstance(raw_name, str):
                raise ValueError("data.source_only_synthetic_evidence_files must contain strings")
            name = raw_name.strip()
            if (
                not name
                or name != raw_name
                or Path(name).name != name
                or "/" in name
                or "\\" in name
                or not name.casefold().endswith(".jsonl")
            ):
                raise ValueError(
                    "data.source_only_synthetic_evidence_files must contain exact basenames"
                )
            folded = name.casefold()
            if folded in seen_casefolded:
                raise ValueError(
                    "data.source_only_synthetic_evidence_files contains a duplicate "
                    f"basename at index {index}"
                )
            seen_casefolded.add(folded)
            normalized.append(name)
        return tuple(normalized)

    def configured_translation_directions(self) -> tuple[tuple[str, str], ...]:
        """Return the directed edges the indexed dataset can actually train."""

        return normalize_translation_directions(
            self.configured_language_pairs(),
            self.translation_directions or None,
            bidirectional=self.bidirectional,
            source_only_languages=self.configured_source_only_languages(),
        )

    def configured_revision_directions(self) -> tuple[tuple[str, str], ...]:
        """Return explicitly authenticated revision edges, if configured."""

        trained = set(self.configured_translation_directions())
        directions: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for index, raw_direction in enumerate(self.revision_directions):
            direction = canonicalize_language_pair(
                raw_direction,
                field=f"data.revision_directions[{index}]",
            )
            if direction in seen:
                raise ValueError(
                    "duplicate data.revision_directions after BCP 47 canonicalization: "
                    f"{raw_direction!r}"
                )
            if direction not in trained:
                raise ValueError(
                    "data.revision_directions must be a subset of "
                    f"data.translation_directions; got {direction!r}"
                )
            seen.add(direction)
            directions.append(direction)
        return tuple(directions)

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                language for pair in self.configured_language_pairs() for language in pair
            )
        )


@dataclass
class TrainingConfig:
    # Checkpoint, log, and export root. Translation and augmentation commands use
    # the same default when automatically locating a model.
    output_dir: str = DEFAULT_RUN_DIRECTORY
    seed: int = 20260710
    # Production training completes this many full corpus passes. ``max_steps``
    # is an explicit override for short debugging and legacy configurations.
    num_train_epochs: int = 3
    max_steps: int | None = None
    batch_size_per_gpu: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    precision: str = "bf16"
    compile: bool = False
    # ``auto`` uses lower-communication DDP when one GPU can hold the complete
    # state and selects FSDP2 only when persistent state requires sharding.
    parallel_strategy: str = "auto"
    # Legacy configuration compatibility; new files should use parallel_strategy.
    fsdp2: bool | None = None
    fsdp_reduce_dtype: str = "auto"
    reshard_after_forward: bool = True
    log_every: int = 10
    eval_every: int = 250
    eval_batches: int = 20
    save_every: int = 500
    # Complete this many epochs before early stopping can react to noisy validation.
    early_stopping_min_epochs: int = 2
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    # SFT best/early-stopping metric. Equal weighting of per-direction token NLL
    # prevents a high-volume edge from hiding regression on a smaller edge.
    # Custom callers without direction metadata fall back to global NLL.
    sft_selection_metric: str = "macro_direction_nll"
    # Candidate refinement may be published only after every configured direction
    # improves by at least this held-out token-NLL margin. A positive floor keeps
    # the identity initialization and floating-point noise release-ineligible.
    candidate_refinement_min_worst_direction_nll_gain: float = (
        DEFAULT_CANDIDATE_REFINEMENT_MIN_WORST_DIRECTION_NLL_GAIN
    )
    # Release evidence must contain this many distinct held-out examples for
    # every configured translation direction. Repeating a scarce row would
    # inflate token counts without adding independent evidence.
    candidate_refinement_min_validation_examples_per_direction: int = 32
    # Exponential moving average: after each step, update shadow weights as
    # ``decay * shadow + (1 - decay) * parameter``. Zero disables EMA.
    ema_decay: float = 0.999
    tensorboard: bool = True
    tensorboard_dir: str | None = None
    resume_from: str | None = None
    # Generate final formats once from the selected best weights after training.
    # Intermediate best/latest saves retain FP32 only to avoid long conversion pauses.
    final_export_formats: list[str] = field(
        default_factory=lambda: [
            "fp32",
            "fp16",
            "bf16",
            "int8",
            "int4",
            "gguf_q4_k_m",
            "transformers",
        ]
    )


SUPPORTED_EXPORT_FORMATS = frozenset(
    {
        "fp32",
        "fp16",
        "bf16",
        "int8",
        "int4",
        "fp8",
        "gguf_q4_k_m",
        "transformers",
    }
)


def _validate_export_formats(formats: Sequence[str], *, field_name: str) -> None:
    """Both training stages publish their own artifacts, so both need this."""

    if not formats:
        raise ValueError(f"{field_name}.final_export_formats must contain at least one format")
    if len(set(formats)) != len(formats):
        raise ValueError(f"{field_name}.final_export_formats must not contain duplicates")
    unknown = sorted(set(formats) - SUPPORTED_EXPORT_FORMATS)
    if unknown:
        raise ValueError(
            f"unsupported {field_name}.final_export_formats: "
            f"{unknown}; supported={sorted(SUPPORTED_EXPORT_FORMATS)}"
        )


@dataclass
class FoundationConfig:
    """Configure monolingual pretraining before translation supervision.

    This stage learns span-corruption reconstruction from per-language text
    through ``<denoise_xx>`` tasks. It never sees translation pairs, so its
    output is an encoder-decoder foundation model rather than a translator and
    is saved under the distinct ``sion`` release role.

    Stage order: foundation -> translation SFT -> MRT posttraining.
    """

    # Run automatically when the corpus directory contains eligible data. A
    # missing or empty directory is reported and skipped. False skips even when
    # data exists.
    enabled: bool = True
    corpus_dir: str = DEFAULT_MONOLINGUAL_CORPUS_DIRECTORY
    dataset_dir: str = DEFAULT_FOUNDATION_DATASET_DIRECTORY
    # Public name for this non-translation foundation artifact.
    release_name: str = FOUNDATION_RELEASE_NAME
    # Explicit foundation-only languages. An empty list derives languages from
    # translation nodes after removing source-only varieties.
    languages: list[str] = field(default_factory=list)

    # Corpus composition.
    # Cross-language temperature sampling: 1.0 follows raw size; lower values
    # move closer to equal language mass.
    language_sampling_alpha: float = DEFAULT_LANGUAGE_SAMPLING_ALPHA
    # Warn when a language falls below this share; zero disables the warning.
    minimum_language_share: float = 0.05
    # Whether a configured language with no monolingual data stops preparation.
    # The default warns and proceeds because corpora are often filled in stages.
    require_all_languages: bool = False
    minimum_characters: int = 8
    maximum_characters: int = 4000
    deduplicate: bool = True
    # Target row share for optional structured reasoning files. Traces are longer
    # than ordinary reconstruction targets, so a small row share can still
    # provide substantial decoder-token mass. It is unused when no file exists.
    reasoning_sample_share: float = 0.05

    # Reconstruction task.
    # Ordinary monolingual rows always use denoising. This is the task itself,
    # unlike the SFT denoise probability. The collator explicitly bypasses this
    # corruption for reasoning JSONL rows.
    noise_density: float = 0.15
    mean_span: float = 3.0

    # Tokenizer sampling.
    # Include monolingual text up to each language's parallel sentence count
    # multiplied by this ratio. Omitting all monolingual text hides foundation
    # vocabulary, while using every row lets the largest language dominate.
    # Zero excludes monolingual corpora from tokenizer training.
    tokenizer_sample_ratio: float = 0.4

    # Training.
    num_train_epochs: int = 3
    max_steps: int | None = None
    batch_size_per_gpu: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 2_000
    eval_every: int = 1_000
    eval_batches: int = 50
    save_every: int = 2_000
    early_stopping_min_epochs: int = 2
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.0
    shard_size: int = 200_000
    validation_fraction: float = 0.002
    # Export formats for this stage. Foundation output normally continues into
    # fine-tuning, so defaults retain formats that can resume training.
    final_export_formats: list[str] = field(
        default_factory=lambda: ["fp32", "bf16", "transformers"]
    )

    def validate(self) -> None:
        if not self.corpus_dir:
            raise ValueError("foundation.corpus_dir must be non-empty")
        if not self.dataset_dir:
            raise ValueError("foundation.dataset_dir must be non-empty")
        if (
            not self.release_name
            or self.release_name != self.release_name.strip()
            or not self.release_name.isascii()
        ):
            raise ValueError("foundation.release_name must be a normalized non-empty ASCII name")
        if self.release_name == TRANSLATION_RELEASE_NAME:
            raise ValueError(
                "foundation.release_name must differ from the translation release name "
                f"({TRANSLATION_RELEASE_NAME!r}); the two stages are published separately"
            )
        self.languages = list(
            canonicalize_language_tags(
                self.languages,
                field="foundation.languages",
            )
        )
        if not 0.0 < self.language_sampling_alpha <= 1.0:
            raise ValueError("foundation.language_sampling_alpha must be in (0, 1]")
        if not 0.0 <= self.minimum_language_share < 1.0:
            raise ValueError("foundation.minimum_language_share must be in [0, 1)")
        if self.minimum_characters < 1:
            raise ValueError("foundation.minimum_characters must be positive")
        if self.maximum_characters <= self.minimum_characters:
            raise ValueError(
                "foundation.maximum_characters must be greater than minimum_characters"
            )
        if not 0.0 <= self.reasoning_sample_share <= 0.10:
            raise ValueError("foundation.reasoning_sample_share must be in [0, 0.10]")
        if not 0.0 < self.noise_density < 1.0:
            raise ValueError("foundation.noise_density must be in (0, 1)")
        if self.mean_span <= 0:
            raise ValueError("foundation.mean_span must be positive")
        if self.tokenizer_sample_ratio < 0:
            raise ValueError("foundation.tokenizer_sample_ratio must be non-negative")
        for name, value in (
            ("num_train_epochs", self.num_train_epochs),
            ("batch_size_per_gpu", self.batch_size_per_gpu),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("eval_every", self.eval_every),
            ("eval_batches", self.eval_batches),
            ("save_every", self.save_every),
            ("shard_size", self.shard_size),
        ):
            if value <= 0:
                raise ValueError(f"foundation.{name} must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("foundation.max_steps must be positive when specified")
        if self.learning_rate <= 0:
            raise ValueError("foundation.learning_rate must be positive")
        if not 0.0 <= self.min_learning_rate_ratio <= 1.0:
            raise ValueError("foundation.min_learning_rate_ratio must be in [0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("foundation.warmup_steps must be non-negative")
        if self.max_steps is not None and self.warmup_steps > self.max_steps:
            raise ValueError("foundation.warmup_steps cannot exceed max_steps")
        if self.early_stopping_patience < 0:
            raise ValueError("foundation.early_stopping_patience must be non-negative")
        if self.early_stopping_min_epochs <= 0:
            raise ValueError("foundation.early_stopping_min_epochs must be positive")
        if self.max_steps is None and self.early_stopping_min_epochs > self.num_train_epochs:
            raise ValueError("foundation.early_stopping_min_epochs cannot exceed num_train_epochs")
        if self.early_stopping_min_delta < 0:
            raise ValueError("foundation.early_stopping_min_delta must be non-negative")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("foundation.validation_fraction must be in (0, 0.5)")
        _validate_export_formats(self.final_export_formats, field_name="foundation")


@dataclass
class PostTrainingConfig:
    """Configure composite minimum-risk and candidate-preference posttraining."""

    enabled: bool = True
    method: str = "mrt"
    num_train_epochs: int = 2
    max_steps: int | None = None
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-5
    warmup_steps: int = 200
    samples_per_source: int = 2
    # Score this many candidates per source at once instead of materializing all
    # candidate vocabulary logits together. One minimizes peak VRAM.
    candidate_micro_batch: int = 1
    # Recompute large candidate-scoring activations during backward to save memory.
    candidate_gradient_checkpointing: bool = True
    sampling_temperature: float = 1.0
    top_k: int = 64
    max_new_tokens: int = 256
    risk_weight: float = 0.20
    mrt_alpha: float = 1.0
    # Learn the ordering of every candidate pair with a reward difference.
    preference_weight: float = 0.10
    preference_min_gap: float = 0.05
    preference_temperature: float = 1.0
    # Composite reward weights reduce single-metric reward hacking.
    reward_chrf_weight: float = 0.55
    reward_token_f1_weight: float = 0.15
    reward_number_weight: float = 0.10
    reward_structured_weight: float = 0.05
    reward_slot_weight: float = 0.05
    reward_language_weight: float = 0.05
    reward_length_weight: float = 0.05
    reward_repetition_penalty: float = 0.15
    reward_copy_penalty: float = 0.10
    # Numeric corruption is a hard penalty, not only a weighted component. A
    # proportional preservation score can let one invented value win through a
    # small chrF gain. Subtract this value per corruption; zero restores the old
    # weight-only behavior.
    reward_number_corruption_penalty: float = 0.35
    # Translate candidates back to the source language when an authenticated
    # reverse edge exists, adding a cycle-consistency signal.
    roundtrip_enabled: bool = False
    roundtrip_reward_weight: float = 0.20
    roundtrip_failure_penalty: float = 0.15
    roundtrip_min_score: float = 0.55
    roundtrip_num_beams: int = 1
    roundtrip_max_new_tokens: int = 256
    # Candidate sampling and best-model validation use the same constraints as
    # deployment. These fields are part of checkpoint identity because changing
    # them changes both the optimized candidate distribution and the metric used
    # for early stopping.
    validation_num_beams: int = DEFAULT_NUM_BEAMS
    validation_length_penalty: float = DEFAULT_LENGTH_PENALTY
    decode_min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS
    decode_no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE
    decode_max_output_length_ratio: float = DEFAULT_MAX_OUTPUT_LENGTH_RATIO
    decode_max_output_length_margin: int = DEFAULT_MAX_OUTPUT_LENGTH_MARGIN
    # MRT best-selection metric. A mean reward can hide regression on one edge
    # behind improvement on another. The default favors the weakest direction;
    # callers without direction metadata fall back to mean reward.
    selection_metric: str = "worst_direction_reward"
    eval_batch_size_per_gpu: int = 1
    eval_every: int = 250
    save_every: int = 1_000
    early_stopping_min_epochs: int = 2
    early_stopping_patience: int = 5


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    # Stage order: monolingual foundation, translation SFT, then MRT posttraining.
    foundation: FoundationConfig = field(default_factory=FoundationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    posttraining: PostTrainingConfig = field(default_factory=PostTrainingConfig)

    def foundation_languages(self) -> tuple[str, ...]:
        """Return effective foundation languages after excluding source-only nodes."""

        return foundation_languages(
            self.foundation.languages or self.data.languages,
            self.data.configured_source_only_languages(),
        )

    def validate(self) -> None:
        self.model.validate()
        if not 0.0 <= self.data.source_token_dropout < 0.5:
            raise ValueError("source_token_dropout must be in [0, 0.5)")
        if not 0.0 <= self.data.decoder_input_noise < 0.5:
            raise ValueError("decoder_input_noise must be in [0, 0.5)")
        if not self.data.synthetic_prefix:
            raise ValueError("synthetic_prefix must be non-empty")
        self.data.configured_synthetic_prefixes()
        if not 0.0 <= self.data.synthetic_sampling_weight <= 1.0:
            raise ValueError("synthetic_sampling_weight must be in [0, 1]")
        evidence_fraction = cast(object, self.data.refinement_evidence_fraction)
        if (
            isinstance(evidence_fraction, bool)
            or not isinstance(evidence_fraction, (int, float))
            or not math.isfinite(float(evidence_fraction))
            or not 0.0 <= float(evidence_fraction) < 0.5
        ):
            raise ValueError("refinement_evidence_fraction must be in [0, 0.5)")
        evidence_files = self.data.configured_source_only_synthetic_evidence_files()
        if evidence_files and float(evidence_fraction) <= 0.0:
            raise ValueError(
                "source_only_synthetic_evidence_files requires a positive "
                "refinement_evidence_fraction"
            )
        if evidence_files and not self.data.configured_source_only_languages():
            raise ValueError("source_only_synthetic_evidence_files requires source_only_languages")
        if not 0.0 <= self.data.denoise_probability <= 1.0:
            raise ValueError("denoise_probability must be in [0, 1]")
        if not 0.0 <= self.data.validation_denoise_probability <= 1.0:
            raise ValueError("validation_denoise_probability must be in [0, 1]")
        if not 0.0 <= self.data.denoise_noise_density < 1.0:
            raise ValueError("denoise_noise_density must be in [0, 1)")
        if self.data.denoise_mean_span <= 0:
            raise ValueError("denoise_mean_span must be positive")
        if self.data.max_source_length < 2 or self.data.max_target_length < 2:
            raise ValueError("maximum source and target lengths must be at least 2")
        if self.data.max_source_length > self.model.max_seq_len:
            raise ValueError("max_source_length cannot exceed model.max_seq_len")
        if self.data.max_target_length > self.model.max_seq_len:
            raise ValueError("max_target_length cannot exceed model.max_seq_len")
        if self.data.pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be at least 1")
        if self.data.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        loader_timeout = cast(object, self.data.dataloader_timeout_seconds)
        if (
            isinstance(loader_timeout, bool)
            or not isinstance(loader_timeout, (int, float))
            or not math.isfinite(float(loader_timeout))
            or float(loader_timeout) <= 0.0
        ):
            raise ValueError("dataloader_timeout_seconds must be a finite positive number")
        if self.data.bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if self.data.source_sampling_alpha <= 0.0:
            raise ValueError("source_sampling_alpha must be positive")
        if self.data.max_source_upsampling < 1.0:
            raise ValueError("max_source_upsampling must be at least 1")
        if not 0.0 <= self.data.language_pair_sampling_alpha <= 1.0:
            raise ValueError("language_pair_sampling_alpha must be in [0, 1]")
        if any(
            not name or weight < 0 for name, weight in self.data.source_sampling_weights.items()
        ):
            raise ValueError(
                "source_sampling_weights must have non-empty names and non-negative values"
            )
        if self.training.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.training.max_steps is not None and self.training.max_steps <= 0:
            raise ValueError("max_steps must be positive when specified")
        if self.training.batch_size_per_gpu <= 0:
            raise ValueError("batch_size_per_gpu must be positive")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.training.min_learning_rate_ratio <= 1.0:
            raise ValueError("min_learning_rate_ratio must be in [0, 1]")
        if self.training.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if (
            self.training.max_steps is not None
            and self.training.warmup_steps > self.training.max_steps
        ):
            raise ValueError("warmup_steps cannot exceed max_steps")
        if self.training.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.training.adam_beta1 < 1.0:
            raise ValueError("adam_beta1 must be in [0, 1)")
        if not 0.0 <= self.training.adam_beta2 < 1.0:
            raise ValueError("adam_beta2 must be in [0, 1)")
        if self.training.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if self.training.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.training.precision.lower() not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be one of: fp32, bf16, fp16")
        if self.training.parallel_strategy.lower() not in {"auto", "ddp", "fsdp2"}:
            raise ValueError("parallel_strategy must be one of: auto, ddp, fsdp2")
        if self.training.fsdp_reduce_dtype.lower() not in {"auto", "fp32", "bf16"}:
            raise ValueError("fsdp_reduce_dtype must be one of: auto, fp32, bf16")
        if self.training.fsdp2 is not None and self.training.parallel_strategy.lower() != "auto":
            raise ValueError("training.fsdp2 and training.parallel_strategy cannot both be set")
        for name, value in (
            ("log_every", self.training.log_every),
            ("eval_every", self.training.eval_every),
            ("eval_batches", self.training.eval_batches),
            ("save_every", self.training.save_every),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.training.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.training.early_stopping_min_epochs <= 0:
            raise ValueError("early_stopping_min_epochs must be positive")
        if (
            self.training.max_steps is None
            and self.training.early_stopping_min_epochs > self.training.num_train_epochs
        ):
            raise ValueError("early_stopping_min_epochs cannot exceed num_train_epochs")
        if self.training.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        supported_sft_selection_metrics = {
            "global_nll",
            "macro_direction_nll",
            "worst_direction_nll",
        }
        if self.training.sft_selection_metric.lower() not in supported_sft_selection_metrics:
            raise ValueError(
                "sft_selection_metric must be one of: "
                + ", ".join(sorted(supported_sft_selection_metrics))
            )
        refinement_min_gain: object = (
            self.training.candidate_refinement_min_worst_direction_nll_gain
        )
        if (
            isinstance(refinement_min_gain, bool)
            or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                refinement_min_gain, (int, float)
            )
            or not math.isfinite(float(refinement_min_gain))
            or refinement_min_gain <= 0.0
        ):
            raise ValueError(
                "candidate_refinement_min_worst_direction_nll_gain must be finite and positive"
            )
        refinement_min_examples = (
            self.training.candidate_refinement_min_validation_examples_per_direction
        )
        if type(refinement_min_examples) is not int or refinement_min_examples <= 0:
            raise ValueError(
                "candidate_refinement_min_validation_examples_per_direction must be a "
                "positive integer"
            )
        if not 0.0 <= self.training.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        _validate_export_formats(self.training.final_export_formats, field_name="training")
        self.foundation.validate()
        post = self.posttraining
        if post.method != "mrt":
            raise ValueError("posttraining.method must be 'mrt'")
        for name, value in (
            ("num_train_epochs", post.num_train_epochs),
            ("batch_size_per_gpu", post.batch_size_per_gpu),
            ("gradient_accumulation_steps", post.gradient_accumulation_steps),
            ("samples_per_source", post.samples_per_source),
            ("candidate_micro_batch", post.candidate_micro_batch),
            ("max_new_tokens", post.max_new_tokens),
            ("eval_batch_size_per_gpu", post.eval_batch_size_per_gpu),
            ("eval_every", post.eval_every),
            ("save_every", post.save_every),
            ("validation_num_beams", post.validation_num_beams),
            ("roundtrip_num_beams", post.roundtrip_num_beams),
            ("roundtrip_max_new_tokens", post.roundtrip_max_new_tokens),
        ):
            if value <= 0:
                raise ValueError(f"posttraining.{name} must be positive")
        for name, value in (
            ("decode_min_new_tokens", post.decode_min_new_tokens),
            ("decode_no_repeat_ngram_size", post.decode_no_repeat_ngram_size),
            ("decode_max_output_length_margin", post.decode_max_output_length_margin),
        ):
            if type(value) is not int:
                raise ValueError(f"posttraining.{name} must be an integer")
            if value < 0:
                raise ValueError(f"posttraining.{name} must be non-negative")
        if post.decode_min_new_tokens >= post.max_new_tokens:
            raise ValueError(
                "posttraining.decode_min_new_tokens must be smaller than max_new_tokens"
            )
        if (
            type(post.validation_length_penalty) not in (int, float)
            or not math.isfinite(post.validation_length_penalty)
            or post.validation_length_penalty <= 0
        ):
            raise ValueError(
                "posttraining.validation_length_penalty must be a positive finite number"
            )
        if (
            type(post.decode_max_output_length_ratio) not in (int, float)
            or not math.isfinite(post.decode_max_output_length_ratio)
            or post.decode_max_output_length_ratio <= 0
        ):
            raise ValueError(
                "posttraining.decode_max_output_length_ratio must be a positive finite number"
            )
        if post.max_steps is not None and post.max_steps <= 0:
            raise ValueError("posttraining.max_steps must be positive when specified")
        if post.samples_per_source < 2:
            raise ValueError("posttraining.samples_per_source must be at least 2")
        if post.learning_rate <= 0 or post.sampling_temperature <= 0 or post.mrt_alpha <= 0:
            raise ValueError(
                "posttraining learning rate, temperature, and mrt_alpha must be positive"
            )
        if not 0.0 <= post.risk_weight <= 1.0:
            raise ValueError("posttraining.risk_weight must be in [0, 1]")
        if not 0.0 <= post.preference_weight <= 1.0:
            raise ValueError("posttraining.preference_weight must be in [0, 1]")
        if not 0.0 <= post.preference_min_gap < 1.0:
            raise ValueError("posttraining.preference_min_gap must be in [0, 1)")
        if post.preference_temperature <= 0:
            raise ValueError("posttraining.preference_temperature must be positive")
        reward_weights = {
            name: getattr(post, name)
            for name in (
                "reward_chrf_weight",
                "reward_token_f1_weight",
                "reward_number_weight",
                "reward_structured_weight",
                "reward_slot_weight",
                "reward_language_weight",
                "reward_length_weight",
            )
        }
        if any(value < 0 for value in reward_weights.values()):
            raise ValueError("posttraining reward weights must be non-negative")
        if post.reward_number_corruption_penalty < 0:
            raise ValueError("posttraining.reward_number_corruption_penalty must be non-negative")
        if sum(reward_weights.values()) <= 0:
            raise ValueError("at least one posttraining reward weight must be positive")
        if not 0.0 <= post.reward_repetition_penalty <= 1.0:
            raise ValueError("posttraining.reward_repetition_penalty must be in [0, 1]")
        if not 0.0 <= post.reward_copy_penalty <= 1.0:
            raise ValueError("posttraining.reward_copy_penalty must be in [0, 1]")
        if post.roundtrip_reward_weight < 0:
            raise ValueError("posttraining.roundtrip_reward_weight must be non-negative")
        if not 0.0 <= post.roundtrip_failure_penalty <= 1.0:
            raise ValueError("posttraining.roundtrip_failure_penalty must be in [0, 1]")
        if not 0.0 <= post.roundtrip_min_score <= 1.0:
            raise ValueError("posttraining.roundtrip_min_score must be in [0, 1]")
        if post.roundtrip_enabled and post.roundtrip_reward_weight == 0:
            raise ValueError(
                "posttraining.roundtrip_reward_weight must be positive when roundtrip is enabled"
            )
        if post.top_k < 0:
            raise ValueError("posttraining.top_k must be non-negative")
        supported_post_selection = {"reward", "macro_direction_reward", "worst_direction_reward"}
        if post.selection_metric not in supported_post_selection:
            raise ValueError(
                "posttraining.selection_metric must be one of: "
                + ", ".join(sorted(supported_post_selection))
            )
        if post.warmup_steps < 0:
            raise ValueError("posttraining.warmup_steps must be non-negative")
        if post.max_steps is not None and post.warmup_steps > post.max_steps:
            raise ValueError("posttraining.warmup_steps cannot exceed max_steps")
        if post.early_stopping_patience < 0:
            raise ValueError("posttraining.early_stopping_patience must be non-negative")
        if post.early_stopping_min_epochs <= 0:
            raise ValueError("posttraining.early_stopping_min_epochs must be positive")
        if post.max_steps is None and post.early_stopping_min_epochs > post.num_train_epochs:
            raise ValueError(
                "posttraining.early_stopping_min_epochs cannot exceed num_train_epochs"
            )
        raw_pairs = (
            self.data.language_pairs if self.data.language_pairs else [self.data.language_pair]
        )
        pairs = self.data.configured_language_pairs()
        if len(pairs) != len(raw_pairs):
            raise ValueError("duplicate or reversed language pair after BCP 47 canonicalization")
        if self.data.language_pairs:
            self.data.language_pairs = [list(pair) for pair in pairs]
        else:
            self.data.language_pair = list(pairs[0])
        source_only = self.data.configured_source_only_languages()
        self.data.source_only_languages = list(source_only)
        if source_only:
            known = set(self.data.languages)
            unknown = sorted(set(source_only) - known)
            if unknown:
                raise ValueError(
                    "data.source_only_languages must appear in the configured language "
                    f"pairs; {unknown} do not (configured languages: {sorted(known)})"
                )
            for pair in pairs:
                if pair[0] in source_only and pair[1] in source_only:
                    raise ValueError(
                        "at most one side of a language pair may be source-only; "
                        f"both sides of {list(pair)!r} are listed in "
                        "data.source_only_languages"
                    )
        directions = self.data.configured_translation_directions()
        if self.data.translation_directions:
            self.data.translation_directions = [list(direction) for direction in directions]
        revision_directions = self.data.configured_revision_directions()
        if self.data.revision_directions:
            self.data.revision_directions = [list(direction) for direction in revision_directions]
            self.data.revision_examples = True

    def validate_training_supervision(
        self,
        *,
        alignment_targets_available: bool,
    ) -> None:
        """Validate losses whose labels are supplied by the training pipeline.

        ``ModelConfig`` deliberately permits a positive BATS alignment weight:
        research callers can pass ``alignment_targets`` directly to the model.
        The built-in collator does not create those dense alignment matrices,
        however, so the standard CLI must reject that configuration instead of
        silently optimizing a permanently-zero alignment loss.
        """

        experimental = self.model.experimental
        if (
            experimental.bats_enabled
            and experimental.bats_loss_weight > 0
            and not alignment_targets_available
        ):
            raise ValueError(
                "experimental.bats_loss_weight is positive, but this training pipeline "
                "does not provide alignment_targets. Set bats_loss_weight=0, add an "
                "alignment-aware collator/training caller, or explicitly validate the "
                "custom pipeline with alignment_targets_available=True."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_dataclass(cls: type, values: dict[str, Any] | None):
    values = dict(values or {})
    return cls(**values)


_CONFIG_TOP_LEVEL_KEYS = frozenset({"model", "data", "foundation", "training", "posttraining"})


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys instead of keeping the last one."""

    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def parse_raw_config_text(content: str) -> dict[str, Any]:
    """Parse config text and reject ambiguous or misspelled top-level keys."""

    raw = yaml.load(content, Loader=_UniqueKeySafeLoader) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    unknown = set(raw) - _CONFIG_TOP_LEVEL_KEYS
    if unknown:
        rendered = ", ".join(sorted(repr(key) for key in unknown))
        expected = ", ".join(sorted(_CONFIG_TOP_LEVEL_KEYS))
        raise ValueError(f"unknown top-level config key(s): {rendered}; expected only: {expected}")
    return raw


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """Read a config file and reject ambiguous or misspelled top-level keys."""

    return parse_raw_config_text(Path(path).read_text(encoding="utf-8"))


def config_from_raw(raw: dict[str, Any]) -> AppConfig:
    """Build ``AppConfig`` from raw values, using dataclass defaults for omissions."""
    model_values = dict(raw.get("model") or {})
    data_values = dict(raw.get("data") or {})
    if "language_pair" in data_values and "language_pairs" in data_values:
        raise ValueError("data.language_pair and data.language_pairs cannot both be set")
    training_values = dict(raw.get("training") or {})
    if "fsdp2" in training_values and "parallel_strategy" in training_values:
        raise ValueError("training.fsdp2 and training.parallel_strategy cannot both be set")
    experimental = _construct_dataclass(ExperimentalConfig, model_values.pop("experimental", {}))
    model = ModelConfig(experimental=experimental, **model_values)
    return AppConfig(
        model=model,
        data=_construct_dataclass(DataConfig, data_values),
        foundation=_construct_dataclass(FoundationConfig, raw.get("foundation")),
        training=_construct_dataclass(TrainingConfig, training_values),
        posttraining=_construct_dataclass(PostTrainingConfig, raw.get("posttraining")),
    )


def load_config(path: str | Path) -> AppConfig:
    config = config_from_raw(load_raw_config(path))
    config.validate()
    return config
