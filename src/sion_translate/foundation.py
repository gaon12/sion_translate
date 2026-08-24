"""Plan the monolingual foundation stage and derive its configuration.

This logic stays outside the CLI for two reasons. The decision to run the
stage, including its human-readable explanation, must be testable without CLI
wiring. The inheritance rules that derive a foundation configuration from the
translation configuration are also easy to get subtly wrong and deserve a
separate, directly tested boundary.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from sion_translate.artifacts import FOUNDATION_STAGE_DIRECTORY, MODEL_RELEASE_VERSION
from sion_translate.config import AppConfig
from sion_translate.data.monolingual import (
    MonolingualDiscovery,
    assess_language_balance,
    discover_monolingual_sources,
    render_discovery_report,
)


PIPELINE_IDENTITY_SCHEMA = "sion-translation-pipeline-v2"
FOUNDATION_LINEAGE_SCHEMA = "sion-foundation-lineage-v1"


@dataclass(frozen=True)
class FoundationPlan:
    """Record whether this run includes the foundation stage and explain why."""

    enabled: bool
    reason: str
    discovery: MonolingualDiscovery
    languages: tuple[str, ...] = ()
    report: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.enabled


def build_translation_pipeline_identity(
    plan: FoundationPlan,
    *,
    foundation_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the stable ancestry branch required for translation-stage resume.

    ``ran`` is deliberately not part of this identity: a completed foundation
    stage that is reused and one that was trained in the current process have
    the same translation-model ancestry. Paths and human-readable skip reasons
    are likewise runtime details rather than compatibility inputs.
    """

    identity: dict[str, Any] = {
        "schema": PIPELINE_IDENTITY_SCHEMA,
        "branch": "foundation-then-translation" if plan.enabled else "translation-only",
    }
    if not plan.enabled:
        if foundation_lineage is not None:
            raise ValueError("translation-only pipeline cannot carry foundation lineage")
        return identity
    if foundation_lineage is None:
        raise ValueError("foundation-enabled translation pipeline requires resolved lineage")
    lineage = dict(foundation_lineage)
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
    if set(lineage) != expected_fields:
        raise ValueError("foundation lineage fields do not match the v1 contract")
    if lineage.get("schema") != FOUNDATION_LINEAGE_SCHEMA:
        raise ValueError("foundation lineage has an unsupported schema")
    release_name = lineage.get("release_name")
    if (
        not isinstance(release_name, str)
        or not release_name
        or release_name != release_name.strip()
        or not release_name.isascii()
    ):
        raise ValueError("foundation lineage release_name must be non-empty ASCII")
    if lineage.get("release_version") != MODEL_RELEASE_VERSION:
        raise ValueError("foundation lineage release_version does not match this package")
    if (
        isinstance(lineage.get("selected_step"), bool)
        or not isinstance(lineage.get("selected_step"), int)
        or cast(int, lineage.get("selected_step")) < 0
    ):
        raise ValueError("foundation lineage selected_step must be a non-negative integer")
    languages = lineage.get("languages")
    if not isinstance(languages, list):
        raise ValueError("foundation lineage languages do not match the current plan")
    language_values = cast(list[object], languages)
    if not all(isinstance(value, str) for value in language_values) or language_values != list(
        plan.languages
    ):
        raise ValueError("foundation lineage languages do not match the current plan")
    for field_name in (
        "foundation_manifest_sha256",
        "tokenizer_sha256",
        "checkpoint_identity_sha256",
        "checkpoint_artifact_sha256",
    ):
        value = lineage.get(field_name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or value != value.lower()
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"foundation lineage {field_name} must be a SHA-256 digest")
    identity["foundation"] = lineage
    return identity


def plan_foundation_stage(config: AppConfig) -> FoundationPlan:
    """Inspect the corpus and decide whether to run the foundation stage.

    Automatic execution when a corpus is present makes skipping a normal code
    path. Every skip therefore includes a clear reason. A silent skip could
    otherwise leave an operator believing that gigabytes of monolingual data
    were included when translation training never consumed them.
    """

    languages = config.foundation_languages()
    discovery = discover_monolingual_sources(config.foundation.corpus_dir, languages)
    report = tuple(render_discovery_report(discovery))

    if not config.foundation.enabled:
        return FoundationPlan(
            enabled=False,
            reason="Skipping the foundation stage because foundation.enabled=false.",
            discovery=discovery,
            languages=languages,
            report=report,
        )
    if not discovery.sources:
        return FoundationPlan(
            enabled=False,
            reason=(
                f"Skipping the foundation stage because {config.foundation.corpus_dir} "
                "contains no usable monolingual files. Add .txt or .jsonl files under "
                "language-code directories (for example, "
                f"{config.foundation.corpus_dir}/{languages[0] if languages else 'ko'}/) "
                "and the next run will discover them automatically."
            ),
            discovery=discovery,
            languages=languages,
            report=report,
        )

    balance = assess_language_balance(
        {
            language: sum(
                source.size_bytes for source in discovery.sources if source.language == language
            )
            for language in languages
        },
        alpha=config.foundation.language_sampling_alpha,
        minimum_share=config.foundation.minimum_language_share,
    )
    if config.foundation.require_all_languages and discovery.languages_without_data:
        missing = ", ".join(discovery.languages_without_data)
        raise RuntimeError(
            "foundation.require_all_languages=true, but these configured languages have "
            f"no monolingual data: {missing}. Add the missing data or disable "
            "require_all_languages."
        )
    return FoundationPlan(
        enabled=True,
        reason=f"Found {len(discovery.sources)} usable monolingual corpus files.",
        discovery=discovery,
        languages=languages,
        report=report,
        warnings=balance.warnings,
    )


def foundation_run_directory(config: AppConfig) -> Path:
    return Path(config.training.output_dir) / FOUNDATION_STAGE_DIRECTORY


def build_foundation_config(config: AppConfig) -> AppConfig:
    """Derive a foundation-stage configuration from translation settings.

    The stage inherits model structure, precision, parallel strategy, random
    seed, and tokenizer. It replaces dataset paths, objective settings
    (reconstruction plus optional reasoning), schedule, and output paths.

    Reconstruction probability is 1.0 for both training and validation. A
    normal monolingual shard has ``src == tgt``, so an uncorrupted example only
    teaches copying. Setting validation alone to zero would also turn model
    selection into a meaningless copy-task comparison. Structured reasoning
    rows carry a separate task tag, allowing the collator to bypass denoising
    independently of this probability.
    """

    foundation = config.foundation
    derived = copy.deepcopy(config)
    derived.data.dataset_dir = foundation.dataset_dir
    derived.data.denoise_probability = 1.0
    derived.data.validation_denoise_probability = 1.0
    derived.data.denoise_noise_density = foundation.noise_density
    derived.data.denoise_mean_span = foundation.mean_span
    # Span corruption already supplies noise. Source-token dropout would erase
    # evidence that the model needs in order to reconstruct the target.
    derived.data.source_token_dropout = 0.0
    derived.data.decoder_input_noise = 0.0
    # Foundation shards intentionally exclude source-only languages. Keeping
    # them here could make the collator skip every reconstruction example.
    derived.data.source_only_languages = []
    # Keep ``data.language_pairs`` from the translation configuration. A
    # reconstruction pair would be self-directed, such as (ko, ko), while the
    # shared validator correctly requires distinct endpoints for translation.
    # Introducing an exception here would weaken that invariant. The foundation
    # manifest selects the languages actually read by ``IndexedParallelDataset``,
    # and the collator uses each record's source and target language fields, so
    # configured translation pairs are not consulted on this path.

    training = derived.training
    training.output_dir = str(foundation_run_directory(config))
    training.num_train_epochs = foundation.num_train_epochs
    training.max_steps = foundation.max_steps
    training.batch_size_per_gpu = foundation.batch_size_per_gpu
    training.gradient_accumulation_steps = foundation.gradient_accumulation_steps
    training.learning_rate = foundation.learning_rate
    training.min_learning_rate_ratio = foundation.min_learning_rate_ratio
    training.warmup_steps = foundation.warmup_steps
    training.eval_every = foundation.eval_every
    training.eval_batches = foundation.eval_batches
    training.save_every = foundation.save_every
    training.early_stopping_min_epochs = foundation.early_stopping_min_epochs
    training.early_stopping_patience = foundation.early_stopping_patience
    training.early_stopping_min_delta = foundation.early_stopping_min_delta
    training.final_export_formats = list(foundation.final_export_formats)
    training.resume_from = None
    training.tensorboard_dir = None
    # Direction-specific metrics describe translation, not reconstruction.
    training.sft_selection_metric = "global_nll"

    # A reconstruction-only stage does not run translation post-training.
    derived.posttraining.enabled = False
    derived.foundation.enabled = False
    return derived


@dataclass
class FoundationOutcome:
    """Record the stage outcome for translation provenance."""

    ran: bool
    reason: str
    best_checkpoint: str | None = None
    selected_step: int | None = None
    languages: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "ran": self.ran,
            "reason": self.reason,
            "best_checkpoint": self.best_checkpoint,
            "selected_step": self.selected_step,
            "languages": list(self.languages),
            "warnings": list(self.warnings),
        }


__all__ = [
    "FoundationOutcome",
    "FoundationPlan",
    "PIPELINE_IDENTITY_SCHEMA",
    "build_foundation_config",
    "build_translation_pipeline_identity",
    "foundation_run_directory",
    "plan_foundation_stage",
]
