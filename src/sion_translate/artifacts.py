"""Canonical public paths for generated artifacts and training runs.

Compatibility is established by artifact metadata and content hashes, not by a
release-looking directory name.  Keep these paths stable for scripts and users.
"""

from __future__ import annotations

from sion_translate._version import MODEL_RELEASE_VERSION

DEFAULT_ARTIFACT_ROOT = "artifacts"
DEFAULT_TOKENIZER_MODEL = f"{DEFAULT_ARTIFACT_ROOT}/tokenizer/sion.model"
DEFAULT_TOKENIZER_FEATURES = f"{DEFAULT_ARTIFACT_ROOT}/tokenizer/token_features.npz"
DEFAULT_DATASET_DIRECTORY = f"{DEFAULT_ARTIFACT_ROOT}/dataset"
DEFAULT_RUN_DIRECTORY = "runs/auto"

# Keep monolingual foundation inputs and indexed outputs separate from the
# parallel dataset. They share a tokenizer but use different record schemas and
# split policies, so mixing their paths would make artifact identity ambiguous.
DEFAULT_MONOLINGUAL_CORPUS_DIRECTORY = "data/corpus"
DEFAULT_FOUNDATION_DATASET_DIRECTORY = f"{DEFAULT_ARTIFACT_ROOT}/foundation_dataset"

# Public release names. The foundation output is an encoder-decoder base model,
# not a translation model; translation stages derive from it. Distinct names
# prevent users from confusing which weights are allowed to translate.
FOUNDATION_RELEASE_NAME = "sion"
TRANSLATION_RELEASE_NAME = "sion_translate"
# A guarded training run places this marker beside any superseded inference
# export. Discovery and direct native loading reject the directory until an
# atomic, guard-approved export replaces it.
RELEASE_INELIGIBLE_FILENAME = "RELEASE_INELIGIBLE.json"
RELEASE_INELIGIBLE_SCHEMA = "sion-release-ineligible-v1"
# Public model generation. The Python package uses semantic versioning, while
# model lineage uses the user-facing major.minor label. Foundation and derived
# translation releases record the same generation value.

# Stage directories inside one run. ``pretrain`` already names translation SFT,
# so the earlier base-model stage uses the unambiguous ``foundation`` name.
FOUNDATION_STAGE_DIRECTORY = "foundation"
SUPERVISED_STAGE_DIRECTORY = "pretrain"
POSTTRAINING_STAGE_DIRECTORY = "posttrain"


__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_DATASET_DIRECTORY",
    "DEFAULT_FOUNDATION_DATASET_DIRECTORY",
    "DEFAULT_MONOLINGUAL_CORPUS_DIRECTORY",
    "DEFAULT_RUN_DIRECTORY",
    "DEFAULT_TOKENIZER_FEATURES",
    "DEFAULT_TOKENIZER_MODEL",
    "FOUNDATION_RELEASE_NAME",
    "FOUNDATION_STAGE_DIRECTORY",
    "MODEL_RELEASE_VERSION",
    "POSTTRAINING_STAGE_DIRECTORY",
    "RELEASE_INELIGIBLE_FILENAME",
    "RELEASE_INELIGIBLE_SCHEMA",
    "SUPERVISED_STAGE_DIRECTORY",
    "TRANSLATION_RELEASE_NAME",
]
