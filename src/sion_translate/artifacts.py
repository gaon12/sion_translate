"""Canonical public paths for generated artifacts and training runs.

Compatibility is established by artifact metadata and content hashes, not by a
release-looking directory name.  Keep these paths stable for scripts and users.
"""

from __future__ import annotations

DEFAULT_ARTIFACT_ROOT = "artifacts"
DEFAULT_TOKENIZER_MODEL = f"{DEFAULT_ARTIFACT_ROOT}/tokenizer/sion.model"
DEFAULT_TOKENIZER_FEATURES = f"{DEFAULT_ARTIFACT_ROOT}/tokenizer/token_features.npz"
DEFAULT_DATASET_DIRECTORY = f"{DEFAULT_ARTIFACT_ROOT}/dataset"
DEFAULT_RUN_DIRECTORY = "runs/auto"


__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_DATASET_DIRECTORY",
    "DEFAULT_RUN_DIRECTORY",
    "DEFAULT_TOKENIZER_FEATURES",
    "DEFAULT_TOKENIZER_MODEL",
]
