"""Stable paths for artifacts that must be regenerated as one compatible set."""

from __future__ import annotations


ARTIFACT_LAYOUT_VERSION = "sion-v6"
DEFAULT_ARTIFACT_ROOT = f"artifacts/{ARTIFACT_LAYOUT_VERSION}"
DEFAULT_TOKENIZER_MODEL = f"{DEFAULT_ARTIFACT_ROOT}/tokenizer/sion.model"
DEFAULT_TOKENIZER_FEATURES = f"{DEFAULT_ARTIFACT_ROOT}/tokenizer/token_features.npz"
DEFAULT_DATASET_DIRECTORY = f"{DEFAULT_ARTIFACT_ROOT}/dataset"


__all__ = [
    "ARTIFACT_LAYOUT_VERSION",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_DATASET_DIRECTORY",
    "DEFAULT_TOKENIZER_FEATURES",
    "DEFAULT_TOKENIZER_MODEL",
]
