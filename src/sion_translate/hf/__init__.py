"""Hugging Face Transformers integration for Sion."""

# Transformers Auto registries intentionally accept runtime class objects.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from .configuration_sion import SionConfig
from .conversion import save_transformers_checkpoint
from .modeling_sion import SionForConditionalGeneration
from .tokenization_sion import SionTokenizer


def register_sion_auto_classes() -> None:
    """Register Sion with local Transformers Auto classes idempotently."""

    AutoConfig.register(SionConfig.model_type, SionConfig, exist_ok=True)
    AutoModelForSeq2SeqLM.register(
        SionConfig,
        SionForConditionalGeneration,
        exist_ok=True,
    )
    AutoTokenizer.register(
        SionConfig,
        slow_tokenizer_class=SionTokenizer,
        exist_ok=True,
    )


__all__ = [
    "SionConfig",
    "SionForConditionalGeneration",
    "SionTokenizer",
    "register_sion_auto_classes",
    "save_transformers_checkpoint",
]
