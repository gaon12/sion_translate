"""Conversion helpers for standard Transformers checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Sequence

import torch

from sion_translate.config import ModelConfig
from sion_translate.tokenizer import SionTokenizer as NativeSionTokenizer

from .configuration_sion import SionConfig
from .modeling_sion import SionForConditionalGeneration
from .tokenization_sion import SionTokenizer


def save_transformers_checkpoint(
    output_dir: str | Path,
    state_dict: dict[str, torch.Tensor],
    model_config: ModelConfig,
    *,
    pad_id: int = 0,
    tokenizer_path: str | Path | None = None,
    languages: Sequence[str] | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
    max_shard_size: str = "5GB",
) -> Path:
    """Save native Sion weights as a safe, AutoClass-compatible directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bos_id = 2
    eos_id = 3
    tokenizer: NativeSionTokenizer | None = None
    if tokenizer_path is not None:
        tokenizer = NativeSionTokenizer(tokenizer_path)
        bos_id = tokenizer.bos_id
        eos_id = tokenizer.eos_id
        if languages is None:
            languages = tokenizer.languages
    pairs = [list(pair) for pair in (language_pairs or [])]
    SionConfig.register_for_auto_class()
    SionForConditionalGeneration.register_for_auto_class("AutoModelForSeq2SeqLM")
    SionTokenizer.register_for_auto_class()
    config = SionConfig.from_model_config(
        model_config,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        languages=list(languages or []),
        language_pairs=pairs,
    )
    model = SionForConditionalGeneration(config)
    model.model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    if tokenizer_path is not None:
        hf_tokenizer = SionTokenizer(
            str(tokenizer_path),
            model_max_length=model_config.max_seq_len,
        )
        hf_tokenizer.save_pretrained(output_dir)

    module_dir = Path(__file__).parent
    for filename in (
        "configuration_sion.py",
        "modeling_sion.py",
        "tokenization_sion.py",
    ):
        shutil.copyfile(module_dir / filename, output_dir / filename)
    metadata = {
        "format": "transformers-safetensors-v1",
        "languages": list(languages or []),
        "language_pairs": pairs,
        "native_state_dict_prefix": "model.",
        "transformers_auto_classes": [
            "AutoConfig",
            "AutoModelForSeq2SeqLM",
            "AutoTokenizer",
        ],
        "requires": ["sion-translate", "sentencepiece", "transformers>=5,<6"],
    }
    (output_dir / "sion_export.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir
