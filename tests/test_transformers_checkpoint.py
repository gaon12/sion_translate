from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from sion_translate.config import ModelConfig
from sion_translate.hf import (
    SionConfig,
    SionForConditionalGeneration,
    register_sion_auto_classes,
    save_transformers_checkpoint,
)
from sion_translate.model import (
    SionForConditionalGeneration as NativeSionForConditionalGeneration,
)
from sion_translate.tokenizer import SionTokenizer as NativeSionTokenizer
from sion_translate.tokenizer import train_tokenizer


def tiny_model_config(vocab_size: int = 128) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
    )


def train_tiny_tokenizer(tmp_path: Path) -> Path:
    source = tmp_path / "parallel.jsonl"
    rows = [
        {
            "ko": f"한국어 체크포인트 예문 {index}입니다.",
            "ja": f"日本語チェックポイント例文{index}です。",
        }
        for index in range(40)
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return train_tokenizer(
        [str(source)],
        tmp_path / "tokenizer",
        vocab_size=512,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        validation_fraction=0.0,
        test_fraction=0.0,
        num_workers=1,
        num_threads=1,
    )


def test_transformers_wrapper_matches_native_forward_and_config() -> None:
    torch.manual_seed(17)
    native_config = tiny_model_config()
    native = NativeSionForConditionalGeneration(native_config, pad_id=0).eval()
    config = SionConfig.from_model_config(
        native_config,
        languages=["ko", "ja"],
        language_pairs=[["ko", "ja"]],
        project_revision="fixture",
    )
    restored_config = SionConfig.from_dict(config.to_dict())
    assert restored_config.to_model_config() == native_config
    assert restored_config.project_revision == "fixture"
    assert restored_config.languages == ["ko", "ja"]

    model = SionForConditionalGeneration(config).eval()
    model.model.load_state_dict(native.state_dict(), strict=True)
    input_ids = torch.tensor([[4, 7, 8, 3], [4, 9, 3, 0]])
    attention_mask = input_ids.ne(0)
    labels = torch.tensor([[11, 12, 3], [13, 3, -100]])
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)

    native_output = native(
        input_ids,
        attention_mask,
        decoder_input_ids,
        labels=labels,
    )
    transformers_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    torch.testing.assert_close(transformers_output.logits, native_output.logits)
    torch.testing.assert_close(transformers_output.loss, native_output.loss)

    model.gradient_checkpointing_enable()
    assert model.config.gradient_checkpointing
    assert model.model.config.gradient_checkpointing
    model.gradient_checkpointing_disable()
    assert not model.config.gradient_checkpointing


def test_transformers_checkpoint_auto_classes_and_safe_weights(
    tmp_path: Path,
) -> None:
    tokenizer_path = train_tiny_tokenizer(tmp_path)
    tokenizer = NativeSionTokenizer(tokenizer_path)
    native_config = tiny_model_config(len(tokenizer))
    torch.manual_seed(23)
    native = NativeSionForConditionalGeneration(native_config, pad_id=tokenizer.pad_id).eval()
    output_dir = tmp_path / "transformers"

    save_transformers_checkpoint(
        output_dir,
        native.state_dict(),
        native_config,
        pad_id=tokenizer.pad_id,
        tokenizer_path=tokenizer_path,
        languages=["ko", "ja"],
        language_pairs=[["ko", "ja"]],
        max_shard_size="1MB",
    )

    required_files = {
        "config.json",
        "configuration_sion.py",
        "generation_config.json",
        "modeling_sion.py",
        "sion_export.json",
        "tokenization_sion.py",
        "tokenizer.model",
        "tokenizer_config.json",
    }
    assert required_files <= {path.name for path in output_dir.iterdir()}
    weight_files = sorted(output_dir.glob("model*.safetensors"))
    assert weight_files
    saved_keys: set[str] = set()
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt") as handle:
            saved_keys.update(handle.keys())
    assert "model.token_embedding.weight" in saved_keys

    remote_model = AutoModelForSeq2SeqLM.from_pretrained(
        output_dir,
        trust_remote_code=True,
    ).eval()
    remote_tokenizer = AutoTokenizer.from_pretrained(
        output_dir,
        trust_remote_code=True,
    )
    assert remote_model.__class__.__module__.startswith("transformers_modules.")
    assert remote_tokenizer.__class__.__module__.startswith("transformers_modules.")
    torch.testing.assert_close(
        remote_model.model.token_embedding.weight,
        native.token_embedding.weight,
    )

    register_sion_auto_classes()
    local_config = AutoConfig.from_pretrained(output_dir, trust_remote_code=False)
    local_model = AutoModelForSeq2SeqLM.from_pretrained(
        output_dir,
        trust_remote_code=False,
    ).eval()
    local_tokenizer = AutoTokenizer.from_pretrained(
        output_dir,
        trust_remote_code=False,
    )
    assert isinstance(local_config, SionConfig)
    assert isinstance(local_model, SionForConditionalGeneration)
    local_tokenizer.src_lang = "ko"
    local_tokenizer.tgt_lang = "ja"
    encoded = local_tokenizer("한국어 번역 테스트입니다.", return_tensors="pt")
    assert encoded.input_ids[0, 0].item() == local_tokenizer.language_tags["ja"]
    generated = local_model.generate(**encoded, max_new_tokens=2)
    assert generated.shape == (1, 3)
