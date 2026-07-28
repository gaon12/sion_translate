from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

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
from sion_translate.training.export import (
    export_state_dict_formats,
    validate_export_directory,
)


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


def train_tiny_tokenizer(
    tmp_path: Path,
    *,
    language_pairs: list[tuple[str, str]] | None = None,
) -> Path:
    source = tmp_path / "parallel.jsonl"
    rows = [
        {
            "ko": f"한국어 체크포인트 예문 {index}입니다.",
            "ja": f"日本語チェックポイント例文{index}です。",
            "en": f"English checkpoint sentence {index}.",
            "ru": f"Русское предложение контрольной точки {index}.",
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
        language_pairs=language_pairs,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_path = train_tiny_tokenizer(tmp_path)
    tokenizer = NativeSionTokenizer(tokenizer_path)
    native_config = tiny_model_config(len(tokenizer))
    native_config.experimental.morphoscript_enabled = True
    native_config.experimental.tetm_enabled = True
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
        "sion_native_config.py",
        "sion_native_experimental.py",
        "sion_native_layers.py",
        "sion_native_transformer.py",
        "sion_export.json",
        "token_features.npz",
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
    assert remote_model.model.__class__.__module__.startswith("transformers_modules.")
    assert remote_model.config.slot_token_ids == tokenizer.slot_ids
    assert remote_model.config.token_features_shapes == {
        "script": [len(tokenizer)],
        "onset": [len(tokenizer)],
        "vowel": [len(tokenizer)],
        "coda": [len(tokenizer)],
    }
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
    encoded = local_tokenizer(
        "한국어 <slot_0> 번역 <slot_0> 테스트입니다.",
        return_tensors="pt",
    )
    assert encoded.input_ids[0, 0].item() == local_tokenizer.language_tags["ja"]
    for feature_name in (
        "src_script_ids",
        "src_onset_ids",
        "src_vowel_ids",
        "src_coda_ids",
    ):
        assert encoded[feature_name].shape == encoded.input_ids.shape
    assert encoded.memory_token_ids.shape == (1, 2, 1)
    assert encoded.memory_mask.tolist() == [[True, True]]
    assert encoded.memory_type_ids.tolist() == [[8, 8]]
    assert encoded.memory_mode_ids.tolist() == [[4, 4]]
    captured_generate: dict[str, object] = {}
    native_generate = local_model.model.generate

    def capture_generate(*args, **kwargs):
        captured_generate.update(kwargs)
        return native_generate(*args, **kwargs)

    monkeypatch.setattr(local_model.model, "generate", capture_generate)
    generated = local_model.generate(**encoded, max_new_tokens=2)
    assert generated.shape == (1, 3)
    assert {
        "src_script_ids",
        "src_onset_ids",
        "src_vowel_ids",
        "src_coda_ids",
        "memory_token_ids",
        "memory_mask",
        "memory_type_ids",
        "memory_mode_ids",
    } <= captured_generate.keys()
    trainer_style = local_model.generate(
        **encoded,
        labels=torch.tensor([[11, 3]]),
        max_new_tokens=2,
        generation_config=local_model.generation_config,
        synced_gpus=False,
    )
    assert trainer_style.shape == (1, 3)
    inputs_alias = local_model.generate(
        inputs=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        max_new_tokens=2,
    )
    assert inputs_alias.shape == (1, 3)
    with pytest.raises(ValueError, match="only one of inputs or input_ids"):
        local_model.generate(
            inputs=encoded.input_ids,
            input_ids=encoded.input_ids,
            max_new_tokens=2,
        )
    with pytest.raises(NotImplementedError, match="synced_gpus=True"):
        local_model.generate(**encoded, max_new_tokens=2, synced_gpus=True)
    attention_config = copy.deepcopy(local_model.generation_config)
    attention_config.output_attentions = True
    with pytest.raises(NotImplementedError, match="output_attentions"):
        local_model.generate(
            **encoded,
            max_new_tokens=2,
            generation_config=attention_config,
        )

    collator = DataCollatorForSeq2Seq(
        tokenizer=local_tokenizer,
        model=local_model,
        return_tensors="pt",
    )
    collated = collator(
        [
            local_tokenizer("짧은 문장"),
            local_tokenizer("길이가 서로 다른 문장을 표준 collator로 묶습니다."),
        ]
    )
    assert collated["src_script_ids"].shape == collated["input_ids"].shape
    assert collated["memory_token_ids"].shape[:2] == (2, 1)
    assert not collated["memory_mask"].any()
    tokenizer_copy = tmp_path / "tokenizer-copy"
    local_tokenizer.save_pretrained(tokenizer_copy)
    assert (tokenizer_copy / "token_features.npz").is_file()
    reloaded_copy = AutoTokenizer.from_pretrained(tokenizer_copy, trust_remote_code=True)
    assert reloaded_copy.token_features is not None

    captured_sample: dict[str, object] = {}
    native_sample = local_model.model.sample

    def capture_sample(*args, **kwargs):
        captured_sample.update(kwargs)
        return native_sample(*args, **kwargs)

    monkeypatch.setattr(local_model.model, "sample", capture_sample)
    sampling_encoded = {
        name: value for name, value in encoded.items() if not name.startswith("memory_")
    }
    sampled = local_model.generate(
        **sampling_encoded,
        do_sample=True,
        temperature=0.8,
        top_k=8,
        num_return_sequences=3,
        max_length=4,
        return_dict_in_generate=True,
        generator=torch.Generator().manual_seed(9),
    )
    assert sampled.sequences.shape == (3, 4)
    assert captured_sample["num_samples"] == 3
    assert "src_script_ids" in captured_sample
    beam_hypotheses = local_model.generate(
        **encoded,
        num_beams=3,
        num_return_sequences=2,
        max_length=4,
    )
    assert beam_hypotheses.shape == (2, 4)
    with pytest.raises(NotImplementedError, match="top_p"):
        local_model.generate(**encoded, max_new_tokens=2, top_p=0.9)
    with pytest.raises(NotImplementedError, match="output_scores"):
        local_model.generate(**encoded, max_new_tokens=2, output_scores=True)

    clean_environment = textwrap.dedent(
        """
        import importlib.abc
        import sys

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        class BlockSionPackage(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "sion_translate" or fullname.startswith("sion_translate."):
                    raise ModuleNotFoundError(
                        "sion_translate intentionally hidden for self-contained export test"
                    )
                return None

        sys.meta_path.insert(0, BlockSionPackage())
        checkpoint = sys.argv[1]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        ).eval()
        tokenizer.src_lang = "ko"
        tokenizer.tgt_lang = "ja"
        encoded = tokenizer("독립 실행 <slot_0>", return_tensors="pt")
        generated = model.generate(**encoded, max_length=3)
        assert generated.shape[0] == 1
        assert generated.shape[1] <= 3
        """
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, "-c", clean_environment, str(output_dir)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_transformers_checkpoint_preserves_source_precision(tmp_path: Path) -> None:
    native_config = tiny_model_config()
    native = NativeSionForConditionalGeneration(native_config, pad_id=0).eval()
    for name, dtype, serialized_dtype in (
        ("fp32", torch.float32, "float32"),
        ("fp16", torch.float16, "float16"),
        ("bf16", torch.bfloat16, "bfloat16"),
    ):
        state = {
            key: value.to(dtype=dtype) if value.is_floating_point() else value
            for key, value in native.state_dict().items()
        }
        output_dir = tmp_path / name
        save_transformers_checkpoint(
            output_dir,
            state,
            native_config,
        )
        saved_dtypes: set[torch.dtype] = set()
        for weight_file in output_dir.glob("model*.safetensors"):
            with safe_open(weight_file, framework="pt") as handle:
                saved_dtypes.update(handle.get_tensor(key).dtype for key in handle.keys())
        assert saved_dtypes == {dtype}
        metadata = json.loads((output_dir / "sion_export.json").read_text())
        assert metadata["dtype"] == serialized_dtype


def test_transformers_export_rejects_incompatible_tokenizer_and_features(
    tmp_path: Path,
) -> None:
    tokenizer_path = train_tiny_tokenizer(tmp_path)
    tokenizer = NativeSionTokenizer(tokenizer_path)
    config = tiny_model_config(len(tokenizer))
    native = NativeSionForConditionalGeneration(config, pad_id=tokenizer.pad_id).eval()

    with pytest.raises(ValueError, match="vocabulary size"):
        save_transformers_checkpoint(
            tmp_path / "wrong-vocab",
            native.state_dict(),
            tiny_model_config(len(tokenizer) + 1),
            tokenizer_path=tokenizer_path,
        )
    with pytest.raises(ValueError, match="pad ID"):
        save_transformers_checkpoint(
            tmp_path / "wrong-pad",
            native.state_dict(),
            config,
            pad_id=tokenizer.pad_id + 1,
            tokenizer_path=tokenizer_path,
        )
    with pytest.raises(ValueError, match="language tags"):
        save_transformers_checkpoint(
            tmp_path / "wrong-languages",
            native.state_dict(),
            config,
            tokenizer_path=tokenizer_path,
            languages=["ko", "en"],
        )

    wrong_features = tmp_path / "wrong-token-features.npz"
    np.savez_compressed(
        wrong_features,
        script=np.zeros(len(tokenizer) - 1, dtype=np.uint8),
        onset=np.zeros(len(tokenizer) - 1, dtype=np.uint8),
        vowel=np.zeros(len(tokenizer) - 1, dtype=np.uint8),
        coda=np.zeros(len(tokenizer) - 1, dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="expected"):
        save_transformers_checkpoint(
            tmp_path / "wrong-features",
            native.state_dict(),
            config,
            tokenizer_path=tokenizer_path,
            token_features_path=wrong_features,
        )


def test_transformers_tokenizer_rejects_disconnected_language_edge(tmp_path: Path) -> None:
    tokenizer_path = train_tiny_tokenizer(
        tmp_path,
        language_pairs=[("ko", "ja"), ("en", "ru")],
    )
    tokenizer = NativeSionTokenizer(tokenizer_path)
    config = tiny_model_config(len(tokenizer))
    native = NativeSionForConditionalGeneration(config, pad_id=tokenizer.pad_id)
    output_dir = tmp_path / "multilingual-transformers"
    save_transformers_checkpoint(
        output_dir,
        native.state_dict(),
        config,
        pad_id=tokenizer.pad_id,
        tokenizer_path=tokenizer_path,
        language_pairs=[("ko", "ja"), ("en", "ru")],
    )
    restored = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    with pytest.raises(ValueError, match="unsupported translation edge"):
        restored._build_translation_inputs(
            "연결되지 않은 방향",
            return_tensors="pt",
            src_lang="ko",
            tgt_lang="ru",
        )


def test_hf_tokenizer_caps_protected_slot_occurrences_and_checks_feature_hash(
    tmp_path: Path,
) -> None:
    tokenizer_path = train_tiny_tokenizer(tmp_path)
    tokenizer = NativeSionTokenizer(tokenizer_path)
    config = tiny_model_config(len(tokenizer))
    native = NativeSionForConditionalGeneration(config, pad_id=tokenizer.pad_id).eval()
    output_dir = tmp_path / "transformers"
    save_transformers_checkpoint(
        output_dir,
        native.state_dict(),
        config,
        tokenizer_path=tokenizer_path,
    )
    restored = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    encoded = restored(" ".join(["<slot_0>"] * 70), return_tensors="np")
    assert encoded["memory_token_ids"].shape == (1, 64, 1)
    assert encoded["memory_mask"].all()

    features_path = output_dir / "token_features.npz"
    raw = bytearray(features_path.read_bytes())
    raw[-1] ^= 1
    features_path.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)


def test_export_pipeline_writes_and_validates_hf_tokenizer(tmp_path: Path) -> None:
    tokenizer_path = train_tiny_tokenizer(tmp_path)
    tokenizer = NativeSionTokenizer(tokenizer_path)
    config = tiny_model_config(len(tokenizer))
    model = NativeSionForConditionalGeneration(config, pad_id=tokenizer.pad_id)
    output_dir = tmp_path / "export"
    manifest = export_state_dict_formats(
        output_dir,
        model.state_dict(),
        config,
        tokenizer.pad_id,
        formats=("transformers",),
        tokenizer_path=tokenizer_path,
        language_pairs=(("ko", "ja"),),
    )
    entry = manifest["formats"]["transformers"]
    assert entry["status"] == "ok"
    checkpoint = output_dir / entry["file"]
    assert (checkpoint / "tokenizer.model").is_file()
    assert validate_export_directory(output_dir)["valid"]
    restored = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    assert restored.vocab_size == len(tokenizer)
