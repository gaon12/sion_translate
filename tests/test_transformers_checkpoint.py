from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
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

import sion_translate.training.export as export_module
import sion_translate.hf.conversion as hf_conversion
from sion_translate.config import ModelConfig
from sion_translate.hf import (
    SionConfig,
    SionForConditionalGeneration,
    SionTokenizer as HFSionTokenizer,
    register_sion_auto_classes,
    save_transformers_checkpoint,
)
from sion_translate.model import (
    SionForConditionalGeneration as NativeSionForConditionalGeneration,
)
from sion_translate.tokenizer import SionTokenizer as NativeSionTokenizer
from sion_translate.tokenizer import train_tokenizer
from sion_translate.training.export import (
    _inspect_transformers_checkpoint,
    build_export_metadata,
    convert_export,
    export_state_dict_formats,
    validate_export_directory,
)


TRANSLATION_PIPELINE_IDENTITY = {
    "schema": "sion-translation-pipeline-v2",
    "branch": "translation-only",
}


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
    foundation_languages: tuple[str, ...] = (),
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
        foundation_languages=foundation_languages,
        num_workers=1,
        num_threads=1,
    )


class _LargeControlVocabulary:
    """SentencePiece layout with language controls beyond token ID 256."""

    def __init__(self, languages: list[str]) -> None:
        from sion_translate.tokenizer import (
            OPTIONAL_CONTROL_SYMBOLS,
            SHARED_CONTROL_SYMBOLS,
            SLOT_SYMBOLS,
        )

        self._pieces = ["<pad>", "<unk>", "<s>", "</s>"]
        self._pieces += [f"<2{language}>" for language in languages]
        self._pieces += [f"<denoise_{language}>" for language in languages]
        self._pieces += SHARED_CONTROL_SYMBOLS + OPTIONAL_CONTROL_SYMBOLS + SLOT_SYMBOLS
        self._pieces += [f"<0x{value:02X}>" for value in range(256)]
        # This is a learned piece, not a reserved language control, despite its
        # surface form.  The byte-fallback boundary must keep it out.
        self._pieces += ["▁alpha", "<2learned>", "▁omega"]
        self._index = {piece: token_id for token_id, piece in enumerate(self._pieces)}

    def vocab_size(self) -> int:
        return len(self._pieces)

    def id_to_piece(self, token_id: int) -> str:
        return self._pieces[token_id]

    def piece_to_id(self, piece: str) -> int:
        return self._index.get(piece, 1)

    def encode(self, _text: str, *, out_type: type[str]) -> list[str]:
        assert out_type is str
        return ["▁alpha"]

    def decode(self, pieces: list[str]) -> str:
        return "".join(pieces)


def test_hf_tokenizer_discovers_every_reserved_language_past_id_256(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import sion_translate.hf.tokenization_sion as tokenizer_module

    languages = [f"l{index:03d}" for index in range(150)]
    processor = _LargeControlVocabulary(languages)
    assert processor.piece_to_id(f"<denoise_{languages[-1]}>") > 256

    def processor_factory(**_kwargs: object) -> _LargeControlVocabulary:
        return processor

    monkeypatch.setattr(
        tokenizer_module.spm,
        "SentencePieceProcessor",
        processor_factory,
    )
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"stub")

    tokenizer = HFSionTokenizer(str(tokenizer_path), translation_capable=False)

    assert set(tokenizer.language_tags) == set(languages)
    assert set(tokenizer.denoise_tags) == set(languages)
    assert "learned" not in tokenizer.language_tags
    assert "<2learned>" not in tokenizer.all_special_tokens


def test_hf_multilingual_tokenizer_requires_an_explicit_pair_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import sion_translate.hf.tokenization_sion as tokenizer_module

    processor = _LargeControlVocabulary(["de", "fr", "sw", "ar"])

    def processor_factory(**_kwargs: object) -> _LargeControlVocabulary:
        return processor

    monkeypatch.setattr(
        tokenizer_module.spm,
        "SentencePieceProcessor",
        processor_factory,
    )
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"stub")

    with pytest.raises(ValueError, match="require language_pairs metadata"):
        HFSionTokenizer(str(tokenizer_path))

    with pytest.raises(ValueError, match="require explicit translation_directions"):
        HFSionTokenizer(
            str(tokenizer_path),
            language_pairs=[["de", "fr"], ["sw", "ar"]],
            release_name="sion_translate",
            release_version="1.5",
        )

    legacy = HFSionTokenizer(
        str(tokenizer_path),
        language_pairs=[["de", "fr"], ["sw", "ar"]],
        release_name="sion_translate",
        release_version="1.4",
    )
    assert legacy.translation_directions == [
        ["de", "fr"],
        ["fr", "de"],
        ["sw", "ar"],
        ["ar", "sw"],
    ]


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


def test_transformers_config_rejects_an_uncovered_language_pair() -> None:
    with pytest.raises(ValueError, match="every language pair"):
        SionConfig.from_model_config(
            tiny_model_config(),
            languages=["de", "fr", "sw", "ar"],
            language_pairs=[["de", "fr"], ["sw", "ar"]],
            translation_directions=[["de", "fr"]],
        )


def test_current_transformers_config_rejects_a_missing_direction_graph() -> None:
    with pytest.raises(ValueError, match="require explicit translation_directions"):
        SionConfig.from_model_config(
            tiny_model_config(),
            languages=["de", "fr"],
            language_pairs=[["de", "fr"]],
            release_name="sion_translate",
            release_version="1.5",
        )


@pytest.mark.parametrize(
    "language_pairs",
    [
        ("de",),
        ({"source": "de", "target": "fr"},),
    ],
)
def test_transformers_config_rejects_non_sequence_pair_entries(
    language_pairs: object,
) -> None:
    with pytest.raises(ValueError, match="two-item language sequence"):
        SionConfig.from_model_config(
            tiny_model_config(),
            language_pairs=language_pairs,  # type: ignore[arg-type]
        )


def test_transformers_conversion_rejects_an_uncovered_pair_before_writing(
    tmp_path: Path,
) -> None:
    config = tiny_model_config()
    native = NativeSionForConditionalGeneration(config, pad_id=0)
    output_dir = tmp_path / "invalid-transformers"

    with pytest.raises(ValueError, match="every language pair"):
        save_transformers_checkpoint(
            output_dir,
            native.state_dict(),
            config,
            languages=["de", "fr", "sw", "ar"],
            language_pairs=[["de", "fr"], ["sw", "ar"]],
            translation_directions=[["de", "fr"]],
        )

    assert not output_dir.exists()


def test_hf_multi_return_beam_keeps_positive_penalty_future_winner_alive() -> None:
    native_config = tiny_model_config()
    config = SionConfig.from_model_config(
        native_config,
        languages=["ko", "ja"],
        language_pairs=[["ko", "ja"]],
    )
    model = SionForConditionalGeneration(config).eval()
    vocab = native_config.vocab_size
    eos_id = int(config.eos_token_id)

    step0 = torch.full((1, vocab), float("-inf"))
    step0[0, eos_id] = torch.log(torch.tensor(0.5))
    step0[0, 10] = torch.log(torch.tensor(0.499))
    step0[0, 11] = torch.log(torch.tensor(0.001))
    step1 = torch.full((1, vocab), float("-inf"))
    step1[0, eos_id] = torch.log(torch.tensor(0.4))
    step1[0, 20] = torch.log(torch.tensor(0.3))
    step1[0, 21] = torch.log(torch.tensor(0.2))
    step1[0, 22] = torch.log(torch.tensor(0.1))
    continuation = torch.full((1, vocab), float("-inf"))
    continuation[0, 30] = 0.0
    steps = iter([step0, step1, *([continuation] * 4)])

    def scripted_logits(hidden: torch.Tensor) -> torch.Tensor:
        plan = next(steps)
        return plan.expand(hidden.shape[0], plan.shape[-1]).clone()

    model.model._logits = scripted_logits  # type: ignore[method-assign]
    input_ids = torch.randint(4, vocab, (1, 5))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    output = model._beam_generate_multiple(
        input_ids,
        attention_mask,
        max_new_tokens=6,
        num_beams=2,
        num_return_sequences=2,
        length_penalty=2.0,
        native_inputs={},
    )

    assert output[0, :3].tolist() == [int(config.decoder_start_token_id), 10, 20]


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
        revision_trained=True,
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
    assert remote_model.config.revision_trained is True
    assert remote_model.config.release_name == "sion_translate"
    assert remote_model.config.release_version == "1.5"
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
    assert local_config.revision_trained is True
    export_metadata = json.loads((output_dir / "sion_export.json").read_text(encoding="utf-8"))
    assert export_metadata["capabilities"]["revision_trained"] is True
    assert export_metadata["release_name"] == "sion_translate"
    assert export_metadata["release_version"] == "1.5"

    for text in (
        "  앞뒤 공백  ",
        "\u1100\u1161",  # decomposed Hangul: NFC must match the native tokenizer
        "\r\n줄바꿈과 탭\t",
        "한국어 <slot_0> 번역",
    ):
        assert local_tokenizer(text, add_special_tokens=False).input_ids == tokenizer.encode(text)

    for source_language, target_language, source_text in (
        ("ko", "ja", "양방향 한국어 입력"),
        ("ja", "ko", "双方向の日本語入力"),
    ):
        directional = local_tokenizer._build_translation_inputs(
            source_text,
            return_tensors="pt",
            src_lang=source_language,
            tgt_lang=target_language,
        )
        assert directional.input_ids[0].tolist() == [
            local_tokenizer.language_tags[target_language],
            *tokenizer.encode(source_text),
            tokenizer.eos_id,
        ]
        shifted = local_model.prepare_decoder_input_ids_from_labels(
            torch.tensor([[17, tokenizer.eos_id]])
        )
        assert shifted[0, 0].item() == tokenizer.bos_id

    generation_config = local_model.generation_config
    assert generation_config.num_beams == 4
    assert generation_config.length_penalty == 1.0
    assert generation_config.max_new_tokens == min(256, native_config.max_seq_len)
    assert generation_config.no_repeat_ngram_size == 4
    suppressed = set(generation_config.suppress_tokens)
    expected_control_ids = {
        tokenizer.pad_id,
        tokenizer.unk_id,
        tokenizer.bos_id,
        tokenizer.mask_id,
        *tokenizer.language_tags.values(),
        *tokenizer.denoise_tags.values(),
    }
    assert expected_control_ids <= suppressed
    assert tokenizer.eos_id not in suppressed
    assert tokenizer.slot_ids[0] not in suppressed

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
    assert generated.shape[0] == 1
    assert 2 <= generated.shape[1] <= 3
    assert generated[0, 0].item() == tokenizer.bos_id
    assert captured_generate["num_beams"] == 4
    assert captured_generate["no_repeat_ngram_size"] == 4
    assert set(captured_generate["forbidden_token_ids"]) == suppressed
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
    captured_generate.clear()
    explicitly_overridden = local_model.generate(
        **encoded,
        max_new_tokens=2,
        num_beams=1,
        no_repeat_ngram_size=0,
        suppress_tokens=[],
    )
    assert explicitly_overridden.shape[0] == 1
    assert 2 <= explicitly_overridden.shape[1] <= 3
    assert captured_generate["num_beams"] == 1
    assert captured_generate["no_repeat_ngram_size"] == 0
    assert captured_generate["forbidden_token_ids"] == ()
    trainer_style = local_model.generate(
        **encoded,
        labels=torch.tensor([[11, 3]]),
        max_new_tokens=2,
        generation_config=local_model.generation_config,
        synced_gpus=False,
    )
    assert trainer_style.shape[0] == 1
    assert 2 <= trainer_style.shape[1] <= 3
    inputs_alias = local_model.generate(
        inputs=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        max_new_tokens=2,
    )
    assert inputs_alias.shape[0] == 1
    assert 2 <= inputs_alias.shape[1] <= 3
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
    assert sampled.sequences.shape[0] == 3
    assert 2 <= sampled.sequences.shape[1] <= 4
    assert captured_sample["num_samples"] == 3
    assert "src_script_ids" in captured_sample
    beam_hypotheses = local_model.generate(
        **encoded,
        num_beams=3,
        num_return_sequences=2,
        max_length=4,
    )
    assert beam_hypotheses.shape[0] == 2
    assert 2 <= beam_hypotheses.shape[1] <= 4
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
        # Keep the ``python -c`` payload ASCII-only so Windows does not corrupt
        # the Korean sample while transferring it through the command line.
        encoded = tokenizer("\\ub3c5\\ub9bd \\uc2e4\\ud589 <slot_0>", return_tensors="pt")
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


def test_candidate_refinement_transformers_checkpoint_is_self_contained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_config = tiny_model_config()
    native_config.experimental.candidate_refinement_enabled = True
    native_config.experimental.candidate_refinement_steps = 1
    native_config.experimental.candidate_refinement_vocab_chunk_size = 16
    native = NativeSionForConditionalGeneration(native_config, pad_id=0).eval()
    save_transformers_checkpoint(tmp_path, native.state_dict(), native_config)

    generation = json.loads((tmp_path / "generation_config.json").read_text(encoding="utf-8"))
    serialized_config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / "sion_export.json").read_text(encoding="utf-8"))
    assert generation["reasoning_level"] == 9
    assert serialized_config["default_reasoning_level"] == 9
    assert metadata["generation_defaults"]["reasoning_level"] == 9
    restored = AutoModelForSeq2SeqLM.from_pretrained(tmp_path, trust_remote_code=True).eval()
    assert restored.model.candidate_refinement is not None
    assert restored.config.default_reasoning_level == 9
    assert restored.generation_config.reasoning_level == 9
    input_ids = torch.tensor([[4, 5, 3]])
    output = restored(
        input_ids=input_ids,
        attention_mask=input_ids.ne(0),
        decoder_input_ids=torch.tensor([[2, 6]]),
    )
    assert output.logits.shape == (1, 2, native_config.vocab_size)

    captured_generate: dict[str, object] = {}
    native_generate = restored.model.generate

    def capture_generate(*args, **kwargs):
        captured_generate.update(kwargs)
        return native_generate(*args, **kwargs)

    monkeypatch.setattr(restored.model, "generate", capture_generate)
    encoded = {"input_ids": input_ids, "attention_mask": input_ids.ne(0)}
    restored.generate(**encoded, max_new_tokens=1)
    assert captured_generate["reasoning_level"] == 9
    captured_generate.clear()
    restored.generate(**encoded, max_new_tokens=1, reasoning_level=0)
    assert captured_generate["reasoning_level"] == 0
    captured_generate.clear()

    for invalid_type in (True, 1.0, "1"):
        with pytest.raises(TypeError, match="integer from 0 to 9"):
            restored.generate(
                **encoded,
                max_new_tokens=1,
                reasoning_level=invalid_type,
            )
        assert not captured_generate
    for out_of_range in (-1, 10):
        with pytest.raises(ValueError, match="between 0 and 9"):
            restored.generate(
                **encoded,
                max_new_tokens=1,
                reasoning_level=out_of_range,
            )
        assert not captured_generate

    round_trip = tmp_path / "round-trip"
    restored.save_pretrained(round_trip)
    reloaded = AutoModelForSeq2SeqLM.from_pretrained(round_trip, trust_remote_code=True).eval()
    assert reloaded.config.default_reasoning_level == 9
    assert reloaded.generation_config.reasoning_level == 9

    contradictory_checkpoint = tmp_path / "contradictory"
    shutil.copytree(round_trip, contradictory_checkpoint)
    round_trip_generation = json.loads(
        (contradictory_checkpoint / "generation_config.json").read_text(encoding="utf-8")
    )
    round_trip_generation["reasoning_level"] = 0
    (contradictory_checkpoint / "generation_config.json").write_text(
        json.dumps(round_trip_generation),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="default_reasoning_level.*disagree"):
        AutoModelForSeq2SeqLM.from_pretrained(
            contradictory_checkpoint,
            trust_remote_code=True,
        )


@pytest.mark.parametrize("default_reasoning_level", [True, 1.0, "1"])
def test_transformers_config_rejects_non_integer_reasoning_default(
    default_reasoning_level: object,
) -> None:
    with pytest.raises(TypeError, match="integer from 0 to 9"):
        SionConfig(  # pyright: ignore[reportArgumentType]
            default_reasoning_level=default_reasoning_level,
        )


@pytest.mark.parametrize("default_reasoning_level", [-1, 10])
def test_transformers_config_rejects_out_of_range_reasoning_default(
    default_reasoning_level: int,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 9"):
        SionConfig(default_reasoning_level=default_reasoning_level)


def test_transformers_config_rejects_non_boolean_revision_capability() -> None:
    with pytest.raises(ValueError, match="revision_trained must be a boolean"):
        SionConfig(revision_trained="true")


@pytest.mark.parametrize(
    ("release_name", "release_version"),
    [(123, "1.5"), ("sion", 1.5), ("", "1.5"), ("sion", "")],
)
def test_transformers_config_rejects_invalid_release_identity(
    release_name: object,
    release_version: object,
) -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        SionConfig(  # pyright: ignore[reportArgumentType]
            release_name=release_name,
            release_version=release_version,
        )


@pytest.mark.parametrize("release_version", ["1", "v1.5", "1.5-beta"])
def test_transformers_config_rejects_malformed_release_version(
    release_version: str,
) -> None:
    with pytest.raises(ValueError, match="numeric major.minor"):
        SionConfig(release_name="sion", release_version=release_version)


@pytest.mark.parametrize(
    ("release_name", "translation_capable"),
    [("sion", True), ("sion_translate", False)],
)
def test_transformers_config_rejects_contradictory_repository_role(
    release_name: str,
    translation_capable: bool,
) -> None:
    with pytest.raises(ValueError, match="translation-capable"):
        SionConfig(
            release_name=release_name,
            release_version="1.5",
            translation_capable=translation_capable,
        )


def test_export_validation_enforces_expected_repository_role(tmp_path: Path) -> None:
    config = tiny_model_config()
    native = NativeSionForConditionalGeneration(config, pad_id=0)
    metadata = build_export_metadata(
        config,
        pipeline_identity=TRANSLATION_PIPELINE_IDENTITY,
    )
    manifest = export_state_dict_formats(
        tmp_path,
        native.state_dict(),
        config,
        0,
        formats=("fp32",),
        metadata=metadata,
    )
    assert manifest["metadata"]["release_name"] == "sion_translate"

    mismatch = validate_export_directory(
        tmp_path,
        expected_release_name="sion",
        expected_release_version="1.5",
        expected_translation_capable=False,
    )
    assert not mismatch["valid"]
    assert {error["error_type"] for error in mismatch["errors"]} == {"UnexpectedIdentity"}

    manifest_path = tmp_path / "export_manifest.json"
    corrupted = json.loads(manifest_path.read_text(encoding="utf-8"))
    corrupted["metadata"]["release_name"] = "sion"
    manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
    invalid_role = validate_export_directory(tmp_path)
    assert not invalid_role["valid"]
    assert "InvalidReleaseIdentity" in {error["error_type"] for error in invalid_role["errors"]}

    with pytest.raises(ValueError, match="expected sion foundation identity"):
        validate_export_directory(
            tmp_path,
            expected_release_name="sion",
            expected_release_version="1.5",
            expected_translation_capable=True,
        )


def test_transformers_inspection_rejects_internal_release_disagreement(
    tmp_path: Path,
) -> None:
    config = tiny_model_config()
    native = NativeSionForConditionalGeneration(config, pad_id=0)
    save_transformers_checkpoint(tmp_path, native.state_dict(), config)
    export_path = tmp_path / "sion_export.json"
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    payload["release_version"] = "1.4"
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="disagree about release_version"):
        _inspect_transformers_checkpoint(tmp_path)


def test_transformers_inspection_subprocess_has_a_finite_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def time_out(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == export_module._TRANSFORMERS_INSPECTION_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(cmd="inspection", timeout=float(kwargs["timeout"]))

    monkeypatch.setattr(export_module.subprocess, "run", time_out)

    with pytest.raises(RuntimeError, match="inspection timed out"):
        _inspect_transformers_checkpoint(tmp_path)


def test_transformers_inspection_rejects_tampered_reasoning_endpoint(
    tmp_path: Path,
) -> None:
    config = tiny_model_config()
    config.experimental.candidate_refinement_enabled = True
    native = NativeSionForConditionalGeneration(config, pad_id=0)
    save_transformers_checkpoint(tmp_path, native.state_dict(), config)
    export_path = tmp_path / "sion_export.json"
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    payload["generation_defaults"] = {"reasoning_level": 0}
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match model features"):
        _inspect_transformers_checkpoint(tmp_path)


def test_transformers_inspection_rejects_tampered_generation_reasoning_endpoint(
    tmp_path: Path,
) -> None:
    config = tiny_model_config()
    config.experimental.candidate_refinement_enabled = True
    native = NativeSionForConditionalGeneration(config, pad_id=0)
    save_transformers_checkpoint(tmp_path, native.state_dict(), config)
    generation_path = tmp_path / "generation_config.json"
    payload = json.loads(generation_path.read_text(encoding="utf-8"))
    payload["reasoning_level"] = 0
    generation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="disagree about reasoning_level"):
        _inspect_transformers_checkpoint(tmp_path)


def test_transformers_inspection_binds_reasoning_sidecars_to_config_default(
    tmp_path: Path,
) -> None:
    config = tiny_model_config()
    config.experimental.candidate_refinement_enabled = True
    native = NativeSionForConditionalGeneration(config, pad_id=0)
    save_transformers_checkpoint(tmp_path, native.state_dict(), config)
    config_path = tmp_path / "config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["default_reasoning_level"] = 8
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="config and generation sidecars disagree"):
        _inspect_transformers_checkpoint(tmp_path)


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
    with pytest.raises(ValueError, match="language tags"):
        save_transformers_checkpoint(
            tmp_path / "incomplete-languages",
            native.state_dict(),
            config,
            tokenizer_path=tokenizer_path,
            languages=["ko"],
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
    with pytest.raises(ValueError, match="unsupported translation direction"):
        restored._build_translation_inputs(
            "연결되지 않은 방향",
            return_tensors="pt",
            src_lang="ko",
            tgt_lang="ru",
        )


def test_transformers_tokenizer_enforces_trained_direction(tmp_path: Path) -> None:
    tokenizer_path = train_tiny_tokenizer(tmp_path)
    tokenizer = NativeSionTokenizer(tokenizer_path)
    config = tiny_model_config(len(tokenizer))
    native = NativeSionForConditionalGeneration(config, pad_id=tokenizer.pad_id)
    output_dir = tmp_path / "unidirectional-transformers"
    save_transformers_checkpoint(
        output_dir,
        native.state_dict(),
        config,
        pad_id=tokenizer.pad_id,
        tokenizer_path=tokenizer_path,
        language_pairs=[("ko", "ja")],
        translation_directions=[("ko", "ja")],
    )
    restored = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    encoded = restored._build_translation_inputs(
        "정방향",
        return_tensors="pt",
        src_lang="ko",
        tgt_lang="ja",
    )
    assert encoded.input_ids.shape[0] == 1
    with pytest.raises(ValueError, match="unsupported translation direction"):
        restored._build_translation_inputs(
            "역방향",
            return_tensors="pt",
            src_lang="ja",
            tgt_lang="ko",
        )
    tokenizer_metadata = json.loads(
        (output_dir / "tokenizer_metadata.json").read_text(encoding="utf-8")
    )
    assert tokenizer_metadata["language_pairs"] == [["ko", "ja"]]
    assert tokenizer_metadata["translation_directions"] == [["ko", "ja"]]
    for sidecar_name in ("config.json", "sion_export.json"):
        sidecar_path = output_dir / sidecar_name
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["pipeline"] = TRANSLATION_PIPELINE_IDENTITY
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert _inspect_transformers_checkpoint(output_dir)["translation_directions"] == [["ko", "ja"]]

    tokenizer_metadata["translation_directions"].append(["ja", "ko"])
    (output_dir / "tokenizer_metadata.json").write_text(
        json.dumps(tokenizer_metadata),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="tokenizer metadata disagree"):
        _inspect_transformers_checkpoint(output_dir)


def test_transformers_export_failure_leaves_no_partial_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_path = train_tiny_tokenizer(tmp_path)
    tokenizer = NativeSionTokenizer(tokenizer_path)
    config = tiny_model_config(len(tokenizer))
    native = NativeSionForConditionalGeneration(config, pad_id=tokenizer.pad_id)
    output_dir = tmp_path / "failed-transformers"

    def fail_runtime_copy(_output_dir: Path) -> list[str]:
        raise RuntimeError("runtime copy failed")

    monkeypatch.setattr(hf_conversion, "_copy_self_contained_runtime", fail_runtime_copy)

    with pytest.raises(RuntimeError, match="runtime copy failed"):
        save_transformers_checkpoint(
            output_dir,
            native.state_dict(),
            config,
            pad_id=tokenizer.pad_id,
            tokenizer_path=tokenizer_path,
            language_pairs=[["ko", "ja"]],
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(f".{output_dir.name}.staging-*"))

    output_dir.mkdir()
    marker = output_dir / "complete.marker"
    marker.write_text("previous generation", encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime copy failed"):
        save_transformers_checkpoint(
            output_dir,
            native.state_dict(),
            config,
            pad_id=tokenizer.pad_id,
            tokenizer_path=tokenizer_path,
            language_pairs=[["ko", "ja"]],
        )
    assert marker.read_text(encoding="utf-8") == "previous generation"
    assert {path.name for path in output_dir.iterdir()} == {"complete.marker"}


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
    metadata = build_export_metadata(
        config,
        tokenizer_path=tokenizer_path,
        language_pairs=(("ko", "ja"),),
        pipeline_identity=TRANSLATION_PIPELINE_IDENTITY,
    )
    manifest = export_state_dict_formats(
        output_dir,
        model.state_dict(),
        config,
        tokenizer.pad_id,
        formats=("transformers",),
        metadata=metadata,
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


def test_transformers_export_promotes_tokenizer_language_discovery_to_manifest(
    tmp_path: Path,
) -> None:
    tokenizer_path = train_tiny_tokenizer(tmp_path)
    tokenizer = NativeSionTokenizer(tokenizer_path)
    config = tiny_model_config(len(tokenizer))
    model = NativeSionForConditionalGeneration(config, pad_id=tokenizer.pad_id)
    output_dir = tmp_path / "legacy-export"
    metadata = build_export_metadata(
        config,
        tokenizer_path=tokenizer_path,
        pipeline_identity=TRANSLATION_PIPELINE_IDENTITY,
    )

    manifest = export_state_dict_formats(
        output_dir,
        model.state_dict(),
        config,
        tokenizer.pad_id,
        formats=("transformers",),
        metadata=metadata,
        tokenizer_path=tokenizer_path,
    )

    assert set(manifest["metadata"]["languages"]) == set(tokenizer.denoise_languages)
    checkpoint = output_dir / manifest["formats"]["transformers"]["file"]
    config_payload = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    assert set(config_payload["languages"]) == set(tokenizer.languages)
    assert validate_export_directory(output_dir)["valid"]


def test_foundation_conversion_preserves_release_and_rejects_translation(
    tmp_path: Path,
) -> None:
    tokenizer_path = train_tiny_tokenizer(
        tmp_path,
        foundation_languages=("ko", "ja", "en"),
    )
    tokenizer = NativeSionTokenizer(tokenizer_path)
    assert set(tokenizer.languages) == {"ko", "ja"}
    assert set(tokenizer.denoise_languages) == {"ko", "ja", "en"}
    config = tiny_model_config(len(tokenizer))
    model = NativeSionForConditionalGeneration(config, pad_id=tokenizer.pad_id)
    source_dir = tmp_path / "foundation-native"
    metadata = build_export_metadata(
        config,
        tokenizer_path=tokenizer_path,
        languages=["ko", "ja", "en"],
        release_name="sion",
        translation_capable=False,
    )
    export_state_dict_formats(
        source_dir,
        model.state_dict(),
        config,
        tokenizer.pad_id,
        formats=("fp32",),
        metadata=metadata,
        tokenizer_path=tokenizer_path,
    )

    converted = convert_export(
        source_dir / "model.pt",
        tmp_path / "foundation-transformers",
        formats=("transformers",),
    )

    assert converted["metadata"]["release_name"] == "sion"
    assert converted["metadata"]["translation_capable"] is False
    assert converted["metadata"]["release_version"] == "1.5"
    assert converted["metadata"]["languages"] == ["ko", "ja", "en"]
    checkpoint = tmp_path / "foundation-transformers" / converted["formats"]["transformers"]["file"]
    restored = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    assert restored.translation_capable is False
    with pytest.raises(ValueError, match="foundation model.*not translation-capable"):
        restored._build_translation_inputs(
            "번역 불가",
            return_tensors="pt",
            src_lang="ko",
            tgt_lang="ja",
        )
    assert validate_export_directory(tmp_path / "foundation-transformers")["valid"]
