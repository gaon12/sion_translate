from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import sion_translate.fp8_runtime as fp8_runtime
import sion_translate.inference as inference
from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.fp8 import FORWARD_DTYPE, Fp8Policy
from sion_translate.fp8_runtime import Fp8Linear, apply_fp8_weights
from sion_translate.model import SionForConditionalGeneration
from sion_translate.training.export import _pack_fp8_state


_UNSET = object()


class FakeTokenizer:
    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3
    mask_id = 6
    language_tags = {"ja": 4, "ko": 5}
    denoise_tags = {"ja": 7, "ko": 8}
    reasoning_tags = {"ja": 16}
    reasoning_trace_ids = {
        "<think>": 17,
        "</think>": 18,
        "<answer>": 19,
        "</answer>": 21,
    }
    slot_ids = [20]
    draft_id = 9
    splits_digits = True
    languages = ("ja", "ko")

    def __init__(self, _path):
        pass

    def __len__(self):
        return 64

    @staticmethod
    def encode(_text):
        return [20, 11]

    @staticmethod
    def decode(ids):
        return " ".join(map(str, ids))


class ArbitraryGraphFakeTokenizer(FakeTokenizer):
    language_tags = {"de": 4, "fr": 5, "sw": 12, "ar": 13}
    denoise_tags = {"de": 7, "fr": 8, "sw": 14, "ar": 15}
    reasoning_tags = {}
    languages = ("de", "fr", "sw", "ar")


class BCP47GraphFakeTokenizer(FakeTokenizer):
    language_tags = {"pt-BR": 4, "zh-Hant": 5}
    denoise_tags = {"pt-BR": 7, "zh-Hant": 8}
    reasoning_tags = {}
    languages = ("pt-BR", "zh-Hant")


def runtime_config(*, experimental: ExperimentalConfig | None = None) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
        gradient_checkpointing=False,
        experimental=experimental or ExperimentalConfig(),
    )


def make_translator(
    monkeypatch,
    tmp_path: Path,
    config: ModelConfig,
    *,
    revision_trained: bool | None = False,
    language_pairs: tuple[tuple[str, str], ...] = (("ko", "ja"),),
    translation_directions: tuple[tuple[str, str], ...] | None = None,
    tokenizer_class: type[FakeTokenizer] = FakeTokenizer,
    quantization: dict[str, object] | None = None,
    feature_sha256: str | None = None,
    feature_filename: str = "token_features.npz",
    explicit_features: bool = True,
    device: str = "cpu",
    raw_revision_capability: object = _UNSET,
    feature_arrays: dict[str, np.ndarray] | None = None,
    loaded_model: SionForConditionalGeneration | None = None,
) -> inference.Translator:
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"fake tokenizer")
    digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    model = loaded_model if loaded_model is not None else SionForConditionalGeneration(config)
    features = tmp_path / feature_filename
    zeros = np.zeros(64, dtype=np.uint8)
    np.savez_compressed(
        features,
        **(
            feature_arrays
            or {
                "script": zeros,
                "onset": zeros,
                "vowel": zeros,
                "coda": zeros,
            }
        ),
    )
    metadata: dict[str, object] = {
        "tokenizer": {"sha256": digest},
        "token_features": {
            "filename": features.name,
            "size": features.stat().st_size,
            "sha256": feature_sha256 or hashlib.sha256(features.read_bytes()).hexdigest(),
        },
        "language_pairs": [list(pair) for pair in language_pairs],
        "feature_flags": {
            "bats": config.experimental.bats_enabled,
            "core": config.experimental.core_enabled,
            "tetm": config.experimental.tetm_enabled,
            "morphoscript": config.experimental.morphoscript_enabled,
            "evidence_repair": config.experimental.evidence_repair_enabled,
            "candidate_refinement": config.experimental.candidate_refinement_enabled,
            "semantic_parity": config.experimental.semantic_parity_enabled,
            "situglu": config.experimental.situglu_enabled,
            "recurrent_block": False,
        },
        "legacy": False,
        "format": "fp32",
    }
    if raw_revision_capability is not _UNSET:
        metadata["capabilities"] = {"revision_trained": raw_revision_capability}
    elif revision_trained is not None:
        metadata["capabilities"] = {"revision_trained": revision_trained}
    if quantization is not None:
        metadata["quantization"] = quantization
    authenticated_directions = translation_directions or tuple(
        direction for pair in language_pairs for direction in (pair, (pair[1], pair[0]))
    )
    if authenticated_directions:
        metadata["translation_directions"] = [
            list(direction) for direction in authenticated_directions
        ]
    monkeypatch.setattr(inference, "SionTokenizer", tokenizer_class)
    monkeypatch.setattr(
        inference,
        "load_exported_model",
        lambda *_args, **_kwargs: (model, config, 0, metadata),
    )
    return inference.Translator(
        tmp_path / "model.pt",
        tokenizer_path,
        device=device,
        token_features_path=features if explicit_features else None,
    )


def test_translator_connects_morphoscript_and_typed_memory(monkeypatch, tmp_path: Path) -> None:
    config = runtime_config(
        experimental=ExperimentalConfig(
            tetm_enabled=True,
            morphoscript_enabled=True,
            morphoscript_interval=1,
        )
    )
    translator = make_translator(monkeypatch, tmp_path, config)
    morph_calls: list[int] = []
    memory_calls: list[int] = []
    morph_hook = translator.model.morphoscript.register_forward_hook(
        lambda *_args: morph_calls.append(1)
    )
    memory_hook = translator.model.typed_memory.register_forward_hook(
        lambda *_args: memory_calls.append(1)
    )
    try:
        output = translator.translate(
            ["용어"],
            target_language="ja",
            num_beams=1,
            max_new_tokens=2,
            batch_size=1,
        )
    finally:
        morph_hook.remove()
        memory_hook.remove()
    assert len(output) == 1
    assert morph_calls
    assert memory_calls


def test_translator_validates_generation_lengths(monkeypatch, tmp_path: Path) -> None:
    translator = make_translator(monkeypatch, tmp_path, runtime_config())
    monkeypatch.setattr(
        translator.model,
        "generate",
        lambda input_ids, *_args, **_kwargs: torch.tensor([[2, 3]]).expand(input_ids.shape[0], -1),
    )
    assert translator.translate(
        ["문장"],
        target_language="ja",
        max_new_tokens=translator.model_config.max_seq_len,
    )
    with pytest.raises(ValueError, match="batch_size"):
        translator.translate(["문장"], target_language="ja", batch_size=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        translator.translate(
            ["문장"],
            target_language="ja",
            max_new_tokens=translator.model_config.max_seq_len + 1,
        )
    with pytest.raises(ValueError, match="between 0 and 9"):
        translator.translate(["문장"], target_language="ja", max_new_tokens=2, reasoning_level=10)
    with pytest.raises(TypeError, match="integer from 0 to 9"):
        translator.translate(["문장"], target_language="ja", max_new_tokens=2, reasoning_level=True)


def test_revision_requires_exported_training_capability(monkeypatch, tmp_path: Path) -> None:
    translator = make_translator(
        monkeypatch,
        tmp_path,
        runtime_config(),
        revision_trained=False,
    )
    with pytest.raises(ValueError, match="revision capability"):
        translator.revise(
            ["원문"],
            ["초안"],
            target_language="ja",
            max_new_tokens=2,
        )


@pytest.mark.parametrize("invalid_value", [None, "false", 0, 1, [], {}])
def test_revision_capability_rejects_non_boolean_manifest_values(
    monkeypatch,
    tmp_path: Path,
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match="revision_trained must be a boolean"):
        make_translator(
            monkeypatch,
            tmp_path,
            runtime_config(),
            raw_revision_capability=invalid_value,
        )


def test_translator_rejects_model_tokenizer_vocab_mismatch(monkeypatch, tmp_path: Path) -> None:
    config = runtime_config()
    config.vocab_size = 65
    with pytest.raises(ValueError, match="tokenizer vocab"):
        make_translator(monkeypatch, tmp_path, config)


def test_find_exported_model_preserves_best_semantic_priority(tmp_path: Path) -> None:
    exports = tmp_path / "run" / "posttrain" / "exports"

    def write_export(stage: str, timestamp: float) -> Path:
        directory = exports / stage
        directory.mkdir(parents=True)
        artifact = directory / "model.pt"
        artifact.touch()
        artifact_set_id = "a" * 64
        (directory / "export_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "sion-export-manifest-v2",
                    "created_unix": timestamp,
                    "state_sha256": "b" * 64,
                    "artifact_set_id": artifact_set_id,
                    "formats": {
                        "fp32": {
                            "status": "ok",
                            "file": artifact.name,
                            "size": artifact.stat().st_size,
                            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            "artifact_set_id": artifact_set_id,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return artifact

    best = write_export("best", 10.0)
    write_export("latest", 20.0)
    assert inference.find_exported_model(tmp_path / "run") == best


def test_find_exported_model_rejects_manifest_artifact_tampering(tmp_path: Path) -> None:
    directory = tmp_path / "run" / "exports" / "best"
    directory.mkdir(parents=True)
    artifact = directory / "model.pt"
    artifact.write_bytes(b"original")
    artifact_set_id = "a" * 64
    (directory / "export_manifest.json").write_text(
        json.dumps(
            {
                "schema": "sion-export-manifest-v2",
                "state_sha256": "b" * 64,
                "artifact_set_id": artifact_set_id,
                "formats": {
                    "fp32": {
                        "status": "ok",
                        "file": artifact.name,
                        "size": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        "artifact_set_id": artifact_set_id,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    artifact.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size or SHA256"):
        inference.find_exported_model(tmp_path / "run")


def test_find_exported_model_rejects_manifest_path_escape(tmp_path: Path) -> None:
    directory = tmp_path / "run" / "exports" / "best"
    directory.mkdir(parents=True)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    artifact_set_id = "a" * 64
    (directory / "export_manifest.json").write_text(
        json.dumps(
            {
                "schema": "sion-export-manifest-v2",
                "state_sha256": "b" * 64,
                "artifact_set_id": artifact_set_id,
                "formats": {
                    "fp32": {
                        "status": "ok",
                        "file": "../../../../outside.pt",
                        "size": outside.stat().st_size,
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                        "artifact_set_id": artifact_set_id,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes export directory"):
        inference.find_exported_model(tmp_path / "run")


def test_find_exported_model_keeps_legacy_no_manifest_fallback(tmp_path: Path) -> None:
    artifact = tmp_path / "run" / "exports" / "best" / "model.pt"
    artifact.parent.mkdir(parents=True)
    artifact.touch()

    assert inference.find_exported_model(tmp_path / "run") == artifact


def test_translator_sampling_seed_builds_reproducible_generator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    translator = make_translator(monkeypatch, tmp_path, runtime_config())
    draws: list[torch.Tensor] = []

    def sample(*_args, generator: torch.Generator, **_kwargs) -> torch.Tensor:
        draws.append(torch.randint(0, 1_000_000, (8,), generator=generator))
        return torch.tensor([[[2, 3]]])

    monkeypatch.setattr(translator.model, "sample", sample)
    kwargs = {
        "target_language": "ja",
        "num_beams": 1,
        "max_new_tokens": 2,
        "batch_size": 1,
        "num_candidates": 1,
    }
    translator.translate(["문장"], seed=41, **kwargs)
    translator.translate(["문장"], sampling_seed=41, **kwargs)
    translator.translate(["문장"], seed=42, **kwargs)
    torch.testing.assert_close(draws[0], draws[1])
    assert not torch.equal(draws[0], draws[2])

    with pytest.raises(ValueError, match="mutually exclusive"):
        translator.translate(
            ["문장"],
            seed=41,
            generator=torch.Generator(),
            **kwargs,
        )


def test_candidate_reranking_encodes_each_source_batch_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    translator = make_translator(monkeypatch, tmp_path, runtime_config())
    encode_calls = 0
    real_encode = translator.model.encode

    def count_encode(*args, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        return real_encode(*args, **kwargs)

    monkeypatch.setattr(translator.model, "encode", count_encode)
    output = translator.translate(
        ["문장"],
        target_language="ja",
        num_beams=2,
        max_new_tokens=2,
        batch_size=1,
        num_candidates=2,
        seed=17,
    )

    assert len(output) == 1
    assert encode_calls == 1


def test_translator_applies_safe_decode_limits_and_control_token_mask(
    monkeypatch,
    tmp_path: Path,
) -> None:
    translator = make_translator(monkeypatch, tmp_path, runtime_config())
    captured: dict[str, object] = {}

    def capture_generate(input_ids, _attention_mask, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[2, 3]]).expand(input_ids.shape[0], -1)

    monkeypatch.setattr(translator.model, "generate", capture_generate)
    translator.translate(
        ["문장"],
        target_language="ja",
        max_new_tokens=15,
        max_output_length_ratio=2.0,
        max_output_length_margin=1,
        no_repeat_ngram_size=4,
        reasoning_level=0,
    )

    # FakeTokenizer.encode()는 본문 토큰 2개를 내므로 2*2 + margin 1입니다.
    assert captured["max_new_tokens"] == 5
    torch.testing.assert_close(
        captured["max_new_tokens_per_row"],
        torch.tensor([5]),
    )
    assert captured["no_repeat_ngram_size"] == 4
    assert captured["min_new_tokens"] == 1
    assert captured["reasoning_level"] == 0
    forbidden = set(captured["forbidden_token_ids"])
    assert {0, 2, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 21} <= forbidden
    assert 3 not in forbidden


def test_translator_uses_exported_reasoning_endpoint_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    translator = make_translator(monkeypatch, tmp_path, runtime_config())
    translator.export_metadata["generation_defaults"] = {"reasoning_level": 9}
    captured: dict[str, object] = {}

    def capture_generate(input_ids, _attention_mask, **kwargs):
        captured.update(kwargs)
        return torch.tensor([[2, 3]]).expand(input_ids.shape[0], -1)

    monkeypatch.setattr(translator.model, "generate", capture_generate)
    translator.translate(["문장"], target_language="ja", max_new_tokens=2)

    assert captured["reasoning_level"] == 9


def test_native_sampling_generator_is_reproducible() -> None:
    model = SionForConditionalGeneration(runtime_config())
    input_ids = torch.tensor([[4, 11, 3]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    kwargs = {
        "bos_id": 2,
        "eos_id": 3,
        "num_samples": 2,
        "max_new_tokens": 3,
        "temperature": 0.8,
        "top_k": 16,
    }
    first = model.sample(
        input_ids,
        attention_mask,
        generator=torch.Generator().manual_seed(123),
        **kwargs,
    )
    second = model.sample(
        input_ids,
        attention_mask,
        generator=torch.Generator().manual_seed(123),
        **kwargs,
    )
    torch.testing.assert_close(first, second)


def test_disconnected_multilingual_graph_rejects_untrained_direction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    translator = make_translator(
        monkeypatch,
        tmp_path,
        runtime_config(),
        language_pairs=(("de", "fr"), ("sw", "ar")),
        tokenizer_class=ArbitraryGraphFakeTokenizer,
    )
    assert translator._resolve_source_language("de", "fr") == "de"
    with pytest.raises(ValueError, match="학습되지 않은 번역 방향"):
        translator.translate(
            ["ein Satz"],
            source_language="de",
            target_language="ar",
            max_new_tokens=2,
        )


def test_multilingual_inference_requires_an_explicit_trained_graph(
    monkeypatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="language_pairs metadata"):
        make_translator(
            monkeypatch,
            tmp_path,
            runtime_config(),
            language_pairs=(),
            tokenizer_class=ArbitraryGraphFakeTokenizer,
        )


def test_unidirectional_export_rejects_untrained_reverse_direction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    translator = make_translator(
        monkeypatch,
        tmp_path,
        runtime_config(),
        translation_directions=(("ko", "ja"),),
    )
    assert translator._resolve_source_language("ko", "ja") == "ko"
    with pytest.raises(ValueError, match="학습되지 않은 번역 방향"):
        translator._resolve_source_language("ja", "ko")


def test_inference_canonicalizes_bcp47_aliases_at_the_api_boundary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    translator = make_translator(
        monkeypatch,
        tmp_path,
        runtime_config(),
        language_pairs=(("PT-br", "zh-hant"),),
        translation_directions=(("PT-br", "zh-hant"),),
        tokenizer_class=BCP47GraphFakeTokenizer,
    )

    assert translator.language_pairs == (("pt-BR", "zh-Hant"),)
    assert translator.translation_directions == (("pt-BR", "zh-Hant"),)
    assert translator._resolve_source_language("PT-br", "zh-hant") == "pt-BR"


def test_inference_rejects_sidecar_and_export_direction_graph_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        inference,
        "load_tokenizer_metadata",
        lambda _path: {
            "language_pairs": [["ko", "ja"]],
            "translation_directions": [["ja", "ko"]],
        },
    )

    with pytest.raises(ValueError, match="translation directions do not match"):
        make_translator(
            monkeypatch,
            tmp_path,
            runtime_config(),
            translation_directions=(("ko", "ja"),),
        )


def test_translation_direction_metadata_rejects_disconnected_edges(
    monkeypatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="invalid translation direction metadata"):
        make_translator(
            monkeypatch,
            tmp_path,
            runtime_config(),
            translation_directions=(("ko", "ru"),),
        )


def test_translation_direction_metadata_rejects_an_uncovered_pair(
    monkeypatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cover every language pair"):
        make_translator(
            monkeypatch,
            tmp_path,
            runtime_config(),
            language_pairs=(("de", "fr"), ("sw", "ar")),
            translation_directions=(("de", "fr"),),
            tokenizer_class=ArbitraryGraphFakeTokenizer,
        )


def test_current_translation_metadata_rejects_a_missing_direction_graph() -> None:
    metadata = {
        "release_name": "sion_translate",
        "release_version": "1.5",
        "translation_capable": True,
        "pipeline": {
            "schema": "sion-translation-pipeline-v2",
            "branch": "translation-only",
        },
        "language_pairs": [["de", "fr"], ["sw", "ar"]],
    }

    with pytest.raises(ValueError, match="requires explicit translation_directions"):
        inference._translation_directions_from_metadata(metadata)


def test_legacy_missing_direction_graph_remains_unknown() -> None:
    metadata = {
        "release_name": "sion_translate",
        "release_version": "1.4",
        "translation_capable": True,
        "language_pairs": [["de", "fr"], ["sw", "ar"]],
    }

    assert inference._translation_directions_from_metadata(metadata) == ()


def test_cpu_only_quantization_metadata_overrides_requested_cuda(
    monkeypatch,
    tmp_path: Path,
) -> None:
    with pytest.warns(RuntimeWarning, match="CPU 전용"):
        translator = make_translator(
            monkeypatch,
            tmp_path,
            runtime_config(),
            quantization={"backend": "torchao", "runtime_device": "cpu"},
            device="cuda",
        )
    assert translator.device == torch.device("cpu")
    assert translator.quantized


def test_fp8_translator_prepares_a_bf16_model_for_fp16_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = runtime_config()
    model = SionForConditionalGeneration(config).eval().to(torch.bfloat16)
    packed, _quantization = _pack_fp8_state(
        model.state_dict(),
        Fp8Policy(enabled=True, block=32),
    )
    assert apply_fp8_weights(model, packed) > 0
    monkeypatch.setattr(
        fp8_runtime,
        "resolve_fp8_compute_dtype",
        lambda _device: torch.float16,
    )

    translator = make_translator(
        monkeypatch,
        tmp_path,
        config,
        quantization={"format": "fp8", "runtime_device": "cuda"},
        loaded_model=model,
    )

    assert translator.fp8_runtime is not None
    assert "FP16 즉시 역양자화" in translator.fp8_runtime
    assert {parameter.dtype for parameter in translator.model.parameters()} == {torch.float16}
    fp8_modules = [module for module in translator.model.modules() if isinstance(module, Fp8Linear)]
    assert fp8_modules
    assert all(module.weight.dtype is FORWARD_DTYPE for module in fp8_modules)
    assert all(module.scales.dtype is torch.float32 for module in fp8_modules)

    input_ids = torch.randint(4, 60, (2, 8))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    decoder_ids = torch.randint(4, 60, (2, 4))
    with torch.no_grad():
        logits = translator.model(input_ids, attention_mask, decoder_ids).logits
    assert torch.isfinite(logits).all()


def test_typed_memory_matches_training_slot_cap(monkeypatch, tmp_path: Path) -> None:
    translator = make_translator(
        monkeypatch,
        tmp_path,
        runtime_config(experimental=ExperimentalConfig(tetm_enabled=True)),
    )
    features = translator._generation_features(torch.full((1, 100), 20, dtype=torch.long))
    assert features["memory_token_ids"].shape == (1, 64, 1)
    assert int(features["memory_mask"].sum()) == 64


def test_token_feature_identity_mismatch_is_rejected(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="token feature SHA256"):
        make_translator(
            monkeypatch,
            tmp_path,
            runtime_config(
                experimental=ExperimentalConfig(
                    morphoscript_enabled=True,
                    morphoscript_interval=1,
                )
            ),
            feature_sha256="0" * 64,
        )


def test_token_feature_archive_rejects_extra_arrays(monkeypatch, tmp_path: Path) -> None:
    zeros = np.zeros(64, dtype=np.uint8)
    with pytest.raises(ValueError, match="must contain exactly"):
        make_translator(
            monkeypatch,
            tmp_path,
            runtime_config(),
            feature_arrays={
                "script": zeros,
                "onset": zeros,
                "vowel": zeros,
                "coda": zeros,
                "unexpected": zeros,
            },
        )


def test_token_feature_archive_rejects_non_integer_arrays(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zeros = np.zeros(64, dtype=np.uint8)
    with pytest.raises(ValueError, match="script must use an integer dtype"):
        make_translator(
            monkeypatch,
            tmp_path,
            runtime_config(),
            feature_arrays={
                "script": np.zeros(64, dtype=np.float32),
                "onset": zeros,
                "vowel": zeros,
                "coda": zeros,
            },
        )


def test_token_features_follow_export_metadata_filename(monkeypatch, tmp_path: Path) -> None:
    translator = make_translator(
        monkeypatch,
        tmp_path,
        runtime_config(
            experimental=ExperimentalConfig(
                morphoscript_enabled=True,
                morphoscript_interval=1,
            )
        ),
        feature_filename="custom_morphoscript_features.npz",
        explicit_features=False,
    )
    assert translator.token_features is not None
    assert set(translator.token_features) == {"script", "onset", "vowel", "coda"}


def test_unknown_legacy_revision_capability_warns_but_remains_usable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    translator = make_translator(
        monkeypatch,
        tmp_path,
        runtime_config(),
        revision_trained=None,
    )
    monkeypatch.setattr(
        translator,
        "translate",
        lambda texts, **_kwargs: list(texts),
    )
    with pytest.warns(RuntimeWarning, match="기록되어 있지 않습니다"):
        revised = translator.revise(
            ["원문"],
            ["초안"],
            target_language="ja",
            max_new_tokens=2,
        )
    assert revised == ["원문 <draft> 초안"]
