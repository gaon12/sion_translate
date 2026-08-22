from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import subprocess
import sys
import textwrap
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import sion_translate.training.export as export_module
from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.inference import find_exported_model
from sion_translate.model import SionForConditionalGeneration
from sion_translate.model.layers import RotaryEmbedding, SwiGLU
from sion_translate.training.distributed import DistributedContext
from sion_translate.training.export import (
    EXPORT_SCHEMA,
    _cpu_model,
    _quantize_affine_k,
    build_export_metadata as _build_export_metadata,
    convert_export,
    export_inference_models,
    export_state_dict_formats as _export_state_dict_formats,
    load_exported_model,
    validate_export_directory,
)


def export_config(*, d_model: int = 32) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=d_model,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=d_model * 2,
        max_seq_len=16,
        dropout=0.0,
        gradient_checkpointing=False,
        experimental=ExperimentalConfig(),
    )


def translation_pipeline_identity() -> dict[str, Any]:
    return {
        "schema": "sion-translation-pipeline-v2",
        "branch": "translation-only",
    }


def foundation_pipeline_identity() -> dict[str, Any]:
    return {
        "schema": "sion-translation-pipeline-v2",
        "branch": "foundation-then-translation",
        "foundation": {
            "schema": "sion-foundation-lineage-v1",
            "release_name": "sion",
            "release_version": "1.5",
            "languages": ["ko", "ja"],
            "selected_step": 7,
            "foundation_manifest_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
            "checkpoint_identity_sha256": "c" * 64,
            "checkpoint_artifact_sha256": "d" * 64,
        },
    }


def build_export_metadata(
    model_config: ModelConfig,
    **kwargs: Any,
) -> dict[str, Any]:
    """Default existing tests to the exact pipeline contract introduced in 1.5."""

    release_version = kwargs.get("release_version", "1.5")
    translation_capable = bool(kwargs.get("translation_capable", True))
    if "pipeline_identity" not in kwargs and translation_capable and release_version == "1.5":
        kwargs["pipeline_identity"] = translation_pipeline_identity()
    return _build_export_metadata(model_config, **kwargs)


def export_state_dict_formats(
    directory: str | Path,
    state_dict: Mapping[str, torch.Tensor],
    model_config: ModelConfig,
    pad_id: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Give unrelated export tests the valid current translation ancestry."""

    if kwargs.get("metadata") is None:
        translation_capable = bool(kwargs.get("translation_capable", True))
        kwargs["metadata"] = build_export_metadata(
            model_config,
            release_name=str(kwargs.get("release_name", "sion_translate")),
            translation_capable=translation_capable,
            pipeline_identity=(translation_pipeline_identity() if translation_capable else None),
        )
    return _export_state_dict_formats(
        directory,
        state_dict,
        model_config,
        pad_id,
        **kwargs,
    )


def _write_0b_style_legacy_gguf(
    path: Path,
    state_dict: Mapping[str, torch.Tensor],
    *,
    language_pairs: list[list[str]],
) -> dict[str, int]:
    gguf = pytest.importorskip("gguf")
    counts = {"q4_k": 0, "q5_k": 0, "f16": 0}
    writer = gguf.GGUFWriter(path, "sion")
    try:
        writer.add_name("sion_translate")
        writer.add_string("sion.export_schema", EXPORT_SCHEMA)
        if language_pairs:
            writer.add_string(
                "sion.language_pairs",
                json.dumps(language_pairs, ensure_ascii=False, separators=(",", ":")),
            )
        for name, tensor in state_dict.items():
            writer.add_tensor(name, tensor.detach().float().cpu().numpy().astype(np.float16))
            counts["f16"] += 1
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        writer = None
    finally:
        if writer is not None:
            writer.close()
    return counts


def _store_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    (directory / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_loader_resolves_pre_rename_kjx_module_pickles(tmp_path: Path) -> None:
    legacy_root = types.ModuleType("kjx")
    legacy_root.__path__ = []
    legacy_model_package = types.ModuleType("kjx.model")
    legacy_model_package.__path__ = []
    legacy_module = types.ModuleType("kjx.model.kjx")
    legacy_class = type(
        "KJXForConditionalGeneration",
        (SionForConditionalGeneration,),
        {"__module__": "kjx.model.kjx"},
    )
    legacy_module.KJXForConditionalGeneration = legacy_class
    temporary_modules = {
        "kjx": legacy_root,
        "kjx.model": legacy_model_package,
        "kjx.model.kjx": legacy_module,
    }
    sys.modules.update(temporary_modules)
    path = tmp_path / "legacy.pt"
    config = export_config()
    legacy_model = legacy_class(config)
    del legacy_model._synchronize_generation_across_ranks
    del legacy_model.recurrent_block_layers
    del legacy_model.recurrent_steps
    for module in legacy_model.modules():
        if isinstance(module, SwiGLU):
            del module.gate_beta
            del module.up_beta
        elif isinstance(module, RotaryEmbedding):
            del module.head_dim
            del module.max_seq_len
            del module.base
            del module._cache_device
    try:
        torch.save(
            {
                "model_config": asdict(config),
                "pad_id": 0,
                "model": legacy_model,
            },
            path,
        )
    finally:
        for name in reversed(temporary_modules):
            sys.modules.pop(name, None)

    with pytest.raises(ValueError, match="safe weights-only loader"):
        load_exported_model(path)
    with pytest.warns(RuntimeWarning, match="execute code"):
        loaded, loaded_config, pad_id = load_exported_model(
            path,
            unsafe_allow_pickle=True,
        )

    assert type(loaded) is SionForConditionalGeneration
    assert loaded_config == config
    assert pad_id == 0
    assert loaded._synchronize_generation_across_ranks is False
    assert loaded.recurrent_block_layers == 0
    assert loaded.recurrent_steps == 1
    assert all(
        hasattr(module, "gate_beta") and hasattr(module, "up_beta")
        for module in loaded.modules()
        if isinstance(module, SwiGLU)
    )
    assert all(
        module.head_dim == config.d_model // config.num_heads
        and module.max_seq_len == config.max_seq_len
        and module.base == 10000.0
        and module._cache_device == str(module.cos.device)
        for module in loaded.modules()
        if isinstance(module, RotaryEmbedding)
    )
    generated = loaded.generate(
        torch.tensor([[4, 5, 3]]),
        torch.ones(1, 3, dtype=torch.bool),
        bos_id=2,
        eos_id=3,
        max_new_tokens=2,
    )
    assert generated.shape[0] == 1


@pytest.mark.parametrize("bits", (4, 5))
def test_numpy_k_quant_fallback_round_trips_with_gguf(bits: int) -> None:
    gguf = pytest.importorskip("gguf")
    source = np.random.default_rng(7).normal(size=(3, 512)).astype(np.float32)
    quantized = _quantize_affine_k(source, bits=bits)
    qtype = gguf.GGMLQuantizationType.Q4_K if bits == 4 else gguf.GGMLQuantizationType.Q5_K
    restored = gguf.quants.dequantize(quantized, qtype)
    assert restored.shape == source.shape
    assert np.sqrt(np.mean((source - restored) ** 2)) < (0.10 if bits == 4 else 0.06)


def test_stable_precision_and_packed_int4_exports_reload(tmp_path: Path) -> None:
    pytest.importorskip("torchao")
    pytest.importorskip("gguf")
    config = export_config(d_model=256)
    model = SionForConditionalGeneration(config)
    metadata = build_export_metadata(
        config,
        language_pair=("ko", "ja"),
        revision_trained=False,
        step=3,
    )
    manifest = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        step=3,
        formats=(
            "fp32",
            "fp16",
            "bf16",
            "int8",
            "int4",
            "gguf_q4_k_m",
            "transformers",
        ),
        metadata=metadata,
        int4_backend="packed",
    )
    assert all(entry["status"] == "ok" for entry in manifest["formats"].values())
    assert manifest["formats"]["transformers"]["revision_trained"] is False
    packed_payload = torch.load(tmp_path / "model_int4.pt", weights_only=True)
    assert packed_payload["schema"] == EXPORT_SCHEMA
    assert not isinstance(packed_payload["model"], torch.nn.Module)
    assert packed_payload["quantization"]["backend"] == "sion-packed"
    assert packed_payload["metadata"]["language_pair"] == ["ko", "ja"]

    input_ids = torch.tensor([[4, 10, 3]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    decoder_input_ids = torch.tensor([[2, 20]])
    for filename in (
        "model.pt",
        "model_fp16.pt",
        "model_bf16.pt",
        "model_int8.pt",
        "model_int4.pt",
    ):
        restored, restored_config, pad_id = load_exported_model(tmp_path / filename)
        assert restored_config.vocab_size == config.vocab_size
        assert pad_id == 0
        output = restored(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        assert output.logits.shape == (1, 2, config.vocab_size)
    validation = validate_export_directory(tmp_path)
    assert validation["valid"]
    assert set(validation["formats"]) == {
        "fp32",
        "fp16",
        "bf16",
        "int8",
        "int4",
        "gguf_q4_k_m",
        "transformers",
    }


def test_native_loader_constructs_on_meta_before_binding_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = export_config()
    source = SionForConditionalGeneration(config)
    export_state_dict_formats(
        tmp_path,
        source.state_dict(),
        config,
        0,
        step=1,
        formats=("fp32",),
    )
    original_constructor = export_module.SionForConditionalGeneration
    constructor_devices: list[str] = []

    def recording_constructor(*args: object, **kwargs: object) -> SionForConditionalGeneration:
        model = original_constructor(*args, **kwargs)
        constructor_devices.append(next(model.parameters()).device.type)
        return model

    monkeypatch.setattr(export_module, "SionForConditionalGeneration", recording_constructor)
    restored, _, _ = load_exported_model(tmp_path / "model.pt")
    assert constructor_devices == ["meta"]
    assert next(restored.parameters()).device.type == "cpu"


def test_export_metadata_records_tokenizer_hash(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"tokenizer fixture")
    metadata = build_export_metadata(
        export_config(),
        tokenizer_path=tokenizer,
        language_pair=("ko", "ja"),
        revision_trained=True,
    )
    assert metadata["tokenizer"]["sha256"] == hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    assert metadata["capabilities"]["revision_trained"] is True


def test_export_metadata_records_evidence_and_parity_architecture() -> None:
    config = export_config()
    config.experimental.evidence_repair_enabled = True
    config.experimental.candidate_refinement_enabled = True
    config.experimental.semantic_parity_enabled = True

    metadata = build_export_metadata(config)

    assert metadata["feature_flags"]["evidence_repair"] is True
    assert metadata["feature_flags"]["candidate_refinement"] is True
    assert metadata["feature_flags"]["semantic_parity"] is True
    assert metadata["generation_defaults"]["reasoning_level"] == 9


def test_export_metadata_records_an_independent_json_safe_pipeline_identity(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"pipeline tokenizer fixture")
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    lineage: dict[str, object] = {
        "schema": "sion-foundation-lineage-v1",
        "release_name": "sion",
        "release_version": "1.5",
        "languages": ["ko", "ja"],
        "selected_step": 7,
        "foundation_manifest_sha256": "a" * 64,
        "tokenizer_sha256": tokenizer_sha256,
        "checkpoint_identity_sha256": "c" * 64,
        "checkpoint_artifact_sha256": "d" * 64,
    }
    pipeline = {
        "schema": "sion-translation-pipeline-v2",
        "branch": "foundation-then-translation",
        "foundation": lineage,
    }

    metadata = build_export_metadata(
        export_config(),
        tokenizer_path=tokenizer,
        languages=("ko", "ja"),
        pipeline_identity=pipeline,
    )
    lineage["foundation_manifest_sha256"] = "changed"

    assert metadata["pipeline"] == {
        "schema": "sion-translation-pipeline-v2",
        "branch": "foundation-then-translation",
        "foundation": {
            "schema": "sion-foundation-lineage-v1",
            "release_name": "sion",
            "release_version": "1.5",
            "languages": ["ko", "ja"],
            "selected_step": 7,
            "foundation_manifest_sha256": "a" * 64,
            "tokenizer_sha256": tokenizer_sha256,
            "checkpoint_identity_sha256": "c" * 64,
            "checkpoint_artifact_sha256": "d" * 64,
        },
    }
    json.dumps(metadata, allow_nan=False)


def test_foundation_pipeline_accepts_a_configured_distinct_base_release_name(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"custom foundation release tokenizer")
    pipeline = foundation_pipeline_identity()
    pipeline["foundation"]["release_name"] = "my_base"
    pipeline["foundation"]["tokenizer_sha256"] = hashlib.sha256(tokenizer.read_bytes()).hexdigest()

    metadata = _build_export_metadata(
        export_config(),
        tokenizer_path=tokenizer,
        languages=("ko", "ja", "en"),
        pipeline_identity=pipeline,
    )

    assert metadata["pipeline"]["foundation"]["release_name"] == "my_base"


@pytest.mark.parametrize(
    "pipeline_identity",
    [
        {"objective": object()},
        {"loss": float("nan")},
    ],
)
def test_export_metadata_rejects_non_json_pipeline_identity(
    pipeline_identity: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="JSON-safe"):
        build_export_metadata(export_config(), pipeline_identity=pipeline_identity)


@pytest.mark.parametrize(
    "pipeline_identity",
    [
        {"schema": "sion-translation-pipeline-v1", "branch": "translation-only"},
        {
            "schema": "sion-translation-pipeline-v2",
            "branch": "translation-only",
            "unverifiable": True,
        },
        {
            **foundation_pipeline_identity(),
            "foundation": {
                **foundation_pipeline_identity()["foundation"],
                "checkpoint_artifact_sha256": "D" * 64,
            },
        },
        {
            **foundation_pipeline_identity(),
            "foundation": {
                key: value
                for key, value in foundation_pipeline_identity()["foundation"].items()
                if key != "release_version"
            },
        },
    ],
)
def test_export_metadata_rejects_nonexact_pipeline_contracts(
    pipeline_identity: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="pipeline|lineage"):
        _build_export_metadata(export_config(), pipeline_identity=pipeline_identity)


def test_foundation_export_metadata_forbids_translation_pipeline_identity() -> None:
    with pytest.raises(ValueError, match="foundation-only.*must not contain pipeline"):
        _build_export_metadata(
            export_config(),
            release_name="sion",
            translation_capable=False,
            pipeline_identity=translation_pipeline_identity(),
        )


def test_foundation_pipeline_binds_tokenizer_without_conflating_translation_languages(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"foundation binding fixture")
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    pipeline = foundation_pipeline_identity()
    pipeline["foundation"]["tokenizer_sha256"] = tokenizer_sha256

    metadata = build_export_metadata(
        export_config(),
        tokenizer_path=tokenizer,
        languages=("ko", "ja"),
        pipeline_identity=pipeline,
    )

    assert metadata["pipeline"] == pipeline
    wrong_tokenizer = json.loads(json.dumps(pipeline))
    wrong_tokenizer["foundation"]["tokenizer_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="exactly match metadata.tokenizer"):
        _build_export_metadata(
            export_config(),
            tokenizer_path=tokenizer,
            languages=("ko", "ja"),
            pipeline_identity=wrong_tokenizer,
        )
    independent_languages = _build_export_metadata(
        export_config(),
        tokenizer_path=tokenizer,
        languages=("ko", "en"),
        pipeline_identity=pipeline,
    )
    assert independent_languages["languages"] == ["ko", "en"]
    assert independent_languages["pipeline"]["foundation"]["languages"] == ["ko", "ja"]


def test_export_revalidates_pipeline_after_tokenizer_and_language_overrides(
    tmp_path: Path,
) -> None:
    tokenizer_a = tmp_path / "tokenizer-a.model"
    tokenizer_b = tmp_path / "tokenizer-b.model"
    tokenizer_a.write_bytes(b"foundation tokenizer A")
    tokenizer_b.write_bytes(b"foundation tokenizer B")
    pipeline = foundation_pipeline_identity()
    pipeline["foundation"]["tokenizer_sha256"] = hashlib.sha256(
        tokenizer_a.read_bytes()
    ).hexdigest()
    config = export_config()
    model = SionForConditionalGeneration(config)
    metadata = build_export_metadata(
        config,
        tokenizer_path=tokenizer_a,
        languages=("ko", "ja"),
        pipeline_identity=pipeline,
    )

    tokenizer_destination = tmp_path / "wrong-tokenizer"
    with pytest.raises(ValueError, match="exactly match metadata.tokenizer"):
        export_module.export_state_dict_formats(
            tokenizer_destination,
            model.state_dict(),
            config,
            0,
            formats=("fp32",),
            metadata=metadata,
            tokenizer_path=tokenizer_b,
        )
    assert not (tokenizer_destination / tokenizer_b.name).exists()
    assert not (tokenizer_destination / "model.pt").exists()
    assert not (tokenizer_destination / "export_manifest.json").exists()

    language_destination = tmp_path / "relabeled-languages"
    with pytest.raises(ValueError, match="omits configured language-pair members"):
        export_module.export_state_dict_formats(
            language_destination,
            model.state_dict(),
            config,
            0,
            formats=("fp32",),
            metadata=metadata,
            tokenizer_path=tokenizer_a,
            language_pairs=(("ko", "en"),),
        )
    assert not (language_destination / "model.pt").exists()
    assert not (language_destination / "export_manifest.json").exists()


def test_foundation_languages_are_independent_of_translation_capabilities(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"production language-union fixture")
    pipeline = foundation_pipeline_identity()
    pipeline["foundation"]["languages"] = ["ko", "ja", "en"]
    pipeline["foundation"]["tokenizer_sha256"] = hashlib.sha256(tokenizer.read_bytes()).hexdigest()

    metadata = build_export_metadata(
        export_config(),
        tokenizer_path=tokenizer,
        language_pairs=(("ko", "ja"),),
        pipeline_identity=pipeline,
    )

    assert metadata["languages"] == ["ko", "ja"]
    assert metadata["pipeline"] == pipeline


def test_validation_rejects_consistently_replicated_false_foundation_tokenizer(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"foundation validation fixture")
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    pipeline = foundation_pipeline_identity()
    pipeline["foundation"]["tokenizer_sha256"] = tokenizer_sha256
    metadata = build_export_metadata(
        export_config(),
        tokenizer_path=tokenizer,
        languages=("ko", "ja"),
        pipeline_identity=pipeline,
    )
    model = SionForConditionalGeneration(export_config())
    manifest = export_state_dict_formats(
        tmp_path / "export",
        model.state_dict(),
        model.config,
        0,
        formats=("fp32",),
        metadata=metadata,
        tokenizer_path=tokenizer,
    )
    export_dir = tmp_path / "export"
    artifact = export_dir / "model.pt"
    payload = torch.load(artifact, weights_only=True)
    false_digest = "e" * 64
    payload["metadata"]["pipeline"]["foundation"]["tokenizer_sha256"] = false_digest
    manifest["metadata"]["pipeline"]["foundation"]["tokenizer_sha256"] = false_digest
    torch.save(payload, artifact)
    manifest["metadata_compatibility_id"] = export_module._metadata_compatibility_id(
        manifest["metadata"]
    )
    manifest["formats"]["fp32"].update(export_module._file_entry(artifact))
    (export_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    validation = validate_export_directory(export_dir)

    assert not validation["valid"]
    assert any(
        error["error_type"] == "InvalidPipelineIdentity"
        and "metadata.tokenizer" in error["message"]
        for error in validation["errors"]
    )


def test_export_rejects_a_reasoning_endpoint_that_bypasses_trained_refinement(
    tmp_path: Path,
) -> None:
    config = export_config()
    config.experimental.candidate_refinement_enabled = True
    model = SionForConditionalGeneration(config)
    metadata = build_export_metadata(config)
    metadata["generation_defaults"] = {"reasoning_level": 0}

    with pytest.raises(ValueError, match="does not match model features"):
        export_state_dict_formats(
            tmp_path,
            model.state_dict(),
            config,
            0,
            formats=("fp32",),
            metadata=metadata,
        )


def test_native_loader_rejects_tampered_reasoning_endpoint(tmp_path: Path) -> None:
    config = export_config()
    config.experimental.candidate_refinement_enabled = True
    model = SionForConditionalGeneration(config)
    export_state_dict_formats(tmp_path, model.state_dict(), config, 0, formats=("fp32",))
    payload = torch.load(tmp_path / "model.pt", weights_only=True)
    payload["metadata"]["generation_defaults"] = {"reasoning_level": 0}
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)

    with pytest.raises(ValueError, match="does not match model features"):
        load_exported_model(tampered)


def test_native_loader_rejects_current_schema_without_pipeline_identity(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    export_state_dict_formats(tmp_path, model.state_dict(), config, 0, formats=("fp32",))
    payload = torch.load(tmp_path / "model.pt", weights_only=True)
    payload["metadata"].pop("pipeline")
    tampered = tmp_path / "missing-pipeline.pt"
    torch.save(payload, tampered)

    with pytest.raises(ValueError, match="requires pipeline identity"):
        load_exported_model(tampered)


def test_native_loader_rejects_schema_stripping_from_declared_1_5(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    export_state_dict_formats(tmp_path, model.state_dict(), config, 0, formats=("fp32",))
    payload = torch.load(tmp_path / "model.pt", weights_only=True)
    payload.pop("schema")
    payload["metadata"].pop("pipeline")
    tampered = tmp_path / "schema-stripped-1.5.pt"
    torch.save(payload, tampered)

    with pytest.raises(ValueError, match="requires pipeline identity"):
        load_exported_model(tampered)


def test_native_loader_accepts_schema_less_declared_1_0(tmp_path: Path) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    metadata = build_export_metadata(config, release_version="1.0")
    export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
        metadata=metadata,
    )
    payload = torch.load(tmp_path / "model.pt", weights_only=True)
    payload.pop("schema")
    declared_legacy = tmp_path / "schema-less-1.0.pt"
    torch.save(payload, declared_legacy)

    _, loaded_config, pad_id = load_exported_model(declared_legacy)

    assert loaded_config == config
    assert pad_id == 0


def test_native_loader_rejects_current_schema_role_capability_mismatch(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    export_state_dict_formats(tmp_path, model.state_dict(), config, 0, formats=("fp32",))
    payload = torch.load(tmp_path / "model.pt", weights_only=True)
    payload["metadata"]["release_name"] = "sion"
    tampered = tmp_path / "contradictory-role.pt"
    torch.save(payload, tampered)

    with pytest.raises(ValueError, match="foundation release cannot be translation-capable"):
        load_exported_model(tampered)


def test_export_metadata_identifies_the_target_1_5_model_generation() -> None:
    metadata = build_export_metadata(export_config())

    assert metadata["release_name"] == "sion_translate"
    assert metadata["release_version"] == "1.5"


@pytest.mark.parametrize("release_version", [None, 15, "", "v1.5", "1"])
def test_export_metadata_rejects_malformed_model_versions(release_version: object) -> None:
    with pytest.raises(ValueError, match="numeric major.minor"):
        build_export_metadata(export_config(), release_version=release_version)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("release_name", "translation_capable"),
    [("sion", True), ("sion_translate", False)],
)
def test_export_metadata_rejects_contradictory_repository_roles(
    release_name: str,
    translation_capable: bool,
) -> None:
    with pytest.raises(ValueError, match="translation-capable"):
        build_export_metadata(
            export_config(),
            release_name=release_name,
            translation_capable=translation_capable,
        )


def test_export_rejects_caller_metadata_that_bypasses_repository_role(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    metadata = build_export_metadata(config)
    metadata["release_name"] = "sion"

    with pytest.raises(ValueError, match="foundation release cannot be translation-capable"):
        export_state_dict_formats(
            tmp_path,
            model.state_dict(),
            config,
            0,
            formats=("fp32",),
            metadata=metadata,
        )


def test_export_metadata_records_exact_trained_translation_directions() -> None:
    metadata = build_export_metadata(
        export_config(),
        language_pairs=(("ko", "ja"), ("en", "ru")),
        bidirectional=False,
    )
    assert metadata["language_pairs"] == [["ko", "ja"], ["en", "ru"]]
    assert metadata["translation_directions"] == [["ko", "ja"], ["en", "ru"]]


def test_cpu_export_model_reuses_stable_snapshot_storage() -> None:
    config = export_config()
    source = SionForConditionalGeneration(config).state_dict()
    restored = _cpu_model(config, source, 0)
    for name, tensor in restored.state_dict().items():
        assert tensor.data_ptr() == source[name].data_ptr()


def test_conversion_inherits_source_tokenizer_hash_when_path_is_omitted(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"tokenizer fixture")
    source_dir = tmp_path / "source"
    metadata = build_export_metadata(config, tokenizer_path=tokenizer)
    export_state_dict_formats(
        source_dir,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
        metadata=metadata,
        tokenizer_path=tokenizer,
    )

    converted = convert_export(
        source_dir / "model.pt",
        tmp_path / "converted",
        formats=("fp16",),
    )
    assert converted["metadata"]["tokenizer"] == metadata["tokenizer"]
    payload = torch.load(tmp_path / "converted" / "model_fp16.pt", weights_only=True)
    assert payload["metadata"]["tokenizer"] == metadata["tokenizer"]

    (source_dir / tokenizer.name).unlink()
    missing_tokenizer = convert_export(
        # Simulate moving only model.pt and its metadata, without the embedded
        # sidecar that normal v2 exports now carry.
        source_dir / "model.pt",
        tmp_path / "missing-tokenizer",
        formats=("transformers",),
    )
    entry = missing_tokenizer["formats"]["transformers"]
    assert entry["status"] == "error"
    assert "tokenizer_path" in entry["message"]


def test_conversion_preserves_old_model_generation_and_redacts_source_path(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    source_dir = tmp_path / "old-generation"
    export_state_dict_formats(
        source_dir,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
        metadata=build_export_metadata(config, release_version="1.0"),
    )
    source = source_dir / "model.pt"

    converted = convert_export(
        source,
        tmp_path / "converted",
        formats=("fp16",),
    )

    assert converted["metadata"]["release_version"] == "1.0"
    assert converted["metadata"]["source"] == {
        "filename": source.name,
        "size": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert str(tmp_path) not in json.dumps(converted["metadata"])


def test_conversion_requires_explicit_identity_for_unversioned_legacy_weights(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    source_dir = tmp_path / "source"
    export_state_dict_formats(
        source_dir,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
    )
    legacy_source = tmp_path / "legacy.pt"
    payload = torch.load(source_dir / "model.pt", weights_only=True)
    payload.pop("schema")
    payload["metadata"].pop("release_name")
    payload["metadata"].pop("release_version")
    payload["metadata"].pop("translation_capable")
    payload["metadata"].pop("pipeline")
    torch.save(payload, legacy_source)

    with pytest.raises(ValueError, match="pass --release-name explicitly"):
        convert_export(legacy_source, tmp_path / "ambiguous", formats=("fp16",))

    converted = convert_export(
        legacy_source,
        tmp_path / "identified",
        formats=("fp16",),
        release_name="sion",
        release_version="0.9",
        translation_capable=False,
    )
    assert converted["metadata"]["release_name"] == "sion"
    assert converted["metadata"]["release_version"] == "0.9"


def test_conversion_rejects_release_identity_relabeling(tmp_path: Path) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    source_dir = tmp_path / "source"
    export_state_dict_formats(source_dir, model.state_dict(), config, 0, formats=("fp32",))

    with pytest.raises(ValueError, match="conversion cannot relabel weights"):
        convert_export(
            source_dir / "model.pt",
            tmp_path / "relabeled",
            formats=("fp16",),
            release_version="1.4",
        )

    with pytest.raises(ValueError, match="translation_capable conflicts"):
        convert_export(
            source_dir / "model.pt",
            tmp_path / "recapabilitized",
            formats=("fp16",),
            translation_capable=False,
        )


def test_conversion_rejects_tokenizer_identity_relabeling(tmp_path: Path) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    tokenizer_a = tmp_path / "tokenizer-a.model"
    tokenizer_b = tmp_path / "tokenizer-b.model"
    tokenizer_a.write_bytes(b"trusted tokenizer A")
    tokenizer_b.write_bytes(b"different tokenizer B")
    source_dir = tmp_path / "source"
    export_state_dict_formats(
        source_dir,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
        metadata=build_export_metadata(config, tokenizer_path=tokenizer_a),
        tokenizer_path=tokenizer_a,
    )

    with pytest.raises(ValueError, match="tokenizer conflicts.*cannot relabel"):
        convert_export(
            source_dir / "model.pt",
            tmp_path / "relabeled-tokenizer",
            formats=("fp16",),
            tokenizer_path=tokenizer_b,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"language_pairs": (("en", "ru"),)},
        {"bidirectional": True},
        {"revision_trained": False},
    ],
    ids=("language-pairs", "directions", "revision-capability"),
)
def test_conversion_rejects_translation_capability_relabeling(
    tmp_path: Path,
    overrides: dict[str, Any],
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    source_dir = tmp_path / "source"
    export_state_dict_formats(
        source_dir,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
        metadata=build_export_metadata(
            config,
            language_pairs=(("ko", "ja"),),
            bidirectional=False,
            revision_trained=True,
        ),
    )

    with pytest.raises(ValueError, match="conflict.*source"):
        convert_export(
            source_dir / "model.pt",
            tmp_path / "relabeled-capability",
            formats=("fp16",),
            **overrides,
        )


def test_gguf_only_conversion_resolves_inherited_foundation_tokenizer(
    tmp_path: Path,
) -> None:
    pytest.importorskip("gguf")
    config = export_config()
    model = SionForConditionalGeneration(config)
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"foundation-bound tokenizer")
    pipeline = foundation_pipeline_identity()
    pipeline["foundation"]["tokenizer_sha256"] = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    source_dir = tmp_path / "source"
    export_state_dict_formats(
        source_dir,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
        metadata=build_export_metadata(
            config,
            tokenizer_path=tokenizer,
            language_pairs=(("ko", "ja"),),
            pipeline_identity=pipeline,
        ),
        tokenizer_path=tokenizer,
    )

    converted = convert_export(
        source_dir / "model.pt",
        tmp_path / "converted",
        formats=("gguf_q4_k_m",),
    )

    assert converted["formats"]["gguf_q4_k_m"]["status"] == "ok"
    assert (
        converted["metadata"]["tokenizer"]["sha256"] == pipeline["foundation"]["tokenizer_sha256"]
    )


def test_transformers_directory_hash_is_deterministic_and_tamper_evident(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    half_state = {
        name: tensor.half() if tensor.is_floating_point() else tensor
        for name, tensor in model.state_dict().items()
    }
    pairs = (("ko", "ja"), ("en", "ru"))
    metadata = build_export_metadata(config, language_pairs=pairs)

    manifests = []
    for name in ("first", "second"):
        manifests.append(
            export_state_dict_formats(
                tmp_path / name,
                half_state,
                config,
                0,
                formats=("transformers",),
                metadata=metadata,
                language_pairs=pairs,
            )
        )
    first_entry = manifests[0]["formats"]["transformers"]
    second_entry = manifests[1]["formats"]["transformers"]
    assert first_entry["artifact_type"] == "directory"
    assert first_entry["sha256"] == second_entry["sha256"]
    assert first_entry["size"] == sum(item["size"] for item in first_entry["files"])
    assert first_entry["dtypes"] == ["torch.float16"]

    checkpoint_dir = tmp_path / "first" / first_entry["file"]
    config_json = json.loads((checkpoint_dir / "config.json").read_text())
    assert config_json["language_pairs"] == [["ko", "ja"], ["en", "ru"]]
    assert config_json["translation_directions"] == [
        ["ko", "ja"],
        ["ja", "ko"],
        ["en", "ru"],
        ["ru", "en"],
    ]
    assert validate_export_directory(tmp_path / "first")["valid"]

    config_path = checkpoint_dir / "config.json"
    config_path.write_text(config_path.read_text() + "\n", encoding="utf-8")
    invalid = validate_export_directory(tmp_path / "first")
    assert not invalid["valid"]
    assert invalid["formats"]["transformers"]["error_type"] == "RuntimeError"


def test_0b_style_transformers_sidecars_are_legacy_only_before_1_5(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    pairs = (("ko", "ja"),)
    metadata = build_export_metadata(
        config,
        release_version="1.0",
        language_pairs=pairs,
    )
    manifest = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("transformers",),
        metadata=metadata,
        language_pairs=pairs,
    )
    checkpoint = tmp_path / "transformers"
    for sidecar_name in ("config.json", "sion_export.json"):
        sidecar_path = checkpoint / sidecar_name
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar.pop("release_name", None)
        sidecar.pop("release_version", None)
        sidecar.pop("pipeline", None)
        if sidecar_name == "sion_export.json":
            sidecar.pop("generation_defaults", None)
        sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    generation_path = checkpoint / "generation_config.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation.pop("reasoning_level", None)
    generation_path.write_text(
        json.dumps(generation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["metadata"].pop("generation_defaults", None)
    manifest["metadata_compatibility_id"] = export_module._metadata_compatibility_id(
        manifest["metadata"]
    )
    manifest["formats"]["transformers"].update(export_module._directory_entry(checkpoint))
    _store_manifest(tmp_path, manifest)

    legacy_validation = validate_export_directory(tmp_path)

    assert legacy_validation["valid"]
    assert (
        legacy_validation["formats"]["transformers"]["inspection"]["identity_source"]
        == "legacy-manifest"
    )

    manifest["metadata"]["release_version"] = "1.5"
    manifest["metadata"]["pipeline"] = translation_pipeline_identity()
    manifest["metadata"]["generation_defaults"] = {"reasoning_level": 0}
    manifest["metadata_compatibility_id"] = export_module._metadata_compatibility_id(
        manifest["metadata"]
    )
    _store_manifest(tmp_path, manifest)

    current_validation = validate_export_directory(tmp_path)

    assert not current_validation["valid"]
    assert any(
        error.get("format") == "transformers"
        and "release_name must be a non-empty string" in error["message"]
        for error in current_validation["errors"]
    )


def test_transformers_export_rejects_broken_bundled_remote_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sion_translate.hf import conversion as hf_conversion

    config = export_config()
    model = SionForConditionalGeneration(config)
    original_save = hf_conversion.save_transformers_checkpoint

    def corrupt_remote_code(output_dir, *args, **kwargs):
        destination = original_save(output_dir, *args, **kwargs)
        (Path(output_dir) / "modeling_sion.py").write_text(
            "this is not valid Python !!!\n",
            encoding="utf-8",
        )
        return destination

    monkeypatch.setattr(
        hf_conversion,
        "save_transformers_checkpoint",
        corrupt_remote_code,
    )
    manifest = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("transformers",),
    )
    entry = manifest["formats"]["transformers"]
    assert entry["status"] == "error"
    assert entry["error_type"] in {"RuntimeError", "SyntaxError"}


def test_validator_requires_v2_integrity_fields(tmp_path: Path) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
    )
    manifest_path = tmp_path / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["state_sha256"]
    del manifest["formats"]["fp32"]["sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_export_directory(tmp_path)
    assert not validation["valid"]
    assert validation["errors"]


def test_validator_preserves_training_only_config_fields(tmp_path: Path) -> None:
    config = export_config()
    config.gradient_checkpointing = True
    model = SionForConditionalGeneration(config)
    export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
    )
    loaded_model, loaded_config, _ = load_exported_model(tmp_path / "model.pt")
    assert loaded_config.gradient_checkpointing is True
    assert loaded_model.config.gradient_checkpointing is False
    assert validate_export_directory(tmp_path)["valid"]


def test_transformers_options_flow_through_conversion_and_training_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sion_translate.hf import conversion as hf_conversion

    config = export_config()
    model = SionForConditionalGeneration(config)
    source_dir = tmp_path / "source"
    export_state_dict_formats(
        source_dir,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
    )
    tokenizer_marker = tmp_path / "tokenizer.model"
    tokenizer_marker.write_bytes(b"tokenizer propagation fixture")
    pairs = (("ko", "ja"), ("en", "ru"))
    calls: list[dict[str, object]] = []
    real_save = hf_conversion.save_transformers_checkpoint

    def capture_save(*args: object, **kwargs: object) -> Path:
        calls.append(dict(kwargs))
        forwarded = dict(kwargs)
        forwarded["tokenizer_path"] = None
        return real_save(*args, **forwarded)

    monkeypatch.setattr(hf_conversion, "save_transformers_checkpoint", capture_save)
    converted = convert_export(
        source_dir / "model.pt",
        tmp_path / "converted",
        formats=("transformers",),
        tokenizer_path=tokenizer_marker,
        language_pairs=pairs,
    )
    assert converted["formats"]["transformers"]["status"] == "ok"
    assert converted["metadata"]["language_pairs"] == [["ko", "ja"], ["en", "ru"]]

    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    trained = export_inference_models(
        tmp_path / "trained",
        model,
        config,
        context,
        7,
        formats=("transformers",),
        tokenizer_path=tokenizer_marker,
        language_pairs=pairs,
        pipeline_identity=translation_pipeline_identity(),
    )
    assert trained is not None
    assert trained["formats"]["transformers"]["status"] == "ok"
    assert len(calls) == 2
    for call in calls:
        assert call["tokenizer_path"] == tokenizer_marker
        assert call["languages"] == ["ko", "ja", "en", "ru"]
        assert call["language_pairs"] == [["ko", "ja"], ["en", "ru"]]
        assert call["translation_directions"] == [
            ["ko", "ja"],
            ["ja", "ko"],
            ["en", "ru"],
            ["ru", "en"],
        ]


def test_export_cli_accepts_repeated_language_pairs_and_transformers_default() -> None:
    from sion_translate.cli.export import build_parser

    args = build_parser().parse_args(
        [
            "model.pt",
            "--output",
            "converted",
            "--language-pair",
            "ko",
            "ja",
            "--language-pair",
            "en",
            "ru",
            "--unidirectional",
            "--release-name",
            "sion",
            "--release-version",
            "1.0",
            "--foundation-only",
        ]
    )
    assert args.language_pairs == [["ko", "ja"], ["en", "ru"]]
    assert args.unidirectional is True
    assert "transformers" in args.formats.split(",")
    assert args.release_name == "sion"
    assert args.release_version == "1.0"
    assert args.translation_capable is False


def test_training_export_manifest_merges_base_and_quantized_formats(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torchao")
    config = export_config()
    model = SionForConditionalGeneration(config)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    manifest = export_inference_models(
        tmp_path,
        model,
        config,
        context,
        4,
        ema=None,
        formats=("fp32", "int8"),
        pipeline_identity=translation_pipeline_identity(),
    )
    assert manifest is not None
    assert set(manifest["formats"]) == {"fp32", "int8"}
    stored = json.loads((tmp_path / "export_manifest.json").read_text())
    assert set(stored["formats"]) == {"fp32", "int8"}


@pytest.mark.parametrize("strict", [False, True])
def test_training_export_preserves_pipeline_identity_in_every_mode(
    tmp_path: Path,
    strict: bool,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    pipeline = translation_pipeline_identity()

    manifest = export_inference_models(
        tmp_path / ("strict" if strict else "regular"),
        model,
        config,
        context,
        11,
        formats=("fp32",),
        pipeline_identity=pipeline,
        strict=strict,
    )

    assert manifest is not None
    assert manifest["metadata"]["pipeline"] == pipeline


def test_translation_1_5_export_fails_closed_without_pipeline_identity(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    destination = tmp_path / "missing-pipeline"

    with pytest.raises(ValueError, match="requires pipeline identity"):
        export_module.export_state_dict_formats(
            destination,
            model.state_dict(),
            config,
            0,
            formats=("fp32",),
        )

    assert not (destination / "model.pt").exists()
    assert not (destination / "export_manifest.json").exists()


def test_export_validation_fails_closed_when_pipeline_identity_is_removed(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    manifest = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
    )
    manifest["metadata"].pop("pipeline")
    (tmp_path / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    validation = validate_export_directory(tmp_path)

    assert not validation["valid"]
    assert any(error["error_type"] == "InvalidPipelineIdentity" for error in validation["errors"])


def test_conversion_preserves_inherited_pipeline_identity(tmp_path: Path) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    pipeline = translation_pipeline_identity()
    source_dir = tmp_path / "source"
    export_state_dict_formats(
        source_dir,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
        metadata=build_export_metadata(config, pipeline_identity=pipeline),
    )

    converted = convert_export(
        source_dir / "model.pt",
        tmp_path / "converted",
        formats=("fp16",),
    )

    assert converted["metadata"]["pipeline"] == pipeline


def test_transformers_pipeline_sidecars_cross_validate_exact_identity(tmp_path: Path) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    pipeline = translation_pipeline_identity()
    metadata = build_export_metadata(config, pipeline_identity=pipeline)

    manifest = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("transformers",),
        metadata=metadata,
    )

    assert manifest["formats"]["transformers"]["status"] == "ok"
    transformers_dir = tmp_path / "transformers"
    config_payload = json.loads((transformers_dir / "config.json").read_text(encoding="utf-8"))
    export_payload = json.loads((transformers_dir / "sion_export.json").read_text(encoding="utf-8"))
    assert config_payload["pipeline"] == pipeline
    assert export_payload["pipeline"] == pipeline
    assert validate_export_directory(tmp_path)["valid"]

    config_payload["pipeline"]["branch"] = "tampered"
    (transformers_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["formats"]["transformers"].update(export_module._directory_entry(transformers_dir))
    (tmp_path / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    validation = validate_export_directory(tmp_path)
    assert not validation["valid"]
    assert any(
        "disagree about pipeline identity" in error["message"] for error in validation["errors"]
    )


def test_transformers_tokenizer_identity_is_cross_checked_with_manifest(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    manifest = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("transformers",),
    )
    transformers_dir = tmp_path / "transformers"
    false_digest = "e" * 64
    for sidecar_name in ("config.json", "sion_export.json"):
        sidecar_path = transformers_dir / sidecar_name
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        payload["tokenizer_sha256"] = false_digest
        sidecar_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    manifest["formats"]["transformers"].update(export_module._directory_entry(transformers_dir))
    (tmp_path / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    validation = validate_export_directory(tmp_path)

    assert not validation["valid"]
    assert any(
        "tokenizer identity does not match the manifest" in error["message"]
        for error in validation["errors"]
    )


def test_transformers_sion_export_contract_is_cross_checked_with_config(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    manifest = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("transformers",),
    )
    transformers_dir = tmp_path / "transformers"
    sidecar_path = transformers_dir / "sion_export.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["translation_capable"] = False
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["formats"]["transformers"].update(export_module._directory_entry(transformers_dir))
    (tmp_path / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    validation = validate_export_directory(tmp_path)

    assert not validation["valid"]
    assert any(
        "disagree about translation_capable" in error["message"] for error in validation["errors"]
    )


def _mock_distributed_export_control_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, BaseException | None]]:
    synchronizations: list[tuple[int, BaseException | None]] = []

    def record_synchronization(
        context: DistributedContext,
        error: BaseException | None,
    ) -> None:
        synchronizations.append((context.rank, error))
        if error is not None:
            raise error

    def clone_state(
        model: torch.nn.Module,
        _context: DistributedContext,
    ) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    monkeypatch.setattr(
        export_module,
        "_broadcast_training_export_invocation",
        lambda _context, _invocation: "test-export-invocation",
    )
    monkeypatch.setattr(
        export_module,
        "_all_ranks_observe_training_export_status",
        lambda _context, _visible: True,
    )
    monkeypatch.setattr(
        export_module,
        "_wait_for_training_export_acknowledgements",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        export_module,
        "_synchronize_rank0_exception",
        record_synchronization,
    )
    monkeypatch.setattr(export_module, "gather_full_state_dict", clone_state)
    return synchronizations


def test_distributed_regular_export_waits_on_durable_success_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronizations = _mock_distributed_export_control_plane(monkeypatch)
    config = export_config()
    model = SionForConditionalGeneration(config)
    directory = tmp_path / "regular"
    main = DistributedContext(0, 0, 2, torch.device("cpu"), True, "gloo")
    peer = DistributedContext(1, 1, 2, torch.device("cpu"), True, "gloo")

    manifest = export_inference_models(
        directory,
        model,
        config,
        main,
        17,
        formats=("fp32",),
        pipeline_identity=translation_pipeline_identity(),
    )
    peer_manifest = export_inference_models(
        directory,
        model,
        config,
        peer,
        17,
        formats=("fp32",),
        pipeline_identity=translation_pipeline_identity(),
    )

    assert manifest is not None
    assert peer_manifest is None
    statuses = export_module._read_training_export_status(
        export_module._training_export_status_path(
            directory,
            invocation="test-export-invocation",
        )
    )
    assert any(
        status.get("invocation") == "test-export-invocation" and status.get("state") == "complete"
        for status in statuses
    )
    # Each invocation uses only short setup/status-publication collectives before
    # rank 0 starts conversion. No peer waits in a process-group collective while
    # rank 0 performs conversion or publishing.
    assert synchronizations == [(0, None), (0, None), (1, None), (1, None)]


def test_distributed_regular_export_propagates_failure_through_durable_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronizations = _mock_distributed_export_control_plane(monkeypatch)
    config = export_config()
    model = SionForConditionalGeneration(config)
    directory = tmp_path / "regular"
    main = DistributedContext(0, 0, 2, torch.device("cpu"), True, "gloo")
    peer = DistributedContext(1, 1, 2, torch.device("cpu"), True, "gloo")

    def fail_export(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("injected long export failure")

    monkeypatch.setattr(export_module, "export_state_dict_formats", fail_export)

    with pytest.raises(RuntimeError, match="injected long export failure"):
        export_inference_models(
            directory,
            model,
            config,
            main,
            19,
            formats=("fp32",),
            pipeline_identity=translation_pipeline_identity(),
        )
    with pytest.raises(
        RuntimeError,
        match="rank 0 training export failed: RuntimeError: injected long export failure",
    ):
        export_inference_models(
            directory,
            model,
            config,
            peer,
            19,
            formats=("fp32",),
            pipeline_identity=translation_pipeline_identity(),
        )

    assert synchronizations == [(0, None), (0, None), (1, None), (1, None)]


def test_training_export_wait_rejects_a_stale_completion_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = {
        "schema": export_module._TRAINING_EXPORT_STATUS_SCHEMA,
        "state": "complete",
        "invocation": "stale-invocation",
        "step": 23,
        "release_name": "sion_translate",
    }
    current_running = {
        **stale,
        "state": "running",
        "invocation": "current-invocation",
    }
    current_complete = {**current_running, "state": "complete"}
    observations = iter([[stale], [current_running], [current_complete]])
    reads = 0

    def observe(_status_path: Path) -> list[dict[str, object]]:
        nonlocal reads
        reads += 1
        return next(observations)

    monkeypatch.setattr(export_module, "_read_training_export_status", observe)
    monkeypatch.setattr(export_module.time, "sleep", lambda _seconds: None)

    export_module._wait_for_training_export_status(
        tmp_path / "status.json",
        invocation="current-invocation",
        step=23,
        release_name="sion_translate",
    )

    assert reads == 3


def test_training_export_wait_fails_when_rank_zero_heartbeat_stalls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = {
        "schema": export_module._TRAINING_EXPORT_STATUS_SCHEMA,
        "state": "running",
        "invocation": "heartbeat-stall",
        "step": 27,
        "release_name": "sion_translate",
        "heartbeat_sequence": 0,
    }
    clock = iter((0.0, 1.0, 3.0))
    monkeypatch.setattr(
        export_module,
        "_read_training_export_status",
        lambda _path: [running],
    )
    monkeypatch.setattr(export_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(export_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match="stopped publishing heartbeats"):
        export_module._wait_for_training_export_status(
            tmp_path / "status.json",
            invocation="heartbeat-stall",
            step=27,
            release_name="sion_translate",
            stale_timeout_seconds=2.0,
        )


def test_training_export_terminal_status_cannot_be_overwritten_by_heartbeat(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "status.json"
    running = {
        "schema": export_module._TRAINING_EXPORT_STATUS_SCHEMA,
        "state": "running",
        "invocation": "heartbeat-race",
        "step": 28,
        "release_name": "sion_translate",
        "heartbeat_sequence": 0,
    }
    handles = export_module._initialize_training_export_status(status_path, running)
    heartbeat = export_module._start_training_export_heartbeat(
        handles,
        running,
        interval_seconds=0.001,
    )
    try:
        export_module.time.sleep(0.01)
        assert not export_module._stop_training_export_heartbeat(heartbeat)
        export_module._publish_training_export_status(
            handles,
            {**running, "state": "complete"},
        )
        export_module.time.sleep(0.005)
    finally:
        export_module._close_training_export_status(handles)

    statuses = export_module._read_training_export_status(status_path)
    assert statuses
    assert {status["state"] for status in statuses} == {"complete"}


def test_training_export_rank_zero_rejects_peer_observer_failure(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    export_module._publish_training_export_acknowledgement(
        status_path,
        invocation="observer-failure",
        step=30,
        release_name="sion_translate",
        rank=1,
        observed_state="observer_failed",
        error=TimeoutError("lost heartbeat"),
    )

    with pytest.raises(RuntimeError, match=r"rank 1.*TimeoutError.*lost heartbeat"):
        export_module._wait_for_training_export_acknowledgements(
            status_path,
            invocation="observer-failure",
            step=30,
            release_name="sion_translate",
            world_size=2,
            terminal_state="complete",
            timeout_seconds=1.0,
        )


def test_training_export_terminal_status_survives_one_channel_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "status.json"
    running = {
        "schema": export_module._TRAINING_EXPORT_STATUS_SCHEMA,
        "state": "running",
        "invocation": "redundancy-test",
        "step": 29,
        "release_name": "sion_translate",
    }
    complete = {**running, "state": "complete"}
    handles = export_module._initialize_training_export_status(status_path, running)
    real_overwrite = export_module._overwrite_training_export_status

    def fail_primary(
        handle: object,
        payload: dict[str, object],
    ) -> None:
        if handle is handles[0] and payload.get("state") == "complete":
            raise OSError("injected primary status failure")
        real_overwrite(handle, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(
        export_module,
        "_overwrite_training_export_status",
        fail_primary,
    )
    try:
        export_module._publish_training_export_status(handles, complete)
    finally:
        export_module._close_training_export_status(handles)

    statuses = export_module._read_training_export_status(status_path)
    assert any(status.get("state") == "complete" for status in statuses)


def test_concurrent_training_exports_keep_status_generations_isolated(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "shared-export"
    paths = {
        invocation: export_module._training_export_status_path(
            directory,
            invocation=invocation,
        )
        for invocation in ("invocation-a", "invocation-b")
    }
    assert paths["invocation-a"] != paths["invocation-b"]

    def status(invocation: str, state: str) -> dict[str, object]:
        return {
            "schema": export_module._TRAINING_EXPORT_STATUS_SCHEMA,
            "state": state,
            "invocation": invocation,
            "step": 31,
            "release_name": "sion_translate",
        }

    # Interleave both generations exactly as concurrent rank-0 exporters can:
    # B initializes after A, then A publishes through its still-open handles.
    handles_a = export_module._initialize_training_export_status(
        paths["invocation-a"],
        status("invocation-a", "running"),
    )
    handles_b = export_module._initialize_training_export_status(
        paths["invocation-b"],
        status("invocation-b", "running"),
    )
    try:
        export_module._publish_training_export_status(
            handles_a,
            status("invocation-a", "complete"),
        )
        export_module._publish_training_export_status(
            handles_b,
            status("invocation-b", "complete"),
        )
    finally:
        export_module._close_training_export_status(handles_a)
        export_module._close_training_export_status(handles_b)

    for invocation, path in paths.items():
        statuses = export_module._read_training_export_status(path)
        assert any(
            item.get("invocation") == invocation and item.get("state") == "complete"
            for item in statuses
        )


def test_subset_export_preserves_unrequested_manifest_entries(
    tmp_path: Path,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    first = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("fp32", "fp16"),
    )
    fp32_entry = dict(first["formats"]["fp32"])

    second = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("bf16",),
    )
    assert second["formats"]["fp32"] == fp32_entry
    assert set(second["formats"]) == {"fp32", "fp16", "bf16"}
    assert second["last_requested_formats"] == ["bf16"]


def test_subset_export_does_not_merge_artifacts_from_different_weights(
    tmp_path: Path,
) -> None:
    config = export_config()
    first_model = SionForConditionalGeneration(config)
    export_state_dict_formats(
        tmp_path,
        first_model.state_dict(),
        config,
        0,
        formats=("fp32", "fp16"),
    )
    second_model = SionForConditionalGeneration(config)
    second = export_state_dict_formats(
        tmp_path,
        second_model.state_dict(),
        config,
        0,
        formats=("bf16",),
    )
    assert set(second["formats"]) == {"bf16"}
    assert second["requested_formats"] == ["bf16"]


def test_subset_export_does_not_merge_incompatible_metadata(tmp_path: Path) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    first = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
        metadata=build_export_metadata(
            config,
            language_pairs=(("ko", "ja"),),
        ),
    )
    second = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("bf16",),
        metadata=build_export_metadata(
            config,
            language_pairs=(("en", "ru"),),
        ),
    )
    assert first["artifact_set_id"] == second["artifact_set_id"]
    assert first["metadata_compatibility_id"] != second["metadata_compatibility_id"]
    assert set(second["formats"]) == {"bf16"}
    assert validate_export_directory(tmp_path)["valid"]


@pytest.mark.parametrize(
    ("first_overrides", "second_overrides"),
    [
        ({"step": 1}, {"step": 2}),
        (
            {"language_pairs": (("ko", "ja"),), "languages": ("ko", "ja", "en")},
            {"language_pairs": (("ko", "ja"),), "languages": ("ko", "ja", "ru")},
        ),
        (
            {"source": "source-a"},
            {"source": "source-b"},
        ),
    ],
)
def test_subset_export_does_not_merge_contradictory_provenance(
    tmp_path: Path,
    first_overrides: dict[str, Any],
    second_overrides: dict[str, Any],
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    first_overrides = dict(first_overrides)
    second_overrides = dict(second_overrides)
    for overrides in (first_overrides, second_overrides):
        source_name = overrides.get("source")
        if isinstance(source_name, str):
            source = tmp_path / source_name
            source.write_bytes(source_name.encode("utf-8"))
            overrides["source"] = source
    first = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
        metadata=build_export_metadata(config, **first_overrides),
    )
    second = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("bf16",),
        metadata=build_export_metadata(config, **second_overrides),
    )

    assert first["artifact_set_id"] == second["artifact_set_id"]
    assert first["metadata_compatibility_id"] != second["metadata_compatibility_id"]
    assert set(second["formats"]) == {"bf16"}
    assert validate_export_directory(tmp_path)["valid"]


def test_new_weight_generation_removes_unreferenced_known_artifacts(
    tmp_path: Path,
) -> None:
    config = export_config()
    first_model = SionForConditionalGeneration(config)
    second_state = {
        name: tensor.detach().clone() for name, tensor in first_model.state_dict().items()
    }
    first_name = next(iter(second_state))
    second_state[first_name] = second_state[first_name] + 1
    export_state_dict_formats(
        tmp_path,
        first_model.state_dict(),
        config,
        0,
        formats=("fp32",),
    )
    assert (tmp_path / "model.pt").is_file()

    manifest = export_state_dict_formats(
        tmp_path,
        second_state,
        config,
        0,
        formats=("bf16",),
    )

    assert set(manifest["formats"]) == {"bf16"}
    assert not (tmp_path / "model.pt").exists()
    assert (tmp_path / "model_bf16.pt").is_file()
    assert validate_export_directory(tmp_path)["valid"]


def test_failed_reexport_does_not_delete_existing_valid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torchao")
    config = export_config()
    model = SionForConditionalGeneration(config)
    export_dir = tmp_path / "run" / "exports" / "best"
    first = export_state_dict_formats(
        export_dir,
        model.state_dict(),
        config,
        0,
        formats=("int8",),
    )
    artifact = export_dir / first["formats"]["int8"]["file"]
    original = artifact.read_bytes()

    def fail_save(payload: object, path: Path) -> None:
        raise RuntimeError("intentional conversion failure")

    monkeypatch.setattr(
        "sion_translate.training.export._atomic_torch_save",
        fail_save,
    )
    for attempt in (1, 2):
        retried = export_state_dict_formats(
            export_dir,
            model.state_dict(),
            config,
            0,
            formats=("int8",),
        )
        entry = retried["formats"]["int8"]
        assert entry["status"] == "ok"
        assert entry["last_error"]["status"] == "error"
        assert entry["failed_export_attempts"] == attempt
        assert find_exported_model(tmp_path / "run", int8=True) == artifact
    assert artifact.read_bytes() == original


def test_custom_sion_gguf_is_real_mixed_k_quant(tmp_path: Path) -> None:
    gguf = pytest.importorskip("gguf")
    config = export_config(d_model=256)
    model = SionForConditionalGeneration(config)
    pipeline = translation_pipeline_identity()
    manifest = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("gguf_q4_k_m",),
        metadata=build_export_metadata(
            config,
            language_pairs=(("ko", "ja"),),
            pipeline_identity=pipeline,
        ),
    )
    entry = manifest["formats"]["gguf_q4_k_m"]
    assert entry["status"] == "ok"
    assert entry["runtime_supported_by_llama_cpp"] is False
    assert entry["tensor_counts"]["q4_k"] > 0
    assert entry["tensor_counts"]["q5_k"] > 0
    reader = gguf.GGUFReader(tmp_path / entry["file"])
    assert json.loads(reader.fields["sion.pipeline"].contents()) == pipeline
    assert json.loads(reader.fields["sion.model_config"].contents()) == asdict(config)
    assert manifest["model_config"] == asdict(config)
    assert manifest["pad_id"] == 0
    assert json.loads(reader.fields["sion.languages"].contents()) == ["ko", "ja"]
    assert json.loads(reader.fields["sion.language_pairs"].contents()) == [["ko", "ja"]]
    assert json.loads(reader.fields["sion.translation_directions"].contents()) == [
        ["ko", "ja"],
        ["ja", "ko"],
    ]
    tensor_types = {tensor.tensor_type.name for tensor in reader.tensors}
    assert {"Q4_K", "Q5_K", "F16"} <= tensor_types
    state = model.state_dict()
    for tensor in reader.tensors:
        assert tuple(map(int, tensor.shape)) == tuple(reversed(state[tensor.name].shape))
    stored_manifest = json.loads((tmp_path / "export_manifest.json").read_text())
    assert stored_manifest["formats"]["gguf_q4_k_m"]["sha256"] == entry["sha256"]
    assert validate_export_directory(tmp_path)["valid"]


def test_0b_style_gguf_identity_fallback_is_legacy_only_before_1_5(
    tmp_path: Path,
) -> None:
    pytest.importorskip("gguf")
    config = export_config()
    model = SionForConditionalGeneration(config)
    pairs = (("ko", "ja"),)
    metadata = build_export_metadata(
        config,
        release_version="1.0",
        language_pairs=pairs,
    )
    manifest = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("gguf_q4_k_m",),
        metadata=metadata,
    )
    artifact = tmp_path / manifest["formats"]["gguf_q4_k_m"]["file"]
    artifact.unlink()
    counts = _write_0b_style_legacy_gguf(
        artifact,
        model.state_dict(),
        language_pairs=[["ko", "ja"]],
    )
    manifest["formats"]["gguf_q4_k_m"].update(export_module._file_entry(artifact))
    manifest["formats"]["gguf_q4_k_m"]["tensor_counts"] = counts
    _store_manifest(tmp_path, manifest)

    legacy_validation = validate_export_directory(tmp_path)

    assert legacy_validation["valid"]
    assert (
        legacy_validation["formats"]["gguf_q4_k_m"]["inspection"]["identity_source"]
        == "legacy-manifest"
    )

    manifest["metadata"]["release_version"] = "1.5"
    manifest["metadata"]["pipeline"] = translation_pipeline_identity()
    manifest["metadata_compatibility_id"] = export_module._metadata_compatibility_id(
        manifest["metadata"]
    )
    _store_manifest(tmp_path, manifest)

    current_validation = validate_export_directory(tmp_path)

    assert not current_validation["valid"]
    assert any(
        error.get("format") == "gguf_q4_k_m"
        and "sion.translation_capable must be a boolean" in error["message"]
        for error in current_validation["errors"]
    )


def test_foundation_gguf_records_its_role_without_translation_pipeline(
    tmp_path: Path,
) -> None:
    gguf = pytest.importorskip("gguf")
    config = export_config()
    model = SionForConditionalGeneration(config)
    metadata = build_export_metadata(
        config,
        release_name="sion",
        translation_capable=False,
    )

    manifest = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("gguf_q4_k_m",),
        metadata=metadata,
        release_name="sion",
        translation_capable=False,
    )

    entry = manifest["formats"]["gguf_q4_k_m"]
    assert entry["status"] == "ok"
    reader = gguf.GGUFReader(tmp_path / entry["file"])
    assert reader.fields["general.name"].contents() == "sion"
    assert reader.fields["sion.release_name"].contents() == "sion"
    assert reader.fields["sion.release_version"].contents() == "1.5"
    assert reader.fields["sion.translation_capable"].contents() is False
    assert "sion.pipeline" not in reader.fields
    assert validate_export_directory(tmp_path)["valid"]


def test_foundation_branch_translation_gguf_binds_lineage_inputs(
    tmp_path: Path,
) -> None:
    gguf = pytest.importorskip("gguf")
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"foundation branch GGUF tokenizer")
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    pipeline = foundation_pipeline_identity()
    pipeline["foundation"]["tokenizer_sha256"] = tokenizer_sha256
    config = export_config()
    model = SionForConditionalGeneration(config)
    metadata = build_export_metadata(
        config,
        tokenizer_path=tokenizer,
        languages=("ko", "ja"),
        pipeline_identity=pipeline,
    )

    manifest = export_state_dict_formats(
        tmp_path / "export",
        model.state_dict(),
        config,
        0,
        formats=("gguf_q4_k_m",),
        metadata=metadata,
        tokenizer_path=tokenizer,
    )

    entry = manifest["formats"]["gguf_q4_k_m"]
    assert entry["status"] == "ok"
    reader = gguf.GGUFReader(tmp_path / "export" / entry["file"])
    assert json.loads(reader.fields["sion.languages"].contents()) == ["ko", "ja"]
    assert reader.fields["sion.tokenizer.sha256"].contents() == tokenizer_sha256
    assert json.loads(reader.fields["sion.pipeline"].contents()) == pipeline
    assert validate_export_directory(tmp_path / "export")["valid"]


def test_gguf_inspection_is_cross_checked_with_manifest_language_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    metadata = build_export_metadata(
        config,
        language_pairs=(("ko", "ja"),),
        pipeline_identity=translation_pipeline_identity(),
    )
    export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("gguf_q4_k_m",),
        metadata=metadata,
    )
    real_inspect = export_module._inspect_sion_gguf

    def tampered_inspection(path: Path, **kwargs: Any) -> dict[str, Any]:
        inspection = real_inspect(path, **kwargs)
        inspection["language_pairs"] = [["en", "ru"]]
        return inspection

    monkeypatch.setattr(export_module, "_inspect_sion_gguf", tampered_inspection)

    validation = validate_export_directory(tmp_path)

    assert not validation["valid"]
    assert any(
        "GGUF language pairs do not match the manifest" in error["message"]
        for error in validation["errors"]
    )


def test_gguf_inspection_is_cross_checked_with_manifest_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = export_config()
    model = SionForConditionalGeneration(config)
    export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("gguf_q4_k_m",),
        metadata=build_export_metadata(
            config,
            language_pairs=(("ko", "ja"),),
            pipeline_identity=translation_pipeline_identity(),
        ),
    )
    real_inspect = export_module._inspect_sion_gguf

    def tampered_inspection(path: Path, **kwargs: Any) -> dict[str, Any]:
        inspection = real_inspect(path, **kwargs)
        inspection["model_config"] = dict(inspection["model_config"])
        inspection["model_config"]["num_heads"] += 1
        return inspection

    monkeypatch.setattr(export_module, "_inspect_sion_gguf", tampered_inspection)

    validation = validate_export_directory(tmp_path)

    assert not validation["valid"]
    assert any(
        "GGUF model_config does not match the manifest" in error["message"]
        for error in validation["errors"]
    )


def test_native_export_embeds_and_validates_token_feature_identity(tmp_path: Path) -> None:
    tokenizer = tmp_path / "source-tokenizer.model"
    tokenizer.write_bytes(b"tokenizer identity fixture")
    features = tmp_path / "token_features.npz"
    zeros = np.zeros(64, dtype=np.uint8)
    np.savez_compressed(
        features,
        script=zeros,
        onset=zeros,
        vowel=zeros,
        coda=zeros,
    )
    config = export_config()
    model = SionForConditionalGeneration(config)
    output = tmp_path / "embedded"
    manifest = export_state_dict_formats(
        output,
        model.state_dict(),
        config,
        0,
        formats=("fp32",),
        tokenizer_path=tokenizer,
        token_features_path=features,
    )
    assert manifest["metadata"]["embedded_sidecars"] == [
        "tokenizer",
        "token_features",
    ]
    assert (output / tokenizer.name).read_bytes() == tokenizer.read_bytes()
    assert (output / features.name).read_bytes() == features.read_bytes()
    assert validate_export_directory(output)["valid"]

    with (output / features.name).open("ab") as handle:
        handle.write(b"tampered")
    invalid = validate_export_directory(output)
    assert not invalid["valid"]
    assert any(error["error_type"] == "SidecarMismatch" for error in invalid["errors"])


def test_strict_final_export_is_directory_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = export_config()
    destination = tmp_path / "best"
    original_model = SionForConditionalGeneration(config)
    export_state_dict_formats(
        destination,
        original_model.state_dict(),
        config,
        0,
        formats=("fp32",),
    )
    original_manifest = (destination / "export_manifest.json").read_bytes()
    original_artifact = (destination / "model.pt").read_bytes()

    def fail_gguf(*_args: object, **_kwargs: object) -> dict[str, int]:
        raise RuntimeError("intentional final GGUF failure")

    monkeypatch.setattr(
        "sion_translate.training.export._write_sion_gguf",
        fail_gguf,
    )
    replacement_model = SionForConditionalGeneration(config)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    with pytest.raises(RuntimeError, match="required model exports failed"):
        export_inference_models(
            destination,
            replacement_model,
            config,
            context,
            2,
            formats=("fp32", "gguf_q4_k_m"),
            pipeline_identity=translation_pipeline_identity(),
            strict=True,
        )

    assert (destination / "export_manifest.json").read_bytes() == original_manifest
    assert (destination / "model.pt").read_bytes() == original_artifact
    assert not list(tmp_path.glob(".best.tmp-*"))


def test_atomic_replace_directory_restores_backup_after_partial_fallback_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "best"
    destination.mkdir()
    (destination / "original.txt").write_text("known good", encoding="utf-8")
    temporary = tmp_path / ".best.tmp-test"
    temporary.mkdir()
    for name in ("first.txt", "second.txt", "third.txt"):
        (temporary / name).write_text(name, encoding="utf-8")

    real_replace = export_module.os.replace
    child_moves = 0

    def fail_after_one_child(source: str | Path, target: str | Path) -> None:
        nonlocal child_moves
        source_path = Path(source)
        target_path = Path(target)
        if source_path == temporary and target_path == destination:
            raise PermissionError("injected Windows directory rename failure")
        if source_path.parent == temporary:
            child_moves += 1
            if child_moves == 2:
                raise OSError("injected child move failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(export_module.os, "replace", fail_after_one_child)

    with pytest.raises(OSError, match="injected child move failure"):
        export_module._atomic_replace_directory(temporary, destination)

    assert child_moves == 2
    assert (destination / "original.txt").read_text(encoding="utf-8") == "known good"
    assert {path.name for path in destination.iterdir()} == {"original.txt"}
    assert not list(tmp_path.glob(".best.backup-*"))


def test_atomic_replace_directory_uses_a_cross_process_publish_lock(tmp_path: Path) -> None:
    destination = tmp_path / "exports" / "best"
    destination.mkdir(parents=True)
    (destination / "generation.txt").write_text("old", encoding="utf-8")
    temporary = tmp_path / "exports" / ".best.tmp-child"
    temporary.mkdir()
    (temporary / "generation.txt").write_text("new", encoding="utf-8")
    lock_root = export_module._export_publish_lock_root(destination)
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        import sion_translate.training.export as export_module

        export_module._EXPORT_LOCK_TIMEOUT_SECONDS = 0.15
        try:
            export_module._atomic_replace_directory(
                Path({str(temporary)!r}),
                Path({str(destination)!r}),
            )
        except RuntimeError as error:
            print(error)
            sys.exit(3)
        """
    )

    with export_module.artifact_lock(lock_root):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            timeout=10.0,
        )

    assert result.returncode == 3, result.stderr
    assert "다른 프로세스에 잠겨" in result.stdout
    assert (destination / "generation.txt").read_text(encoding="utf-8") == "old"
    assert (temporary / "generation.txt").read_text(encoding="utf-8") == "new"


def test_complete_file_and_manifest_generation_uses_a_cross_process_lock(tmp_path: Path) -> None:
    destination = tmp_path / "exports" / "best"
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        import torch
        import sion_translate.training.export as export_module
        from sion_translate.config import ExperimentalConfig, ModelConfig

        export_module._EXPORT_LOCK_TIMEOUT_SECONDS = 0.15
        config = ModelConfig(
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
            experimental=ExperimentalConfig(),
        )
        try:
            export_module.export_state_dict_formats(
                Path({str(destination)!r}),
                {{"sentinel": torch.tensor([2.0])}},
                config,
                0,
                formats=("fp32",),
                metadata=export_module.build_export_metadata(
                    config,
                    pipeline_identity={{
                        "schema": "sion-translation-pipeline-v2",
                        "branch": "translation-only",
                    }},
                ),
            )
        except RuntimeError as error:
            print(error)
            sys.exit(3)
        """
    )

    with export_module.artifact_lock(export_module._export_publish_lock_root(destination)):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            timeout=10.0,
        )

    assert result.returncode == 3, result.stderr
    assert "다른 프로세스에 잠겨" in result.stdout
    assert not destination.exists()


def test_handoff_cleanup_failure_never_exposes_a_partial_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "best"
    destination.mkdir()
    (destination / "original.txt").write_text("known good", encoding="utf-8")
    temporary = tmp_path / ".best.tmp-test"
    temporary.mkdir()
    for name in ("first.txt", "second.txt"):
        (temporary / name).write_text(name, encoding="utf-8")

    real_remove = export_module._remove_artifact
    real_replace = export_module.os.replace
    child_moves = 0

    def fail_install(source: str | Path, target: str | Path) -> None:
        nonlocal child_moves
        source_path = Path(source)
        target_path = Path(target)
        if source_path == temporary and target_path == destination:
            raise PermissionError("injected Windows directory rename failure")
        if source_path.parent == temporary:
            child_moves += 1
            if child_moves == 2:
                raise OSError("injected child move failure")
        real_replace(source_path, target_path)

    def fail_handoff_cleanup(path: Path) -> None:
        if path.name.startswith(".best.handoff-"):
            raise PermissionError("injected handoff cleanup failure")
        real_remove(path)

    monkeypatch.setattr(export_module.os, "replace", fail_install)
    monkeypatch.setattr(export_module, "_remove_artifact", fail_handoff_cleanup)

    with pytest.raises(OSError, match="injected child move failure"):
        export_module._atomic_replace_directory(temporary, destination)

    assert child_moves == 2
    assert {path.name for path in destination.iterdir()} == {"original.txt"}
    assert not (destination / "first.txt").exists()
    handoffs = list(tmp_path.glob(".best.handoff-*"))
    assert len(handoffs) == 1
    assert (handoffs[0] / "first.txt").is_file()


def test_restore_failure_preserves_the_original_install_error_and_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "best"
    destination.mkdir()
    (destination / "original.txt").write_text("known good", encoding="utf-8")
    temporary = tmp_path / ".best.tmp-test"
    temporary.mkdir()
    (temporary / "new.txt").write_text("new", encoding="utf-8")

    real_replace = export_module.os.replace

    def fail_install_and_restore(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if source_path == temporary and target_path == destination:
            raise PermissionError("injected Windows directory rename failure")
        if source_path.parent == temporary:
            raise OSError("injected child install failure")
        if source_path.name.startswith(".best.backup-") and target_path == destination:
            raise PermissionError("injected restore failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(export_module.os, "replace", fail_install_and_restore)

    with pytest.raises(RuntimeError, match="recoverable backup") as captured:
        export_module._atomic_replace_directory(temporary, destination)

    assert isinstance(captured.value.__cause__, OSError)
    assert "injected child install failure" in str(captured.value.__cause__)
    backups = list(tmp_path.glob(".best.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "original.txt").read_text(encoding="utf-8") == "known good"
    assert not destination.exists()


def test_locked_empty_staging_shell_does_not_turn_success_into_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".transformers.tmp-locked"
    staging.mkdir()

    def locked_shell(_path: Path) -> None:
        raise PermissionError("injected locked empty shell")

    monkeypatch.setattr(export_module, "_remove_artifact", locked_shell)

    export_module._remove_staging_artifact(staging)

    assert staging.is_dir()
    assert not list(staging.iterdir())
