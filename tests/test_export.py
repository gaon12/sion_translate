from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
import types
from dataclasses import asdict
from pathlib import Path

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
    build_export_metadata,
    convert_export,
    export_inference_models,
    export_state_dict_formats,
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
    payload["metadata"].pop("release_name")
    payload["metadata"].pop("release_version")
    payload["metadata"].pop("translation_capable")
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
    )
    assert manifest is not None
    assert set(manifest["formats"]) == {"fp32", "int8"}
    stored = json.loads((tmp_path / "export_manifest.json").read_text())
    assert set(stored["formats"]) == {"fp32", "int8"}


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
    manifest = export_state_dict_formats(
        tmp_path,
        model.state_dict(),
        config,
        0,
        formats=("gguf_q4_k_m",),
    )
    entry = manifest["formats"]["gguf_q4_k_m"]
    assert entry["status"] == "ok"
    assert entry["runtime_supported_by_llama_cpp"] is False
    assert entry["tensor_counts"]["q4_k"] > 0
    assert entry["tensor_counts"]["q5_k"] > 0
    reader = gguf.GGUFReader(tmp_path / entry["file"])
    tensor_types = {tensor.tensor_type.name for tensor in reader.tensors}
    assert {"Q4_K", "Q5_K", "F16"} <= tensor_types
    state = model.state_dict()
    for tensor in reader.tensors:
        assert tuple(map(int, tensor.shape)) == tuple(reversed(state[tensor.name].shape))
    stored_manifest = json.loads((tmp_path / "export_manifest.json").read_text())
    assert stored_manifest["formats"]["gguf_q4_k_m"]["sha256"] == entry["sha256"]


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
