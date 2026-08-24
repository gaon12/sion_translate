"""foundation 단계를 실제로 한 번 돌려 본다.

계획·설정 유도는 `test_foundation_stage.py` 가 봅니다. 여기서는 그 설정으로
정말 학습이 돌고, **두 번째 실행이 다시 학습하지 않는지** 를 봅니다. 이
단계는 파이프라인에서 가장 오래 걸리는 구간이라, 번역 학습이 실패해 다시
실행할 때마다 며칠짜리 사전학습을 반복하면 안 됩니다.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sion_translate.cli.train import (
    FOUNDATION_COMPLETION_FILENAME,
    _foundation_checkpoint_artifact_sha256,
    _foundation_completion_matches_inputs,
    _foundation_plan_for_lineage,
    _foundation_source_sampling_weights,
    resolve_foundation_lineage,
    run_foundation_stage,
)
from sion_translate.config import AppConfig, ExperimentalConfig, ModelConfig
from sion_translate.data.prepare_foundation import prepare_foundation_dataset
from sion_translate.foundation import (
    build_foundation_config,
    build_translation_pipeline_identity,
    foundation_run_directory,
    plan_foundation_stage,
)
from sion_translate.fingerprint import file_sha256
from sion_translate.model import SionForConditionalGeneration
from sion_translate.tokenizer import SionTokenizer, train_tokenizer
from sion_translate.training.distributed import DistributedContext
from sion_translate.training.checkpoint import DCP_COMPLETION_FILENAME


@pytest.fixture(scope="module")
def tokenizer_model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("foundation_tokenizer")
    shard = directory / "pairs.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for index in range(400):
            handle.write(
                json.dumps(
                    {
                        "ko": f"한국어 문장 {index} 입니다 그리고 조금 더 깁니다",
                        "ja": f"日本語の文 {index} です そしてもう少し長いです",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return train_tokenizer(
        [str(shard)],
        directory / "out",
        vocab_size=700,
        num_workers=1,
        language_pairs=[["kj", "ko"], ["kj", "ja"], ["ko", "ja"]],
        source_only_languages=["kj"],
    )


def _prepared(tmp_path, tokenizer_model, *, empty_language: str | None = None):
    """토크나이저·코퍼스·데이터셋이 준비된 설정과 모델을 만든다."""

    corpus = tmp_path / "corpus"
    for language, template in (
        ("ko", "한국어 단일어 문장 {} 입니다 조금 더 깁니다"),
        ("ja", "日本語の単言語文 {} です もう少し長いです"),
    ):
        (corpus / language).mkdir(parents=True)
        lines = (
            ["짧음" if language == "ko" else "短い"] * 240
            if language == empty_language
            else [template.format(index) for index in range(240)]
        )
        (corpus / language / "a.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    config = AppConfig()
    config.data.language_pairs = [["ko", "ja"]]
    config.data.tokenizer_model = str(tokenizer_model)
    config.data.tokenizer_features = str(tokenizer_model.parent / "token_features.npz")
    config.data.dataset_dir = str(tmp_path / "dataset")
    config.data.num_workers = 0
    config.data.bucket_size = 16
    config.data.max_source_length = 32
    config.data.max_target_length = 32
    config.foundation.corpus_dir = str(corpus)
    config.foundation.dataset_dir = str(tmp_path / "foundation_dataset")
    config.foundation.max_steps = 2
    config.foundation.warmup_steps = 1
    config.foundation.batch_size_per_gpu = 2
    config.foundation.eval_every = 1
    config.foundation.eval_batches = 1
    config.foundation.save_every = 1
    config.foundation.shard_size = 64
    config.foundation.validation_fraction = 0.1
    config.foundation.final_export_formats = ["fp32"]
    config.training.output_dir = str(tmp_path / "runs")
    config.training.tensorboard = False
    config.training.ema_decay = 0.0
    config.training.precision = "fp32"

    tokenizer = SionTokenizer(tokenizer_model)
    config.model = ModelConfig(
        vocab_size=len(tokenizer),
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
        experimental=ExperimentalConfig(),
    )
    config.validate()

    plan = plan_foundation_stage(config)
    prepare_foundation_dataset(
        plan.discovery,
        tokenizer_model,
        config.foundation.dataset_dir,
        minimum_characters=config.foundation.minimum_characters,
        maximum_characters=config.foundation.maximum_characters,
        max_tokens=config.data.max_source_length - 2,
        max_target_tokens=config.data.max_target_length - 1,
        deduplicate=config.foundation.deduplicate,
        shard_size=config.foundation.shard_size,
        validation_fraction=config.foundation.validation_fraction,
        language_sampling_alpha=config.foundation.language_sampling_alpha,
        minimum_language_share=config.foundation.minimum_language_share,
        reasoning_sample_share=config.foundation.reasoning_sample_share,
        release_name=config.foundation.release_name,
    )
    model = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    return config, plan, model, tokenizer, context


def _mutate_checkpoint_weights(checkpoint: Path, amount: float = 7.0) -> None:
    payload = torch.load(checkpoint / "checkpoint.pt", map_location="cpu", weights_only=True)
    payload["model"]["token_embedding.weight"] = (
        payload["model"]["token_embedding.weight"].clone() + amount
    )
    temporary = checkpoint / "checkpoint.tampered.pt"
    torch.save(payload, temporary)
    temporary.replace(checkpoint / "checkpoint.pt")


@pytest.mark.parametrize("distributed_artifact", [DCP_COMPLETION_FILENAME, ".metadata"])
def test_foundation_digest_rejects_mixed_checkpoint_formats(
    tmp_path: Path,
    distributed_artifact: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.pt").write_bytes(b"local")
    (checkpoint / distributed_artifact).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="mixes local and distributed"):
        _foundation_checkpoint_artifact_sha256(checkpoint)


def test_the_stage_trains_and_marks_itself_complete(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    before = model.token_embedding.weight.detach().clone()

    import sion_translate.cli.train as train_module

    sampler_arguments: list[dict[str, object]] = []
    original_sampler = train_module.DistributedBucketBatchSampler

    def recording_sampler(*args, **kwargs):
        sampler_arguments.append(dict(kwargs))
        return original_sampler(*args, **kwargs)

    monkeypatch.setattr(train_module, "DistributedBucketBatchSampler", recording_sampler)

    outcome = run_foundation_stage(config, plan, model, tokenizer, context)

    assert outcome.ran
    assert outcome.selected_step is not None
    run_root = foundation_run_directory(config)
    assert (run_root / FOUNDATION_COMPLETION_FILENAME).is_file()
    assert (run_root / "checkpoints" / "best").exists()
    # 실제로 학습이 일어났다면 가중치가 움직여야 한다.
    assert not torch.allclose(model.token_embedding.weight, before)

    marker = json.loads((run_root / FOUNDATION_COMPLETION_FILENAME).read_text(encoding="utf-8"))
    assert marker["stage"] == "foundation"
    assert marker["schema"] == "sion-foundation-completion-v2"
    assert marker["release_name"] == "sion"
    assert marker["release_version"] == "1.5"
    assert sorted(marker["languages"]) == ["ja", "ko"]
    assert len(marker["foundation_manifest_sha256"]) == 64
    assert len(marker["tokenizer_sha256"]) == 64
    assert len(marker["checkpoint_identity_sha256"]) == 64
    assert len(marker["checkpoint_artifact_sha256"]) == 64
    assert len(marker["export_manifest_sha256"]) == 64
    lineage = resolve_foundation_lineage(config, plan, context)
    assert lineage == {
        "schema": "sion-foundation-lineage-v1",
        "release_name": "sion",
        "release_version": "1.5",
        "languages": list(plan.languages),
        "selected_step": marker["selected_step"],
        "foundation_manifest_sha256": marker["foundation_manifest_sha256"],
        "tokenizer_sha256": marker["tokenizer_sha256"],
        "checkpoint_identity_sha256": marker["checkpoint_identity_sha256"],
        "checkpoint_artifact_sha256": marker["checkpoint_artifact_sha256"],
    }
    pipeline = build_translation_pipeline_identity(plan, foundation_lineage=lineage)
    assert pipeline["foundation"] == lineage
    assert sampler_arguments[0]["source_sampling_weights_by_id"]
    assert math.isinf(sampler_arguments[0]["max_source_upsampling"])
    assert "source_sampling_weights_by_id" not in sampler_arguments[1]


def test_foundation_publishes_only_positive_mass_train_languages(
    tmp_path: Path,
    tokenizer_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(
        tmp_path,
        tokenizer_model,
        empty_language="ja",
    )
    assert plan.languages == ("ko", "ja")

    outcome = run_foundation_stage(config, plan, model, tokenizer, context)

    assert outcome.ran
    assert outcome.languages == ("ko",)
    run_root = foundation_run_directory(config)
    completion = run_root / FOUNDATION_COMPLETION_FILENAME
    marker = json.loads(completion.read_text(encoding="utf-8"))
    assert marker["languages"] == ["ko"]
    lineage = resolve_foundation_lineage(config, plan, context)
    assert lineage["languages"] == ["ko"]
    effective_plan = _foundation_plan_for_lineage(plan, lineage)
    pipeline = build_translation_pipeline_identity(
        effective_plan,
        foundation_lineage=lineage,
    )
    assert pipeline["foundation"]["languages"] == ["ko"]

    latest_payload = torch.load(
        run_root / "exports" / "latest" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    final_payload = torch.load(
        run_root / "exports" / "best" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert latest_payload["metadata"]["languages"] == ["ko"]
    assert final_payload["metadata"]["languages"] == ["ko"]

    import sion_translate.cli.train as train_module

    shutil.rmtree(run_root / "exports" / "best")
    original_export = train_module.export_final_model
    repaired_languages: list[tuple[str, ...]] = []

    def recording_export(*args, **kwargs):
        repaired_languages.append(tuple(kwargs["languages"]))
        return original_export(*args, **kwargs)

    monkeypatch.setattr(train_module, "export_final_model", recording_export)
    monkeypatch.setattr(
        train_module,
        "train",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("effective-language export repair must not retrain foundation")
        ),
    )
    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)

    reused = run_foundation_stage(config, plan, fresh, tokenizer, context)

    assert not reused.ran
    assert reused.languages == ("ko",)
    assert repaired_languages == [("ko",)]
    repaired_payload = torch.load(
        run_root / "exports" / "best" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert repaired_payload["metadata"]["languages"] == ["ko"]


def test_empty_prepared_language_has_zero_sampling_mass(tmp_path) -> None:
    manifest = {
        "language_sampling": {
            "counts": {"ko": 100, "ja": 0},
            "weights": {"ko": 1.0, "ja": 0.0},
        },
        "sources": [
            {"id": 0, "language": "ko", "name": "ko.txt"},
            {"id": 1, "language": "ja", "name": "ja.txt"},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = SimpleNamespace(dataset_root=tmp_path, source_names=["ko.txt", "ja.txt"])

    weights = _foundation_source_sampling_weights(dataset)

    assert weights == {0: 1.0, 1: 0.0}


def test_structured_reasoning_sampling_is_capped_at_the_manifest_row_share(tmp_path) -> None:
    manifest = {
        "language_sampling": {
            "counts": {"ko": 900, "ja": 1_000},
            "weights": {"ko": 0.5, "ja": 0.5},
        },
        "reasoning": {"sample_share": 0.05},
        "sources": [
            {"id": 0, "language": "ko", "name": "ko.txt", "task": "denoising"},
            {"id": 1, "language": "ja", "name": "ja.txt", "task": "denoising"},
            {
                "id": 2,
                "language": "ja",
                "name": "reasoning_math.jsonl",
                "task": "reasoning",
            },
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pair_source_ids = np.asarray([0] * 900 + [1] * 900 + [2] * 100, dtype=np.uint16)
    dataset = SimpleNamespace(
        dataset_root=tmp_path,
        source_names=["ko.txt", "ja.txt", "reasoning_math.jsonl"],
        pair_source_ids=pair_source_ids,
    )

    weights = _foundation_source_sampling_weights(dataset)
    masses = {
        source_id: float(np.count_nonzero(pair_source_ids == source_id)) * weight
        for source_id, weight in weights.items()
    }
    reasoning_share = masses[2] / sum(masses.values())

    assert reasoning_share == pytest.approx(0.05)
    assert masses[0] == pytest.approx(masses[1])


def test_foundation_completion_is_invalidated_by_new_prepared_inputs(tmp_path) -> None:
    dataset_dir = tmp_path / "foundation_dataset"
    dataset_dir.mkdir()
    manifest = dataset_dir / "manifest.json"
    tokenizer = tmp_path / "tokenizer.model"
    completion = tmp_path / FOUNDATION_COMPLETION_FILENAME
    manifest.write_text('{"objective":"denoising"}\n', encoding="utf-8")
    tokenizer.write_bytes(b"tokenizer-v1")
    completion.write_text(
        json.dumps(
            {
                "foundation_manifest_sha256": file_sha256(manifest),
                "tokenizer_sha256": file_sha256(tokenizer),
            }
        ),
        encoding="utf-8",
    )

    assert _foundation_completion_matches_inputs(
        completion,
        dataset_dir=dataset_dir,
        tokenizer_path=tokenizer,
    )
    manifest.write_text('{"objective":"denoising+reasoning"}\n', encoding="utf-8")
    assert not _foundation_completion_matches_inputs(
        completion,
        dataset_dir=dataset_dir,
        tokenizer_path=tokenizer,
    )


def test_a_second_run_reuses_the_weights_instead_of_retraining(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    """가장 비싼 단계를 반복하지 않는 것이 이 표시의 존재 이유다."""
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    run_foundation_stage(config, plan, model, tokenizer, context)
    trained = model.token_embedding.weight.detach().clone()
    completion = foundation_run_directory(config) / FOUNDATION_COMPLETION_FILENAME
    original_marker = completion.read_bytes()

    import sion_translate.cli.train as train_module

    def _fail(*args, **kwargs):
        raise AssertionError("완료된 foundation 단계를 다시 학습하려 했습니다")

    monkeypatch.setattr(train_module, "train", _fail)
    monkeypatch.setattr(train_module, "export_final_model", _fail)

    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    outcome = run_foundation_stage(config, plan, fresh, tokenizer, context)

    assert not outcome.ran
    assert "Reused" in outcome.reason
    assert outcome.best_checkpoint is not None
    assert torch.allclose(fresh.token_embedding.weight, trained)
    assert completion.read_bytes() == original_marker


def test_a_missing_final_export_is_rebuilt_without_retraining(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    first = run_foundation_stage(config, plan, model, tokenizer, context)
    run_root = foundation_run_directory(config)
    completion = run_root / FOUNDATION_COMPLETION_FILENAME
    original_marker = json.loads(completion.read_text(encoding="utf-8"))
    export_dir = run_root / "exports" / "best"
    shutil.rmtree(export_dir)

    import sion_translate.cli.train as train_module

    monkeypatch.setattr(
        train_module,
        "train",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing export must not retrain foundation")
        ),
    )
    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)

    second = run_foundation_stage(config, plan, fresh, tokenizer, context)

    assert not second.ran
    assert second.selected_step == first.selected_step
    assert (export_dir / "export_manifest.json").is_file()
    assert (export_dir / "model.pt").is_file()
    repaired_marker = json.loads(completion.read_text(encoding="utf-8"))
    assert {
        key: value for key, value in repaired_marker.items() if key != "export_manifest_sha256"
    } == {key: value for key, value in original_marker.items() if key != "export_manifest_sha256"}
    assert repaired_marker["export_manifest_sha256"] == file_sha256(
        export_dir / "export_manifest.json"
    )


def test_a_truncated_completion_marker_resumes_training_instead_of_rebinding(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    run_foundation_stage(config, plan, model, tokenizer, context)
    run_root = foundation_run_directory(config)
    completion = run_root / FOUNDATION_COMPLETION_FILENAME
    completion.write_text('{"schema":', encoding="utf-8")

    import sion_translate.cli.train as train_module

    monkeypatch.setattr(
        train_module,
        "export_final_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an untrusted marker must not publish an export")
        ),
    )

    def require_resume(*args, **_kwargs):
        foundation_config = args[3]
        assert foundation_config.training.resume_from is not None
        assert Path(foundation_config.training.resume_from).name == "latest"
        assert not completion.exists()
        assert (run_root / "checkpoints" / "best" / "checkpoint.pt").is_file()
        assert not (run_root / "exports" / "best").exists()
        raise RuntimeError("authenticated resume requested")

    monkeypatch.setattr(train_module, "train", require_resume)

    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)

    with pytest.raises(RuntimeError, match="authenticated resume requested"):
        run_foundation_stage(config, plan, fresh, tokenizer, context)

    quarantines = list(run_root.parent.glob(f"{run_root.name}.untrusted-derived-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / FOUNDATION_COMPLETION_FILENAME).read_text(
        encoding="utf-8"
    ) == '{"schema":'
    assert not (quarantines[0] / "checkpoints" / "best").exists()
    assert (quarantines[0] / "exports" / "best").is_dir()


def test_truncated_marker_cannot_launder_an_unbound_best_checkpoint(
    tmp_path,
    tokenizer_model,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    run_foundation_stage(config, plan, model, tokenizer, context)
    run_root = foundation_run_directory(config)
    latest = run_root / "checkpoints" / "latest"
    best = run_root / "checkpoints" / "best"
    completion = run_root / FOUNDATION_COMPLETION_FILENAME
    trusted_latest = torch.load(
        latest / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )["model"]["token_embedding.weight"].clone()
    _mutate_checkpoint_weights(best)
    untrusted_best = torch.load(
        best / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )["model"]["token_embedding.weight"].clone()
    completion.write_text('{"schema":', encoding="utf-8")

    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    outcome = run_foundation_stage(config, plan, fresh, tokenizer, context)

    rebound_best = torch.load(
        best / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )["model"]["token_embedding.weight"]
    assert outcome.ran
    torch.testing.assert_close(rebound_best, trusted_latest)
    torch.testing.assert_close(fresh.token_embedding.weight, trusted_latest)
    assert not torch.allclose(rebound_best, untrusted_best)
    marker = json.loads(completion.read_text(encoding="utf-8"))
    assert marker["checkpoint_artifact_sha256"] == file_sha256(best / "checkpoint.pt")
    quarantines = list(run_root.parent.glob(f"{run_root.name}.untrusted-derived-*"))
    assert len(quarantines) == 1
    quarantined_best = torch.load(
        quarantines[0] / "checkpoints" / "best" / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )["model"]["token_embedding.weight"]
    torch.testing.assert_close(quarantined_best, untrusted_best)


def test_truncated_marker_with_changed_objective_archives_before_fresh_training(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    run_foundation_stage(config, plan, model, tokenizer, context)
    run_root = foundation_run_directory(config)
    completion = run_root / FOUNDATION_COMPLETION_FILENAME
    completion.write_text('{"schema":', encoding="utf-8")
    config.foundation.learning_rate *= 0.5

    import sion_translate.cli.train as train_module

    monkeypatch.setattr(
        train_module,
        "export_final_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an incompatible resume checkpoint must never be exported")
        ),
    )

    def require_fresh_training(*args, **_kwargs):
        foundation_config = args[3]
        assert foundation_config.training.resume_from is None
        assert not run_root.exists()
        raise RuntimeError("fresh foundation training requested")

    monkeypatch.setattr(train_module, "train", require_fresh_training)

    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    with pytest.raises(RuntimeError, match="fresh foundation training requested"):
        run_foundation_stage(config, plan, fresh, tokenizer, context)

    assert len(list(run_root.parent.glob(f"{run_root.name}.stale-*"))) == 1


def test_a_foreign_valid_foundation_export_is_replaced_from_the_bound_checkpoint(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    first = run_foundation_stage(config, plan, model, tokenizer, context)
    expected = model.token_embedding.weight.detach().clone()
    completion = foundation_run_directory(config) / FOUNDATION_COMPLETION_FILENAME
    original_marker = json.loads(completion.read_text(encoding="utf-8"))

    import sion_translate.cli.train as train_module

    with torch.no_grad():
        model.token_embedding.weight.add_(1.0)
    train_module.export_final_model(
        model,
        build_foundation_config(config),
        context,
        Path(config.training.output_dir),
        stage="foundation",
        step=int(first.selected_step or 0),
        formats=("fp32",),
        release_name=config.foundation.release_name,
        translation_capable=False,
        languages=plan.languages,
    )

    original_export = train_module.export_final_model
    export_calls = 0

    def recording_export(*args, **kwargs):
        nonlocal export_calls
        export_calls += 1
        return original_export(*args, **kwargs)

    monkeypatch.setattr(train_module, "export_final_model", recording_export)
    monkeypatch.setattr(
        train_module,
        "train",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("foreign export must be repaired from checkpoint")
        ),
    )
    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)

    outcome = run_foundation_stage(config, plan, fresh, tokenizer, context)

    payload = torch.load(
        foundation_run_directory(config) / "exports" / "best" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert not outcome.ran
    assert export_calls == 1
    torch.testing.assert_close(fresh.token_embedding.weight, expected)
    torch.testing.assert_close(payload["model"]["token_embedding.weight"], expected)
    repaired_marker = json.loads(completion.read_text(encoding="utf-8"))
    assert {
        key: value for key, value in repaired_marker.items() if key != "export_manifest_sha256"
    } == {key: value for key, value in original_marker.items() if key != "export_manifest_sha256"}
    assert repaired_marker["export_manifest_sha256"] == file_sha256(
        foundation_run_directory(config) / "exports" / "best" / "export_manifest.json"
    )


def test_tampered_export_binding_forces_a_checkpoint_based_reexport(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    run_foundation_stage(config, plan, model, tokenizer, context)
    completion = foundation_run_directory(config) / FOUNDATION_COMPLETION_FILENAME
    marker = json.loads(completion.read_text(encoding="utf-8"))
    checkpoint_digest = marker["checkpoint_artifact_sha256"]
    marker["export_manifest_sha256"] = "0" * 64
    completion.write_text(json.dumps(marker), encoding="utf-8")

    import sion_translate.cli.train as train_module

    original_export = train_module.export_final_model
    export_calls = 0

    def recording_export(*args, **kwargs):
        nonlocal export_calls
        export_calls += 1
        return original_export(*args, **kwargs)

    monkeypatch.setattr(train_module, "export_final_model", recording_export)
    monkeypatch.setattr(
        train_module,
        "train",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("export binding repair must not retrain foundation")
        ),
    )

    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    outcome = run_foundation_stage(config, plan, fresh, tokenizer, context)

    repaired = json.loads(completion.read_text(encoding="utf-8"))
    assert not outcome.ran
    assert export_calls == 1
    assert repaired["checkpoint_artifact_sha256"] == checkpoint_digest
    assert repaired["export_manifest_sha256"] != "0" * 64


def test_tampered_checkpoint_binding_archives_instead_of_rebinding(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    run_foundation_stage(config, plan, model, tokenizer, context)
    run_root = foundation_run_directory(config)
    completion = run_root / FOUNDATION_COMPLETION_FILENAME
    marker = json.loads(completion.read_text(encoding="utf-8"))
    marker["checkpoint_artifact_sha256"] = "0" * 64
    completion.write_text(json.dumps(marker), encoding="utf-8")

    import sion_translate.cli.train as train_module

    monkeypatch.setattr(
        train_module,
        "export_final_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an unauthenticated checkpoint must never be exported")
        ),
    )
    monkeypatch.setattr(
        train_module,
        "train",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("fresh foundation training requested")
        ),
    )

    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    with pytest.raises(RuntimeError, match="fresh foundation training requested"):
        run_foundation_stage(config, plan, fresh, tokenizer, context)

    assert not run_root.exists()
    assert len(list(run_root.parent.glob(f"{run_root.name}.stale-*"))) == 1


def test_tampered_selected_step_archives_before_loading_or_exporting(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    run_foundation_stage(config, plan, model, tokenizer, context)
    run_root = foundation_run_directory(config)
    completion = run_root / FOUNDATION_COMPLETION_FILENAME
    marker = json.loads(completion.read_text(encoding="utf-8"))
    marker["selected_step"] = int(marker["selected_step"]) + 1
    completion.write_text(json.dumps(marker), encoding="utf-8")

    import sion_translate.cli.train as train_module

    monkeypatch.setattr(
        train_module,
        "export_final_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a step-mismatched checkpoint must never be exported")
        ),
    )
    monkeypatch.setattr(
        train_module,
        "train",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("fresh foundation training requested")
        ),
    )

    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    with pytest.raises(RuntimeError, match="fresh foundation training requested"):
        run_foundation_stage(config, plan, fresh, tokenizer, context)

    assert not run_root.exists()
    assert len(list(run_root.parent.glob(f"{run_root.name}.stale-*"))) == 1


def test_modified_checkpoint_contents_are_never_promoted_to_a_new_binding(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    run_foundation_stage(config, plan, model, tokenizer, context)
    run_root = foundation_run_directory(config)
    _mutate_checkpoint_weights(run_root / "checkpoints" / "best")

    import sion_translate.cli.train as train_module

    monkeypatch.setattr(
        train_module,
        "export_final_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("modified checkpoint contents must never be exported")
        ),
    )
    monkeypatch.setattr(
        train_module,
        "train",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("fresh foundation training requested")
        ),
    )

    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    with pytest.raises(RuntimeError, match="fresh foundation training requested"):
        run_foundation_stage(config, plan, fresh, tokenizer, context)

    assert not run_root.exists()
    assert len(list(run_root.parent.glob(f"{run_root.name}.stale-*"))) == 1


def test_marker_bound_previous_checkpoint_is_selected_without_rewriting(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    run_foundation_stage(config, plan, model, tokenizer, context)
    expected = model.token_embedding.weight.detach().clone()
    run_root = foundation_run_directory(config)
    completion = run_root / FOUNDATION_COMPLETION_FILENAME
    original_marker = completion.read_bytes()
    best = run_root / "checkpoints" / "best"
    previous = best.with_name(f".{best.name}.previous")
    shutil.copytree(best, previous)
    _mutate_checkpoint_weights(best)

    import sion_translate.cli.train as train_module

    def fail(*_args, **_kwargs):
        raise AssertionError("a marker-bound previous checkpoint must be reused directly")

    monkeypatch.setattr(train_module, "train", fail)
    monkeypatch.setattr(train_module, "export_final_model", fail)

    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    outcome = run_foundation_stage(config, plan, fresh, tokenizer, context)

    assert not outcome.ran
    assert outcome.best_checkpoint == str(previous)
    torch.testing.assert_close(fresh.token_embedding.weight, expected)
    assert completion.read_bytes() == original_marker


def test_foundation_resume_discovers_a_retained_local_previous_generation(
    tmp_path: Path,
) -> None:
    import sion_translate.cli.train as train_module

    config = AppConfig()
    config.training.output_dir = str(tmp_path / "foundation-run")
    latest = Path(config.training.output_dir) / "checkpoints" / "latest"
    previous = latest.with_name(f".{latest.name}.previous")
    previous.mkdir(parents=True)
    payload = previous / "checkpoint.pt"
    payload.write_bytes(b"retained local foundation generation")
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    resolution = train_module._inspect_foundation_resume(config, context)

    assert resolution == {
        "state": "available",
        "generation": "previous",
        "checkpoint_artifact_sha256": file_sha256(payload),
    }


def test_changed_foundation_objective_archives_the_completed_run_before_retraining(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    run_foundation_stage(config, plan, model, tokenizer, context)
    config.foundation.learning_rate *= 0.5

    import sion_translate.cli.train as train_module

    monkeypatch.setattr(
        train_module,
        "train",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("fresh foundation training requested")
        ),
    )
    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)

    with pytest.raises(RuntimeError, match="fresh foundation training requested"):
        run_foundation_stage(config, plan, fresh, tokenizer, context)

    run_root = foundation_run_directory(config)
    assert not run_root.exists()
    assert len(list(run_root.parent.glob(f"{run_root.name}.stale-*"))) == 1


def test_completion_marker_replace_failure_preserves_the_previous_generation(
    tmp_path,
    monkeypatch,
) -> None:
    import sion_translate.cli.train as train_module

    completion = tmp_path / FOUNDATION_COMPLETION_FILENAME
    original = b'{"generation":"known-good"}\n'
    completion.write_bytes(original)

    def reject_replace(_source, _destination) -> None:
        raise PermissionError("injected marker replace failure")

    monkeypatch.setattr(train_module.os, "replace", reject_replace)

    with pytest.raises(PermissionError, match="injected marker replace failure"):
        train_module._atomic_write_foundation_completion(completion, {"generation": "retry"})

    assert completion.read_bytes() == original
    assert not list(tmp_path.glob(f".{FOUNDATION_COMPLETION_FILENAME}.*.tmp"))


def test_rank_zero_marker_failure_is_propagated_before_any_peer_barrier(
    monkeypatch,
) -> None:
    import sion_translate.cli.train as train_module

    context = DistributedContext(0, 0, 2, torch.device("cpu"), True, "gloo")
    broadcasts: list[bool] = []

    def record_failure(value: bool, _context: DistributedContext) -> bool:
        broadcasts.append(value)
        return value

    monkeypatch.setattr(train_module, "broadcast_bool", record_failure)

    with pytest.raises(PermissionError, match="injected publication failure"):
        train_module._run_rank_zero_action(
            context,
            lambda: (_ for _ in ()).throw(PermissionError("injected publication failure")),
            description="publishing a test marker",
        )

    assert broadcasts == [True]


def test_peer_fails_from_rank_zero_marker_failure_without_running_the_action(
    monkeypatch,
) -> None:
    import sion_translate.cli.train as train_module

    context = DistributedContext(1, 1, 2, torch.device("cpu"), True, "gloo")
    monkeypatch.setattr(train_module, "broadcast_bool", lambda _value, _context: True)

    with pytest.raises(RuntimeError, match="rank 0 failed while publishing a test marker"):
        train_module._run_rank_zero_action(
            context,
            lambda: (_ for _ in ()).throw(AssertionError("peer action must not run")),
            description="publishing a test marker",
        )


def test_two_gloo_ranks_exit_together_when_rank_zero_marker_write_fails(
    tmp_path: Path,
) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("distributed marker failure test requires Gloo")

    script = tmp_path / "marker_failure_worker.py"
    results = tmp_path / "results"
    results.mkdir()
    script.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path

            from sion_translate.cli.train import _run_rank_zero_action
            from sion_translate.training.distributed import (
                cleanup_distributed,
                initialize_distributed,
            )

            result_dir = Path(sys.argv[1])
            context = initialize_distributed()
            try:
                try:
                    _run_rank_zero_action(
                        context,
                        lambda: (_ for _ in ()).throw(
                            PermissionError("injected rank-zero marker failure")
                        ),
                        description="publishing a test marker",
                    )
                except BaseException as error:
                    (result_dir / f"rank-{context.rank}.txt").write_text(
                        f"{type(error).__name__}:{error}",
                        encoding="utf-8",
                    )
                else:
                    raise SystemExit("coordinated action unexpectedly succeeded")
            finally:
                cleanup_distributed(context)
            """
        ),
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONUTF8": "1",
        "USE_LIBUV": "0",
    }
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        rendezvous_port = int(port_probe.getsockname()[1])

    processes: list[subprocess.Popen[str]] = []
    for rank in range(2):
        worker_environment = {
            **environment,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(rendezvous_port),
            "WORLD_SIZE": "2",
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
        }
        processes.append(
            subprocess.Popen(
                [sys.executable, str(script), str(results)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=worker_environment,
            )
        )
    outputs: list[tuple[str, str]] = []
    try:
        for process in processes:
            outputs.append(process.communicate(timeout=60))
    except subprocess.TimeoutExpired:
        for process in processes:
            process.kill()
        for process in processes:
            process.communicate()
        raise

    for process, (stdout, stderr) in zip(processes, outputs, strict=True):
        assert process.returncode == 0, stderr or stdout
    assert (
        (results / "rank-0.txt")
        .read_text(encoding="utf-8")
        .startswith("PermissionError:injected rank-zero marker failure")
    )
    assert (
        (results / "rank-1.txt")
        .read_text(encoding="utf-8")
        .startswith("RuntimeError:rank 0 failed while publishing a test marker")
    )


def test_a_disabled_plan_does_nothing_and_says_why(tmp_path, tokenizer_model) -> None:
    config, _, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    config.foundation.enabled = False
    plan = plan_foundation_stage(config)

    outcome = run_foundation_stage(config, plan, model, tokenizer, context)

    assert not outcome.ran
    assert "foundation.enabled=false" in outcome.reason
    assert outcome.best_checkpoint is None
    assert not (foundation_run_directory(config) / FOUNDATION_COMPLETION_FILENAME).exists()


def test_a_missing_corpus_never_trains_or_exports_a_foundation_model(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, _, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    config.foundation.corpus_dir = str(tmp_path / "missing-corpus")
    plan = plan_foundation_stage(config)
    run_root = foundation_run_directory(config)

    import sion_translate.cli.train as train_module

    def _fail(*_args, **_kwargs):
        raise AssertionError("missing corpus must not train or export sion")

    monkeypatch.setattr(train_module, "train", _fail)
    monkeypatch.setattr(train_module, "export_final_model", _fail)

    outcome = run_foundation_stage(config, plan, model, tokenizer, context)

    assert not outcome.ran
    assert outcome.best_checkpoint is None
    assert not run_root.exists()


def test_the_stage_publishes_under_the_foundation_release_name(tmp_path, tokenizer_model) -> None:
    """foundation 산출물은 번역 모델이 아니라 그 파운데이션이다."""
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    # The translation graph also contains a source-only variety, but the
    # foundation plan deliberately targets only ko/ja.
    config.data.language_pairs = [["kj", "ko"], ["kj", "ja"], ["ko", "ja"]]
    config.data.source_only_languages = ["kj"]
    config.foundation.final_export_formats = ["fp32", "transformers"]
    run_foundation_stage(config, plan, model, tokenizer, context)

    export = foundation_run_directory(config) / "exports" / "best"
    assert export.is_dir()
    # `best` is replaced by the final multi-format export. `latest` is the
    # training-loop export that would remain if the run were interrupted after
    # a save, so it must carry the same non-translation contract too.
    latest = foundation_run_directory(config) / "exports" / "latest" / "model.pt"
    latest_payload = torch.load(latest, map_location="cpu", weights_only=True)
    latest_metadata = latest_payload["metadata"]
    assert latest_metadata["release_name"] == "sion"
    assert latest_metadata["translation_capable"] is False
    assert latest_metadata["languages"] == ["ko", "ja"]
    assert "translation_directions" not in latest_metadata
    assert "language_pair" not in latest_metadata
    assert "language_pairs" not in latest_metadata

    payload = torch.load(export / "model.pt", map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    assert metadata["release_name"] == "sion"
    assert metadata["translation_capable"] is False
    assert metadata["languages"] == ["ko", "ja"]
    # 번역 방향을 적지 않는다. 번역할 수 없는 가중치이기 때문이다.
    assert "translation_directions" not in metadata
    assert "language_pair" not in metadata
    assert "language_pairs" not in metadata

    transformers_config = json.loads(
        (export / "transformers" / "config.json").read_text(encoding="utf-8")
    )
    assert transformers_config["languages"] == ["ko", "ja"]
    assert transformers_config["language_pairs"] == []
    assert transformers_config["translation_directions"] == []
    assert transformers_config["translation_capable"] is False

    tokenizer_config = json.loads(
        (export / "transformers" / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert tokenizer_config["translation_capable"] is False

    manifest = json.loads((export / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formats"]["transformers"]["languages"] == metadata["languages"]
    assert manifest["formats"]["transformers"]["translation_capable"] is False

    from transformers import AutoConfig, AutoTokenizer

    hf_config = AutoConfig.from_pretrained(export / "transformers", trust_remote_code=True)
    hf_tokenizer = AutoTokenizer.from_pretrained(
        export / "transformers",
        trust_remote_code=True,
    )
    assert hf_config.translation_capable is False
    assert hf_tokenizer.translation_capable is False
    with pytest.raises(ValueError, match="foundation model.*not translation-capable"):
        hf_tokenizer._build_translation_inputs(
            "번역을 시도하면 안 됩니다.",
            return_tensors="pt",
            src_lang="ko",
            tgt_lang="ja",
        )


def test_a_foundation_export_is_refused_by_the_translator(tmp_path, tokenizer_model) -> None:
    """A foundation artifact must not accept translation direction tags."""
    from sion_translate.inference import Translator

    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    config.foundation.final_export_formats = ["fp32"]
    run_foundation_stage(config, plan, model, tokenizer, context)
    export = foundation_run_directory(config) / "exports" / "best" / "model.pt"

    with pytest.raises(ValueError, match="not a translation model"):
        Translator(str(export), str(tokenizer_model))
