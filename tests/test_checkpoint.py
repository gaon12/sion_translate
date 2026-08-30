from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from functools import partial
import hashlib
import json
import multiprocessing
import random
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sion_translate.training.checkpoint as checkpoint_module
from sion_translate.config import AppConfig, DataConfig, ExperimentalConfig, ModelConfig
from sion_translate.model import SionForConditionalGeneration
from sion_translate.training.checkpoint import (
    build_checkpoint_identity,
    initialize_model_from_checkpoint,
    CHECKPOINT_SCHEMA,
    load_checkpoint,
    preflight_checkpoint_identity,
    resolve_checkpoint_source,
    save_checkpoint,
)
from sion_translate.training.distributed import DistributedContext
from sion_translate.training.ema import EMAWeights
from sion_translate.training.trainer import (
    build_training_checkpoint_identity,
    cosine_scheduler,
)


class FakeScaler:
    def __init__(self, scale: float):
        self.scale = scale

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = state["scale"]


def _components() -> tuple[
    SionForConditionalGeneration,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    DistributedContext,
]:
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        max_seq_len=16,
        gradient_checkpointing=False,
        experimental=ExperimentalConfig(
            bats_enabled=False,
            core_enabled=False,
            tetm_enabled=False,
            morphoscript_enabled=False,
        ),
    )
    model = SionForConditionalGeneration(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = cosine_scheduler(optimizer, warmup_steps=1, max_steps=10, min_ratio=0.1)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    return model, optimizer, scheduler, context


def _distributed_save_heartbeat_worker(
    rank: int,
    world_size: int,
    rendezvous_uri: str,
    checkpoint_text: str,
    result_directory_text: str,
    mode: str,
) -> None:
    checkpoint = Path(checkpoint_text)
    result_directory = Path(result_directory_text)
    result = ""
    try:
        torch.distributed.init_process_group(
            "gloo",
            init_method=rendezvous_uri,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=4),
        )
        checkpoint_module._CHECKPOINT_IO_HEARTBEAT_SECONDS = 0.1
        if rank == 0 and mode == "prepare-failure":
            real_remove_path = checkpoint_module._remove_path

            def fail_staging_cleanup(path: Path) -> None:
                time.sleep(5.0)
                if path.name.endswith(".staging"):
                    raise OSError("injected staging cleanup failure")
                real_remove_path(path)

            checkpoint_module._remove_path = fail_staging_cleanup
        if rank == 0 and mode in {"publish-success", "publish-failure"}:
            real_write_completion = checkpoint_module._write_dcp_completion

            def delayed_completion(*args, **kwargs):
                time.sleep(5.0)
                if mode == "publish-failure":
                    raise OSError("injected completion publication failure")
                return real_write_completion(*args, **kwargs)

            checkpoint_module._write_dcp_completion = delayed_completion

        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(4, 4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        context = DistributedContext(
            rank,
            rank,
            world_size,
            torch.device("cpu"),
            True,
            "gloo",
        )
        try:
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                3,
                context,
            )
        except BaseException as error:
            result = f"{type(error).__name__}|{error}"
        else:
            result = "ok"
    except BaseException as error:
        result = f"worker-{type(error).__name__}|{error}"
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        result_directory.mkdir(parents=True, exist_ok=True)
        (result_directory / f"rank-{rank}.txt").write_text(result, encoding="utf-8")


def test_local_checkpoint_round_trip(tmp_path: Path) -> None:
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "checkpoint"
    expected = model.token_embedding.weight.detach().clone()
    scaler = FakeScaler(1024.0)
    progress = {
        "best_validation_loss": 1.25,
        "early_stopping_bad_evals": 2,
        "epoch": 3,
    }
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        7,
        context,
        scaler=scaler,
        training_state=progress,
    )
    assert resolve_checkpoint_source(checkpoint, context) == checkpoint
    with torch.no_grad():
        model.token_embedding.weight.add_(1.0)
    scaler.scale = 1.0
    restored_progress: dict = {}
    step = load_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        context,
        scaler=scaler,
        training_state=restored_progress,
    )
    assert step == 7
    assert scaler.scale == 1024.0
    assert restored_progress == progress
    torch.testing.assert_close(model.token_embedding.weight, expected)

    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=True)
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert "rng_state" in payload


@pytest.mark.parametrize("distributed", [False, True])
def test_checkpoint_source_rejects_mixed_local_and_distributed_formats(
    tmp_path: Path,
    distributed: bool,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.pt").write_bytes(b"local")
    (checkpoint / ".metadata").write_bytes(b"distributed")
    context = DistributedContext(0, 0, 1, torch.device("cpu"), distributed)

    with pytest.raises(ValueError, match="mixes local checkpoint.pt"):
        resolve_checkpoint_source(checkpoint, context)


@pytest.mark.parametrize("local_generation", ["current", "previous"])
@pytest.mark.parametrize("distributed", [False, True])
def test_checkpoint_pair_rejects_mixed_formats_through_every_load_entrypoint(
    tmp_path: Path,
    local_generation: str,
    distributed: bool,
) -> None:
    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    current.mkdir()
    previous.mkdir()
    local = current if local_generation == "current" else previous
    dcp = previous if local_generation == "current" else current
    (local / "checkpoint.pt").write_bytes(b"local checkpoint")
    (dcp / ".metadata").write_bytes(b"distributed checkpoint")
    context = DistributedContext(0, 0, 1, torch.device("cpu"), distributed, "gloo")
    model, optimizer, scheduler, _ = _components()

    for requested in (current, previous):
        operations = (
            partial(resolve_checkpoint_source, requested, context),
            partial(preflight_checkpoint_identity, requested, context, None),
            partial(initialize_model_from_checkpoint, requested, model, context),
            partial(load_checkpoint, requested, model, optimizer, scheduler, context),
        )
        for operation in operations:
            with pytest.raises(ValueError, match="mixes local checkpoint.pt"):
                operation()


def test_local_checkpoint_replace_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, optimizer, scheduler, 1, context)
    original = (checkpoint / "checkpoint.pt").read_bytes()

    def fail_after_partial_write(payload: object, handle: object) -> None:
        del payload
        handle.write(b"partial")
        handle.flush()
        raise RuntimeError("intentional save failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="intentional save failure"):
        save_checkpoint(checkpoint, model, optimizer, scheduler, 2, context)

    assert (checkpoint / "checkpoint.pt").read_bytes() == original
    assert not list(checkpoint.glob(".checkpoint.pt.*.tmp"))


def test_checkpoint_identity_mismatch_fails_before_model_mutation(tmp_path: Path) -> None:
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        1,
        context,
        identity={"schema": "test", "tokenizer": {"sha256": "a" * 64}},
    )
    with torch.no_grad():
        model.token_embedding.weight.add_(1.0)
    before_load = model.token_embedding.weight.detach().clone()

    with pytest.raises(ValueError, match="identity does not match"):
        load_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            context,
            expected_identity={"schema": "test", "tokenizer": {"sha256": "b" * 64}},
        )

    torch.testing.assert_close(model.token_embedding.weight, before_load)


def _identity_fixture(root: Path, *, approximate_split: bool = True) -> dict:
    tokenizer = root / "artifacts" / "tokenizer" / "sion.model"
    dataset = root / "artifacts" / "dataset"
    tokenizer.parent.mkdir(parents=True)
    dataset.mkdir(parents=True)
    tokenizer.write_bytes(b"portable tokenizer")
    (dataset / "manifest.json").write_text('{"format":"test"}\n', encoding="utf-8")
    (dataset / "raw_fingerprint.json").write_text(
        '{"schema":"test","sha256":"abc"}\n',
        encoding="utf-8",
    )
    data_config = DataConfig(
        raw_dir=str(root / "data"),
        tokenizer_model=str(tokenizer),
        tokenizer_features=str(tokenizer.parent / "token_features.npz"),
        dataset_dir=str(dataset),
        approximate_split=approximate_split,
    )
    return checkpoint_module.build_checkpoint_identity(
        model_config=ModelConfig(vocab_size=64),
        tokenizer_path=tokenizer,
        token_features_path=None,
        dataset_dir=dataset,
        data_config=data_config,
    )


def test_checkpoint_identity_ignores_runtime_storage_locations(tmp_path: Path) -> None:
    disk_identity = _identity_fixture(tmp_path / "disk")
    ram_identity = _identity_fixture(tmp_path / "dev-shm")

    assert disk_identity == ram_identity


def test_legacy_path_bearing_identity_normalizes_during_resume(tmp_path: Path) -> None:
    expected = _identity_fixture(tmp_path / "current")
    legacy = deepcopy(expected)
    legacy_config = legacy["data"]["config"]
    legacy_config.update(
        {
            "raw_dir": "/dev/shm/sion/data",
            "tokenizer_model": "/dev/shm/sion/artifacts/tokenizer/sion.model",
            "tokenizer_features": "/dev/shm/sion/artifacts/tokenizer/token_features.npz",
            "dataset_dir": "/dev/shm/sion/artifacts/dataset",
        }
    )
    legacy["data"]["config_sha256"] = "legacy-path-dependent-hash"

    checkpoint_module._validate_identity({"identity": legacy}, expected)


def test_legacy_identity_accepts_only_disabled_candidate_refinement_defaults(
    tmp_path: Path,
) -> None:
    expected = _identity_fixture(tmp_path / "disabled")
    legacy = deepcopy(expected)
    legacy_experimental = legacy["model"]["config"]["experimental"]
    for name in tuple(legacy_experimental):
        if name.startswith("candidate_refinement_"):
            legacy_experimental.pop(name)
    legacy["model"]["config_sha256"] = hashlib.sha256(
        checkpoint_module._canonical_json(legacy["model"]["config"]).encode("utf-8")
    ).hexdigest()

    checkpoint_module._validate_identity({"identity": legacy}, expected)

    enabled = deepcopy(expected)
    enabled["model"]["config"]["experimental"]["candidate_refinement_enabled"] = True
    enabled["model"]["config_sha256"] = "enabled-candidate-hash"
    with pytest.raises(ValueError, match="identity does not match"):
        checkpoint_module._validate_identity({"identity": legacy}, enabled)


def test_checkpoint_identity_still_rejects_semantic_data_changes(tmp_path: Path) -> None:
    expected = _identity_fixture(tmp_path / "expected", approximate_split=True)
    incompatible = _identity_fixture(tmp_path / "incompatible", approximate_split=False)

    with pytest.raises(ValueError, match="identity does not match"):
        checkpoint_module._validate_identity({"identity": incompatible}, expected)


def test_dcp_preflight_round_trips_the_full_training_identity(tmp_path: Path) -> None:
    import torch.distributed.checkpoint as dcp

    tokenizer = tmp_path / "artifacts" / "tokenizer" / "sion.model"
    dataset = tmp_path / "artifacts" / "dataset"
    tokenizer.parent.mkdir(parents=True)
    dataset.mkdir(parents=True)
    tokenizer.write_bytes(b"production identity tokenizer")
    (dataset / "manifest.json").write_text('{"format":"test"}\n', encoding="utf-8")
    (dataset / "raw_fingerprint.json").write_text(
        '{"schema":"test","sha256":"abc"}\n',
        encoding="utf-8",
    )
    config = AppConfig(
        model=ModelConfig(vocab_size=64),
        data=DataConfig(
            tokenizer_model=str(tokenizer),
            tokenizer_features=str(tokenizer.parent / "token_features.npz"),
            dataset_dir=str(dataset),
            language_pairs=[["kj", "ko"], ["kj", "ja"]],
            source_only_languages=["kj"],
        ),
    )
    context = DistributedContext(0, 0, 2, torch.device("cpu"), True, "gloo")
    identity = build_training_checkpoint_identity(
        config,
        batch_sampler=SimpleNamespace(
            seed=17,
            batch_size=3,
            drop_last=True,
            bucket_size=19,
        ),
        context=context,
        stage_name="pretrain/SFT",
        include_posttraining=False,
        pipeline_identity={
            "schema": "sion-translation-pipeline-v1",
            "branch": "translation-only",
        },
    )
    checkpoint = tmp_path / "dcp-full-identity"
    dcp.save({"identity": identity, "step": 13}, checkpoint_id=checkpoint)

    assert checkpoint_module._preflight_dcp_identity(checkpoint, identity) == 13
    checkpoint_module._preflight_dcp_stage_transfer(checkpoint, identity)
    metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    probe = checkpoint_module._dcp_identity_probe(metadata)
    dcp.load(probe, checkpoint_id=checkpoint)
    checkpoint_module._restore_expected_empty_mappings(probe["identity"], identity)

    assert probe == {"identity": identity}


def _write_test_dcp_generation(
    checkpoint: Path,
    state: dict[str, object],
    *,
    marker_step: int,
) -> None:
    import torch.distributed.checkpoint as dcp

    dcp.save(state, checkpoint_id=checkpoint)
    _seal_test_dcp(checkpoint, step=marker_step)


def _seal_test_dcp(checkpoint: Path, *, step: int) -> None:
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "rng_state": checkpoint_module._capture_rng_state(),
        },
        checkpoint / "rng-rank-00000.pt",
    )
    checkpoint_module._write_dcp_completion(
        checkpoint,
        step=step,
        world_size=1,
    )


def _clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _assert_model_state_unchanged(
    model: torch.nn.Module,
    expected: dict[str, torch.Tensor],
) -> None:
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name])


def test_dcp_resume_rejects_a_non_integer_step_before_model_mutation(tmp_path: Path) -> None:
    from torch.distributed.checkpoint.state_dict import get_state_dict

    source, source_optimizer, source_scheduler, _ = _components()
    target, target_optimizer, target_scheduler, _ = _components()
    with torch.no_grad():
        source.token_embedding.weight.fill_(9.0)
        target.token_embedding.weight.fill_(1.0)
    model_state, optimizer_state = get_state_dict(source, source_optimizer)
    checkpoint = tmp_path / "bad-resume-step"
    _write_test_dcp_generation(
        checkpoint,
        {
            "schema": CHECKPOINT_SCHEMA,
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": source_scheduler.state_dict(),
            "step": "not-an-integer",
            "training_state": {},
        },
        marker_step=3,
    )
    before = _clone_model_state(target)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")

    with pytest.raises(ValueError, match="step must be an integer"):
        load_checkpoint(
            checkpoint,
            target,
            target_optimizer,
            target_scheduler,
            context,
            expected_identity=None,
        )

    _assert_model_state_unchanged(target, before)


@pytest.mark.parametrize(
    "release_fields",
    [
        {},
        {
            "best_candidate_refinement_guard_schema": ("sion-candidate-refinement-release-v3"),
            "best_candidate_refinement_deployed_family": "ema",
            "best_candidate_refinement_direction_fingerprint": "b" * 64,
            "best_candidate_refinement_direction_count": 2,
            "best_candidate_refinement_release_guard_passed": True,
            "best_candidate_refinement_worst_direction_nll_gain": 0.0125,
            "best_candidate_refinement_min_worst_direction_nll_gain": 1e-5,
            "best_candidate_refinement_validation_cohort_fingerprint": "c" * 64,
            "best_candidate_refinement_deployment_state_sha256": "d" * 64,
            "candidate_refinement_sft_baseline_loss": 0.75,
            "candidate_refinement_sft_baseline_selection_metric": (
                "validation_ema_macro_direction_nll"
            ),
            "candidate_refinement_sft_baseline_validation_cohort_fingerprint": "c" * 64,
        },
    ],
    ids=("legacy-without-release-attestation", "versioned-release-attestation"),
)
def test_dcp_resume_restores_optional_best_generation_binding_fields(
    tmp_path: Path,
    release_fields: dict[str, object],
) -> None:
    from torch.distributed.checkpoint.state_dict import get_state_dict

    source, source_optimizer, source_scheduler, _ = _components()
    target, target_optimizer, target_scheduler, _ = _components()
    model_state, optimizer_state = get_state_dict(source, source_optimizer)
    checkpoint = tmp_path / "resume-best-binding"
    expected_progress = {
        "best_validation_loss": 1.25,
        "best_step": 3,
        "early_stopping_bad_evals": 2,
        "epoch": 1,
        "batch_in_epoch": 4,
        "configured_selection_metric": "macro_direction_nll",
        "best_selection_metric": "validation_macro_direction_nll",
        "best_checkpoint_artifact_sha256": "a" * 64,
        **release_fields,
    }
    _write_test_dcp_generation(
        checkpoint,
        {
            "schema": CHECKPOINT_SCHEMA,
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": source_scheduler.state_dict(),
            "step": 3,
            "training_state": expected_progress,
        },
        marker_step=3,
    )
    restored_progress = {
        "best_validation_loss": float("inf"),
        "best_step": -1,
        "early_stopping_bad_evals": 0,
        "epoch": 0,
        "batch_in_epoch": 0,
    }
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")

    step = load_checkpoint(
        checkpoint,
        target,
        target_optimizer,
        target_scheduler,
        context,
        training_state=restored_progress,
        expected_identity=None,
    )

    assert step == 3
    assert restored_progress == expected_progress
    assert (
        checkpoint_module.inspect_checkpoint_training_state(checkpoint, context)
        == expected_progress
    )


def test_dcp_stage_transfer_rejects_a_non_integer_step_before_model_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed.checkpoint as dcp

    source, _, _, _ = _components()
    target, _, _, _ = _components()
    with torch.no_grad():
        source.token_embedding.weight.fill_(8.0)
        target.token_embedding.weight.fill_(2.0)
    identity = {
        "schema": "test",
        "tokenizer": {"sha256": "a" * 64},
        "model": {"config_sha256": "b" * 64},
    }
    checkpoint = tmp_path / "bad-stage-transfer-step"
    _write_test_dcp_generation(
        checkpoint,
        {
            "model": source.state_dict(),
            "step": "not-an-integer",
            "identity": identity,
        },
        marker_step=3,
    )
    before = _clone_model_state(target)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    real_load = dcp.load
    preflight_no_dist: list[bool] = []

    def record_preflight_load(*args, **kwargs):
        preflight_no_dist.append(bool(kwargs.get("no_dist")))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(dcp, "load", record_preflight_load)

    with pytest.raises(ValueError, match="stage-transfer checkpoint step must be an integer"):
        initialize_model_from_checkpoint(
            checkpoint,
            target,
            context,
            expected_identity=identity,
        )

    assert preflight_no_dist == [True]
    _assert_model_state_unchanged(target, before)


def test_dcp_resume_normalizes_a_second_load_checkpoint_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    source, source_optimizer, source_scheduler, _ = _components()
    target, target_optimizer, target_scheduler, _ = _components()
    identity = {"schema": "test", "tokenizer": {"sha256": "a" * 64}}
    model_state, optimizer_state = get_state_dict(source, source_optimizer)
    checkpoint = tmp_path / "second-load-failure"
    _write_test_dcp_generation(
        checkpoint,
        {
            "schema": CHECKPOINT_SCHEMA,
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": source_scheduler.state_dict(),
            "step": 5,
            "training_state": {},
            "identity": identity,
        },
        marker_step=5,
    )
    before = _clone_model_state(target)
    real_load = dcp.load
    load_calls = 0

    def fail_second_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        if load_calls == 1:
            assert kwargs.get("no_dist") is True
            return real_load(*args, **kwargs)
        assert kwargs.get("no_dist", False) is False
        raise dcp.CheckpointException("injected second-load failure", {})

    monkeypatch.setattr(dcp, "load", fail_second_load)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")

    with pytest.raises(ValueError, match="distributed checkpoint payload could not be loaded"):
        load_checkpoint(
            checkpoint,
            target,
            target_optimizer,
            target_scheduler,
            context,
            expected_identity=identity,
        )

    assert load_calls == 2
    _assert_model_state_unchanged(target, before)


def test_dcp_resume_normalizes_a_second_metadata_checkpoint_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    source, source_optimizer, source_scheduler, _ = _components()
    target, target_optimizer, target_scheduler, _ = _components()
    identity = {"schema": "test", "tokenizer": {"sha256": "a" * 64}}
    source_ema = EMAWeights(source, decay=0.9)
    target_ema = EMAWeights(target, decay=0.9)
    model_state, optimizer_state = get_state_dict(source, source_optimizer)
    checkpoint = tmp_path / "second-metadata-failure"
    _write_test_dcp_generation(
        checkpoint,
        {
            "schema": CHECKPOINT_SCHEMA,
            "model": model_state,
            "optimizer": optimizer_state,
            "scheduler": source_scheduler.state_dict(),
            "step": 5,
            "training_state": {},
            "identity": identity,
            "ema": source_ema.state_dict(),
        },
        marker_step=5,
    )
    before = _clone_model_state(target)
    real_read_metadata = dcp.FileSystemReader.read_metadata
    metadata_calls = 0

    def fail_second_metadata_read(reader):
        nonlocal metadata_calls
        metadata_calls += 1
        if metadata_calls <= 2:
            return real_read_metadata(reader)
        raise dcp.CheckpointException("injected second-metadata failure", {})

    monkeypatch.setattr(dcp.FileSystemReader, "read_metadata", fail_second_metadata_read)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")

    with pytest.raises(ValueError, match="distributed checkpoint metadata could not be loaded"):
        load_checkpoint(
            checkpoint,
            target,
            target_optimizer,
            target_scheduler,
            context,
            ema=target_ema,
            expected_identity=identity,
        )

    assert metadata_calls == 3
    _assert_model_state_unchanged(target, before)


def test_dcp_identity_probe_preserves_integer_list_paths(tmp_path: Path) -> None:
    import torch.distributed.checkpoint as dcp

    state = {
        "identity": {
            "schema": "test",
            "languages": [
                {"code": "ko", "weight": 1.0},
                {"code": "ja", "weight": 0.5},
            ],
        }
    }
    checkpoint = tmp_path / "dcp-list-identity"
    dcp.save(state, checkpoint_id=checkpoint)
    metadata = dcp.FileSystemReader(checkpoint).read_metadata()

    probe = checkpoint_module._dcp_identity_probe(metadata)
    assert isinstance(probe["identity"]["languages"], list)
    dcp.load(probe, checkpoint_id=checkpoint)

    assert probe == state


def test_checkpoint_identity_rejects_a_different_pipeline_branch(tmp_path: Path) -> None:
    expected = _identity_fixture(tmp_path / "expected")
    expected["pipeline"] = {
        "schema": "sion-translation-pipeline-v1",
        "branch": "translation-only",
    }
    incompatible = deepcopy(expected)
    incompatible["pipeline"]["branch"] = "foundation-then-translation"

    with pytest.raises(ValueError, match=r"identity\.pipeline\.branch"):
        checkpoint_module._validate_identity({"identity": incompatible}, expected)


def test_checkpoint_identity_upgrades_exact_legacy_translation_only_pipeline() -> None:
    legacy = {
        "schema": "test",
        "pipeline": {
            "schema": "sion-translation-pipeline-v1",
            "branch": "translation-only",
        },
    }
    expected = {
        "schema": "test",
        "pipeline": {
            "schema": "sion-translation-pipeline-v2",
            "branch": "translation-only",
        },
    }

    checkpoint_module._validate_identity({"identity": legacy}, expected)


def test_checkpoint_identity_does_not_upgrade_nonexact_legacy_translation_pipeline() -> None:
    legacy = {
        "schema": "test",
        "pipeline": {
            "schema": "sion-translation-pipeline-v1",
            "branch": "translation-only",
            "unverifiable_lineage": "present",
        },
    }
    expected = {
        "schema": "test",
        "pipeline": {
            "schema": "sion-translation-pipeline-v2",
            "branch": "translation-only",
        },
    }

    with pytest.raises(ValueError, match=r"identity\.pipeline"):
        checkpoint_module._validate_identity({"identity": legacy}, expected)


def test_checkpoint_identity_keeps_legacy_foundation_pipeline_unverifiable() -> None:
    legacy = {
        "schema": "test",
        "pipeline": {
            "schema": "sion-translation-pipeline-v1",
            "branch": "foundation-then-translation",
        },
    }
    expected = {
        "schema": "test",
        "pipeline": {
            "schema": "sion-translation-pipeline-v2",
            "branch": "foundation-then-translation",
            "foundation": {
                "schema": "sion-foundation-lineage-v1",
                "checkpoint_artifact_sha256": "a" * 64,
            },
        },
    }

    with pytest.raises(ValueError, match=r"identity\.pipeline"):
        checkpoint_module._validate_identity({"identity": legacy}, expected)


@pytest.mark.parametrize("state", [{}, {"identity": {"schema": "legacy"}}])
def test_required_pipeline_identity_rejects_unverifiable_legacy_resume(state: dict) -> None:
    expected = {
        "schema": "test",
        "pipeline": {
            "schema": "sion-translation-pipeline-v1",
            "branch": "translation-only",
        },
    }

    with pytest.raises(ValueError, match="no recorded pipeline identity"):
        checkpoint_module._validate_identity(state, expected)


@pytest.mark.parametrize(
    ("compiled_at_save", "compiled_at_load"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_checkpoint_model_and_ema_keys_are_compile_independent(
    tmp_path: Path,
    compiled_at_save: bool,
    compiled_at_load: bool,
) -> None:
    source, source_optimizer, source_scheduler, context = _components()
    source_model = torch.compile(source, backend="eager") if compiled_at_save else source
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-3)
    source_scheduler = cosine_scheduler(
        source_optimizer,
        warmup_steps=0,
        max_steps=2,
        min_ratio=0.1,
    )
    source_ema = EMAWeights(source_model, 0.9)
    with torch.no_grad():
        for parameter in source_model.parameters():
            parameter.add_(0.25)
    source_ema.update(source_model)
    checkpoint = tmp_path / f"{compiled_at_save}-{compiled_at_load}"
    save_checkpoint(
        checkpoint,
        source_model,
        source_optimizer,
        source_scheduler,
        1,
        context,
        ema=source_ema,
    )

    target, target_optimizer, target_scheduler, _ = _components()
    target_model = torch.compile(target, backend="eager") if compiled_at_load else target
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=1e-3)
    target_scheduler = cosine_scheduler(
        target_optimizer,
        warmup_steps=0,
        max_steps=2,
        min_ratio=0.1,
    )
    target_ema = EMAWeights(target_model, 0.9)
    load_checkpoint(
        checkpoint,
        target_model,
        target_optimizer,
        target_scheduler,
        context,
        ema=target_ema,
    )

    source_state = checkpoint_module._unwrap_compiled_model(source_model).state_dict()
    target_state = checkpoint_module._unwrap_compiled_model(target_model).state_dict()
    assert source_state.keys() == target_state.keys()
    assert all(not name.startswith("_orig_mod.") for name in source_state)
    for name in source_state:
        torch.testing.assert_close(target_state[name], source_state[name])
    assert source_ema.shadow.keys() == target_ema.shadow.keys()
    for name in source_ema.shadow:
        torch.testing.assert_close(target_ema.shadow[name], source_ema.shadow[name])


@pytest.mark.parametrize("tamper", ["missing", "unexpected", "shape", "dtype"])
def test_ema_state_validation_is_exact_and_non_mutating(tamper: str) -> None:
    model, _, _, _ = _components()
    ema = EMAWeights(model, 0.9)
    state = {name: tensor.clone() for name, tensor in ema.state_dict().items()}
    first_name = next(iter(state))
    if tamper == "missing":
        state.pop(first_name)
    elif tamper == "unexpected":
        state["injected.unexpected"] = torch.tensor([1.0])
    elif tamper == "shape":
        state[first_name] = state[first_name].reshape(-1)[:1]
    else:
        state[first_name] = state[first_name].double()
    before = {name: tensor.clone() for name, tensor in ema.shadow.items()}

    with pytest.raises(ValueError, match="EMA checkpoint"):
        ema.load_state_dict(state)

    for name, tensor in ema.shadow.items():
        torch.testing.assert_close(tensor, before[name])


@pytest.mark.parametrize("tamper", ["missing", "partial"])
def test_resume_requires_a_complete_ema_before_mutating_training_state(
    tmp_path: Path,
    tamper: str,
) -> None:
    model, optimizer, scheduler, context = _components()
    ema = EMAWeights(model, 0.9)
    checkpoint = tmp_path / f"resume-{tamper}"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        3,
        context,
        ema=ema,
    )
    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=True)
    if tamper == "missing":
        payload.pop("ema")
    else:
        payload["ema"].pop(next(iter(payload["ema"])))
    torch.save(payload, checkpoint / "checkpoint.pt")

    fresh, fresh_optimizer, fresh_scheduler, _ = _components()
    fresh_ema = EMAWeights(fresh, 0.9)
    model_before = {name: tensor.clone() for name, tensor in fresh.state_dict().items()}
    optimizer_before = deepcopy(fresh_optimizer.state_dict())
    scheduler_before = deepcopy(fresh_scheduler.state_dict())
    ema_before = {name: tensor.clone() for name, tensor in fresh_ema.shadow.items()}

    with pytest.raises(ValueError, match="EMA"):
        load_checkpoint(
            checkpoint,
            fresh,
            fresh_optimizer,
            fresh_scheduler,
            context,
            ema=fresh_ema,
        )

    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, model_before[name])
    assert fresh_optimizer.state_dict() == optimizer_before
    assert fresh_scheduler.state_dict() == scheduler_before
    for name, tensor in fresh_ema.shadow.items():
        torch.testing.assert_close(tensor, ema_before[name])


@pytest.mark.parametrize("missing_field", ["rng_state", "scaler"])
def test_resume_requires_current_rng_and_active_scaler_before_model_mutation(
    tmp_path: Path,
    missing_field: str,
) -> None:
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / f"resume-{missing_field}"
    source_scaler = FakeScaler(8.0) if missing_field == "scaler" else None
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        3,
        context,
        scaler=source_scaler,
    )
    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=True)
    payload.pop(missing_field)
    torch.save(payload, checkpoint / "checkpoint.pt")

    fresh, fresh_optimizer, fresh_scheduler, _ = _components()
    fresh_scaler = FakeScaler(2.0) if missing_field == "scaler" else None
    model_before = {name: tensor.clone() for name, tensor in fresh.state_dict().items()}
    optimizer_before = deepcopy(fresh_optimizer.state_dict())
    scheduler_before = deepcopy(fresh_scheduler.state_dict())

    with pytest.raises(ValueError, match="RNG|scaler"):
        load_checkpoint(
            checkpoint,
            fresh,
            fresh_optimizer,
            fresh_scheduler,
            context,
            scaler=fresh_scaler,
        )

    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, model_before[name])
    assert fresh_optimizer.state_dict() == optimizer_before
    assert fresh_scheduler.state_dict() == scheduler_before
    if fresh_scaler is not None:
        assert fresh_scaler.scale == 2.0


def test_local_checkpoint_restores_python_numpy_and_torch_rng(tmp_path: Path) -> None:
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "checkpoint"
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    save_checkpoint(checkpoint, model, optimizer, scheduler, 1, context)
    expected = (random.random(), float(np.random.random()), torch.rand(4))

    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    load_checkpoint(checkpoint, model, optimizer, scheduler, context)
    actual = (random.random(), float(np.random.random()), torch.rand(4))

    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


def test_cuda_rng_capture_reads_only_the_current_device(monkeypatch) -> None:
    expected = torch.tensor([1, 2, 3], dtype=torch.uint8)
    monkeypatch.setattr(checkpoint_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(checkpoint_module.torch.cuda, "get_rng_state", lambda: expected)
    monkeypatch.setattr(
        checkpoint_module.torch.cuda,
        "get_rng_state_all",
        lambda: (_ for _ in ()).throw(AssertionError("must not touch peer devices")),
    )

    state = checkpoint_module._capture_rng_state()

    torch.testing.assert_close(state["torch_cuda"], expected)


def test_legacy_cuda_rng_lists_restore_only_the_current_device(monkeypatch) -> None:
    state = checkpoint_module._capture_rng_state()
    first = torch.tensor([1, 2], dtype=torch.uint8)
    second = torch.tensor([3, 4], dtype=torch.uint8)
    state["torch_cuda"] = [first, second]
    restored: list[torch.Tensor] = []
    monkeypatch.setattr(checkpoint_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(checkpoint_module.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        checkpoint_module.torch.cuda,
        "set_rng_state",
        lambda value: restored.append(value.clone()),
    )
    monkeypatch.setattr(
        checkpoint_module.torch.cuda,
        "set_rng_state_all",
        lambda values: (_ for _ in ()).throw(AssertionError(f"unexpected states: {values}")),
    )

    checkpoint_module._restore_rng_state(state)

    assert len(restored) == 1
    torch.testing.assert_close(restored[0], second)


def test_distributed_checkpoint_publishes_complete_directory_and_falls_back(
    tmp_path: Path,
) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("distributed checkpoint test requires Gloo")
    if torch.distributed.is_initialized():
        pytest.skip("test requires ownership of the default process group")

    torch.distributed.init_process_group(
        "gloo",
        init_method=(tmp_path / "dcp-rendezvous").resolve().as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        model, optimizer, scheduler, _ = _components()
        context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
        checkpoint = tmp_path / "latest"
        identity = {"schema": "test", "tokenizer": {"sha256": "a" * 64}}

        # Materialize Adam moments so a failed identity check can prove that
        # neither model nor optimizer state was populated by DCP first.
        sum(parameter.sum() for parameter in model.parameters()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            1,
            context,
            identity=identity,
        )
        first_weight = model.token_embedding.weight.detach().clone()
        with torch.no_grad():
            model.token_embedding.weight.add_(1.0)
            optimizer.state[model.token_embedding.weight]["exp_avg"].add_(1.0)
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            2,
            context,
            identity=identity,
        )

        marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
        previous = checkpoint.with_name(".latest.previous")
        assert marker.is_file()
        assert (previous / checkpoint_module.DCP_COMPLETION_FILENAME).is_file()
        assert not checkpoint.with_name(".latest.staging").exists()
        with torch.no_grad():
            model.token_embedding.weight.add_(2.0)
            optimizer.state[model.token_embedding.weight]["exp_avg"].add_(2.0)
        live_weight = model.token_embedding.weight.detach().clone()
        live_exp_avg = optimizer.state[model.token_embedding.weight]["exp_avg"].detach().clone()
        with pytest.raises(ValueError, match="identity does not match"):
            load_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                context,
                expected_identity={"schema": "test", "tokenizer": {"sha256": "b" * 64}},
            )
        torch.testing.assert_close(model.token_embedding.weight, live_weight)
        torch.testing.assert_close(
            optimizer.state[model.token_embedding.weight]["exp_avg"],
            live_exp_avg,
        )
        with pytest.raises(ValueError, match="no recorded pipeline identity"):
            load_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                context,
                expected_identity={
                    "schema": "test",
                    "tokenizer": {"sha256": "a" * 64},
                    "pipeline": {
                        "schema": "sion-translation-pipeline-v1",
                        "branch": "translation-only",
                    },
                },
            )
        torch.testing.assert_close(model.token_embedding.weight, live_weight)
        torch.testing.assert_close(
            optimizer.state[model.token_embedding.weight]["exp_avg"],
            live_exp_avg,
        )
        with pytest.raises(ValueError, match=r"identity\.tokenizer\.metadata"):
            load_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                context,
                expected_identity={
                    "schema": "test",
                    "tokenizer": {
                        "sha256": "a" * 64,
                        "metadata": {"sha256": "c" * 64},
                    },
                },
            )
        torch.testing.assert_close(model.token_embedding.weight, live_weight)
        torch.testing.assert_close(
            optimizer.state[model.token_embedding.weight]["exp_avg"],
            live_exp_avg,
        )

        fresh, _, _, _ = _components()
        fresh_weight = fresh.token_embedding.weight.detach().clone()
        with pytest.raises(ValueError, match="tokenizer/model identity"):
            initialize_model_from_checkpoint(
                checkpoint,
                fresh,
                context,
                expected_identity={
                    "schema": "test",
                    "tokenizer": {"sha256": "b" * 64},
                },
            )
        torch.testing.assert_close(fresh.token_embedding.weight, fresh_weight)

        # An incomplete current publication must not destroy the retained
        # checkpoint. Removing its marker simulates interruption around publish.
        marker.unlink()
        with pytest.warns(RuntimeWarning, match="retained checkpoint"):
            step = load_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                context,
                expected_identity=identity,
            )
        assert step == 1
        torch.testing.assert_close(model.token_embedding.weight, first_weight)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "mode",
    ["prepare-failure", "publish-success", "publish-failure"],
)
def test_distributed_save_long_io_heartbeats_and_propagates_rank_zero_result(
    tmp_path: Path,
    mode: str,
) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("distributed checkpoint test requires Gloo")
    if torch.distributed.is_initialized():
        pytest.skip("test requires ownership of the default process group")

    checkpoint = tmp_path / "latest"
    result_directory = tmp_path / "results"
    rendezvous_uri = (tmp_path / "heartbeat-rendezvous").resolve().as_uri()
    process_context = multiprocessing.get_context("spawn")
    processes = [
        process_context.Process(
            target=_distributed_save_heartbeat_worker,
            args=(
                rank,
                2,
                rendezvous_uri,
                str(checkpoint),
                str(result_directory),
                mode,
            ),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 30.0
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    hanging = [process for process in processes if process.is_alive()]
    for process in hanging:
        process.terminate()
        process.join(5.0)
    assert not hanging, "distributed checkpoint save processes hung"
    assert [process.exitcode for process in processes] == [0, 0]

    results = [
        (result_directory / f"rank-{rank}.txt").read_text(encoding="utf-8") for rank in range(2)
    ]
    assert all("timeout" not in result.lower() for result in results)
    if mode == "publish-success":
        assert results == ["ok", "ok"]
        assert (checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME).is_file()
    else:
        expected_detail = (
            "injected staging cleanup failure"
            if mode == "prepare-failure"
            else "injected completion publication failure"
        )
        assert results[0].startswith("OSError|")
        assert results[1].startswith("RuntimeError|")
        assert all(expected_detail in result for result in results)


def test_checkpoint_io_collective_failure_waits_for_worker_ownership_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_started = threading.Event()
    release_action = threading.Event()
    action_completed = threading.Event()
    context = DistributedContext(0, 0, 2, torch.device("cpu"), True, "gloo")

    def blocking_action() -> None:
        action_started.set()
        release_action.wait(10.0)
        action_completed.set()

    def fail_heartbeat(*_args, **_kwargs) -> None:
        assert action_started.wait(1.0)
        raise RuntimeError("injected heartbeat collective failure")

    monkeypatch.setattr(checkpoint_module.torch.distributed, "all_reduce", fail_heartbeat)
    releaser = threading.Timer(0.1, release_action.set)
    releaser.start()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="injected heartbeat collective failure"):
            checkpoint_module._run_checkpoint_io_action(
                blocking_action,
                context,
                operation="blocked test action",
                rank_zero_only=True,
            )
    finally:
        release_action.set()
        releaser.join()

    assert action_completed.is_set()
    assert time.monotonic() - started >= 0.05


def _write_fake_complete_dcp(path: Path, *, step: int, world_size: int = 1) -> object:
    path.mkdir(parents=True)
    (path / ".metadata").write_bytes(f"metadata-{step}".encode())
    (path / "__0_0.distcp").write_bytes(f"shard-{step}".encode())
    for rank in range(world_size):
        (path / f"rng-rank-{rank:05d}.pt").write_bytes(f"rng-{rank}-{step}".encode())
    return checkpoint_module._write_dcp_completion(
        path,
        step=step,
        world_size=world_size,
    )


def test_checkpoint_discovery_includes_a_retained_local_previous_generation(
    tmp_path: Path,
) -> None:
    latest = tmp_path / "latest"
    previous = tmp_path / ".latest.previous"
    previous.mkdir()
    (previous / "checkpoint.pt").write_bytes(b"retained local checkpoint")

    assert checkpoint_module.checkpoint_path_exists(latest)
    assert checkpoint_module.checkpoint_path_exists(previous)


def _write_markerless_scalar_dcp(path: Path, *, step: int) -> None:
    import torch.distributed.checkpoint as dcp

    dcp.save({"step": step}, checkpoint_id=path)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "rng_state": checkpoint_module._capture_rng_state(),
        },
        path / "rng-rank-00000.pt",
    )


def test_checkpoint_generation_candidates_orders_current_then_previous_without_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    _write_fake_complete_dcp(current, step=2)
    _write_fake_complete_dcp(previous, step=1)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    monkeypatch.setattr(
        checkpoint_module,
        "_sha256_file",
        lambda _path: pytest.fail("candidate ordering must not hash shard contents"),
    )

    assert checkpoint_module.checkpoint_generation_candidates(current, context) == (
        current,
        previous,
    )
    assert checkpoint_module.checkpoint_generation_candidates(previous, context) == (
        current,
        previous,
    )


def test_local_checkpoint_generation_bindings_preserve_digest_order(tmp_path: Path) -> None:
    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    current.mkdir()
    previous.mkdir()
    current_payload = current / "checkpoint.pt"
    previous_payload = previous / "checkpoint.pt"
    current_payload.write_bytes(b"current generation")
    previous_payload.write_bytes(b"previous generation")
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    bindings = checkpoint_module.checkpoint_generation_bindings(current, context)

    assert [binding.source for binding in bindings] == [current, previous]
    assert [binding.artifact_sha256 for binding in bindings] == [
        hashlib.sha256(current_payload.read_bytes()).hexdigest(),
        hashlib.sha256(previous_payload.read_bytes()).hexdigest(),
    ]


@pytest.mark.parametrize("malformed_field", ["optimizer", "model"])
def test_local_load_structure_preflight_reaches_previous_without_mutating_model(
    tmp_path: Path,
    malformed_field: str,
) -> None:
    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    source, source_optimizer, source_scheduler, context = _components()
    save_checkpoint(current, source, source_optimizer, source_scheduler, 5, context)
    save_checkpoint(previous, source, source_optimizer, source_scheduler, 4, context)
    current_payload = torch.load(current / "checkpoint.pt", weights_only=True)
    if malformed_field == "optimizer":
        current_payload["optimizer"] = {}
    else:
        model_state = current_payload["model"]
        first_name = next(iter(model_state))
        model_state[first_name] = model_state[first_name][:1]
    torch.save(current_payload, current / "checkpoint.pt")

    target, _, _, _ = _components()
    before = _clone_model_state(target)
    selected: Path | None = None
    for binding in checkpoint_module.checkpoint_generation_bindings(current, context):
        try:
            with checkpoint_module.verified_checkpoint_generation_lease(
                current,
                context,
                expected_artifact_sha256=binding.artifact_sha256,
            ) as generation:
                checkpoint_module.preflight_checkpoint_load_structure(
                    generation.source,
                    target,
                    context,
                )
                selected = generation.source
        except (RuntimeError, ValueError):
            continue
        break

    assert selected == previous
    _assert_model_state_unchanged(target, before)


def test_dcp_load_structure_preflight_reaches_previous_after_model_shape_mismatch(
    tmp_path: Path,
) -> None:
    from torch.distributed.checkpoint.state_dict import get_state_dict

    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    source, source_optimizer, source_scheduler, _ = _components()
    model_state, optimizer_state = get_state_dict(source, source_optimizer)
    malformed_model_state = dict(model_state)
    first_name = next(iter(malformed_model_state))
    malformed_model_state[first_name] = malformed_model_state[first_name][:1]
    shared_state = {
        "schema": CHECKPOINT_SCHEMA,
        "optimizer": optimizer_state,
        "scheduler": source_scheduler.state_dict(),
        "training_state": {"epoch": 0, "batch_in_epoch": 0},
    }
    _write_test_dcp_generation(
        current,
        {**shared_state, "model": malformed_model_state, "step": 5},
        marker_step=5,
    )
    _write_test_dcp_generation(
        previous,
        {**shared_state, "model": model_state, "step": 4},
        marker_step=4,
    )
    target, _, _, _ = _components()
    before = _clone_model_state(target)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    selected: Path | None = None

    for binding in checkpoint_module.checkpoint_generation_bindings(current, context):
        try:
            with checkpoint_module.verified_checkpoint_generation_lease(
                current,
                context,
                expected_artifact_sha256=binding.artifact_sha256,
            ) as generation:
                checkpoint_module.preflight_checkpoint_load_structure(
                    generation.source,
                    target,
                    context,
                )
                selected = generation.source
        except (RuntimeError, ValueError):
            continue
        break

    assert selected == previous
    _assert_model_state_unchanged(target, before)


def test_dcp_checkpoint_generation_bindings_preserve_marker_order(tmp_path: Path) -> None:
    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    _write_fake_complete_dcp(current, step=2)
    _write_fake_complete_dcp(previous, step=1)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")

    bindings = checkpoint_module.checkpoint_generation_bindings(current, context)

    assert [binding.source for binding in bindings] == [current.resolve(), previous.resolve()]
    assert [binding.artifact_sha256 for binding in bindings] == [
        hashlib.sha256(
            (current / checkpoint_module.DCP_COMPLETION_FILENAME).read_bytes()
        ).hexdigest(),
        hashlib.sha256(
            (previous / checkpoint_module.DCP_COMPLETION_FILENAME).read_bytes()
        ).hexdigest(),
    ]


def test_checkpoint_generation_candidate_metadata_reads_only_the_small_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "candidate"
    _write_fake_complete_dcp(checkpoint, step=7)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    monkeypatch.setattr(
        checkpoint_module,
        "_sha256_file",
        lambda _path: pytest.fail("candidate metadata must not hash shard contents"),
    )

    artifact_sha256, marker_step = checkpoint_module.checkpoint_generation_candidate_metadata(
        checkpoint, context
    )

    assert artifact_sha256 == hashlib.sha256(marker.read_bytes()).hexdigest()
    assert marker_step == 7


@pytest.mark.parametrize("invalid_current", ["broken-marker", "missing-rng"])
def test_checkpoint_generation_candidates_skips_structurally_invalid_current(
    tmp_path: Path,
    invalid_current: str,
) -> None:
    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    _write_fake_complete_dcp(current, step=2)
    _write_fake_complete_dcp(previous, step=1)
    if invalid_current == "broken-marker":
        (current / checkpoint_module.DCP_COMPLETION_FILENAME).write_bytes(b"{broken")
    else:
        (current / "rng-rank-00000.pt").unlink()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")

    assert checkpoint_module.checkpoint_generation_candidates(current, context) == (previous,)


def test_checkpoint_generation_candidates_excludes_unsealed_legacy_dcp(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy"
    _write_markerless_scalar_dcp(checkpoint, step=3)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")

    assert checkpoint_module.checkpoint_generation_candidates(checkpoint, context) == ()


def test_checkpoint_candidate_lease_falls_back_from_corrupt_current_to_previous(
    tmp_path: Path,
) -> None:
    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    _write_fake_complete_dcp(current, step=2)
    _write_fake_complete_dcp(previous, step=1)
    (current / "__0_0.distcp").write_bytes(b"broken")
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    selected: Path | None = None

    for candidate in checkpoint_module.checkpoint_generation_candidates(current, context):
        marker = candidate / checkpoint_module.DCP_COMPLETION_FILENAME
        marker_digest = hashlib.sha256(marker.read_bytes()).hexdigest()
        try:
            with checkpoint_module.verified_checkpoint_source_lease(
                candidate,
                context,
                marker_digest,
            ) as leased:
                selected = leased
        except (RuntimeError, ValueError):
            continue
        break

    assert selected == previous


def test_verified_generation_lease_falls_back_after_identity_preflight(
    tmp_path: Path,
) -> None:
    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    expected_identity = {"schema": "test", "tokenizer": {"sha256": "a" * 64}}
    _write_test_dcp_generation(
        current,
        {
            "identity": {"schema": "test", "tokenizer": {"sha256": "b" * 64}},
            "step": 8,
        },
        marker_step=8,
    )
    _write_test_dcp_generation(
        previous,
        {"identity": expected_identity, "step": 7},
        marker_step=7,
    )
    previous_marker = previous / checkpoint_module.DCP_COMPLETION_FILENAME
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")

    with checkpoint_module.verified_checkpoint_generation_lease(
        current,
        context,
        expected_identity,
        expected_step=7,
    ) as selected:
        assert selected.source == previous
        assert selected.step == 7
        assert selected.artifact_sha256 == hashlib.sha256(previous_marker.read_bytes()).hexdigest()
        assert checkpoint_module._active_checkpoint_lease() is not None

    assert checkpoint_module._active_checkpoint_lease() is None


def test_verified_generation_lease_honors_an_exact_previous_marker_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = tmp_path / "best"
    previous = current.with_name(".best.previous")
    _write_test_dcp_generation(current, {"step": 9}, marker_step=9)
    _write_test_dcp_generation(previous, {"step": 8}, marker_step=8)
    previous_marker = previous / checkpoint_module.DCP_COMPLETION_FILENAME
    previous_digest = hashlib.sha256(previous_marker.read_bytes()).hexdigest()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    real_sha256_file = checkpoint_module._sha256_file
    hashed_paths: list[Path] = []

    def record_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", record_sha256)

    with checkpoint_module.verified_checkpoint_generation_lease(
        current,
        context,
        expected_artifact_sha256=previous_digest,
        expected_step=8,
    ) as selected:
        assert selected.source == previous

    assert hashed_paths
    assert all(path.parent == previous for path in hashed_paths)
    assert len(hashed_paths) == 3


def test_checkpoint_generation_candidates_preserves_mixed_pair_guard(tmp_path: Path) -> None:
    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    current.mkdir()
    (current / "checkpoint.pt").write_bytes(b"local")
    _write_fake_complete_dcp(previous, step=1)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")

    with pytest.raises(ValueError, match="mixes local checkpoint.pt"):
        checkpoint_module.checkpoint_generation_candidates(current, context)


def test_dcp_completion_rehashes_inventory_at_each_public_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "cached"
    _write_fake_complete_dcp(checkpoint, step=4)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    real_sha256_file = checkpoint_module._sha256_file
    hashed_paths: list[Path] = []

    def record_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", record_sha256)

    assert resolve_checkpoint_source(checkpoint, context) == checkpoint
    first_hash_count = len(hashed_paths)
    assert {path.name for path in hashed_paths} == {
        ".metadata",
        "__0_0.distcp",
        "rng-rank-00000.pt",
    }
    assert resolve_checkpoint_source(checkpoint, context) == checkpoint
    assert len(hashed_paths) == first_hash_count * 2


def test_dcp_completion_revalidates_after_marker_content_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "changed-marker"
    _write_fake_complete_dcp(checkpoint, step=5)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    real_sha256_file = checkpoint_module._sha256_file
    hash_count = 0

    def count_sha256(path: Path) -> str:
        nonlocal hash_count
        hash_count += 1
        return real_sha256_file(path)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", count_sha256)
    assert resolve_checkpoint_source(checkpoint, context) == checkpoint
    first_hash_count = hash_count
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    marker.write_bytes(marker.read_bytes() + b" ")

    assert resolve_checkpoint_source(checkpoint, context) == checkpoint
    assert hash_count == first_hash_count * 2


def test_dcp_completion_rejects_same_marker_after_shard_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "replaced-shard"
    _write_fake_complete_dcp(checkpoint, step=6)
    real_sha256_file = checkpoint_module._sha256_file
    hashed_paths: list[Path] = []

    def record_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", record_sha256)
    assert checkpoint_module._dcp_completion_status(checkpoint, world_size=1) == "valid"
    first_hash_count = len(hashed_paths)
    shard = checkpoint / "__0_0.distcp"
    replacement = checkpoint / "replacement.tmp"
    replacement.write_bytes(b"broken-")
    checkpoint_module.os.replace(replacement, shard)

    assert checkpoint_module._dcp_completion_status(checkpoint, world_size=1) == "invalid"
    assert len(hashed_paths) > first_hash_count


def test_dcp_completion_rejects_same_length_metadata_mutation_with_restored_mtime(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "restored-mtime"
    _write_fake_complete_dcp(checkpoint, step=6)
    metadata = checkpoint / ".metadata"
    original_stat = metadata.stat()
    assert checkpoint_module._dcp_completion_status(checkpoint, world_size=1) == "valid"
    metadata.write_bytes(b"tampered-6")
    checkpoint_module.os.utime(
        metadata,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    assert metadata.stat().st_size == original_stat.st_size
    assert metadata.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert checkpoint_module._dcp_completion_status(checkpoint, world_size=1) == "invalid"


def test_peer_registration_hashes_every_visible_inventory_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "peer-cache"
    _write_fake_complete_dcp(checkpoint, step=6, world_size=2)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    marker_digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    context = DistributedContext(1, 1, 2, torch.device("cpu"), False)
    real_sha256_file = checkpoint_module._sha256_file
    hashed_paths: list[Path] = []

    def record_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", record_sha256)

    assert (
        checkpoint_module.register_verified_checkpoint_source(
            checkpoint,
            context,
            marker_digest,
        )
        == checkpoint
    )
    assert {path.name for path in hashed_paths} == {
        ".metadata",
        "__0_0.distcp",
        "rng-rank-00000.pt",
        "rng-rank-00001.pt",
    }


def test_peer_registration_rejects_a_missing_inventory_shard(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "peer-missing-shard"
    _write_fake_complete_dcp(checkpoint, step=7, world_size=2)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    marker_digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    (checkpoint / "__0_0.distcp").unlink()
    context = DistributedContext(1, 1, 2, torch.device("cpu"), False)

    with pytest.raises(ValueError, match="inventory verification failed"):
        checkpoint_module.register_verified_checkpoint_source(
            checkpoint,
            context,
            marker_digest,
        )


def test_peer_registration_rejects_marker_digest_mismatch_without_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "peer-mismatch"
    _write_fake_complete_dcp(checkpoint, step=7, world_size=2)
    context = DistributedContext(1, 1, 2, torch.device("cpu"), False)
    monkeypatch.setattr(
        checkpoint_module,
        "_sha256_file",
        lambda _path: pytest.fail("digest mismatch must be rejected before shard hashing"),
    )

    with pytest.raises(ValueError, match="marker digest does not match"):
        checkpoint_module.register_verified_checkpoint_source(
            checkpoint,
            context,
            "0" * 64,
        )


@pytest.mark.parametrize("marker_kind", ["legacy", "missing-required-file"])
def test_peer_registration_rejects_non_v2_or_invalid_markers_without_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_kind: str,
) -> None:
    checkpoint = tmp_path / marker_kind
    _write_fake_complete_dcp(checkpoint, step=8, world_size=2)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    if marker_kind == "legacy":
        marker.write_text(
            json.dumps(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "step": 8,
                    "world_size": 2,
                }
            ),
            encoding="utf-8",
        )
    else:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["files"] = [
            entry for entry in payload["files"] if entry["path"] != "rng-rank-00001.pt"
        ]
        marker.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    marker_digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    context = DistributedContext(1, 1, 2, torch.device("cpu"), False)
    monkeypatch.setattr(
        checkpoint_module,
        "_sha256_file",
        lambda _path: pytest.fail("invalid peer marker must not trigger shard hashing"),
    )

    with pytest.raises(ValueError, match="not v2|missing required files"):
        checkpoint_module.register_verified_checkpoint_source(
            checkpoint,
            context,
            marker_digest,
        )


def test_peer_registration_keeps_mixed_pair_guard_after_verification(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mixed-after-cache"
    _write_fake_complete_dcp(checkpoint, step=9, world_size=2)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    marker_digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    context = DistributedContext(1, 1, 2, torch.device("cpu"), False)
    checkpoint_module.register_verified_checkpoint_source(
        checkpoint,
        context,
        marker_digest,
    )
    (checkpoint / "checkpoint.pt").write_bytes(b"mixed local payload")

    with pytest.raises(ValueError, match="mixes local checkpoint.pt"):
        checkpoint_module.register_verified_checkpoint_source(
            checkpoint,
            context,
            marker_digest,
        )


def test_verified_checkpoint_lease_hashes_once_then_expires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "leased"
    _write_fake_complete_dcp(checkpoint, step=9)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    marker_digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    real_sha256_file = checkpoint_module._sha256_file
    hashed_paths: list[Path] = []

    def record_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", record_sha256)

    with checkpoint_module.verified_checkpoint_source_lease(
        checkpoint,
        context,
        marker_digest,
    ) as source:
        assert source == checkpoint
        first_hash_count = len(hashed_paths)
        assert first_hash_count == 3
        assert resolve_checkpoint_source(source, context) == checkpoint
        assert resolve_checkpoint_source(source, context) == checkpoint
        assert len(hashed_paths) == first_hash_count

    assert resolve_checkpoint_source(checkpoint, context) == checkpoint
    assert len(hashed_paths) == first_hash_count * 2


def test_verified_checkpoint_lease_covers_repeated_public_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed.checkpoint as dcp

    checkpoint = tmp_path / "leased-preflight"
    dcp.save({"step": 15}, checkpoint_id=checkpoint)
    _seal_test_dcp(checkpoint, step=15)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    marker_digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    real_sha256_file = checkpoint_module._sha256_file
    hashed_paths: list[Path] = []

    def record_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", record_sha256)

    with checkpoint_module.verified_checkpoint_source_lease(
        checkpoint,
        context,
        marker_digest,
    ) as source:
        initial_hash_count = len(hashed_paths)
        assert preflight_checkpoint_identity(source, context, None) == 15
        assert preflight_checkpoint_identity(source, context, None) == 15
        assert len(hashed_paths) == initial_hash_count

    assert initial_hash_count == 3


def test_verified_checkpoint_lease_revokes_on_marker_change(tmp_path: Path) -> None:
    checkpoint = tmp_path / "changed-lease-marker"
    _write_fake_complete_dcp(checkpoint, step=10)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    marker_digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")

    with checkpoint_module.verified_checkpoint_source_lease(
        checkpoint,
        context,
        marker_digest,
    ):
        marker.write_bytes(marker.read_bytes() + b" ")
        with pytest.raises(ValueError, match="marker digest changed"):
            resolve_checkpoint_source(checkpoint, context)

    assert checkpoint_module._active_checkpoint_lease() is None


def test_verified_checkpoint_lease_rejects_another_source_without_hashing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "leased-source"
    other = tmp_path / "other-source"
    _write_fake_complete_dcp(checkpoint, step=11)
    _write_fake_complete_dcp(other, step=12)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    marker_digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    real_sha256_file = checkpoint_module._sha256_file
    hashed_paths: list[Path] = []

    def record_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", record_sha256)

    with checkpoint_module.verified_checkpoint_source_lease(
        checkpoint,
        context,
        marker_digest,
    ):
        with pytest.raises(ValueError, match="lease source does not match"):
            resolve_checkpoint_source(other, context)

    assert all(path.parent != other for path in hashed_paths)


@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
def test_full_checkpoint_load_consumes_a_verification_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fails: bool,
) -> None:
    checkpoint = tmp_path / "load-lease"
    _write_fake_complete_dcp(checkpoint, step=11)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    marker_digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    model, optimizer, scheduler, _ = _components()

    def fake_load(*_args, **_kwargs) -> int:
        if fails:
            raise RuntimeError("injected full load failure")
        return 11

    monkeypatch.setattr(checkpoint_module, "_load_checkpoint_impl", fake_load)

    with checkpoint_module.verified_checkpoint_source_lease(
        checkpoint,
        context,
        marker_digest,
    ):
        if fails:
            with pytest.raises(RuntimeError, match="injected full load failure"):
                load_checkpoint(checkpoint, model, optimizer, scheduler, context)
        else:
            assert load_checkpoint(checkpoint, model, optimizer, scheduler, context) == 11
        assert checkpoint_module._active_checkpoint_lease() is None


@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
def test_stage_transfer_load_consumes_a_verification_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fails: bool,
) -> None:
    checkpoint = tmp_path / "stage-lease"
    _write_fake_complete_dcp(checkpoint, step=12)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    marker_digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    model, _, _, _ = _components()

    def fake_load(*_args, **_kwargs) -> dict[str, object]:
        if fails:
            raise RuntimeError("injected stage load failure")
        return {"source": str(checkpoint), "step": 12}

    monkeypatch.setattr(
        checkpoint_module,
        "_initialize_model_from_checkpoint_impl",
        fake_load,
    )

    with checkpoint_module.verified_checkpoint_source_lease(
        checkpoint,
        context,
        marker_digest,
    ):
        if fails:
            with pytest.raises(RuntimeError, match="injected stage load failure"):
                initialize_model_from_checkpoint(checkpoint, model, context)
        else:
            assert initialize_model_from_checkpoint(checkpoint, model, context)["step"] == 12
        assert checkpoint_module._active_checkpoint_lease() is None


def test_markerless_dcp_is_not_a_default_resume_source(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unsealed-legacy"
    _write_markerless_scalar_dcp(checkpoint, step=10)

    with pytest.raises(FileNotFoundError, match="authenticated v2"):
        checkpoint_module._resolve_dcp_checkpoint(checkpoint, world_size=1)


def test_explicit_upgrade_markerless_dcp_publishes_v2_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "legacy-markerless"
    _write_markerless_scalar_dcp(checkpoint, step=10)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    assert not marker.exists()

    marker_digest = checkpoint_module.upgrade_legacy_dcp_completion(
        checkpoint,
        world_size=1,
        step=10,
    )

    assert marker_digest == hashlib.sha256(marker.read_bytes()).hexdigest()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["schema"] == checkpoint_module.DCP_COMPLETION_SCHEMA
    assert payload["step"] == 10
    real_sha256_file = checkpoint_module._sha256_file
    hashed_paths: list[Path] = []

    def record_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(checkpoint_module, "_sha256_file", record_sha256)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    assert resolve_checkpoint_source(checkpoint, context) == checkpoint
    assert {path.name for path in hashed_paths} == {
        ".metadata",
        "__0_0.distcp",
        "rng-rank-00000.pt",
    }


def test_upgrade_existing_v2_marker_is_a_verified_byte_for_byte_noop(tmp_path: Path) -> None:
    checkpoint = tmp_path / "already-v2"
    _write_markerless_scalar_dcp(checkpoint, step=11)
    first_digest = checkpoint_module.upgrade_legacy_dcp_completion(checkpoint, 1, 11)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    original_marker = marker.read_bytes()

    second_digest = checkpoint_module.upgrade_legacy_dcp_completion(checkpoint, 1, 11)

    assert second_digest == first_digest
    assert marker.read_bytes() == original_marker


@pytest.mark.parametrize(
    ("world_size", "step", "match"),
    [(1, 13, "payload step"), (2, 12, "RNG files")],
)
def test_upgrade_markerless_dcp_rejects_step_or_world_mismatch_without_marker(
    tmp_path: Path,
    world_size: int,
    step: int,
    match: str,
) -> None:
    checkpoint = tmp_path / f"mismatch-{world_size}-{step}"
    _write_markerless_scalar_dcp(checkpoint, step=12)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME

    with pytest.raises(ValueError, match=match):
        checkpoint_module.upgrade_legacy_dcp_completion(
            checkpoint,
            world_size=world_size,
            step=step,
        )

    assert not marker.exists()


def test_upgrade_rejects_an_invalid_existing_marker_without_overwriting_it(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "invalid-marker"
    _write_markerless_scalar_dcp(checkpoint, step=14)
    marker = checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME
    marker.write_bytes(b"{broken")
    original_marker = marker.read_bytes()

    with pytest.raises(ValueError, match="completion marker is invalid"):
        checkpoint_module.upgrade_legacy_dcp_completion(checkpoint, 1, 14)

    assert marker.read_bytes() == original_marker


@pytest.mark.parametrize("tamper", ["shard", "rng", "marker"])
def test_dcp_completion_inventory_falls_back_from_corruption(
    tmp_path: Path,
    tamper: str,
) -> None:
    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    _write_fake_complete_dcp(previous, step=1)
    _write_fake_complete_dcp(current, step=2)
    if tamper == "shard":
        (current / "__0_0.distcp").write_bytes(b"corrupt")
    elif tamper == "rng":
        (current / "rng-rank-00000.pt").unlink()
    else:
        (current / checkpoint_module.DCP_COMPLETION_FILENAME).write_text(
            "{broken",
            encoding="utf-8",
        )

    with pytest.warns(RuntimeWarning, match="retained checkpoint"):
        resolved = checkpoint_module._resolve_dcp_checkpoint(current, world_size=1)

    assert resolved == previous


def test_invalid_dcp_marker_cannot_fall_through_to_legacy_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "latest"
    _write_fake_complete_dcp(checkpoint, step=1)
    (checkpoint / checkpoint_module.DCP_COMPLETION_FILENAME).write_text(
        "{broken",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="current=invalid"):
        checkpoint_module._resolve_dcp_checkpoint(checkpoint, world_size=1)


def test_dcp_publish_keeps_valid_previous_when_corrupt_current_install_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = tmp_path / "latest"
    previous = current.with_name(".latest.previous")
    staging = current.with_name(".latest.staging")
    _write_fake_complete_dcp(previous, step=1)
    _write_fake_complete_dcp(current, step=2)
    verified_staging = _write_fake_complete_dcp(staging, step=3)
    (current / "__0_0.distcp").write_bytes(b"corrupt")
    real_replace = checkpoint_module.os.replace

    def fail_staging_install(source: str | Path, destination: str | Path) -> None:
        if Path(source) == staging and Path(destination) == current:
            raise OSError("injected staging install failure")
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_staging_install)
    with pytest.raises(OSError, match="injected staging install failure"):
        checkpoint_module._publish_dcp_staging(
            staging,
            current,
            world_size=1,
            verified=verified_staging,
        )

    assert not current.exists()
    assert staging.exists()
    assert checkpoint_module._dcp_completion_status(previous, world_size=1) == "valid"
    with pytest.warns(RuntimeWarning, match="retained checkpoint"):
        assert checkpoint_module._resolve_dcp_checkpoint(current, world_size=1) == previous


@pytest.mark.parametrize("missing", [".metadata", "rng-rank-00000.pt"])
def test_dcp_completion_writer_requires_the_exact_core_inventory(
    tmp_path: Path,
    missing: str,
) -> None:
    staging = tmp_path / f"missing-{missing.lstrip('.')}"
    staging.mkdir()
    (staging / ".metadata").write_bytes(b"metadata")
    (staging / "__0_0.distcp").write_bytes(b"shard")
    (staging / "rng-rank-00000.pt").write_bytes(b"rng")
    (staging / missing).unlink()

    with pytest.raises(ValueError, match="missing required files"):
        checkpoint_module._write_dcp_completion(staging, step=1, world_size=1)

    assert not (staging / checkpoint_module.DCP_COMPLETION_FILENAME).exists()


def test_dcp_completion_writer_rejects_rng_files_outside_world_size(tmp_path: Path) -> None:
    staging = tmp_path / "extra-rng"
    staging.mkdir()
    (staging / ".metadata").write_bytes(b"metadata")
    (staging / "__0_0.distcp").write_bytes(b"shard")
    (staging / "rng-rank-00000.pt").write_bytes(b"rng-0")
    (staging / "rng-rank-00001.pt").write_bytes(b"rng-1")

    with pytest.raises(ValueError, match="unexpected RNG files"):
        checkpoint_module._write_dcp_completion(staging, step=1, world_size=1)


def test_dcp_publish_requires_a_capability_for_the_exact_staging_source(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".latest.staging"
    other = tmp_path / ".other.staging"
    destination = tmp_path / "latest"
    _write_fake_complete_dcp(staging, step=2)
    wrong_capability = _write_fake_complete_dcp(other, step=2)

    with pytest.raises(ValueError, match="capability source does not match"):
        checkpoint_module._publish_dcp_staging(
            staging,
            destination,
            world_size=1,
            verified=wrong_capability,
        )

    assert staging.is_dir()
    assert not destination.exists()


def test_dcp_publish_rejects_a_marker_changed_after_capability_issue(tmp_path: Path) -> None:
    staging = tmp_path / ".latest.staging"
    destination = tmp_path / "latest"
    capability = _write_fake_complete_dcp(staging, step=3)
    marker = staging / checkpoint_module.DCP_COMPLETION_FILENAME
    marker.write_bytes(marker.read_bytes() + b" ")

    with pytest.raises(ValueError, match="marker changed before publication"):
        checkpoint_module._publish_dcp_staging(
            staging,
            destination,
            world_size=1,
            verified=capability,
        )

    assert staging.is_dir()
    assert not destination.exists()


def test_distributed_stage_transfer_supports_ddp_canonical_model_keys(
    tmp_path: Path,
) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("distributed checkpoint test requires Gloo")
    if torch.distributed.is_initialized():
        pytest.skip("test requires ownership of the default process group")

    torch.distributed.init_process_group(
        "gloo",
        init_method=(tmp_path / "dcp-stage-rendezvous").resolve().as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        source, _, _, _ = _components()
        source_ddp = torch.nn.parallel.DistributedDataParallel(source)
        optimizer = torch.optim.AdamW(source_ddp.parameters(), lr=1e-3)
        scheduler = cosine_scheduler(optimizer, warmup_steps=0, max_steps=2, min_ratio=0.1)
        with torch.no_grad():
            source.token_embedding.weight.fill_(1.0)
        ema = EMAWeights(source_ddp, decay=0.9)
        with torch.no_grad():
            ema.shadow["module.token_embedding.weight"].fill_(2.0)
        checkpoint = tmp_path / "foundation-ddp"
        context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
        save_checkpoint(
            checkpoint,
            source_ddp,
            optimizer,
            scheduler,
            9,
            context,
            ema=ema,
        )

        fresh, _, _, _ = _components()
        fresh_ddp = torch.nn.parallel.DistributedDataParallel(fresh)
        provenance = initialize_model_from_checkpoint(checkpoint, fresh_ddp, context)

        assert torch.all(fresh.token_embedding.weight == 2.0)
        assert provenance["weights"] == "ema"
        assert provenance["step"] == 9
    finally:
        torch.distributed.destroy_process_group()


# ── Stage transfer (foundation -> SFT) ──────────────────────────────────


def _identity(model, tmp_path: Path, *, dataset_name: str, stage: str) -> dict:
    tokenizer = tmp_path / "tok" / "sion.model"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    if not tokenizer.exists():
        tokenizer.write_bytes(b"tokenizer-bytes")
    dataset = tmp_path / dataset_name
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "manifest.json").write_text(f'{{"name": "{dataset_name}"}}', encoding="utf-8")
    return build_checkpoint_identity(
        model_config=model.config,
        tokenizer_path=tokenizer,
        token_features_path=None,
        dataset_dir=dataset,
        stage_name=stage,
    )


def test_stage_transfer_loads_weights_without_optimizer_or_step(tmp_path: Path) -> None:
    """Moving from foundation training to SFT is not a resume operation.

    The new stage has a new objective and learning-rate schedule. Carrying over
    Adam moments and the step counter would skip warmup and apply momentum from
    a different loss surface.
    """
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "foundation"
    with torch.no_grad():
        model.token_embedding.weight.fill_(0.5)
    trained = model.token_embedding.weight.detach().clone()
    for _ in range(5):
        scheduler.step()
    save_checkpoint(checkpoint, model, optimizer, scheduler, 4200, context)

    fresh, fresh_optimizer, fresh_scheduler, _ = _components()
    with torch.no_grad():
        fresh.token_embedding.weight.fill_(-1.0)
    scheduler_before = deepcopy(fresh_scheduler.state_dict())
    optimizer_before = deepcopy(fresh_optimizer.state_dict())

    provenance = initialize_model_from_checkpoint(checkpoint, fresh, context)

    assert torch.allclose(fresh.token_embedding.weight, trained)
    # Report the old step only as provenance; the new stage starts at step zero.
    assert provenance["step"] == 4200
    assert fresh_scheduler.state_dict() == scheduler_before
    assert fresh_optimizer.state_dict() == optimizer_before


def test_stage_transfer_selects_ema_weights_without_inheriting_optimizer_state(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "foundation"
    with torch.no_grad():
        model.token_embedding.weight.fill_(1.0)
    ema = EMAWeights(model, decay=0.9)
    with torch.no_grad():
        ema.shadow["token_embedding.weight"].fill_(2.0)
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        7,
        context,
        ema=ema,
    )

    fresh, fresh_optimizer, fresh_scheduler, _ = _components()
    optimizer_before = deepcopy(fresh_optimizer.state_dict())
    scheduler_before = deepcopy(fresh_scheduler.state_dict())
    provenance = initialize_model_from_checkpoint(checkpoint, fresh, context)

    assert torch.all(fresh.token_embedding.weight == 2.0)
    assert provenance == {
        "source": str(checkpoint),
        "step": 7,
        "stage": None,
        "weights": "ema",
    }
    assert fresh_optimizer.state_dict() == optimizer_before
    assert fresh_scheduler.state_dict() == scheduler_before


def test_stage_transfer_rejects_missing_required_ema_before_mutating_the_model(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "foundation"
    identity = _identity(
        model,
        tmp_path,
        dataset_name="foundation_dataset",
        stage="foundation",
    )
    training = AppConfig().training
    training.ema_decay = 0.999
    identity["objective"] = checkpoint_module.build_objective_identity(training)
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        7,
        context,
        identity=identity,
    )

    fresh, _, _, _ = _components()
    before = {name: tensor.clone() for name, tensor in fresh.state_dict().items()}
    with pytest.raises(ValueError, match="trained with EMA enabled"):
        initialize_model_from_checkpoint(checkpoint, fresh, context)

    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


def test_stage_transfer_rejects_ema_that_contradicts_a_disabled_source_policy(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "foundation"
    identity = _identity(
        model,
        tmp_path,
        dataset_name="foundation_dataset",
        stage="foundation",
    )
    training = AppConfig().training
    training.ema_decay = 0.0
    identity["objective"] = checkpoint_module.build_objective_identity(training)
    ema = EMAWeights(model, decay=0.9)
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        7,
        context,
        identity=identity,
        ema=ema,
    )

    fresh, _, _, _ = _components()
    before = {name: tensor.clone() for name, tensor in fresh.state_dict().items()}
    with pytest.raises(ValueError, match="records EMA as disabled"):
        initialize_model_from_checkpoint(checkpoint, fresh, context)

    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


def test_stage_transfer_validates_all_raw_tensors_before_loading_any(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "foundation"
    save_checkpoint(checkpoint, model, optimizer, scheduler, 7, context)
    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=True)
    model_names = list(payload["model"])
    payload["model"][model_names[0]] = torch.full_like(payload["model"][model_names[0]], 9.0)
    last_name = model_names[-1]
    payload["model"][last_name] = payload["model"][last_name].reshape(-1)[:1]
    torch.save(payload, checkpoint / "checkpoint.pt")

    fresh, _, _, _ = _components()
    before = {name: tensor.clone() for name, tensor in fresh.state_dict().items()}
    with pytest.raises(ValueError, match="model tensor metadata mismatch"):
        initialize_model_from_checkpoint(checkpoint, fresh, context)

    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


def test_stage_transfer_validates_all_ema_tensors_before_copying_any(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "foundation"
    ema = EMAWeights(model, decay=0.9)
    save_checkpoint(checkpoint, model, optimizer, scheduler, 7, context, ema=ema)
    payload = torch.load(checkpoint / "checkpoint.pt", weights_only=True)
    ema_names = list(payload["ema"])
    payload["ema"][ema_names[0]] = torch.full_like(payload["ema"][ema_names[0]], 9.0)
    last_name = ema_names[-1]
    payload["ema"][last_name] = payload["ema"][last_name].reshape(-1)[:1]
    torch.save(payload, checkpoint / "checkpoint.pt")

    fresh, _, _, _ = _components()
    before = {name: tensor.clone() for name, tensor in fresh.state_dict().items()}
    with pytest.raises(ValueError, match="tensor metadata mismatch"):
        initialize_model_from_checkpoint(checkpoint, fresh, context)

    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


def test_distributed_stage_transfer_selects_the_checkpoint_ema_weights(
    tmp_path: Path,
) -> None:
    import torch.distributed.checkpoint as dcp

    model, _, _, _ = _components()
    with torch.no_grad():
        model.token_embedding.weight.fill_(1.0)
    ema = EMAWeights(model, decay=0.9)
    with torch.no_grad():
        ema.shadow["token_embedding.weight"].fill_(2.0)
    checkpoint = tmp_path / "foundation-dcp"
    dcp.save(
        {
            "model": model.state_dict(),
            "step": 9,
            "ema": ema.state_dict(),
        },
        checkpoint_id=checkpoint,
    )
    _seal_test_dcp(checkpoint, step=9)

    fresh, _, _, _ = _components()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    provenance = initialize_model_from_checkpoint(checkpoint, fresh, context)

    assert torch.all(fresh.token_embedding.weight == 2.0)
    assert provenance["weights"] == "ema"
    assert provenance["step"] == 9


@pytest.mark.parametrize("tamper", ["dtype", "unexpected"])
def test_distributed_stage_transfer_preflights_ema_metadata(
    tmp_path: Path,
    tamper: str,
) -> None:
    import torch.distributed.checkpoint as dcp

    model, _, _, _ = _components()
    ema = EMAWeights(model, decay=0.9)
    ema_state = ema.state_dict()
    if tamper == "dtype":
        first_name = next(iter(ema_state))
        ema_state[first_name] = ema_state[first_name].double()
    else:
        ema_state["injected.unexpected"] = torch.tensor([3.0])
    checkpoint = tmp_path / f"foundation-dcp-{tamper}"
    dcp.save(
        {
            "model": model.state_dict(),
            "step": 9,
            "ema": ema_state,
        },
        checkpoint_id=checkpoint,
    )
    _seal_test_dcp(checkpoint, step=9)

    fresh, _, _, _ = _components()
    before = {name: tensor.clone() for name, tensor in fresh.state_dict().items()}
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    with pytest.raises(ValueError, match="distributed checkpoint ema"):
        initialize_model_from_checkpoint(checkpoint, fresh, context)

    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


def test_stage_transfer_accepts_a_different_dataset(tmp_path: Path) -> None:
    """The two stages normally use different datasets: monolingual and parallel."""
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "foundation"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        1,
        context,
        identity=_identity(model, tmp_path, dataset_name="foundation_dataset", stage="foundation"),
    )

    fresh, _, _, _ = _components()
    initialize_model_from_checkpoint(
        checkpoint,
        fresh,
        context,
        expected_identity=_identity(model, tmp_path, dataset_name="dataset", stage="pretrain"),
    )


def test_stage_transfer_refuses_a_different_tokenizer(tmp_path: Path) -> None:
    """A different tokenizer changes what each embedding row represents.

    The shapes still match, so ``load_state_dict`` succeeds. Without this guard,
    the new stage would silently inherit meaningless weights.
    """
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "foundation"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        1,
        context,
        identity=_identity(model, tmp_path, dataset_name="foundation_dataset", stage="foundation"),
    )

    other = tmp_path / "other"
    other.mkdir()
    expected = _identity(model, tmp_path, dataset_name="dataset", stage="pretrain")
    expected["tokenizer"]["model"]["sha256"] = "0" * 64

    fresh, _, _, _ = _components()
    with pytest.raises(ValueError, match="tokenizer/model identity"):
        initialize_model_from_checkpoint(checkpoint, fresh, context, expected_identity=expected)


def test_stage_transfer_refuses_a_different_model_config(tmp_path: Path) -> None:
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "foundation"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        1,
        context,
        identity=_identity(model, tmp_path, dataset_name="foundation_dataset", stage="foundation"),
    )

    expected = _identity(model, tmp_path, dataset_name="dataset", stage="pretrain")
    expected["model"]["config_sha256"] = "0" * 64

    fresh, _, _, _ = _components()
    with pytest.raises(ValueError, match="tokenizer/model identity"):
        initialize_model_from_checkpoint(checkpoint, fresh, context, expected_identity=expected)


def test_stage_transfer_refuses_a_checkpoint_without_an_identity(tmp_path: Path) -> None:
    """Stopping is safer than inheriting weights whose identity cannot be verified."""
    model, optimizer, scheduler, context = _components()
    checkpoint = tmp_path / "foundation"
    save_checkpoint(checkpoint, model, optimizer, scheduler, 1, context)

    fresh, _, _, _ = _components()
    with pytest.raises(ValueError, match="no recorded identity"):
        initialize_model_from_checkpoint(
            checkpoint,
            fresh,
            context,
            expected_identity=_identity(model, tmp_path, dataset_name="dataset", stage="pretrain"),
        )


# ── Objective identity: what the run optimizes is part of its identity ───


def _objective_config():
    from sion_translate.config import AppConfig

    return AppConfig()


def test_the_objective_identity_covers_the_optimizer_schedule() -> None:
    """Changing optimizer settings makes inherited momentum describe another curve."""
    from sion_translate.training.checkpoint import build_objective_identity

    config = _objective_config()
    baseline = build_objective_identity(config.training)
    for field, value in (
        ("learning_rate", 1e-5),
        ("adam_beta2", 0.98),
        ("weight_decay", 0.0),
        ("precision", "fp32"),
        ("ema_decay", 0.0),
        ("sft_selection_metric", "global_nll"),
        ("candidate_refinement_min_worst_direction_nll_gain", 2e-5),
    ):
        changed = _objective_config()
        setattr(changed.training, field, value)
        assert build_objective_identity(changed.training) != baseline, field


def test_legacy_objective_identity_injects_only_the_authenticated_release_default() -> None:
    """Old optimizer progress remains usable without trusting a forged objective."""

    current = checkpoint_module.build_objective_identity(_objective_config().training)
    legacy = deepcopy(current)
    release_margin_field = "candidate_refinement_min_worst_direction_nll_gain"
    legacy["supervised"].pop(release_margin_field)
    unhashed_legacy = {key: value for key, value in legacy.items() if key != "sha256"}
    legacy["sha256"] = hashlib.sha256(
        checkpoint_module._canonical_json(unhashed_legacy).encode("utf-8")
    ).hexdigest()

    normalized = checkpoint_module._normalize_identity_for_comparison({"objective": legacy})
    assert normalized["objective"] == current

    forged = deepcopy(legacy)
    forged["sha256"] = "0" * 64
    normalized_forged = checkpoint_module._normalize_identity_for_comparison({"objective": forged})
    assert release_margin_field not in normalized_forged["objective"]["supervised"]


def test_reward_weights_are_part_of_the_posttraining_identity() -> None:
    """For MRT, the reward definition is also the selection metric.

    Changing one weight puts ``validation_reward`` on a different scale, while
    early stopping still compares it with the previous best value.
    """
    from sion_translate.training.checkpoint import build_objective_identity

    config = _objective_config()
    baseline = build_objective_identity(
        config.training, config.posttraining, include_posttraining=True
    )
    for field, value in (
        ("reward_chrf_weight", 0.1),
        ("roundtrip_enabled", True),
        ("roundtrip_max_new_tokens", 128),
        ("samples_per_source", 8),
        ("selection_metric", "macro_direction_reward"),
        ("validation_num_beams", 1),
        ("validation_length_penalty", 0.8),
        ("decode_min_new_tokens", 2),
        ("decode_no_repeat_ngram_size", 3),
        ("decode_max_output_length_ratio", 2.5),
        ("decode_max_output_length_margin", 8),
    ):
        changed = _objective_config()
        setattr(changed.posttraining, field, value)
        assert (
            build_objective_identity(
                changed.training, changed.posttraining, include_posttraining=True
            )
            != baseline
        ), field


def test_posttraining_settings_do_not_affect_a_supervised_identity() -> None:
    """An MRT-only setting change must not prevent an SFT run from resuming."""
    from sion_translate.training.checkpoint import build_objective_identity

    config = _objective_config()
    baseline = build_objective_identity(config.training)
    changed = _objective_config()
    changed.posttraining.reward_chrf_weight = 0.1
    assert build_objective_identity(changed.training) == baseline


def test_resuming_with_a_changed_objective_is_refused(tmp_path: Path) -> None:
    from sion_translate.training.checkpoint import build_objective_identity

    model, optimizer, scheduler, context = _components()
    config = _objective_config()
    checkpoint = tmp_path / "checkpoint"

    def identity(app_config):
        return build_checkpoint_identity(
            model_config=model.config,
            tokenizer_path=tmp_path / "sion.model",
            token_features_path=None,
            dataset_dir=tmp_path,
            objective_identity=build_objective_identity(app_config.training),
        )

    save_checkpoint(checkpoint, model, optimizer, scheduler, 1, context, identity=identity(config))

    changed = _objective_config()
    changed.training.learning_rate = 1e-6
    with pytest.raises(ValueError, match="identity does not match"):
        load_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            context,
            expected_identity=identity(changed),
        )


def test_a_legacy_checkpoint_without_an_objective_warns_instead_of_failing(
    tmp_path: Path,
) -> None:
    """Legacy resume remains allowed but reports which checks are unavailable."""
    import warnings as warnings_module

    from sion_translate.training.checkpoint import build_objective_identity

    model, optimizer, scheduler, context = _components()
    legacy = build_checkpoint_identity(
        model_config=model.config,
        tokenizer_path=tmp_path / "sion.model",
        token_features_path=None,
        dataset_dir=tmp_path,
    )
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, optimizer, scheduler, 1, context, identity=legacy)

    modern = build_checkpoint_identity(
        model_config=model.config,
        tokenizer_path=tmp_path / "sion.model",
        token_features_path=None,
        dataset_dir=tmp_path,
        objective_identity=build_objective_identity(_objective_config().training),
    )
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        load_checkpoint(checkpoint, model, optimizer, scheduler, context, expected_identity=modern)
    assert any("objective/optimization identity" in str(entry.message) for entry in caught)
