from __future__ import annotations

from copy import deepcopy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import sion_translate.training.checkpoint as checkpoint_module
from sion_translate.config import DataConfig, ExperimentalConfig, ModelConfig
from sion_translate.model import SionForConditionalGeneration
from sion_translate.training.checkpoint import (
    CHECKPOINT_SCHEMA,
    load_checkpoint,
    save_checkpoint,
)
from sion_translate.training.distributed import DistributedContext
from sion_translate.training.trainer import cosine_scheduler


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


def test_checkpoint_identity_still_rejects_semantic_data_changes(tmp_path: Path) -> None:
    expected = _identity_fixture(tmp_path / "expected", approximate_split=True)
    incompatible = _identity_fixture(tmp_path / "incompatible", approximate_split=False)

    with pytest.raises(ValueError, match="identity does not match"):
        checkpoint_module._validate_identity({"identity": incompatible}, expected)


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
        with pytest.raises(ValueError, match="identity does not match"):
            load_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                context,
                expected_identity={"schema": "test", "tokenizer": {"sha256": "b" * 64}},
            )

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
