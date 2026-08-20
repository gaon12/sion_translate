from __future__ import annotations

from copy import deepcopy
import hashlib
import random
from pathlib import Path
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
    dcp.save({"identity": identity}, checkpoint_id=checkpoint)

    checkpoint_module._preflight_dcp_identity(checkpoint, identity)
    checkpoint_module._preflight_dcp_stage_transfer(checkpoint, identity)
    metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    probe = checkpoint_module._dcp_identity_probe(metadata)
    dcp.load(probe, checkpoint_id=checkpoint)
    checkpoint_module._restore_expected_empty_mappings(probe["identity"], identity)

    assert probe == {"identity": identity}


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


# ── 단계 인계 (foundation → SFT) ────────────────────────────────────────


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
    """foundation → SFT 는 재개가 아니다.

    새 단계는 새 목적함수와 새 LR schedule 을 갖습니다. 이전 단계의 Adam
    moment 와 step 카운터를 이어받으면 warmup 이 건너뛰어지고 momentum 이
    다른 loss 표면의 것을 가리킵니다.
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
    # step 은 반환값으로만 알려 주고, 새 단계는 0 에서 시작한다.
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

    fresh, _, _, _ = _components()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    with pytest.warns(RuntimeWarning, match="legacy distributed checkpoint"):
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

    fresh, _, _, _ = _components()
    before = {name: tensor.clone() for name, tensor in fresh.state_dict().items()}
    context = DistributedContext(0, 0, 1, torch.device("cpu"), True, "gloo")
    with pytest.warns(RuntimeWarning, match="legacy distributed checkpoint"):
        with pytest.raises(ValueError, match="distributed checkpoint ema"):
            initialize_model_from_checkpoint(checkpoint, fresh, context)

    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


def test_stage_transfer_accepts_a_different_dataset(tmp_path: Path) -> None:
    """두 단계가 서로 다른 데이터셋을 쓰는 것은 정상이다 (단일어 대 병렬)."""
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
    """토크나이저가 다르면 임베딩 행이 가리키는 것이 달라진다.

    모양은 맞으므로 load_state_dict 는 성공합니다. 즉 막지 않으면 조용히
    무의미한 가중치를 물려받습니다.
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
    """검증할 수 없는 가중치를 물려받느니 멈추는 편이 낫다."""
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


# ── 목적함수 identity: 무엇을 최적화하는지도 정체성이다 ─────────────────


def _objective_config():
    from sion_translate.config import AppConfig

    return AppConfig()


def test_the_objective_identity_covers_the_optimizer_schedule() -> None:
    """학습률·Adam 계수를 바꾸고 optimizer state 를 이어받으면 momentum 이
    다른 곡률을 가리킨다."""
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
    ):
        changed = _objective_config()
        setattr(changed.training, field, value)
        assert build_objective_identity(changed.training) != baseline, field


def test_reward_weights_are_part_of_the_posttraining_identity() -> None:
    """MRT 는 reward 정의가 곧 선택 지표다.

    가중치 하나만 바꿔도 validation_reward 는 다른 축의 수치가 되는데
    early stopping 은 과거 best 와 비교합니다.
    """
    from sion_translate.training.checkpoint import build_objective_identity

    config = _objective_config()
    baseline = build_objective_identity(
        config.training, config.posttraining, include_posttraining=True
    )
    for field, value in (
        ("reward_chrf_weight", 0.1),
        ("roundtrip_enabled", True),
        ("samples_per_source", 8),
        ("validation_num_beams", 1),
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
    """SFT 재개를 MRT 설정 변경만으로 거부하면 안 된다."""
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
    """구버전 체크포인트의 재개를 막지는 않되, 검사하지 못한 것을 밝힌다."""
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
    assert any("목적함수" in str(entry.message) for entry in caught)
