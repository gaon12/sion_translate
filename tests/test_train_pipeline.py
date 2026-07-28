from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sion_translate.cli.train import (
    dataloader_runtime_kwargs,
    export_final_model,
    find_existing_checkpoint,
    release_stage_resources,
    requires_ddp_unused_parameter_detection,
    shutdown_dataloader,
    tokenizer_policy_problem,
    validate_training_capacity,
)
from sion_translate.config import AppConfig, config_from_raw
from sion_translate.fingerprint import file_sha256
from sion_translate.training.distributed import DistributedContext
from sion_translate.training.distributed import (
    fsdp_reduce_dtype,
    initialize_distributed,
    parallelize_model,
    resolve_parallel_strategy,
)


class WorkerIterator:
    def __init__(self) -> None:
        self.stopped = False

    def _shutdown_workers(self) -> None:
        self.stopped = True


class LoaderStub:
    def __init__(self) -> None:
        self._iterator = WorkerIterator()


def test_dataloader_runtime_settings_separate_training_and_validation() -> None:
    device = torch.device("cuda")
    training = dataloader_runtime_kwargs(12, device, training=True)
    validation = dataloader_runtime_kwargs(3, device, training=False)
    single_process = dataloader_runtime_kwargs(0, torch.device("cpu"), training=True)

    assert training == {
        "num_workers": 12,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    assert validation == {
        "num_workers": 3,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
    assert single_process == {"num_workers": 0, "pin_memory": False}


def test_stage_release_stops_persistent_workers_on_cpu() -> None:
    loader = LoaderStub()
    iterator = loader._iterator
    shutdown_dataloader(loader)  # type: ignore[arg-type]
    assert iterator.stopped
    assert loader._iterator is None

    second = LoaderStub()
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        distributed=False,
    )
    assert release_stage_resources(context, second) == {}  # type: ignore[arg-type]
    assert second._iterator is None


def test_final_export_wires_all_formats_and_model_sidecars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = AppConfig()
    config.data.tokenizer_model = str(tmp_path / "sion.model")
    config.data.tokenizer_features = str(tmp_path / "token_features.npz")
    config.data.language_pairs = [["ko", "ja"], ["en", "ru"]]
    config.data.bidirectional = False
    config.data.revision_examples = True
    config.model.experimental.morphoscript_enabled = True
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def capture(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "sion_translate.cli.train.export_inference_models",
        capture,
    )
    destination = export_final_model(
        torch.nn.Linear(1, 1),
        config,
        context,
        tmp_path / "run",
        stage="posttrain",
        step=17,
    )

    assert destination == tmp_path / "run" / "posttrain" / "exports" / "best"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == destination
    assert kwargs["formats"] == tuple(config.training.final_export_formats)
    assert kwargs["tokenizer_path"] == config.data.tokenizer_model
    assert kwargs["token_features_path"] == config.data.tokenizer_features
    assert kwargs["language_pairs"] == (("ko", "ja"), ("en", "ru"))
    assert kwargs["bidirectional"] is False
    assert kwargs["revision_trained"] is True
    assert kwargs["strict"] is True


def test_parallel_strategy_prefers_ddp_and_supports_legacy_fsdp() -> None:
    distributed = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=4,
        device=torch.device("cuda"),
        distributed=True,
        backend="nccl",
    )
    single = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        distributed=False,
    )
    assert resolve_parallel_strategy("auto", distributed) == "ddp"
    assert resolve_parallel_strategy("auto", distributed, legacy_fsdp2=True) == "fsdp2"
    assert resolve_parallel_strategy("fsdp2", single) == "single"
    assert fsdp_reduce_dtype("auto", torch.bfloat16) == torch.bfloat16
    assert fsdp_reduce_dtype("auto", torch.float32) == torch.float32


def test_parallel_strategy_config_rejects_ambiguous_legacy_override() -> None:
    config = config_from_raw(
        {
            "training": {
                "parallel_strategy": "fsdp2",
                "fsdp_reduce_dtype": "bf16",
            }
        }
    )
    config.validate()
    assert config.training.parallel_strategy == "fsdp2"

    try:
        config_from_raw(
            {
                "training": {
                    "parallel_strategy": "ddp",
                    "fsdp2": True,
                }
            }
        )
    except ValueError as exc:
        assert "cannot both be set" in str(exc)
    else:
        raise AssertionError("ambiguous parallel settings must be rejected")


def test_cuda_multi_gpu_fails_before_process_group_without_nccl(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: False)
    initialized = False

    def record_initialization(**_kwargs: object) -> None:
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        record_initialization,
    )

    with pytest.raises(RuntimeError, match="requires the NCCL"):
        initialize_distributed()
    assert initialized is False


def test_fsdp2_registers_custom_generation_forward_methods(monkeypatch) -> None:
    from torch.distributed import fsdp as fsdp_api

    class GeneratingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(2, 2)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.projection(inputs)

        def generate(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.forward(inputs)

        def sample(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.forward(inputs)

    class Policy:
        def __init__(self, **_kwargs: object) -> None:
            pass

    model = GeneratingModel()
    events: list[tuple[str, object]] = []

    def fully_shard(module: torch.nn.Module, **_kwargs: object) -> None:
        events.append(("shard", module))

    def register(module: torch.nn.Module, method_name: str) -> None:
        events.append((method_name, module))

    monkeypatch.setattr(fsdp_api, "MixedPrecisionPolicy", Policy)
    monkeypatch.setattr(fsdp_api, "fully_shard", fully_shard)
    monkeypatch.setattr(fsdp_api, "register_fsdp_forward_method", register)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
        distributed=True,
        backend="gloo",
    )

    result = parallelize_model(
        model,
        context,
        strategy="fsdp2",
        precision="fp32",
        reshard_after_forward=True,
        materialize_meta=False,
    )

    assert result is model
    assert events == [
        ("shard", model),
        ("generate", model),
        ("sample", model),
    ]


def test_fsdp2_reports_missing_custom_forward_registration_api(
    monkeypatch,
) -> None:
    from torch.distributed import fsdp as fsdp_api

    monkeypatch.setattr(fsdp_api, "register_fsdp_forward_method", None)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
        distributed=True,
        backend="gloo",
    )

    with pytest.raises(RuntimeError, match="register_fsdp_forward_method"):
        parallelize_model(
            torch.nn.Linear(2, 2),
            context,
            strategy="fsdp2",
            precision="fp32",
            reshard_after_forward=True,
            materialize_meta=False,
        )


def test_ddp_unused_parameter_detection_covers_bats_stage_transition() -> None:
    config = AppConfig()
    config.model.experimental.bats_enabled = True
    config.model.experimental.bats_coverage_weight = 0.01
    config.posttraining.enabled = True
    assert requires_ddp_unused_parameter_detection(config) is True

    config.posttraining.enabled = False
    assert requires_ddp_unused_parameter_detection(config) is False

    config.model.experimental.bats_coverage_weight = 0.0
    assert requires_ddp_unused_parameter_detection(config) is True

    config.model.experimental.bats_enabled = False
    config.posttraining.enabled = True
    assert requires_ddp_unused_parameter_detection(config) is False


def test_h100_capacity_gate_requires_four_gpus_for_8b_and_sixteen_for_32b() -> None:
    def context(world_size: int) -> DistributedContext:
        return DistributedContext(
            rank=0,
            local_rank=0,
            world_size=world_size,
            device=torch.device("cuda"),
            distributed=True,
            backend="nccl",
        )

    with pytest.raises(RuntimeError, match="at least 4 GPUs"):
        validate_training_capacity(
            8_000_000_000,
            context(2),
            parallel_strategy="fsdp2",
            ema_enabled=True,
            per_gpu_vram_gib=80.0,
        )
    eight_billion = validate_training_capacity(
        8_000_000_000,
        context(4),
        parallel_strategy="fsdp2",
        ema_enabled=True,
        per_gpu_vram_gib=80.0,
    )
    assert eight_billion is not None
    assert eight_billion["per_rank_state_gib"] < eight_billion["state_budget_gib"]

    with pytest.raises(RuntimeError, match="at least 16 GPUs"):
        validate_training_capacity(
            32_083_082_800,
            context(8),
            parallel_strategy="fsdp2",
            ema_enabled=True,
            per_gpu_vram_gib=80.0,
        )
    thirty_two_billion = validate_training_capacity(
        32_083_082_800,
        context(16),
        parallel_strategy="fsdp2",
        ema_enabled=True,
        per_gpu_vram_gib=80.0,
    )
    assert thirty_two_billion is not None
    assert thirty_two_billion["minimum_world_size"] == 16


def test_existing_checkpoint_search_covers_stage_directories(tmp_path: Path) -> None:
    config = AppConfig()
    config.training.output_dir = str(tmp_path / "run")
    assert find_existing_checkpoint(config) is None

    checkpoint = tmp_path / "run" / "pretrain" / "checkpoints" / "latest"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.pt").write_bytes(b"weights")
    assert find_existing_checkpoint(config) == checkpoint


def test_tokenizer_policy_requires_digit_splitting_and_matching_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "sion.model"
    model_path.write_bytes(b"tokenizer")
    vocab_path = tmp_path / "sion.vocab"
    vocab_path.write_bytes(b"vocabulary")
    pairs = (("ko", "ja"),)

    class DigitTokenizer:
        splits_digits = True

    metadata = {
        "version": 2,
        "split_digits": True,
        "model_sha256": file_sha256(model_path),
        "vocab_sha256": file_sha256(vocab_path),
        "language_pairs": [["ko", "ja"]],
    }
    monkeypatch.setattr("sion_translate.cli.train.SionTokenizer", lambda _: DigitTokenizer())
    monkeypatch.setattr(
        "sion_translate.cli.train.load_tokenizer_metadata",
        lambda _: metadata,
    )
    monkeypatch.setattr(
        "sion_translate.cli.train.tokenizer_split_digits_policy",
        lambda _: True,
    )
    assert tokenizer_policy_problem(model_path, pairs) is None

    class MergedDigitTokenizer:
        splits_digits = False

    monkeypatch.setattr(
        "sion_translate.cli.train.SionTokenizer",
        lambda _: MergedDigitTokenizer(),
    )
    assert "split_digits=False" in str(tokenizer_policy_problem(model_path, pairs))
