from __future__ import annotations

from pathlib import Path

import torch

from sion_translate.cli.train import (
    dataloader_runtime_kwargs,
    export_final_model,
    find_existing_checkpoint,
    release_stage_resources,
    shutdown_dataloader,
    tokenizer_policy_problem,
)
from sion_translate.config import AppConfig, config_from_raw
from sion_translate.fingerprint import file_sha256
from sion_translate.training.distributed import DistributedContext
from sion_translate.training.distributed import (
    fsdp_reduce_dtype,
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
