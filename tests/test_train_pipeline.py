from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sion_translate.cli.train as train_module
import sion_translate.training.distributed as distributed_module
from sion_translate.cli.train import (
    build_collator_args,
    construct_training_model,
    dataloader_runtime_kwargs,
    export_final_model,
    find_existing_checkpoint,
    missing_final_export_dependencies,
    preflight_final_export_dependencies,
    release_stage_resources,
    requires_ddp_unused_parameter_detection,
    shutdown_dataloader,
    tokenizer_policy_problem,
    validate_training_capacity,
)
from sion_translate.config import (
    AppConfig,
    ExperimentalConfig,
    ModelConfig,
    config_from_raw,
)
from sion_translate.fingerprint import file_sha256
from sion_translate.model.transformer import SionForConditionalGeneration
from sion_translate.training.distributed import DistributedContext
from sion_translate.training.distributed import (
    distributed_precision_dtype,
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


def test_collator_pipeline_passes_configured_source_only_languages() -> None:
    config = config_from_raw(
        {
            "data": {
                "language_pairs": [
                    ["kj", "ko"],
                    ["kj", "ja"],
                    ["kd", "ko"],
                    ["jd", "ja"],
                    ["ko", "ja"],
                ],
                "source_only_languages": ["kj", "kd", "jd"],
            }
        }
    )
    tokenizer = SimpleNamespace()

    args = build_collator_args(config, tokenizer)  # type: ignore[arg-type]

    assert args["source_only_languages"] == ("kj", "kd", "jd")


def test_final_export_dependency_preflight_fails_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = {"torchao": False, "gguf": False}
    monkeypatch.setattr(
        train_module.importlib.util,
        "find_spec",
        lambda name: object() if available.get(name, True) else None,
    )

    assert missing_final_export_dependencies(["fp32", "transformers"]) == {}
    assert missing_final_export_dependencies(["int8", "gguf_q4_k_m"]) == {
        "int8": "torchao",
        "gguf_q4_k_m": "gguf-python",
    }
    with pytest.raises(RuntimeError, match=r"int8.*torchao.*gguf_q4_k_m.*gguf-python"):
        preflight_final_export_dependencies(["int8", "gguf_q4_k_m"])

    available["torchao"] = True
    available["gguf"] = True
    preflight_final_export_dependencies(["int8", "gguf_q4_k_m"])


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


def test_final_export_advertises_only_the_eight_trained_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig()
    config.data.language_pairs = [
        ["kj", "ko"],
        ["kj", "ja"],
        ["kd", "ko"],
        ["kd", "ja"],
        ["jd", "ko"],
        ["jd", "ja"],
        ["ko", "ja"],
    ]
    config.data.source_only_languages = ["kj", "kd", "jd"]
    config.data.bidirectional = True
    captured: dict[str, object] = {}

    def capture(*_args: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(train_module, "export_inference_models", capture)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    export_final_model(
        torch.nn.Linear(1, 1),
        config,
        context,
        tmp_path / "run",
        stage="posttrain",
        step=23,
    )

    assert captured["translation_directions"] == (
        ("kj", "ko"),
        ("kj", "ja"),
        ("kd", "ko"),
        ("kd", "ja"),
        ("jd", "ko"),
        ("jd", "ja"),
        ("ko", "ja"),
        ("ja", "ko"),
    )


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


def test_distributed_bf16_support_is_resolved_collectively(monkeypatch) -> None:
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cuda"),
        distributed=True,
        backend="nccl",
    )
    original_tensor = torch.tensor
    monkeypatch.setattr(
        distributed_module.torch,
        "tensor",
        lambda value, **kwargs: original_tensor(value, dtype=kwargs.get("dtype")),
    )
    monkeypatch.setattr(distributed_module.torch.cuda, "is_bf16_supported", lambda: True)

    monkeypatch.setattr(
        distributed_module.dist,
        "all_reduce",
        lambda value, **_kwargs: value.fill_(1),
    )
    assert distributed_precision_dtype("bf16", context) == torch.bfloat16

    monkeypatch.setattr(
        distributed_module.dist,
        "all_reduce",
        lambda value, **_kwargs: value.fill_(0),
    )
    with pytest.raises(RuntimeError, match="at least one distributed CUDA rank"):
        distributed_precision_dtype("bf16", context)


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

    class Policy:
        def __init__(self, **_kwargs: object) -> None:
            pass

    model = SionForConditionalGeneration(
        ModelConfig(
            vocab_size=32,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            num_heads=4,
            num_kv_heads=2,
            d_ff=32,
            max_seq_len=16,
            dropout=0.0,
        )
    )
    encoder_layer = model.encoder_layers[0]
    decoder_layer = model.decoder_layers[0]
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
    assert model._synchronize_generation_across_ranks is True
    assert events == [
        ("shard", encoder_layer),
        ("shard", decoder_layer),
        ("shard", model),
        ("project_cross_key_value", decoder_layer),
        ("forward_step", decoder_layer),
        ("generate", model),
        ("sample", model),
    ]


def test_fsdp2_cpu_gloo_forward_generate_and_sample_smoke(tmp_path: Path) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("CPU FSDP2 smoke test requires torch.distributed with Gloo")
    if torch.distributed.is_initialized():
        pytest.skip("test requires ownership of the default process group")

    rendezvous = (tmp_path / "fsdp2-rendezvous").resolve().as_uri()
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=0,
        world_size=1,
    )
    try:
        with torch.device("meta"):
            model = SionForConditionalGeneration(
                ModelConfig(
                    vocab_size=32,
                    d_model=16,
                    encoder_layers=1,
                    decoder_layers=1,
                    num_heads=4,
                    num_kv_heads=2,
                    d_ff=32,
                    max_seq_len=16,
                    dropout=0.0,
                    label_smoothing=0.0,
                    z_loss_weight=0.0,
                    experimental=ExperimentalConfig(
                        bats_enabled=True,
                        bats_coverage_weight=0.01,
                        core_enabled=True,
                        tetm_enabled=True,
                        morphoscript_enabled=True,
                        morphoscript_interval=1,
                        evidence_repair_enabled=True,
                        semantic_parity_enabled=True,
                    ),
                )
            )
        assert model.register_state is not None
        assert model.typed_memory is not None
        assert model.register_state.inject_gate.shape == (1,)
        assert model.typed_memory.gate.shape == (1,)

        context = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
            distributed=True,
            backend="gloo",
        )
        model = parallelize_model(
            model,
            context,
            strategy="fsdp2",
            precision="fp32",
            reduce_dtype="fp32",
            reshard_after_forward=True,
            materialize_meta=True,
        )
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        assert torch.count_nonzero(model.morph_gates) == 0
        assert torch.count_nonzero(model.evidence_repair.repair_scale) == 0
        assert torch.count_nonzero(model.register_state.inject_gate) == 0
        assert torch.count_nonzero(model.typed_memory.gate) == 0
        assert torch.count_nonzero(model.alignment_head.null_source) == 0

        input_ids = torch.tensor([[4, 5, 3]])
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        decoder_input_ids = torch.tensor([[2, 6, 7]])
        labels = torch.tensor([[6, 7, 3]])
        memory_inputs = {
            "memory_token_ids": torch.tensor([[[8, 9], [10, 0]]]),
            "memory_mask": torch.tensor([[True, True]]),
            "memory_type_ids": torch.tensor([[1, 8]]),
            "memory_mode_ids": torch.tensor([[1, 4]]),
        }
        morphoscript_inputs = {
            "src_script_ids": torch.zeros_like(input_ids),
            "src_onset_ids": torch.zeros_like(input_ids),
            "src_vowel_ids": torch.zeros_like(input_ids),
            "src_coda_ids": torch.zeros_like(input_ids),
        }

        output = model(
            input_ids,
            attention_mask,
            decoder_input_ids,
            labels,
            register_labels=torch.tensor([1]),
            **memory_inputs,
            **morphoscript_inputs,
        )
        assert output.loss is not None
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        assert model.register_state.inject_gate.grad is not None
        assert model.typed_memory.gate.grad is not None

        generated = model.generate(
            input_ids,
            attention_mask,
            bos_id=2,
            eos_id=3,
            max_new_tokens=3,
            num_beams=1,
            min_new_tokens=2,
            **memory_inputs,
            **morphoscript_inputs,
        )
        sampled = model.sample(
            input_ids,
            attention_mask,
            bos_id=2,
            eos_id=3,
            num_samples=2,
            max_new_tokens=3,
            min_new_tokens=2,
            **memory_inputs,
            **morphoscript_inputs,
        )
        assert generated.shape == (1, 4)
        assert sampled.shape == (1, 2, 4)
    finally:
        torch.distributed.destroy_process_group()


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


def test_ddp_unused_parameter_detection_covers_supervised_only_heads() -> None:
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

    config.model.experimental.semantic_parity_enabled = True
    assert requires_ddp_unused_parameter_detection(config) is True

    config.posttraining.enabled = False
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


def test_single_gpu_capacity_gate_runs_before_parameter_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction_devices: list[str] = []

    class OversizedModel(torch.nn.Module):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1))
            construction_devices.append(self.weight.device.type)

        @staticmethod
        def parameter_count() -> int:
            return 32_083_082_800

    monkeypatch.setattr(train_module, "SionForConditionalGeneration", OversizedModel)
    monkeypatch.setattr(
        train_module.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=80 * 2**30),
    )
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cuda"),
        distributed=False,
    )
    with pytest.raises(RuntimeError, match="Switch to FSDP2"):
        construct_training_model(
            AppConfig(),
            context,
            pad_id=0,
            parallel_strategy="single",
        )
    assert construction_devices == ["meta"]


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
        languages = ("ko", "ja")

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

    class WrongLanguageTokenizer:
        splits_digits = True
        languages = ("ko",)

    monkeypatch.setattr(
        "sion_translate.cli.train.SionTokenizer",
        lambda _: WrongLanguageTokenizer(),
    )
    assert "언어 집합" in str(tokenizer_policy_problem(model_path, pairs))
