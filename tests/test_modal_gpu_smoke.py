from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.model import SionForConditionalGeneration


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "modal_gpu_smoke.py"
SPEC = importlib.util.spec_from_file_location("modal_gpu_smoke_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _distributed_rank_report(rank: int, **overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "rank": rank,
        "world_size": 2,
        "backend": "nccl",
        "norm": 0.75,
        "loss": 1.25,
        "parameter_count": MODULE.DISTRIBUTED_CANARY_PARAMETER_COUNT,
        "candidate_refinement_steps": 1,
        "candidate_refinement_loss": 1.3,
        "candidate_refinement_gain": 0.01,
        "candidate_refinement_token_nll_gain": 0.01,
        "candidate_refinement_gradient_count": 6,
        "candidate_refinement_update": True,
        "optimizer_state_count": 12,
        "checkpoint_step": 1,
        "checkpoint_roundtrip": True,
        "ema_roundtrip": True,
        "optimizer_roundtrip": True,
        "scheduler_roundtrip": True,
        "rng_roundtrip": True,
        "device_name": "NVIDIA A100-SXM4-40GB",
        "peak_allocated_gib": 1.5 + rank,
        "peak_reserved_gib": 2.0 + rank,
    }
    report.update(overrides)
    return report


def _runtime_device(target: str, index: int) -> dict[str, object]:
    if target.startswith("a100"):
        return {
            "index": index,
            "name": ("NVIDIA A100-SXM4-80GB" if target == "a100-80gb" else "NVIDIA A100-SXM4-40GB"),
            "capability": [8, 0],
            "total_memory_gib": 79.2 if target == "a100-80gb" else 39.4,
            "bf16_native": True,
        }
    return {
        "index": index,
        "name": "NVIDIA H100 80GB HBM3",
        "capability": [9, 0],
        "total_memory_gib": 79.2,
        "bf16_native": True,
    }


def _distributed_phase() -> dict[str, object]:
    ranks = [_distributed_rank_report(rank) for rank in (0, 1)]
    return {
        "elapsed_seconds": 0.5,
        "report": {
            "ranks": ranks,
            "max_peak_allocated_gib": 2.5,
            "max_peak_reserved_gib": 3.0,
        },
        "stdout_tail": "rank reports",
        "stderr_tail": "",
    }


def _remote_result(target: str = "a100-40gb") -> dict[str, object]:
    elapsed = 12.25
    phases: dict[str, object] = {
        "attention_optimizer": {"elapsed_seconds": 0.5, "loss": 0.25},
        "production_model": {
            "elapsed_seconds": 11.0,
            "parameter_count": 287_127_073,
            "dropout": 0.1,
            "gradient_checkpointing": target in {"a100-40gb", "a100-40gb-x2"},
            "loss": 4.0,
            "gradient_norm": 0.5,
            "candidate_refinement_loss": 4.1,
            "candidate_refinement_steps": 1,
            "candidate_refinement_gain": 0.01,
            "candidate_refinement_token_nll_gain": 0.01,
            "candidate_refinement_scale_gradient": 0.02,
            "candidate_refinement_scale": -0.0003,
            "candidate_refinement_gradient_count": 6,
            "candidate_refinement_update": True,
            "cached_refinement_calls": 3,
            "generated_shape": [1, 4],
            "checkpoint_bytes": 4_000_000_000,
            "checkpoint_save_seconds": 2.0,
            "checkpoint_load_seconds": 2.0,
            "ema_decay": 0.999,
            "ema_roundtrip": True,
            "optimizer_roundtrip": True,
            "scheduler_roundtrip": True,
            "rng_roundtrip": True,
        },
    }
    if target == "a100-40gb-x2":
        phases["distributed_fsdp2"] = _distributed_phase()
    return {
        "status": "passed",
        "target": target,
        "runtime": {
            "python": "3.11.9",
            "python_implementation": "CPython",
            "libc": {"name": "glibc", "version": "2.36"},
            "versions": dict(MODULE.EXPECTED_RUNTIME_VERSIONS),
            "cuda_runtime": "12.8",
            "cudnn": MODULE.EXPECTED_CUDNN_VERSION,
            "nccl_available": True,
            "nccl_version": list(MODULE.EXPECTED_NCCL_VERSION),
            "devices": [
                _runtime_device(target, index)
                for index in range(int(MODULE.TARGETS[target]["gpu_count"]))
            ],
            "lock_sha256": MODULE.LOCK_SHA256,
        },
        "phases": phases,
        "elapsed_seconds": elapsed,
        "estimated_function_compute_charge_usd": MODULE.estimated_function_compute_charge(
            target, elapsed
        ),
        "authorization_compute_charge_usd": MODULE.authorization_compute_charge(target),
        "peak_allocated_gib": 8.0,
        "peak_reserved_gib": 9.0,
    }


def test_target_allowlist_uses_exact_modal_gpu_names() -> None:
    assert MODULE.TARGETS == {
        "a100-40gb": {
            "gpu": "A100-40GB",
            "gpu_count": 1,
            "usd_per_second": 0.000583,
        },
        "a100-80gb": {
            "gpu": "A100-80GB",
            "gpu_count": 1,
            "usd_per_second": 0.000694,
        },
        "h100": {"gpu": "H100!", "gpu_count": 1, "usd_per_second": 0.001097},
        "a100-40gb-x2": {
            "gpu": "A100-40GB:2",
            "gpu_count": 2,
            "usd_per_second": 0.000583,
        },
    }


def test_cli_requires_exactly_one_known_target_and_an_authorization_threshold() -> None:
    parser = MODULE.build_parser()
    parsed = parser.parse_args(["--target", "h100", "--max-dollars", "1.10"])
    assert parsed.target == "h100"
    assert parsed.max_dollars == 1.10

    with pytest.raises(SystemExit):
        parser.parse_args(["--max-dollars", "1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--target", "all", "--max-dollars", "1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--target", "h100"])


def test_cost_guard_uses_startup_function_recovery_and_gpu_count() -> None:
    assert MODULE.authorization_compute_charge("a100-40gb") == pytest.approx(0.68100816)
    assert MODULE.authorization_compute_charge("a100-80gb") == pytest.approx(0.78801216)
    assert MODULE.authorization_compute_charge("h100") == pytest.approx(1.17650416)
    assert MODULE.authorization_compute_charge("a100-40gb-x2") == pytest.approx(1.24302016)
    assert MODULE.validate_cost_guard("h100", 1.18) == pytest.approx(1.17650416)

    with pytest.raises(ValueError, match="exceeds"):
        MODULE.validate_cost_guard("h100", 1.17)
    for invalid in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite positive"):
            MODULE.validate_cost_guard("h100", invalid)


def test_function_body_cost_estimate_rounds_up_and_includes_idle_window() -> None:
    rate = 0.000583 + 4 * 0.0000131 + 32 * 0.00000222
    assert MODULE.estimated_function_compute_charge("a100-40gb", 0.01) == pytest.approx(3 * rate)
    assert MODULE.estimated_function_compute_charge("a100-40gb", 1.01) == pytest.approx(4 * rate)
    assert MODULE.estimated_function_compute_charge("a100-40gb", 999) == pytest.approx(
        (999 + MODULE.SCALEDOWN_WINDOW_SECONDS) * rate
    )
    for invalid in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="observed elapsed"):
            MODULE.estimated_function_compute_charge("a100-40gb", invalid)


def test_modal_functions_are_timeout_configured_without_launching_gpu() -> None:
    if MODULE.modal is None:
        pytest.skip("Modal client is not installed")
    assert MODULE.FUNCTION_TIMEOUT_SECONDS == 300
    assert MODULE.STARTUP_TIMEOUT_SECONDS == 180
    assert MODULE.CHILD_TIMEOUT_SECONDS == 150
    assert MODULE.PARENT_CLEANUP_MARGIN_SECONDS == 20
    assert MODULE._common_options == {
        "retries": 0,
        "timeout": 300,
        "startup_timeout": 180,
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "scaledown_window": 2,
        "single_use_containers": True,
        "cpu": 4.0,
        "memory": 32_768,
        "ephemeral_disk": 16_384,
        "volumes": {str(MODULE.RESULT_MOUNT): MODULE.result_volume},
        "include_source": True,
    }


def test_local_modal_client_version_is_reviewed() -> None:
    if MODULE.modal is None:
        pytest.skip("Modal client is not installed")
    MODULE._validate_modal_client_version()
    assert MODULE.importlib.metadata.version("modal") == MODULE.EXPECTED_MODAL_CLIENT_VERSION


def test_unreviewed_modal_client_is_rejected_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE.importlib.metadata, "version", lambda _package: "9.9.9")

    with pytest.raises(RuntimeError, match="requires local Modal client"):
        MODULE._validate_modal_client_version()


def test_authenticated_bootstrap_and_gpu_lock_are_pinned() -> None:
    lock = MODULE.REPOSITORY_ROOT / MODULE.LOCK_RELATIVE_PATH
    assert MODULE._sha256(lock) == MODULE.LOCK_SHA256
    bootstrap = (MODULE.REPOSITORY_ROOT / MODULE.UV_BOOTSTRAP_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert "uv==0.12.3" in bootstrap
    assert "sha256:1482d1462b1aecd18ee33627363fe1c63d6a194f12d40d37efc446d9e0d800a1" in bootstrap


def test_documented_commands_cover_current_authorization_values() -> None:
    documentation = (MODULE.REPOSITORY_ROOT / "docs" / "MODAL_GPU_SMOKE.md").read_text(
        encoding="utf-8"
    )
    for target, ceiling in {
        "a100-40gb": "0.69",
        "a100-80gb": "0.80",
        "h100": "1.19",
        "a100-40gb-x2": "1.25",
    }.items():
        assert f"--target {target} --max-dollars {ceiling}" in documentation
        assert MODULE.validate_cost_guard(target, float(ceiling)) <= float(ceiling)
    assert "not a Modal account-level\nspending cap" in documentation


@pytest.mark.parametrize("target", tuple(MODULE.TARGETS))
def test_production_model_config_has_exact_reviewed_architecture(target: str) -> None:
    expected = ModelConfig(
        vocab_size=48_000,
        d_model=864,
        encoder_layers=18,
        decoder_layers=9,
        num_heads=12,
        num_kv_heads=6,
        d_ff=2_304,
        max_seq_len=2_048,
        dropout=0.1,
        gradient_checkpointing=target in {"a100-40gb", "a100-40gb-x2"},
        experimental=ExperimentalConfig(
            candidate_refinement_enabled=True,
            candidate_refinement_steps=1,
            candidate_refinement_temperature=1.0,
            candidate_refinement_loss_weight=0.25,
            candidate_refinement_vocab_chunk_size=2_048,
        ),
    )
    actual = MODULE._production_config(target)
    assert actual == expected


def test_production_parameter_count_matches_reviewed_capacity() -> None:
    actual = MODULE._production_config("h100")
    with torch.device("meta"):
        model = SionForConditionalGeneration(actual)
    assert model.parameter_count() == MODULE.PRODUCTION_PARAMETER_COUNT


def test_distributed_canary_shape_matches_reviewed_capacity() -> None:
    config = ModelConfig(
        vocab_size=128,
        d_model=72,
        encoder_layers=2,
        decoder_layers=1,
        num_heads=6,
        num_kv_heads=3,
        d_ff=192,
        max_seq_len=32,
        dropout=0.1,
        gradient_checkpointing=True,
        experimental=ExperimentalConfig(candidate_refinement_enabled=True),
    )
    with torch.device("meta"):
        model = SionForConditionalGeneration(config)
    assert model.parameter_count() == MODULE.DISTRIBUTED_CANARY_PARAMETER_COUNT
    assert len(list(model.candidate_refinement.named_parameters())) == (
        MODULE.EXPECTED_REFINEMENT_GRADIENT_COUNT
    )


def test_distributed_checkpoint_load_template_declares_custom_progress_key() -> None:
    assert MODULE._distributed_smoke_training_state_template() == {"modal_distributed_smoke": False}


def test_single_gpu_checkpoint_load_template_declares_custom_progress_key() -> None:
    assert MODULE._single_gpu_smoke_training_state_template() == {"smoke_complete": False}


def test_optimizer_checkpoint_probe_corrupts_a_non_scalar_adam_moment() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    state_name, saved = MODULE._first_optimizer_state_tensor(torch, optimizer)
    corrupted_name, corrupted = MODULE._corrupt_first_optimizer_state_tensor(torch, optimizer)

    assert state_name == corrupted_name
    assert state_name.split(":", 1)[1] in MODULE.OPTIMIZER_MOMENT_NAMES
    assert state_name.split(":", 1)[1] != "step"
    assert saved.numel() > 1
    assert not torch.equal(saved, corrupted)


def test_token_refinement_gain_reduces_only_target_tokens() -> None:
    labels = torch.tensor([[7, 8, -100], [9, -100, -100]])
    token_gain = torch.tensor([[0.3, 0.6, 0.0], [0.9, 0.0, 0.0]])

    mean_gain = MODULE._mean_refinement_token_nll_gain(
        torch,
        token_gain,
        labels,
        "test refinement",
    )

    assert mean_gain == pytest.approx(0.6)

    with pytest.raises(RuntimeError, match="shape"):
        MODULE._mean_refinement_token_nll_gain(
            torch,
            token_gain[:, :2],
            labels,
            "test refinement",
        )
    invalid_padding = token_gain.clone()
    invalid_padding[0, 2] = 1.0
    with pytest.raises(RuntimeError, match="outside target tokens"):
        MODULE._mean_refinement_token_nll_gain(
            torch,
            invalid_padding,
            labels,
            "test refinement",
        )
    invalid_finite = token_gain.clone()
    invalid_finite[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        MODULE._mean_refinement_token_nll_gain(
            torch,
            invalid_finite,
            labels,
            "test refinement",
        )


def test_distributed_report_requires_finite_evidence_from_both_ranks() -> None:
    stdout = "torchrun diagnostic\n" + "\n".join(
        json.dumps(_distributed_rank_report(rank), allow_nan=False) for rank in (0, 1)
    )
    report = MODULE._validated_distributed_report(stdout)
    assert [rank["rank"] for rank in report["ranks"]] == [0, 1]
    assert report["max_peak_allocated_gib"] == pytest.approx(2.5)
    assert report["max_peak_reserved_gib"] == pytest.approx(3.0)

    invalid_reports = (
        [_distributed_rank_report(0)],
        [_distributed_rank_report(0), _distributed_rank_report(0)],
        [_distributed_rank_report(0), _distributed_rank_report(1, backend="gloo")],
        [_distributed_rank_report(0), _distributed_rank_report(1, norm=float("nan"))],
        [
            _distributed_rank_report(0),
            _distributed_rank_report(1, checkpoint_roundtrip=False),
        ],
        [_distributed_rank_report(0), _distributed_rank_report(1, ema_roundtrip=False)],
        [_distributed_rank_report(0), _distributed_rank_report(1, optimizer_roundtrip=False)],
        [_distributed_rank_report(0), _distributed_rank_report(1, scheduler_roundtrip=False)],
        [_distributed_rank_report(0), _distributed_rank_report(1, rng_roundtrip=False)],
        [
            _distributed_rank_report(0),
            _distributed_rank_report(1, candidate_refinement_steps=0),
        ],
        [
            _distributed_rank_report(0),
            _distributed_rank_report(1, candidate_refinement_gradient_count=8),
        ],
        [
            _distributed_rank_report(0),
            _distributed_rank_report(1, candidate_refinement_update=False),
        ],
        [
            _distributed_rank_report(0),
            _distributed_rank_report(1, candidate_refinement_token_nll_gain=0.5),
        ],
        [_distributed_rank_report(0), _distributed_rank_report(1, device_name="H100")],
    )
    for reports in invalid_reports:
        output = "\n".join(json.dumps(item) for item in reports)
        with pytest.raises(RuntimeError, match="distributed smoke"):
            MODULE._validated_distributed_report(output)


def test_parent_deadline_reserves_cleanup_time() -> None:
    assert MODULE._remaining_child_timeout(0.0) == 150.0
    assert MODULE._remaining_child_timeout(200.0) == 80.0
    with pytest.raises(TimeoutError, match="not enough parent"):
        MODULE._remaining_child_timeout(251.0)
    for invalid in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="parent elapsed"):
            MODULE._remaining_child_timeout(invalid)


def test_two_gpu_timeout_terminates_the_complete_child_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminated: list[object] = []

    class TimedOutProcess:
        returncode = None

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            raise subprocess.TimeoutExpired("torchrun", timeout or 0.0)

    process = TimedOutProcess()

    def launch(*_args: Any, **_kwargs: Any) -> TimedOutProcess:
        return process

    def terminate(candidate: object) -> None:
        terminated.append(candidate)

    monkeypatch.setattr(MODULE.subprocess, "Popen", launch)
    monkeypatch.setattr(MODULE, "_terminate_child_group", terminate)

    with pytest.raises(TimeoutError, match="exceeded"):
        MODULE._two_gpu_canary(45.0)

    assert terminated == [process]


def test_two_gpu_spawn_failure_restores_the_parent_signal_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited_mask = {signal.SIGTERM}
    restored_masks: list[set[signal.Signals] | None] = []
    monkeypatch.setattr(MODULE, "_block_guarded_child_signals", lambda: inherited_mask)
    monkeypatch.setattr(MODULE, "_restore_guarded_child_signals", restored_masks.append)
    monkeypatch.setattr(
        MODULE.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected spawn failure")),
    )

    with pytest.raises(OSError, match="injected spawn failure"):
        MODULE._two_gpu_canary(45.0)

    assert restored_masks == [inherited_mask]


def test_two_gpu_signal_mask_restore_failure_terminates_spawned_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminated: list[object] = []

    class LiveProcess:
        returncode = None

    process = LiveProcess()
    inherited_mask = {signal.SIGTERM}

    def launch(*_args: Any, **_kwargs: Any) -> LiveProcess:
        return process

    def fail_restore(_mask: set[signal.Signals] | None) -> None:
        raise RuntimeError("injected restore failure")

    monkeypatch.setattr(MODULE, "_block_guarded_child_signals", lambda: inherited_mask)
    monkeypatch.setattr(MODULE.subprocess, "Popen", launch)
    monkeypatch.setattr(MODULE, "_restore_guarded_child_signals", fail_restore)
    monkeypatch.setattr(MODULE, "_terminate_child_group", terminated.append)

    with pytest.raises(RuntimeError, match="injected restore failure"):
        MODULE._two_gpu_canary(45.0)

    assert terminated == [process]


def test_two_gpu_success_validates_both_rank_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    restored_masks: list[set[signal.Signals] | None] = []
    stdout = "\n".join(
        json.dumps(_distributed_rank_report(rank), allow_nan=False) for rank in (0, 1)
    )

    class SuccessfulProcess:
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            observed["timeout"] = timeout
            return stdout, "warning"

    def launch(command: list[str], **kwargs: Any) -> SuccessfulProcess:
        observed["command"] = command
        observed["options"] = kwargs
        return SuccessfulProcess()

    monkeypatch.setattr(MODULE.subprocess, "Popen", launch)
    inherited_mask = {signal.SIGTERM}
    monkeypatch.setattr(MODULE, "_block_guarded_child_signals", lambda: inherited_mask)
    monkeypatch.setattr(MODULE, "_restore_guarded_child_signals", restored_masks.append)
    result = MODULE._two_gpu_canary(45.0)

    assert result["report"]["ranks"][1]["rank"] == 1
    assert result["elapsed_seconds"] > 0.0
    assert observed["timeout"] == 45.0
    options = cast(dict[str, Any], observed["options"])
    assert options["start_new_session"] is True
    assert options["env"]["SION_DISTRIBUTED_TIMEOUT_SECONDS"] == "60"
    assert "SION_MODAL_CHECKPOINT_DIR" in options["env"]
    assert options["env"][MODULE.EXPECTED_PARENT_PID_ENVIRONMENT] == str(MODULE.os.getpid())
    assert options["env"][MODULE.INHERITED_SIGNAL_MASK_ENVIRONMENT] == str(int(signal.SIGTERM))
    assert str(MODULE.REMOTE_ROOT) in options["env"]["PYTHONPATH"]
    command = cast(list[str], observed["command"])
    assert "--nproc-per-node=2" in command
    assert "--max-restarts=0" in command
    assert command.count("sion_translate.process_guard") == 2
    assert "launcher" in command
    assert "worker" in command
    assert "scripts.modal_gpu_smoke" in command
    assert restored_masks == [inherited_mask]


@pytest.mark.parametrize("target", tuple(MODULE.TARGETS))
def test_remote_success_result_requires_exact_finite_evidence(target: str) -> None:
    result = _remote_result(target)
    assert MODULE._validated_remote_result(target, result) == result

    invalid = dict(result)
    invalid["target"] = "a100-40gb" if target == "h100" else "h100"
    with pytest.raises(RuntimeError, match="mismatched success"):
        MODULE._validated_remote_result(target, invalid)

    invalid = dict(result)
    invalid["elapsed_seconds"] = float("nan")
    with pytest.raises(RuntimeError, match="not finite"):
        MODULE._validated_remote_result(target, invalid)

    invalid = dict(result)
    invalid["authorization_compute_charge_usd"] = 0.0
    with pytest.raises(RuntimeError, match="top-level metrics"):
        MODULE._validated_remote_result(target, invalid)


def test_remote_result_rejects_hostile_runtime_envelopes() -> None:
    invalid_results: list[dict[str, object]] = []
    runtime_cases: tuple[tuple[str, object], ...] = (
        ("python", "3.10.14"),
        ("python_implementation", "PyPy"),
        ("versions", {}),
        ("cuda_runtime", "12.7"),
        ("cudnn", 1),
        ("nccl_available", False),
        ("nccl_version", [2, 26, 0]),
        ("lock_sha256", "0" * 64),
    )
    for field, bad_value in runtime_cases:
        invalid = deepcopy(_remote_result())
        cast(dict[str, object], invalid["runtime"])[field] = bad_value
        invalid_results.append(invalid)

    invalid = deepcopy(_remote_result())
    cast(dict[str, object], invalid["runtime"])["libc"] = {
        "name": "glibc",
        "version": "2.27",
    }
    invalid_results.append(invalid)

    for field, bad_value in (
        ("index", True),
        ("name", "NVIDIA H200"),
        ("capability", [9, 0]),
        ("total_memory_gib", -1.0),
        ("bf16_native", False),
    ):
        invalid = deepcopy(_remote_result())
        runtime = cast(dict[str, object], invalid["runtime"])
        device = cast(list[dict[str, object]], runtime["devices"])[0]
        device[field] = bad_value
        invalid_results.append(invalid)

    invalid = deepcopy(_remote_result())
    runtime = cast(dict[str, object], invalid["runtime"])
    device = cast(list[dict[str, object]], runtime["devices"])[0]
    del device["index"]
    invalid_results.append(invalid)

    invalid = deepcopy(_remote_result())
    runtime = cast(dict[str, object], invalid["runtime"])
    device = cast(list[dict[str, object]], runtime["devices"])[0]
    device["name"] = "NVIDIA H200 A100"
    invalid_results.append(invalid)

    invalid = deepcopy(_remote_result("a100-80gb"))
    runtime = cast(dict[str, object], invalid["runtime"])
    device = cast(list[dict[str, object]], runtime["devices"])[0]
    device["total_memory_gib"] = 1e100
    with pytest.raises(RuntimeError):
        MODULE._validated_remote_result("a100-80gb", invalid)

    for invalid in invalid_results:
        with pytest.raises(RuntimeError):
            MODULE._validated_remote_result("a100-40gb", invalid)


def test_remote_result_rejects_hostile_attention_and_production_evidence() -> None:
    invalid_results: list[dict[str, object]] = []
    for phase_name, field, bad_value in (
        ("attention_optimizer", "elapsed_seconds", -1.0),
        ("attention_optimizer", "loss", "0.25"),
        ("production_model", "elapsed_seconds", 0.0),
        ("production_model", "loss", -1.0),
        ("production_model", "gradient_norm", 0.0),
        ("production_model", "candidate_refinement_loss", "4.1"),
        ("production_model", "candidate_refinement_steps", 0),
        ("production_model", "candidate_refinement_scale_gradient", 0.0),
        ("production_model", "candidate_refinement_gradient_count", True),
        ("production_model", "candidate_refinement_gradient_count", 8),
        ("production_model", "candidate_refinement_update", False),
        ("production_model", "candidate_refinement_token_nll_gain", 0.5),
        ("production_model", "checkpoint_bytes", True),
        ("production_model", "checkpoint_bytes", 1),
        ("production_model", "checkpoint_save_seconds", 0.0),
        ("production_model", "checkpoint_load_seconds", 0.0),
        ("production_model", "generated_shape", [1, 5]),
        ("production_model", "generated_shape", [True, 2]),
        ("production_model", "optimizer_roundtrip", False),
    ):
        invalid = deepcopy(_remote_result())
        phases = cast(dict[str, object], invalid["phases"])
        phase = cast(dict[str, object], phases[phase_name])
        phase[field] = bad_value
        invalid_results.append(invalid)

    invalid = deepcopy(_remote_result())
    phases = cast(dict[str, object], invalid["phases"])
    production = cast(dict[str, object], phases["production_model"])
    del production["candidate_refinement_token_nll_gain"]
    invalid_results.append(invalid)

    invalid = deepcopy(_remote_result())
    phases = cast(dict[str, object], invalid["phases"])
    attention = cast(dict[str, object], phases["attention_optimizer"])
    attention["unreviewed"] = True
    invalid_results.append(invalid)

    for invalid in invalid_results:
        with pytest.raises(RuntimeError, match="Modal GPU smoke"):
            MODULE._validated_remote_result("a100-40gb", invalid)


def test_remote_result_rejects_hostile_distributed_evidence() -> None:
    invalid_results: list[dict[str, object]] = []

    invalid = deepcopy(_remote_result("a100-40gb-x2"))
    phases = cast(dict[str, object], invalid["phases"])
    distributed = cast(dict[str, object], phases["distributed_fsdp2"])
    report = cast(dict[str, object], distributed["report"])
    report["ranks"] = []
    invalid_results.append(invalid)

    invalid = deepcopy(_remote_result("a100-40gb-x2"))
    phases = cast(dict[str, object], invalid["phases"])
    distributed = cast(dict[str, object], phases["distributed_fsdp2"])
    report = cast(dict[str, object], distributed["report"])
    report["max_peak_reserved_gib"] = 99.0
    invalid_results.append(invalid)

    invalid = deepcopy(_remote_result("a100-40gb-x2"))
    phases = cast(dict[str, object], invalid["phases"])
    distributed = cast(dict[str, object], phases["distributed_fsdp2"])
    report = cast(dict[str, object], distributed["report"])
    ranks = cast(list[dict[str, object]], report["ranks"])
    ranks[1]["optimizer_roundtrip"] = False
    invalid_results.append(invalid)

    invalid = deepcopy(_remote_result("a100-40gb-x2"))
    phases = cast(dict[str, object], invalid["phases"])
    distributed = cast(dict[str, object], phases["distributed_fsdp2"])
    del distributed["stderr_tail"]
    invalid_results.append(invalid)

    invalid = deepcopy(_remote_result("a100-40gb-x2"))
    phases = cast(dict[str, object], invalid["phases"])
    distributed = cast(dict[str, object], phases["distributed_fsdp2"])
    distributed["elapsed_seconds"] = -1.0
    invalid_results.append(invalid)

    invalid = deepcopy(_remote_result("a100-40gb-x2"))
    phases = cast(dict[str, object], invalid["phases"])
    distributed = cast(dict[str, object], phases["distributed_fsdp2"])
    report = cast(dict[str, object], distributed["report"])
    ranks = cast(list[dict[str, object]], report["ranks"])
    for rank in ranks:
        rank["peak_allocated_gib"] = 40.0
        rank["peak_reserved_gib"] = 41.0
    report["max_peak_allocated_gib"] = 40.0
    report["max_peak_reserved_gib"] = 41.0
    invalid_results.append(invalid)

    invalid = deepcopy(_remote_result("a100-40gb-x2"))
    phases = cast(dict[str, object], invalid["phases"])
    distributed = cast(dict[str, object], phases["distributed_fsdp2"])
    report = cast(dict[str, object], distributed["report"])
    ranks = cast(list[dict[str, object]], report["ranks"])
    ranks[1]["device_name"] = "NVIDIA A100-PCIE-40GB"
    invalid_results.append(invalid)

    for invalid in invalid_results:
        with pytest.raises(RuntimeError, match="distributed"):
            MODULE._validated_remote_result("a100-40gb-x2", invalid)


def test_remote_result_rejects_inconsistent_top_level_metrics() -> None:
    invalid_results: list[dict[str, object]] = []
    for field, bad_value in (
        ("elapsed_seconds", "12.25"),
        ("estimated_function_compute_charge_usd", -1.0),
        ("peak_allocated_gib", 0.0),
        ("peak_reserved_gib", 0.0),
        ("peak_allocated_gib", 10.0),
        ("peak_reserved_gib", 7.0),
    ):
        invalid = deepcopy(_remote_result())
        invalid[field] = bad_value
        invalid_results.append(invalid)
    for invalid in invalid_results:
        with pytest.raises(RuntimeError, match="Modal GPU smoke"):
            MODULE._validated_remote_result("a100-40gb", invalid)

    invalid = deepcopy(_remote_result())
    invalid["elapsed_seconds"] = 11.0
    invalid["estimated_function_compute_charge_usd"] = MODULE.estimated_function_compute_charge(
        "a100-40gb", 11.0
    )
    with pytest.raises(RuntimeError, match="inconsistent timing"):
        MODULE._validated_remote_result("a100-40gb", invalid)

    invalid = deepcopy(_remote_result())
    phases = cast(dict[str, object], invalid["phases"])
    production = cast(dict[str, object], phases["production_model"])
    production["checkpoint_save_seconds"] = 6.0
    production["checkpoint_load_seconds"] = 6.0
    with pytest.raises(RuntimeError, match="inconsistent timing"):
        MODULE._validated_remote_result("a100-40gb", invalid)


def test_gpu_smoke_contract_hashes_every_reviewed_runtime_byte(tmp_path: Path) -> None:
    (tmp_path / MODULE.LOCK_RELATIVE_PATH).parent.mkdir(parents=True)
    (tmp_path / MODULE.LOCK_RELATIVE_PATH).write_text("lock\n", encoding="utf-8")
    (tmp_path / MODULE.UV_BOOTSTRAP_RELATIVE_PATH).write_text("bootstrap\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "modal_gpu_smoke.py").write_text("print('smoke')\n", encoding="utf-8")
    (tmp_path / "src" / "package").mkdir(parents=True)
    source = tmp_path / "src" / "package" / "runtime.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    first = MODULE.gpu_smoke_contract_sha256(tmp_path)
    assert MODULE.SHA256_PATTERN.fullmatch(first)
    assert MODULE.gpu_smoke_contract_sha256(tmp_path) == first

    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert MODULE.gpu_smoke_contract_sha256(tmp_path) != first


def test_executed_entrypoint_must_match_reviewed_image_copy(tmp_path: Path) -> None:
    executed = tmp_path / "mounted" / "modal_gpu_smoke.py"
    reviewed = tmp_path / "image" / "modal_gpu_smoke.py"
    executed.parent.mkdir()
    reviewed.parent.mkdir()
    executed.write_text("VALUE = 1\n", encoding="utf-8")
    reviewed.write_text("VALUE = 1\n", encoding="utf-8")

    MODULE._verify_executed_entrypoint(executed, reviewed)
    reviewed.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="differs from its reviewed image copy"):
        MODULE._verify_executed_entrypoint(executed, reviewed)


def test_durable_journal_commits_progress_and_final_result(tmp_path: Path) -> None:
    commits: list[int] = []
    result = _remote_result()
    journal = MODULE._DurableRunJournal(
        tmp_path,
        run_id="smoke-20260901t120000z-0123456789abcdef",
        target="a100-40gb",
        function_call_id="fc-0123456789abcdef",
        max_dollars=1.0,
        expected_contract_sha256="a" * 64,
        commit=lambda: commits.append(len(commits) + 1),
    )

    journal.started()
    journal.contract_verified("a" * 64)
    journal.phase_completed("attention_optimizer")
    journal.phase_completed("production_model")
    journal.passed(result)

    run_root = tmp_path / "runs" / "smoke-20260901t120000z-0123456789abcdef"
    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    assert commits == [1, 2, 3, 4, 5]
    assert status["state"] == "passed"
    assert status["sequence"] == 5
    assert status["function_call_id"] == "fc-0123456789abcdef"
    assert status["completed_phases"] == ["attention_optimizer", "production_model"]
    assert json.loads((run_root / "result.json").read_text(encoding="utf-8")) == result
    assert len(list((run_root / "events").glob("*.json"))) == 5
    assert not list(run_root.rglob("*.tmp"))


def test_durable_failure_removes_prior_terminal_artifacts(tmp_path: Path) -> None:
    journal = MODULE._DurableRunJournal(
        tmp_path,
        run_id="smoke-20260901t120000z-fedcba9876543210",
        target="a100-40gb",
        function_call_id="fc-fedcba9876543210",
        max_dollars=1.0,
        expected_contract_sha256="b" * 64,
        commit=lambda: None,
    )
    run_root = tmp_path / "runs" / "smoke-20260901t120000z-fedcba9876543210"
    run_root.mkdir(parents=True)
    (run_root / "result.json").write_text("{}\n", encoding="utf-8")
    (run_root / "failure.json").write_text("{}\n", encoding="utf-8")

    journal.started()
    assert not (run_root / "result.json").exists()
    assert not (run_root / "failure.json").exists()
    journal.failed(RuntimeError("injected durable failure"))

    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    failure = json.loads((run_root / "failure.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert "injected durable failure" in failure["message"]
    assert not (run_root / "result.json").exists()


def test_durable_contract_mismatch_fails_before_gpu_canary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(MODULE, "gpu_smoke_contract_sha256", lambda _root: "d" * 64)
    monkeypatch.setattr(
        MODULE,
        "_run_remote",
        lambda *_args, **_kwargs: pytest.fail("GPU canary must not run after contract mismatch"),
    )

    with pytest.raises(RuntimeError, match="inspect its Volume journal"):
        MODULE._run_durable_remote(
            "a100-40gb",
            "smoke-20260901t120000z-aabbccddeeff0011",
            "fc-aabbccddeeff0011",
            1.0,
            "c" * 64,
            journal_root=tmp_path,
            commit=lambda: None,
        )

    run_root = tmp_path / "runs" / "smoke-20260901t120000z-aabbccddeeff0011"
    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert (run_root / "failure.json").is_file()
    assert not (run_root / "result.json").exists()


def test_durable_timeout_is_wrapped_after_failure_journaling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(MODULE, "gpu_smoke_contract_sha256", lambda _root: "e" * 64)
    monkeypatch.setattr(MODULE, "_verify_executed_entrypoint", lambda *_args: None)
    monkeypatch.setattr(
        MODULE,
        "_run_remote",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("child timed out")),
    )

    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._run_durable_remote(
            "a100-40gb",
            "smoke-20260901t120000z-1122334455667788",
            "fc-1122334455667788",
            1.0,
            "e" * 64,
            journal_root=tmp_path,
            commit=lambda: None,
        )

    assert isinstance(captured.value.__cause__, TimeoutError)


def test_durable_start_commit_failure_prevents_gpu_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def run_remote(*_args: Any, **_kwargs: Any) -> dict[str, object]:
        nonlocal called
        called = True
        return _remote_result()

    monkeypatch.setattr(MODULE, "_run_remote", run_remote)

    with pytest.raises(RuntimeError, match="inspect its Volume journal"):
        MODULE._run_durable_remote(
            "a100-40gb",
            "smoke-20260901t120000z-8877665544332211",
            "fc-8877665544332211",
            1.0,
            "f" * 64,
            journal_root=tmp_path,
            commit=lambda: (_ for _ in ()).throw(OSError("commit failed")),
        )

    assert called is False


def test_durable_result_commit_failure_cannot_return_false_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commits = 0

    def commit() -> None:
        nonlocal commits
        commits += 1
        if commits == 3:
            raise OSError("injected terminal commit failure")

    monkeypatch.setattr(MODULE, "gpu_smoke_contract_sha256", lambda _root: "1" * 64)
    monkeypatch.setattr(MODULE, "_verify_executed_entrypoint", lambda *_args: None)
    monkeypatch.setattr(MODULE, "_run_remote", lambda *_args, **_kwargs: _remote_result())

    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._run_durable_remote(
            "a100-40gb",
            "smoke-20260901t120000z-1234567890abcdef",
            "fc-1234567890abcdef",
            1.0,
            "1" * 64,
            journal_root=tmp_path,
            commit=commit,
        )

    assert isinstance(captured.value.__cause__, OSError)
    run_root = tmp_path / "runs" / "smoke-20260901t120000z-1234567890abcdef"
    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert not (run_root / "result.json").exists()


def test_failure_journal_error_does_not_replace_original_gpu_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commits = 0

    def commit() -> None:
        nonlocal commits
        commits += 1
        if commits == 3:
            raise OSError("injected failure-journal commit error")

    monkeypatch.setattr(MODULE, "gpu_smoke_contract_sha256", lambda _root: "2" * 64)
    monkeypatch.setattr(MODULE, "_verify_executed_entrypoint", lambda *_args: None)
    monkeypatch.setattr(
        MODULE,
        "_run_remote",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("original GPU error")),
    )

    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._run_durable_remote(
            "a100-40gb",
            "smoke-20260901t120000z-abcdef1234567890",
            "fc-abcdef1234567890",
            1.0,
            "2" * 64,
            journal_root=tmp_path,
            commit=commit,
        )

    assert isinstance(captured.value.__cause__, ValueError)
    assert str(captured.value.__cause__) == "original GPU error"


def test_modal_dispatch_binds_the_actual_function_call_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def dispatch(*args: object, **kwargs: object) -> dict[str, object]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"status": "sentinel"}

    monkeypatch.setattr(
        MODULE,
        "modal",
        SimpleNamespace(current_function_call_id=lambda: "fc-0011223344556677"),
    )
    monkeypatch.setattr(MODULE, "_dispatch_remote", dispatch)

    result = MODULE._dispatch_modal_function(
        "a100-40gb",
        "smoke-20260901t120000z-0011223344556677",
        1.0,
        "3" * 64,
        journal_root=tmp_path,
        commit=lambda: None,
    )

    assert result == {"status": "sentinel"}
    assert observed["args"] == (
        "a100-40gb",
        "smoke-20260901t120000z-0011223344556677",
        "fc-0011223344556677",
        1.0,
        "3" * 64,
    )


def test_durable_dispatch_rejects_partial_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supplied together"):
        MODULE._dispatch_remote(
            "a100-40gb",
            run_id="smoke-20260901t120000z-0011223344556677",
            journal_root=tmp_path,
            commit=lambda: None,
        )
