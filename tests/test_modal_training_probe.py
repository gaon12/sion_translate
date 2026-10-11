"""CPU-only contract checks before any paid training-capacity probe."""

from __future__ import annotations

import ast
from dataclasses import asdict
import importlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import modal_training_probe as probe
from sion_translate.auto import EnvironmentInfo
from sion_translate.config import config_from_raw, load_raw_config


def test_modal_app_definition_is_valid_without_submitting(tmp_path):
    pytest.importorskip("modal")
    app, function = probe.build_app(tmp_path, "a100-80gb")
    assert app.name == "sion-real-data-training-probe"
    assert function is not None


def test_probe_resource_limits_match_the_compute_contingency(tmp_path, monkeypatch):
    modal = pytest.importorskip("modal")
    observed = {}

    def function(**kwargs):
        observed.update(kwargs)
        return lambda worker: worker

    monkeypatch.setattr(modal, "App", lambda *args, **kwargs: SimpleNamespace(function=function))
    probe.build_app(tmp_path, "h100")
    assert observed["cpu"] == (probe.CPU, probe.CPU)
    assert observed["memory"] == (probe.MEMORY_GIB * 1024, probe.MEMORY_GIB * 1024)
    assert observed["retries"] == 0
    assert observed["timeout"] == probe.TIMEOUT


def test_headroom_cases_keep_all_stages_and_explicit_candidate_batching():
    cases = probe.CASE_SETS["headroom"]
    assert cases["foundation"] == [(24, None)]
    assert cases["sft"] == [(24, None)]
    assert cases["mrt"] == [(8, 8), (16, 8), (32, 8)]
    assert probe.compute_contingency("h100") < 4


def test_balanced_cases_preserve_sft_and_mrt_effective_batch_divisibility():
    cases = probe.CASE_SETS["balanced"]
    assert cases["foundation"] == [(32, None)]
    assert cases["sft"] == [(32, None)]
    assert 256 % cases["sft"][0][0] == 0
    assert cases["mrt"] == [(8, 4), (16, 4)]
    assert all(16 % batch == 0 for batch, _ in cases["mrt"])


def test_profiler_preserves_events_under_strict_warnings():
    import torch

    with probe.make_profiler(torch, cuda=False) as profiler:
        torch.ones(4).sum()
    assert profiler.acc_events is True
    assert any(event.key == "aten::sum" for event in profiler.key_averages())


def _plan(tmp_path: Path):
    asset = tmp_path / "tokenizer" / "sion.model"
    asset.parent.mkdir()
    asset.write_bytes(b"authenticated tokenizer")
    plan = {
        "schema": "sion-training-probe-data-v1",
        "provenance": {"complete_indexed_inventories_verified": True},
        "files": {
            "tokenizer/sion.model": {"size": asset.stat().st_size, "sha256": probe.sha256(asset)}
        },
    }
    probe.write_json(tmp_path / "plan.json", plan)
    return plan


def test_plan_and_all_payload_bytes_are_authenticated(tmp_path: Path):
    plan = _plan(tmp_path)
    digest = probe.sha256(tmp_path / "plan.json")
    assert probe.verify_plan(tmp_path, digest) == plan
    with pytest.raises(ValueError, match="plan SHA-256"):
        probe.verify_plan(tmp_path, "0" * 64)
    (tmp_path / "tokenizer/sion.model").write_bytes(b"modified")
    with pytest.raises(ValueError, match="asset integrity"):
        probe.verify_plan(tmp_path, digest)


def test_plan_rejects_unverified_inventory_and_path_escape(tmp_path: Path):
    plan = _plan(tmp_path)
    plan["provenance"]["complete_indexed_inventories_verified"] = False
    probe.write_json(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="not authenticated"):
        probe.verify_plan(tmp_path, probe.sha256(tmp_path / "plan.json"))
    plan["provenance"]["complete_indexed_inventories_verified"] = True
    plan["files"] = {"../escaped.model": {"size": 1, "sha256": "0" * 64}}
    probe.write_json(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="escapes"):
        probe.verify_plan(tmp_path, probe.sha256(tmp_path / "plan.json"))


def test_length_strata_do_not_repeat_or_mutate_source_rows():
    rows = [{"src": list(range(i)), "tgt": [1], "id": i} for i in range(10, 0, -1)]
    before = list(rows)
    assert [row["id"] for row in probe.batch_at_fraction(rows, 3, 0)] == [1, 2, 3]
    assert [row["id"] for row in probe.batch_at_fraction(rows, 3, 1)] == [8, 9, 10]
    assert rows == before
    with pytest.raises(ValueError):
        probe.batch_at_fraction(rows, 11, 0.5)


def test_timing_projects_optimizer_cost_at_preserved_effective_batch():
    samples = [{"seconds": 1.0, "tokens": 40}, {"seconds": 3.0, "tokens": 80}]
    result = probe.summarize_measurements(samples, 0.5, batch_size=8, effective_batch=32)
    assert result["projected_accumulation"] == 4
    assert result["projected_effective_batch"] == 32
    assert result["examples_per_second"] == pytest.approx(16 / 4.25)
    assert result["tokens_per_second"] == pytest.approx(120 / 4.25)
    larger = probe.summarize_measurements(samples, 0.5, batch_size=64, effective_batch=32)
    assert larger["projected_effective_batch"] == 64


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timing_is_not_a_speed_result(seconds: float):
    with pytest.raises(ValueError):
        probe.summarize_measurements([{"seconds": seconds, "tokens": 1}], 0.1, 8, 32)


def test_production_config_uses_full_token_count_and_mocked_80gb_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import sion_translate.auto as automatic

    raw = load_raw_config(Path(__file__).resolve().parents[1] / "sion_translate.yaml")
    config = config_from_raw(raw)
    config.model.vocab_size = 48_000
    plan = {
        "config": asdict(config),
        "raw_config": raw,
        "stages": {
            "sft": {
                "examples_per_epoch": 56_109_623,
                "physical_pairs": 28_148_216,
                "physical_tokens": 1_550_725_850,
                "representative": [{"src": [4], "tgt": [5]}],
            }
        },
    }
    environment = EnvironmentInfo(True, 1, 1, "NVIDIA A100 80GB", 80.0, True, 8, "Linux")
    monkeypatch.setattr(automatic, "probe_environment", lambda: environment)
    observed_counts = {}
    apply_auto_settings = automatic.apply_auto_settings

    def record_auto_settings(*args, **kwargs):
        observed_counts.update(kwargs)
        return apply_auto_settings(*args, **kwargs)

    monkeypatch.setattr(automatic, "apply_auto_settings", record_auto_settings)
    resolved, _decisions = probe.resolved_config(plan, tmp_path)
    assert resolved.training.precision == "bf16"
    assert resolved.training.compile is False
    assert resolved.model.vocab_size == 48_000
    assert resolved.model.d_model == 832
    assert resolved.model.encoder_layers == 18 and resolved.model.decoder_layers == 9
    assert resolved.training.num_train_epochs == 3
    # Explicit architecture pins skip the automatic sizing log. Verify the
    # complete dataset reaches the policy rather than relying on that message.
    assert observed_counts["train_examples"] == 56_109_623
    assert observed_counts["physical_train_pairs"] == 28_148_216
    assert observed_counts["physical_train_tokens"] == 1_550_725_850
    assert resolved.data.tokenizer_model == str(tmp_path / "tokenizer/sion.model")
    assert plan["config"] == asdict(config)


def test_worker_imports_and_scheduler_checkpoint_bindings_match_real_apis():
    """Check startup bindings without initializing CUDA or constructing a model."""
    syntax = ast.parse(inspect.getsource(probe.run_case))
    local_symbols = {}
    for node in ast.walk(syntax):
        if isinstance(node, ast.ImportFrom):
            module = importlib.import_module(node.module)
            for alias in node.names:
                local_symbols[alias.asname or alias.name] = getattr(module, alias.name)
    for node in ast.walk(syntax):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in {"cosine_scheduler", "save_checkpoint", "load_checkpoint"}:
                inspect.signature(local_symbols[name]).bind(
                    *([None] * len(node.args)),
                    **{keyword.arg: None for keyword in node.keywords},
                )


def test_invalid_derived_stage_stops_before_model_allocation(tmp_path, monkeypatch):
    import torch
    import sion_translate.foundation as foundation

    config = config_from_raw(
        load_raw_config(Path(__file__).resolve().parents[1] / "sion_translate.yaml")
    )
    config.data.source_only_languages = []
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "manual_seed", lambda _: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None)
    monkeypatch.setattr(probe, "verify_plan", lambda *_: {})
    monkeypatch.setattr(probe, "resolved_config", lambda *_: (config, []))
    monkeypatch.setattr(foundation, "build_foundation_config", lambda _: config)
    with pytest.raises(ValueError, match="requires source_only_languages"):
        probe.run_case(tmp_path, "0" * 64, "foundation", 32, tmp_path / "result.json")


@pytest.mark.parametrize("state", ["submission_intent", "failed"])
def test_status_before_spawn_does_not_access_a_call_or_overwrite_receipt(
    tmp_path, monkeypatch, capsys, state
):
    receipt = tmp_path / "receipt.json"
    probe.write_json(receipt, {"state": state, "app_id": "ap-building"})
    before = receipt.read_bytes()
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace())
    probe.status(SimpleNamespace(receipt=receipt))
    status = json.loads(capsys.readouterr().out)
    assert status["state"] == state
    assert status["app_id"] == "ap-building"
    assert receipt.read_bytes() == before


def test_status_rejects_a_submitted_receipt_without_a_call_id(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    probe.write_json(receipt, {"state": "submitted"})
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace())
    with pytest.raises(ValueError, match="missing its function call ID"):
        probe.status(SimpleNamespace(receipt=receipt))


@pytest.mark.parametrize("outcome", ["success", "error", "pending"])
def test_completed_journal_requires_successful_function_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
):
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "run_id": "test-run",
        "target": "a100-80gb",
        "call_id": "fc-test",
        "plan_sha256": "1" * 64,
        "probe_sha256": "2" * 64,
        "source_contract_sha256": "3" * 64,
        "state": "submitted",
        "volume": "test-volume",
    }
    journal = {key: value for key, value in receipt.items() if key not in {"state", "volume"}}
    journal["status"] = "completed"
    probe.write_json(receipt_path, receipt)

    def get(*, timeout):
        assert timeout == 0
        if outcome == "error":
            raise RuntimeError("shutdown failure after journal publication")
        if outcome == "pending":
            raise TimeoutError("still running")
        return journal

    call = SimpleNamespace(get=get, logs=SimpleNamespace(tail=lambda **_: []))
    volume = SimpleNamespace(read_file=lambda _: iter([json.dumps(journal).encode()]))
    modal = SimpleNamespace(
        FunctionCall=SimpleNamespace(from_id=lambda _: call),
        Volume=SimpleNamespace(from_name=lambda _: volume),
        exception=SimpleNamespace(TimeoutError=TimeoutError),
    )
    monkeypatch.setitem(sys.modules, "modal", modal)
    probe.status(SimpleNamespace(receipt=receipt_path))
    observed = probe.read_json(receipt_path)
    assert (
        observed["state"]
        == {
            "success": "completed",
            "error": "needs_reconciliation",
            "pending": "submitted",
        }[outcome]
    )
    if outcome == "error":
        assert "shutdown failure" in observed["call_error"]
