from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, datetime
import importlib.util
import json
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "modal_gpu_smoke_control.py"
SPEC = importlib.util.spec_from_file_location("modal_gpu_smoke_control_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

RUN_ID = "smoke-20260901t120000z-0123456789abcdef"
CALL_ID = "fc-0123456789abcdef"
CONTRACT_SHA256 = "a" * 64


def test_smoke_entrypoint_has_a_registered_file_module() -> None:
    function = MODULE.SMOKE.smoke_a100_40gb.get_raw_f()
    assert inspect.getmodule(function) is MODULE.SMOKE
    from modal._utils.function_utils import FunctionInfo

    info = FunctionInfo(function)
    assert not info.is_serialized()
    assert info.module_name == "modal_gpu_smoke"


def test_workspace_guard_accepts_authorized_thirty_dollar_budget() -> None:
    assert MODULE._validate_workspace_budget_guard("a100-40gb", 1.0, 30.0, 0.0) == 30.0


def _identity_remote_result(_target: str, value: object) -> object:
    return value


def _receipt(*, state: str = "submitted", call_id: str | None = CALL_ID) -> dict[str, object]:
    target = "a100-40gb"
    max_dollars = 1.0
    workspace_budget = 2.0
    workspace_usage = 1.0
    return {
        "receipt_version": MODULE.RECEIPT_VERSION,
        "run_id": RUN_ID,
        "target": target,
        "app_name": MODULE.SMOKE.APP_NAME,
        "function_name": MODULE.SMOKE.REMOTE_FUNCTION_NAMES[target],
        "result_volume_name": MODULE.SMOKE.RESULT_VOLUME_NAME,
        "remote_run_path": f"runs/{RUN_ID}",
        "contract_sha256": CONTRACT_SHA256,
        "authorization_compute_charge_usd": MODULE.SMOKE.authorization_compute_charge(target),
        "max_dollars": max_dollars,
        "workspace_budget_usd": workspace_budget,
        "workspace_usage_before_submit_usd": workspace_usage,
        "workspace_budget_headroom_usd": workspace_budget - workspace_usage,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": "b" * 40,
        "submission_state": state,
        "function_call_id": call_id,
        "submission_error": (
            None
            if state != "submission-unknown"
            else {
                "error_type": "ConnectionError",
                "message": "ambiguous",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
            }
        ),
    }


def _write_receipt(root: Path, value: dict[str, object] | None = None) -> Path:
    receipt_path = root / RUN_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, _receipt() if value is None else value)
    return receipt_path


class _RunContext(AbstractContextManager[None]):
    def __init__(
        self,
        observed: dict[str, object],
        *,
        enter_error: BaseException | None = None,
        exit_error: BaseException | None = None,
    ):
        self.observed = observed
        self.enter_error = enter_error
        self.exit_error = exit_error

    def __enter__(self) -> None:
        self.observed["enter_attempted"] = True
        if self.enter_error is not None:
            raise self.enter_error
        self.observed["entered"] = True
        return None

    def __exit__(self, *_args: object) -> bool:
        self.observed["exited"] = True
        if self.exit_error is not None:
            raise self.exit_error
        return False


def _configure_submit(
    monkeypatch: pytest.MonkeyPatch,
    receipt_root: Path,
    *,
    enter_error: BaseException | None = None,
    spawn_error: BaseException | None = None,
    exit_error: BaseException | None = None,
) -> tuple[dict[str, object], list[tuple[object, ...]]]:
    observed: dict[str, object] = {}
    calls: list[tuple[object, ...]] = []

    class FakeApp:
        def run(self, *, detach: bool = False) -> _RunContext:
            observed["detach"] = detach
            return _RunContext(
                observed,
                enter_error=enter_error,
                exit_error=exit_error,
            )

    class FakeFunction:
        def spawn(self, *arguments: object) -> object:
            intent = receipt_root / RUN_ID / "receipt.json"
            assert intent.is_file()
            assert (receipt_root / MODULE.SUBMISSION_LOCK_NAME).is_dir()
            assert json.loads(intent.read_text(encoding="utf-8"))["submission_state"] == (
                "submitting"
            )
            calls.append(arguments)
            if spawn_error is not None:
                raise spawn_error
            return SimpleNamespace(object_id=CALL_ID)

    def require_modal() -> object:
        return object()

    def new_run_id() -> str:
        return RUN_ID

    def git_commit() -> str:
        return "b" * 40

    def contract_sha256(_root: Path) -> str:
        return CONTRACT_SHA256

    monkeypatch.setattr(MODULE, "_require_modal", require_modal)
    monkeypatch.setattr(MODULE, "_new_run_id", new_run_id)
    monkeypatch.setattr(MODULE, "_git_commit", git_commit)
    monkeypatch.setattr(MODULE.SMOKE, "gpu_smoke_contract_sha256", contract_sha256)
    monkeypatch.setattr(MODULE.SMOKE, "app", FakeApp())
    monkeypatch.setattr(MODULE.SMOKE, "smoke_functions", {"a100-40gb": FakeFunction()})
    return observed, calls


def test_submit_persists_intent_then_detached_function_call_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed, calls = _configure_submit(monkeypatch, tmp_path)

    receipt_path = MODULE.submit(
        "a100-40gb",
        1.0,
        tmp_path,
        workspace_budget=2.0,
        workspace_usage=1.0,
    )

    receipt = MODULE._read_receipt(receipt_path)
    assert calls == [(RUN_ID, 1.0, CONTRACT_SHA256)]
    assert observed == {
        "detach": True,
        "enter_attempted": True,
        "entered": True,
        "exited": True,
    }
    assert receipt["submission_state"] == "submitted"
    assert receipt["function_call_id"] == CALL_ID
    assert receipt["submission_error"] is None
    assert not (tmp_path / MODULE.SUBMISSION_LOCK_NAME).exists()


def test_context_exit_failure_keeps_already_persisted_call_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_submit(monkeypatch, tmp_path, exit_error=OSError("detach exit failed"))

    with pytest.raises(OSError, match="detach exit failed"):
        MODULE.submit(
            "a100-40gb",
            1.0,
            tmp_path,
            workspace_budget=2.0,
            workspace_usage=1.0,
        )

    receipt = MODULE._read_receipt(tmp_path / RUN_ID / "receipt.json")
    assert receipt["submission_state"] == "submitted"
    assert receipt["function_call_id"] == CALL_ID
    assert receipt["submission_error"]["error_type"] == "OSError"


def test_ambiguous_spawn_is_not_retried_and_blocks_another_submission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _observed, calls = _configure_submit(
        monkeypatch, tmp_path, spawn_error=ConnectionError("response lost")
    )

    with pytest.raises(ConnectionError, match="response lost"):
        MODULE.submit(
            "a100-40gb",
            1.0,
            tmp_path,
            workspace_budget=2.0,
            workspace_usage=1.0,
        )

    receipt = MODULE._read_receipt(tmp_path / RUN_ID / "receipt.json")
    assert len(calls) == 1
    assert receipt["submission_state"] == "submission-unknown"
    assert receipt["function_call_id"] is None
    with pytest.raises(RuntimeError, match="earlier Modal GPU smoke"):
        MODULE.submit(
            "a100-40gb",
            1.0,
            tmp_path,
            workspace_budget=2.0,
            workspace_usage=1.0,
        )
    assert len(calls) == 1
    assert not (tmp_path / MODULE.SUBMISSION_LOCK_NAME).exists()


def test_function_creation_failure_is_terminal_without_a_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed, calls = _configure_submit(
        monkeypatch,
        tmp_path,
        enter_error=RuntimeError("function definition was rejected"),
    )

    with pytest.raises(RuntimeError, match="function definition was rejected"):
        MODULE.submit(
            "a100-40gb",
            1.0,
            tmp_path,
            workspace_budget=2.0,
            workspace_usage=1.0,
        )

    receipt = MODULE._read_receipt(tmp_path / RUN_ID / "receipt.json")
    assert observed["enter_attempted"] is True
    assert "entered" not in observed
    assert calls == []
    assert receipt["submission_state"] == "submission-failed"
    assert receipt["function_call_id"] is None
    assert receipt["submission_error"]["error_type"] == "RuntimeError"


def test_existing_submission_lock_fails_closed_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _observed, calls = _configure_submit(monkeypatch, tmp_path)
    lock_path = tmp_path / MODULE.SUBMISSION_LOCK_NAME
    lock_path.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="submission may be active"):
        MODULE.submit(
            "a100-40gb",
            1.0,
            tmp_path,
            workspace_budget=2.0,
            workspace_usage=1.0,
        )

    assert calls == []
    assert not (tmp_path / RUN_ID).exists()
    assert lock_path.is_dir()


@pytest.mark.parametrize(
    ("workspace_budget", "workspace_usage", "message"),
    (
        (1.5, 1.0, "does not cover"),
        (32.0, 1.0, r"exceeds the \$30"),
        (float("nan"), 0.0, "finite non-negative"),
        (True, 0.0, "finite non-negative"),
    ),
)
def test_workspace_hard_budget_guard_rejects_unsafe_headroom(
    workspace_budget: object,
    workspace_usage: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE._validate_workspace_budget_guard("a100-40gb", 1.0, workspace_budget, workspace_usage)


class _FakeVolume:
    def __init__(self, files: dict[str, object]):
        self.files = files

    def read_file(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield MODULE._json_bytes(self.files[path])


def _remote_status(state: str) -> dict[str, object]:
    return {
        "journal_version": MODULE.SMOKE.JOURNAL_VERSION,
        "run_id": RUN_ID,
        "target": "a100-40gb",
        "function_call_id": CALL_ID,
        "state": state,
        "sequence": 2,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "completed_phases": [],
        "max_dollars": 1.0,
        "authorization_compute_charge_usd": MODULE.SMOKE.authorization_compute_charge("a100-40gb"),
        "expected_contract_sha256": CONTRACT_SHA256,
        "event": "test",
    }


def _configure_status(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, object],
    get_result: object,
    *,
    output_expired_type: type[BaseException] | None = None,
    modal_timeout_type: type[BaseException] | None = None,
    function_timeout_type: type[BaseException] | None = None,
    internal_failure_type: type[BaseException] | None = None,
) -> None:
    volume = _FakeVolume(files)

    class FakeCall:
        class Logs:
            @staticmethod
            def tail(**_kwargs: object) -> list[object]:
                return []

        logs = Logs()

        def get(self, *, timeout: float) -> object:
            assert timeout == 0
            if isinstance(get_result, BaseException):
                raise get_result
            return get_result

    class FakeFunctionCall:
        @staticmethod
        def from_id(call_id: str) -> FakeCall:
            assert call_id == CALL_ID
            return FakeCall()

    fake_exception = SimpleNamespace(
        OutputExpiredError=output_expired_type,
        TimeoutError=modal_timeout_type,
        FunctionTimeoutError=function_timeout_type,
        InternalFailure=internal_failure_type,
    )

    class FakeVolumeType:
        @staticmethod
        def from_name(_name: str) -> _FakeVolume:
            return volume

    fake_modal = SimpleNamespace(
        Volume=FakeVolumeType,
        FunctionCall=FakeFunctionCall,
        exception=fake_exception,
    )

    def require_modal() -> object:
        return fake_modal

    monkeypatch.setattr(MODULE, "_require_modal", require_modal)


def test_status_distinguishes_pending_from_output_expiration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_path = _write_receipt(tmp_path)
    root = f"runs/{RUN_ID}"
    _configure_status(
        monkeypatch,
        {f"{root}/status.json": _remote_status("running")},
        TimeoutError(),
    )
    assert MODULE.status(receipt_path, include_logs=False)["recovered_state"] == "running"

    class OutputExpiredError(Exception):
        pass

    _configure_status(
        monkeypatch,
        {},
        OutputExpiredError(),
        output_expired_type=OutputExpiredError,
    )
    snapshot = MODULE.status(receipt_path, include_logs=False)
    assert snapshot["function_call_state"] == "output-expired"
    assert snapshot["recovered_state"] == "output-expired"


def test_status_accepts_exact_modal_poll_timeout_as_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ModalTimeoutError(Exception):
        pass

    receipt_path = _write_receipt(tmp_path)
    _configure_status(
        monkeypatch,
        {},
        ModalTimeoutError(),
        modal_timeout_type=ModalTimeoutError,
    )

    snapshot = MODULE.status(receipt_path, include_logs=False)
    assert snapshot["function_call_state"] == "running"
    assert snapshot["recovered_state"] == "submitted"


def test_status_treats_function_timeout_as_terminal_despite_stale_journal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ModalTimeoutError(Exception):
        pass

    class FunctionTimeoutError(ModalTimeoutError):
        pass

    receipt_path = _write_receipt(tmp_path)
    root = f"runs/{RUN_ID}"
    _configure_status(
        monkeypatch,
        {f"{root}/status.json": _remote_status("running")},
        FunctionTimeoutError("function exceeded 300 seconds"),
        modal_timeout_type=ModalTimeoutError,
        function_timeout_type=FunctionTimeoutError,
    )

    snapshot = MODULE.status(receipt_path, include_logs=False)
    assert snapshot["function_call_state"] == "failed"
    assert snapshot["recovered_state"] == "failed"
    MODULE._assert_no_unresolved_receipts(tmp_path)


def test_status_output_expiration_overrides_stale_running_journal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class OutputExpiredError(Exception):
        pass

    receipt_path = _write_receipt(tmp_path)
    root = f"runs/{RUN_ID}"
    _configure_status(
        monkeypatch,
        {f"{root}/status.json": _remote_status("running")},
        OutputExpiredError(),
        output_expired_type=OutputExpiredError,
    )

    snapshot = MODULE.status(receipt_path, include_logs=False)
    assert snapshot["function_call_state"] == "output-expired"
    assert snapshot["recovered_state"] == "output-expired"


def test_status_keeps_client_errors_non_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_path = _write_receipt(tmp_path)
    _configure_status(monkeypatch, {}, ConnectionError("temporary client failure"))

    snapshot = MODULE.status(receipt_path, include_logs=False)
    assert snapshot["function_call_state"] == "unavailable"
    assert snapshot["recovered_state"] == "status-unavailable"
    with pytest.raises(RuntimeError, match="no recovered terminal state"):
        MODULE._assert_no_unresolved_receipts(tmp_path)


def test_status_keeps_retriable_modal_internal_failure_non_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class InternalFailure(Exception):
        pass

    receipt_path = _write_receipt(tmp_path)
    _configure_status(
        monkeypatch,
        {},
        InternalFailure("temporary Modal internal failure"),
        internal_failure_type=InternalFailure,
    )

    snapshot = MODULE.status(receipt_path, include_logs=False)
    assert snapshot["function_call_state"] == "unavailable"
    assert snapshot["recovered_state"] == "status-unavailable"
    with pytest.raises(RuntimeError, match="no recovered terminal state"):
        MODULE._assert_no_unresolved_receipts(tmp_path)


def test_status_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_path = _write_receipt(tmp_path)
    _configure_status(monkeypatch, {}, KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        MODULE.status(receipt_path, include_logs=False)
    assert not (receipt_path.parent / "status-latest.json").exists()


def test_status_does_not_trust_volume_failure_after_unclassified_call_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_path = _write_receipt(tmp_path)
    root = f"runs/{RUN_ID}"
    failure = {"error_type": "RuntimeError", "message": "failed", "traceback_tail": ""}
    _configure_status(
        monkeypatch,
        {
            f"{root}/status.json": _remote_status("failed"),
            f"{root}/failure.json": failure,
        },
        RuntimeError("remote failure"),
    )

    snapshot = MODULE.status(receipt_path, include_logs=False)
    assert snapshot["recovered_state"] == "status-unavailable"
    assert snapshot["remote_failure"] == failure
    assert (receipt_path.parent / "status-latest.json").is_file()
    with pytest.raises(RuntimeError, match="no recovered terminal state"):
        MODULE._assert_no_unresolved_receipts(tmp_path)


def test_status_recognizes_exact_durable_remote_failure_without_journal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_path = _write_receipt(tmp_path)
    error = RuntimeError(f"durable Modal GPU smoke {RUN_ID} failed; inspect its Volume journal")
    _configure_status(monkeypatch, {}, error)

    snapshot = MODULE.status(receipt_path, include_logs=False)
    assert snapshot["function_call_state"] == "failed"
    assert snapshot["recovered_state"] == "failed"
    MODULE._assert_no_unresolved_receipts(tmp_path)


def test_status_discovers_ambiguous_submission_call_id_from_volume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_path = _write_receipt(
        tmp_path,
        _receipt(state="submission-unknown", call_id=None),
    )
    root = f"runs/{RUN_ID}"
    _configure_status(
        monkeypatch,
        {f"{root}/status.json": _remote_status("running")},
        TimeoutError(),
    )

    snapshot = MODULE.status(receipt_path, include_logs=False)
    assert snapshot["observed_function_call_id"] == CALL_ID
    assert snapshot["function_call_id_source"] == "volume-status"
    assert snapshot["function_call_state"] == "running"
    assert snapshot["recovered_state"] == "running"


def test_status_requires_terminal_call_after_committed_volume_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = {"proof": "validated"}
    monkeypatch.setattr(
        MODULE.SMOKE,
        "_validated_remote_result",
        _identity_remote_result,
    )
    receipt_path = _write_receipt(tmp_path)
    root = f"runs/{RUN_ID}"
    _configure_status(
        monkeypatch,
        {
            f"{root}/status.json": _remote_status("passed"),
            f"{root}/result.json": result,
        },
        TimeoutError(),
    )

    snapshot = MODULE.status(receipt_path, include_logs=False)
    assert snapshot["function_call_state"] == "running"
    assert snapshot["recovered_state"] == "terminal-journal-pending-call"
    with pytest.raises(RuntimeError, match="no recovered terminal state"):
        MODULE._assert_no_unresolved_receipts(tmp_path)


def test_status_rejects_volume_success_and_terminal_call_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ModalTimeoutError(Exception):
        pass

    class FunctionTimeoutError(ModalTimeoutError):
        pass

    monkeypatch.setattr(
        MODULE.SMOKE,
        "_validated_remote_result",
        _identity_remote_result,
    )
    receipt_path = _write_receipt(tmp_path)
    root = f"runs/{RUN_ID}"
    _configure_status(
        monkeypatch,
        {
            f"{root}/status.json": _remote_status("passed"),
            f"{root}/result.json": {"proof": "validated"},
        },
        FunctionTimeoutError("terminal timeout"),
        modal_timeout_type=ModalTimeoutError,
        function_timeout_type=FunctionTimeoutError,
    )

    with pytest.raises(RuntimeError, match="Volume success and FunctionCall failure"):
        MODULE.status(receipt_path, include_logs=False)


def test_status_rejects_volume_failure_and_function_call_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = {"proof": "validated"}
    monkeypatch.setattr(
        MODULE.SMOKE,
        "_validated_remote_result",
        _identity_remote_result,
    )
    receipt_path = _write_receipt(tmp_path)
    root = f"runs/{RUN_ID}"
    _configure_status(
        monkeypatch,
        {
            f"{root}/status.json": _remote_status("failed"),
            f"{root}/failure.json": {
                "error_type": "RuntimeError",
                "message": "failed",
                "traceback_tail": "",
            },
        },
        result,
    )

    with pytest.raises(RuntimeError, match="Volume failure and FunctionCall success"):
        MODULE.status(receipt_path, include_logs=False)


def test_status_rejects_function_call_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_path = _write_receipt(tmp_path)
    root = f"runs/{RUN_ID}"
    status = _remote_status("running")
    status["function_call_id"] = "fc-fedcba9876543210"
    _configure_status(monkeypatch, {f"{root}/status.json": status}, TimeoutError())

    with pytest.raises(RuntimeError, match="identity or state"):
        MODULE.status(receipt_path, include_logs=False)
