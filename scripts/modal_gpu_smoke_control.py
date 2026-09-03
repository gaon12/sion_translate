"""Submit and recover detached ephemeral Modal GPU smoke calls.

The controller writes an intent receipt before ``Function.spawn``, starts the App
with ``detach=True``, and records the FunctionCall ID as soon as Modal accepts the
input. The remote function independently journals progress and its final result in
a named Volume, so either identity can recover the run after the client exits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
from typing import Any, cast, Generator, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPOSITORY_ROOT / "scripts" / "modal_gpu_smoke.py"
SMOKE_SPEC = importlib.util.spec_from_file_location("modal_gpu_smoke", SMOKE_SCRIPT)
if SMOKE_SPEC is None or SMOKE_SPEC.loader is None:  # pragma: no cover - import invariant
    raise RuntimeError(f"cannot load the Modal GPU smoke module: {SMOKE_SCRIPT}")
SMOKE: Any = importlib.util.module_from_spec(SMOKE_SPEC)
# Modal inspects the registered module to mount the entrypoint by filename.
# An unregistered dynamic module instead becomes a serialized closure.
_previous_smoke_module = sys.modules.get(SMOKE_SPEC.name)
sys.modules[SMOKE_SPEC.name] = SMOKE
try:
    SMOKE_SPEC.loader.exec_module(SMOKE)
except BaseException:
    if _previous_smoke_module is None:
        sys.modules.pop(SMOKE_SPEC.name, None)
    else:
        sys.modules[SMOKE_SPEC.name] = _previous_smoke_module
    raise

try:
    import modal
except ModuleNotFoundError:  # Unit tests exercise the controller with a fake client.
    modal = None


RECEIPT_VERSION = 1
DEFAULT_RECEIPT_ROOT = REPOSITORY_ROOT / "artifacts" / "modal-runs"
MAX_REMOTE_JSON_BYTES = 8 * 1024 * 1024
MAX_LOG_ENTRIES = 200
MAX_LOG_BYTES = 2 * 1024 * 1024
MAX_WORKSPACE_BUDGET_HEADROOM_USD = 30.0
SUBMISSION_LOCK_NAME = ".submission-lock"
RECEIPT_FIELDS = {
    "receipt_version",
    "run_id",
    "target",
    "app_name",
    "function_name",
    "result_volume_name",
    "remote_run_path",
    "contract_sha256",
    "authorization_compute_charge_usd",
    "max_dollars",
    "workspace_budget_usd",
    "workspace_usage_before_submit_usd",
    "workspace_budget_headroom_usd",
    "created_at_utc",
    "git_commit",
    "submission_state",
    "function_call_id",
    "submission_error",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_bytes(value: object) -> bytes:
    SMOKE._validate_json_value(value)
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, value: object) -> None:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    commit = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise RuntimeError("the local Git commit identity is invalid")
    return commit


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    run_id = f"smoke-{timestamp}-{secrets.token_hex(8)}"
    return SMOKE._validated_run_id(run_id)


def _receipt_path(receipt_root: Path, run_id: str) -> Path:
    resolved_root = receipt_root.resolve()
    path = resolved_root / run_id / "receipt.json"
    if path.parent.parent != resolved_root:
        raise RuntimeError("Modal receipt path escaped its configured root")
    return path


def _validated_receipt(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Modal run receipt fields are invalid")
    receipt = cast(dict[str, Any], value.copy())
    if set(receipt) != RECEIPT_FIELDS:
        raise ValueError("Modal run receipt fields are invalid")
    if receipt.get("receipt_version") != RECEIPT_VERSION:
        raise ValueError("Modal run receipt version is unsupported")
    run_id = SMOKE._validated_run_id(receipt.get("run_id"))
    target = receipt.get("target")
    if not isinstance(target, str) or target not in SMOKE.TARGETS:
        raise ValueError("Modal run receipt target is invalid")
    expected_function = SMOKE.REMOTE_FUNCTION_NAMES[target]
    if (
        receipt.get("app_name") != SMOKE.APP_NAME
        or receipt.get("function_name") != expected_function
        or receipt.get("result_volume_name") != SMOKE.RESULT_VOLUME_NAME
        or receipt.get("remote_run_path") != f"runs/{run_id}"
    ):
        raise ValueError("Modal run receipt remote identity is invalid")
    contract = SMOKE._validated_contract_sha256(receipt.get("contract_sha256"))
    del contract
    max_dollars = receipt.get("max_dollars")
    if isinstance(max_dollars, bool) or not isinstance(max_dollars, (int, float)):
        raise ValueError("Modal run receipt budget is invalid")
    authorized = SMOKE.validate_cost_guard(target, float(max_dollars))
    recorded_authorization = receipt.get("authorization_compute_charge_usd")
    if (
        isinstance(recorded_authorization, bool)
        or not isinstance(recorded_authorization, (int, float))
        or not math.isclose(
            float(recorded_authorization),
            authorized,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Modal run receipt authorization is invalid")
    workspace_budget = receipt.get("workspace_budget_usd")
    workspace_usage = receipt.get("workspace_usage_before_submit_usd")
    expected_headroom = _validate_workspace_budget_guard(
        target,
        float(max_dollars),
        workspace_budget,
        workspace_usage,
    )
    recorded_headroom = receipt.get("workspace_budget_headroom_usd")
    if (
        isinstance(recorded_headroom, bool)
        or not isinstance(recorded_headroom, (int, float))
        or not math.isclose(float(recorded_headroom), expected_headroom, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("Modal run receipt Workspace budget headroom is invalid")
    if not isinstance(receipt.get("created_at_utc"), str):
        raise ValueError("Modal run receipt creation time is invalid")
    if re.fullmatch(r"[0-9a-f]{40,64}", str(receipt.get("git_commit"))) is None:
        raise ValueError("Modal run receipt Git identity is invalid")
    if receipt.get("submission_state") not in {
        "submitting",
        "submitted",
        "submission-unknown",
        "submission-failed",
    }:
        raise ValueError("Modal run receipt submission state is invalid")
    call_id = receipt.get("function_call_id")
    if call_id is not None and (
        not isinstance(call_id, str) or SMOKE.FUNCTION_CALL_ID_PATTERN.fullmatch(call_id) is None
    ):
        raise ValueError("Modal run receipt FunctionCall ID is invalid")
    submission_error = receipt.get("submission_error")
    if submission_error is not None and not isinstance(submission_error, dict):
        raise ValueError("Modal run receipt submission error is invalid")
    if receipt["submission_state"] == "submitted" and call_id is None:
        raise ValueError("submitted Modal run receipt has no FunctionCall ID")
    if receipt["submission_state"] in {"submitting", "submission-unknown"} and call_id is not None:
        raise ValueError("unconfirmed Modal run receipt unexpectedly has a FunctionCall ID")
    if receipt["submission_state"] == "submission-unknown" and submission_error is None:
        raise ValueError("ambiguous Modal run receipt has no submission diagnostic")
    SMOKE._validate_json_value(receipt)
    return receipt


def _read_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Modal run receipt is not a regular file: {path}")
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Modal run receipt is unreasonably large")
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Modal run receipt: {path}") from error
    return _validated_receipt(value)


def _require_modal() -> Any:
    if modal is None:
        raise RuntimeError("install the pinned Modal client before managing a remote run")
    SMOKE._validate_modal_client_version()
    return modal


def _read_local_recovered_state(
    run_directory: Path,
    expected_receipt: dict[str, Any],
) -> str | None:
    snapshot_path = run_directory / "status-latest.json"
    if not snapshot_path.exists():
        return None
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        raise RuntimeError(f"Modal status snapshot is not a regular file: {snapshot_path}")
    if snapshot_path.stat().st_size > MAX_REMOTE_JSON_BYTES:
        raise RuntimeError(f"Modal status snapshot is unreasonably large: {snapshot_path}")
    try:
        value: object = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read Modal status snapshot: {snapshot_path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Modal status snapshot is not a JSON object: {snapshot_path}")
    snapshot = cast(dict[str, Any], value)
    recovered_state = snapshot.get("recovered_state")
    if snapshot.get("receipt") != expected_receipt or recovered_state not in {
        "passed",
        "failed",
        "running",
        "submitted",
        "output-expired",
        "status-unavailable",
        "terminal-journal-pending-call",
        "submitting",
        "submission-unknown",
        "submission-failed",
    }:
        raise RuntimeError(f"Modal status snapshot identity or state is invalid: {snapshot_path}")
    if recovered_state == "passed" and not (
        snapshot.get("function_call_state") == "passed"
        or (
            snapshot.get("function_call_state") == "output-expired"
            and snapshot.get("remote_result") is not None
        )
    ):
        raise RuntimeError(f"Modal passed snapshot has no terminal call evidence: {snapshot_path}")
    if recovered_state == "failed" and not (
        snapshot.get("function_call_state") == "failed"
        or (
            snapshot.get("function_call_state") == "output-expired"
            and snapshot.get("remote_failure") is not None
        )
    ):
        raise RuntimeError(f"Modal failed snapshot has no terminal call evidence: {snapshot_path}")
    assert isinstance(recovered_state, str)
    return recovered_state


def _assert_no_unresolved_receipts(receipt_root: Path) -> None:
    resolved_root = receipt_root.resolve()
    if not resolved_root.exists():
        return
    if resolved_root.is_symlink() or not resolved_root.is_dir():
        raise RuntimeError(f"Modal receipt root is not a regular directory: {resolved_root}")
    unresolved: list[str] = []
    for receipt_path in sorted(resolved_root.glob("*/receipt.json")):
        receipt = _read_receipt(receipt_path)
        recovered_state = _read_local_recovered_state(receipt_path.parent, receipt)
        if recovered_state in {"passed", "failed"}:
            continue
        if receipt["submission_state"] == "submission-failed":
            continue
        unresolved.append(str(receipt_path))
    if unresolved:
        raise RuntimeError(
            "an earlier Modal GPU smoke has no recovered terminal state; run the status "
            f"command before submitting another target: {unresolved}"
        )


@contextmanager
def _exclusive_submission(receipt_root: Path) -> Generator[Path, None, None]:
    """Serialize paid submissions with an atomic, fail-closed local directory lock."""

    resolved_root = receipt_root.resolve()
    if resolved_root.exists() and (resolved_root.is_symlink() or not resolved_root.is_dir()):
        raise RuntimeError(f"Modal receipt root is not a regular directory: {resolved_root}")
    resolved_root.mkdir(parents=True, exist_ok=True)
    lock_path = resolved_root / SUBMISSION_LOCK_NAME
    try:
        lock_path.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "another Modal GPU submission may be active or an interrupted submission "
            f"left a fail-closed lock: {lock_path}"
        ) from error
    owner_path = lock_path / "owner.json"
    try:
        _write_json_atomic(
            owner_path,
            {
                "lock_version": 1,
                "process_id": os.getpid(),
                "acquired_at_utc": _utc_now(),
            },
        )
        yield resolved_root
    finally:
        owner_path.unlink(missing_ok=True)
        try:
            lock_path.rmdir()
        except FileNotFoundError as error:
            raise RuntimeError(
                f"Modal submission lock disappeared while held: {lock_path}"
            ) from error


def _validate_workspace_budget_guard(
    target: str,
    max_dollars: float,
    workspace_budget: object,
    workspace_usage: object,
) -> float:
    """Require a small platform-enforced hard-cap headroom before GPU submission."""

    authorized = SMOKE.validate_cost_guard(target, max_dollars)
    if (
        isinstance(workspace_budget, bool)
        or not isinstance(workspace_budget, (int, float))
        or not math.isfinite(float(workspace_budget))
        or float(workspace_budget) < 0.0
        or isinstance(workspace_usage, bool)
        or not isinstance(workspace_usage, (int, float))
        or not math.isfinite(float(workspace_usage))
        or float(workspace_usage) < 0.0
    ):
        raise ValueError("Workspace budget and current usage must be finite non-negative numbers")
    budget = float(workspace_budget)
    usage = float(workspace_usage)
    headroom = budget - usage
    if headroom + 1e-9 < authorized:
        raise ValueError(
            "Workspace budget headroom does not cover the selected target authorization"
        )
    if headroom > MAX_WORKSPACE_BUDGET_HEADROOM_USD + 1e-9:
        raise ValueError(
            "Workspace budget headroom exceeds the $30 safety ceiling; lower the hard "
            "Workspace budget before submitting"
        )
    return headroom


def submit(
    target: str,
    max_dollars: float,
    receipt_root: Path,
    *,
    workspace_budget: float,
    workspace_usage: float,
) -> Path:
    """Persist intent, detach one ephemeral input, and persist its durable identity."""

    _require_modal()
    authorized = SMOKE.validate_cost_guard(target, max_dollars)
    budget_headroom = _validate_workspace_budget_guard(
        target,
        max_dollars,
        workspace_budget,
        workspace_usage,
    )
    contract_sha256 = SMOKE.gpu_smoke_contract_sha256(REPOSITORY_ROOT)
    with _exclusive_submission(receipt_root) as resolved_receipt_root:
        _assert_no_unresolved_receipts(resolved_receipt_root)
        run_id = _new_run_id()
        receipt_path = _receipt_path(resolved_receipt_root, run_id)
        if receipt_path.parent.exists():
            raise FileExistsError(
                f"Modal run receipt directory already exists: {receipt_path.parent}"
            )
        receipt: dict[str, Any] = {
            "receipt_version": RECEIPT_VERSION,
            "run_id": run_id,
            "target": target,
            "app_name": SMOKE.APP_NAME,
            "function_name": SMOKE.REMOTE_FUNCTION_NAMES[target],
            "result_volume_name": SMOKE.RESULT_VOLUME_NAME,
            "remote_run_path": f"runs/{run_id}",
            "contract_sha256": contract_sha256,
            "authorization_compute_charge_usd": authorized,
            "max_dollars": max_dollars,
            "workspace_budget_usd": workspace_budget,
            "workspace_usage_before_submit_usd": workspace_usage,
            "workspace_budget_headroom_usd": budget_headroom,
            "created_at_utc": _utc_now(),
            "git_commit": _git_commit(),
            "submission_state": "submitting",
            "function_call_id": None,
            "submission_error": None,
        }
        _write_json_atomic(receipt_path, receipt)
        app_entered = False
        try:
            if SMOKE.app is None or target not in SMOKE.smoke_functions:
                raise RuntimeError("the Modal GPU smoke App is unavailable")
            with SMOKE.app.run(detach=True):
                app_entered = True
                function_call = SMOKE.smoke_functions[target].spawn(
                    run_id, max_dollars, contract_sha256
                )
                call_id = function_call.object_id
                if (
                    not isinstance(call_id, str)
                    or SMOKE.FUNCTION_CALL_ID_PATTERN.fullmatch(call_id) is None
                ):
                    raise RuntimeError("Modal returned an invalid FunctionCall ID")
                receipt["submission_state"] = "submitted"
                receipt["function_call_id"] = call_id
                _write_json_atomic(receipt_path, receipt)
        except BaseException as error:
            if receipt["submission_state"] != "submitted":
                receipt["submission_state"] = (
                    "submission-unknown" if app_entered else "submission-failed"
                )
            receipt["submission_error"] = {
                "error_type": type(error).__name__,
                "message": str(error)[:4_000],
                "recorded_at_utc": _utc_now(),
            }
            _write_json_atomic(receipt_path, receipt)
            raise
    print(_json_bytes({"receipt_path": str(receipt_path), **receipt}).decode(), end="")
    return receipt_path


def _read_volume_json(volume: Any, path: str) -> object | None:
    try:
        chunks = volume.read_file(path)
        payload = bytearray()
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise RuntimeError(f"Modal Volume returned a non-byte chunk for {path}")
            payload.extend(chunk)
            if len(payload) > MAX_REMOTE_JSON_BYTES:
                raise RuntimeError(f"Modal Volume JSON is unreasonably large: {path}")
    except FileNotFoundError:
        return None
    try:
        return json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Modal Volume contains invalid JSON: {path}") from error


def _validated_remote_status(value: object, receipt: dict[str, Any]) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("durable Modal status is not a JSON object")
    status = cast(dict[str, Any], value.copy())
    if (
        status.get("journal_version") != SMOKE.JOURNAL_VERSION
        or status.get("run_id") != receipt["run_id"]
        or status.get("target") != receipt["target"]
        or (
            receipt.get("function_call_id") is not None
            and status.get("function_call_id") != receipt["function_call_id"]
        )
        or not isinstance(status.get("function_call_id"), str)
        or SMOKE.FUNCTION_CALL_ID_PATTERN.fullmatch(status["function_call_id"]) is None
        or status.get("state") not in {"running", "passed", "failed"}
        or isinstance(status.get("sequence"), bool)
        or not isinstance(status.get("sequence"), int)
        or status["sequence"] < 1
        or status.get("expected_contract_sha256") != receipt["contract_sha256"]
    ):
        raise RuntimeError("durable Modal status identity or state is invalid")
    SMOKE._validate_json_value(status)
    return status


def _tail_logs(function_call: Any) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    total_bytes = 0
    for entry in function_call.logs.tail(entries=MAX_LOG_ENTRIES):
        message = str(entry.message)
        total_bytes += len(message.encode("utf-8", errors="replace"))
        if total_bytes > MAX_LOG_BYTES:
            raise RuntimeError("Modal FunctionCall log tail exceeded the local safety limit")
        entries.append(
            {
                "timestamp": entry.timestamp.isoformat(),
                "source": str(entry.source),
                "message": message,
                "object_id": str(entry.object_id),
            }
        )
    return entries


def status(receipt_path: Path, *, include_logs: bool = True) -> dict[str, Any]:
    """Recover the latest journal, result, FunctionCall state, and bounded log tail."""

    modal_module = _require_modal()
    receipt = _read_receipt(receipt_path.resolve())
    remote_root = str(receipt["remote_run_path"])
    volume = modal_module.Volume.from_name(str(receipt["result_volume_name"]))
    remote_status = _validated_remote_status(
        _read_volume_json(volume, f"{remote_root}/status.json"), receipt
    )
    raw_remote_result = _read_volume_json(volume, f"{remote_root}/result.json")
    remote_result = (
        None
        if raw_remote_result is None
        else SMOKE._validated_remote_result(receipt["target"], raw_remote_result)
    )
    remote_failure = _read_volume_json(volume, f"{remote_root}/failure.json")
    if remote_failure is not None:
        SMOKE._validate_json_value(remote_failure)

    receipt_call_id = receipt.get("function_call_id")
    discovered_call_id = None if remote_status is None else remote_status.get("function_call_id")
    call_id = receipt_call_id if isinstance(receipt_call_id, str) else discovered_call_id
    call_id_source = (
        "receipt"
        if isinstance(receipt_call_id, str)
        else "volume-status"
        if isinstance(discovered_call_id, str)
        else "unavailable"
    )
    call_state = "identity-unavailable"
    call_result: dict[str, Any] | None = None
    call_error: dict[str, str] | None = None
    log_entries: list[dict[str, object]] = []
    logs_error: dict[str, str] | None = None
    if isinstance(call_id, str):
        function_call: Any | None = None
        try:
            loaded_call = modal_module.FunctionCall.from_id(call_id)
        except Exception as error:
            call_state = "unavailable"
            call_error = {
                "error_type": type(error).__name__,
                "message": str(error)[:4_000],
            }
        else:
            function_call = loaded_call
            try:
                call_result = SMOKE._validated_remote_result(
                    receipt["target"], loaded_call.get(timeout=0)
                )
                call_state = "passed"
            except Exception as error:
                modal_exception_module = getattr(modal_module, "exception", None)
                output_expired_type = getattr(modal_exception_module, "OutputExpiredError", None)
                modal_timeout_type = getattr(modal_exception_module, "TimeoutError", None)
                is_exact_modal_poll_timeout = (
                    isinstance(modal_timeout_type, type) and type(error) is modal_timeout_type
                )
                is_durable_remote_failure = type(error) is RuntimeError and str(error) == (
                    f"durable Modal GPU smoke {receipt['run_id']} failed; "
                    "inspect its Volume journal"
                )
                terminal_error_types = tuple(
                    exception_type
                    for name in (
                        "FunctionTimeoutError",
                        "RemoteError",
                        "ExecutionError",
                    )
                    if isinstance(
                        exception_type := getattr(modal_exception_module, name, None),
                        type,
                    )
                )
                if isinstance(output_expired_type, type) and isinstance(error, output_expired_type):
                    call_state = "output-expired"
                elif isinstance(error, TimeoutError) or is_exact_modal_poll_timeout:
                    call_state = "running"
                elif is_durable_remote_failure or (
                    terminal_error_types and isinstance(error, terminal_error_types)
                ):
                    # These exceptions are produced only after Modal returns a terminal
                    # function output. In particular, a hard function timeout can kill
                    # the worker before it replaces the last durable "running" event.
                    call_state = "failed"
                    call_error = {
                        "error_type": type(error).__name__,
                        "message": str(error)[:4_000],
                    }
                else:
                    # An unclassified client-side error does not prove that the remote
                    # input terminated. Keeping it unresolved prevents paid overlap.
                    call_state = "unavailable"
                    call_error = {
                        "error_type": type(error).__name__,
                        "message": str(error)[:4_000],
                    }
        if include_logs and function_call is not None:
            try:
                log_entries = _tail_logs(function_call)
            except Exception as error:
                logs_error = {
                    "error_type": type(error).__name__,
                    "message": str(error)[:4_000],
                }

    if remote_result is not None and call_result is not None and remote_result != call_result:
        raise RuntimeError("Modal FunctionCall and Volume results disagree")
    if remote_result is not None and remote_failure is not None:
        raise RuntimeError("durable Modal run contains both success and failure artifacts")
    if remote_result is not None and call_state == "failed":
        raise RuntimeError("Modal Volume success and FunctionCall failure disagree")
    if remote_failure is not None and call_result is not None:
        raise RuntimeError("Modal Volume failure and FunctionCall success disagree")
    if remote_result is not None and (remote_status is None or remote_status["state"] != "passed"):
        raise RuntimeError("durable Modal result has no matching passed status")
    if remote_failure is not None and (remote_status is None or remote_status["state"] != "failed"):
        raise RuntimeError("durable Modal failure has no matching failed status")
    if remote_status is not None and remote_status["state"] == "passed" and remote_result is None:
        raise RuntimeError("durable Modal status passed without a persisted result")
    if remote_status is not None and remote_status["state"] == "failed" and remote_failure is None:
        raise RuntimeError("durable Modal status failed without persisted diagnostics")

    if call_result is not None:
        recovered_state = "passed"
    elif call_state == "failed":
        recovered_state = "failed"
    elif call_state == "output-expired":
        if remote_result is not None:
            recovered_state = "passed"
        elif remote_failure is not None:
            recovered_state = "failed"
        else:
            recovered_state = "output-expired"
    elif call_state == "unavailable":
        recovered_state = "status-unavailable"
    elif call_state == "running" and (remote_result is not None or remote_failure is not None):
        recovered_state = "terminal-journal-pending-call"
    elif remote_status is not None:
        recovered_state = str(remote_status["state"])
    elif call_state == "running":
        recovered_state = "submitted"
    else:
        recovered_state = str(receipt["submission_state"])
    snapshot: dict[str, Any] = {
        "recovered_at_utc": _utc_now(),
        "recovered_state": recovered_state,
        "receipt": receipt,
        "remote_status": remote_status,
        "remote_result": remote_result,
        "remote_failure": remote_failure,
        "observed_function_call_id": call_id,
        "function_call_id_source": call_id_source,
        "function_call_state": call_state,
        "function_call_result": call_result,
        "function_call_error": call_error,
        "logs": log_entries,
        "logs_error": logs_error,
    }
    _write_json_atomic(receipt_path.resolve().parent / "status-latest.json", snapshot)
    print(_json_bytes(snapshot).decode(), end="")
    return snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit or recover one detached ephemeral Modal GPU smoke call."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit_parser = subparsers.add_parser("submit", help="Spawn one durable paid GPU input.")
    submit_parser.add_argument("--target", required=True, choices=tuple(SMOKE.TARGETS))
    submit_parser.add_argument("--max-dollars", required=True, type=float)
    submit_parser.add_argument(
        "--workspace-budget",
        required=True,
        type=float,
        help="Hard Workspace usage budget currently set in Modal.",
    )
    submit_parser.add_argument(
        "--workspace-usage",
        required=True,
        type=float,
        help="Current-cycle usage shown immediately before submission.",
    )
    submit_parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPT_ROOT)
    status_parser = subparsers.add_parser("status", help="Recover one run without waiting.")
    status_parser.add_argument("--receipt", required=True, type=Path)
    status_parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Skip the bounded FunctionCall log-tail request.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    if parsed.command == "submit":
        submit(
            parsed.target,
            parsed.max_dollars,
            parsed.receipt_root,
            workspace_budget=parsed.workspace_budget,
            workspace_usage=parsed.workspace_usage,
        )
        return 0
    if parsed.command == "status":
        status(parsed.receipt, include_logs=not parsed.no_logs)
        return 0
    raise RuntimeError(f"unsupported Modal controller command: {parsed.command}")


if __name__ == "__main__":
    raise SystemExit(main())
