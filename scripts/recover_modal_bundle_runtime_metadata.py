"""Recover one authenticated bundle after Modal omitted distribution metadata.

This controller never uploads, copies, or moves the prepared archive. It binds
one fresh attempt ID to the exact failed source receipt and Volume journal with
a deterministic no-overwrite claim. The attested CPU worker then reads the
original incoming directory directly and writes only a fresh operation journal.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence, cast


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE_SCRIPT = REPOSITORY_ROOT / "scripts" / "modal_stage_gpu_bundle.py"
RECOVERY_SCHEMA = "sion-modal-runtime-metadata-recovery-v1"
MAX_REMOTE_JSON_BYTES = 8 * 1024 * 1024
APP_ID_PATTERN = re.compile(r"^ap-[A-Za-z0-9_-]{8,128}$")


def _load_stage_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "modal_stage_gpu_bundle",
        STAGE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Modal bundle stager: {STAGE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise
    return module


STAGE: Any = _load_stage_module()


class RuntimeMetadataRecoveryError(RuntimeError):
    """Raised when the exact supported recovery cannot be proven safe."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_bytes(value: object) -> bytes:
    return STAGE._json_bytes(value)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = STAGE._read_local_json(path, label)
    if not isinstance(value, dict):
        raise RuntimeMetadataRecoveryError(f"{label} is not a JSON object")
    return cast(dict[str, Any], value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, label: str) -> str:
    _size, digest = STAGE._hash_regular_file_stable(path, label)
    return digest


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeMetadataRecoveryError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _replacement_revision() -> tuple[str, str]:
    commit = _git("rev-parse", "HEAD")
    if commit != _git("rev-parse", "origin/main"):
        raise RuntimeMetadataRecoveryError("replacement commit is not pushed to origin/main")
    paths = (
        *STAGE.finalizer_runtime_contract_paths(REPOSITORY_ROOT),
        Path(__file__).relative_to(REPOSITORY_ROOT),
    )
    tracked_source = set(
        _git("ls-tree", "-r", "--name-only", commit, "--", "src/sion_translate").splitlines()
    )
    actual_source = {
        path.as_posix() for path in paths if path.parts[:2] == ("src", "sion_translate")
    }
    if actual_source != tracked_source:
        raise RuntimeMetadataRecoveryError(
            "runtime source inventory differs from the pushed commit"
        )
    for path in paths:
        # Apply the repository's checkout filters, including Windows CRLF.
        # The submitted runtime contract separately binds the exact image bytes.
        completed = subprocess.run(
            ["git", "cat-file", "--filters", f"{commit}:{path.as_posix()}"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
        )
        observed = _sha256_file(REPOSITORY_ROOT / path, "reviewed runtime file")
        raw = subprocess.run(
            ["git", "cat-file", "blob", f"{commit}:{path.as_posix()}"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
        )
        if (
            completed.returncode != 0
            or raw.returncode != 0
            or observed
            not in {
                _sha256_bytes(completed.stdout),
                _sha256_bytes(raw.stdout),
            }
        ):
            raise RuntimeMetadataRecoveryError(
                f"runtime file differs from the pushed commit: {path}"
            )
    return commit, _git("rev-parse", "HEAD^{tree}")


def _is_modal_not_found(error: BaseException) -> bool:
    return isinstance(error, FileNotFoundError) or (
        type(error).__module__ == "modal.exception" and type(error).__name__ == "NotFoundError"
    )


def _read_volume_bytes(volume: Any, path: str, label: str) -> bytes | None:
    try:
        payload = bytearray()
        for chunk in volume.read_file(path):
            if not isinstance(chunk, bytes):
                raise RuntimeMetadataRecoveryError(
                    f"Modal Volume returned a non-byte chunk for {label}"
                )
            payload.extend(chunk)
            if len(payload) > MAX_REMOTE_JSON_BYTES:
                raise RuntimeMetadataRecoveryError(f"{label} is unreasonably large")
    except BaseException as error:
        if _is_modal_not_found(error):
            return None
        raise
    return bytes(payload)


def _read_volume_json(volume: Any, path: str, label: str) -> tuple[bytes, dict[str, Any]] | None:
    payload = _read_volume_bytes(volume, path, label)
    if payload is None:
        return None
    try:
        value: object = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeMetadataRecoveryError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeMetadataRecoveryError(f"{label} is not a JSON object")
    return payload, cast(dict[str, Any], value)


def _assert_remote_path_absent(volume: Any, path: str, label: str) -> None:
    try:
        list(volume.iterdir(path, recursive=False))
    except BaseException as error:
        if _is_modal_not_found(error):
            return
        raise
    raise RuntimeMetadataRecoveryError(f"{label} already exists: {path}")


def _validate_remote_incoming(volume: Any, receipt: Mapping[str, object]) -> None:
    incoming_path = cast(str, receipt["remote_incoming_path"])
    entries = list(volume.iterdir(incoming_path, recursive=False))
    if len(entries) != 1:
        raise RuntimeMetadataRecoveryError(
            "source incoming directory does not contain exactly one archive"
        )
    entry = entries[0]
    entry_type = getattr(getattr(entry, "type", None), "name", None)
    entry_path = getattr(entry, "path", None)
    entry_size = getattr(entry, "size", None)
    if (
        entry_type != "FILE"
        or not isinstance(entry_path, str)
        or entry_path.lstrip("/") != f"{incoming_path.lstrip('/')}/bundle.zip"
        or entry_size != receipt["bundle_size"]
    ):
        raise RuntimeMetadataRecoveryError("source incoming archive identity is invalid")


def _read_exact_app_lifecycle(app_id: str) -> tuple[str, float]:
    # Use the same read-only endpoint as the pinned SDK's exact-ID CLI lookup.
    # AppList only retains recently stopped Apps; absence is not proof of exit.
    from modal.client import _Client
    from modal_proto import api_pb2

    async def read() -> tuple[str, float]:
        client = await _Client.from_env()
        response = await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id))
        lifecycle = response.lifecycle
        return api_pb2.AppState.Name(lifecycle.app_state), lifecycle.stopped_at

    return asyncio.run(read())


def _assert_app_stopped(app_id: str) -> None:
    if APP_ID_PATTERN.fullmatch(app_id) is None:
        raise RuntimeMetadataRecoveryError("failed Modal App ID is invalid")
    completed = subprocess.run(
        [sys.executable, "-m", "modal", "app", "list", "--json"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        apps: object = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeMetadataRecoveryError("cannot parse Modal App state") from error
    if completed.returncode != 0 or not isinstance(apps, list):
        raise RuntimeMetadataRecoveryError("cannot read Modal App state")
    app_values = cast(list[object], apps)
    matches: list[dict[str, object]] = []
    for value in app_values:
        if not isinstance(value, dict):
            continue
        app = cast(dict[str, object], value)
        if app.get("app_id") == app_id:
            matches.append(app)
    if not matches:
        state, stopped_at = _read_exact_app_lifecycle(app_id)
        if state != "APP_STATE_STOPPED" or not math.isfinite(stopped_at) or stopped_at <= 0:
            raise RuntimeMetadataRecoveryError("exact failed Modal App lifecycle is not stopped")
        return
    if len(matches) != 1 or matches[0].get("state") != "stopped" or matches[0].get("tasks") != "0":
        raise RuntimeMetadataRecoveryError("failed Modal App is not stopped with zero tasks")


def _assert_terminal_failed_call(modal_module: Any, call_id: str, upload_id: str) -> None:
    call = modal_module.FunctionCall.from_id(call_id)
    expected = f"durable Modal bundle finalizer {upload_id} failed; inspect its Volume journal"
    try:
        call.get(timeout=0)
    except TimeoutError as error:
        raise RuntimeMetadataRecoveryError("failed Modal FunctionCall is still pending") from error
    except BaseException as error:
        if type(error) is not RuntimeError or str(error) != expected:
            raise RuntimeMetadataRecoveryError(
                "failed Modal FunctionCall has an unsupported terminal result"
            ) from error
    else:
        raise RuntimeMetadataRecoveryError("failed Modal FunctionCall unexpectedly succeeded")


def _validated_source_evidence(
    volume: Any,
    receipt_path: Path,
    receipt: Mapping[str, object],
) -> dict[str, object]:
    call_id = cast(str, receipt["function_call_id"])
    snapshot = _read_json(receipt_path.parent / "status-latest.json", "failed status snapshot")
    status_remote = f"{receipt['remote_operation_path']}/status.json"
    failure_remote = f"{receipt['remote_operation_path']}/failure.json"
    result_remote = f"{receipt['remote_operation_path']}/result.json"
    status_record = _read_volume_json(volume, status_remote, "source operation status")
    failure_record = _read_volume_json(volume, failure_remote, "source operation failure")
    if status_record is None or failure_record is None:
        raise RuntimeMetadataRecoveryError("source operation journal is incomplete")
    status_bytes, status_value = status_record
    failure_bytes, failure_value = failure_record
    status = STAGE._validated_operation_status(status_value, receipt)
    failure = STAGE._validated_failure(failure_value, receipt)
    if (
        status is None
        or status.get("state") != "failed"
        or status.get("sequence") != 2
        or failure is None
        or failure.get("error_type") != STAGE.SOURCE_RECOVERY_ERROR_TYPE
        or failure.get("message") != STAGE.SOURCE_RECOVERY_ERROR_MESSAGE
        or "importlib.metadata.PackageNotFoundError" not in cast(str, failure.get("traceback_tail"))
        or _read_volume_bytes(volume, result_remote, "source operation result") is not None
    ):
        raise RuntimeMetadataRecoveryError("source operation is not the supported metadata failure")
    expected_terminal = (
        f"durable Modal bundle finalizer {receipt['upload_id']} failed; inspect its Volume journal"
    )
    raw_call_error = snapshot.get("function_call_error")
    call_error = (
        cast(dict[str, object], raw_call_error) if isinstance(raw_call_error, dict) else None
    )
    if (
        snapshot.get("receipt") != receipt
        or snapshot.get("recovered_state") != "failed"
        or snapshot.get("function_call_state") != "failed"
        or snapshot.get("observed_function_call_id") != call_id
        or snapshot.get("remote_status") != status
        or snapshot.get("remote_failure") != failure
        or snapshot.get("remote_result") is not None
        or not isinstance(call_error, dict)
        or call_error.get("error_type") != "RuntimeError"
        or call_error.get("message") != expected_terminal
    ):
        raise RuntimeMetadataRecoveryError("local status snapshot disagrees with source evidence")
    return {
        "source_status_sha256": _sha256_bytes(status_bytes),
        "source_failure_sha256": _sha256_bytes(failure_bytes),
        "source_status": status,
        "source_failure": failure,
    }


def _attempt_receipt(
    source: Mapping[str, object],
    attempt_upload_id: str,
    runtime_contract_sha256: str,
    created_at_utc: str,
    *,
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
    recovery_claim: Mapping[str, object],
) -> dict[str, Any]:
    incoming, final, operation = STAGE._remote_paths(
        attempt_upload_id, cast(str, source["bundle_sha256"])
    )
    observation = STAGE._budget_observation(max_dollars, workspace_budget, workspace_usage)
    receipt: dict[str, Any] = {
        "receipt_version": STAGE.SOURCE_RECOVERY_RECEIPT_VERSION,
        "source_recovery_claim": dict(recovery_claim),
        "upload_id": attempt_upload_id,
        "volume_name": source["volume_name"],
        "volume_version": STAGE.VOLUME_VERSION,
        "app_name": STAGE.APP_NAME,
        "function_name": STAGE.FINALIZER_FUNCTION_NAME,
        "runtime_contract_sha256": runtime_contract_sha256,
        "local_bundle_path": source["local_bundle_path"],
        "bundle_sha256": source["bundle_sha256"],
        "bundle_size": source["bundle_size"],
        "verification": source["verification"],
        "remote_incoming_path": recovery_claim["source_remote_incoming_path"],
        "remote_final_path": final,
        "remote_operation_path": operation,
        "created_at_utc": created_at_utc,
        "authorization_compute_charge_usd": observation["authorization_compute_charge_usd"],
        "max_dollars": observation["max_dollars"],
        "workspace_budget_usd": observation["workspace_budget_usd"],
        "workspace_usage_before_submit_usd": observation["workspace_usage_usd"],
        "workspace_budget_headroom_usd": observation["workspace_budget_headroom_usd"],
        "budget_observations": [observation],
        "upload_state": "source-bound",
        "upload_error": None,
        "finalizer_state": "not-submitted",
        "function_call_id": None,
        "finalizer_error": None,
        "submission_claim_id": None,
        "remote_submission_claim_path": STAGE._remote_submission_claim_path(attempt_upload_id),
        "submission_claim_state": "not-created",
        "submission_claim_error": None,
    }
    return STAGE._validated_receipt(receipt)


def _validate_attempt_identity(
    receipt: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    mutable_fields = {
        "finalizer_state",
        "function_call_id",
        "finalizer_error",
        "submission_claim_id",
        "submission_claim_state",
        "submission_claim_error",
        "authorization_compute_charge_usd",
        "max_dollars",
        "workspace_budget_usd",
        "workspace_usage_before_submit_usd",
        "workspace_budget_headroom_usd",
        "budget_observations",
    }
    if set(receipt) != set(expected) or any(
        receipt[key] != value for key, value in expected.items() if key not in mutable_fields
    ):
        raise RuntimeMetadataRecoveryError("attempt receipt conflicts with source identity")
    if receipt["finalizer_state"] != "not-submitted" or receipt["function_call_id"] is not None:
        raise RuntimeMetadataRecoveryError("attempt was already submitted or is ambiguous")


def _put_recovery_claim(volume: Any, claim: Mapping[str, object]) -> None:
    path = cast(str, claim["remote_recovery_claim_path"])
    existing = _read_volume_json(volume, path, "source recovery claim")
    if existing is None:
        with volume.batch_upload(force=False) as batch:
            batch.put_file(io.BytesIO(_json_bytes(dict(claim))), path)
        existing = _read_volume_json(volume, path, "source recovery claim")
    if existing is None or existing[1] != claim:
        raise RuntimeMetadataRecoveryError("source recovery claim is missing or conflicts")


def _submit_attempt(
    modal_module: Any,
    volume: Any,
    receipt_path: Path,
    receipt: dict[str, Any],
    source_upload_id: str,
    recovery_claim: dict[str, object],
) -> None:
    if (
        receipt["finalizer_state"] != "not-submitted"
        or receipt["function_call_id"] is not None
        or receipt["submission_claim_state"] not in {"not-created", "created"}
    ):
        raise RuntimeMetadataRecoveryError("recovery attempt receipt is not fresh")
    if (
        receipt.get("source_recovery_claim") != recovery_claim
        or recovery_claim.get("source_upload_id") != source_upload_id
    ):
        raise RuntimeMetadataRecoveryError("attempt source claim identity changed")
    remote_source_claim = _read_volume_json(
        volume, cast(str, recovery_claim["remote_recovery_claim_path"]), "source recovery claim"
    )
    if remote_source_claim is None or remote_source_claim[1] != recovery_claim:
        raise RuntimeMetadataRecoveryError("remote source recovery claim is missing or conflicts")
    try:
        app, finalizer = STAGE.build_source_recovery_runtime(modal_module, volume)
    except BaseException as error:
        receipt["finalizer_error"] = STAGE._error_record(error)
        STAGE._write_json_atomic(receipt_path, receipt)
        raise
    if receipt["submission_claim_state"] == "not-created":
        STAGE._create_submission_claim(volume, receipt_path, receipt)
    claim_record = _read_volume_json(
        volume, receipt["remote_submission_claim_path"], "attempt submission claim"
    )
    if claim_record is None or STAGE._validated_submission_claim(claim_record[1], receipt) is None:
        raise RuntimeMetadataRecoveryError("attempt submission claim is missing or invalid")
    # A copied local intent must not authorize a second controller to spawn.
    # This marker is never overwritten or removed, even if its upload response
    # is lost. Ambiguous submission requires inspection, never an automatic retry.
    fence_path = cast(str, recovery_claim["remote_recovery_claim_path"]) + ".spawn.json"
    with volume.batch_upload(force=False) as batch:
        batch.put_file(
            io.BytesIO(
                _json_bytes(
                    {
                        "schema": "sion-modal-source-recovery-spawn-v1",
                        "attempt_upload_id": receipt["upload_id"],
                        "source_recovery_claim_sha256": _sha256_bytes(_json_bytes(recovery_claim)),
                    }
                )
            ),
            fence_path,
        )
    receipt["finalizer_state"] = "submitting"
    receipt["finalizer_error"] = None
    STAGE._write_json_atomic(receipt_path, receipt)
    app_entered = False
    try:
        with app.run(detach=True):
            app_entered = True
            function_call = finalizer.spawn(
                source_upload_id,
                receipt["upload_id"],
                receipt["bundle_sha256"],
                receipt["bundle_size"],
                receipt["runtime_contract_sha256"],
                recovery_claim,
            )
            call_id = function_call.object_id
            if (
                not isinstance(call_id, str)
                or STAGE.FUNCTION_CALL_ID_PATTERN.fullmatch(call_id) is None
            ):
                raise RuntimeError("Modal returned an invalid recovery FunctionCall ID")
            receipt["function_call_id"] = call_id
            receipt["finalizer_state"] = "submitted"
            STAGE._write_json_atomic(receipt_path, receipt)
    except BaseException as error:
        if receipt["function_call_id"] is None:
            receipt["finalizer_state"] = "submission-unknown" if app_entered else "not-submitted"
        receipt["finalizer_error"] = STAGE._error_record(error)
        STAGE._write_json_atomic(receipt_path, receipt)
        raise


def recover(
    receipt_path: Path,
    failed_app_id: str,
    *,
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
    preflight_only: bool,
) -> Path:
    resolved_source_path = receipt_path.resolve()
    source = STAGE._read_receipt(resolved_source_path)
    source_call_id = source.get("function_call_id")
    if source.get("finalizer_state") != "submitted" or not isinstance(source_call_id, str):
        raise RuntimeMetadataRecoveryError("source receipt is not a submitted failed call")
    STAGE.validate_finalizer_cost_guard(max_dollars)
    STAGE._validate_workspace_budget_guard(max_dollars, workspace_budget, workspace_usage)
    modal_module = STAGE._require_modal()
    _assert_app_stopped(failed_app_id)
    _assert_terminal_failed_call(modal_module, source_call_id, cast(str, source["upload_id"]))
    replacement_commit, replacement_tree = _replacement_revision()
    replacement_runtime = STAGE.finalizer_runtime_contract_sha256(REPOSITORY_ROOT)
    builder_sha256 = _sha256_file(Path(__file__), "runtime metadata recovery controller")
    source_receipt_sha256 = _sha256_file(resolved_source_path, "source Modal receipt")
    local_size, local_sha256 = STAGE._hash_regular_file_stable(
        Path(cast(str, source["local_bundle_path"])),
        "local prepared bundle",
    )
    if (local_size, local_sha256) != (source["bundle_size"], source["bundle_sha256"]):
        raise RuntimeMetadataRecoveryError("local archive no longer matches the source receipt")

    volume = modal_module.Volume.from_name(
        source["volume_name"],
        create_if_missing=False,
        version=STAGE.VOLUME_VERSION,
    )
    evidence = _validated_source_evidence(volume, resolved_source_path, source)
    _validate_remote_incoming(volume, source)
    original_claim_record = _read_volume_json(
        volume,
        cast(str, source["remote_submission_claim_path"]),
        "original submission claim",
    )
    if original_claim_record is None:
        raise RuntimeMetadataRecoveryError("original submission claim is missing")
    original_claim_bytes, original_claim = original_claim_record
    original_runtime = STAGE._validate_sha256(original_claim.get("runtime_contract_sha256"))
    if (
        STAGE._validated_submission_claim(
            original_claim, {**source, "runtime_contract_sha256": original_runtime}
        )
        is None
    ):
        raise RuntimeMetadataRecoveryError("original submission claim is invalid")

    preflight = {
        "source_receipt_path": str(resolved_source_path),
        "source_upload_id": source["upload_id"],
        "source_function_call_id": source_call_id,
        "source_app_id": failed_app_id,
        "bundle_sha256": local_sha256,
        "bundle_size": local_size,
        "replacement_commit": replacement_commit,
        "replacement_tree": replacement_tree,
        "replacement_runtime_contract_sha256": replacement_runtime,
        "recovery_builder_sha256": builder_sha256,
        "source_receipt_sha256": source_receipt_sha256,
        "source_status_sha256": evidence["source_status_sha256"],
        "source_failure_sha256": evidence["source_failure_sha256"],
        "original_submission_claim_sha256": _sha256_bytes(original_claim_bytes),
        "remote_source_valid": True,
        "archive_reuploaded": False,
        "replacement_submitted": False,
    }
    if preflight_only:
        print(_json_bytes(preflight).decode(), end="")
        return resolved_source_path

    receipt_root = STAGE._existing_receipt_root(resolved_source_path, source)
    recovery_root = resolved_source_path.parent / "runtime-metadata-recovery"
    intent_path = resolved_source_path.parent / "runtime-metadata-recovery-intent.json"
    with STAGE._exclusive_submission(receipt_root):
        if STAGE._read_receipt(resolved_source_path) != source:
            raise RuntimeMetadataRecoveryError("source receipt changed during recovery validation")
        if intent_path.exists():
            intent = _read_json(intent_path, "runtime metadata recovery intent")
            attempt_upload_id = STAGE._validate_upload_id(intent.get("attempt_upload_id"))
            recovery_claim_id = STAGE._validate_submission_claim_id(intent.get("recovery_claim_id"))
        else:
            attempt_upload_id = STAGE._new_upload_id()
            recovery_claim_id = STAGE._new_submission_claim_id()
            intent = {
                "schema": RECOVERY_SCHEMA,
                "created_at_utc": _utc_now(),
                **preflight,
                "attempt_upload_id": attempt_upload_id,
                "attempt_receipt_path": str(STAGE._receipt_path(recovery_root, attempt_upload_id)),
                "recovery_claim_id": recovery_claim_id,
                "state": "intent",
            }
            STAGE._write_json_atomic(intent_path, intent)
        immutable = {
            "schema": RECOVERY_SCHEMA,
            **preflight,
            "attempt_upload_id": attempt_upload_id,
            "attempt_receipt_path": str(STAGE._receipt_path(recovery_root, attempt_upload_id)),
            "recovery_claim_id": recovery_claim_id,
        }
        if any(intent.get(key) != value for key, value in immutable.items()):
            raise RuntimeMetadataRecoveryError("runtime metadata recovery intent identity changed")
        if intent.get("state") not in {"intent", "source-claimed", "not-submitted"}:
            raise RuntimeMetadataRecoveryError("runtime metadata recovery was already submitted")

        recovery_claim = STAGE._source_recovery_claim_payload(
            source_upload_id=source["upload_id"],
            attempt_upload_id=attempt_upload_id,
            bundle_sha256=source["bundle_sha256"],
            bundle_size=source["bundle_size"],
            source_function_call_id=source_call_id,
            source_app_id=failed_app_id,
            source_runtime_contract_sha256=source["runtime_contract_sha256"],
            replacement_runtime_contract_sha256=replacement_runtime,
            replacement_commit=replacement_commit,
            replacement_tree=replacement_tree,
            recovery_builder_sha256=builder_sha256,
            source_status_sha256=evidence["source_status_sha256"],
            source_failure_sha256=evidence["source_failure_sha256"],
            source_receipt_sha256=source_receipt_sha256,
            original_submission_claim_sha256=_sha256_bytes(original_claim_bytes),
            original_submission_claim_id=source["submission_claim_id"],
            original_submission_runtime_contract_sha256=original_runtime,
            source_receipt_created_at_utc=source["created_at_utc"],
            recovery_claim_id=recovery_claim_id,
        )
        attempt_receipt_path = STAGE._receipt_path(recovery_root, attempt_upload_id)
        expected_receipt = _attempt_receipt(
            source,
            attempt_upload_id,
            replacement_runtime,
            cast(str, intent["created_at_utc"]),
            max_dollars=max_dollars,
            workspace_budget=workspace_budget,
            workspace_usage=workspace_usage,
            recovery_claim=recovery_claim,
        )
        attempt_receipt = (
            STAGE._read_receipt(attempt_receipt_path)
            if attempt_receipt_path.exists()
            else expected_receipt
        )
        _validate_attempt_identity(attempt_receipt, expected_receipt)
        STAGE._refresh_receipt_budget(
            attempt_receipt,
            max_dollars=max_dollars,
            workspace_budget=workspace_budget,
            workspace_usage=workspace_usage,
        )
        STAGE._write_json_atomic(attempt_receipt_path, attempt_receipt)
        _put_recovery_claim(volume, recovery_claim)
        intent["state"] = "source-claimed"
        intent["remote_recovery_claim_path"] = recovery_claim["remote_recovery_claim_path"]
        STAGE._write_json_atomic(intent_path, intent)

        _assert_app_stopped(failed_app_id)
        _assert_terminal_failed_call(modal_module, source_call_id, cast(str, source["upload_id"]))
        current_evidence = _validated_source_evidence(volume, resolved_source_path, source)
        if (
            current_evidence["source_status_sha256"] != evidence["source_status_sha256"]
            or current_evidence["source_failure_sha256"] != evidence["source_failure_sha256"]
        ):
            raise RuntimeMetadataRecoveryError("source journal changed before submission")
        _validate_remote_incoming(volume, source)
        if _replacement_revision() != (replacement_commit, replacement_tree) or (
            STAGE.finalizer_runtime_contract_sha256(REPOSITORY_ROOT) != replacement_runtime
            or _sha256_file(resolved_source_path, "source receipt") != source_receipt_sha256
        ):
            raise RuntimeMetadataRecoveryError("local provenance changed before submission")
        _assert_remote_path_absent(
            volume,
            cast(str, attempt_receipt["remote_operation_path"]),
            "recovery attempt operation",
        )
        intent["state"] = "submitting"
        STAGE._write_json_atomic(intent_path, intent)
        try:
            _submit_attempt(
                modal_module,
                volume,
                attempt_receipt_path,
                attempt_receipt,
                cast(str, source["upload_id"]),
                recovery_claim,
            )
        except BaseException:
            observed = STAGE._read_receipt(attempt_receipt_path)
            intent["state"] = observed["finalizer_state"]
            intent["replacement_function_call_id"] = observed["function_call_id"]
            STAGE._write_json_atomic(intent_path, intent)
            raise
        submitted = STAGE._read_receipt(attempt_receipt_path)
        intent["state"] = "submitted"
        intent["replacement_function_call_id"] = submitted["function_call_id"]
        STAGE._write_json_atomic(intent_path, intent)

    print(
        _json_bytes(
            {
                "receipt_path": str(attempt_receipt_path),
                "source_receipt_path": str(resolved_source_path),
                "source_function_call_id": source_call_id,
                "replacement_function_call_id": submitted["function_call_id"],
                "replacement_commit": replacement_commit,
                "replacement_runtime_contract_sha256": replacement_runtime,
                "archive_reuploaded": False,
            }
        ).decode(),
        end="",
    )
    return attempt_receipt_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--failed-app-id", required=True)
    parser.add_argument("--max-dollars", type=float, required=True)
    parser.add_argument("--workspace-budget", type=float, required=True)
    parser.add_argument("--workspace-usage", type=float, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    recover(
        arguments.receipt,
        arguments.failed_app_id,
        max_dollars=arguments.max_dollars,
        workspace_budget=arguments.workspace_budget,
        workspace_usage=arguments.workspace_usage,
        preflight_only=arguments.preflight_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
