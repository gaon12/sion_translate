"""Recover one legacy Modal bundle upload after a pre-spawn disk rejection.

Modal changed its container disk contract so requests below the default 512 GiB
quota are rejected while the Function definition is created. Older
``modal_stage_gpu_bundle.py`` revisions requested 2 GiB explicitly. The upload
and its immutable submission claim can already exist when that rejection is
reported.

This tool never uploads an archive. It reconstructs the exact runtime recorded
by the receipt, proves that the rejected ``FunctionCreate`` happened before the
``app.run`` body and ``spawn``, validates the remote claim and absence of an
operation journal, and submits the original finalizer once with the disk option
set to ``None`` in memory. The reviewed image bytes remain unchanged, so the
receipt and remote claim keep their original runtime identity.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence, cast


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_STAGE_PATH = Path("scripts/modal_stage_gpu_bundle.py")
COPIED_RUNTIME_PATHS = (
    Path("scripts/package_gpu_bundle.py"),
    Path("requirements/modal-bundle-stage.txt"),
)
SOURCE_PACKAGE_PATH = Path("src/sion_translate")
RECOVERY_SCHEMA = "sion-modal-bundle-disk-recovery-v1"
DESERIALIZATION_RECOVERY_SCHEMA = "sion-modal-bundle-deserialization-recovery-v1"
DESERIALIZATION_CLAIM_SCHEMA = "sion-modal-bundle-deserialization-claim-v1"
MOUNT_RECOVERY_SCHEMA = "sion-modal-bundle-mount-recovery-v1"
MOUNT_RECOVERY_CLAIM_SCHEMA = "sion-modal-bundle-mount-recovery-claim-v1"
EXPECTED_ERROR_TYPE = "InvalidError"
EXPECTED_ERROR_MESSAGE = (
    "Function disk request out of bounds: 2048 MiB. Must be between 524288 and 3145728 MiB."
)
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
APP_ID_PATTERN = re.compile(r"^ap-[A-Za-z0-9_-]{8,128}$")
EXPECTED_DESERIALIZATION_MODULE = "sion_legacy_modal_stage_runtime"
IMPORTABLE_STAGE_MODULE = "modal_stage_gpu_bundle"
EXPECTED_DESERIALIZATION_ROOT_CAUSE = (
    f"ModuleNotFoundError: No module named '{EXPECTED_DESERIALIZATION_MODULE}'"
)
EXPECTED_DESERIALIZATION_ERROR = (
    "modal.exception.DeserializationError: Deserialization failed because the "
    f"'{EXPECTED_DESERIALIZATION_MODULE}' module is not available in the remote environment."
)
EXPECTED_STOP_LOG = "Stopping app - user stopped from CLI."
EXPECTED_MOUNT_ERROR = (
    "modal_stage_gpu_bundle.BundleStageError: Modal Volume mount must be a regular "
    "non-symlink directory: /mnt/sion-bundles"
)
EXPECTED_TERMINAL_MESSAGE = (
    "durable Modal bundle finalizer bundle-20260902t065624z-174a5e7835536906 failed; "
    "inspect its Volume journal"
)


class LegacyRecoveryError(RuntimeError):
    """Raised when historical evidence is insufficient for a safe retry."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LegacyRecoveryError(f"{label} is not a regular file: {path}")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise LegacyRecoveryError(f"{label} is unreasonably large: {path}")
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LegacyRecoveryError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise LegacyRecoveryError(f"{label} is not a JSON object: {path}")
    return cast(dict[str, Any], value)


def _git_blob(repository_root: Path, commit: str, relative_path: Path) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "show",
            f"{commit}:{relative_path.as_posix()}",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace")[-2_000:]
        raise LegacyRecoveryError(
            f"cannot reconstruct {relative_path.as_posix()} from {commit}: {diagnostic}"
        )
    return completed.stdout


def _copy_regular_file(source: Path, destination: Path) -> None:
    metadata = source.lstat()
    if source.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise LegacyRecoveryError(f"legacy runtime source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _materialize_legacy_runtime(
    repository_root: Path,
    commit: str,
    destination: Path,
) -> Path:
    """Rebuild the receipt runtime without changing its recorded image bytes."""

    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise LegacyRecoveryError("legacy runtime commit is invalid")
    destination.mkdir(parents=True, exist_ok=False)
    stage_destination = destination / LEGACY_STAGE_PATH
    stage_destination.parent.mkdir(parents=True, exist_ok=True)
    stage_destination.write_bytes(_git_blob(repository_root, commit, LEGACY_STAGE_PATH))
    for relative_path in COPIED_RUNTIME_PATHS:
        _copy_regular_file(repository_root / relative_path, destination / relative_path)
    source_root = repository_root / SOURCE_PACKAGE_PATH
    if source_root.is_symlink() or not source_root.is_dir():
        raise LegacyRecoveryError(
            f"legacy source package is not a regular directory: {source_root}"
        )
    for source in sorted(source_root.rglob("*"), key=lambda path: path.as_posix()):
        relative = source.relative_to(repository_root)
        if "__pycache__" in relative.parts or source.suffix in {".pyc", ".pyo"}:
            continue
        metadata = source.lstat()
        if source.is_symlink():
            raise LegacyRecoveryError(f"legacy source package contains a link: {source}")
        if stat.S_ISDIR(metadata.st_mode):
            (destination / relative).mkdir(parents=True, exist_ok=True)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise LegacyRecoveryError(f"legacy source package contains a special file: {source}")
        _copy_regular_file(source, destination / relative)
    return stage_destination


def _materialize_committed_runtime(
    repository_root: Path,
    commit: str,
    destination: Path,
) -> Path:
    """Materialize every worker byte from one reviewed Git commit."""

    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise LegacyRecoveryError("replacement runtime commit is invalid")
    listing = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            SOURCE_PACKAGE_PATH.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if listing.returncode != 0:
        raise LegacyRecoveryError(
            f"cannot list committed runtime source: {listing.stderr[-2_000:]}"
        )
    source_files = [Path(line) for line in listing.stdout.splitlines() if line]
    if not source_files or any(path.suffix != ".py" for path in source_files):
        raise LegacyRecoveryError("committed runtime source listing is empty or invalid")
    runtime_files = (LEGACY_STAGE_PATH, *COPIED_RUNTIME_PATHS, *source_files)
    destination.mkdir(parents=True, exist_ok=False)
    for relative_path in runtime_files:
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_git_blob(repository_root, commit, relative_path))
    return destination / LEGACY_STAGE_PATH


def _load_legacy_module(stage_path: Path) -> ModuleType:
    name = IMPORTABLE_STAGE_MODULE
    spec = importlib.util.spec_from_file_location(name, stage_path)
    if spec is None or spec.loader is None:
        raise LegacyRecoveryError(f"cannot import reconstructed runtime: {stage_path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def _build_importable_finalizer_runtime(
    legacy: Any,
    modal_module: Any,
    volume: Any,
) -> tuple[Any, Any]:
    """Build the historical worker while importing its attested image source remotely."""

    remote_source = legacy.REMOTE_ROOT / "src"
    remote_scripts = legacy.REMOTE_ROOT / "scripts"
    image = (
        modal_module.Image.debian_slim(python_version="3.11")
        .pip_install_from_requirements(
            str(legacy.FINALIZER_REQUIREMENTS),
            extra_options="--require-hashes --only-binary=:all: --no-cache-dir",
        )
        .add_local_dir(
            str(legacy.REPOSITORY_ROOT / legacy.SOURCE_PACKAGE_RELATIVE_PATH),
            str(legacy.REMOTE_SOURCE_PACKAGE),
            copy=True,
            ignore=("**/__pycache__/**", "**/*.pyc", "**/*.pyo"),
        )
        .add_local_file(
            str(legacy.PACKAGE_SCRIPT),
            str(legacy.REMOTE_PACKAGE_SCRIPT),
            copy=True,
        )
        .add_local_file(
            str(legacy.STAGE_SCRIPT),
            str(legacy.REMOTE_STAGE_SCRIPT),
            copy=True,
        )
        .add_local_file(
            str(legacy.FINALIZER_REQUIREMENTS),
            str(legacy.REMOTE_FINALIZER_REQUIREMENTS),
            copy=True,
        )
        .env(
            {
                "PYTHONPATH": f"{remote_source}:{remote_scripts}",
                "PYTHONUNBUFFERED": "1",
            }
        )
    )
    app = modal_module.App(legacy.APP_NAME, image=image, include_source=False)

    @app.function(
        name=legacy.FINALIZER_FUNCTION_NAME,
        volumes={str(legacy.VOLUME_MOUNT): volume},
        cpu=legacy.FINALIZER_CPU_CORES,
        memory=legacy.FINALIZER_MEMORY_MIB,
        timeout=legacy.FINALIZER_TIMEOUT_SECONDS,
        retries=0,
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=legacy.FINALIZER_SCALEDOWN_WINDOW_SECONDS,
        single_use_containers=True,
        serialized=True,
        include_source=False,
    )
    def finalize(
        upload_id: str,
        expected_sha256: str,
        expected_size: int,
        expected_runtime_contract_sha256: str,
    ) -> dict[str, object]:
        function_call_id = modal_module.current_function_call_id()
        if function_call_id is None:
            raise RuntimeError("Modal did not expose the bundle finalizer FunctionCall ID")
        package_module = legacy._load_package_module(Path(str(legacy.REMOTE_PACKAGE_SCRIPT)))
        result = legacy._finalize_bundle(
            volume,
            Path(str(legacy.VOLUME_MOUNT)),
            upload_id,
            expected_sha256,
            expected_size,
            function_call_id,
            expected_runtime_contract_sha256,
            package_module=package_module,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return result

    return app, finalize


def _install_importable_runtime_builder(legacy: Any) -> None:
    legacy.FINALIZER_EPHEMERAL_DISK_MIB = None
    legacy._build_finalizer_runtime = lambda modal_module, volume: (
        _build_importable_finalizer_runtime(legacy, modal_module, volume)
    )


def _validate_rejected_receipt(receipt: Mapping[str, object]) -> None:
    error = receipt.get("finalizer_error")
    if (
        receipt.get("upload_state") != "uploaded"
        or receipt.get("submission_claim_state") != "created"
        or receipt.get("finalizer_state") != "submission-unknown"
        or receipt.get("function_call_id") is not None
        or not isinstance(error, dict)
        or error.get("error_type") != EXPECTED_ERROR_TYPE
        or error.get("message") != EXPECTED_ERROR_MESSAGE
    ):
        raise LegacyRecoveryError(
            "receipt is not the exact pre-spawn Modal disk rejection supported by this tool"
        )


def _validate_status_snapshot(
    snapshot: Mapping[str, object], receipt: Mapping[str, object]
) -> None:
    if (
        snapshot.get("receipt") != receipt
        or snapshot.get("recovered_state") != "submission-unknown"
        or snapshot.get("function_call_state") != "identity-unavailable"
        or snapshot.get("function_call_id_source") != "unavailable"
        or snapshot.get("observed_function_call_id") is not None
        or snapshot.get("function_call_result") is not None
        or snapshot.get("function_call_error") is not None
        or snapshot.get("remote_status") is not None
        or snapshot.get("remote_result") is not None
        or snapshot.get("remote_failure") is not None
    ):
        raise LegacyRecoveryError(
            "status snapshot does not prove that the rejected finalizer has no remote identity"
        )


def _validate_failed_deserialization_receipt(receipt: Mapping[str, object]) -> str:
    call_id = receipt.get("function_call_id")
    if (
        receipt.get("upload_state") != "uploaded"
        or receipt.get("submission_claim_state") != "created"
        or receipt.get("finalizer_state") != "submitted"
        or not isinstance(call_id, str)
        or receipt.get("finalizer_error") is not None
    ):
        raise LegacyRecoveryError(
            "receipt is not a submitted Modal finalizer eligible for deserialization recovery"
        )
    return call_id


def _validate_failed_deserialization_status(
    snapshot: Mapping[str, object],
    receipt: Mapping[str, object],
    call_id: str,
) -> None:
    error = snapshot.get("function_call_error")
    if (
        snapshot.get("receipt") != receipt
        or snapshot.get("recovered_state") != "failed"
        or snapshot.get("function_call_state") != "failed"
        or snapshot.get("function_call_id_source") != "receipt"
        or snapshot.get("observed_function_call_id") != call_id
        or snapshot.get("function_call_result") is not None
        or not isinstance(error, dict)
        or error.get("error_type") != "RemoteError"
        or snapshot.get("remote_status") is not None
        or snapshot.get("remote_result") is not None
        or snapshot.get("remote_failure") is not None
    ):
        raise LegacyRecoveryError(
            "status snapshot does not prove one terminal pre-journal finalizer failure"
        )


def _validate_failed_mount_status(
    snapshot: Mapping[str, object],
    receipt: Mapping[str, object],
    call_id: str,
) -> None:
    error = snapshot.get("function_call_error")
    expected_message = (
        f"durable Modal bundle finalizer {receipt['upload_id']} failed; inspect its Volume journal"
    )
    if (
        snapshot.get("receipt") != receipt
        or snapshot.get("recovered_state") != "failed"
        or snapshot.get("function_call_state") != "failed"
        or snapshot.get("function_call_id_source") != "receipt"
        or snapshot.get("observed_function_call_id") != call_id
        or snapshot.get("function_call_result") is not None
        or not isinstance(error, dict)
        or error.get("error_type") != "RuntimeError"
        or error.get("message") != expected_message
        or snapshot.get("remote_status") is not None
        or snapshot.get("remote_result") is not None
        or snapshot.get("remote_failure") is not None
    ):
        raise LegacyRecoveryError(
            "status snapshot does not prove the supported pre-journal mount failure"
        )


def _fetch_failed_app_logs(app_id: str, call_id: str) -> str:
    if APP_ID_PATTERN.fullmatch(app_id) is None:
        raise LegacyRecoveryError("failed Modal App ID is invalid")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "app",
            "logs",
            app_id,
            "--tail",
            "1000",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    logs = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise LegacyRecoveryError(f"cannot fetch stopped Modal App logs: {logs[-2_000:]}")
    if (
        EXPECTED_DESERIALIZATION_ROOT_CAUSE not in logs
        or EXPECTED_DESERIALIZATION_ERROR not in logs
        or EXPECTED_STOP_LOG not in logs
    ):
        raise LegacyRecoveryError(
            "stopped Modal App logs do not prove the supported deserialization failure"
        )
    return logs


def _fetch_mount_failure_logs(app_id: str) -> str:
    if APP_ID_PATTERN.fullmatch(app_id) is None:
        raise LegacyRecoveryError("failed Modal App ID is invalid")
    completed = subprocess.run(
        [sys.executable, "-m", "modal", "app", "logs", app_id, "--tail", "1000"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    logs = completed.stdout + completed.stderr
    if completed.returncode != 0 or EXPECTED_MOUNT_ERROR not in logs:
        raise LegacyRecoveryError(
            f"stopped Modal App logs do not prove the mount failure: {logs[-2_000:]}"
        )
    return logs


def _assert_app_stopped(app_id: str) -> None:
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
        raise LegacyRecoveryError("cannot parse Modal App state") from error
    if completed.returncode != 0 or not isinstance(apps, list):
        raise LegacyRecoveryError("cannot read Modal App state")
    matches = [app for app in apps if isinstance(app, dict) and app.get("app_id") == app_id]
    if len(matches) != 1 or matches[0].get("state") != "stopped" or matches[0].get("tasks") != "0":
        raise LegacyRecoveryError("failed Modal App is not stopped with zero tasks")


def _assert_terminal_failed_call(
    modal_module: Any,
    call_id: str,
    *,
    expected_error_type: str = "RemoteError",
    expected_message: str = "",
) -> None:
    call = modal_module.FunctionCall.from_id(call_id)
    try:
        call.get(timeout=0)
    except TimeoutError as error:
        raise LegacyRecoveryError("failed Modal FunctionCall is still pending") from error
    except BaseException as error:
        if type(error).__name__ != expected_error_type or str(error) != expected_message:
            raise LegacyRecoveryError(
                "failed Modal FunctionCall has an unsupported terminal result"
            ) from error
    else:
        raise LegacyRecoveryError("failed Modal FunctionCall unexpectedly returned a result")


def _deserialization_claim_path(upload_id: str) -> str:
    return f"/recovery-claims/{upload_id}/deserialization-v1.json"


def _mount_claim_path(upload_id: str, failed_call_id: str) -> str:
    return f"/recovery-claims/{upload_id}/mount-symlink-{failed_call_id}.json"


def _deserialization_claim_payload(
    receipt: Mapping[str, object],
    *,
    failed_app_id: str,
    failed_call_id: str,
    recovery_claim_id: str,
) -> dict[str, object]:
    return {
        "schema": DESERIALIZATION_CLAIM_SCHEMA,
        "upload_id": receipt["upload_id"],
        "bundle_sha256": receipt["bundle_sha256"],
        "bundle_size": receipt["bundle_size"],
        "runtime_contract_sha256": receipt["runtime_contract_sha256"],
        "original_submission_claim_id": receipt["submission_claim_id"],
        "failed_app_id": failed_app_id,
        "failed_function_call_id": failed_call_id,
        "failure_module": EXPECTED_DESERIALIZATION_MODULE,
        "recovery_claim_id": recovery_claim_id,
    }


def _runtime_builder_sha256() -> str:
    source = inspect.getsource(_build_importable_finalizer_runtime).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _receipt_identity_projection(receipt: Mapping[str, object]) -> dict[str, object]:
    """Select fields a replacement finalizer is never allowed to redirect."""

    fields = (
        "receipt_version",
        "upload_id",
        "volume_name",
        "volume_version",
        "app_name",
        "function_name",
        "local_bundle_path",
        "bundle_sha256",
        "bundle_size",
        "verification",
        "remote_incoming_path",
        "remote_final_path",
        "remote_operation_path",
        "created_at_utc",
        "upload_state",
        "upload_error",
        "submission_claim_id",
        "remote_submission_claim_path",
        "submission_claim_state",
        "submission_claim_error",
    )
    try:
        return {field: receipt[field] for field in fields}
    except KeyError as error:
        raise LegacyRecoveryError(
            f"Modal recovery receipt is missing identity field {error.args[0]}"
        ) from error


def _assert_remote_path_absent(volume: Any, path: str, label: str) -> None:
    """Distinguish a missing Volume path from an existing empty directory."""

    try:
        iterator = iter(volume.iterdir(path, recursive=False))
        try:
            next(iterator)
        except StopIteration:
            pass
    except BaseException as error:
        is_local_not_found = isinstance(error, FileNotFoundError)
        is_modal_not_found = (
            type(error).__module__ == "modal.exception" and type(error).__name__ == "NotFoundError"
        )
        if is_local_not_found or is_modal_not_found:
            return
        raise
    raise LegacyRecoveryError(f"{label} already exists on the Modal Volume: {path}")


def _validate_remote_incoming_archive(volume: Any, receipt: Mapping[str, object]) -> None:
    incoming_path = cast(str, receipt["remote_incoming_path"])
    entries = list(volume.iterdir(incoming_path, recursive=False))
    if len(entries) != 1:
        raise LegacyRecoveryError("remote incoming directory does not contain exactly one entry")
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
        raise LegacyRecoveryError("remote incoming archive identity is invalid")


def _preserve_recovery_evidence(
    legacy: Any,
    receipt_path: Path,
    receipt: Mapping[str, object],
    *,
    runtime_commit: str,
) -> None:
    preserved_path = receipt_path.parent / "receipt-before-disk-recovery.json"
    if preserved_path.exists():
        if _read_json(preserved_path, "preserved recovery receipt") != receipt:
            raise LegacyRecoveryError("preserved recovery receipt conflicts with current evidence")
    else:
        legacy._write_json_atomic(preserved_path, receipt)
    intent_path = receipt_path.parent / "disk-recovery-intent.json"
    if intent_path.exists():
        raise LegacyRecoveryError(
            "a disk-recovery intent already exists; inspect it before retrying"
        )
    legacy._write_json_atomic(
        intent_path,
        {
            "schema": RECOVERY_SCHEMA,
            "recorded_at_utc": _utc_now(),
            "runtime_commit": runtime_commit,
            "runtime_contract_sha256": receipt["runtime_contract_sha256"],
            "bundle_sha256": receipt["bundle_sha256"],
            "upload_id": receipt["upload_id"],
            "supported_error_type": EXPECTED_ERROR_TYPE,
            "supported_error_message": EXPECTED_ERROR_MESSAGE,
            "archive_reuploaded": False,
        },
    )


def _recover_with_legacy_module(
    legacy: Any,
    receipt_path: Path,
    *,
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
) -> Path:
    """Validate the historical operation and submit only its original finalizer."""

    resolved_receipt_path = receipt_path.resolve()
    initial_receipt = legacy._read_receipt(resolved_receipt_path)
    _validate_rejected_receipt(initial_receipt)
    snapshot = _read_json(
        resolved_receipt_path.parent / "status-latest.json",
        "pre-recovery Modal status snapshot",
    )
    _validate_status_snapshot(snapshot, initial_receipt)
    runtime_commit = initial_receipt.get("verification", {}).get("git_commit")
    if not isinstance(runtime_commit, str) or COMMIT_PATTERN.fullmatch(runtime_commit) is None:
        raise LegacyRecoveryError("receipt does not contain a valid runtime commit")
    observed_runtime = legacy.finalizer_runtime_contract_sha256(legacy.REPOSITORY_ROOT)
    if observed_runtime != initial_receipt.get("runtime_contract_sha256"):
        raise LegacyRecoveryError("reconstructed runtime does not match the immutable receipt")
    bundle_path = Path(cast(str, initial_receipt["local_bundle_path"]))
    observed_size, observed_sha256 = legacy._hash_regular_file_stable(
        bundle_path,
        "legacy prepared GPU bundle",
    )
    if (observed_size, observed_sha256) != (
        initial_receipt["bundle_size"],
        initial_receipt["bundle_sha256"],
    ):
        raise LegacyRecoveryError("local archive no longer matches the uploaded receipt")
    modal_module = legacy._require_modal()
    legacy.validate_finalizer_cost_guard(max_dollars)
    legacy._validate_workspace_budget_guard(max_dollars, workspace_budget, workspace_usage)
    receipt_root = legacy._existing_receipt_root(resolved_receipt_path, initial_receipt)
    with legacy._exclusive_submission(receipt_root):
        receipt = legacy._read_receipt(resolved_receipt_path)
        if receipt != initial_receipt:
            raise LegacyRecoveryError("receipt changed after recovery validation")
        legacy._assert_no_unresolved_receipts(
            receipt_root,
            exclude_upload_id=cast(str, receipt["upload_id"]),
        )
        volume = modal_module.Volume.from_name(
            receipt["volume_name"],
            create_if_missing=False,
            version=legacy.VOLUME_VERSION,
        )
        operation_path = cast(str, receipt["remote_operation_path"])
        remote_operation = {
            name: legacy._read_volume_json(volume, f"{operation_path}/{name}.json")
            for name in ("status", "result", "failure")
        }
        if any(value is not None for value in remote_operation.values()):
            raise LegacyRecoveryError(
                "a remote finalizer journal now exists; recover status instead of resubmitting"
            )
        raw_claim = legacy._read_volume_json(
            volume,
            cast(str, receipt["remote_submission_claim_path"]),
        )
        if legacy._validated_submission_claim(raw_claim, receipt) is None:
            raise LegacyRecoveryError("the immutable remote submission claim is missing")
        _preserve_recovery_evidence(
            legacy,
            resolved_receipt_path,
            receipt,
            runtime_commit=runtime_commit,
        )
        receipt["finalizer_state"] = "not-submitted"
        legacy._refresh_receipt_budget(
            receipt,
            max_dollars=max_dollars,
            workspace_budget=workspace_budget,
            workspace_usage=workspace_usage,
        )
        legacy._write_json_atomic(resolved_receipt_path, receipt)
        if not hasattr(legacy, "FINALIZER_EPHEMERAL_DISK_MIB"):
            raise LegacyRecoveryError("reconstructed runtime is not the supported legacy revision")
        legacy.FINALIZER_EPHEMERAL_DISK_MIB = None
        legacy._submit_finalizer(modal_module, volume, resolved_receipt_path, receipt)
    print(
        legacy._json_bytes(
            {
                "receipt_path": str(resolved_receipt_path),
                "archive_reuploaded": False,
                **receipt,
            }
        ).decode(),
        end="",
    )
    return resolved_receipt_path


def _recover_deserialization_with_legacy_module(
    legacy: Any,
    receipt_path: Path,
    *,
    failed_app_id: str,
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
    preflight_only: bool = False,
) -> Path:
    """Replace one terminal pre-user-code call without uploading its archive again."""

    resolved_receipt_path = receipt_path.resolve()
    initial_receipt = legacy._read_receipt(resolved_receipt_path)
    failed_call_id = _validate_failed_deserialization_receipt(initial_receipt)
    snapshot = _read_json(
        resolved_receipt_path.parent / "status-latest.json",
        "failed Modal status snapshot",
    )
    _validate_failed_deserialization_status(snapshot, initial_receipt, failed_call_id)
    logs = _fetch_failed_app_logs(failed_app_id, failed_call_id)
    modal_module = legacy._require_modal()
    _assert_terminal_failed_call(modal_module, failed_call_id)
    legacy.validate_finalizer_cost_guard(max_dollars)
    legacy._validate_workspace_budget_guard(max_dollars, workspace_budget, workspace_usage)

    observed_runtime = legacy.finalizer_runtime_contract_sha256(legacy.REPOSITORY_ROOT)
    if observed_runtime != initial_receipt.get("runtime_contract_sha256"):
        raise LegacyRecoveryError("reconstructed runtime does not match the immutable receipt")
    bundle_path = Path(cast(str, initial_receipt["local_bundle_path"]))
    observed_size, observed_sha256 = legacy._hash_regular_file_stable(
        bundle_path,
        "legacy prepared GPU bundle",
    )
    if (observed_size, observed_sha256) != (
        initial_receipt["bundle_size"],
        initial_receipt["bundle_sha256"],
    ):
        raise LegacyRecoveryError("local archive no longer matches the uploaded receipt")

    receipt_root = legacy._existing_receipt_root(resolved_receipt_path, initial_receipt)
    recovery_root = resolved_receipt_path.parent / "deserialization-recovery"
    recovery_receipt_path = legacy._receipt_path(
        recovery_root,
        cast(str, initial_receipt["upload_id"]),
    )
    intent_path = resolved_receipt_path.parent / "deserialization-recovery-intent.json"
    remote_claim_path = _deserialization_claim_path(cast(str, initial_receipt["upload_id"]))

    with legacy._exclusive_submission(receipt_root):
        if legacy._read_receipt(resolved_receipt_path) != initial_receipt:
            raise LegacyRecoveryError("original receipt changed during recovery validation")
        volume = modal_module.Volume.from_name(
            initial_receipt["volume_name"],
            create_if_missing=False,
            version=legacy.VOLUME_VERSION,
        )
        operation_path = cast(str, initial_receipt["remote_operation_path"])
        remote_operation = {
            name: legacy._read_volume_json(volume, f"{operation_path}/{name}.json")
            for name in ("status", "result", "failure")
        }
        if any(value is not None for value in remote_operation.values()):
            raise LegacyRecoveryError(
                "a remote finalizer journal now exists; recover status instead of replacing the call"
            )
        original_claim = legacy._read_volume_json(
            volume,
            cast(str, initial_receipt["remote_submission_claim_path"]),
        )
        if legacy._validated_submission_claim(original_claim, initial_receipt) is None:
            raise LegacyRecoveryError("the original immutable submission claim is missing")
        _validate_remote_incoming_archive(volume, initial_receipt)
        if preflight_only:
            print(
                legacy._json_bytes(
                    {
                        "receipt_path": str(resolved_receipt_path),
                        "failed_app_id": failed_app_id,
                        "failed_function_call_id": failed_call_id,
                        "runtime_contract_sha256": observed_runtime,
                        "bundle_sha256": observed_sha256,
                        "bundle_size": observed_size,
                        "remote_journal_absent": True,
                        "remote_incoming_archive_valid": True,
                        "original_submission_claim_valid": True,
                        "archive_reuploaded": False,
                        "replacement_submitted": False,
                    }
                ).decode(),
                end="",
            )
            return resolved_receipt_path

        if intent_path.exists():
            intent = _read_json(intent_path, "deserialization recovery intent")
            recovery_claim_id = intent.get("recovery_claim_id")
            if not isinstance(recovery_claim_id, str):
                raise LegacyRecoveryError("deserialization recovery intent has no claim identity")
        else:
            recovery_claim_id = legacy._new_submission_claim_id()
            intent = {
                "schema": DESERIALIZATION_RECOVERY_SCHEMA,
                "recorded_at_utc": _utc_now(),
                "upload_id": initial_receipt["upload_id"],
                "bundle_sha256": initial_receipt["bundle_sha256"],
                "runtime_contract_sha256": initial_receipt["runtime_contract_sha256"],
                "failed_app_id": failed_app_id,
                "failed_function_call_id": failed_call_id,
                "failure_module": EXPECTED_DESERIALIZATION_MODULE,
                "failure_log_sha256": hashlib.sha256(logs.encode("utf-8")).hexdigest(),
                "recovery_claim_id": recovery_claim_id,
                "remote_recovery_claim_path": remote_claim_path,
                "recovery_receipt_path": str(recovery_receipt_path),
                "state": "intent",
                "replacement_function_call_id": None,
                "archive_reuploaded": False,
            }
            legacy._write_json_atomic(intent_path, intent)

        immutable_intent = {
            "schema": DESERIALIZATION_RECOVERY_SCHEMA,
            "upload_id": initial_receipt["upload_id"],
            "bundle_sha256": initial_receipt["bundle_sha256"],
            "runtime_contract_sha256": initial_receipt["runtime_contract_sha256"],
            "failed_app_id": failed_app_id,
            "failed_function_call_id": failed_call_id,
            "failure_module": EXPECTED_DESERIALIZATION_MODULE,
            "recovery_claim_id": recovery_claim_id,
            "remote_recovery_claim_path": remote_claim_path,
            "recovery_receipt_path": str(recovery_receipt_path),
            "archive_reuploaded": False,
        }
        if any(intent.get(key) != value for key, value in immutable_intent.items()):
            raise LegacyRecoveryError("deserialization recovery intent identity changed")
        if intent.get("state") in {"submitted", "submission-unknown"}:
            raise LegacyRecoveryError(
                "deserialization replacement was already submitted or became ambiguous"
            )

        claim_payload = _deserialization_claim_payload(
            initial_receipt,
            failed_app_id=failed_app_id,
            failed_call_id=failed_call_id,
            recovery_claim_id=recovery_claim_id,
        )
        remote_recovery_claim = legacy._read_volume_json(volume, remote_claim_path)
        if remote_recovery_claim is None:
            with volume.batch_upload(force=False) as batch:
                batch.put_file(io.BytesIO(legacy._json_bytes(claim_payload)), remote_claim_path)
            remote_recovery_claim = legacy._read_volume_json(volume, remote_claim_path)
        if remote_recovery_claim != claim_payload:
            raise LegacyRecoveryError(
                "remote deserialization recovery claim is missing or conflicts"
            )
        intent["state"] = "claim-created"
        legacy._write_json_atomic(intent_path, intent)

        if recovery_receipt_path.exists():
            recovery_receipt = legacy._read_receipt(recovery_receipt_path)
            if recovery_receipt["function_call_id"] is not None or recovery_receipt[
                "finalizer_state"
            ] in {"submitted", "submission-unknown", "submitting"}:
                raise LegacyRecoveryError(
                    "replacement receipt already has a submitted or ambiguous FunctionCall"
                )
        else:
            recovery_receipt = dict(initial_receipt)
            recovery_receipt["finalizer_state"] = "not-submitted"
            recovery_receipt["function_call_id"] = None
            recovery_receipt["finalizer_error"] = None
        legacy._refresh_receipt_budget(
            recovery_receipt,
            max_dollars=max_dollars,
            workspace_budget=workspace_budget,
            workspace_usage=workspace_usage,
        )
        legacy._write_json_atomic(recovery_receipt_path, recovery_receipt)
        _assert_terminal_failed_call(modal_module, failed_call_id)
        final_remote_operation = {
            name: legacy._read_volume_json(volume, f"{operation_path}/{name}.json")
            for name in ("status", "result", "failure")
        }
        if any(value is not None for value in final_remote_operation.values()):
            raise LegacyRecoveryError(
                "a finalizer journal appeared immediately before replacement submission"
            )
        _validate_remote_incoming_archive(volume, initial_receipt)
        try:
            legacy._submit_finalizer(
                modal_module,
                volume,
                recovery_receipt_path,
                recovery_receipt,
            )
        except BaseException:
            observed = legacy._read_receipt(recovery_receipt_path)
            intent["state"] = observed["finalizer_state"]
            intent["replacement_function_call_id"] = observed["function_call_id"]
            legacy._write_json_atomic(intent_path, intent)
            raise
        submitted = legacy._read_receipt(recovery_receipt_path)
        intent["state"] = "submitted"
        intent["replacement_function_call_id"] = submitted["function_call_id"]
        legacy._write_json_atomic(intent_path, intent)

    print(
        legacy._json_bytes(
            {
                "receipt_path": str(recovery_receipt_path),
                "original_failed_function_call_id": failed_call_id,
                "archive_reuploaded": False,
                **submitted,
            }
        ).decode(),
        end="",
    )
    return recovery_receipt_path


def _recover_mount_with_committed_runtime(
    runtime: Any,
    source_receipt_path: Path,
    *,
    failed_app_id: str,
    replacement_commit: str,
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
    preflight_only: bool = False,
) -> Path:
    """Replace one terminal mount-contract failure with the reviewed current worker."""

    resolved_source_path = source_receipt_path.resolve()
    source_receipt = runtime._read_receipt(resolved_source_path)
    failed_call_id = _validate_failed_deserialization_receipt(source_receipt)
    snapshot = _read_json(
        resolved_source_path.parent / "status-latest.json",
        "failed mount status snapshot",
    )
    _validate_failed_mount_status(snapshot, source_receipt, failed_call_id)
    logs = _fetch_mount_failure_logs(failed_app_id)
    _assert_app_stopped(failed_app_id)
    modal_module = runtime._require_modal()
    expected_terminal_message = (
        f"durable Modal bundle finalizer {source_receipt['upload_id']} failed; "
        "inspect its Volume journal"
    )
    _assert_terminal_failed_call(
        modal_module,
        failed_call_id,
        expected_error_type="RuntimeError",
        expected_message=expected_terminal_message,
    )
    runtime.validate_finalizer_cost_guard(max_dollars)
    runtime._validate_workspace_budget_guard(max_dollars, workspace_budget, workspace_usage)

    replacement_runtime_hash = runtime.finalizer_runtime_contract_sha256(runtime.REPOSITORY_ROOT)
    bundle_path = Path(cast(str, source_receipt["local_bundle_path"]))
    observed_size, observed_sha256 = runtime._hash_regular_file_stable(
        bundle_path,
        "prepared GPU bundle for mount recovery",
    )
    if (observed_size, observed_sha256) != (
        source_receipt["bundle_size"],
        source_receipt["bundle_sha256"],
    ):
        raise LegacyRecoveryError("local archive no longer matches the failed receipt")

    source_root = runtime._existing_receipt_root(resolved_source_path, source_receipt)
    recovery_root = resolved_source_path.parent / "mount-recovery"
    recovery_receipt_path = runtime._receipt_path(
        recovery_root,
        cast(str, source_receipt["upload_id"]),
    )
    intent_path = resolved_source_path.parent / "mount-recovery-intent.json"
    remote_claim_path = _mount_claim_path(
        cast(str, source_receipt["upload_id"]),
        failed_call_id,
    )
    builder_hash = _runtime_builder_sha256()
    source_identity = _receipt_identity_projection(source_receipt)
    source_identity_hash = _json_sha256(source_identity)
    replacement_receipt_identity_hash = _json_sha256(
        {
            "identity": source_identity,
            "runtime_contract_sha256": replacement_runtime_hash,
        }
    )

    with runtime._exclusive_submission(source_root):
        if runtime._read_receipt(resolved_source_path) != source_receipt:
            raise LegacyRecoveryError("failed mount receipt changed during validation")
        volume = modal_module.Volume.from_name(
            source_receipt["volume_name"],
            create_if_missing=False,
            version=runtime.VOLUME_VERSION,
        )
        operation_path = cast(str, source_receipt["remote_operation_path"])
        _assert_remote_path_absent(
            volume,
            operation_path,
            "failed mount operation directory",
        )
        original_claim = runtime._read_volume_json(
            volume,
            cast(str, source_receipt["remote_submission_claim_path"]),
        )
        if runtime._validated_submission_claim(original_claim, source_receipt) is None:
            raise LegacyRecoveryError("the original immutable submission claim is missing")
        original_claim_hash = _json_sha256(original_claim)
        _validate_remote_incoming_archive(volume, source_receipt)
        if preflight_only:
            print(
                runtime._json_bytes(
                    {
                        "source_receipt_path": str(resolved_source_path),
                        "failed_app_id": failed_app_id,
                        "failed_function_call_id": failed_call_id,
                        "replacement_commit": replacement_commit,
                        "replacement_runtime_contract_sha256": replacement_runtime_hash,
                        "recovery_builder_sha256": builder_hash,
                        "source_receipt_identity_sha256": source_identity_hash,
                        "replacement_receipt_identity_sha256": (replacement_receipt_identity_hash),
                        "original_submission_claim_sha256": original_claim_hash,
                        "bundle_sha256": observed_sha256,
                        "bundle_size": observed_size,
                        "remote_journal_absent": True,
                        "remote_incoming_archive_valid": True,
                        "original_submission_claim_valid": True,
                        "archive_reuploaded": False,
                        "replacement_submitted": False,
                    }
                ).decode(),
                end="",
            )
            return resolved_source_path

        if recovery_receipt_path.exists():
            recovery_receipt = runtime._read_receipt(recovery_receipt_path)
            if (
                _receipt_identity_projection(recovery_receipt) != source_identity
                or recovery_receipt["runtime_contract_sha256"] != replacement_runtime_hash
                or recovery_receipt["function_call_id"] is not None
                or recovery_receipt["finalizer_state"] != "not-submitted"
                or recovery_receipt["finalizer_error"] is not None
            ):
                raise LegacyRecoveryError(
                    "existing mount recovery receipt is not the exact authorized replacement"
                )
        else:
            recovery_receipt = dict(source_receipt)
            recovery_receipt["runtime_contract_sha256"] = replacement_runtime_hash
            recovery_receipt["finalizer_state"] = "not-submitted"
            recovery_receipt["function_call_id"] = None
            recovery_receipt["finalizer_error"] = None

        if intent_path.exists():
            intent = _read_json(intent_path, "mount recovery intent")
            recovery_claim_id = intent.get("recovery_claim_id")
            if not isinstance(recovery_claim_id, str):
                raise LegacyRecoveryError("mount recovery intent has no claim identity")
        else:
            recovery_claim_id = runtime._new_submission_claim_id()
            intent = {
                "schema": MOUNT_RECOVERY_SCHEMA,
                "recorded_at_utc": _utc_now(),
                "upload_id": source_receipt["upload_id"],
                "bundle_sha256": source_receipt["bundle_sha256"],
                "source_runtime_contract_sha256": source_receipt["runtime_contract_sha256"],
                "replacement_runtime_contract_sha256": replacement_runtime_hash,
                "replacement_commit": replacement_commit,
                "recovery_builder_sha256": builder_hash,
                "source_receipt_identity_sha256": source_identity_hash,
                "replacement_receipt_identity_sha256": replacement_receipt_identity_hash,
                "original_submission_claim_sha256": original_claim_hash,
                "failed_app_id": failed_app_id,
                "failed_function_call_id": failed_call_id,
                "failure_log_sha256": hashlib.sha256(logs.encode("utf-8")).hexdigest(),
                "recovery_claim_id": recovery_claim_id,
                "remote_recovery_claim_path": remote_claim_path,
                "recovery_receipt_path": str(recovery_receipt_path),
                "state": "intent",
                "replacement_function_call_id": None,
                "archive_reuploaded": False,
            }
            runtime._write_json_atomic(intent_path, intent)
        immutable = {
            "schema": MOUNT_RECOVERY_SCHEMA,
            "upload_id": source_receipt["upload_id"],
            "bundle_sha256": source_receipt["bundle_sha256"],
            "source_runtime_contract_sha256": source_receipt["runtime_contract_sha256"],
            "replacement_runtime_contract_sha256": replacement_runtime_hash,
            "replacement_commit": replacement_commit,
            "recovery_builder_sha256": builder_hash,
            "source_receipt_identity_sha256": source_identity_hash,
            "replacement_receipt_identity_sha256": replacement_receipt_identity_hash,
            "original_submission_claim_sha256": original_claim_hash,
            "failed_app_id": failed_app_id,
            "failed_function_call_id": failed_call_id,
            "recovery_claim_id": recovery_claim_id,
            "remote_recovery_claim_path": remote_claim_path,
            "recovery_receipt_path": str(recovery_receipt_path),
            "archive_reuploaded": False,
        }
        if any(intent.get(key) != value for key, value in immutable.items()):
            raise LegacyRecoveryError("mount recovery intent identity changed")
        if intent.get("state") in {"submitted", "submission-unknown"}:
            raise LegacyRecoveryError("mount replacement was already submitted or ambiguous")

        claim_payload = {
            **immutable,
            "schema": MOUNT_RECOVERY_CLAIM_SCHEMA,
            "original_submission_claim_id": source_receipt["submission_claim_id"],
            "source_receipt_identity": source_identity,
        }
        remote_claim = runtime._read_volume_json(volume, remote_claim_path)
        if remote_claim is None:
            with volume.batch_upload(force=False) as batch:
                batch.put_file(io.BytesIO(runtime._json_bytes(claim_payload)), remote_claim_path)
            remote_claim = runtime._read_volume_json(volume, remote_claim_path)
        if remote_claim != claim_payload:
            raise LegacyRecoveryError("remote mount recovery claim is missing or conflicts")
        intent["state"] = "claim-created"
        runtime._write_json_atomic(intent_path, intent)
        runtime._refresh_receipt_budget(
            recovery_receipt,
            max_dollars=max_dollars,
            workspace_budget=workspace_budget,
            workspace_usage=workspace_usage,
        )
        runtime._write_json_atomic(recovery_receipt_path, recovery_receipt)

        _assert_app_stopped(failed_app_id)
        _assert_terminal_failed_call(
            modal_module,
            failed_call_id,
            expected_error_type="RuntimeError",
            expected_message=expected_terminal_message,
        )
        _assert_remote_path_absent(
            volume,
            operation_path,
            "mount operation directory immediately before recovery",
        )
        _validate_remote_incoming_archive(volume, source_receipt)
        try:
            runtime._submit_finalizer(
                modal_module,
                volume,
                recovery_receipt_path,
                recovery_receipt,
            )
        except BaseException:
            observed = runtime._read_receipt(recovery_receipt_path)
            intent["state"] = observed["finalizer_state"]
            intent["replacement_function_call_id"] = observed["function_call_id"]
            runtime._write_json_atomic(intent_path, intent)
            raise
        submitted = runtime._read_receipt(recovery_receipt_path)
        intent["state"] = "submitted"
        intent["replacement_function_call_id"] = submitted["function_call_id"]
        runtime._write_json_atomic(intent_path, intent)

    print(
        runtime._json_bytes(
            {
                "receipt_path": str(recovery_receipt_path),
                "source_failed_function_call_id": failed_call_id,
                "replacement_commit": replacement_commit,
                "recovery_builder_sha256": builder_hash,
                "archive_reuploaded": False,
                **submitted,
            }
        ).decode(),
        end="",
    )
    return recovery_receipt_path


def recover(
    receipt_path: Path,
    *,
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
) -> Path:
    initial = _read_json(receipt_path.resolve(), "legacy Modal receipt")
    verification = initial.get("verification")
    runtime_commit = verification.get("git_commit") if isinstance(verification, dict) else None
    if not isinstance(runtime_commit, str):
        raise LegacyRecoveryError("legacy Modal receipt has no runtime commit")
    with tempfile.TemporaryDirectory(prefix="sion-modal-legacy-runtime-") as temporary:
        runtime_root = Path(temporary) / "runtime"
        stage_path = _materialize_legacy_runtime(
            REPOSITORY_ROOT,
            runtime_commit,
            runtime_root,
        )
        legacy = _load_legacy_module(stage_path)
        _install_importable_runtime_builder(legacy)
        return _recover_with_legacy_module(
            legacy,
            receipt_path,
            max_dollars=max_dollars,
            workspace_budget=workspace_budget,
            workspace_usage=workspace_usage,
        )


def recover_deserialization_failure(
    receipt_path: Path,
    *,
    failed_app_id: str,
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
    preflight_only: bool = False,
) -> Path:
    initial = _read_json(receipt_path.resolve(), "failed Modal receipt")
    verification = initial.get("verification")
    runtime_commit = verification.get("git_commit") if isinstance(verification, dict) else None
    if not isinstance(runtime_commit, str):
        raise LegacyRecoveryError("failed Modal receipt has no runtime commit")
    with tempfile.TemporaryDirectory(prefix="sion-modal-legacy-runtime-") as temporary:
        runtime_root = Path(temporary) / "runtime"
        stage_path = _materialize_legacy_runtime(
            REPOSITORY_ROOT,
            runtime_commit,
            runtime_root,
        )
        legacy = _load_legacy_module(stage_path)
        _install_importable_runtime_builder(legacy)
        return _recover_deserialization_with_legacy_module(
            legacy,
            receipt_path,
            failed_app_id=failed_app_id,
            max_dollars=max_dollars,
            workspace_budget=workspace_budget,
            workspace_usage=workspace_usage,
            preflight_only=preflight_only,
        )


def recover_mount_failure(
    receipt_path: Path,
    *,
    failed_app_id: str,
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
    preflight_only: bool = False,
) -> Path:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    replacement_commit = completed.stdout.strip()
    if completed.returncode != 0 or COMMIT_PATTERN.fullmatch(replacement_commit) is None:
        raise LegacyRecoveryError("cannot resolve the reviewed replacement commit")
    with tempfile.TemporaryDirectory(prefix="sion-modal-current-runtime-") as temporary:
        runtime_root = Path(temporary) / "runtime"
        stage_path = _materialize_committed_runtime(
            REPOSITORY_ROOT,
            replacement_commit,
            runtime_root,
        )
        runtime = _load_legacy_module(stage_path)
        _install_importable_runtime_builder(runtime)
        return _recover_mount_with_committed_runtime(
            runtime,
            receipt_path,
            failed_app_id=failed_app_id,
            replacement_commit=replacement_commit,
            max_dollars=max_dollars,
            workspace_budget=workspace_budget,
            workspace_usage=workspace_usage,
            preflight_only=preflight_only,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover one exact legacy Modal disk-rejection receipt without reuploading."
    )
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--max-dollars", required=True, type=float)
    parser.add_argument("--workspace-budget", required=True, type=float)
    parser.add_argument("--workspace-usage", required=True, type=float)
    parser.add_argument(
        "--failed-app-id",
        help="Recover a terminal legacy-module deserialization failure from this stopped App.",
    )
    parser.add_argument(
        "--mount-failure-app-id",
        help="Recover a terminal pre-journal Modal Volume mount failure from this stopped App.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate a deserialization recovery without writing a claim or submitting a call.",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    if parsed.failed_app_id is not None and parsed.mount_failure_app_id is not None:
        raise LegacyRecoveryError("choose exactly one supported recovery failure")
    if parsed.mount_failure_app_id is not None:
        recover_mount_failure(
            parsed.receipt,
            failed_app_id=parsed.mount_failure_app_id,
            max_dollars=parsed.max_dollars,
            workspace_budget=parsed.workspace_budget,
            workspace_usage=parsed.workspace_usage,
            preflight_only=parsed.preflight_only,
        )
        return 0
    if parsed.failed_app_id is None:
        if parsed.preflight_only:
            raise LegacyRecoveryError("--preflight-only requires --failed-app-id")
        recover(
            parsed.receipt,
            max_dollars=parsed.max_dollars,
            workspace_budget=parsed.workspace_budget,
            workspace_usage=parsed.workspace_usage,
        )
    else:
        recover_deserialization_failure(
            parsed.receipt,
            failed_app_id=parsed.failed_app_id,
            max_dollars=parsed.max_dollars,
            workspace_budget=parsed.workspace_budget,
            workspace_usage=parsed.workspace_usage,
            preflight_only=parsed.preflight_only,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
