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
from contextlib import contextmanager
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Generator, Mapping, Sequence, cast


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_STAGE_PATH = Path("scripts/modal_stage_gpu_bundle.py")
COPIED_RUNTIME_PATHS = (
    Path("scripts/package_gpu_bundle.py"),
    Path("requirements/modal-bundle-stage.txt"),
)
SOURCE_PACKAGE_PATH = Path("src/sion_translate")
RECOVERY_SCHEMA = "sion-modal-bundle-disk-recovery-v1"
EXPECTED_ERROR_TYPE = "InvalidError"
EXPECTED_ERROR_MESSAGE = (
    "Function disk request out of bounds: 2048 MiB. Must be between 524288 and 3145728 MiB."
)
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


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


def _load_legacy_module(stage_path: Path) -> ModuleType:
    name = f"sion_legacy_modal_stage_{stage_path.parent.parent.name}"
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


@contextmanager
def _pickle_legacy_runtime_by_value(legacy: ModuleType) -> Generator[None, None, None]:
    """Keep reconstructed modules import-independent during Modal serialization."""

    cloudpickle = importlib.import_module("modal._vendor.cloudpickle")
    raw_package = getattr(legacy, "PACKAGE", None)
    if not isinstance(raw_package, ModuleType):
        raise LegacyRecoveryError("reconstructed runtime verifier is not a module")
    modules = tuple(dict.fromkeys((legacy, raw_package)))
    registered_names = cloudpickle.list_registry_pickle_by_value()
    added: list[ModuleType] = []
    try:
        for module in modules:
            if module.__name__ not in registered_names:
                cloudpickle.register_pickle_by_value(module)
                added.append(module)
        yield
    finally:
        for module in reversed(added):
            cloudpickle.unregister_pickle_by_value(module)


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
        with _pickle_legacy_runtime_by_value(legacy):
            return _recover_with_legacy_module(
                legacy,
                receipt_path,
                max_dollars=max_dollars,
                workspace_budget=workspace_budget,
                workspace_usage=workspace_usage,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover one exact legacy Modal disk-rejection receipt without reuploading."
    )
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--max-dollars", required=True, type=float)
    parser.add_argument("--workspace-budget", required=True, type=float)
    parser.add_argument("--workspace-usage", required=True, type=float)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    recover(
        parsed.receipt,
        max_dollars=parsed.max_dollars,
        workspace_budget=parsed.workspace_budget,
        workspace_usage=parsed.workspace_usage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
