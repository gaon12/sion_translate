"""Stage one verified prepared-only training bundle on a Modal Volume.

The command deliberately separates the large, non-idempotent upload from the
detached CPU finalizer. A local receipt is written before the upload begins. If
the upload response is lost, the receipt records that ambiguity and this tool
does not retry automatically under a second path.

The finalizer never uses a GPU and is not deployed. It verifies the uploaded
archive again, extracts it without ``ZipFile.extractall``, verifies the tree,
writes ``READY`` last, and publishes the complete incoming directory through a
same-volume no-replace rename. Durable operation files and the FunctionCall ID
allow status recovery after the submitting terminal or Codex task disappears.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import UTC, datetime
import errno
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import socket
import stat
import sys
import tempfile
import traceback
from typing import Any, Generator, IO, Mapping, Sequence, cast
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE_SCRIPT_RELATIVE_PATH = Path("scripts/modal_stage_gpu_bundle.py")
PACKAGE_SCRIPT_RELATIVE_PATH = Path("scripts/package_gpu_bundle.py")
FINALIZER_REQUIREMENTS_RELATIVE_PATH = Path("requirements/modal-bundle-stage.txt")
STAGE_SCRIPT = REPOSITORY_ROOT / STAGE_SCRIPT_RELATIVE_PATH
PACKAGE_SCRIPT = REPOSITORY_ROOT / PACKAGE_SCRIPT_RELATIVE_PATH
FINALIZER_REQUIREMENTS = REPOSITORY_ROOT / FINALIZER_REQUIREMENTS_RELATIVE_PATH
REMOTE_ROOT = PurePosixPath("/opt/sion-bundle-stage")
REMOTE_STAGE_SCRIPT = REMOTE_ROOT / STAGE_SCRIPT_RELATIVE_PATH.as_posix()
REMOTE_PACKAGE_SCRIPT = REMOTE_ROOT / PACKAGE_SCRIPT_RELATIVE_PATH.as_posix()
REMOTE_FINALIZER_REQUIREMENTS = REMOTE_ROOT / FINALIZER_REQUIREMENTS_RELATIVE_PATH.as_posix()
SOURCE_PACKAGE_RELATIVE_PATH = Path("src/sion_translate")
REMOTE_SOURCE_PACKAGE = REMOTE_ROOT / SOURCE_PACKAGE_RELATIVE_PATH.as_posix()
VOLUME_MOUNT = PurePosixPath("/mnt/sion-bundles")


def _load_package_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("sion_package_gpu_bundle_for_stage", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"cannot load GPU bundle verifier: {path}")
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


PACKAGE: Any = _load_package_module(PACKAGE_SCRIPT)

try:
    import modal
except ModuleNotFoundError:  # Unit tests use explicit fake Modal clients.
    modal = None


EXPECTED_MODAL_CLIENT_VERSION = "1.5.3"
VOLUME_VERSION = 1
APP_NAME = "sion-prepared-bundle-finalizer"
FINALIZER_FUNCTION_NAME = "finalize_prepared_bundle"
FINALIZER_TIMEOUT_SECONDS = 4 * 60 * 60
FINALIZER_CPU_CORES = 2.0
FINALIZER_MEMORY_MIB = 8 * 1024
FINALIZER_SCALEDOWN_WINDOW_SECONDS = 2
FINALIZER_ATTEMPT_CONTINGENCY = 2
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222
MAX_WORKSPACE_BUDGET_HEADROOM_USD = 5.0
RECEIPT_VERSION = 1
OPERATION_SCHEMA = "sion-modal-bundle-operation-v1"
RESULT_SCHEMA = "sion-modal-bundle-result-v1"
READY_SCHEMA = "sion-modal-prepared-bundle-ready-v1"
SUBMISSION_CLAIM_SCHEMA = "sion-modal-bundle-submission-claim-v1"
DEFAULT_RECEIPT_ROOT = REPOSITORY_ROOT / "artifacts" / "modal-bundle-uploads"
SUBMISSION_LOCK_NAME = ".submission-lock"
COPY_BUFFER_SIZE = 8 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024 * 1024
CONTENT_PREFIX_LENGTH = 2
UPLOAD_ID_PATTERN = re.compile(r"^bundle-[0-9]{8}t[0-9]{6}z-[0-9a-f]{16}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FUNCTION_CALL_ID_PATTERN = re.compile(r"^fc-[A-Za-z0-9_-]{8,128}$")
SUBMISSION_CLAIM_ID_PATTERN = re.compile(r"^claim-[0-9a-f]{32}$")
VOLUME_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
RECEIPT_FIELDS = {
    "receipt_version",
    "upload_id",
    "volume_name",
    "volume_version",
    "app_name",
    "function_name",
    "runtime_contract_sha256",
    "local_bundle_path",
    "bundle_sha256",
    "bundle_size",
    "verification",
    "remote_incoming_path",
    "remote_final_path",
    "remote_operation_path",
    "created_at_utc",
    "authorization_compute_charge_usd",
    "max_dollars",
    "workspace_budget_usd",
    "workspace_usage_before_submit_usd",
    "workspace_budget_headroom_usd",
    "budget_observations",
    "upload_state",
    "upload_error",
    "finalizer_state",
    "function_call_id",
    "finalizer_error",
    "submission_claim_id",
    "remote_submission_claim_path",
    "submission_claim_state",
    "submission_claim_error",
}


class BundleStageError(RuntimeError):
    """Raised when a bundle cannot be staged without weakening its contract."""


class BundleConflictError(BundleStageError):
    """Raised when content-addressed storage already contains conflicting data."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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


def _error_record(error: BaseException) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "message": str(error)[:8_000],
        "recorded_at_utc": _utc_now(),
    }


def _is_link_like(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(file_attributes & reparse_flag)


def _regular_file_metadata(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise BundleStageError(f"{label} does not exist: {path}") from error
    if _is_link_like(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise BundleStageError(f"{label} must be a regular non-symlink file: {path}")
    return metadata


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _hash_regular_file_stable(path: Path, label: str) -> tuple[int, str]:
    before = _regular_file_metadata(path, label)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(COPY_BUFFER_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    after = _regular_file_metadata(path, label)
    if _file_identity(before) != _file_identity(after) or size != after.st_size:
        raise BundleStageError(f"{label} changed while it was being hashed: {path}")
    return size, digest.hexdigest()


def finalizer_runtime_contract_sha256(root: Path) -> str:
    """Hash every reviewed runtime byte copied into the CPU finalizer image."""

    resolved_root = root.resolve()
    relative_paths = [
        STAGE_SCRIPT_RELATIVE_PATH,
        PACKAGE_SCRIPT_RELATIVE_PATH,
        FINALIZER_REQUIREMENTS_RELATIVE_PATH,
    ]
    source_root = resolved_root / SOURCE_PACKAGE_RELATIVE_PATH
    if source_root.is_symlink() or not source_root.is_dir():
        raise BundleStageError(
            f"Modal bundle finalizer source root is not a regular directory: {source_root}"
        )
    copied_source_paths: list[Path] = []
    for path in sorted(source_root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        relative_path = path.relative_to(resolved_root)
        if "__pycache__" in relative_path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        metadata = path.lstat()
        if _is_link_like(metadata):
            raise BundleStageError(f"Modal bundle finalizer source contains a link: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise BundleStageError(f"Modal bundle finalizer source contains a special file: {path}")
        copied_source_paths.append(relative_path)
    if not copied_source_paths:
        raise BundleStageError("Modal bundle finalizer source package is empty")
    relative_paths.extend(copied_source_paths)
    if len(relative_paths) != len(set(relative_paths)):
        raise BundleStageError("Modal bundle finalizer contract contains duplicate paths")
    contract = hashlib.sha256()
    for relative_path in relative_paths:
        path = resolved_root / relative_path
        size, digest = _hash_regular_file_stable(path, "Modal bundle finalizer contract file")
        contract.update(f"{relative_path.as_posix()}\0{size}\0{digest}\n".encode("utf-8"))
    return contract.hexdigest()


def _verify_executed_entrypoint(executed_path: Path, reviewed_path: Path) -> None:
    """Bind Modal's mounted finalizer source to its reviewed image copy."""

    executed = _hash_regular_file_stable(executed_path, "executed Modal finalizer")
    reviewed = _hash_regular_file_stable(reviewed_path, "reviewed Modal finalizer")
    if executed != reviewed:
        raise BundleStageError(
            "Modal's executed bundle finalizer differs from its reviewed image copy"
        )


def _verify_remote_runtime_contract(expected_sha256: str) -> str:
    expected = _validate_sha256(expected_sha256)
    _validate_modal_client_version()
    observed = finalizer_runtime_contract_sha256(Path(str(REMOTE_ROOT)))
    if observed != expected:
        raise BundleStageError(
            "deployed Modal bundle finalizer bytes differ from the submitted contract"
        )
    _verify_executed_entrypoint(Path(__file__), Path(str(REMOTE_STAGE_SCRIPT)))
    return observed


def _verification_dict(result: object) -> dict[str, object]:
    file_count = getattr(result, "file_count", None)
    total_bytes = getattr(result, "total_bytes", None)
    git_commit = getattr(result, "git_commit", None)
    git_tree = getattr(result, "git_tree", None)
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count < 1
        or isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or total_bytes < 1
        or not isinstance(git_commit, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", git_commit) is None
        or not isinstance(git_tree, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", git_tree) is None
    ):
        raise BundleStageError("GPU bundle verification returned an invalid summary")
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "git_commit": git_commit,
        "git_tree": git_tree,
    }


def _assert_prepared_only_archive(path: Path, package_module: Any = PACKAGE) -> None:
    """Require the exact artifact selection produced by ``--prepared-only``."""

    manifest_name = f"{package_module.ARCHIVE_ROOT}/{package_module.MANIFEST_NAME}"
    try:
        with zipfile.ZipFile(path, mode="r", allowZip64=True) as archive:
            info = archive.getinfo(manifest_name)
            if info.file_size > MAX_MANIFEST_BYTES:
                raise BundleStageError("GPU bundle manifest is unreasonably large")
            with archive.open(info, mode="r") as source:
                payload = source.read(MAX_MANIFEST_BYTES + 1)
    except (KeyError, zipfile.BadZipFile) as error:
        raise BundleStageError("GPU bundle has no readable package manifest") from error
    if len(payload) > MAX_MANIFEST_BYTES:
        raise BundleStageError("GPU bundle manifest exceeds its read limit")
    try:
        raw: object = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BundleStageError("GPU bundle manifest is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise BundleStageError("GPU bundle manifest is not a JSON object")
    manifest = cast(dict[str, object], raw)
    raw_files = manifest.get("files")
    contract = manifest.get("training_contract")
    if not isinstance(raw_files, list) or not isinstance(contract, dict):
        raise BundleStageError("GPU bundle manifest omits its prepared-artifact contract")
    records = cast(list[object], raw_files)
    paths: set[str] = set()
    origins: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise BundleStageError("GPU bundle manifest contains an invalid file record")
        record_values = cast(dict[str, object], record)
        path_value = record_values.get("path")
        origin = record_values.get("origin")
        if not isinstance(path_value, str) or not isinstance(origin, str):
            raise BundleStageError("GPU bundle manifest file identity is invalid")
        paths.add(path_value)
        origins.add(origin)
    contract_values = cast(dict[str, object], contract)
    if contract_values.get("raw_parallel_data_included") is not False:
        raise BundleStageError("bundle is not prepared-only: raw parallel data is included")
    if origins.intersection({"data-jsonl", "monolingual-corpus"}):
        raise BundleStageError("bundle is not prepared-only: raw preparation inputs are included")

    def contains_tree(root: str) -> bool:
        return any(path.startswith(f"{root}/") for path in paths)

    if not contains_tree(package_module.TOKENIZER_ROOT_PATH):
        raise BundleStageError("prepared-only bundle omits the tokenizer")
    if not contains_tree(package_module.TRANSLATION_DATASET_ROOT_PATH):
        raise BundleStageError("prepared-only bundle omits the translation dataset")
    foundation_enabled = contract_values.get("foundation_enabled")
    if not isinstance(foundation_enabled, bool):
        raise BundleStageError("prepared-only bundle has an invalid foundation contract")
    foundation_present = contains_tree(package_module.FOUNDATION_DATASET_ROOT_PATH)
    if foundation_present != foundation_enabled:
        raise BundleStageError(
            "prepared-only bundle foundation dataset does not match the selected config"
        )


def _validate_local_bundle(path: Path) -> tuple[Path, int, str, dict[str, object]]:
    absolute = Path(os.path.abspath(path))
    if absolute.suffix.lower() != ".zip":
        raise BundleStageError("prepared GPU bundle must use the .zip extension")
    first_size, first_sha256 = _hash_regular_file_stable(absolute, "prepared GPU bundle")
    verification = _verification_dict(PACKAGE.verify_archive(absolute))
    _assert_prepared_only_archive(absolute)
    second_size, second_sha256 = _hash_regular_file_stable(absolute, "prepared GPU bundle")
    if (first_size, first_sha256) != (second_size, second_sha256):
        raise BundleStageError("prepared GPU bundle changed during local verification")
    return absolute, first_size, first_sha256, verification


def _validate_upload_id(value: object) -> str:
    if not isinstance(value, str) or UPLOAD_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("Modal bundle upload ID is invalid")
    return value


def _validate_sha256(value: object) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("Modal bundle SHA-256 is invalid")
    return value


def _validate_function_call_id(value: object) -> str:
    if not isinstance(value, str) or FUNCTION_CALL_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("Modal bundle FunctionCall ID is invalid")
    return value


def _validate_submission_claim_id(value: object) -> str:
    if not isinstance(value, str) or SUBMISSION_CLAIM_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("Modal bundle submission claim ID is invalid")
    return value


def _validate_positive_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Modal bundle size is invalid")
    return value


def finalizer_authorization_compute_charge() -> float:
    """Return a two-attempt CPU and memory contingency for the finalizer."""

    per_second = (
        FINALIZER_CPU_CORES * CPU_USD_PER_CORE_SECOND
        + (FINALIZER_MEMORY_MIB / 1024) * MEMORY_USD_PER_GIB_SECOND
    )
    billed_seconds = FINALIZER_TIMEOUT_SECONDS + FINALIZER_SCALEDOWN_WINDOW_SECONDS
    return per_second * billed_seconds * FINALIZER_ATTEMPT_CONTINGENCY


def validate_finalizer_cost_guard(max_dollars: object) -> float:
    authorized = finalizer_authorization_compute_charge()
    if (
        isinstance(max_dollars, bool)
        or not isinstance(max_dollars, (int, float))
        or not math.isfinite(float(max_dollars))
        or float(max_dollars) < authorized
    ):
        raise ValueError(
            "--max-dollars must be finite and cover the two-attempt CPU finalizer "
            f"contingency of ${authorized:.4f}"
        )
    return authorized


def _validate_workspace_budget_guard(
    max_dollars: object,
    workspace_budget: object,
    workspace_usage: object,
) -> float:
    authorized = validate_finalizer_cost_guard(max_dollars)
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
    headroom = float(workspace_budget) - float(workspace_usage)
    if headroom + 1e-9 < authorized:
        raise ValueError("Workspace budget headroom does not cover the CPU finalizer contingency")
    if headroom > MAX_WORKSPACE_BUDGET_HEADROOM_USD + 1e-9:
        raise ValueError(
            "Workspace budget headroom exceeds the $5 safety ceiling; lower the hard "
            "Workspace budget before staging"
        )
    return headroom


def _budget_observation(
    max_dollars: object,
    workspace_budget: object,
    workspace_usage: object,
) -> dict[str, object]:
    authorized = validate_finalizer_cost_guard(max_dollars)
    headroom = _validate_workspace_budget_guard(
        max_dollars,
        workspace_budget,
        workspace_usage,
    )
    assert isinstance(max_dollars, (int, float)) and not isinstance(max_dollars, bool)
    assert isinstance(workspace_budget, (int, float)) and not isinstance(workspace_budget, bool)
    assert isinstance(workspace_usage, (int, float)) and not isinstance(workspace_usage, bool)
    return {
        "observed_at_utc": _utc_now(),
        "authorization_compute_charge_usd": authorized,
        "max_dollars": float(max_dollars),
        "workspace_budget_usd": float(workspace_budget),
        "workspace_usage_usd": float(workspace_usage),
        "workspace_budget_headroom_usd": headroom,
    }


def _validated_budget_observation(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Modal bundle budget observation is not a JSON object")
    observation = cast(dict[str, object], value.copy())
    if set(observation) != {
        "observed_at_utc",
        "authorization_compute_charge_usd",
        "max_dollars",
        "workspace_budget_usd",
        "workspace_usage_usd",
        "workspace_budget_headroom_usd",
    } or not isinstance(observation.get("observed_at_utc"), str):
        raise ValueError("Modal bundle budget observation fields are invalid")
    authorized = validate_finalizer_cost_guard(observation.get("max_dollars"))
    headroom = _validate_workspace_budget_guard(
        observation.get("max_dollars"),
        observation.get("workspace_budget_usd"),
        observation.get("workspace_usage_usd"),
    )
    recorded_authorization = observation.get("authorization_compute_charge_usd")
    recorded_headroom = observation.get("workspace_budget_headroom_usd")
    if (
        isinstance(recorded_authorization, bool)
        or not isinstance(recorded_authorization, (int, float))
        or not math.isclose(float(recorded_authorization), authorized, rel_tol=0.0, abs_tol=1e-12)
        or isinstance(recorded_headroom, bool)
        or not isinstance(recorded_headroom, (int, float))
        or not math.isclose(float(recorded_headroom), headroom, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("Modal bundle budget observation values are invalid")
    return observation


def _validate_volume_name(value: object) -> str:
    if not isinstance(value, str) or VOLUME_NAME_PATTERN.fullmatch(value) is None:
        raise ValueError("Modal Volume name is invalid")
    return value


def _new_upload_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz").lower()
    return _validate_upload_id(f"bundle-{stamp}-{secrets.token_hex(8)}")


def _new_submission_claim_id() -> str:
    return _validate_submission_claim_id(f"claim-{secrets.token_hex(16)}")


def _remote_submission_claim_path(upload_id: str) -> str:
    return f"/submission-claims/{_validate_upload_id(upload_id)}.json"


def _remote_paths(upload_id: str, sha256: str) -> tuple[str, str, str]:
    validated_upload_id = _validate_upload_id(upload_id)
    validated_sha256 = _validate_sha256(sha256)
    incoming = f"/incoming/{validated_upload_id}"
    final = f"/bundles/sha256/{validated_sha256[:CONTENT_PREFIX_LENGTH]}/{validated_sha256}"
    operation = f"/operations/{validated_upload_id}"
    return incoming, final, operation


def _receipt_path(receipt_root: Path, upload_id: str) -> Path:
    root = receipt_root.resolve()
    path = root / _validate_upload_id(upload_id) / "receipt.json"
    if path.parent.parent != root:
        raise BundleStageError("Modal bundle receipt escaped its configured root")
    return path


def _validated_verification(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Modal bundle verification summary is invalid")
    fields = cast(dict[str, object], value.copy())
    if set(fields) != {"file_count", "total_bytes", "git_commit", "git_tree"}:
        raise ValueError("Modal bundle verification summary fields are invalid")
    return _verification_dict(type("Verification", (), fields)())


def _validated_receipt(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Modal bundle receipt is not a JSON object")
    receipt = cast(dict[str, Any], value.copy())
    if set(receipt) != RECEIPT_FIELDS or receipt.get("receipt_version") != RECEIPT_VERSION:
        raise ValueError("Modal bundle receipt fields or version are invalid")
    upload_id = _validate_upload_id(receipt.get("upload_id"))
    volume_name = _validate_volume_name(receipt.get("volume_name"))
    del volume_name
    if receipt.get("volume_version") != VOLUME_VERSION:
        raise ValueError("Modal bundle receipt Volume version is invalid")
    if (
        receipt.get("app_name") != APP_NAME
        or receipt.get("function_name") != FINALIZER_FUNCTION_NAME
    ):
        raise ValueError("Modal bundle receipt finalizer identity is invalid")
    _validate_sha256(receipt.get("runtime_contract_sha256"))
    local_path = receipt.get("local_bundle_path")
    if not isinstance(local_path, str) or not Path(local_path).is_absolute():
        raise ValueError("Modal bundle receipt local path is invalid")
    sha256 = _validate_sha256(receipt.get("bundle_sha256"))
    _validate_positive_size(receipt.get("bundle_size"))
    _validated_verification(receipt.get("verification"))
    incoming, final, operation = _remote_paths(upload_id, sha256)
    claim_path = _remote_submission_claim_path(upload_id)
    if (
        receipt.get("remote_incoming_path") != incoming
        or receipt.get("remote_final_path") != final
        or receipt.get("remote_operation_path") != operation
        or receipt.get("remote_submission_claim_path") != claim_path
    ):
        raise ValueError("Modal bundle receipt remote paths are invalid")
    if not isinstance(receipt.get("created_at_utc"), str):
        raise ValueError("Modal bundle receipt creation time is invalid")
    max_dollars = receipt.get("max_dollars")
    authorized = validate_finalizer_cost_guard(max_dollars)
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
        raise ValueError("Modal bundle receipt finalizer authorization is invalid")
    expected_headroom = _validate_workspace_budget_guard(
        max_dollars,
        receipt.get("workspace_budget_usd"),
        receipt.get("workspace_usage_before_submit_usd"),
    )
    recorded_headroom = receipt.get("workspace_budget_headroom_usd")
    if (
        isinstance(recorded_headroom, bool)
        or not isinstance(recorded_headroom, (int, float))
        or not math.isclose(float(recorded_headroom), expected_headroom, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("Modal bundle receipt Workspace budget headroom is invalid")
    raw_observations = receipt.get("budget_observations")
    if not isinstance(raw_observations, list):
        raise ValueError("Modal bundle receipt budget history is invalid")
    observation_values = cast(list[object], raw_observations)
    if not observation_values or len(observation_values) > 32:
        raise ValueError("Modal bundle receipt budget history is invalid")
    observations = [_validated_budget_observation(value) for value in observation_values]
    latest_observation = observations[-1]
    if (
        latest_observation["authorization_compute_charge_usd"]
        != receipt["authorization_compute_charge_usd"]
        or latest_observation["max_dollars"] != receipt["max_dollars"]
        or latest_observation["workspace_budget_usd"] != receipt["workspace_budget_usd"]
        or latest_observation["workspace_usage_usd"] != receipt["workspace_usage_before_submit_usd"]
        or latest_observation["workspace_budget_headroom_usd"]
        != receipt["workspace_budget_headroom_usd"]
    ):
        raise ValueError("Modal bundle receipt latest budget values disagree with history")
    if receipt.get("upload_state") not in {"intent", "uploaded", "upload-unknown"}:
        raise ValueError("Modal bundle receipt upload state is invalid")
    if receipt.get("finalizer_state") not in {
        "not-submitted",
        "submitting",
        "submitted",
        "submission-unknown",
    }:
        raise ValueError("Modal bundle receipt finalizer state is invalid")
    call_id = receipt.get("function_call_id")
    if call_id is not None and (
        not isinstance(call_id, str) or FUNCTION_CALL_ID_PATTERN.fullmatch(call_id) is None
    ):
        raise ValueError("Modal bundle receipt FunctionCall ID is invalid")
    if receipt["finalizer_state"] == "submitted" and call_id is None:
        raise ValueError("submitted Modal bundle receipt has no FunctionCall ID")
    if receipt["finalizer_state"] != "submitted" and call_id is not None:
        raise ValueError("unsubmitted Modal bundle receipt unexpectedly has a FunctionCall ID")
    claim_state = receipt.get("submission_claim_state")
    if claim_state not in {
        "not-created",
        "creating",
        "created",
        "creation-unknown",
    }:
        raise ValueError("Modal bundle receipt submission claim state is invalid")
    for field in ("upload_error", "finalizer_error", "submission_claim_error"):
        if receipt.get(field) is not None and not isinstance(receipt[field], dict):
            raise ValueError(f"Modal bundle receipt {field} is invalid")
    if receipt["upload_state"] == "upload-unknown" and receipt.get("upload_error") is None:
        raise ValueError("ambiguous Modal bundle upload has no diagnostic")
    if (
        receipt["finalizer_state"] == "submission-unknown"
        and receipt.get("finalizer_error") is None
    ):
        raise ValueError("ambiguous Modal finalizer submission has no diagnostic")
    if claim_state == "creation-unknown" and receipt.get("submission_claim_error") is None:
        raise ValueError("ambiguous Modal submission claim has no diagnostic")
    if claim_state == "not-created" and receipt.get("submission_claim_error") is not None:
        raise ValueError("unattempted Modal submission claim has an unexpected diagnostic")
    claim_id = receipt.get("submission_claim_id")
    if claim_state == "not-created":
        if claim_id is not None:
            raise ValueError("unattempted Modal submission claim already has an ID")
    else:
        _validate_submission_claim_id(claim_id)
    if receipt["finalizer_state"] != "not-submitted" and claim_state != "created":
        raise ValueError("Modal finalizer submission has no durable remote claim")
    return receipt


def _read_local_json(path: Path, label: str) -> object:
    metadata = _regular_file_metadata(path, label)
    if metadata.st_size > MAX_JSON_BYTES:
        raise ValueError(f"{label} is unreasonably large: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error


def _read_receipt(path: Path) -> dict[str, Any]:
    return _validated_receipt(_read_local_json(path, "Modal bundle receipt"))


def _read_local_recovered_state(
    run_directory: Path,
    expected_receipt: Mapping[str, object],
) -> str | None:
    snapshot_path = run_directory / "status-latest.json"
    if not snapshot_path.exists():
        return None
    value = _read_local_json(snapshot_path, "Modal bundle status snapshot")
    if not isinstance(value, dict):
        raise BundleStageError("Modal bundle status snapshot is not a JSON object")
    snapshot = cast(dict[str, object], value)
    recovered_state = snapshot.get("recovered_state")
    if snapshot.get("receipt") != expected_receipt or recovered_state not in {
        "passed",
        "failed",
        "pending",
        "output-expired",
        "status-unavailable",
        "terminal-journal-pending-call",
        "upload-unknown",
        "not-submitted",
        "submitting",
        "submission-unknown",
        "claim-creating",
        "claim-creation-unknown",
        "claim-missing",
        "claimed-without-call",
    }:
        raise BundleStageError("Modal bundle status snapshot identity or state is invalid")
    if recovered_state == "passed" and not (
        snapshot.get("function_call_state") == "passed"
        or (
            snapshot.get("function_call_state") == "output-expired"
            and snapshot.get("remote_result") is not None
        )
    ):
        raise BundleStageError("Modal bundle passed snapshot has no terminal call evidence")
    if recovered_state == "failed" and not (
        snapshot.get("function_call_state") == "failed"
        or (
            snapshot.get("function_call_state") == "output-expired"
            and snapshot.get("remote_failure") is not None
        )
    ):
        raise BundleStageError("Modal bundle failed snapshot has no terminal call evidence")
    assert isinstance(recovered_state, str)
    return recovered_state


def _assert_no_unresolved_receipts(
    receipt_root: Path,
    *,
    exclude_upload_id: str | None = None,
) -> None:
    resolved_root = receipt_root.resolve()
    excluded = None if exclude_upload_id is None else _validate_upload_id(exclude_upload_id)
    if not resolved_root.exists():
        return
    if resolved_root.is_symlink() or not resolved_root.is_dir():
        raise BundleStageError(
            f"Modal bundle receipt root is not a regular directory: {resolved_root}"
        )
    unresolved: list[str] = []
    for receipt_path in sorted(resolved_root.glob("*/receipt.json")):
        receipt = _read_receipt(receipt_path)
        if receipt["upload_id"] == excluded:
            continue
        recovered_state = _read_local_recovered_state(receipt_path.parent, receipt)
        if recovered_state in {"passed", "failed"}:
            continue
        unresolved.append(str(receipt_path))
    if unresolved:
        raise BundleStageError(
            "an earlier Modal bundle stage has no recovered terminal state; run the "
            f"status command before uploading again: {unresolved}"
        )


def _existing_receipt_root(
    receipt_path: Path,
    receipt: Mapping[str, object],
) -> Path:
    resolved_receipt_path = receipt_path.resolve()
    upload_id = _validate_upload_id(receipt.get("upload_id"))
    if resolved_receipt_path.name != "receipt.json":
        raise BundleStageError("Modal bundle recovery path must name receipt.json")
    receipt_root = resolved_receipt_path.parent.parent
    if _receipt_path(receipt_root, upload_id) != resolved_receipt_path:
        raise BundleStageError("Modal bundle recovery receipt is not below its upload-ID directory")
    return receipt_root


def _process_instance_state(process_id: int) -> tuple[str, str | None]:
    """Return a process-liveness state bound to an OS process instance."""

    if isinstance(process_id, bool) or process_id <= 0:
        return "unknown", None
    if os.name == "nt":
        process_query_limited_information = 0x1000
        error_invalid_parameter = 87
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class FileTime(ctypes.Structure):
            _fields_ = [
                ("low", ctypes.c_ulong),
                ("high", ctypes.c_ulong),
            ]

        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetProcessTimes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
        ]
        kernel32.GetProcessTimes.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            0,
            process_id,
        )
        if not handle:
            error_code = ctypes.get_last_error()
            if error_code == error_invalid_parameter:
                return "absent", None
            return "unknown", f"windows-error:{error_code}"
        try:
            creation = FileTime()
            exit_time = FileTime()
            kernel_time = FileTime()
            user_time = FileTime()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                return "unknown", f"windows-error:{ctypes.get_last_error()}"
            creation_ticks = (creation.high << 32) | creation.low
            return "running", f"windows-filetime:{creation_ticks}"
        finally:
            kernel32.CloseHandle(handle)
    if sys.platform.startswith("linux"):
        stat_path = Path(f"/proc/{process_id}/stat")
        try:
            content = stat_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "absent", None
        except OSError as error:
            return "unknown", f"linux-error:{error.errno}"
        command_end = content.rfind(")")
        fields = content[command_end + 2 :].split() if command_end >= 0 else []
        if len(fields) <= 19:
            return "unknown", None
        return "running", f"linux-start-ticks:{fields[19]}"
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return "absent", None
    except (OSError, PermissionError) as error:
        return "unknown", f"generic-error:{getattr(error, 'errno', None)}"
    return "running", f"pid-only:{process_id}"


def _validated_lock_owner(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BundleStageError("Modal bundle submission lock owner is not a JSON object")
    owner = cast(dict[str, object], value.copy())
    if (
        set(owner)
        != {
            "lock_version",
            "process_id",
            "process_instance_identity",
            "host_name",
            "acquired_at_utc",
        }
        or owner.get("lock_version") != 2
        or isinstance(owner.get("process_id"), bool)
        or not isinstance(owner.get("process_id"), int)
        or cast(int, owner["process_id"]) <= 0
        or not isinstance(owner.get("process_instance_identity"), str)
        or not isinstance(owner.get("host_name"), str)
        or not isinstance(owner.get("acquired_at_utc"), str)
    ):
        raise BundleStageError("Modal bundle submission lock owner is invalid")
    return owner


@contextmanager
def _exclusive_submission(receipt_root: Path) -> Generator[Path, None, None]:
    """Serialize large uploads and finalizer submissions with an atomic lock."""

    process_state, process_identity = _process_instance_state(os.getpid())
    if process_state != "running" or process_identity is None:
        raise BundleStageError("cannot bind the Modal submission lock to this process")
    resolved_root = receipt_root.resolve()
    if resolved_root.exists() and (resolved_root.is_symlink() or not resolved_root.is_dir()):
        raise BundleStageError(
            f"Modal bundle receipt root is not a regular directory: {resolved_root}"
        )
    resolved_root.mkdir(parents=True, exist_ok=True)
    lock_path = resolved_root / SUBMISSION_LOCK_NAME
    try:
        lock_path.mkdir()
    except FileExistsError as error:
        raise BundleStageError(
            "another Modal bundle stage may be active or an interrupted stage left a "
            f"fail-closed lock: {lock_path}"
        ) from error
    owner_path = lock_path / "owner.json"
    try:
        _write_json_atomic(
            owner_path,
            {
                "lock_version": 2,
                "process_id": os.getpid(),
                "process_instance_identity": process_identity,
                "host_name": socket.gethostname(),
                "acquired_at_utc": _utc_now(),
            },
        )
        yield resolved_root
    finally:
        owner_path.unlink(missing_ok=True)
        try:
            lock_path.rmdir()
        except FileNotFoundError as error:
            raise BundleStageError(
                f"Modal bundle submission lock disappeared while held: {lock_path}"
            ) from error


def recover_submission_lock(receipt_root: Path) -> Path:
    """Remove only a lock whose exact owning process instance has ended."""

    resolved_root = receipt_root.resolve()
    if resolved_root.is_symlink() or not resolved_root.is_dir():
        raise BundleStageError(
            f"Modal bundle receipt root is not a regular directory: {resolved_root}"
        )
    lock_path = resolved_root / SUBMISSION_LOCK_NAME
    _ensure_directory(lock_path, "Modal bundle submission lock")
    names = {entry.name for entry in lock_path.iterdir()}
    if names != {"owner.json"}:
        raise BundleStageError("Modal bundle submission lock has unexpected entries")
    owner_path = lock_path / "owner.json"
    owner = _validated_lock_owner(
        _read_local_json(owner_path, "Modal bundle submission lock owner")
    )
    if owner["host_name"] != socket.gethostname():
        raise BundleStageError("cannot recover a Modal bundle lock created on another host")
    process_id = cast(int, owner["process_id"])
    state, observed_identity = _process_instance_state(process_id)
    if state == "unknown":
        raise BundleStageError("cannot prove that the Modal bundle lock owner has ended")
    if state == "running" and observed_identity == owner["process_instance_identity"]:
        raise BundleStageError("the Modal bundle submission lock owner is still running")
    recovery_path = resolved_root / (f"{SUBMISSION_LOCK_NAME}.recovering-{secrets.token_hex(8)}")
    os.rename(lock_path, recovery_path)
    try:
        recovered_owner = _validated_lock_owner(
            _read_local_json(
                recovery_path / "owner.json",
                "recovered Modal bundle submission lock owner",
            )
        )
        if recovered_owner != owner:
            raise BundleStageError("Modal bundle submission lock changed during recovery")
        (recovery_path / "owner.json").unlink()
        recovery_path.rmdir()
    except BaseException:
        if recovery_path.exists() and not lock_path.exists():
            recovered_owner_path = recovery_path / "owner.json"
            if not recovered_owner_path.exists():
                _write_json_atomic(recovered_owner_path, owner)
            os.rename(recovery_path, lock_path)
        raise
    print(_json_bytes({"recovered_submission_lock": str(lock_path)}).decode(), end="")
    return lock_path


def _validate_modal_client_version() -> None:
    observed = importlib.metadata.version("modal")
    if observed != EXPECTED_MODAL_CLIENT_VERSION:
        raise RuntimeError(
            "the Modal bundle stager requires local Modal client "
            f"{EXPECTED_MODAL_CLIENT_VERSION}, got {observed}"
        )


def _require_modal() -> Any:
    if modal is None:
        raise RuntimeError("install the pinned Modal client before staging a GPU bundle")
    _validate_modal_client_version()
    return modal


def _build_finalizer_runtime(modal_module: Any, volume: Any) -> tuple[Any, Any]:
    """Build one ephemeral, CPU-only finalizer definition for ``App.run``."""

    image = (
        modal_module.Image.debian_slim(python_version="3.11")
        .pip_install_from_requirements(
            str(FINALIZER_REQUIREMENTS),
            extra_options="--require-hashes --only-binary=:all: --no-cache-dir",
        )
        .add_local_dir(
            str(REPOSITORY_ROOT / SOURCE_PACKAGE_RELATIVE_PATH),
            str(REMOTE_SOURCE_PACKAGE),
            copy=True,
            ignore=("**/__pycache__/**", "**/*.pyc", "**/*.pyo"),
        )
        .add_local_file(
            str(PACKAGE_SCRIPT),
            str(REMOTE_PACKAGE_SCRIPT),
            copy=True,
        )
        .add_local_file(
            str(STAGE_SCRIPT),
            str(REMOTE_STAGE_SCRIPT),
            copy=True,
        )
        .add_local_file(
            str(FINALIZER_REQUIREMENTS),
            str(REMOTE_FINALIZER_REQUIREMENTS),
            copy=True,
        )
        .env({"PYTHONPATH": str(REMOTE_ROOT / "src"), "PYTHONUNBUFFERED": "1"})
    )
    app = modal_module.App(APP_NAME, image=image, include_source=False)

    @app.function(
        name=FINALIZER_FUNCTION_NAME,
        volumes={str(VOLUME_MOUNT): volume},
        cpu=FINALIZER_CPU_CORES,
        memory=FINALIZER_MEMORY_MIB,
        timeout=FINALIZER_TIMEOUT_SECONDS,
        retries=0,
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=FINALIZER_SCALEDOWN_WINDOW_SECONDS,
        single_use_containers=True,
        serialized=True,
        include_source=True,
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
        package_module = _load_package_module(Path(str(REMOTE_PACKAGE_SCRIPT)))
        result = _finalize_bundle(
            volume,
            Path(str(VOLUME_MOUNT)),
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


def _refresh_receipt_budget(
    receipt: dict[str, Any],
    *,
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
) -> None:
    """Bind the latest operator-observed hard-budget state before submission."""

    observation = _budget_observation(max_dollars, workspace_budget, workspace_usage)
    raw_history = receipt.get("budget_observations")
    if not isinstance(raw_history, list):
        raise BundleStageError("Modal bundle receipt budget history cannot be extended")
    history = cast(list[object], raw_history)
    if len(history) >= 32:
        raise BundleStageError("Modal bundle receipt budget history cannot be extended")
    history.append(observation)
    receipt["authorization_compute_charge_usd"] = observation["authorization_compute_charge_usd"]
    receipt["max_dollars"] = observation["max_dollars"]
    receipt["workspace_budget_usd"] = observation["workspace_budget_usd"]
    receipt["workspace_usage_before_submit_usd"] = observation["workspace_usage_usd"]
    receipt["workspace_budget_headroom_usd"] = observation["workspace_budget_headroom_usd"]


def _verify_receipt_runtime_contract(receipt: Mapping[str, object]) -> None:
    """Require the exact reviewed runtime that created the upload receipt."""

    runtime_contract = finalizer_runtime_contract_sha256(REPOSITORY_ROOT)
    if runtime_contract != receipt.get("runtime_contract_sha256"):
        raise BundleStageError("local Modal finalizer runtime no longer matches the upload receipt")


def _submission_claim_payload(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": SUBMISSION_CLAIM_SCHEMA,
        "upload_id": _validate_upload_id(receipt.get("upload_id")),
        "bundle_sha256": _validate_sha256(receipt.get("bundle_sha256")),
        "bundle_size": _validate_positive_size(receipt.get("bundle_size")),
        "runtime_contract_sha256": _validate_sha256(receipt.get("runtime_contract_sha256")),
        "submission_claim_id": _validate_submission_claim_id(receipt.get("submission_claim_id")),
        "receipt_created_at_utc": receipt.get("created_at_utc"),
    }


def _validated_submission_claim(
    value: object,
    receipt: Mapping[str, object],
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BundleStageError("Modal submission claim is not a JSON object")
    claim = cast(dict[str, object], value.copy())
    try:
        expected = _submission_claim_payload(receipt)
    except ValueError as error:
        raise BundleStageError(
            "Modal submission claim exists without a matching local claim identity"
        ) from error
    if claim != expected or not isinstance(claim.get("receipt_created_at_utc"), str):
        raise BundleStageError("Modal submission claim identity is invalid")
    return claim


def _create_submission_claim(
    volume: Any,
    receipt_path: Path,
    receipt: dict[str, Any],
) -> None:
    """Acquire the remote no-overwrite fence immediately before one spawn."""

    if (
        receipt["submission_claim_state"] != "not-created"
        or receipt["finalizer_state"] != "not-submitted"
        or receipt["function_call_id"] is not None
        or receipt["submission_claim_id"] is not None
    ):
        raise BundleStageError("Modal bundle submission claim is not fresh")
    claim_path = cast(str, receipt["remote_submission_claim_path"])
    receipt["submission_claim_id"] = _new_submission_claim_id()
    receipt["submission_claim_state"] = "creating"
    receipt["submission_claim_error"] = None
    _write_json_atomic(receipt_path, receipt)
    payload = _submission_claim_payload(receipt)
    try:
        content = io.BytesIO(_json_bytes(payload))
        with volume.batch_upload(force=False) as batch:
            batch.put_file(content, claim_path)
    except BaseException as error:
        receipt["submission_claim_state"] = "creation-unknown"
        receipt["submission_claim_error"] = _error_record(error)
        _write_json_atomic(receipt_path, receipt)
        raise
    receipt["submission_claim_state"] = "created"
    _write_json_atomic(receipt_path, receipt)


def _submit_finalizer(
    modal_module: Any,
    volume: Any,
    receipt_path: Path,
    receipt: dict[str, Any],
) -> None:
    """Submit exactly one detached finalizer and persist every ambiguous edge."""

    try:
        app, finalizer = _build_finalizer_runtime(modal_module, volume)
    except BaseException as error:
        receipt["finalizer_error"] = _error_record(error)
        _write_json_atomic(receipt_path, receipt)
        raise
    if receipt["submission_claim_state"] == "not-created":
        _create_submission_claim(volume, receipt_path, receipt)
    elif receipt["submission_claim_state"] != "created":
        raise BundleStageError("Modal bundle submission claim is not resolved")
    receipt["finalizer_state"] = "submitting"
    receipt["finalizer_error"] = None
    _write_json_atomic(receipt_path, receipt)
    app_entered = False
    try:
        with app.run(detach=True):
            app_entered = True
            function_call = finalizer.spawn(
                receipt["upload_id"],
                receipt["bundle_sha256"],
                receipt["bundle_size"],
                receipt["runtime_contract_sha256"],
            )
            call_id = function_call.object_id
            if not isinstance(call_id, str) or FUNCTION_CALL_ID_PATTERN.fullmatch(call_id) is None:
                raise RuntimeError("Modal returned an invalid finalizer FunctionCall ID")
            receipt["function_call_id"] = call_id
            receipt["finalizer_state"] = "submitted"
            _write_json_atomic(receipt_path, receipt)
    except BaseException as error:
        if receipt["function_call_id"] is None:
            receipt["finalizer_state"] = "submission-unknown" if app_entered else "not-submitted"
        receipt["finalizer_error"] = _error_record(error)
        _write_json_atomic(receipt_path, receipt)
        raise


def stage(
    bundle_path: Path,
    volume_name: str,
    receipt_root: Path,
    *,
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
) -> Path:
    """Verify, upload once, and detach one CPU finalizer with a durable receipt."""

    modal_module = _require_modal()
    validated_volume_name = _validate_volume_name(volume_name)
    authorized = validate_finalizer_cost_guard(max_dollars)
    budget_headroom = _validate_workspace_budget_guard(
        max_dollars,
        workspace_budget,
        workspace_usage,
    )
    with _exclusive_submission(receipt_root) as resolved_receipt_root:
        _assert_no_unresolved_receipts(resolved_receipt_root)
        runtime_contract_sha256 = finalizer_runtime_contract_sha256(REPOSITORY_ROOT)
        local_path, size, sha256, verification = _validate_local_bundle(bundle_path)
        initial_budget_observation = _budget_observation(
            max_dollars,
            workspace_budget,
            workspace_usage,
        )
        upload_id = _new_upload_id()
        incoming, final, operation = _remote_paths(upload_id, sha256)
        receipt_path = _receipt_path(resolved_receipt_root, upload_id)
        if receipt_path.parent.exists():
            raise FileExistsError(
                f"Modal bundle receipt directory already exists: {receipt_path.parent}"
            )
        receipt: dict[str, Any] = {
            "receipt_version": RECEIPT_VERSION,
            "upload_id": upload_id,
            "volume_name": validated_volume_name,
            "volume_version": VOLUME_VERSION,
            "app_name": APP_NAME,
            "function_name": FINALIZER_FUNCTION_NAME,
            "runtime_contract_sha256": runtime_contract_sha256,
            "local_bundle_path": str(local_path),
            "bundle_sha256": sha256,
            "bundle_size": size,
            "verification": verification,
            "remote_incoming_path": incoming,
            "remote_final_path": final,
            "remote_operation_path": operation,
            "created_at_utc": _utc_now(),
            "authorization_compute_charge_usd": authorized,
            "max_dollars": max_dollars,
            "workspace_budget_usd": workspace_budget,
            "workspace_usage_before_submit_usd": workspace_usage,
            "workspace_budget_headroom_usd": budget_headroom,
            "budget_observations": [initial_budget_observation],
            "upload_state": "intent",
            "upload_error": None,
            "finalizer_state": "not-submitted",
            "function_call_id": None,
            "finalizer_error": None,
            "submission_claim_id": None,
            "remote_submission_claim_path": _remote_submission_claim_path(upload_id),
            "submission_claim_state": "not-created",
            "submission_claim_error": None,
        }
        _write_json_atomic(receipt_path, receipt)

        volume = modal_module.Volume.from_name(
            validated_volume_name,
            create_if_missing=True,
            version=VOLUME_VERSION,
        )
        try:
            with volume.batch_upload(force=False) as batch:
                batch.put_file(str(local_path), f"{incoming}/bundle.zip")
            observed_size, observed_sha256 = _hash_regular_file_stable(
                local_path, "prepared GPU bundle"
            )
            if (observed_size, observed_sha256) != (size, sha256):
                raise BundleStageError("prepared GPU bundle changed during its Modal upload")
            receipt["upload_state"] = "uploaded"
            _write_json_atomic(receipt_path, receipt)
        except BaseException as error:
            receipt["upload_state"] = "upload-unknown"
            receipt["upload_error"] = _error_record(error)
            _write_json_atomic(receipt_path, receipt)
            raise

        _submit_finalizer(modal_module, volume, receipt_path, receipt)
    print(_json_bytes({"receipt_path": str(receipt_path), **receipt}).decode(), end="")
    return receipt_path


def resume_finalizer(
    receipt_path: Path,
    *,
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
) -> Path:
    """Finalize one interrupted upload without transmitting the archive again."""

    modal_module = _require_modal()
    validate_finalizer_cost_guard(max_dollars)
    _validate_workspace_budget_guard(max_dollars, workspace_budget, workspace_usage)
    resolved_receipt_path = receipt_path.resolve()
    initial_receipt = _read_receipt(resolved_receipt_path)
    receipt_root = _existing_receipt_root(resolved_receipt_path, initial_receipt)
    with _exclusive_submission(receipt_root):
        receipt = _read_receipt(resolved_receipt_path)
        if _existing_receipt_root(resolved_receipt_path, receipt) != receipt_root:
            raise BundleStageError("Modal bundle receipt identity changed during recovery")
        upload_id = cast(str, receipt["upload_id"])
        _assert_no_unresolved_receipts(
            receipt_root,
            exclude_upload_id=upload_id,
        )
        if receipt["finalizer_state"] != "not-submitted":
            raise BundleStageError(
                "resume-finalizer requires a receipt whose finalizer was never submitted"
            )
        _verify_receipt_runtime_contract(receipt)
        _refresh_receipt_budget(
            receipt,
            max_dollars=max_dollars,
            workspace_budget=workspace_budget,
            workspace_usage=workspace_usage,
        )
        _write_json_atomic(resolved_receipt_path, receipt)
        try:
            volume = modal_module.Volume.from_name(
                receipt["volume_name"],
                create_if_missing=False,
                version=VOLUME_VERSION,
            )
            operation_path = cast(str, receipt["remote_operation_path"])
            existing_operation_files = {
                name: _read_volume_json(volume, f"{operation_path}/{name}.json")
                for name in ("status", "result", "failure")
            }
            if any(value is not None for value in existing_operation_files.values()):
                raise BundleStageError(
                    "a durable finalizer operation already exists; run status instead of "
                    "submitting another finalizer"
                )
            raw_claim = _read_volume_json(
                volume,
                cast(str, receipt["remote_submission_claim_path"]),
            )
            claim_state = receipt["submission_claim_state"]
            if claim_state == "not-created":
                if raw_claim is not None:
                    raise BundleStageError(
                        "a remote submission claim exists without this receipt's identity"
                    )
            else:
                remote_claim = _validated_submission_claim(raw_claim, receipt)
                if remote_claim is None:
                    raise BundleStageError(
                        "the receipt records an attempted submission claim, but the remote "
                        "claim is not durably visible"
                    )
                receipt["submission_claim_state"] = "created"
                receipt["submission_claim_error"] = None
                _write_json_atomic(resolved_receipt_path, receipt)
        except BaseException as error:
            receipt["finalizer_error"] = _error_record(error)
            _write_json_atomic(resolved_receipt_path, receipt)
            raise
        _submit_finalizer(modal_module, volume, resolved_receipt_path, receipt)
    print(
        _json_bytes({"receipt_path": str(resolved_receipt_path), **receipt}).decode(),
        end="",
    )
    return resolved_receipt_path


def _relative_remote_path(remote_path: str) -> PurePosixPath:
    if not remote_path.startswith("/") or remote_path.startswith("//"):
        raise BundleStageError("internal Modal Volume path is not canonical and absolute")
    if "\\" in remote_path:
        raise BundleStageError("internal Modal Volume path contains a backslash")
    raw_parts = remote_path[1:].split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise BundleStageError("internal Modal Volume path is unsafe")
    relative = PurePosixPath(*raw_parts)
    if f"/{relative.as_posix()}" != remote_path:
        raise BundleStageError("internal Modal Volume path is not canonical")
    return relative


def _mounted_path(root: Path, remote_path: str) -> Path:
    return root.joinpath(*_relative_remote_path(remote_path).parts)


def _ensure_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise BundleStageError(f"{label} does not exist: {path}") from error
    if _is_link_like(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise BundleStageError(f"{label} must be a regular non-symlink directory: {path}")


def _mkdir_chain_no_links(root: Path, relative: PurePosixPath) -> Path:
    current = root
    _ensure_directory(root, "extraction root")
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        _ensure_directory(current, "extraction directory")
    return current


def _existing_directory_chain_no_links(
    root: Path,
    relative: PurePosixPath,
    label: str,
) -> Path:
    current = root
    _ensure_directory(root, "Modal Volume mount")
    for part in relative.parts:
        current = current / part
        _ensure_directory(current, label)
    return current


def _path_exists_no_follow(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _copy_stream(source: IO[bytes], destination: IO[bytes]) -> int:
    size = 0
    while True:
        chunk = source.read(COPY_BUFFER_SIZE)
        if not chunk:
            break
        destination.write(chunk)
        size += len(chunk)
    return size


def _stream_extract_archive(
    archive_path: Path,
    extraction_root: Path,
    *,
    package_module: Any = PACKAGE,
) -> None:
    """Stream regular ZIP members into a fresh tree without path traversal."""

    _ensure_directory(extraction_root, "extraction root")
    with zipfile.ZipFile(archive_path, mode="r", allowZip64=True) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.flag_bits & 0x1:
                raise BundleStageError(f"unsafe ZIP member for extraction: {info.filename!r}")
            original_filename = info.orig_filename
            member = PurePosixPath(info.filename)
            if (
                member.is_absolute()
                or original_filename != info.filename
                or member.as_posix() != info.filename
                or "\\" in original_filename
                or len(member.parts) < 2
                or member.parts[0] != package_module.ARCHIVE_ROOT
            ):
                raise BundleStageError(f"unsafe ZIP member path: {info.filename!r}")
            relative_payload = PurePosixPath(*member.parts[1:]).as_posix()
            package_module._validated_relative_path(relative_payload)
            unix_mode = (info.external_attr >> 16) & 0o177777
            if not stat.S_ISREG(unix_mode):
                raise BundleStageError(f"ZIP member is not a regular file: {info.filename!r}")
            parent_relative = PurePosixPath(*member.parts[:-1])
            parent = _mkdir_chain_no_links(extraction_root, parent_relative)
            destination = parent / member.name
            try:
                descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as error:
                raise BundleStageError(
                    f"ZIP extraction destination already exists: {info.filename!r}"
                ) from error
            try:
                with archive.open(info, mode="r") as source, os.fdopen(descriptor, "wb") as output:
                    descriptor = -1
                    written = _copy_stream(source, output)
                    output.flush()
                    os.fsync(output.fileno())
                if written != info.file_size:
                    raise BundleStageError(
                        f"ZIP member size changed during extraction: {info.filename!r}"
                    )
                os.chmod(destination, unix_mode & 0o777)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)


def _ready_payload(
    sha256: str,
    size: int,
    verification: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": READY_SCHEMA,
        "bundle_sha256": _validate_sha256(sha256),
        "bundle_size": size,
        "verification": dict(verification),
    }


def _read_json_file(path: Path, label: str) -> object:
    return _read_local_json(path, label)


def _verify_existing_final(
    final_directory: Path,
    expected_sha256: str,
    expected_size: int,
    *,
    package_module: Any = PACKAGE,
) -> dict[str, object]:
    """Verify an existing content-addressed artifact before allowing reuse."""

    try:
        _ensure_directory(final_directory, "content-addressed bundle")
        names = {entry.name for entry in final_directory.iterdir()}
        if names != {"bundle.zip", "tree", "READY"}:
            raise BundleStageError("content-addressed bundle has unexpected top-level entries")
        archive_path = final_directory / "bundle.zip"
        size, sha256 = _hash_regular_file_stable(archive_path, "content-addressed bundle")
        if (size, sha256) != (expected_size, expected_sha256):
            raise BundleStageError("content-addressed bundle archive identity conflicts")
        archive_verification = _verification_dict(package_module.verify_archive(archive_path))
        _assert_prepared_only_archive(archive_path, package_module)
        tree_verification = _verification_dict(package_module.verify_tree(final_directory / "tree"))
        if tree_verification != archive_verification:
            raise BundleStageError("content-addressed archive and tree verification disagree")
        ready = _read_json_file(final_directory / "READY", "content-addressed READY marker")
        if ready != _ready_payload(expected_sha256, expected_size, archive_verification):
            raise BundleStageError("content-addressed READY marker conflicts with its payload")
        return archive_verification
    except BaseException as error:
        if isinstance(error, BundleConflictError):
            raise
        raise BundleConflictError(
            f"refusing to overwrite conflicting content-addressed bundle: {final_directory}"
        ) from error


def _remove_verified_directory(path: Path, label: str) -> None:
    """Remove one already bounded non-link directory without following links."""

    _ensure_directory(path, label)
    shutil.rmtree(path)
    if path.exists():
        raise BundleStageError(f"{label} still exists after cleanup: {path}")


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing every existing destination."""

    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise BundleStageError("Linux runtime does not provide renameat2(RENAME_NOREPLACE)")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        result = renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise FileExistsError(observed_errno, os.strerror(observed_errno), destination)
        raise OSError(observed_errno, os.strerror(observed_errno), destination)
    os.rename(source, destination)


def _operation_status(
    upload_id: str,
    sha256: str,
    size: int,
    state: str,
    sequence: int,
    *,
    function_call_id: str,
    runtime_contract_sha256: str,
) -> dict[str, object]:
    if state not in {"running", "passed", "failed"}:
        raise ValueError("Modal bundle operation state is invalid")
    return {
        "schema": OPERATION_SCHEMA,
        "upload_id": _validate_upload_id(upload_id),
        "bundle_sha256": _validate_sha256(sha256),
        "bundle_size": _validate_positive_size(size),
        "function_call_id": _validate_function_call_id(function_call_id),
        "runtime_contract_sha256": _validate_sha256(runtime_contract_sha256),
        "state": state,
        "sequence": sequence,
        "updated_at_utc": _utc_now(),
    }


def _assert_existing_operation_owner(
    status_path: Path,
    upload_id: str,
    sha256: str,
    size: int,
    function_call_id: str,
    runtime_contract_sha256: str,
) -> None:
    value = _read_json_file(status_path, "existing Modal bundle operation status")
    if not isinstance(value, dict):
        raise BundleConflictError(
            "existing Modal bundle operation journal belongs to another or invalid "
            "FunctionCall and was preserved"
        )
    status = cast(dict[str, object], value)
    expected_identity = {
        "schema": OPERATION_SCHEMA,
        "upload_id": upload_id,
        "bundle_sha256": sha256,
        "bundle_size": size,
        "function_call_id": function_call_id,
        "runtime_contract_sha256": runtime_contract_sha256,
    }
    if (
        set(status)
        != {
            *expected_identity,
            "state",
            "sequence",
            "updated_at_utc",
        }
        or any(status.get(key) != expected for key, expected in expected_identity.items())
        or status.get("state") not in {"running", "passed", "failed"}
        or isinstance(status.get("sequence"), bool)
        or not isinstance(status.get("sequence"), int)
        or cast(int, status["sequence"]) < 1
        or not isinstance(status.get("updated_at_utc"), str)
    ):
        raise BundleConflictError(
            "existing Modal bundle operation journal belongs to another or invalid "
            "FunctionCall and was preserved"
        )


def _result_payload(
    upload_id: str,
    sha256: str,
    size: int,
    final_path: str,
    verification: Mapping[str, object],
    *,
    reused: bool,
    function_call_id: str,
    runtime_contract_sha256: str,
) -> dict[str, object]:
    return {
        "schema": RESULT_SCHEMA,
        "upload_id": _validate_upload_id(upload_id),
        "bundle_sha256": _validate_sha256(sha256),
        "bundle_size": size,
        "function_call_id": _validate_function_call_id(function_call_id),
        "runtime_contract_sha256": _validate_sha256(runtime_contract_sha256),
        "final_path": final_path,
        "reused": reused,
        "verification": dict(verification),
    }


def _finalize_materialization(
    volume: Any,
    mount_root: Path,
    upload_id: str,
    expected_sha256: str,
    expected_size: int,
    function_call_id: str,
    runtime_contract_sha256: str,
    *,
    package_module: Any,
) -> dict[str, object]:
    incoming_remote, final_remote, _operation_remote = _remote_paths(upload_id, expected_sha256)
    incoming_relative = _relative_remote_path(incoming_remote)
    final_relative = _relative_remote_path(final_remote)
    incoming = _mounted_path(mount_root, incoming_remote)
    final = _mounted_path(mount_root, final_remote)
    final_parent = _mkdir_chain_no_links(mount_root, final_relative.parent)
    if final_parent != final.parent:
        raise BundleStageError("internal content-addressed bundle path is inconsistent")
    if _path_exists_no_follow(final):
        _existing_directory_chain_no_links(
            mount_root,
            final_relative,
            "content-addressed bundle",
        )
        verification = _verify_existing_final(
            final,
            expected_sha256,
            expected_size,
            package_module=package_module,
        )
        if _path_exists_no_follow(incoming):
            _existing_directory_chain_no_links(
                mount_root,
                incoming_relative,
                "duplicate incoming bundle",
            )
            incoming_names = {entry.name for entry in incoming.iterdir()}
            if incoming_names != {"bundle.zip"}:
                raise BundleStageError(
                    "duplicate incoming bundle has unexpected entries and was preserved"
                )
            duplicate_size, duplicate_sha256 = _hash_regular_file_stable(
                incoming / "bundle.zip", "duplicate incoming bundle"
            )
            if (duplicate_size, duplicate_sha256) != (expected_size, expected_sha256):
                raise BundleStageError("duplicate incoming bundle identity conflicts")
            _remove_verified_directory(incoming, "duplicate incoming bundle")
        return _result_payload(
            upload_id,
            expected_sha256,
            expected_size,
            final_remote,
            verification,
            reused=True,
            function_call_id=function_call_id,
            runtime_contract_sha256=runtime_contract_sha256,
        )

    observed_incoming = _existing_directory_chain_no_links(
        mount_root,
        incoming_relative,
        "incoming bundle directory",
    )
    if observed_incoming != incoming:
        raise BundleStageError("internal incoming bundle path is inconsistent")
    archive_path = incoming / "bundle.zip"
    observed_size, observed_sha256 = _hash_regular_file_stable(
        archive_path, "uploaded prepared GPU bundle"
    )
    if (observed_size, observed_sha256) != (expected_size, expected_sha256):
        raise BundleStageError("uploaded bundle size or SHA-256 does not match the receipt")
    archive_verification = _verification_dict(package_module.verify_archive(archive_path))
    _assert_prepared_only_archive(archive_path, package_module)
    after_size, after_sha256 = _hash_regular_file_stable(
        archive_path, "uploaded prepared GPU bundle"
    )
    if (after_size, after_sha256) != (expected_size, expected_sha256):
        raise BundleStageError("uploaded bundle changed during archive verification")

    extraction_root = incoming / "tree"
    incoming_names = {entry.name for entry in incoming.iterdir()}
    if "READY" in incoming_names:
        if incoming_names != {"bundle.zip", "tree", "READY"}:
            raise BundleStageError("ready incoming bundle has unexpected entries")
        archive_verification = _verify_existing_final(
            incoming,
            expected_sha256,
            expected_size,
            package_module=package_module,
        )
    else:
        if not incoming_names.issubset({"bundle.zip", "tree"}):
            raise BundleStageError("incomplete incoming bundle has unexpected entries")
        if extraction_root.exists():
            _remove_verified_directory(extraction_root, "incomplete extraction tree")
        extraction_root.mkdir()
        _stream_extract_archive(
            archive_path,
            extraction_root,
            package_module=package_module,
        )
        final_size, final_sha256 = _hash_regular_file_stable(
            archive_path, "uploaded prepared GPU bundle"
        )
        if (final_size, final_sha256) != (expected_size, expected_sha256):
            raise BundleStageError("uploaded bundle changed during extraction")
        tree_verification = _verification_dict(package_module.verify_tree(extraction_root))
        if tree_verification != archive_verification:
            raise BundleStageError("uploaded archive and extracted tree verification disagree")
        _write_json_atomic(
            incoming / "READY",
            _ready_payload(expected_sha256, expected_size, archive_verification),
        )
    if not any(incoming.iterdir()):
        raise BundleStageError("incoming bundle directory is empty before publication")
    observed_final_parent = _existing_directory_chain_no_links(
        mount_root,
        final_relative.parent,
        "content-addressed bundle parent",
    )
    if observed_final_parent != final.parent:
        raise BundleStageError("internal content-addressed bundle path is inconsistent")
    try:
        _rename_directory_no_replace(incoming, final)
        reused = False
        verification = archive_verification
    except FileExistsError:
        verification = _verify_existing_final(
            final,
            expected_sha256,
            expected_size,
            package_module=package_module,
        )
        reused = True
        _remove_verified_directory(incoming, "verified duplicate incoming bundle")
    volume.commit()
    return _result_payload(
        upload_id,
        expected_sha256,
        expected_size,
        final_remote,
        verification,
        reused=reused,
        function_call_id=function_call_id,
        runtime_contract_sha256=runtime_contract_sha256,
    )


def _finalize_bundle_with_journal(
    volume: Any,
    mount_root: Path,
    upload_id: str,
    expected_sha256: str,
    expected_size: int,
    function_call_id: str,
    expected_runtime_contract_sha256: str,
    *,
    package_module: Any = PACKAGE,
) -> dict[str, object]:
    """Journal and finalize one already uploaded bundle on its mounted Volume."""

    validated_upload_id = _validate_upload_id(upload_id)
    validated_sha256 = _validate_sha256(expected_sha256)
    expected_size = _validate_positive_size(expected_size)
    validated_call_id = _validate_function_call_id(function_call_id)
    validated_runtime_contract = _validate_sha256(expected_runtime_contract_sha256)
    _ensure_directory(mount_root, "Modal Volume mount")
    # The upload is committed by a different client before this fresh container
    # starts. Reload explicitly so a reused worker cannot observe an older view.
    volume.reload()
    _incoming, _final, operation_remote = _remote_paths(validated_upload_id, validated_sha256)
    operation_relative = _relative_remote_path(operation_remote)
    operation_parent = _mkdir_chain_no_links(mount_root, operation_relative.parent)
    operation = operation_parent / operation_relative.name
    if operation != _mounted_path(mount_root, operation_remote):
        raise BundleStageError("internal Modal bundle operation path is inconsistent")
    status_path = operation / "status.json"
    result_path = operation / "result.json"
    failure_path = operation / "failure.json"
    try:
        operation.mkdir()
    except FileExistsError:
        _existing_directory_chain_no_links(
            mount_root,
            operation_relative,
            "existing Modal bundle operation directory",
        )
        _assert_existing_operation_owner(
            status_path,
            validated_upload_id,
            validated_sha256,
            expected_size,
            validated_call_id,
            validated_runtime_contract,
        )
    try:
        result_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        _write_json_atomic(
            status_path,
            _operation_status(
                validated_upload_id,
                validated_sha256,
                expected_size,
                "running",
                1,
                function_call_id=validated_call_id,
                runtime_contract_sha256=validated_runtime_contract,
            ),
        )
        volume.commit()
        _verify_remote_runtime_contract(validated_runtime_contract)
        result = _finalize_materialization(
            volume,
            mount_root,
            validated_upload_id,
            validated_sha256,
            expected_size,
            validated_call_id,
            validated_runtime_contract,
            package_module=package_module,
        )
        failure_path.unlink(missing_ok=True)
        _write_json_atomic(result_path, result)
        _write_json_atomic(
            status_path,
            _operation_status(
                validated_upload_id,
                validated_sha256,
                expected_size,
                "passed",
                2,
                function_call_id=validated_call_id,
                runtime_contract_sha256=validated_runtime_contract,
            ),
        )
        volume.commit()
        return result
    except BaseException as error:
        result_path.unlink(missing_ok=True)
        failure = {
            "schema": OPERATION_SCHEMA,
            "upload_id": validated_upload_id,
            "bundle_sha256": validated_sha256,
            "bundle_size": expected_size,
            "function_call_id": validated_call_id,
            "runtime_contract_sha256": validated_runtime_contract,
            **_error_record(error),
            "traceback_tail": traceback.format_exc()[-32_000:],
        }
        try:
            _write_json_atomic(failure_path, failure)
            _write_json_atomic(
                status_path,
                _operation_status(
                    validated_upload_id,
                    validated_sha256,
                    expected_size,
                    "failed",
                    2,
                    function_call_id=validated_call_id,
                    runtime_contract_sha256=validated_runtime_contract,
                ),
            )
            volume.commit()
        except BaseException as journal_error:
            print(
                "Failed to persist the durable Modal bundle failure journal: "
                f"{type(journal_error).__name__}: {journal_error}",
                file=sys.stderr,
                flush=True,
            )
        raise RuntimeError(
            f"durable Modal bundle finalizer {validated_upload_id} failed; "
            "inspect its Volume journal"
        ) from error


def _finalize_bundle(
    volume: Any,
    mount_root: Path,
    upload_id: str,
    expected_sha256: str,
    expected_size: int,
    function_call_id: str,
    expected_runtime_contract_sha256: str,
    *,
    package_module: Any = PACKAGE,
) -> dict[str, object]:
    """Return only a result or one run-bound terminal failure envelope."""

    validated_upload_id = _validate_upload_id(upload_id)
    terminal_message = (
        f"durable Modal bundle finalizer {validated_upload_id} failed; inspect its Volume journal"
    )
    try:
        return _finalize_bundle_with_journal(
            volume,
            mount_root,
            validated_upload_id,
            expected_sha256,
            expected_size,
            function_call_id,
            expected_runtime_contract_sha256,
            package_module=package_module,
        )
    except BaseException as error:
        if type(error) is RuntimeError and str(error) == terminal_message:
            raise
        raise RuntimeError(terminal_message) from error


def _read_volume_json(volume: Any, path: str) -> object | None:
    try:
        payload = bytearray()
        for chunk in volume.read_file(path):
            if not isinstance(chunk, bytes):
                raise BundleStageError(f"Modal Volume returned a non-byte chunk for {path}")
            payload.extend(chunk)
            if len(payload) > MAX_JSON_BYTES:
                raise BundleStageError(f"Modal Volume JSON is unreasonably large: {path}")
    except FileNotFoundError:
        return None
    try:
        return json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BundleStageError(f"Modal Volume contains invalid JSON: {path}") from error


def _validated_operation_status(
    value: object,
    receipt: Mapping[str, object],
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BundleStageError("durable bundle operation status is not a JSON object")
    status_value = cast(dict[str, object], value.copy())
    if (
        set(status_value)
        != {
            "schema",
            "upload_id",
            "bundle_sha256",
            "bundle_size",
            "function_call_id",
            "runtime_contract_sha256",
            "state",
            "sequence",
            "updated_at_utc",
        }
        or status_value.get("schema") != OPERATION_SCHEMA
        or status_value.get("upload_id") != receipt["upload_id"]
        or status_value.get("bundle_sha256") != receipt["bundle_sha256"]
        or status_value.get("bundle_size") != receipt["bundle_size"]
        or not isinstance(status_value.get("function_call_id"), str)
        or FUNCTION_CALL_ID_PATTERN.fullmatch(cast(str, status_value["function_call_id"])) is None
        or (
            receipt.get("function_call_id") is not None
            and status_value.get("function_call_id") != receipt["function_call_id"]
        )
        or status_value.get("runtime_contract_sha256") != receipt["runtime_contract_sha256"]
        or status_value.get("state") not in {"running", "passed", "failed"}
        or isinstance(status_value.get("sequence"), bool)
        or not isinstance(status_value.get("sequence"), int)
        or cast(int, status_value["sequence"]) < 1
        or not isinstance(status_value.get("updated_at_utc"), str)
    ):
        raise BundleStageError("durable bundle operation status is invalid")
    return status_value


def _validated_result(value: object, receipt: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BundleStageError("durable bundle result is not a JSON object")
    result = cast(dict[str, object], value.copy())
    if (
        set(result)
        != {
            "schema",
            "upload_id",
            "bundle_sha256",
            "bundle_size",
            "function_call_id",
            "runtime_contract_sha256",
            "final_path",
            "reused",
            "verification",
        }
        or result.get("schema") != RESULT_SCHEMA
        or result.get("upload_id") != receipt["upload_id"]
        or result.get("bundle_sha256") != receipt["bundle_sha256"]
        or result.get("bundle_size") != receipt["bundle_size"]
        or not isinstance(result.get("function_call_id"), str)
        or FUNCTION_CALL_ID_PATTERN.fullmatch(cast(str, result["function_call_id"])) is None
        or (
            receipt.get("function_call_id") is not None
            and result.get("function_call_id") != receipt["function_call_id"]
        )
        or result.get("runtime_contract_sha256") != receipt["runtime_contract_sha256"]
        or result.get("final_path") != receipt["remote_final_path"]
        or not isinstance(result.get("reused"), bool)
    ):
        raise BundleStageError("durable bundle result identity is invalid")
    _validated_verification(result.get("verification"))
    return result


def _validated_failure(
    value: object,
    receipt: Mapping[str, object],
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise BundleStageError("durable bundle failure is not a JSON object")
    failure = cast(dict[str, object], value.copy())
    if (
        set(failure)
        != {
            "schema",
            "upload_id",
            "bundle_sha256",
            "bundle_size",
            "function_call_id",
            "runtime_contract_sha256",
            "error_type",
            "message",
            "recorded_at_utc",
            "traceback_tail",
        }
        or failure.get("schema") != OPERATION_SCHEMA
        or failure.get("upload_id") != receipt["upload_id"]
        or failure.get("bundle_sha256") != receipt["bundle_sha256"]
        or failure.get("bundle_size") != receipt["bundle_size"]
        or not isinstance(failure.get("function_call_id"), str)
        or FUNCTION_CALL_ID_PATTERN.fullmatch(cast(str, failure["function_call_id"])) is None
        or (
            receipt.get("function_call_id") is not None
            and failure.get("function_call_id") != receipt["function_call_id"]
        )
        or failure.get("runtime_contract_sha256") != receipt["runtime_contract_sha256"]
        or not all(
            isinstance(failure.get(field), str)
            for field in ("error_type", "message", "recorded_at_utc", "traceback_tail")
        )
    ):
        raise BundleStageError("durable bundle failure identity or fields are invalid")
    return failure


def status(receipt_path: Path) -> dict[str, object]:
    """Recover durable Volume state and a non-blocking detached call state."""

    modal_module = _require_modal()
    resolved_receipt_path = receipt_path.resolve()
    receipt = _read_receipt(resolved_receipt_path)
    volume = modal_module.Volume.from_name(
        receipt["volume_name"],
        create_if_missing=False,
        version=VOLUME_VERSION,
    )
    operation = cast(str, receipt["remote_operation_path"])
    remote_status = _validated_operation_status(
        _read_volume_json(volume, f"{operation}/status.json"), receipt
    )
    raw_result = _read_volume_json(volume, f"{operation}/result.json")
    remote_result = None if raw_result is None else _validated_result(raw_result, receipt)
    remote_failure = _validated_failure(
        _read_volume_json(volume, f"{operation}/failure.json"), receipt
    )

    call_state = "identity-unavailable"
    call_result: dict[str, object] | None = None
    call_error: dict[str, str] | None = None
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
    if isinstance(call_id, str):
        try:
            function_call = modal_module.FunctionCall.from_id(call_id)
        except Exception as error:
            call_state = "unavailable"
            call_error = _error_record(error)
        else:
            try:
                raw_call_result = function_call.get(timeout=0)
            except Exception as error:
                modal_exception_module = getattr(modal_module, "exception", None)
                output_expired_type = getattr(modal_exception_module, "OutputExpiredError", None)
                modal_timeout_type = getattr(modal_exception_module, "TimeoutError", None)
                is_exact_modal_poll_timeout = (
                    isinstance(modal_timeout_type, type) and type(error) is modal_timeout_type
                )
                is_durable_remote_failure = type(error) is RuntimeError and str(error) == (
                    f"durable Modal bundle finalizer {receipt['upload_id']} failed; "
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
                    call_state = "pending"
                elif is_durable_remote_failure or (
                    terminal_error_types and isinstance(error, terminal_error_types)
                ):
                    call_state = "failed"
                    call_error = _error_record(error)
                else:
                    call_state = "unavailable"
                    call_error = _error_record(error)
            else:
                call_result = _validated_result(raw_call_result, receipt)
                call_state = "passed"

    if remote_result is not None and call_result is not None and remote_result != call_result:
        raise BundleStageError("Modal FunctionCall and durable bundle results disagree")
    if remote_result is not None and remote_failure is not None:
        raise BundleStageError("durable bundle operation has both result and failure")
    if remote_result is not None and call_state == "failed":
        raise BundleStageError("Modal Volume success and finalizer FunctionCall failure disagree")
    if remote_failure is not None and call_result is not None:
        raise BundleStageError("Modal Volume failure and finalizer FunctionCall success disagree")
    if remote_result is not None and (
        remote_status is None or remote_status.get("state") != "passed"
    ):
        raise BundleStageError("durable bundle result has no matching passed status")
    if remote_failure is not None and (
        remote_status is None or remote_status.get("state") != "failed"
    ):
        raise BundleStageError("durable bundle failure has no matching failed status")
    if (
        remote_status is not None
        and remote_status.get("state") == "passed"
        and remote_result is None
    ):
        raise BundleStageError("durable bundle status passed without a result")
    if (
        remote_status is not None
        and remote_status.get("state") == "failed"
        and remote_failure is None
    ):
        raise BundleStageError("durable bundle status failed without diagnostics")

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
    elif call_state == "pending" and (remote_result is not None or remote_failure is not None):
        recovered_state = "terminal-journal-pending-call"
    elif remote_status is not None:
        recovered_state = cast(str, remote_status["state"])
    elif call_state == "pending":
        recovered_state = "pending"
    elif receipt["submission_claim_state"] == "creating":
        recovered_state = "claim-creating"
    elif receipt["submission_claim_state"] == "creation-unknown":
        recovered_state = "claim-creation-unknown"
    elif (
        receipt["submission_claim_state"] == "created"
        and receipt["finalizer_state"] == "not-submitted"
    ):
        recovered_state = "claimed-without-call"
    elif receipt["upload_state"] == "upload-unknown":
        recovered_state = "upload-unknown"
    else:
        recovered_state = cast(str, receipt["finalizer_state"])
    snapshot: dict[str, object] = {
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
    }
    _write_json_atomic(resolved_receipt_path.parent / "status-latest.json", snapshot)
    print(_json_bytes(snapshot).decode(), end="")
    return snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage or recover one verified prepared-only Modal GPU bundle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage_parser = subparsers.add_parser(
        "stage", help="Upload once and detach one CPU-only finalizer."
    )
    stage_parser.add_argument("--bundle", required=True, type=Path)
    stage_parser.add_argument("--volume", required=True)

    def add_budget_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--max-dollars",
            required=True,
            type=float,
            help="Required CPU-finalizer contingency authorization.",
        )
        command_parser.add_argument(
            "--workspace-budget",
            required=True,
            type=float,
            help="Hard Workspace usage budget currently set in Modal.",
        )
        command_parser.add_argument(
            "--workspace-usage",
            required=True,
            type=float,
            help="Current-cycle usage shown immediately before staging.",
        )

    add_budget_arguments(stage_parser)
    stage_parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPT_ROOT)
    resume_parser = subparsers.add_parser(
        "resume-finalizer",
        help="Finalize an interrupted upload without uploading its archive again.",
    )
    resume_parser.add_argument("--receipt", required=True, type=Path)
    add_budget_arguments(resume_parser)
    recover_lock_parser = subparsers.add_parser(
        "recover-lock",
        help="Remove a local submission lock only after its process has ended.",
    )
    recover_lock_parser.add_argument(
        "--receipt-root",
        type=Path,
        default=DEFAULT_RECEIPT_ROOT,
    )
    status_parser = subparsers.add_parser("status", help="Recover one operation without waiting.")
    status_parser.add_argument("--receipt", required=True, type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    if parsed.command == "stage":
        stage(
            parsed.bundle,
            parsed.volume,
            parsed.receipt_root,
            max_dollars=parsed.max_dollars,
            workspace_budget=parsed.workspace_budget,
            workspace_usage=parsed.workspace_usage,
        )
        return 0
    if parsed.command == "status":
        status(parsed.receipt)
        return 0
    if parsed.command == "resume-finalizer":
        resume_finalizer(
            parsed.receipt,
            max_dollars=parsed.max_dollars,
            workspace_budget=parsed.workspace_budget,
            workspace_usage=parsed.workspace_usage,
        )
        return 0
    if parsed.command == "recover-lock":
        recover_submission_lock(parsed.receipt_root)
        return 0
    raise RuntimeError(f"unsupported Modal bundle command: {parsed.command}")


if __name__ == "__main__":
    raise SystemExit(main())
