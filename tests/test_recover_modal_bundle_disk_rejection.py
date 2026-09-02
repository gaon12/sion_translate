from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterator

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recover_modal_bundle_disk_rejection.py"
SPEC = importlib.util.spec_from_file_location("recover_modal_bundle_disk_rejection_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

UPLOAD_ID = "bundle-20260902t065624z-174a5e7835536906"
CALL_ID = "fc-0123456789abcdef"
RUNTIME_HASH = "a" * 64
BUNDLE_HASH = "b" * 64
COMMIT = "c" * 40


def _receipt(bundle: Path) -> dict[str, Any]:
    return {
        "upload_id": UPLOAD_ID,
        "upload_state": "uploaded",
        "submission_claim_state": "created",
        "finalizer_state": "submission-unknown",
        "function_call_id": None,
        "finalizer_error": {
            "error_type": MODULE.EXPECTED_ERROR_TYPE,
            "message": MODULE.EXPECTED_ERROR_MESSAGE,
            "recorded_at_utc": "2026-09-02T07:00:39+00:00",
        },
        "runtime_contract_sha256": RUNTIME_HASH,
        "bundle_sha256": BUNDLE_HASH,
        "bundle_size": bundle.stat().st_size,
        "local_bundle_path": str(bundle),
        "verification": {"git_commit": COMMIT},
        "volume_name": "sion-prepared-bundles",
        "remote_operation_path": f"/operations/{UPLOAD_ID}",
        "remote_submission_claim_path": f"/submission-claims/{UPLOAD_ID}.json",
        "submission_claim_id": "claim-0123456789abcdef0123456789abcdef",
        "budget_observations": [],
    }


def _status(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "receipt": receipt,
        "recovered_state": "submission-unknown",
        "function_call_state": "identity-unavailable",
        "function_call_id_source": "unavailable",
        "observed_function_call_id": None,
        "function_call_result": None,
        "function_call_error": None,
        "remote_status": None,
        "remote_result": None,
        "remote_failure": None,
    }


class _FakeLegacy:
    VOLUME_VERSION = 1
    FINALIZER_EPHEMERAL_DISK_MIB: int | None = 2_048

    def __init__(self, runtime_root: Path, remote_files: dict[str, object]) -> None:
        self.REPOSITORY_ROOT = runtime_root
        self.remote_files = remote_files
        self.events: list[object] = []
        outer = self

        class Volume:
            @staticmethod
            def from_name(name: str, *, create_if_missing: bool, version: int) -> object:
                outer.events.append(("volume", name, create_if_missing, version))
                return object()

        self._modal = SimpleNamespace(Volume=Volume)

    def _read_receipt(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def finalizer_runtime_contract_sha256(self, _root: Path) -> str:
        return RUNTIME_HASH

    def _hash_regular_file_stable(self, path: Path, _label: str) -> tuple[int, str]:
        payload = path.read_bytes()
        return len(payload), hashlib.sha256(payload).hexdigest()

    def _require_modal(self) -> object:
        return self._modal

    def validate_finalizer_cost_guard(self, max_dollars: float) -> float:
        self.events.append(("cost", max_dollars))
        return max_dollars

    def _validate_workspace_budget_guard(
        self,
        max_dollars: float,
        workspace_budget: float,
        workspace_usage: float,
    ) -> float:
        self.events.append(("budget", max_dollars, workspace_budget, workspace_usage))
        return workspace_budget - workspace_usage

    def _existing_receipt_root(self, receipt_path: Path, _receipt: object) -> Path:
        return receipt_path.parent.parent

    @contextmanager
    def _exclusive_submission(self, root: Path) -> Iterator[None]:
        self.events.append(("lock-enter", root))
        try:
            yield
        finally:
            self.events.append(("lock-exit", root))

    def _assert_no_unresolved_receipts(self, root: Path, *, exclude_upload_id: str) -> None:
        self.events.append(("scan", root, exclude_upload_id))

    def _read_volume_json(self, _volume: object, path: str) -> object | None:
        self.events.append(("read", path))
        return self.remote_files.get(path)

    def _validated_submission_claim(self, value: object, _receipt: object) -> object | None:
        return value if value == {"claim": "exact"} else None

    def _write_json_atomic(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def _refresh_receipt_budget(
        self,
        receipt: dict[str, Any],
        *,
        max_dollars: float,
        workspace_budget: float,
        workspace_usage: float,
    ) -> None:
        receipt["budget_observations"].append(
            {
                "max_dollars": max_dollars,
                "workspace_budget": workspace_budget,
                "workspace_usage": workspace_usage,
            }
        )

    def _submit_finalizer(
        self,
        _modal: object,
        _volume: object,
        receipt_path: Path,
        receipt: dict[str, Any],
    ) -> None:
        assert self.FINALIZER_EPHEMERAL_DISK_MIB is None
        assert receipt["submission_claim_state"] == "created"
        self.events.append("submitted-without-upload")
        receipt["finalizer_state"] = "submitted"
        receipt["function_call_id"] = CALL_ID
        self._write_json_atomic(receipt_path, receipt)

    @staticmethod
    def _json_bytes(value: object) -> bytes:
        return (json.dumps(value, sort_keys=True) + "\n").encode()


def test_materialize_legacy_runtime_preserves_noncontroller_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    (repository / "scripts").mkdir(parents=True)
    (repository / "requirements").mkdir()
    (repository / "src" / "sion_translate").mkdir(parents=True)
    (repository / "scripts" / "package_gpu_bundle.py").write_bytes(b"package\r\n")
    (repository / "requirements" / "modal-bundle-stage.txt").write_bytes(b"modal\r\n")
    (repository / "src" / "sion_translate" / "__init__.py").write_bytes(b"source\r\n")
    monkeypatch.setattr(MODULE, "_git_blob", lambda *_args: b"legacy-stage\n")

    stage = MODULE._materialize_legacy_runtime(repository, COMMIT, tmp_path / "runtime")

    assert stage.read_bytes() == b"legacy-stage\n"
    assert (tmp_path / "runtime" / "scripts" / "package_gpu_bundle.py").read_bytes() == (
        b"package\r\n"
    )
    assert (
        tmp_path / "runtime" / "src" / "sion_translate" / "__init__.py"
    ).read_bytes() == b"source\r\n"


def test_reconstructed_modules_are_serialized_without_remote_imports(tmp_path: Path) -> None:
    helper_path = tmp_path / "legacy_helper.py"
    helper_path.write_text("VALUE = 'serialized by value'\n", encoding="utf-8")
    helper_spec = importlib.util.spec_from_file_location("sion_legacy_helper_test", helper_path)
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_spec.name] = helper
    helper_spec.loader.exec_module(helper)

    stage_path = tmp_path / "legacy_stage.py"
    stage_path.write_text(
        "import sion_legacy_helper_test as PACKAGE\ndef read_value():\n    return PACKAGE.VALUE\n",
        encoding="utf-8",
    )
    legacy = MODULE._load_legacy_module(stage_path)

    from modal._serialization import deserialize, serialize

    try:
        with MODULE._pickle_legacy_runtime_by_value(legacy):
            payload = serialize(legacy.read_value)
        sys.modules.pop(legacy.__name__, None)
        sys.modules.pop(helper.__name__, None)
        restored = deserialize(payload, None)
        assert restored() == "serialized by value"
    finally:
        sys.modules.pop(legacy.__name__, None)
        sys.modules.pop(helper.__name__, None)


def test_recovery_reuses_the_claim_and_never_uploads_the_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = tmp_path / "prepared.zip"
    bundle.write_bytes(b"prepared archive")
    receipt = _receipt(bundle)
    receipt["bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    receipt_path = tmp_path / "receipts" / UPLOAD_ID / "receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (receipt_path.parent / "status-latest.json").write_text(
        json.dumps(_status(receipt)), encoding="utf-8"
    )
    remote_files = {str(receipt["remote_submission_claim_path"]): {"claim": "exact"}}
    legacy = _FakeLegacy(tmp_path / "runtime", remote_files)

    MODULE._recover_with_legacy_module(
        legacy,
        receipt_path,
        max_dollars=1.27,
        workspace_budget=1.50,
        workspace_usage=0.0,
    )

    recovered = legacy._read_receipt(receipt_path)
    assert recovered["finalizer_state"] == "submitted"
    assert recovered["function_call_id"] == CALL_ID
    assert "submitted-without-upload" in legacy.events
    assert not any(event == "upload" for event in legacy.events)
    assert (receipt_path.parent / "receipt-before-disk-recovery.json").is_file()
    intent = MODULE._read_json(receipt_path.parent / "disk-recovery-intent.json", "recovery intent")
    assert intent["archive_reuploaded"] is False
    assert json.loads(capsys.readouterr().out)["archive_reuploaded"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("upload_state", "upload-unknown"),
        ("submission_claim_state", "creation-unknown"),
        ("finalizer_state", "submitting"),
        ("function_call_id", CALL_ID),
        (
            "finalizer_error",
            {"error_type": "ConnectionError", "message": MODULE.EXPECTED_ERROR_MESSAGE},
        ),
        (
            "finalizer_error",
            {"error_type": MODULE.EXPECTED_ERROR_TYPE, "message": "different rejection"},
        ),
    ),
)
def test_recovery_rejects_any_receipt_evidence_deviation(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    bundle = tmp_path / "prepared.zip"
    bundle.write_bytes(b"prepared archive")
    receipt = _receipt(bundle)
    receipt[field] = value

    with pytest.raises(MODULE.LegacyRecoveryError, match="exact pre-spawn"):
        MODULE._validate_rejected_receipt(receipt)


def test_recovery_rejects_a_remote_journal_or_missing_claim(tmp_path: Path) -> None:
    bundle = tmp_path / "prepared.zip"
    bundle.write_bytes(b"prepared archive")
    receipt = _receipt(bundle)
    receipt["bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    receipt_path = tmp_path / "receipts" / UPLOAD_ID / "receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (receipt_path.parent / "status-latest.json").write_text(
        json.dumps(_status(receipt)), encoding="utf-8"
    )
    status_path = f"{receipt['remote_operation_path']}/status.json"
    claim_path = str(receipt["remote_submission_claim_path"])
    legacy = _FakeLegacy(
        tmp_path / "runtime",
        {status_path: {"state": "running"}, claim_path: {"claim": "exact"}},
    )

    with pytest.raises(MODULE.LegacyRecoveryError, match="journal now exists"):
        MODULE._recover_with_legacy_module(
            legacy,
            receipt_path,
            max_dollars=1.27,
            workspace_budget=1.50,
            workspace_usage=0.0,
        )
    assert "submitted-without-upload" not in legacy.events

    legacy.remote_files = {}
    with pytest.raises(MODULE.LegacyRecoveryError, match="claim is missing"):
        MODULE._recover_with_legacy_module(
            legacy,
            receipt_path,
            max_dollars=1.27,
            workspace_budget=1.50,
            workspace_usage=0.0,
        )
    assert "submitted-without-upload" not in legacy.events
