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
REPLACEMENT_CALL_ID = "fc-fedcba9876543210"
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


def test_exact_finalizer_imports_the_attested_remote_stage_module(tmp_path: Path) -> None:
    stage_source = """
from pathlib import Path, PurePosixPath
REPOSITORY_ROOT = Path('.')
SOURCE_PACKAGE_RELATIVE_PATH = Path('src/sion_translate')
FINALIZER_REQUIREMENTS = Path('requirements/modal-bundle-stage.txt')
PACKAGE_SCRIPT = Path('scripts/package_gpu_bundle.py')
STAGE_SCRIPT = Path('scripts/modal_stage_gpu_bundle.py')
REMOTE_ROOT = PurePosixPath('/opt/sion-bundle-stage')
REMOTE_SOURCE_PACKAGE = REMOTE_ROOT / 'src/sion_translate'
REMOTE_PACKAGE_SCRIPT = REMOTE_ROOT / 'scripts/package_gpu_bundle.py'
REMOTE_STAGE_SCRIPT = REMOTE_ROOT / 'scripts/modal_stage_gpu_bundle.py'
REMOTE_FINALIZER_REQUIREMENTS = REMOTE_ROOT / 'requirements/modal-bundle-stage.txt'
APP_NAME = 'test-finalizer'
FINALIZER_FUNCTION_NAME = 'finalize_prepared_bundle'
VOLUME_MOUNT = PurePosixPath('/mnt/sion-bundles')
FINALIZER_CPU_CORES = 2.0
FINALIZER_MEMORY_MIB = 8192
FINALIZER_TIMEOUT_SECONDS = 14400
FINALIZER_SCALEDOWN_WINDOW_SECONDS = 2
def _load_package_module(path): return path
def _finalize_bundle(*args, **kwargs): return {'state': 'passed'}
"""
    local_stage = tmp_path / "local" / "modal_stage_gpu_bundle.py"
    local_stage.parent.mkdir()
    local_stage.write_text(stage_source, encoding="utf-8")
    legacy = MODULE._load_legacy_module(local_stage)

    class FakeImage:
        environment: dict[str, str] = {}

        @classmethod
        def debian_slim(cls, **_kwargs: object) -> "FakeImage":
            return cls()

        def __getattr__(self, _name: str) -> Any:
            def chain(*_args: object, **_kwargs: object) -> "FakeImage":
                return self

            return chain

        def env(self, values: dict[str, str]) -> "FakeImage":
            type(self).environment = values
            return self

    class FakeApp:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def function(self, **_kwargs: object) -> Any:
            return lambda function: function

    fake_modal = SimpleNamespace(
        Image=FakeImage,
        App=FakeApp,
        current_function_call_id=lambda: CALL_ID,
    )
    _app, finalizer = MODULE._build_importable_finalizer_runtime(
        legacy,
        fake_modal,
        object(),
    )

    from modal._serialization import deserialize, serialize

    payload = serialize(finalizer)
    remote_scripts = tmp_path / "remote" / "scripts"
    remote_scripts.mkdir(parents=True)
    remote_stage = remote_scripts / "modal_stage_gpu_bundle.py"
    remote_stage.write_text(stage_source + "REMOTE_MARKER = True\n", encoding="utf-8")
    sys.modules.pop(legacy.__name__, None)
    sys.path.insert(0, str(remote_scripts))
    try:
        restored = deserialize(payload, None)
        closure = dict(
            zip(
                restored.__code__.co_freevars,
                (cell.cell_contents for cell in restored.__closure__ or ()),
                strict=True,
            )
        )
        restored_legacy = closure["legacy"]
        assert Path(restored_legacy.__file__).resolve() == remote_stage.resolve()
        assert restored_legacy.REMOTE_MARKER is True
        assert FakeImage.environment["PYTHONPATH"] == (
            "/opt/sion-bundle-stage/src:/opt/sion-bundle-stage/scripts"
        )
    finally:
        sys.path.remove(str(remote_scripts))
        sys.modules.pop(legacy.__name__, None)


def test_cli_routes_preflight_only_to_deserialization_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_recovery(receipt: Path, **options: object) -> Path:
        observed["receipt"] = receipt
        observed.update(options)
        return receipt

    monkeypatch.setattr(MODULE, "recover_deserialization_failure", fake_recovery)
    receipt = tmp_path / "receipt.json"
    assert (
        MODULE.main(
            [
                "--receipt",
                str(receipt),
                "--failed-app-id",
                "ap-0123456789abcdef",
                "--max-dollars",
                "1.27",
                "--workspace-budget",
                "1.50",
                "--workspace-usage",
                "0.01",
                "--preflight-only",
            ]
        )
        == 0
    )
    assert observed == {
        "receipt": receipt,
        "failed_app_id": "ap-0123456789abcdef",
        "max_dollars": 1.27,
        "workspace_budget": 1.5,
        "workspace_usage": 0.01,
        "preflight_only": True,
    }


def test_cli_routes_preflight_only_to_mount_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def fake_recovery(receipt: Path, **options: object) -> Path:
        observed["receipt"] = receipt
        observed.update(options)
        return receipt

    monkeypatch.setattr(MODULE, "recover_mount_failure", fake_recovery)
    receipt = tmp_path / "receipt.json"
    assert (
        MODULE.main(
            [
                "--receipt",
                str(receipt),
                "--mount-failure-app-id",
                "ap-fedcba9876543210",
                "--max-dollars",
                "1.27",
                "--workspace-budget",
                "1.50",
                "--workspace-usage",
                "0.02",
                "--preflight-only",
            ]
        )
        == 0
    )
    assert observed == {
        "receipt": receipt,
        "failed_app_id": "ap-fedcba9876543210",
        "max_dollars": 1.27,
        "workspace_budget": 1.5,
        "workspace_usage": 0.02,
        "preflight_only": True,
    }


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


def _mount_source_receipt(bundle: Path) -> dict[str, Any]:
    return {
        "receipt_version": 1,
        "upload_id": UPLOAD_ID,
        "volume_name": "sion-prepared-bundles",
        "volume_version": 1,
        "app_name": "sion-prepared-bundle-finalizer",
        "function_name": "finalize_prepared_bundle",
        "runtime_contract_sha256": "d" * 64,
        "local_bundle_path": str(bundle.resolve()),
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "bundle_size": bundle.stat().st_size,
        "verification": {
            "file_count": 3,
            "total_bytes": 12,
            "git_commit": COMMIT,
            "git_tree": "e" * 40,
        },
        "remote_incoming_path": f"/incoming/{UPLOAD_ID}",
        "remote_final_path": f"/bundles/sha256/bb/{BUNDLE_HASH}",
        "remote_operation_path": f"/operations/{UPLOAD_ID}",
        "created_at_utc": "2026-09-02T10:00:00+00:00",
        "authorization_compute_charge_usd": 1.0,
        "max_dollars": 1.27,
        "workspace_budget_usd": 1.5,
        "workspace_usage_before_submit_usd": 0.01,
        "workspace_budget_headroom_usd": 1.49,
        "budget_observations": [],
        "upload_state": "uploaded",
        "upload_error": None,
        "finalizer_state": "submitted",
        "function_call_id": CALL_ID,
        "finalizer_error": None,
        "submission_claim_id": "claim-0123456789abcdef0123456789abcdef",
        "remote_submission_claim_path": f"/submission-claims/{UPLOAD_ID}.json",
        "submission_claim_state": "created",
        "submission_claim_error": None,
    }


class _FakeMountVolume:
    def __init__(self, receipt: dict[str, Any], *, operation_exists: bool = False) -> None:
        self.receipt = receipt
        self.operation_exists = operation_exists
        self.remote_json: dict[str, object] = {
            str(receipt["remote_submission_claim_path"]): {"claim": "exact"}
        }
        self.uploads: list[tuple[str, object]] = []

    def iterdir(self, path: str, *, recursive: bool) -> Iterator[object]:
        assert recursive is False
        if path == self.receipt["remote_operation_path"]:
            if self.operation_exists:
                return iter(())
            raise FileNotFoundError(path)
        if path == self.receipt["remote_incoming_path"]:
            entry = SimpleNamespace(
                path=f"{path}/bundle.zip",
                size=self.receipt["bundle_size"],
                type=SimpleNamespace(name="FILE"),
            )
            return iter((entry,))
        raise FileNotFoundError(path)

    @contextmanager
    def batch_upload(self, *, force: bool) -> Iterator[Any]:
        assert force is False
        outer = self

        class Batch:
            def put_file(self, content: Any, path: str) -> None:
                value = json.loads(content.read().decode("utf-8"))
                outer.uploads.append((path, value))
                outer.remote_json[path] = value

        yield Batch()


class _FakeMountRuntime(_FakeLegacy):
    FINALIZER_EPHEMERAL_DISK_MIB: int | None = None

    def __init__(self, runtime_root: Path, volume: _FakeMountVolume) -> None:
        super().__init__(runtime_root, volume.remote_json)
        self.volume = volume
        outer = self

        class Volume:
            @staticmethod
            def from_name(name: str, *, create_if_missing: bool, version: int) -> object:
                outer.events.append(("volume", name, create_if_missing, version))
                return outer.volume

        self._modal = SimpleNamespace(Volume=Volume)

    def _receipt_path(self, root: Path, upload_id: str) -> Path:
        return root / upload_id / "receipt.json"

    @staticmethod
    def _new_submission_claim_id() -> str:
        return "claim-fedcba9876543210fedcba9876543210"

    def _read_volume_json(self, _volume: object, path: str) -> object | None:
        self.events.append(("read", path))
        return self.volume.remote_json.get(path)

    def _validated_submission_claim(self, value: object, _receipt: object) -> object | None:
        return value if value == {"claim": "exact"} else None

    def _submit_finalizer(
        self,
        _modal: object,
        _volume: object,
        receipt_path: Path,
        receipt: dict[str, Any],
    ) -> None:
        self.events.append("submitted-without-upload")
        receipt["finalizer_state"] = "submitted"
        receipt["function_call_id"] = REPLACEMENT_CALL_ID
        self._write_json_atomic(receipt_path, receipt)


def _write_mount_failure_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    bundle = tmp_path / "prepared.zip"
    bundle.write_bytes(b"prepared archive")
    receipt = _mount_source_receipt(bundle)
    receipt_path = tmp_path / "receipts" / UPLOAD_ID / "receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    expected_message = (
        f"durable Modal bundle finalizer {UPLOAD_ID} failed; inspect its Volume journal"
    )
    status = {
        "receipt": receipt,
        "recovered_state": "failed",
        "function_call_state": "failed",
        "function_call_id_source": "receipt",
        "observed_function_call_id": CALL_ID,
        "function_call_result": None,
        "function_call_error": {"error_type": "RuntimeError", "message": expected_message},
        "remote_status": None,
        "remote_result": None,
        "remote_failure": None,
    }
    (receipt_path.parent / "status-latest.json").write_text(json.dumps(status), encoding="utf-8")
    return receipt_path, receipt


def _patch_mount_external_proofs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "_fetch_mount_failure_logs", lambda _app: "exact logs")
    monkeypatch.setattr(MODULE, "_assert_app_stopped", lambda _app: None)
    monkeypatch.setattr(MODULE, "_assert_terminal_failed_call", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MODULE, "_runtime_builder_sha256", lambda: "f" * 64)


def test_mount_recovery_binds_identity_and_uploads_only_its_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_mount_external_proofs(monkeypatch)
    receipt_path, source = _write_mount_failure_fixture(tmp_path)
    volume = _FakeMountVolume(source)
    runtime = _FakeMountRuntime(tmp_path / "runtime", volume)

    recovered_path = MODULE._recover_mount_with_committed_runtime(
        runtime,
        receipt_path,
        failed_app_id="ap-0123456789abcdef",
        replacement_commit=COMMIT,
        max_dollars=1.27,
        workspace_budget=1.5,
        workspace_usage=0.01,
    )

    recovered = runtime._read_receipt(recovered_path)
    assert recovered["function_call_id"] == REPLACEMENT_CALL_ID
    assert recovered["finalizer_state"] == "submitted"
    assert recovered["runtime_contract_sha256"] == RUNTIME_HASH
    assert MODULE._receipt_identity_projection(recovered) == MODULE._receipt_identity_projection(
        source
    )
    assert runtime.events.count("submitted-without-upload") == 1
    assert len(volume.uploads) == 1
    claim_path, claim = volume.uploads[0]
    assert claim_path == f"/recovery-claims/{UPLOAD_ID}/mount-symlink-{CALL_ID}.json"
    assert claim["schema"] == MODULE.MOUNT_RECOVERY_CLAIM_SCHEMA
    assert claim["source_receipt_identity"] == MODULE._receipt_identity_projection(source)
    assert "bundle.zip" not in claim_path


def test_mount_recovery_rejects_an_empty_operation_directory_before_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_mount_external_proofs(monkeypatch)
    receipt_path, source = _write_mount_failure_fixture(tmp_path)
    volume = _FakeMountVolume(source, operation_exists=True)
    runtime = _FakeMountRuntime(tmp_path / "runtime", volume)

    with pytest.raises(MODULE.LegacyRecoveryError, match="operation directory"):
        MODULE._recover_mount_with_committed_runtime(
            runtime,
            receipt_path,
            failed_app_id="ap-0123456789abcdef",
            replacement_commit=COMMIT,
            max_dollars=1.27,
            workspace_budget=1.5,
            workspace_usage=0.01,
        )
    assert volume.uploads == []
    assert "submitted-without-upload" not in runtime.events


def test_remote_path_absence_accepts_only_filesystem_or_modal_not_found() -> None:
    class ModalNotFound(Exception):
        pass

    ModalNotFound.__module__ = "modal.exception"
    ModalNotFound.__name__ = "NotFoundError"

    class MissingVolume:
        @staticmethod
        def iterdir(_path: str, *, recursive: bool) -> Iterator[object]:
            assert recursive is False
            raise ModalNotFound("missing")

    MODULE._assert_remote_path_absent(MissingVolume(), "/operations/missing", "operation")

    class BrokenVolume:
        @staticmethod
        def iterdir(_path: str, *, recursive: bool) -> Iterator[object]:
            assert recursive is False
            raise RuntimeError("transport failed")

    with pytest.raises(RuntimeError, match="transport failed"):
        MODULE._assert_remote_path_absent(BrokenVolume(), "/operations/missing", "operation")


def test_mount_recovery_rejects_a_redirected_existing_receipt_before_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_mount_external_proofs(monkeypatch)
    receipt_path, source = _write_mount_failure_fixture(tmp_path)
    volume = _FakeMountVolume(source)
    runtime = _FakeMountRuntime(tmp_path / "runtime", volume)
    recovery_path = receipt_path.parent / "mount-recovery" / UPLOAD_ID / "receipt.json"
    redirected = dict(source)
    redirected["runtime_contract_sha256"] = RUNTIME_HASH
    redirected["volume_name"] = "attacker-volume"
    redirected["finalizer_state"] = "not-submitted"
    redirected["function_call_id"] = None
    recovery_path.parent.mkdir(parents=True)
    recovery_path.write_text(json.dumps(redirected), encoding="utf-8")

    with pytest.raises(MODULE.LegacyRecoveryError, match="exact authorized replacement"):
        MODULE._recover_mount_with_committed_runtime(
            runtime,
            receipt_path,
            failed_app_id="ap-0123456789abcdef",
            replacement_commit=COMMIT,
            max_dollars=1.27,
            workspace_budget=1.5,
            workspace_usage=0.01,
        )
    assert volume.uploads == []
    assert "submitted-without-upload" not in runtime.events
