from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Iterator, cast
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "modal_stage_gpu_bundle.py"
SPEC = importlib.util.spec_from_file_location("modal_stage_gpu_bundle_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

UPLOAD_ID = "bundle-20260901t120000z-0123456789abcdef"
SECOND_UPLOAD_ID = "bundle-20260901t120001z-fedcba9876543210"
CALL_ID = "fc-0123456789abcdef"
OTHER_CALL_ID = "fc-fedcba9876543210"
SUBMISSION_CLAIM_ID = "claim-" + "c" * 32
VOLUME_NAME = "sion-prepared-bundles"
GIT_COMMIT = "a" * 40
GIT_TREE = "b" * 40
RUNTIME_CONTRACT = "e" * 64
MAX_DOLLARS = 1.30
WORKSPACE_BUDGET = 2.00
WORKSPACE_USAGE = 0.50


@pytest.fixture(autouse=True)
def bind_test_runtime_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    def return_expected_contract(expected: str) -> str:
        return expected

    monkeypatch.setattr(
        MODULE,
        "_verify_remote_runtime_contract",
        return_expected_contract,
    )


@dataclass(frozen=True)
class _Verification:
    file_count: int = 3
    total_bytes: int = 12
    git_commit: str = GIT_COMMIT
    git_tree: str = GIT_TREE


VERIFICATION = {
    "file_count": 3,
    "total_bytes": 12,
    "git_commit": GIT_COMMIT,
    "git_tree": GIT_TREE,
}


class _FakePackage:
    ARCHIVE_ROOT = "sion_translate"
    MANIFEST_NAME = "PACKAGE_MANIFEST.json"
    TOKENIZER_ROOT_PATH = "artifacts/tokenizer"
    TRANSLATION_DATASET_ROOT_PATH = "artifacts/dataset"
    FOUNDATION_DATASET_ROOT_PATH = "artifacts/foundation_dataset"

    def __init__(self) -> None:
        self.archive_calls = 0
        self.tree_calls = 0

    @staticmethod
    def _validated_relative_path(path: str) -> object:
        return MODULE.PACKAGE._validated_relative_path(path)

    def verify_archive(self, _path: Path) -> _Verification:
        self.archive_calls += 1
        return _Verification()

    def verify_tree(self, path: Path) -> _Verification:
        assert path.is_dir()
        self.tree_calls += 1
        return _Verification()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _make_prepared_archive(
    path: Path,
    *,
    foundation_enabled: bool = True,
    raw_origin: bool = False,
) -> tuple[int, str]:
    records: list[dict[str, object]] = [
        {
            "path": "artifacts/tokenizer/sion.model",
            "origin": "tokenizer",
            "size": 3,
            "sha256": "0" * 64,
            "mode": "100644",
        },
        {
            "path": "artifacts/dataset/manifest.json",
            "origin": "dataset",
            "size": 2,
            "sha256": "1" * 64,
            "mode": "100644",
        },
    ]
    payloads = {
        "artifacts/tokenizer/sion.model": b"tok",
        "artifacts/dataset/manifest.json": b"{}",
    }
    if foundation_enabled:
        records.append(
            {
                "path": "artifacts/foundation_dataset/manifest.json",
                "origin": "foundation-dataset",
                "size": 2,
                "sha256": "2" * 64,
                "mode": "100644",
            }
        )
        payloads["artifacts/foundation_dataset/manifest.json"] = b"{}"
    if raw_origin:
        records.append(
            {
                "path": "data/raw.jsonl",
                "origin": "data-jsonl",
                "size": 3,
                "sha256": "3" * 64,
                "mode": "100644",
            }
        )
        payloads["data/raw.jsonl"] = b"raw"
    manifest = {
        "files": records,
        "training_contract": {
            "raw_parallel_data_included": raw_origin,
            "foundation_enabled": foundation_enabled,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", allowZip64=True) as archive:
        for relative, payload in payloads.items():
            archive.writestr(_zip_info(f"sion_translate/{relative}"), payload)
        archive.writestr(
            _zip_info("sion_translate/PACKAGE_MANIFEST.json"),
            json.dumps(manifest).encode("utf-8"),
        )
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _directory_symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"directory symlinks are unavailable for this test: {error}")


class _FakeVolume:
    def __init__(self, mount_root: Path | None = None) -> None:
        self.mount_root = mount_root
        self.reload_count = 0
        self.commit_snapshots: list[dict[str, object]] = []

    def reload(self) -> None:
        self.reload_count += 1

    def commit(self) -> None:
        snapshot: dict[str, object] = {}
        if self.mount_root is not None:
            for path in self.mount_root.rglob("*.json"):
                snapshot[path.relative_to(self.mount_root).as_posix()] = json.loads(
                    path.read_text(encoding="utf-8")
                )
            for ready in self.mount_root.rglob("READY"):
                snapshot[ready.relative_to(self.mount_root).as_posix()] = json.loads(
                    ready.read_text(encoding="utf-8")
                )
        self.commit_snapshots.append(snapshot)


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
        self.observed["run_enter_attempted"] = True
        if self.enter_error is not None:
            raise self.enter_error
        self.observed["run_entered"] = True
        return None

    def __exit__(self, *_args: object) -> bool:
        self.observed["run_exited"] = True
        if self.exit_error is not None:
            raise self.exit_error
        return False


class _BatchContext(AbstractContextManager[Any]):
    def __init__(
        self,
        receipt_path: Path,
        observed: dict[str, object],
        bundle_error: BaseException | None = None,
        claim_error: BaseException | None = None,
    ) -> None:
        self.receipt_path = receipt_path
        self.observed = observed
        self.bundle_error = bundle_error
        self.claim_error = claim_error
        self.remote_path: str | None = None

    def __enter__(self) -> Any:
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        assert receipt["upload_state"] in {"intent", "uploaded", "upload-unknown"}
        assert (self.receipt_path.parents[1] / MODULE.SUBMISSION_LOCK_NAME).is_dir()
        previous_count = self.observed.get("batch_enter_count", 0)
        assert isinstance(previous_count, int)
        self.observed["batch_enter_count"] = previous_count + 1
        return self

    def put_file(self, local: object, remote: str) -> None:
        self.remote_path = remote
        put_files = self.observed.setdefault("put_files", [])
        assert isinstance(put_files, list)
        cast(list[object], put_files).append((local, remote))

    def __exit__(self, *_args: object) -> bool:
        self.observed["batch_exited"] = True
        if self.remote_path is not None and self.remote_path.endswith("/bundle.zip"):
            if self.bundle_error is not None:
                raise self.bundle_error
        elif self.claim_error is not None:
            raise self.claim_error
        return False


def _configure_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    upload_error: BaseException | None = None,
    claim_error: BaseException | None = None,
    run_enter_error: BaseException | None = None,
    spawn_error: BaseException | None = None,
    remote_operation_status: object | None = None,
    remote_files: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    bundle = tmp_path / "prepared.zip"
    bundle.write_bytes(b"prepared bundle")
    size = bundle.stat().st_size
    sha256 = hashlib.sha256(bundle.read_bytes()).hexdigest()
    receipt_path = tmp_path / "receipts" / UPLOAD_ID / "receipt.json"
    observed: dict[str, object] = {}

    class FakeVolume:
        def batch_upload(self, *, force: bool) -> _BatchContext:
            observed["force"] = force
            return _BatchContext(
                receipt_path,
                observed,
                upload_error,
                claim_error,
            )

        def read_file(self, path: str) -> Iterator[bytes]:
            observed["read_file"] = path
            if remote_files is not None and path in remote_files:
                yield MODULE._json_bytes(remote_files[path])
                return
            if remote_operation_status is not None and path.endswith("/status.json"):
                yield MODULE._json_bytes(remote_operation_status)
                return
            raise FileNotFoundError(path)

    volume = FakeVolume()

    class FakeVolumeType:
        @staticmethod
        def from_name(
            name: str,
            *,
            create_if_missing: bool,
            version: int,
        ) -> FakeVolume:
            observed["volume"] = (name, create_if_missing, version)
            return volume

    fake_modal = SimpleNamespace(Volume=FakeVolumeType)

    class FakeApp:
        def run(self, *, detach: bool) -> _RunContext:
            observed["detach"] = detach
            return _RunContext(observed, enter_error=run_enter_error)

    class FakeFinalizer:
        def spawn(self, *arguments: object) -> object:
            observed["spawn"] = arguments
            if spawn_error is not None:
                raise spawn_error
            return SimpleNamespace(object_id=CALL_ID)

    def require_modal() -> object:
        return fake_modal

    def new_upload_id() -> str:
        return UPLOAD_ID

    def validate_local_bundle(
        _path: Path,
    ) -> tuple[Path, int, str, dict[str, object]]:
        return bundle.resolve(), size, sha256, cast(dict[str, object], VERIFICATION.copy())

    def build_finalizer_runtime(
        _modal: object,
        observed_volume: object,
    ) -> tuple[FakeApp, FakeFinalizer]:
        if observed_volume is not volume:
            pytest.fail("wrong Volume passed to finalizer builder")
        return FakeApp(), FakeFinalizer()

    monkeypatch.setattr(MODULE, "_require_modal", require_modal)
    monkeypatch.setattr(MODULE, "_new_upload_id", new_upload_id)
    monkeypatch.setattr(
        MODULE,
        "_validate_local_bundle",
        validate_local_bundle,
    )
    monkeypatch.setattr(
        MODULE,
        "_build_finalizer_runtime",
        build_finalizer_runtime,
    )
    return bundle, observed


def test_stage_writes_intent_before_single_upload_and_persists_call_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, observed = _configure_stage(monkeypatch, tmp_path)

    receipt_path = MODULE.stage(
        bundle,
        VOLUME_NAME,
        tmp_path / "receipts",
        max_dollars=MAX_DOLLARS,
        workspace_budget=WORKSPACE_BUDGET,
        workspace_usage=WORKSPACE_USAGE,
    )

    receipt = MODULE._read_receipt(receipt_path)
    assert observed["batch_enter_count"] == 2
    assert observed["force"] is False
    assert observed["volume"] == (VOLUME_NAME, True, 1)
    assert observed["detach"] is True
    assert observed["spawn"] == (
        UPLOAD_ID,
        receipt["bundle_sha256"],
        receipt["bundle_size"],
        receipt["runtime_contract_sha256"],
    )
    assert receipt["upload_state"] == "uploaded"
    assert receipt["finalizer_state"] == "submitted"
    assert receipt["function_call_id"] == CALL_ID
    assert receipt["submission_claim_state"] == "created"
    assert MODULE.SUBMISSION_CLAIM_ID_PATTERN.fullmatch(receipt["submission_claim_id"])
    put_files = cast(list[tuple[object, str]], observed["put_files"])
    assert [remote for _local, remote in put_files] == [
        receipt["remote_incoming_path"] + "/bundle.zip",
        receipt["remote_submission_claim_path"],
    ]
    assert not (tmp_path / "receipts" / MODULE.SUBMISSION_LOCK_NAME).exists()


def test_ambiguous_upload_is_recorded_once_without_finalizer_or_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, observed = _configure_stage(
        monkeypatch,
        tmp_path,
        upload_error=ConnectionError("response was lost"),
    )

    with pytest.raises(ConnectionError, match="response was lost"):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            tmp_path / "receipts",
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )

    receipt = MODULE._read_receipt(tmp_path / "receipts" / UPLOAD_ID / "receipt.json")
    assert observed["batch_enter_count"] == 1
    assert "spawn" not in observed
    assert receipt["upload_state"] == "upload-unknown"
    assert receipt["upload_error"]["error_type"] == "ConnectionError"
    assert receipt["finalizer_state"] == "not-submitted"
    assert receipt["submission_claim_state"] == "not-created"


def test_ambiguous_submission_claim_never_spawns_a_finalizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, observed = _configure_stage(
        monkeypatch,
        tmp_path,
        claim_error=ConnectionError("claim response was lost"),
    )

    with pytest.raises(ConnectionError, match="claim response was lost"):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            tmp_path / "receipts",
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )

    receipt = MODULE._read_receipt(tmp_path / "receipts" / UPLOAD_ID / "receipt.json")
    assert observed["batch_enter_count"] == 2
    assert "spawn" not in observed
    assert receipt["upload_state"] == "uploaded"
    assert receipt["submission_claim_state"] == "creation-unknown"
    assert receipt["submission_claim_error"]["error_type"] == "ConnectionError"
    assert receipt["finalizer_state"] == "not-submitted"


def test_resume_reconciles_the_same_durable_claim_before_spawning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remote_files: dict[str, object] = {}
    bundle, observed = _configure_stage(
        monkeypatch,
        tmp_path,
        claim_error=ConnectionError("claim response was lost"),
        remote_files=remote_files,
    )
    receipt_root = tmp_path / "receipts"
    with pytest.raises(ConnectionError):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            receipt_root,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )
    receipt_path = receipt_root / UPLOAD_ID / "receipt.json"
    receipt = MODULE._read_receipt(receipt_path)
    claim_path = cast(str, receipt["remote_submission_claim_path"])
    remote_files[claim_path] = MODULE._submission_claim_payload(receipt)

    MODULE.resume_finalizer(
        receipt_path,
        max_dollars=MAX_DOLLARS,
        workspace_budget=WORKSPACE_BUDGET,
        workspace_usage=0.60,
    )

    recovered = MODULE._read_receipt(receipt_path)
    assert observed["batch_enter_count"] == 2
    assert recovered["submission_claim_state"] == "created"
    assert recovered["submission_claim_id"] == receipt["submission_claim_id"]
    assert recovered["finalizer_state"] == "submitted"
    assert recovered["function_call_id"] == CALL_ID


def test_resume_keeps_an_ambiguous_missing_claim_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, observed = _configure_stage(
        monkeypatch,
        tmp_path,
        claim_error=ConnectionError("claim response was lost"),
    )
    receipt_root = tmp_path / "receipts"
    with pytest.raises(ConnectionError):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            receipt_root,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )
    receipt_path = receipt_root / UPLOAD_ID / "receipt.json"

    with pytest.raises(MODULE.BundleStageError, match="not durably visible"):
        MODULE.resume_finalizer(
            receipt_path,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=0.60,
        )

    assert "spawn" not in observed
    assert MODULE._read_receipt(receipt_path)["submission_claim_state"] == ("creation-unknown")


def test_resume_never_spawns_for_a_conflicting_remote_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remote_files: dict[str, object] = {}
    bundle, observed = _configure_stage(
        monkeypatch,
        tmp_path,
        claim_error=ConnectionError("claim response was lost"),
        remote_files=remote_files,
    )
    receipt_root = tmp_path / "receipts"
    with pytest.raises(ConnectionError):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            receipt_root,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )
    receipt_path = receipt_root / UPLOAD_ID / "receipt.json"
    receipt = MODULE._read_receipt(receipt_path)
    claim_path = cast(str, receipt["remote_submission_claim_path"])
    conflicting = MODULE._submission_claim_payload(receipt)
    conflicting["submission_claim_id"] = "claim-" + "d" * 32
    remote_files[claim_path] = conflicting

    with pytest.raises(MODULE.BundleStageError, match="claim identity is invalid"):
        MODULE.resume_finalizer(
            receipt_path,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=0.60,
        )

    assert "spawn" not in observed


def test_resume_finalizer_reuses_an_ambiguous_upload_without_uploading_again(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, observed = _configure_stage(
        monkeypatch,
        tmp_path,
        upload_error=ConnectionError("upload response was lost"),
    )
    receipt_root = tmp_path / "receipts"
    with pytest.raises(ConnectionError, match="upload response was lost"):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            receipt_root,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )
    receipt_path = receipt_root / UPLOAD_ID / "receipt.json"

    recovered_path = MODULE.resume_finalizer(
        receipt_path,
        max_dollars=MAX_DOLLARS,
        workspace_budget=WORKSPACE_BUDGET,
        workspace_usage=0.60,
    )

    assert recovered_path == receipt_path.resolve()
    assert observed["batch_enter_count"] == 2
    assert observed["spawn"] == (
        UPLOAD_ID,
        MODULE._read_receipt(receipt_path)["bundle_sha256"],
        bundle.stat().st_size,
        MODULE._read_receipt(receipt_path)["runtime_contract_sha256"],
    )
    receipt = MODULE._read_receipt(receipt_path)
    assert receipt["upload_state"] == "upload-unknown"
    assert receipt["finalizer_state"] == "submitted"
    assert receipt["function_call_id"] == CALL_ID
    assert receipt["workspace_usage_before_submit_usd"] == 0.60
    assert [entry["workspace_usage_usd"] for entry in receipt["budget_observations"]] == [
        WORKSPACE_USAGE,
        0.60,
    ]
    assert not (receipt_root / MODULE.SUBMISSION_LOCK_NAME).exists()


def test_resume_finalizer_refuses_an_existing_remote_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remote_status = {"state": "running", "function_call_id": CALL_ID}
    bundle, observed = _configure_stage(
        monkeypatch,
        tmp_path,
        upload_error=ConnectionError("upload response was lost"),
        remote_operation_status=remote_status,
    )
    receipt_root = tmp_path / "receipts"
    with pytest.raises(ConnectionError):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            receipt_root,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )
    receipt_path = receipt_root / UPLOAD_ID / "receipt.json"

    with pytest.raises(MODULE.BundleStageError, match="already exists"):
        MODULE.resume_finalizer(
            receipt_path,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )

    assert "spawn" not in observed
    receipt = MODULE._read_receipt(receipt_path)
    assert receipt["finalizer_state"] == "not-submitted"
    assert receipt["finalizer_error"]["error_type"] == "BundleStageError"


@pytest.mark.parametrize("operation_file", ("result", "failure"))
def test_resume_finalizer_checks_every_remote_terminal_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation_file: str,
) -> None:
    remote_files: dict[str, object] = {}
    bundle, observed = _configure_stage(
        monkeypatch,
        tmp_path,
        upload_error=ConnectionError("upload response was lost"),
        remote_files=remote_files,
    )
    receipt_root = tmp_path / "receipts"
    with pytest.raises(ConnectionError):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            receipt_root,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )
    receipt_path = receipt_root / UPLOAD_ID / "receipt.json"
    receipt = MODULE._read_receipt(receipt_path)
    operation = cast(str, receipt["remote_operation_path"])
    remote_files[f"{operation}/{operation_file}.json"] = {"unexpected": True}

    with pytest.raises(MODULE.BundleStageError, match="already exists"):
        MODULE.resume_finalizer(
            receipt_path,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )

    assert "spawn" not in observed


def test_resume_finalizer_never_recreates_a_missing_volume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, _observed = _configure_stage(
        monkeypatch,
        tmp_path,
        upload_error=ConnectionError("upload response was lost"),
    )
    receipt_root = tmp_path / "receipts"
    with pytest.raises(ConnectionError):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            receipt_root,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )
    receipt_path = receipt_root / UPLOAD_ID / "receipt.json"
    observed_lookup: list[tuple[str, bool, int]] = []

    class MissingVolumeType:
        @staticmethod
        def from_name(
            name: str,
            *,
            create_if_missing: bool,
            version: int,
        ) -> object:
            observed_lookup.append((name, create_if_missing, version))
            raise RuntimeError("injected missing Modal Volume")

    fake_modal = SimpleNamespace(Volume=MissingVolumeType)

    def require_modal() -> object:
        return fake_modal

    monkeypatch.setattr(MODULE, "_require_modal", require_modal)
    with pytest.raises(RuntimeError, match="missing Modal Volume"):
        MODULE.resume_finalizer(
            receipt_path,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )

    assert observed_lookup == [(VOLUME_NAME, False, MODULE.VOLUME_VERSION)]
    receipt = MODULE._read_receipt(receipt_path)
    assert receipt["finalizer_state"] == "not-submitted"
    assert receipt["submission_claim_state"] == "not-created"


def test_resume_finalizer_refuses_every_previously_submitted_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, observed = _configure_stage(monkeypatch, tmp_path)
    receipt_path = MODULE.stage(
        bundle,
        VOLUME_NAME,
        tmp_path / "receipts",
        max_dollars=MAX_DOLLARS,
        workspace_budget=WORKSPACE_BUDGET,
        workspace_usage=WORKSPACE_USAGE,
    )

    with pytest.raises(MODULE.BundleStageError, match="never submitted"):
        MODULE.resume_finalizer(
            receipt_path,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )

    assert observed["batch_enter_count"] == 2
    assert observed["spawn"]


def test_ambiguous_finalizer_submission_preserves_uploaded_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, observed = _configure_stage(
        monkeypatch,
        tmp_path,
        spawn_error=ConnectionError("spawn response was lost"),
    )

    with pytest.raises(ConnectionError, match="spawn response was lost"):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            tmp_path / "receipts",
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )

    receipt = MODULE._read_receipt(tmp_path / "receipts" / UPLOAD_ID / "receipt.json")
    assert observed["batch_enter_count"] == 2
    assert receipt["upload_state"] == "uploaded"
    assert receipt["finalizer_state"] == "submission-unknown"
    assert receipt["function_call_id"] is None


def test_function_creation_failure_is_recorded_before_any_submission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, observed = _configure_stage(
        monkeypatch,
        tmp_path,
        run_enter_error=RuntimeError("function definition was rejected"),
    )

    with pytest.raises(RuntimeError, match="function definition was rejected"):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            tmp_path / "receipts",
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )

    receipt = MODULE._read_receipt(tmp_path / "receipts" / UPLOAD_ID / "receipt.json")
    assert observed["run_enter_attempted"] is True
    assert "run_entered" not in observed
    assert "spawn" not in observed
    assert receipt["upload_state"] == "uploaded"
    assert receipt["submission_claim_state"] == "created"
    assert receipt["finalizer_state"] == "not-submitted"
    assert receipt["function_call_id"] is None
    assert receipt["finalizer_error"]["error_type"] == "RuntimeError"


def test_existing_submission_lock_blocks_before_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle, observed = _configure_stage(monkeypatch, tmp_path)
    receipt_root = tmp_path / "receipts"
    lock_path = receipt_root / MODULE.SUBMISSION_LOCK_NAME
    lock_path.mkdir(parents=True)

    with pytest.raises(MODULE.BundleStageError, match="stage may be active"):
        MODULE.stage(
            bundle,
            VOLUME_NAME,
            receipt_root,
            max_dollars=MAX_DOLLARS,
            workspace_budget=WORKSPACE_BUDGET,
            workspace_usage=WORKSPACE_USAGE,
        )

    assert "batch_enter_count" not in observed
    assert not (receipt_root / UPLOAD_ID).exists()
    assert lock_path.is_dir()


def test_recover_lock_refuses_the_live_owner_process(tmp_path: Path) -> None:
    receipt_root = tmp_path / "receipts"

    with MODULE._exclusive_submission(receipt_root):
        with pytest.raises(MODULE.BundleStageError, match="owner is still running"):
            MODULE.recover_submission_lock(receipt_root)
        assert (receipt_root / MODULE.SUBMISSION_LOCK_NAME).is_dir()

    assert not (receipt_root / MODULE.SUBMISSION_LOCK_NAME).exists()


def test_recover_lock_removes_only_a_proven_stale_process_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_root = tmp_path / "receipts"
    lock_path = receipt_root / MODULE.SUBMISSION_LOCK_NAME
    lock_path.mkdir(parents=True)
    MODULE._write_json_atomic(
        lock_path / "owner.json",
        {
            "lock_version": 2,
            "process_id": 424242,
            "process_instance_identity": "windows-filetime:123",
            "host_name": MODULE.socket.gethostname(),
            "acquired_at_utc": datetime.now(UTC).isoformat(),
        },
    )

    def absent_process(_process_id: int) -> tuple[str, str | None]:
        return "absent", None

    monkeypatch.setattr(MODULE, "_process_instance_state", absent_process)
    recovered_path = MODULE.recover_submission_lock(receipt_root)

    assert recovered_path == lock_path
    assert not lock_path.exists()
    assert not list(receipt_root.glob(".submission-lock.recovering-*"))


def test_recover_lock_refuses_unknown_or_cross_host_ownership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_root = tmp_path / "receipts"
    lock_path = receipt_root / MODULE.SUBMISSION_LOCK_NAME
    lock_path.mkdir(parents=True)
    owner = {
        "lock_version": 2,
        "process_id": 424242,
        "process_instance_identity": "windows-filetime:123",
        "host_name": "different-host",
        "acquired_at_utc": datetime.now(UTC).isoformat(),
    }
    MODULE._write_json_atomic(lock_path / "owner.json", owner)

    with pytest.raises(MODULE.BundleStageError, match="another host"):
        MODULE.recover_submission_lock(receipt_root)

    owner["host_name"] = MODULE.socket.gethostname()
    MODULE._write_json_atomic(lock_path / "owner.json", owner)

    def unknown_process(_process_id: int) -> tuple[str, str | None]:
        return "unknown", None

    monkeypatch.setattr(MODULE, "_process_instance_state", unknown_process)
    with pytest.raises(MODULE.BundleStageError, match="cannot prove"):
        MODULE.recover_submission_lock(receipt_root)
    assert lock_path.is_dir()


@pytest.mark.parametrize(
    ("max_dollars", "workspace_budget", "workspace_usage", "message"),
    (
        (0.10, WORKSPACE_BUDGET, WORKSPACE_USAGE, "two-attempt CPU"),
        (MAX_DOLLARS, 1.0, 0.0, "does not cover"),
        (MAX_DOLLARS, 8.0, 0.0, r"exceeds the \$5"),
        (float("nan"), WORKSPACE_BUDGET, WORKSPACE_USAGE, "must be finite"),
    ),
)
def test_finalizer_budget_guard_rejects_unsafe_authorization(
    max_dollars: float,
    workspace_budget: float,
    workspace_usage: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MODULE._validate_workspace_budget_guard(
            max_dollars,
            workspace_budget,
            workspace_usage,
        )


@pytest.mark.parametrize(
    "remote_path",
    (
        "relative",
        "/",
        "//outside",
        "/a//b",
        "/a/./b",
        "/a/../b",
        "/a\\b",
    ),
)
def test_remote_volume_paths_reject_noncanonical_input(remote_path: str) -> None:
    with pytest.raises(MODULE.BundleStageError, match="path"):
        MODULE._relative_remote_path(remote_path)


def test_remote_volume_paths_preserve_one_canonical_relative_path(tmp_path: Path) -> None:
    remote_path = f"/incoming/{UPLOAD_ID}/bundle.zip"

    relative = MODULE._relative_remote_path(remote_path)

    assert relative.as_posix() == f"incoming/{UPLOAD_ID}/bundle.zip"
    assert MODULE._mounted_path(tmp_path, remote_path) == (
        tmp_path / "incoming" / UPLOAD_ID / "bundle.zip"
    )


@pytest.mark.parametrize("mismatch", ("size", "digest"))
def test_finalizer_rejects_uploaded_size_or_digest_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    archive = incoming / "bundle.zip"
    size, sha256 = _make_prepared_archive(archive)
    expected_size = size + 1 if mismatch == "size" else size
    expected_sha256 = "f" * 64 if mismatch == "digest" else sha256
    volume = _FakeVolume(mount)
    package = _FakePackage()

    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            volume,
            mount,
            UPLOAD_ID,
            expected_sha256,
            expected_size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=package,
        )
    assert isinstance(captured.value.__cause__, MODULE.BundleStageError)
    assert "size or SHA-256" in str(captured.value.__cause__)

    assert not (incoming / "READY").exists()
    operation = mount / "operations" / UPLOAD_ID
    assert not (operation / "result.json").exists()
    assert json.loads((operation / "status.json").read_text())["state"] == "failed"
    assert json.loads((operation / "failure.json").read_text())["error_type"] == (
        "BundleStageError"
    )


def test_finalizer_rejects_symlinked_operation_ancestor(tmp_path: Path) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    outside = tmp_path / "outside-operations"
    outside.mkdir()
    _directory_symlink_or_skip(mount / "operations", outside)
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")

    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            _FakeVolume(mount),
            mount,
            UPLOAD_ID,
            sha256,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=_FakePackage(),
        )

    assert isinstance(captured.value.__cause__, MODULE.BundleStageError)
    assert "non-symlink directory" in str(captured.value.__cause__)
    assert not list(outside.iterdir())


def test_finalizer_preserves_a_different_function_call_journal(tmp_path: Path) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")
    operation = mount / "operations" / UPLOAD_ID
    operation.mkdir(parents=True)
    status_path = operation / "status.json"
    result_path = operation / "result.json"
    failure_path = operation / "failure.json"
    MODULE._write_json_atomic(
        status_path,
        MODULE._operation_status(
            UPLOAD_ID,
            sha256,
            size,
            "failed",
            2,
            function_call_id=OTHER_CALL_ID,
            runtime_contract_sha256=RUNTIME_CONTRACT,
        ),
    )
    result_path.write_bytes(b"different-call-result")
    failure_path.write_bytes(b"different-call-failure")
    before = {path.name: path.read_bytes() for path in operation.iterdir()}
    package = _FakePackage()

    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            _FakeVolume(mount),
            mount,
            UPLOAD_ID,
            sha256,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=package,
        )

    assert isinstance(captured.value.__cause__, MODULE.BundleConflictError)
    assert "was preserved" in str(captured.value.__cause__)
    assert {path.name: path.read_bytes() for path in operation.iterdir()} == before
    assert package.archive_calls == 0
    assert not (incoming / "tree").exists()
    assert not (mount / "bundles" / "sha256" / sha256[:2] / sha256).exists()


def test_finalizer_rejects_symlinked_incoming_ancestor(tmp_path: Path) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    outside = tmp_path / "outside-incoming"
    outside.mkdir()
    _directory_symlink_or_skip(mount / "incoming", outside)
    archive = outside / UPLOAD_ID / "bundle.zip"
    size, sha256 = _make_prepared_archive(archive)

    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            _FakeVolume(mount),
            mount,
            UPLOAD_ID,
            sha256,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=_FakePackage(),
        )

    assert isinstance(captured.value.__cause__, MODULE.BundleStageError)
    assert "non-symlink directory" in str(captured.value.__cause__)
    assert archive.is_file()
    assert not (archive.parent / "tree").exists()
    assert not (archive.parent / "READY").exists()


def test_finalizer_rejects_symlinked_publication_ancestor(tmp_path: Path) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    outside = tmp_path / "outside-bundles"
    outside.mkdir()
    _directory_symlink_or_skip(mount / "bundles", outside)
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")

    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            _FakeVolume(mount),
            mount,
            UPLOAD_ID,
            sha256,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=_FakePackage(),
        )

    assert isinstance(captured.value.__cause__, MODULE.BundleStageError)
    assert "non-symlink directory" in str(captured.value.__cause__)
    assert not list(outside.iterdir())
    assert incoming.is_dir()
    assert not (incoming / "READY").exists()


def test_finalizer_rejects_a_broken_content_addressed_leaf_link(tmp_path: Path) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")
    final = mount / "bundles" / "sha256" / sha256[:2] / sha256
    final.parent.mkdir(parents=True)
    _directory_symlink_or_skip(final, tmp_path / "missing-outside-bundle")

    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            _FakeVolume(mount),
            mount,
            UPLOAD_ID,
            sha256,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=_FakePackage(),
        )

    assert isinstance(captured.value.__cause__, MODULE.BundleStageError)
    assert "non-symlink directory" in str(captured.value.__cause__)
    assert not (tmp_path / "missing-outside-bundle").exists()
    assert incoming.is_dir()
    assert not (incoming / "READY").exists()


def test_interrupted_extraction_never_writes_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")
    volume = _FakeVolume(mount)
    original_extract = MODULE._stream_extract_archive

    def interrupt(_archive: Path, tree: Path, **_kwargs: object) -> None:
        (tree / "partial.bin").write_bytes(b"partial")
        raise OSError("injected extraction interruption")

    monkeypatch.setattr(MODULE, "_stream_extract_archive", interrupt)
    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            volume,
            mount,
            UPLOAD_ID,
            sha256,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=_FakePackage(),
        )
    assert isinstance(captured.value.__cause__, OSError)
    assert "injected extraction interruption" in str(captured.value.__cause__)

    assert (incoming / "tree" / "partial.bin").is_file()
    assert not (incoming / "READY").exists()
    assert not (mount / "bundles" / "sha256" / sha256[:2] / sha256).exists()

    monkeypatch.setattr(MODULE, "_stream_extract_archive", original_extract)
    recovered = MODULE._finalize_bundle(
        volume,
        mount,
        UPLOAD_ID,
        sha256,
        size,
        CALL_ID,
        RUNTIME_CONTRACT,
        package_module=_FakePackage(),
    )
    assert recovered["reused"] is False
    final = MODULE._mounted_path(mount, str(recovered["final_path"]))
    assert (final / "READY").is_file()
    assert not incoming.exists()


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "sion_translate/../escape.txt",
        "sion_translate/dir/../../escape.txt",
        "outside/payload.txt",
        "sion_translate/dir\\payload.txt",
    ),
)
def test_stream_extraction_rejects_unsafe_paths(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, mode="w") as output:
        output.writestr(_zip_info(unsafe_name), b"escape")
    if "\\" in unsafe_name:
        normalized = unsafe_name.replace("\\", "/").encode("utf-8")
        hostile = unsafe_name.encode("utf-8")
        payload = archive.read_bytes()
        if payload.count(hostile) == 0:
            assert payload.count(normalized) == 2
            archive.write_bytes(payload.replace(normalized, hostile))
        else:
            assert payload.count(hostile) == 2
    extraction = tmp_path / "tree"
    extraction.mkdir()

    with pytest.raises(Exception, match="unsafe|unsupported"):
        MODULE._stream_extract_archive(
            archive,
            extraction,
            package_module=_FakePackage(),
        )

    assert not (tmp_path / "escape.txt").exists()


def test_volume_mount_symlink_is_pinned_to_its_directory_target(tmp_path: Path) -> None:
    target = tmp_path / "volume-target"
    target.mkdir()
    mount = tmp_path / "volume-mount"
    try:
        mount.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    assert MODULE._resolve_volume_mount(mount) == target.resolve()

    unsafe_target = tmp_path / "regular-file"
    unsafe_target.write_bytes(b"not a directory")
    unsafe_mount = tmp_path / "unsafe-mount"
    unsafe_mount.symlink_to(unsafe_target)
    with pytest.raises(Exception, match="directory"):
        MODULE._resolve_volume_mount(unsafe_mount)


def test_exact_existing_artifact_is_verified_and_reused(
    tmp_path: Path,
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")
    volume = _FakeVolume(mount)
    package = _FakePackage()
    first = MODULE._finalize_bundle(
        volume,
        mount,
        UPLOAD_ID,
        sha256,
        size,
        CALL_ID,
        RUNTIME_CONTRACT,
        package_module=package,
    )
    assert first["reused"] is False

    second = MODULE._finalize_bundle(
        volume,
        mount,
        SECOND_UPLOAD_ID,
        sha256,
        size,
        CALL_ID,
        RUNTIME_CONTRACT,
        package_module=package,
    )

    assert second["reused"] is True
    assert second["final_path"] == first["final_path"]
    second_operation = mount / "operations" / SECOND_UPLOAD_ID
    assert json.loads((second_operation / "result.json").read_text()) == second
    assert json.loads((second_operation / "status.json").read_text())["state"] == ("passed")


def test_conflicting_content_addressed_directory_is_never_overwritten(
    tmp_path: Path,
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")
    final = mount / "bundles" / "sha256" / sha256[:2] / sha256
    final.mkdir(parents=True)
    sentinel = final / "do-not-overwrite.txt"
    sentinel.write_text("conflict", encoding="utf-8")
    volume = _FakeVolume(mount)

    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            volume,
            mount,
            UPLOAD_ID,
            sha256,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=_FakePackage(),
        )
    assert isinstance(captured.value.__cause__, MODULE.BundleConflictError)
    assert "refusing to overwrite" in str(captured.value.__cause__)

    assert sentinel.read_text(encoding="utf-8") == "conflict"
    assert incoming.is_dir()
    assert not (incoming / "READY").exists()


def test_exact_destination_winning_publication_race_is_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")

    def publish_first(source: Path, destination: Path) -> None:
        shutil.copytree(source, destination)
        raise FileExistsError("injected exact publication race")

    monkeypatch.setattr(MODULE, "_rename_directory_no_replace", publish_first)
    result = MODULE._finalize_bundle(
        _FakeVolume(mount),
        mount,
        UPLOAD_ID,
        sha256,
        size,
        CALL_ID,
        RUNTIME_CONTRACT,
        package_module=_FakePackage(),
    )

    assert result["reused"] is True
    final = MODULE._mounted_path(mount, str(result["final_path"]))
    assert (final / "READY").is_file()
    assert not incoming.exists()


def test_conflicting_destination_winning_publication_race_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")
    sentinel_payload = "do not overwrite this race winner"

    def publish_conflict(_source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "sentinel.txt").write_text(sentinel_payload, encoding="utf-8")
        raise FileExistsError("injected conflicting publication race")

    monkeypatch.setattr(MODULE, "_rename_directory_no_replace", publish_conflict)
    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            _FakeVolume(mount),
            mount,
            UPLOAD_ID,
            sha256,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=_FakePackage(),
        )

    assert isinstance(captured.value.__cause__, MODULE.BundleConflictError)
    final = mount / "bundles" / "sha256" / sha256[:2] / sha256
    assert (final / "sentinel.txt").read_text(encoding="utf-8") == sentinel_payload
    assert incoming.is_dir()
    assert (incoming / "READY").is_file()


@pytest.mark.skipif(
    not MODULE.sys.platform.startswith("linux"),
    reason="renameat2(RENAME_NOREPLACE) is the Linux finalizer primitive",
)
def test_linux_no_replace_rename_preserves_an_empty_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source", encoding="utf-8")

    with pytest.raises(FileExistsError):
        MODULE._rename_directory_no_replace(source, destination)

    assert (source / "source.txt").read_text(encoding="utf-8") == "source"
    assert destination.is_dir()
    assert not list(destination.iterdir())


def test_success_commits_artifact_before_consistent_final_result(
    tmp_path: Path,
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")
    volume = _FakeVolume(mount)

    result = MODULE._finalize_bundle(
        volume,
        mount,
        UPLOAD_ID,
        sha256,
        size,
        CALL_ID,
        RUNTIME_CONTRACT,
        package_module=_FakePackage(),
    )

    operation_prefix = f"operations/{UPLOAD_ID}"
    assert volume.reload_count == 1
    assert len(volume.commit_snapshots) == 3
    running_status = cast(
        dict[str, object],
        volume.commit_snapshots[0][f"{operation_prefix}/status.json"],
    )
    assert running_status["state"] == "running"
    final_snapshot = volume.commit_snapshots[-1]
    passed_status = cast(
        dict[str, object],
        final_snapshot[f"{operation_prefix}/status.json"],
    )
    assert passed_status["state"] == "passed"
    assert final_snapshot[f"{operation_prefix}/result.json"] == result
    assert f"{operation_prefix}/failure.json" not in final_snapshot
    final = MODULE._mounted_path(mount, str(result["final_path"]))
    assert (final / "READY").is_file()
    assert not incoming.exists()


def test_pre_journal_reload_failure_uses_the_terminal_failure_envelope(
    tmp_path: Path,
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()

    class ReloadFailureVolume(_FakeVolume):
        def reload(self) -> None:
            raise OSError("injected Volume reload failure")

    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            ReloadFailureVolume(mount),
            mount,
            UPLOAD_ID,
            "6" * 64,
            10,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=_FakePackage(),
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert "injected Volume reload failure" in str(captured.value.__cause__)
    assert not (mount / "operations" / UPLOAD_ID).exists()


def test_running_journal_commit_failure_prevents_materialization(
    tmp_path: Path,
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")
    package = _FakePackage()

    class RunningCommitFailureVolume(_FakeVolume):
        def __init__(self, mount_root: Path) -> None:
            super().__init__(mount_root)
            self.commit_attempts = 0

        def commit(self) -> None:
            self.commit_attempts += 1
            if self.commit_attempts == 1:
                raise OSError("injected running-journal commit failure")
            super().commit()

    volume = RunningCommitFailureVolume(mount)
    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            volume,
            mount,
            UPLOAD_ID,
            sha256,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=package,
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert package.archive_calls == 0
    assert not (incoming / "tree").exists()
    operation = mount / "operations" / UPLOAD_ID
    assert json.loads((operation / "status.json").read_text())["state"] == "failed"
    assert (operation / "failure.json").is_file()


def test_terminal_result_commit_failure_cannot_return_false_success(
    tmp_path: Path,
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")

    class TerminalCommitFailureVolume(_FakeVolume):
        def __init__(self, mount_root: Path) -> None:
            super().__init__(mount_root)
            self.commit_attempts = 0

        def commit(self) -> None:
            self.commit_attempts += 1
            if self.commit_attempts == 3:
                raise OSError("injected terminal-result commit failure")
            super().commit()

    volume = TerminalCommitFailureVolume(mount)
    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            volume,
            mount,
            UPLOAD_ID,
            sha256,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=_FakePackage(),
        )

    assert isinstance(captured.value.__cause__, OSError)
    assert "terminal-result commit" in str(captured.value.__cause__)
    final = mount / "bundles" / "sha256" / sha256[:2] / sha256
    assert (final / "READY").is_file()
    operation = mount / "operations" / UPLOAD_ID
    assert not (operation / "result.json").exists()
    assert json.loads((operation / "status.json").read_text())["state"] == "failed"
    assert (operation / "failure.json").is_file()


def test_failure_journal_commit_error_preserves_the_original_cause(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, _sha256 = _make_prepared_archive(incoming / "bundle.zip")

    class FailureJournalCommitFailureVolume(_FakeVolume):
        def __init__(self, mount_root: Path) -> None:
            super().__init__(mount_root)
            self.commit_attempts = 0

        def commit(self) -> None:
            self.commit_attempts += 1
            if self.commit_attempts == 2:
                raise OSError("injected failure-journal commit failure")
            super().commit()

    volume = FailureJournalCommitFailureVolume(mount)
    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            volume,
            mount,
            UPLOAD_ID,
            "7" * 64,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=_FakePackage(),
        )

    assert isinstance(captured.value.__cause__, MODULE.BundleStageError)
    assert "size or SHA-256" in str(captured.value.__cause__)
    assert "failure-journal commit failure" in capsys.readouterr().err


def test_runtime_contract_mismatch_is_journaled_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()
    incoming = mount / "incoming" / UPLOAD_ID
    size, sha256 = _make_prepared_archive(incoming / "bundle.zip")
    package = _FakePackage()

    def reject_runtime_contract(_expected: str) -> str:
        raise MODULE.BundleStageError("injected runtime contract mismatch")

    monkeypatch.setattr(
        MODULE,
        "_verify_remote_runtime_contract",
        reject_runtime_contract,
    )
    with pytest.raises(RuntimeError, match="inspect its Volume journal") as captured:
        MODULE._finalize_bundle(
            _FakeVolume(mount),
            mount,
            UPLOAD_ID,
            sha256,
            size,
            CALL_ID,
            RUNTIME_CONTRACT,
            package_module=package,
        )

    assert isinstance(captured.value.__cause__, MODULE.BundleStageError)
    assert "runtime contract mismatch" in str(captured.value.__cause__)
    assert package.archive_calls == 0
    assert not (incoming / "tree").exists()
    operation = mount / "operations" / UPLOAD_ID
    assert json.loads((operation / "status.json").read_text())["state"] == "failed"
    failure = json.loads((operation / "failure.json").read_text())
    assert failure["error_type"] == "BundleStageError"


def _receipt(
    sha256: str,
    size: int,
    *,
    upload_state: str = "uploaded",
    finalizer_state: str = "submitted",
    call_id: str | None = CALL_ID,
) -> dict[str, object]:
    incoming, final, operation = MODULE._remote_paths(UPLOAD_ID, sha256)
    budget_observation = MODULE._budget_observation(
        MAX_DOLLARS,
        WORKSPACE_BUDGET,
        WORKSPACE_USAGE,
    )
    return {
        "receipt_version": MODULE.RECEIPT_VERSION,
        "upload_id": UPLOAD_ID,
        "volume_name": VOLUME_NAME,
        "volume_version": MODULE.VOLUME_VERSION,
        "app_name": MODULE.APP_NAME,
        "function_name": MODULE.FINALIZER_FUNCTION_NAME,
        "runtime_contract_sha256": RUNTIME_CONTRACT,
        "local_bundle_path": str((Path.cwd() / "prepared.zip").resolve()),
        "bundle_sha256": sha256,
        "bundle_size": size,
        "verification": VERIFICATION.copy(),
        "remote_incoming_path": incoming,
        "remote_final_path": final,
        "remote_operation_path": operation,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authorization_compute_charge_usd": MODULE.finalizer_authorization_compute_charge(),
        "max_dollars": MAX_DOLLARS,
        "workspace_budget_usd": WORKSPACE_BUDGET,
        "workspace_usage_before_submit_usd": WORKSPACE_USAGE,
        "workspace_budget_headroom_usd": WORKSPACE_BUDGET - WORKSPACE_USAGE,
        "budget_observations": [budget_observation],
        "upload_state": upload_state,
        "upload_error": (
            {"error_type": "ConnectionError", "message": "ambiguous"}
            if upload_state == "upload-unknown"
            else None
        ),
        "finalizer_state": finalizer_state,
        "function_call_id": call_id,
        "finalizer_error": (
            {"error_type": "ConnectionError", "message": "ambiguous"}
            if finalizer_state == "submission-unknown"
            else None
        ),
        "submission_claim_id": SUBMISSION_CLAIM_ID,
        "remote_submission_claim_path": MODULE._remote_submission_claim_path(UPLOAD_ID),
        "submission_claim_state": "created",
        "submission_claim_error": None,
    }


class _StatusVolume:
    def __init__(self, files: dict[str, object]):
        self.files = files

    def read_file(self, path: str) -> Iterator[bytes]:
        if path not in self.files:
            raise FileNotFoundError(path)
        yield MODULE._json_bytes(self.files[path])


def _configure_status(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, object],
    call_observation: object,
    *,
    output_expired_type: type[BaseException] | None = None,
    modal_timeout_type: type[BaseException] | None = None,
    function_timeout_type: type[BaseException] | None = None,
    internal_failure_type: type[BaseException] | None = None,
    from_id_error: BaseException | None = None,
) -> None:
    volume = _StatusVolume(files)

    class FakeVolumeType:
        @staticmethod
        def from_name(
            name: str,
            *,
            create_if_missing: bool,
            version: int,
        ) -> _StatusVolume:
            assert (name, create_if_missing, version) == (VOLUME_NAME, False, 1)
            return volume

    class FakeCall:
        def get(self, *, timeout: int) -> object:
            assert timeout == 0
            if isinstance(call_observation, BaseException):
                raise call_observation
            return call_observation

    class FakeFunctionCall:
        @staticmethod
        def from_id(call_id: str) -> FakeCall:
            assert call_id == CALL_ID
            if from_id_error is not None:
                raise from_id_error
            return FakeCall()

    fake_modal = SimpleNamespace(
        Volume=FakeVolumeType,
        FunctionCall=FakeFunctionCall,
        exception=SimpleNamespace(
            OutputExpiredError=output_expired_type,
            TimeoutError=modal_timeout_type,
            FunctionTimeoutError=function_timeout_type,
            InternalFailure=internal_failure_type,
        ),
    )
    monkeypatch.setattr(MODULE, "_require_modal", lambda: fake_modal)


def _durable_failure(receipt: dict[str, object]) -> dict[str, object]:
    return {
        "schema": MODULE.OPERATION_SCHEMA,
        "upload_id": receipt["upload_id"],
        "bundle_sha256": receipt["bundle_sha256"],
        "bundle_size": receipt["bundle_size"],
        "function_call_id": CALL_ID,
        "runtime_contract_sha256": receipt["runtime_contract_sha256"],
        "error_type": "BundleStageError",
        "message": "injected durable failure",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "traceback_tail": "",
    }


def test_status_distinguishes_pending_from_expired_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    size = 10
    sha256 = "c" * 64
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, _receipt(sha256, size))
    _configure_status(monkeypatch, {}, TimeoutError("not complete"))

    pending = MODULE.status(receipt_path)
    assert pending["function_call_state"] == "pending"
    assert pending["recovered_state"] == "pending"

    class OutputExpiredError(Exception):
        pass

    _configure_status(
        monkeypatch,
        {},
        OutputExpiredError("expired"),
        output_expired_type=OutputExpiredError,
    )
    expired = MODULE.status(receipt_path)
    assert expired["function_call_state"] == "output-expired"
    assert expired["recovered_state"] == "output-expired"


@pytest.mark.parametrize(
    ("claim_state", "expected_state"),
    (
        ("creating", "claim-creating"),
        ("creation-unknown", "claim-creation-unknown"),
        ("created", "claimed-without-call"),
    ),
)
def test_status_exposes_every_pre_spawn_claim_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    claim_state: str,
    expected_state: str,
) -> None:
    receipt = _receipt(
        "0" * 64,
        10,
        finalizer_state="not-submitted",
        call_id=None,
    )
    receipt["submission_claim_state"] = claim_state
    receipt["submission_claim_error"] = (
        {
            "error_type": "ConnectionError",
            "message": "ambiguous claim",
            "recorded_at_utc": datetime.now(UTC).isoformat(),
        }
        if claim_state == "creation-unknown"
        else None
    )
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, receipt)
    _configure_status(monkeypatch, {}, None)

    snapshot = MODULE.status(receipt_path)
    assert snapshot["recovered_state"] == expected_state
    with pytest.raises(MODULE.BundleStageError, match="no recovered terminal state"):
        MODULE._assert_no_unresolved_receipts(tmp_path)


def test_status_prefers_consistent_durable_result_after_call_output_expires(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    size = 10
    sha256 = "d" * 64
    receipt = _receipt(sha256, size)
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, receipt)
    operation = str(receipt["remote_operation_path"])
    result = MODULE._result_payload(
        UPLOAD_ID,
        sha256,
        size,
        str(receipt["remote_final_path"]),
        VERIFICATION,
        reused=False,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )
    durable_status = MODULE._operation_status(
        UPLOAD_ID,
        sha256,
        size,
        "passed",
        2,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )

    class OutputExpiredError(Exception):
        pass

    _configure_status(
        monkeypatch,
        {
            f"{operation}/status.json": durable_status,
            f"{operation}/result.json": result,
        },
        OutputExpiredError("expired"),
        output_expired_type=OutputExpiredError,
    )
    snapshot = MODULE.status(receipt_path)
    assert snapshot["function_call_state"] == "output-expired"
    assert snapshot["recovered_state"] == "passed"
    assert snapshot["remote_result"] == result


def test_status_discovers_ambiguous_finalizer_call_id_from_volume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    size = 10
    sha256 = "f" * 64
    receipt = _receipt(
        sha256,
        size,
        finalizer_state="submission-unknown",
        call_id=None,
    )
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, receipt)
    operation = str(receipt["remote_operation_path"])
    durable_status = MODULE._operation_status(
        UPLOAD_ID,
        sha256,
        size,
        "running",
        1,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )
    _configure_status(
        monkeypatch,
        {f"{operation}/status.json": durable_status},
        TimeoutError(),
    )

    snapshot = MODULE.status(receipt_path)
    assert snapshot["observed_function_call_id"] == CALL_ID
    assert snapshot["function_call_id_source"] == "volume-status"
    assert snapshot["function_call_state"] == "pending"
    assert snapshot["recovered_state"] == "running"


def test_status_keeps_terminal_journal_unresolved_while_call_is_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    size = 10
    sha256 = "1" * 64
    receipt = _receipt(sha256, size)
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, receipt)
    operation = str(receipt["remote_operation_path"])
    result = MODULE._result_payload(
        UPLOAD_ID,
        sha256,
        size,
        str(receipt["remote_final_path"]),
        VERIFICATION,
        reused=False,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )
    durable_status = MODULE._operation_status(
        UPLOAD_ID,
        sha256,
        size,
        "passed",
        2,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )
    _configure_status(
        monkeypatch,
        {
            f"{operation}/status.json": durable_status,
            f"{operation}/result.json": result,
        },
        TimeoutError(),
    )

    snapshot = MODULE.status(receipt_path)
    assert snapshot["recovered_state"] == "terminal-journal-pending-call"
    with pytest.raises(MODULE.BundleStageError, match="no recovered terminal state"):
        MODULE._assert_no_unresolved_receipts(tmp_path)


def test_status_recognizes_exact_finalizer_failure_without_journal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt = _receipt("2" * 64, 10)
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, receipt)
    error = RuntimeError(
        f"durable Modal bundle finalizer {UPLOAD_ID} failed; inspect its Volume journal"
    )
    _configure_status(monkeypatch, {}, error)

    snapshot = MODULE.status(receipt_path)
    assert snapshot["function_call_state"] == "failed"
    assert snapshot["recovered_state"] == "failed"
    MODULE._assert_no_unresolved_receipts(tmp_path)


def test_status_keeps_client_error_non_terminal_despite_volume_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    size = 10
    sha256 = "3" * 64
    receipt = _receipt(sha256, size)
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, receipt)
    operation = str(receipt["remote_operation_path"])
    durable_status = MODULE._operation_status(
        UPLOAD_ID,
        sha256,
        size,
        "failed",
        2,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )
    _configure_status(
        monkeypatch,
        {
            f"{operation}/status.json": durable_status,
            f"{operation}/failure.json": _durable_failure(receipt),
        },
        ConnectionError("temporary client failure"),
    )

    snapshot = MODULE.status(receipt_path)
    assert snapshot["function_call_state"] == "unavailable"
    assert snapshot["recovered_state"] == "status-unavailable"
    with pytest.raises(MODULE.BundleStageError, match="no recovered terminal state"):
        MODULE._assert_no_unresolved_receipts(tmp_path)


def test_status_keeps_function_call_lookup_error_non_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    size = 10
    sha256 = "8" * 64
    receipt = _receipt(sha256, size)
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, receipt)
    operation = str(receipt["remote_operation_path"])
    durable_status = MODULE._operation_status(
        UPLOAD_ID,
        sha256,
        size,
        "failed",
        2,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )
    _configure_status(
        monkeypatch,
        {
            f"{operation}/status.json": durable_status,
            f"{operation}/failure.json": _durable_failure(receipt),
        },
        None,
        from_id_error=ConnectionError("FunctionCall lookup was interrupted"),
    )

    snapshot = MODULE.status(receipt_path)
    assert snapshot["function_call_state"] == "unavailable"
    assert snapshot["recovered_state"] == "status-unavailable"


def test_status_keeps_retriable_modal_internal_failure_non_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class InternalFailure(Exception):
        pass

    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, _receipt("b" * 64, 10))
    _configure_status(
        monkeypatch,
        {},
        InternalFailure("temporary Modal internal failure"),
        internal_failure_type=InternalFailure,
    )

    snapshot = MODULE.status(receipt_path)
    assert snapshot["function_call_state"] == "unavailable"
    assert snapshot["recovered_state"] == "status-unavailable"
    with pytest.raises(MODULE.BundleStageError, match="no recovered terminal state"):
        MODULE._assert_no_unresolved_receipts(tmp_path)


def test_status_rejects_volume_success_with_function_call_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    size = 10
    sha256 = "9" * 64
    receipt = _receipt(sha256, size)
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, receipt)
    operation = str(receipt["remote_operation_path"])
    result = MODULE._result_payload(
        UPLOAD_ID,
        sha256,
        size,
        str(receipt["remote_final_path"]),
        VERIFICATION,
        reused=False,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )
    durable_status = MODULE._operation_status(
        UPLOAD_ID,
        sha256,
        size,
        "passed",
        2,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )
    call_error = RuntimeError(
        f"durable Modal bundle finalizer {UPLOAD_ID} failed; inspect its Volume journal"
    )
    _configure_status(
        monkeypatch,
        {
            f"{operation}/status.json": durable_status,
            f"{operation}/result.json": result,
        },
        call_error,
    )

    with pytest.raises(MODULE.BundleStageError, match="success and finalizer.*failure"):
        MODULE.status(receipt_path)


def test_status_rejects_volume_failure_with_function_call_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    size = 10
    sha256 = "a" * 64
    receipt = _receipt(sha256, size)
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, receipt)
    operation = str(receipt["remote_operation_path"])
    result = MODULE._result_payload(
        UPLOAD_ID,
        sha256,
        size,
        str(receipt["remote_final_path"]),
        VERIFICATION,
        reused=False,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )
    durable_status = MODULE._operation_status(
        UPLOAD_ID,
        sha256,
        size,
        "failed",
        2,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )
    _configure_status(
        monkeypatch,
        {
            f"{operation}/status.json": durable_status,
            f"{operation}/failure.json": _durable_failure(receipt),
        },
        result,
    )

    with pytest.raises(MODULE.BundleStageError, match="failure and finalizer.*success"):
        MODULE.status(receipt_path)


def test_status_treats_modal_function_timeout_as_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ModalTimeoutError(Exception):
        pass

    class FunctionTimeoutError(ModalTimeoutError):
        pass

    size = 10
    sha256 = "4" * 64
    receipt = _receipt(sha256, size)
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, receipt)
    operation = str(receipt["remote_operation_path"])
    durable_status = MODULE._operation_status(
        UPLOAD_ID,
        sha256,
        size,
        "running",
        1,
        function_call_id=CALL_ID,
        runtime_contract_sha256=RUNTIME_CONTRACT,
    )
    _configure_status(
        monkeypatch,
        {f"{operation}/status.json": durable_status},
        FunctionTimeoutError("four-hour timeout"),
        modal_timeout_type=ModalTimeoutError,
        function_timeout_type=FunctionTimeoutError,
    )

    snapshot = MODULE.status(receipt_path)
    assert snapshot["function_call_state"] == "failed"
    assert snapshot["recovered_state"] == "failed"


def test_status_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / UPLOAD_ID / "receipt.json"
    MODULE._write_json_atomic(receipt_path, _receipt("5" * 64, 10))
    _configure_status(monkeypatch, {}, KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        MODULE.status(receipt_path)
    assert not (receipt_path.parent / "status-latest.json").exists()


def test_prepared_only_guard_rejects_raw_parallel_payload(tmp_path: Path) -> None:
    archive = tmp_path / "raw.zip"
    _make_prepared_archive(archive, raw_origin=True)

    with pytest.raises(MODULE.BundleStageError, match="not prepared-only"):
        MODULE._assert_prepared_only_archive(archive, _FakePackage())


def test_finalizer_runtime_contract_hashes_every_copied_source(tmp_path: Path) -> None:
    for relative_path, payload in (
        (MODULE.STAGE_SCRIPT_RELATIVE_PATH, b"STAGE = 1\n"),
        (MODULE.PACKAGE_SCRIPT_RELATIVE_PATH, b"PACKAGE = 1\n"),
        (MODULE.FINALIZER_REQUIREMENTS_RELATIVE_PATH, b"numpy==1\n"),
        (MODULE.SOURCE_PACKAGE_RELATIVE_PATH / "runtime.py", b"RUNTIME = 1\n"),
        (MODULE.SOURCE_PACKAGE_RELATIVE_PATH / "schema.json", b"{}\n"),
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    first = MODULE.finalizer_runtime_contract_sha256(tmp_path)
    assert MODULE.SHA256_PATTERN.fullmatch(first)
    assert MODULE.finalizer_runtime_contract_sha256(tmp_path) == first

    (tmp_path / MODULE.SOURCE_PACKAGE_RELATIVE_PATH / "runtime.py").write_text(
        "RUNTIME = 2\n",
        encoding="utf-8",
    )
    assert MODULE.finalizer_runtime_contract_sha256(tmp_path) != first

    second = MODULE.finalizer_runtime_contract_sha256(tmp_path)
    (tmp_path / MODULE.SOURCE_PACKAGE_RELATIVE_PATH / "schema.json").write_text(
        '{"changed": true}\n',
        encoding="utf-8",
    )
    assert MODULE.finalizer_runtime_contract_sha256(tmp_path) != second


def test_finalizer_runtime_is_cpu_only_and_uses_posix_remote_paths() -> None:
    class FakeImage:
        def __init__(self) -> None:
            self.requirements: tuple[str, str] | None = None
            self.local_directories: list[tuple[str, str]] = []
            self.local_files: list[tuple[str, str]] = []
            self.environment: dict[str, str] = {}

        @classmethod
        def debian_slim(cls, *, python_version: str) -> FakeImage:
            assert python_version == "3.11"
            return cls()

        def pip_install_from_requirements(
            self,
            path: str,
            *,
            extra_options: str,
        ) -> FakeImage:
            self.requirements = (path, extra_options)
            return self

        def add_local_dir(
            self,
            local_path: str,
            remote_path: str,
            *,
            copy: bool,
            ignore: tuple[str, ...],
        ) -> FakeImage:
            assert copy is True
            assert ignore == ("**/__pycache__/**", "**/*.pyc", "**/*.pyo")
            self.local_directories.append((local_path, remote_path))
            return self

        def add_local_file(
            self,
            local_path: str,
            remote_path: str,
            *,
            copy: bool,
        ) -> FakeImage:
            assert copy is True
            self.local_files.append((local_path, remote_path))
            return self

        def env(self, environment: dict[str, str]) -> FakeImage:
            self.environment = environment
            return self

    class FakeApp:
        def __init__(self, name: str, *, image: FakeImage, include_source: bool) -> None:
            self.name = name
            self.image = image
            self.include_source = include_source
            self.function_options: dict[str, Any] | None = None

        def function(self, **options: Any) -> Any:
            self.function_options = options

            def decorate(function: Any) -> Any:
                return SimpleNamespace(local=function)

            return decorate

    fake_modal = SimpleNamespace(Image=FakeImage, App=FakeApp)
    volume = object()

    app, _finalizer = MODULE._build_finalizer_runtime(fake_modal, volume)

    assert app.name == MODULE.APP_NAME
    assert app.include_source is False
    assert app.image.requirements == (
        str(MODULE.FINALIZER_REQUIREMENTS),
        "--require-hashes --only-binary=:all: --no-cache-dir",
    )
    assert app.image.environment == {
        "PYTHONPATH": "/opt/sion-bundle-stage/src",
        "PYTHONUNBUFFERED": "1",
    }
    remote_copy_paths = [path for _local, path in app.image.local_directories]
    remote_copy_paths.extend(path for _local, path in app.image.local_files)
    assert remote_copy_paths == [
        "/opt/sion-bundle-stage/src/sion_translate",
        "/opt/sion-bundle-stage/scripts/package_gpu_bundle.py",
        "/opt/sion-bundle-stage/scripts/modal_stage_gpu_bundle.py",
        "/opt/sion-bundle-stage/requirements/modal-bundle-stage.txt",
    ]
    assert all("\\" not in path for path in remote_copy_paths)
    options = app.function_options
    assert options is not None
    assert options["volumes"] == {"/mnt/sion-bundles": volume}
    assert options["cpu"] == MODULE.FINALIZER_CPU_CORES
    assert options["memory"] == MODULE.FINALIZER_MEMORY_MIB
    assert "ephemeral_disk" not in options
    assert options["timeout"] == MODULE.FINALIZER_TIMEOUT_SECONDS
    assert options["retries"] == 0
    assert options["max_containers"] == 1
    assert options["single_use_containers"] is True
    assert options["serialized"] is True
    assert "gpu" not in options


def test_executed_finalizer_must_match_reviewed_image_copy(tmp_path: Path) -> None:
    executed = tmp_path / "mounted.py"
    reviewed = tmp_path / "reviewed.py"
    executed.write_text("VALUE = 1\n", encoding="utf-8")
    reviewed.write_text("VALUE = 1\n", encoding="utf-8")

    MODULE._verify_executed_entrypoint(executed, reviewed)
    reviewed.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(MODULE.BundleStageError, match="differs from its reviewed image copy"):
        MODULE._verify_executed_entrypoint(executed, reviewed)


def test_modal_client_guard_rejects_unreviewed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreviewed_version(_name: str) -> str:
        return "1.5.4"

    monkeypatch.setattr(MODULE.importlib.metadata, "version", unreviewed_version)
    with pytest.raises(RuntimeError, match="requires local Modal client 1.5.3"):
        MODULE._validate_modal_client_version()
