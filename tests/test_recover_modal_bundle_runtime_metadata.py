from __future__ import annotations

from contextlib import AbstractContextManager
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "recover_modal_bundle_runtime_metadata.py"
)
SPEC = importlib.util.spec_from_file_location(
    "recover_modal_bundle_runtime_metadata_test",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SOURCE_UPLOAD_ID = "bundle-20260902t111200z-0123456789abcdef"
ATTEMPT_UPLOAD_ID = "bundle-20260902t120000z-fedcba9876543210"
SOURCE_CALL_ID = "fc-01M1GX40JCW9B1ZGMFDMP6SGEE"
SOURCE_APP_ID = "ap-9MttppC4m4j9CS10yUL6lZ"
ATTEMPT_CALL_ID = "fc-01M1GY00000000000000000000"
SHA256 = "a" * 64
SOURCE_RUNTIME = "b" * 64
REPLACEMENT_RUNTIME = "c" * 64
CLAIM_ID = "claim-" + "d" * 32
ORIGINAL_CLAIM_ID = "claim-" + "e" * 32


class _UploadBatch(AbstractContextManager["_UploadBatch"]):
    def __init__(self, volume: "_Volume") -> None:
        self.volume = volume

    def put_file(self, source: io.BytesIO, destination: str) -> None:
        if destination in self.volume.files:
            raise FileExistsError(destination)
        payload = source.read()
        self.volume.files[destination] = payload
        self.volume.uploads.append((destination, payload))

    def __exit__(self, *_args: object) -> None:
        return None


class _Volume:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.uploads: list[tuple[str, bytes]] = []

    def read_file(self, path: str) -> list[bytes]:
        if path not in self.files:
            raise FileNotFoundError(path)
        return [self.files[path]]

    def batch_upload(self, *, force: bool) -> _UploadBatch:
        assert force is False
        return _UploadBatch(self)


class _RunContext(AbstractContextManager[None]):
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


def _source_receipt(tmp_path: Path) -> dict[str, object]:
    return {
        "volume_name": "sion-prepared-bundles",
        "local_bundle_path": str((tmp_path / "bundle.zip").resolve()),
        "bundle_sha256": SHA256,
        "bundle_size": 1024,
        "verification": {
            "file_count": 10,
            "total_bytes": 2048,
            "git_commit": "1" * 40,
            "git_tree": "2" * 40,
        },
    }


def _claim() -> dict[str, object]:
    return MODULE.STAGE._source_recovery_claim_payload(
        source_upload_id=SOURCE_UPLOAD_ID,
        attempt_upload_id=ATTEMPT_UPLOAD_ID,
        bundle_sha256=SHA256,
        bundle_size=1024,
        source_function_call_id=SOURCE_CALL_ID,
        source_app_id=SOURCE_APP_ID,
        source_runtime_contract_sha256=SOURCE_RUNTIME,
        replacement_runtime_contract_sha256=REPLACEMENT_RUNTIME,
        replacement_commit="7" * 40,
        replacement_tree="8" * 40,
        recovery_builder_sha256="9" * 64,
        source_status_sha256="3" * 64,
        source_failure_sha256="4" * 64,
        source_receipt_sha256="5" * 64,
        original_submission_claim_sha256="6" * 64,
        original_submission_claim_id=ORIGINAL_CLAIM_ID,
        original_submission_runtime_contract_sha256=SOURCE_RUNTIME,
        source_receipt_created_at_utc="2026-09-02T11:11:54+00:00",
        recovery_claim_id=CLAIM_ID,
    )


def test_attempt_receipt_records_no_archive_upload(tmp_path: Path) -> None:
    receipt = MODULE._attempt_receipt(
        _source_receipt(tmp_path),
        ATTEMPT_UPLOAD_ID,
        REPLACEMENT_RUNTIME,
        "2026-09-02T12:00:00+00:00",
        max_dollars=1.27,
        workspace_budget=1.50,
        workspace_usage=0.00001153,
        recovery_claim=_claim(),
    )

    assert receipt["upload_state"] == "source-bound"
    assert receipt["remote_incoming_path"] == f"/incoming/{SOURCE_UPLOAD_ID}"
    assert receipt["receipt_version"] == 2
    assert receipt["function_call_id"] is None
    assert receipt["submission_claim_state"] == "not-created"


def test_recovery_claim_uploads_only_one_small_json_file() -> None:
    volume = _Volume()
    claim = _claim()

    MODULE._put_recovery_claim(volume, claim)
    MODULE._put_recovery_claim(volume, claim)

    assert len(volume.uploads) == 1
    path, payload = volume.uploads[0]
    assert path == claim["remote_recovery_claim_path"]
    assert path.endswith(".json")
    assert "bundle.zip" not in path
    assert json.loads(payload) == claim


def test_attempt_submission_spawns_once_and_never_uploads_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    volume = _Volume()
    receipt = MODULE._attempt_receipt(
        _source_receipt(tmp_path),
        ATTEMPT_UPLOAD_ID,
        REPLACEMENT_RUNTIME,
        "2026-09-02T12:00:00+00:00",
        max_dollars=1.27,
        workspace_budget=1.50,
        workspace_usage=0.00001153,
        recovery_claim=_claim(),
    )
    receipt_path = tmp_path / "attempt" / "receipt.json"
    receipt_path.parent.mkdir()
    MODULE.STAGE._write_json_atomic(receipt_path, receipt)
    spawned: list[tuple[object, ...]] = []
    MODULE._put_recovery_claim(volume, _claim())

    class Finalizer:
        def spawn(self, *arguments: object) -> object:
            spawned.append(arguments)
            return SimpleNamespace(object_id=ATTEMPT_CALL_ID)

    class App:
        def run(self, **options: object) -> _RunContext:
            assert options == {"detach": True}
            return _RunContext()

    app = App()

    def build_runtime(_modal: object, _volume: object) -> tuple[App, Finalizer]:
        return app, Finalizer()

    monkeypatch.setattr(
        MODULE.STAGE,
        "build_source_recovery_runtime",
        build_runtime,
    )

    MODULE._submit_attempt(
        SimpleNamespace(),
        volume,
        receipt_path,
        receipt,
        SOURCE_UPLOAD_ID,
        _claim(),
    )

    observed = MODULE.STAGE._read_receipt(receipt_path)
    assert observed["finalizer_state"] == "submitted"
    assert observed["function_call_id"] == ATTEMPT_CALL_ID
    assert len(spawned) == 1

    cloned = {**observed, "function_call_id": None, "finalizer_state": "not-submitted"}
    with pytest.raises(FileExistsError, match="spawn.json"):
        MODULE._submit_attempt(
            SimpleNamespace(), volume, receipt_path, cloned, SOURCE_UPLOAD_ID, _claim()
        )
    assert len(spawned) == 1
    assert spawned[0][0] == SOURCE_UPLOAD_ID
    assert spawned[0][1] == ATTEMPT_UPLOAD_ID
    assert all(path.endswith(".json") and "bundle.zip" not in path for path, _ in volume.uploads)
    with pytest.raises(MODULE.RuntimeMetadataRecoveryError, match="not fresh"):
        MODULE._submit_attempt(
            SimpleNamespace(),
            volume,
            receipt_path,
            cast(dict[str, Any], observed),
            SOURCE_UPLOAD_ID,
            _claim(),
        )
    assert len(spawned) == 1


def test_exact_worker_deserializes_against_the_copied_remote_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from modal._serialization import deserialize, serialize

    stage = MODULE.STAGE
    monkeypatch.setitem(sys.modules, "modal_stage_gpu_bundle", stage)

    class Image:
        @classmethod
        def debian_slim(cls, **_options: object) -> "Image":
            return cls()

        def __getattr__(self, _name: str) -> Any:
            def chain(*_args: object, **_kwargs: object) -> "Image":
                return self

            return chain

    class App:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def function(self, **options: object) -> Any:
            assert options["include_source"] is False
            assert options["retries"] == 0
            assert "gpu" not in options

            def decorate(function: Any) -> Any:
                return function

            return decorate

    _app, worker = stage.build_source_recovery_runtime(
        SimpleNamespace(Image=Image, App=App), object()
    )
    payload = serialize(worker)
    remote_root = tmp_path / "remote"
    for relative in stage.finalizer_runtime_contract_paths(stage.REPOSITORY_ROOT):
        destination = remote_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stage.REPOSITORY_ROOT / relative, destination)
    # No submitting-process stage or verifier module may survive the round-trip.
    for name in ("modal_stage_gpu_bundle", "sion_package_gpu_bundle_for_stage"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(remote_root / "scripts"))
    restored = deserialize(payload, None)
    remote_function = restored.__globals__["_finalize_bundle"]
    assert (
        Path(remote_function.__code__.co_filename).resolve()
        == (remote_root / "scripts" / "modal_stage_gpu_bundle.py").resolve()
    )
    assert remote_function is not stage._finalize_bundle


@pytest.mark.parametrize(
    "field,value",
    [
        ("volume_name", "another-volume"),
        ("created_at_utc", "changed"),
        ("upload_id", SOURCE_UPLOAD_ID),
        ("remote_operation_path", "/operations/redirected"),
        ("verification", {}),
        ("upload_state", "uploaded"),
    ],
)
def test_attempt_identity_rejects_all_redirects(tmp_path: Path, field: str, value: object) -> None:
    expected = MODULE._attempt_receipt(
        _source_receipt(tmp_path),
        ATTEMPT_UPLOAD_ID,
        REPLACEMENT_RUNTIME,
        "2026-09-02T12:00:00+00:00",
        max_dollars=1.27,
        workspace_budget=1.50,
        workspace_usage=0.0,
        recovery_claim=_claim(),
    )
    changed = {**expected, field: value}
    with pytest.raises(MODULE.RuntimeMetadataRecoveryError, match="conflicts"):
        MODULE._validate_attempt_identity(changed, expected)


@pytest.mark.parametrize("mutation", ["none", "modified", "missing", "extra", "ignored-extra"])
def test_revision_guard_checks_every_runtime_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for relative in (
        "scripts/modal_stage_gpu_bundle.py",
        "scripts/package_gpu_bundle.py",
        "scripts/recover_modal_bundle_runtime_metadata.py",
        "requirements/modal-bundle-stage.txt",
        "src/sion_translate/__init__.py",
        "src/sion_translate/nested/runtime.py",
        ".gitignore",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "*.json\n" if relative == ".gitignore" else "# reviewed\n", encoding="utf-8"
        )

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    git("init", "--quiet")
    git("config", "core.autocrlf", "false")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    commit = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", commit)
    monkeypatch.setattr(MODULE, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(
        MODULE, "__file__", str(root / "scripts/recover_modal_bundle_runtime_metadata.py")
    )
    if mutation == "modified":
        (root / "scripts/package_gpu_bundle.py").write_text("# changed\n", encoding="utf-8")
    elif mutation == "missing":
        (root / "src/sion_translate/nested/runtime.py").unlink()
    elif mutation in {"extra", "ignored-extra"}:
        suffix = "json" if mutation == "ignored-extra" else "py"
        (root / f"src/sion_translate/injected.{suffix}").write_text("{}\n", encoding="utf-8")
    if mutation == "none":
        assert MODULE._replacement_revision() == (commit, git("rev-parse", "HEAD^{tree}"))
    else:
        with pytest.raises(MODULE.RuntimeMetadataRecoveryError, match="differs"):
            MODULE._replacement_revision()


@pytest.mark.parametrize("ambiguous_spawn", [False, True])
def test_full_recovery_preserves_source_and_never_repeats_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambiguous_spawn: bool,
) -> None:
    stage = MODULE.STAGE
    source = MODULE._attempt_receipt(
        _source_receipt(tmp_path),
        ATTEMPT_UPLOAD_ID,
        REPLACEMENT_RUNTIME,
        "2026-09-02T12:00:00+00:00",
        max_dollars=1.27,
        workspace_budget=1.5,
        workspace_usage=0.0,
        recovery_claim=_claim(),
    )
    source.pop("source_recovery_claim")
    source.update(
        receipt_version=1,
        upload_id=SOURCE_UPLOAD_ID,
        runtime_contract_sha256=SOURCE_RUNTIME,
        upload_state="uploaded",
        finalizer_state="submitted",
        function_call_id=SOURCE_CALL_ID,
        submission_claim_id=ORIGINAL_CLAIM_ID,
        submission_claim_state="created",
        remote_submission_claim_path=stage._remote_submission_claim_path(SOURCE_UPLOAD_ID),
    )
    incoming, final, operation = stage._remote_paths(SOURCE_UPLOAD_ID, SHA256)
    source.update(
        remote_incoming_path=incoming, remote_final_path=final, remote_operation_path=operation
    )
    source_path = tmp_path / "receipts" / SOURCE_UPLOAD_ID / "receipt.json"
    stage._write_json_atomic(source_path, source)
    status = stage._operation_status(
        SOURCE_UPLOAD_ID,
        SHA256,
        1024,
        "failed",
        2,
        function_call_id=SOURCE_CALL_ID,
        runtime_contract_sha256=SOURCE_RUNTIME,
    )
    failure = {
        **{
            key: status[key]
            for key in (
                "schema",
                "upload_id",
                "bundle_sha256",
                "bundle_size",
                "function_call_id",
                "runtime_contract_sha256",
            )
        },
        "error_type": "PackageNotFoundError",
        "message": stage.SOURCE_RECOVERY_ERROR_MESSAGE,
        "traceback_tail": "importlib.metadata.PackageNotFoundError",
        "recorded_at_utc": "2026-09-02T12:00:00+00:00",
    }
    stage._write_json_atomic(
        source_path.parent / "status-latest.json",
        {
            "receipt": source,
            "recovered_state": "failed",
            "function_call_state": "failed",
            "observed_function_call_id": SOURCE_CALL_ID,
            "remote_status": status,
            "remote_failure": failure,
            "remote_result": None,
            "function_call_error": {
                "error_type": "RuntimeError",
                "message": (
                    f"durable Modal bundle finalizer {SOURCE_UPLOAD_ID} failed; inspect its Volume journal"
                ),
            },
        },
    )

    class Volume(_Volume):
        def iterdir(self, path: str, *, recursive: bool) -> list[object]:
            assert recursive is False
            if path == incoming:
                return [
                    SimpleNamespace(
                        path=incoming + "/bundle.zip", size=1024, type=SimpleNamespace(name="FILE")
                    )
                ]
            raise FileNotFoundError(path)

    volume = Volume()
    volume.files[operation + "/status.json"] = stage._json_bytes(status)
    volume.files[operation + "/failure.json"] = stage._json_bytes(failure)
    # The first upload's runtime differs from the later failed recovery's runtime.
    volume.files[source["remote_submission_claim_path"]] = stage._json_bytes(
        stage._submission_claim_payload({**source, "runtime_contract_sha256": "f" * 64})
    )
    source_bytes = source_path.read_bytes()
    prior_files = dict(volume.files)
    spawn_calls: list[tuple[object, ...]] = []

    class Finalizer:
        def spawn(self, *args: object) -> object:
            spawn_calls.append(args)
            if ambiguous_spawn:
                raise OSError("lost spawn response")
            return SimpleNamespace(object_id=ATTEMPT_CALL_ID)

    class App:
        def run(self, *, detach: bool) -> _RunContext:
            assert detach
            return _RunContext()

    class VolumeType:
        @staticmethod
        def from_name(*_args: object, **_kwargs: object) -> Volume:
            return volume

    def modal_client() -> object:
        return SimpleNamespace(Volume=VolumeType)

    def revision() -> tuple[str, str]:
        return "7" * 40, "8" * 40

    def no_op(*_args: object) -> None:
        return None

    def runtime_hash(_root: Path) -> str:
        return REPLACEMENT_RUNTIME

    def build(*_args: object) -> tuple[App, Finalizer]:
        return App(), Finalizer()

    actual_hash = stage._hash_regular_file_stable

    def file_hash(path: Path, label: str) -> tuple[int, str]:
        if path == Path(source["local_bundle_path"]):
            return 1024, SHA256
        return actual_hash(path, label)

    monkeypatch.setattr(MODULE, "_assert_app_stopped", no_op)
    monkeypatch.setattr(MODULE, "_assert_terminal_failed_call", no_op)
    monkeypatch.setattr(MODULE, "_replacement_revision", revision)
    monkeypatch.setattr(stage, "_require_modal", modal_client)
    monkeypatch.setattr(stage, "finalizer_runtime_contract_sha256", runtime_hash)
    monkeypatch.setattr(stage, "_hash_regular_file_stable", file_hash)
    monkeypatch.setattr(stage, "build_source_recovery_runtime", build)
    options = dict(max_dollars=1.27, workspace_budget=1.5, workspace_usage=0.0)
    MODULE.recover(source_path, SOURCE_APP_ID, preflight_only=True, **options)
    assert not volume.uploads and not spawn_calls
    assert not (source_path.parent / "runtime-metadata-recovery-intent.json").exists()
    if ambiguous_spawn:
        with pytest.raises(OSError, match="lost spawn"):
            MODULE.recover(source_path, SOURCE_APP_ID, preflight_only=False, **options)
    else:
        receipt_path = MODULE.recover(source_path, SOURCE_APP_ID, preflight_only=False, **options)
        receipt = stage._read_receipt(receipt_path)
        assert receipt["receipt_version"] == 2
        assert receipt["remote_incoming_path"] == incoming
        claim = receipt["source_recovery_claim"]
        result = stage._result_payload(
            receipt["upload_id"],
            SHA256,
            1024,
            final,
            source["verification"],
            reused=False,
            function_call_id=ATTEMPT_CALL_ID,
            runtime_contract_sha256=REPLACEMENT_RUNTIME,
        )
        result.update(
            source_upload_id=SOURCE_UPLOAD_ID,
            source_recovery_claim_sha256=hashlib.sha256(stage._json_bytes(claim)).hexdigest(),
        )
        assert stage._validated_result(result, receipt) == result
    with pytest.raises(MODULE.RuntimeMetadataRecoveryError, match="already submitted"):
        MODULE.recover(source_path, SOURCE_APP_ID, preflight_only=False, **options)
    assert len(spawn_calls) == 1
    assert source_path.read_bytes() == source_bytes
    assert all(volume.files[path] == payload for path, payload in prior_files.items())
    assert all(path.endswith(".json") and "bundle.zip" not in path for path, _ in volume.uploads)
