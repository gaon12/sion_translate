from __future__ import annotations

# Tests intentionally exercise the script's internal safety boundaries.
# pyright: reportPrivateUsage=false

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import scripts.modal_train_tokenizer as modal_tokenizer_script

from scripts.modal_train_tokenizer import (
    CHILD_MODE_FLAG,
    EXPECTED_SENTENCEPIECE_VERSION,
    REQUIRED_ARTIFACTS,
    SOURCE_MANIFEST_VERSION,
    SourceRecord,
    _artifact_records,
    _child_failure_message,
    _file_sha256,
    _load_child_result,
    _manifest_digest,
    _new_run_id,
    _parse_source_manifest,
    _publish_candidate,
    _records_digest,
    _run_training_subprocess,
    _validate_training_metadata,
    _validate_training_source_identities,
)

SAMPLE_RATIO = 0.37


def source_manifest(records: list[SourceRecord]) -> dict[str, object]:
    return {
        "version": SOURCE_MANIFEST_VERSION,
        "git_commit": "a" * 40,
        "config_path": "sion_translate.yaml",
        "config_sha256": "b" * 64,
        "tokenizer_sample_ratio": SAMPLE_RATIO,
        "file_count": len(records),
        "total_bytes": sum(record.size for record in records),
        "files_sha256": _records_digest(records),
        "files": [
            {"path": record.path, "size": record.size, "sha256": record.sha256}
            for record in records
        ],
    }


def test_source_manifest_rejects_tampering_and_unsafe_paths() -> None:
    records = [SourceRecord("data/a.jsonl", 3, hashlib.sha256(b"abc").hexdigest())]
    manifest = source_manifest(records)

    assert _parse_source_manifest(manifest) == records
    assert len(_manifest_digest(manifest)) == 64

    tampered = {**manifest, "total_bytes": 4}
    with pytest.raises(ValueError, match="byte count"):
        _parse_source_manifest(tampered)

    unsafe = source_manifest([SourceRecord("../secret", 3, records[0].sha256)])
    with pytest.raises(ValueError, match="stay below"):
        _parse_source_manifest(unsafe)


def test_generated_run_id_is_publishable(tmp_path: Path) -> None:
    manifest = source_manifest([])
    run_id = _new_run_id("a" * 40, manifest)
    build = tmp_path / "build"
    build.mkdir()
    (build / "artifact").write_text("ok", encoding="utf-8")

    published = _publish_candidate(build, tmp_path / "output", run_id)

    assert published.name == run_id


def test_candidate_publish_is_complete_atomic_and_never_overwrites(tmp_path: Path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    for name in REQUIRED_ARTIFACTS:
        (build / name).write_bytes(name.encode())
    artifacts = _artifact_records(build)
    assert set(artifacts) == REQUIRED_ARTIFACTS
    assert all(value["sha256"] for value in artifacts.values())

    output = tmp_path / "output"
    published = _publish_candidate(build, output, "ratio-040-test")
    assert {path.name for path in published.iterdir()} == REQUIRED_ARTIFACTS
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _publish_candidate(build, output, "ratio-040-test")


def test_training_metadata_accepts_arbitrary_language_counts(tmp_path: Path) -> None:
    contract = {"schema": "sion-tokenizer-training-v3", "languages": ["de", "fr"]}
    metadata = {
        "sentencepiece_version": EXPECTED_SENTENCEPIECE_VERSION,
        "monolingual_sample_ratio": SAMPLE_RATIO,
        "monolingual_sentences": {"fr": 5},
        "corpus_sentences": 100,
        "corpus_sentences_per_language": {"de": 60, "fr": 40},
        "sampled_sentences": 50,
        "sampled_sentences_per_language": {"de": 28, "fr": 22},
        "training_contract": contract,
        "training_contract_sha256": _manifest_digest(contract),
    }
    (tmp_path / "tokenizer_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert _validate_training_metadata(tmp_path, SAMPLE_RATIO) == metadata

    metadata["sampled_sentences_per_language"] = {"de": 28, "fr": 21}
    (tmp_path / "tokenizer_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not sum"):
        _validate_training_metadata(tmp_path, SAMPLE_RATIO)


def test_file_hash_uses_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"tokenizer")
    assert _file_sha256(path) == hashlib.sha256(b"tokenizer").hexdigest()


def test_source_hash_rejects_mutation_during_the_hash_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "data" / "parallel.jsonl"
    source.parent.mkdir()
    source.write_bytes(b"original")
    actual_hash = modal_tokenizer_script._file_sha256

    def mutate_after_read(path: Path) -> str:
        digest = actual_hash(path)
        path.write_bytes(b"replacement with a different size")
        return digest

    monkeypatch.setattr(modal_tokenizer_script, "_file_sha256", mutate_after_read)
    with pytest.raises(RuntimeError, match="changed while it was hashed"):
        modal_tokenizer_script._hash_source(source, tmp_path)


def test_child_training_sources_must_match_the_parent_manifest() -> None:
    expected_digest = hashlib.sha256(b"expected").hexdigest()
    changed_digest = hashlib.sha256(b"changed").hexdigest()
    expected = [SourceRecord("data/parallel.jsonl", 8, expected_digest)]
    metadata: dict[str, object] = {
        "training_contract": {
            "sources": [{"path": "parallel.jsonl", "size": 8, "sha256": expected_digest}]
        }
    }

    _validate_training_source_identities(metadata, expected)

    metadata["training_contract"] = {
        "sources": [{"path": "parallel.jsonl", "size": 7, "sha256": changed_digest}]
    }
    with pytest.raises(RuntimeError, match="differ from the parent manifest"):
        _validate_training_source_identities(metadata, expected)


def test_parent_heartbeats_while_mock_child_holds_the_gil(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeProcess:
        pid = 4242

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.waits = 0

        def wait(self, timeout: float) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["child"], timeout)
            self.returncode = 0
            return 0

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            raise AssertionError("successful child must not be terminated")

        def kill(self) -> None:
            raise AssertionError("successful child must not be killed")

    process = FakeProcess()

    def fake_popen(command: object) -> FakeProcess:
        del command
        return process

    monkeypatch.setattr(
        modal_tokenizer_script.subprocess,
        "Popen",
        fake_popen,
    )

    _run_training_subprocess(["python", "child.py"], heartbeat_seconds=45)

    output = capsys.readouterr().out
    assert "tokenizer child alive" in output
    assert "tokenizer child completed" in output


def test_parent_reports_native_child_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    class CrashedProcess:
        pid = 5252

        @staticmethod
        def wait(timeout: float) -> int:
            del timeout
            return -11

        @staticmethod
        def poll() -> int:
            return -11

        @staticmethod
        def terminate() -> None:
            raise AssertionError("exited child must not be terminated")

        @staticmethod
        def kill() -> None:
            raise AssertionError("exited child must not be killed")

    def fake_popen(command: object) -> CrashedProcess:
        del command
        return CrashedProcess()

    monkeypatch.setattr(
        modal_tokenizer_script.subprocess,
        "Popen",
        fake_popen,
    )

    with pytest.raises(RuntimeError, match="SIGSEGV"):
        _run_training_subprocess(["python", "child.py"])
    assert "SIGSEGV" in _child_failure_message(139)


def test_parent_validates_child_result_json(tmp_path: Path) -> None:
    result_path = tmp_path / "child-result.json"
    result = {
        "sentencepiece_version": EXPECTED_SENTENCEPIECE_VERSION,
        "tokenizer_sample_ratio": SAMPLE_RATIO,
        "corpus_sentences": 100,
        "corpus_sentences_per_language": {"de": 60, "fr": 40},
        "sampled_sentences": 50,
        "sampled_sentences_per_language": {"de": 28, "fr": 22},
        "monolingual_sentences": {"fr": 5},
        "training_contract_sha256": "c" * 64,
        "cpu_plan": {
            "available": 16,
            "preprocess_workers": 8,
            "sentencepiece_threads": 8,
        },
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert _load_child_result(result_path, expected_sample_ratio=SAMPLE_RATIO) == result

    result["sampled_sentences"] = 101
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sampled sentence count"):
        _load_child_result(result_path, expected_sample_ratio=SAMPLE_RATIO)


def test_child_mode_help_does_not_construct_modal_app() -> None:
    script = Path(modal_tokenizer_script.__file__)
    completed = subprocess.run(
        [sys.executable, str(script), CHILD_MODE_FLAG, "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SentencePiece child process" in completed.stdout
