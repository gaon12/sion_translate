from __future__ import annotations

# Tests intentionally exercise the script's internal safety boundaries.
# pyright: reportPrivateUsage=false

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts.modal_train_tokenizer import (
    EXPECTED_MONOLINGUAL_SENTENCES,
    EXPECTED_SENTENCEPIECE_VERSION,
    EXPECTED_TOKENIZER_SAMPLE_RATIO,
    EXPECTED_TOTAL_SENTENCES,
    REQUIRED_ARTIFACTS,
    SourceRecord,
    _artifact_records,
    _file_sha256,
    _manifest_digest,
    _new_run_id,
    _observe_tokenizer_sentence_counts,
    _parse_source_manifest,
    _publish_candidate,
    _records_digest,
    _validate_training_metadata,
)


def source_manifest(records: list[SourceRecord]) -> dict[str, object]:
    return {
        "version": 1,
        "git_commit": "a" * 40,
        "config_path": "sion_translate.yaml",
        "config_sha256": "b" * 64,
        "tokenizer_sample_ratio": EXPECTED_TOKENIZER_SAMPLE_RATIO,
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


def test_iterator_count_mismatch_fails_before_training_continues() -> None:
    class TokenizerModule:
        @staticmethod
        def iter_tokenizer_sentences() -> Iterator[str]:
            yield "one sentence"

    with _observe_tokenizer_sentence_counts(TokenizerModule) as observed:
        with pytest.raises(RuntimeError, match="iterator count differs"):
            list(TokenizerModule.iter_tokenizer_sentences())

    assert observed == []


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


def test_training_metadata_requires_measured_production_counts(tmp_path: Path) -> None:
    metadata = {
        "sentencepiece_version": EXPECTED_SENTENCEPIECE_VERSION,
        "monolingual_sample_ratio": EXPECTED_TOKENIZER_SAMPLE_RATIO,
        "monolingual_sentences": EXPECTED_MONOLINGUAL_SENTENCES,
    }
    (tmp_path / "tokenizer_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert (
        _validate_training_metadata(tmp_path, [EXPECTED_TOTAL_SENTENCES, EXPECTED_TOTAL_SENTENCES])
        == metadata
    )

    metadata["monolingual_sentences"] = {"ja": 1, "ko": 2}
    (tmp_path / "tokenizer_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="monolingual sample counts"):
        _validate_training_metadata(tmp_path, [EXPECTED_TOTAL_SENTENCES, EXPECTED_TOTAL_SENTENCES])


def test_file_hash_uses_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"tokenizer")
    assert _file_sha256(path) == hashlib.sha256(b"tokenizer").hexdigest()
