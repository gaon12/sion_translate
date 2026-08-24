from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import sion_translate.queue_translation as queue_translation_module
from sion_translate.cli.translate_queue import (
    _resolve_run_metadata_seed,
    _validate_resume_runtime_metadata,
    build_parser,
)
from sion_translate.data.records import expand_parallel_record
from sion_translate.queue_translation import (
    PermanentQueueRowError,
    QueueTranslationOptions,
    RetryableQueueTranslationError,
    _accepted_run_lock,
    _queue_run_lock,
    translate_queue as _translate_queue,
)
from sion_translate.scripts_registry import script_letter_count


class FakeTokenizer:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = str(model_path) if model_path is not None else None

    @staticmethod
    def encode(text: str) -> list[str]:
        return [character for character in text if not character.isspace()]


def test_queue_quality_marks_unprofiled_languages_as_unchecked() -> None:
    quality, reasons = queue_translation_module._forward_quality(
        "source language sentence",
        "target language sentence",
        source_language="qaa",
        target_language="qab",
    )

    assert quality["target_language_fraction"] is None
    assert not {"ko_script_mismatch", "ja_script_mismatch"} & set(reasons)


def test_queue_quality_reports_language_independent_script_counts() -> None:
    quality, _reasons = queue_translation_module._forward_quality(
        "한국어 원문 문장입니다.",
        "人工知能研究社会文化交流発展",
        source_language="ko-KR",
        target_language="ja-JP",
    )

    assert quality["target_script_characters"] == {"han": 14}
    assert "ja_no_kana" in quality["pair_warnings"]


def _hold_queue_lock(path: str, ready, release) -> None:
    with _queue_run_lock(Path(path)):
        ready.set()
        release.wait(timeout=10)


class FakeTranslator:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.translation_model_path: str | None = None
        self.tokenizer_model_path: str | None = None
        self.tokenizer_metadata_path: str | None = None
        self.token_features_path: str | None = None
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []
        self.export_metadata: dict[str, object] = {}
        self.tokenizer_metadata: dict[str, object] | None = None
        self.translation_directions = (
            ("ko", "ja"),
            ("ja", "ko"),
        )
        self.mapping = {
            ("ko", "ja", "안녕하세요."): "こんにちは。",
            ("ja", "ko", "こんにちは。"): "안녕하세요.",
            ("ko", "ja", "원래 문장입니다."): "元の文です。",
            ("ja", "ko", "元の文です。"): "전혀 다른 내용입니다.",
            ("ko", "ja", "정상 문장입니다."): "正常な文です。",
            ("ja", "ko", "正常な文です。"): "정상 문장입니다.",
        }

    @property
    def translation_directions(self) -> tuple[tuple[str, str], ...]:
        return self._translation_directions

    @translation_directions.setter
    def translation_directions(self, value: tuple[tuple[str, str], ...]) -> None:
        self._translation_directions = value
        self.export_metadata["translation_directions"] = [list(item) for item in value]

    def translate(
        self,
        texts,
        *,
        source_language,
        target_language,
        num_beams,
        max_new_tokens,
        batch_size,
        max_output_length_ratio,
        max_output_length_margin,
    ):
        del (
            num_beams,
            max_new_tokens,
            batch_size,
            max_output_length_ratio,
            max_output_length_margin,
        )
        self.calls.append((source_language, target_language, tuple(texts)))
        if "고장 문장입니다." in texts:
            raise PermanentQueueRowError("synthetic row failure")
        return [self.mapping[(source_language, target_language, text)] for text in texts]


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _make_writable(path: Path) -> None:
    """Allow an adversarial fixture to modify a queue-owned read-only file."""

    os.chmod(path, 0o600)


def _as_exact_legacy_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Convert a test manifest to the exact pipeline-v1 migration shape."""

    for field in ("signature_version", "parts", "integrity", "training_set"):
        manifest.pop(field)
    manifest["configuration"]["pipeline_version"] = 1
    signature = queue_translation_module._stable_digest(manifest["configuration"])
    manifest["run_signature"] = signature
    manifest["run_id"] = signature[:16]
    return manifest


def _run_metadata(
    translator: FakeTranslator,
    artifact_dir: Path,
    *,
    model: str = "fake-v1",
) -> dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{model}.pt"
    model_path.write_text(
        json.dumps(
            {
                "model": model,
                "translation_directions": [
                    list(item) for item in translator.translation_directions
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tokenizer_path = artifact_dir / "fake-tokenizer.model"
    tokenizer_path.write_bytes(b"recorded fake tokenizer")
    translator.translation_model_path = str(model_path.resolve())
    translator.tokenizer_model_path = str(tokenizer_path.resolve())
    translator.tokenizer_metadata_path = None
    translator.token_features_path = None
    translator.tokenizer = FakeTokenizer(tokenizer_path)
    return {
        "source_dataset": "tests/queue-corpus",
        "source_revision": "sha256:" + hashlib.sha256(b"queue-corpus").hexdigest(),
        "source_license": "LicenseRef-Test-Fixture",
        "translation_model": _identity(model_path),
        "tokenizer": _identity(tokenizer_path),
        "tokenizer_metadata": None,
        "token_features": None,
        "translation_directions": [list(item) for item in translator.translation_directions],
        "translation_graph_source": "translation_model",
    }


def translate_queue(
    input_path: str | Path,
    output_dir: str | Path,
    translator: FakeTranslator,
    **kwargs: Any,
) -> dict[str, Any]:
    artifact_dir = Path(output_dir).parent / ".queue-runtime-artifacts"
    if "run_metadata" not in kwargs:
        kwargs["run_metadata"] = _run_metadata(translator, artifact_dir)
    kwargs.setdefault("allow_unverified_translator", True)
    return _translate_queue(input_path, output_dir, translator, **kwargs)


def _published_accepted_paths(manifest: Mapping[str, Any]) -> list[Path]:
    return [
        Path(part["accepted"]["path"])
        for part in manifest["parts"]
        if part.get("published") is True
    ]


def _training_paths(manifest: Mapping[str, Any]) -> list[Path]:
    training_set = manifest.get("training_set")
    return [Path(training_set["path"])] if isinstance(training_set, Mapping) else []


def _write_queue(path: Path) -> None:
    rows = [
        {
            "id": "one",
            "source_lang": "ko",
            "target_lang": "ja",
            "source": "안녕하세요.",
            "translation": None,
            "status": "pending",
        },
        {
            "id": "bad-cycle",
            "source_lang": "ko",
            "target_lang": "ja",
            "source": "원래 문장입니다.",
            "translation": None,
            "status": "pending",
        },
        {
            "id": "failure",
            "source_lang": "ko",
            "target_lang": "ja",
            "source": "고장 문장입니다.",
            "translation": None,
            "status": "pending",
        },
        {
            "id": "already",
            "source_lang": "ko",
            "target_lang": "ja",
            "source": "이미 처리한 문장입니다.",
            "translation": "処理済みです。",
            "status": "accepted",
        },
        {
            "id": "two",
            "source_lang": "ko",
            "target_lang": "ja",
            "source": "정상 문장입니다.",
            "translation": None,
            "status": "pending",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _options(**updates) -> QueueTranslationOptions:
    values = {
        "batch_size": 4,
        "shard_size": 2,
        "min_roundtrip_score": 0.75,
        "min_target_language_fraction": 0.20,
        "required_target_scripts": (("ja", "kana", 1),),
    }
    values.update(updates)
    return QueueTranslationOptions(**values)


def test_queue_translation_is_resumable_audited_and_failure_isolated(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()

    partial = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        run_metadata=_run_metadata(
            translator,
            tmp_path / ".queue-runtime-artifacts",
            model="fake-v1",
        ),
        max_rows=4,
    )

    assert partial["progress"]["completed_rows"] == 4
    assert not partial["progress"]["complete"]
    assert partial["stats"] == {
        "processed": 4,
        "generated": 2,
        "accepted": 1,
        "rejected": 1,
        "errors": 1,
        "skipped_existing": 1,
    }
    first_rows = [
        json.loads(line)
        for line in (results / "part-000000.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert first_rows[0]["status"] == "accepted"
    assert first_rows[1]["status"] == "rejected"
    assert first_rows[1]["rejection_reasons"] == ["roundtrip_score"]

    translator.calls.clear()
    completed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        run_metadata=_run_metadata(
            translator,
            tmp_path / ".queue-runtime-artifacts",
            model="fake-v1",
        ),
    )

    assert completed["progress"]["complete"]
    assert completed["progress"]["completed_rows"] == 5
    assert completed["stats"]["accepted"] == 2
    replayed_inputs = {text for _source, _target, texts in translator.calls for text in texts}
    # Historical results are reproduced once at the final publication boundary.
    assert "안녕하세요." in replayed_inputs
    assert "원래 문장입니다." in replayed_inputs
    accepted_rows = [
        json.loads(line)
        for path in _published_accepted_paths(completed)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in accepted_rows] == ["one", "two"]
    assert all(row["synthetic"] is True for row in accepted_rows)
    assert all(row["training_direction"] == ["ko", "ja"] for row in accepted_rows)
    assert all(row["source_language"] == "ko" for row in accepted_rows)
    assert all(row["target_language"] == "ja" for row in accepted_rows)
    assert all(row["source"] and row["translation"] for row in accepted_rows)
    assert all(row["provenance"]["run_id"] == completed["run_id"] for row in accepted_rows)
    assert [row["provenance"]["source_index"] for row in accepted_rows] == [0, 4]
    assert all(
        row["provenance"]["source_queue"]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
        for row in accepted_rows
    )
    assert all(
        row["provenance"]["translation_model"]
        == completed["configuration"]["run_metadata"]["translation_model"]
        and row["provenance"]["tokenizer"]
        == completed["configuration"]["run_metadata"]["tokenizer"]
        and row["provenance"]["token_features"] is None
        and row["provenance"]["translation_directions"] == [["ko", "ja"], ["ja", "ko"]]
        and row["provenance"]["translation_graph_source"] == "translation_model"
        for row in accepted_rows
    )
    assert all(
        path.name.endswith(".jsonl.private") for path in _published_accepted_paths(completed)
    )
    assert _training_paths(completed) == sorted(accepted.glob("*.jsonl"))
    assert len(_training_paths(completed)) == 1
    expanded = expand_parallel_record(accepted_rows[0], (("ko", "ja"),))
    assert len(expanded.pairs) == 1
    assert expanded.pairs[0].metadata["training_direction"] == ["ko", "ja"]

    call_count = len(translator.calls)
    unchanged = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        run_metadata=_run_metadata(
            translator,
            tmp_path / ".queue-runtime-artifacts",
            model="fake-v1",
        ),
    )
    assert unchanged["stats"] == completed["stats"]
    assert len(translator.calls) == call_count


def test_transient_runtime_failure_leaves_the_shard_retryable(tmp_path: Path) -> None:
    class FailOnceTranslator(FakeTranslator):
        failed = False

        def translate(self, texts, **kwargs):
            if not self.failed:
                self.failed = True
                self.calls.append(
                    (kwargs["source_language"], kwargs["target_language"], tuple(texts))
                )
                raise RuntimeError("temporary device failure")
            return super().translate(texts, **kwargs)

    source = tmp_path / "queue.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "retryable",
                "source_lang": "ko",
                "target_lang": "ja",
                "source": "안녕하세요.",
                "translation": None,
                "status": "pending",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    translator = FailOnceTranslator()

    with pytest.raises(RetryableQueueTranslationError, match="was not committed"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )

    failed_manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert failed_manifest["progress"]["completed_rows"] == 0
    assert failed_manifest["parts"] == []
    assert not (results / "part-000000.jsonl").exists()
    assert list(accepted.glob("*.jsonl")) == []

    completed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
    )
    assert completed["progress"]["complete"] is True
    assert completed["stats"]["accepted"] == 1
    assert len(_training_paths(completed)) == 1


def test_queue_canonicalizes_bcp47_aliases_before_translation(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "bcp47",
                "source_lang": "PT-br",
                "target_lang": "ZH-hant",
                "source": "Este texto de origem possui conteúdo suficiente.",
                "translation": None,
                "status": "pending",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    translator = FakeTranslator()
    translator.translation_directions = (("pt-BR", "zh-Hant"),)
    translator.mapping[
        (
            "pt-BR",
            "zh-Hant",
            "Este texto de origem possui conteúdo suficiente.",
        )
    ] = "這段來源文本包含足夠的內容。"

    translate_queue(
        source,
        tmp_path / "results",
        translator,
        accepted_dir=tmp_path / "accepted",
        options=_options(roundtrip_enabled=False),
    )

    assert translator.calls == [
        (
            "pt-BR",
            "zh-Hant",
            ("Este texto de origem possui conteúdo suficiente.",),
        )
    ]
    result = json.loads((tmp_path / "results" / "part-000000.jsonl").read_text("utf-8"))
    assert (result["source_lang"], result["target_lang"]) == ("pt-BR", "zh-Hant")


def test_queue_rejects_missing_reverse_direction_before_model_work(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    _write_queue(source)
    translator = FakeTranslator()
    translator.translation_directions = (("ko", "ja"),)

    with pytest.raises(ValueError, match="--no-roundtrip"):
        translate_queue(
            source,
            tmp_path / "results",
            translator,
            accepted_dir=tmp_path / "accepted",
            options=_options(),
        )

    assert translator.calls == []
    assert not (tmp_path / "results" / "manifest.json").exists()


def test_queue_allows_artifact_bound_forward_only_model_when_roundtrip_is_disabled(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    _write_queue(source)
    translator = FakeTranslator()
    translator.translation_directions = (("ko", "ja"),)

    manifest = translate_queue(
        source,
        tmp_path / "results",
        translator,
        accepted_dir=tmp_path / "accepted",
        options=_options(roundtrip_enabled=False),
        max_rows=1,
    )

    assert manifest["stats"]["accepted"] == 1
    assert translator.calls == [("ko", "ja", ("안녕하세요.",))]


def test_queue_rejects_translator_without_artifact_direction_graph(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    _write_queue(source)
    translator = FakeTranslator()
    translator.translation_directions = ()

    with pytest.raises(ValueError, match="no artifact-bound translation_directions"):
        translate_queue(
            source,
            tmp_path / "results",
            translator,
            accepted_dir=tmp_path / "accepted",
            options=_options(roundtrip_enabled=False),
        )

    assert translator.calls == []


def test_queue_rejects_duplicate_ids_before_model_work(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    duplicate = {
        "id": "duplicate",
        "source_lang": "ko",
        "target_lang": "ja",
        "source": "안녕하세요.",
        "translation": None,
        "status": "pending",
    }
    source.write_text(
        "\n".join(json.dumps(duplicate, ensure_ascii=False) for _ in range(2)) + "\n",
        encoding="utf-8",
    )
    translator = FakeTranslator()

    with pytest.raises(ValueError, match="duplicate queue id 'duplicate'"):
        translate_queue(
            source,
            tmp_path / "results",
            translator,
            accepted_dir=tmp_path / "accepted",
            options=_options(),
        )

    assert translator.calls == []
    assert not (tmp_path / "results" / "manifest.json").exists()


def test_processing_stream_rejects_duplicate_ids_seeded_by_committed_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    rows = [
        {
            "id": row_id,
            "source_lang": "ko",
            "target_lang": "ja",
            "source": "안녕하세요.",
            "translation": None,
            "status": "pending",
        }
        for row_id in ("one", "two")
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    original_process = queue_translation_module._process_raw_rows
    processing_calls = 0

    def substitute_duplicate_in_second_processing_batch(raw_rows, **kwargs):
        nonlocal processing_calls
        if kwargs.get("source_index_database") is not None:
            processing_calls += 1
        if processing_calls == 2:
            duplicate = json.loads(raw_rows[0])
            duplicate["id"] = "one"
            raw_rows = [json.dumps(duplicate, ensure_ascii=False).encode() + b"\n"]
        return original_process(raw_rows, **kwargs)

    monkeypatch.setattr(
        queue_translation_module,
        "_process_raw_rows",
        substitute_duplicate_in_second_processing_batch,
    )

    with pytest.raises(ValueError, match="does not bind 'one' to source row 1"):
        translate_queue(
            source,
            results,
            FakeTranslator(),
            accepted_dir=accepted,
            options=_options(shard_size=1),
        )

    persisted = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["progress"]["next_part"] == 1
    assert len(_published_accepted_paths(persisted)) == 1
    assert not (results / "part-000001.jsonl").exists()
    assert list(accepted.glob("*.jsonl")) == []


def test_library_new_run_requires_complete_immutable_lineage(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    _write_queue(source)
    translator = FakeTranslator()

    with pytest.raises(ValueError, match="explicit immutable run_metadata"):
        _translate_queue(
            source,
            tmp_path / "missing-metadata",
            translator,
            accepted_dir=tmp_path / "missing-accepted",
            options=_options(),
            allow_unverified_translator=True,
        )

    missing_artifact = _run_metadata(translator, tmp_path / ".queue-runtime-artifacts")
    missing_artifact["translation_model"] = {
        "path": str(tmp_path / "missing-model.pt"),
        "size": 1,
        "sha256": "0" * 64,
    }
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _translate_queue(
            source,
            tmp_path / "missing-artifact",
            translator,
            accepted_dir=tmp_path / "missing-artifact-accepted",
            options=_options(),
            run_metadata=missing_artifact,
            allow_unverified_translator=True,
        )

    stale_hash = _run_metadata(translator, tmp_path / ".queue-runtime-artifacts")
    Path(str(stale_hash["translation_model"]["path"])).write_bytes(b"tampered model bytes")
    with pytest.raises(ValueError, match="does not match the current artifact bytes"):
        _translate_queue(
            source,
            tmp_path / "stale-artifact-hash",
            translator,
            accepted_dir=tmp_path / "stale-artifact-hash-accepted",
            options=_options(),
            run_metadata=stale_hash,
            allow_unverified_translator=True,
        )

    wrong_artifact = _run_metadata(translator, tmp_path / ".queue-runtime-artifacts")
    decoy_path = tmp_path / ".queue-runtime-artifacts" / "decoy-model.pt"
    decoy_path.write_bytes(b"different but real model artifact")
    wrong_artifact["translation_model"] = _identity(decoy_path)
    with pytest.raises(ValueError, match="differs from the artifact loaded by the translator"):
        _translate_queue(
            source,
            tmp_path / "wrong-runtime-binding",
            translator,
            accepted_dir=tmp_path / "wrong-runtime-binding-accepted",
            options=_options(),
            run_metadata=wrong_artifact,
            allow_unverified_translator=True,
        )

    placeholder = _run_metadata(translator, tmp_path / ".queue-runtime-artifacts")
    placeholder["source_revision"] = "unknown"
    with pytest.raises(ValueError, match="unknown placeholder"):
        _translate_queue(
            source,
            tmp_path / "placeholder-metadata",
            translator,
            accepted_dir=tmp_path / "placeholder-accepted",
            options=_options(),
            run_metadata=placeholder,
            allow_unverified_translator=True,
        )

    mutable_graph = _run_metadata(translator, tmp_path / ".queue-runtime-artifacts")
    mutable_graph["translation_graph_source"] = "tokenizer_metadata"
    translator.export_metadata = {}
    translator.tokenizer_metadata = {"translation_directions": [["ko", "ja"], ["ja", "ko"]]}
    with pytest.raises(ValueError, match="records it as absent"):
        _translate_queue(
            source,
            tmp_path / "unrecorded-sidecar",
            translator,
            accepted_dir=tmp_path / "unrecorded-sidecar-accepted",
            options=_options(),
            run_metadata=mutable_graph,
            allow_unverified_translator=True,
        )
    assert translator.calls == []

    recorded_sidecar = dict(mutable_graph)
    sidecar_path = tmp_path / ".queue-runtime-artifacts" / "tokenizer_metadata.json"
    sidecar_path.write_text(
        json.dumps(translator.tokenizer_metadata, ensure_ascii=False),
        encoding="utf-8",
    )
    translator.tokenizer_metadata_path = str(sidecar_path.resolve())
    recorded_sidecar["tokenizer_metadata"] = _identity(sidecar_path)
    manifest = _translate_queue(
        source,
        tmp_path / "recorded-sidecar",
        translator,
        accepted_dir=tmp_path / "recorded-sidecar-accepted",
        options=_options(roundtrip_enabled=False),
        run_metadata=recorded_sidecar,
        max_rows=1,
        allow_unverified_translator=True,
    )
    assert manifest["stats"]["accepted"] == 1


def test_custom_translator_requires_an_explicit_unverified_boundary(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    first_row = {
        "id": "one",
        "source_lang": "ko",
        "target_lang": "ja",
        "source": "안녕하세요.",
        "translation": None,
        "status": "pending",
    }
    source.write_text(json.dumps(first_row, ensure_ascii=False) + "\n", encoding="utf-8")
    translator = FakeTranslator()
    run_metadata = _run_metadata(translator, tmp_path / ".queue-runtime-artifacts")

    with pytest.raises(TypeError, match="allow_unverified_translator=True"):
        _translate_queue(
            source,
            tmp_path / "rejected-results",
            translator,
            accepted_dir=tmp_path / "rejected-accepted",
            options=_options(roundtrip_enabled=False),
            run_metadata=run_metadata,
        )

    manifest = _translate_queue(
        source,
        tmp_path / "explicit-results",
        translator,
        accepted_dir=tmp_path / "explicit-accepted",
        options=_options(roundtrip_enabled=False),
        run_metadata=run_metadata,
        allow_unverified_translator=True,
    )
    row = json.loads(_published_accepted_paths(manifest)[0].read_text(encoding="utf-8"))
    assert manifest["configuration"]["runtime_verification"] == "unverified_custom_translator"
    assert row["provenance"]["runtime_verification"] == "unverified_custom_translator"


def test_load_identity_binding_rejects_a_post_load_artifact_replacement(tmp_path: Path) -> None:
    translator = FakeTranslator()
    run_metadata = _run_metadata(translator, tmp_path / "artifacts")
    model_metadata = run_metadata["translation_model"]
    assert isinstance(model_metadata, Mapping)
    model_path = Path(str(model_metadata["path"]))
    loaded_identity = {
        **_identity(model_path),
        "device": model_path.stat().st_dev,
        "inode": model_path.stat().st_ino,
        "mtime_ns": model_path.stat().st_mtime_ns,
    }
    translator.translation_model_identity = loaded_identity
    model_path.write_bytes(b"replacement bytes not loaded by Translator")
    replacement = queue_translation_module._content_identity(
        {
            **_identity(model_path),
            "device": model_path.stat().st_dev,
            "inode": model_path.stat().st_ino,
            "mtime_ns": model_path.stat().st_mtime_ns,
        },
        field="translation_model",
    )

    with pytest.raises(ValueError, match="differs from the artifact identity loaded"):
        queue_translation_module._bind_translator_artifact(
            translator,
            field="translation_model",
            identity=replacement,
            verify_load_identity=True,
        )


def test_legacy_resume_rejects_an_unrecorded_graph_bearing_tokenizer_sidecar(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    current_metadata = _run_metadata(translator, tmp_path / ".queue-runtime-artifacts")
    legacy_metadata = {
        "source_dataset": "legacy/corpus",
        "source_revision": "legacy-snapshot-1",
        "source_license": "CC-BY-4.0",
        "translation_model": current_metadata["translation_model"],
        "tokenizer": current_metadata["tokenizer"],
        "token_features": None,
    }
    manifest = queue_translation_module._new_manifest(
        source=queue_translation_module._source_identity(source, None, force_hash=True),
        options=_options(roundtrip_enabled=False),
        run_metadata=legacy_metadata,
        accepted_dir=accepted,
        accepted_shard_prefix=None,
        teacher_pilot_rows=None,
    )
    _as_exact_legacy_manifest(manifest)
    results.mkdir()
    queue_translation_module._atomic_write_json(results / "manifest.json", manifest)
    translator.export_metadata = {}
    translator.tokenizer_metadata = {"translation_directions": [["ko", "ja"], ["ja", "ko"]]}

    with pytest.raises(ValueError, match="records it as absent"):
        _translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(roundtrip_enabled=False),
            run_metadata=legacy_metadata,
            allow_unverified_translator=True,
        )

    assert translator.calls == []


def test_authentic_legacy_resume_still_rejects_placeholder_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    current_metadata = _run_metadata(translator, tmp_path / ".queue-runtime-artifacts")
    legacy_metadata = {
        "source_dataset": "legacy/corpus",
        "source_revision": "unknown",
        "source_license": "CC-BY-4.0",
        "translation_model": current_metadata["translation_model"],
        "tokenizer": current_metadata["tokenizer"],
        "token_features": None,
    }
    manifest = queue_translation_module._new_manifest(
        source=queue_translation_module._source_identity(source, None, force_hash=True),
        options=_options(roundtrip_enabled=False),
        run_metadata=legacy_metadata,
        accepted_dir=accepted,
        accepted_shard_prefix=None,
        teacher_pilot_rows=None,
    )
    _as_exact_legacy_manifest(manifest)
    results.mkdir()
    queue_translation_module._atomic_write_json(results / "manifest.json", manifest)

    with pytest.raises(ValueError, match="cannot use an unknown placeholder"):
        _translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(roundtrip_enabled=False),
            run_metadata=legacy_metadata,
            allow_unverified_translator=True,
        )

    assert translator.calls == []


def test_queue_cli_requires_explicit_non_placeholder_provenance_for_new_runs(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    parsed_without_provenance = parser.parse_args(["--input", "queue.jsonl"])
    with pytest.raises(SystemExit, match="new queue runs require explicit provenance"):
        _resolve_run_metadata_seed(
            tmp_path / "new-results",
            source_dataset=parsed_without_provenance.source_dataset,
            source_revision=parsed_without_provenance.source_revision,
            source_license=parsed_without_provenance.source_license,
        )

    required = [
        "--input",
        "queue.jsonl",
        "--source-dataset",
        "local/corpus",
        "--source-revision",
        "sha256:abc123",
        "--source-license",
        "CC-BY-4.0",
    ]
    parsed = parser.parse_args(required)
    metadata, resuming = _resolve_run_metadata_seed(
        tmp_path / "new-results",
        source_dataset=parsed.source_dataset,
        source_revision=parsed.source_revision,
        source_license=parsed.source_license,
    )
    assert not resuming
    assert metadata == {
        "source_dataset": "local/corpus",
        "source_revision": "sha256:abc123",
        "source_license": "CC-BY-4.0",
    }
    placeholder = required.copy()
    placeholder[5] = "unknown"
    with pytest.raises(SystemExit):
        parser.parse_args(placeholder)


def test_legacy_cli_metadata_resumes_without_signature_drift(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    current_metadata = _run_metadata(translator, tmp_path / ".queue-runtime-artifacts")
    runtime_metadata = {
        "translation_model": current_metadata["translation_model"],
        "tokenizer": current_metadata["tokenizer"],
        "tokenizer_metadata": None,
        "token_features": None,
        "translation_directions": [["ko", "ja"], ["ja", "ko"]],
        "translation_graph_source": "translation_model",
    }
    legacy_metadata = {
        "source_dataset": "heegyu/namuwiki",
        "source_revision": "legacy-snapshot-1",
        "source_license": "CC-BY-NC-SA-2.0",
        "translation_model": runtime_metadata["translation_model"],
        "tokenizer": runtime_metadata["tokenizer"],
        "token_features": None,
    }
    partial = queue_translation_module._new_manifest(
        source=queue_translation_module._source_identity(source, None, force_hash=True),
        options=_options(),
        run_metadata=legacy_metadata,
        accepted_dir=accepted,
        accepted_shard_prefix=None,
        teacher_pilot_rows=None,
    )
    _as_exact_legacy_manifest(partial)
    legacy_signature = partial["run_signature"]
    results.mkdir()
    queue_translation_module._atomic_write_json(results / "manifest.json", partial)

    resolved, resuming = _resolve_run_metadata_seed(
        results,
        source_dataset=None,
        source_revision=None,
        source_license=None,
    )
    assert resuming
    assert resolved == legacy_metadata
    _validate_resume_runtime_metadata(
        resolved,
        runtime_metadata,
    )
    with pytest.raises(SystemExit, match="cannot change recorded source_dataset"):
        _resolve_run_metadata_seed(
            results,
            source_dataset="different/corpus",
            source_revision=None,
            source_license=None,
        )

    completed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        run_metadata=resolved,
    )

    assert completed["run_signature"] != legacy_signature
    assert completed["configuration"]["run_metadata"] == {
        **legacy_metadata,
        "tokenizer_metadata": None,
        "translation_directions": [["ko", "ja"], ["ja", "ko"]],
        "translation_graph_source": "translation_model",
    }
    tampered = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    tampered["configuration"]["run_metadata"]["source_revision"] = "forged"
    (results / "manifest.json").write_text(
        json.dumps(tampered, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="configuration digest is invalid"):
        _resolve_run_metadata_seed(
            results,
            source_dataset=None,
            source_revision=None,
            source_license=None,
        )


def test_queue_resume_rejects_quality_or_model_changes(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        run_metadata=_run_metadata(
            translator,
            tmp_path / ".queue-runtime-artifacts",
            model="fake-v1",
        ),
        max_rows=1,
    )

    with pytest.raises(ValueError, match="resume configuration changed"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(min_roundtrip_score=0.10),
            run_metadata=_run_metadata(
                translator,
                tmp_path / ".queue-runtime-artifacts",
                model="fake-v1",
            ),
        )
    with pytest.raises(ValueError, match="resume configuration changed"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
            run_metadata=_run_metadata(
                translator,
                tmp_path / ".queue-runtime-artifacts",
                model="fake-v2",
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("batch_size", True),
        ("shard_size", 0),
        ("num_beams", 0),
        ("max_new_tokens", 0),
        ("max_output_length_ratio", 0),
        ("max_output_length_margin", -1),
        ("max_output_length_margin", False),
        ("min_roundtrip_score", 1.1),
        ("min_pair_score", 101),
        ("min_pair_score", True),
        ("min_target_language_fraction", -0.1),
        ("required_target_scripts", (("ja", "kana", 0),)),
        ("required_target_scripts", (("ja", "kana", True),)),
    ],
)
def test_queue_options_validate(field: str, value: object) -> None:
    options = _options(**{field: value})
    with pytest.raises(ValueError):
        options.validate()


def test_target_script_requirements_support_arbitrary_bcp47_languages() -> None:
    options = _options(
        required_target_scripts=(
            ("sr-Cyrl", "cyrillic", 3),
            ("az-Arab", "arabic", 2),
        )
    )

    options.validate()
    assert options.target_script_requirements("sr-Cyrl") == {"cyrillic": 3}
    assert options.target_script_requirements("az-Arab") == {"arabic": 2}
    assert options.target_script_requirements("en") == {}


def test_cli_parses_repeatable_target_script_requirements() -> None:
    args = build_parser().parse_args(
        [
            "--input",
            "queue.jsonl",
            "--require-target-script",
            "SR-cyrl=CYRILLIC:3",
            "--require-target-script",
            "az-Arab=arabic:2",
        ]
    )

    assert args.require_target_script == [
        ("sr-Cyrl", "cyrillic", 3),
        ("az-Arab", "arabic", 2),
    ]


def test_teacher_pilot_requires_review_before_approval(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    _write_queue(source)
    translator = FakeTranslator()

    with pytest.raises(ValueError, match="cannot be approved before"):
        translate_queue(
            source,
            tmp_path / "fresh-results",
            translator,
            accepted_dir=tmp_path / "fresh-accepted",
            options=_options(),
            teacher_pilot_rows=2,
            approve_teacher=True,
            approval_actor="reviewer",
        )

    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    pilot = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        teacher_pilot_rows=2,
    )
    assert pilot["progress"]["completed_rows"] == 2
    assert pilot["stats"]["generated"] == 2
    assert pilot["teacher_review"] == {
        "pilot_rows": 2,
        "review_required": True,
        "approved": False,
        "approved_at": None,
        "approved_by": None,
    }
    assert list(accepted.glob("*.jsonl")) == []

    with pytest.raises(ValueError, match="pilot size is fixed"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
            teacher_pilot_rows=3,
        )

    completed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        teacher_pilot_rows=2,
        approve_teacher=True,
        approval_actor="reviewer",
    )
    assert completed["progress"]["complete"]
    assert completed["teacher_review"]["approved"]
    assert completed["teacher_review"]["approved_by"] == "reviewer"
    assert completed["teacher_review"]["approved_at"]
    assert _training_paths(completed) == sorted(accepted.glob("*.jsonl"))
    assert len(_training_paths(completed)) == 1
    assert completed["training_set"]["rows"] == sum(
        part["accepted"]["rows"] for part in completed["parts"]
    )


def test_manifest_rejects_truthy_string_completion_before_resume(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    manifest_path = results / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["progress"]["complete"] = "yes"
    queue_translation_module._atomic_write_json(manifest_path, manifest)
    translator.calls.clear()

    with pytest.raises(ValueError, match=r"progress\.complete must be a boolean"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )
    assert translator.calls == []


def test_manifest_rejects_truthy_string_teacher_approval(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        teacher_pilot_rows=1,
    )
    manifest_path = results / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["teacher_review"]["approved"] = "yes"
    queue_translation_module._atomic_write_json(manifest_path, manifest)
    translator.calls.clear()

    with pytest.raises(ValueError, match=r"teacher_review\.approved must be a boolean"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
            teacher_pilot_rows=1,
        )
    assert translator.calls == []
    assert list(accepted.glob("*.jsonl")) == []


def test_existing_manifest_can_adopt_a_frozen_teacher_review_policy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    legacy = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=2,
    )
    legacy.pop("teacher_review")
    legacy["stats"].pop("generated")
    (results / "manifest.json").write_text(
        json.dumps(legacy, ensure_ascii=False),
        encoding="utf-8",
    )

    resumed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        teacher_pilot_rows=2,
        approve_teacher=True,
        approval_actor="migration-reviewer",
    )

    assert resumed["teacher_review"]["approved"]
    assert resumed["teacher_review"]["pilot_rows"] == 2
    assert resumed["stats"]["generated"] >= 2


def test_queue_resume_ignores_mtime_but_records_content_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    partial = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    original_signature = partial["run_signature"]
    artifact = partial["parts"][0]
    assert artifact["published"] is True
    assert artifact["result"]["rows"] == 1
    assert len(artifact["result"]["sha256"]) == 64
    assert artifact["accepted"]["rows"] == 1
    assert artifact["status_counts"]["accepted"] == 1
    assert artifact["generated_rows"] == 1

    stat = source.stat()
    os.utime(
        source,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
    )
    completed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
    )

    assert completed["progress"]["complete"]
    assert completed["run_signature"] == original_signature
    assert completed["signature_version"] == 2
    assert completed["configuration"]["source"]["mtime_ns"] == source.stat().st_mtime_ns


def test_completed_queue_rehashes_source_even_if_size_and_mtime_match(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
    )
    stat = source.stat()
    original = source.read_bytes()
    tampered = original.replace(
        "안녕하세요".encode(),
        "반갑습니다".encode(),
        1,
    )
    assert len(tampered) == len(original)
    source.write_bytes(tampered)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    with pytest.raises(ValueError, match="input source content changed"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )


def test_queue_resume_rejects_corrupted_committed_shard(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    corrupted_result_path = results / "part-000000.jsonl"
    _make_writable(corrupted_result_path)
    with corrupted_result_path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ValueError, match="integrity mismatch"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )


def test_queue_resume_rejects_missing_committed_shard(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    missing_path = Path(manifest["parts"][0]["accepted"]["path"])
    _make_writable(missing_path)
    missing_path.unlink()

    with pytest.raises(FileNotFoundError, match="accepted part"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )


def test_legacy_manifest_signature_and_shards_migrate_on_resume(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    recorded_metadata = _run_metadata(
        translator,
        tmp_path / ".queue-runtime-artifacts",
    )
    legacy_metadata = {
        "source_dataset": "legacy/corpus",
        "source_revision": "legacy-snapshot-1",
        "source_license": "CC-BY-4.0",
        "translation_model": recorded_metadata["translation_model"],
        "tokenizer": recorded_metadata["tokenizer"],
        "token_features": None,
    }
    initial_metadata = {
        **legacy_metadata,
        "source_revision": "legacy-snapshot-1",
        "tokenizer_metadata": None,
        "translation_directions": [["ko", "ja"], ["ja", "ko"]],
        "translation_graph_source": "translation_model",
    }
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        run_metadata=initial_metadata,
        max_rows=1,
    )
    manifest["configuration"].pop("accepted_shard_prefix")
    manifest["configuration"].pop("source_snapshot")
    manifest["configuration"].pop("source_index")
    manifest["configuration"].pop("runtime_verification")
    manifest["configuration"]["pipeline_version"] = 1
    manifest["configuration"]["run_metadata"] = legacy_metadata
    configuration_bytes = json.dumps(
        manifest["configuration"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    legacy_signature = hashlib.sha256(configuration_bytes).hexdigest()
    old_accepted_path = Path(manifest["parts"][0]["accepted"]["path"])
    _make_writable(old_accepted_path)
    legacy_run_id = legacy_signature[:16]
    legacy_accepted_path = accepted / f"bt_{source.stem}_{legacy_run_id}_000000.jsonl"
    result_path = results / "part-000000.jsonl"
    _make_writable(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["run_id"] = legacy_run_id
    result_path.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    accepted_row = json.loads(old_accepted_path.read_text(encoding="utf-8"))
    old_shape = {
        "KO": accepted_row["source"],
        "JA": accepted_row["translation"],
        "id": accepted_row["id"],
        "synthetic": True,
        "provenance": {
            "type": "machine_translation",
            "queue_id": accepted_row["id"],
            "run_id": legacy_run_id,
            "roundtrip_score": accepted_row["provenance"]["roundtrip_score"],
        },
    }
    old_accepted_path.write_text(
        json.dumps(old_shape, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    old_accepted_path.rename(legacy_accepted_path)
    manifest["run_id"] = legacy_run_id
    manifest["run_signature"] = legacy_signature
    manifest.pop("signature_version")
    manifest.pop("parts")
    manifest.pop("integrity")
    manifest.pop("training_set")
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    migrated_partial = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        run_metadata=legacy_metadata,
        max_rows=1,
    )

    assert not migrated_partial["progress"]["complete"]
    assert not legacy_accepted_path.exists()
    assert list(accepted.glob("*.jsonl")) == []

    migrated = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        run_metadata=migrated_partial["configuration"]["run_metadata"],
    )

    assert migrated["progress"]["complete"]
    assert migrated["run_id"] == legacy_run_id
    assert migrated["run_signature"] != legacy_signature
    assert migrated["signature_version"] == 2
    assert len(migrated["parts"]) == migrated["progress"]["next_part"]
    migrated_accepted_path = Path(migrated["parts"][0]["accepted"]["path"])
    assert migrated_accepted_path.name.endswith(".jsonl.private")
    assert not legacy_accepted_path.exists()
    migrated_row = json.loads(migrated_accepted_path.read_text(encoding="utf-8"))
    assert migrated_row["source_language"] == "ko"
    assert migrated_row["target_language"] == "ja"
    assert migrated_row["training_direction"] == ["ko", "ja"]
    assert migrated_row["provenance"]["source_dataset"] == "legacy/corpus"
    assert migrated_row["provenance"]["source_revision"] == "legacy-snapshot-1"
    assert migrated_row["provenance"]["translation_model"] == recorded_metadata["translation_model"]
    assert migrated_row["provenance"]["tokenizer"] == recorded_metadata["tokenizer"]
    assert migrated_row["provenance"]["translation_directions"] == [
        ["ko", "ja"],
        ["ja", "ko"],
    ]
    assert migrated_row["provenance"]["translation_graph_source"] == "translation_model"
    assert (
        migrated_row["provenance"]["source_queue"]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert list(accepted.glob("bt_*.jsonl")) == []
    assert list(accepted.glob("queue_bt_*.jsonl")) == _training_paths(migrated)


def test_legacy_accepted_row_rejects_text_that_disagrees_with_result(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    manifest = translate_queue(
        source,
        results,
        FakeTranslator(),
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    accepted_path = Path(manifest["parts"][0]["accepted"]["path"])
    _make_writable(accepted_path)
    current = json.loads(accepted_path.read_text(encoding="utf-8"))
    old_shape = {
        "KO": current["source"],
        "JA": "結果シャードと一致しない改ざん文です。",
        "id": current["id"],
        "synthetic": True,
        "provenance": {
            key: value
            for key, value in current["provenance"].items()
            if key in {"type", "queue_id", "run_id", "roundtrip_score"}
        },
    }
    accepted_path.write_text(
        json.dumps(old_shape, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["parts"][0]["accepted"] = queue_translation_module._jsonl_artifact(accepted_path)
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    resumed_translator = FakeTranslator()

    with pytest.raises(ValueError, match="source/translation mismatch"):
        translate_queue(
            source,
            results,
            resumed_translator,
            accepted_dir=accepted,
            options=_options(),
        )

    assert resumed_translator.calls == []


def test_accepted_migration_recovers_after_rewrite_before_manifest_commit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    manifest = translate_queue(
        source,
        results,
        FakeTranslator(),
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    accepted_path = Path(manifest["parts"][0]["accepted"]["path"])
    _make_writable(accepted_path)
    canonical_row = json.loads(accepted_path.read_text(encoding="utf-8"))
    old_shape = {
        "ko": canonical_row["source"],
        "ja": canonical_row["translation"],
        "id": canonical_row["id"],
        "synthetic": True,
        "provenance": {
            key: value
            for key, value in canonical_row["provenance"].items()
            if key in {"type", "queue_id", "run_id", "roundtrip_score"}
        },
    }
    accepted_path.write_text(
        json.dumps(old_shape, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["parts"][0]["accepted"] = queue_translation_module._jsonl_artifact(accepted_path)
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    # Simulate the process stopping after the atomic accepted-row rewrite but
    # before the updated artifact digest reaches manifest.json.
    _make_writable(accepted_path)
    accepted_path.write_text(
        json.dumps(canonical_row, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    recovered = translate_queue(
        source,
        results,
        FakeTranslator(),
        accepted_dir=accepted,
        options=_options(),
    )

    assert recovered["parts"][0]["accepted"] == queue_translation_module._jsonl_artifact(
        accepted_path
    )
    assert json.loads(accepted_path.read_text(encoding="utf-8")) == canonical_row


@pytest.mark.parametrize("parts_as_null", [False, True])
def test_current_manifest_cannot_drop_parts_to_rebaseline_target_tampering(
    tmp_path: Path,
    parts_as_null: bool,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    calls_before_resume = list(translator.calls)
    result_path = results / "part-000000.jsonl"
    _make_writable(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["translation"] = "poisoned training target"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    accepted_path = Path(manifest["parts"][0]["accepted"]["path"])
    _make_writable(accepted_path)
    accepted_row = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted_row["translation"] = result["translation"]
    accepted_path.write_text(
        json.dumps(accepted_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if parts_as_null:
        manifest["parts"] = None
    else:
        manifest.pop("parts")
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refusing legacy downgrade"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )
    assert translator.calls == calls_before_resume


def test_stripped_current_manifest_cannot_impersonate_the_exact_legacy_schema(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    calls_before_resume = list(translator.calls)
    for field in ("signature_version", "parts", "integrity", "training_set"):
        manifest.pop(field)
    configuration = manifest["configuration"]
    for field in (
        "source_snapshot",
        "source_index",
        "runtime_verification",
        "accepted_shard_prefix",
    ):
        configuration.pop(field)
    forged_signature = queue_translation_module._stable_digest(configuration)
    manifest["run_signature"] = forged_signature
    manifest["run_id"] = forged_signature[:16]
    queue_translation_module._atomic_write_json(results / "manifest.json", manifest)

    with pytest.raises(ValueError, match="refusing legacy downgrade"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )

    assert translator.calls == calls_before_resume
    assert list(accepted.glob("*.jsonl")) == []


def test_public_training_copy_rejects_manifest_attested_poisoned_target(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    result_path = results / "part-000000.jsonl"
    _make_writable(result_path)
    accepted_path = Path(manifest["parts"][0]["accepted"]["path"])
    _make_writable(accepted_path)
    training_path = Path(manifest["parts"][0]["training"]["path"])
    original_training_bytes = training_path.read_bytes()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    accepted_row = json.loads(accepted_path.read_text(encoding="utf-8"))
    result["translation"] = "POISONED TARGET THAT THE MODEL NEVER GENERATED"
    accepted_row["translation"] = result["translation"]
    result_path.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    accepted_path.write_text(
        json.dumps(accepted_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["parts"][0]["result"] = queue_translation_module._jsonl_artifact(result_path)
    manifest["parts"][0]["accepted"] = queue_translation_module._jsonl_artifact(accepted_path)
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    calls_before_resume = len(translator.calls)

    with pytest.raises(ValueError, match="private training stage 000000 failed verification"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )

    assert len(translator.calls) == calls_before_resume
    assert training_path.read_bytes() == original_training_bytes


def test_current_multi_part_manifest_cannot_drop_its_committed_ledger(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(shard_size=1),
        max_rows=2,
    )
    calls_before_resume = list(translator.calls)
    second_result_path = results / "part-000001.jsonl"
    _make_writable(second_result_path)
    second_result = json.loads(second_result_path.read_text(encoding="utf-8"))
    second_result["id"] = "one"
    second_result_path.write_text(
        json.dumps(second_result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest.pop("parts")
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refusing legacy downgrade"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(shard_size=1),
        )

    assert translator.calls == calls_before_resume


def test_current_manifest_missing_parts_fails_before_source_substitution_migration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    calls_before_resume = list(translator.calls)
    result_path = results / "part-000000.jsonl"
    _make_writable(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["id"] = "substituted"
    result["source"] = "바꿔치기한 원문입니다."
    result_path.write_text(
        json.dumps(result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    accepted_path = Path(manifest["parts"][0]["accepted"]["path"])
    _make_writable(accepted_path)
    accepted_row = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted_row["id"] = "substituted"
    accepted_row["source"] = result["source"]
    accepted_row["provenance"]["queue_id"] = "substituted"
    accepted_path.write_text(
        json.dumps(accepted_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest.pop("parts")
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refusing legacy downgrade"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )

    assert translator.calls == calls_before_resume


def test_current_manifest_missing_parts_fails_before_status_migration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(shard_size=4),
        max_rows=4,
    )
    calls_before_resume = list(translator.calls)
    result_path = results / "part-000000.jsonl"
    _make_writable(result_path)
    result_rows = [
        json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result_rows[3]["status"] == "skipped_existing"
    result_rows[3]["status"] = "accepted"
    result_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in result_rows) + "\n",
        encoding="utf-8",
    )
    manifest.pop("parts")
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refusing legacy downgrade"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(shard_size=4),
        )

    assert translator.calls == calls_before_resume


def test_manifest_cannot_mark_an_uncommitted_source_tail_complete(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(shard_size=1),
        max_rows=1,
    )
    manifest_path = results / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["progress"]["complete"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unconsumed source tail"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(shard_size=1),
        )


def test_private_source_snapshot_isolates_processing_from_later_input_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    original_bytes = source.read_bytes()
    translator = FakeTranslator()
    run_metadata = _run_metadata(translator, tmp_path / ".queue-runtime-artifacts")
    original_validate = queue_translation_module._validated_run_lineage

    def mutate_after_initial_source_hash(*args, **kwargs):
        lineage = original_validate(*args, **kwargs)
        rows = source.read_text(encoding="utf-8").splitlines()
        first = json.loads(rows[0])
        first["source"] = "정상 문장입니다."
        rows[0] = json.dumps(first, ensure_ascii=False)
        source.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return lineage

    monkeypatch.setattr(
        queue_translation_module,
        "_validated_run_lineage",
        mutate_after_initial_source_hash,
    )

    manifest = _translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        run_metadata=run_metadata,
        max_rows=1,
        allow_unverified_translator=True,
    )

    assert source.read_bytes() != original_bytes
    assert translator.calls
    assert manifest["progress"]["next_part"] == 1
    result = json.loads((results / "part-000000.jsonl").read_text(encoding="utf-8"))
    assert result["source"] == "안녕하세요."
    assert (
        manifest["configuration"]["source"]["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    )
    assert _published_accepted_paths(manifest)[0].is_file()
    assert list(accepted.glob("*.jsonl")) == []


def test_new_run_recovers_queue_owned_snapshot_and_manifest_temp_files(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    _write_queue(source)
    results.mkdir()
    stale_snapshot_temp = results / (
        f".{queue_translation_module.SOURCE_SNAPSHOT_FILENAME}.999999.tmp"
    )
    stale_manifest_temp = results / ".manifest.json.999999.tmp"
    stale_snapshot_temp.write_bytes(b"partial snapshot")
    stale_manifest_temp.write_bytes(b"partial manifest")

    manifest = translate_queue(
        source,
        results,
        FakeTranslator(),
        accepted_dir=tmp_path / "accepted",
        options=_options(),
        max_rows=1,
    )

    assert manifest["progress"]["next_part"] == 1
    assert not stale_snapshot_temp.exists()
    assert not stale_manifest_temp.exists()


def test_queue_output_allows_only_one_writer(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    _write_queue(source)

    with _queue_run_lock(results):
        with pytest.raises(RuntimeError, match="already being translated"):
            translate_queue(
                source,
                results,
                FakeTranslator(),
                accepted_dir=tmp_path / "accepted",
                options=_options(),
            )


def test_queue_lock_excludes_a_second_process(tmp_path: Path) -> None:
    results = tmp_path / "results"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_queue_lock,
        args=(str(results), ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=10)
        with pytest.raises(RuntimeError, match="already being translated"):
            with _queue_run_lock(results):
                pass
    finally:
        release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    assert process.exitcode == 0


def test_shared_accepted_namespace_allows_only_one_publisher(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    accepted = tmp_path / "accepted"
    _write_queue(source)

    with _accepted_run_lock(accepted):
        with pytest.raises(RuntimeError, match="accepted queue namespace"):
            translate_queue(
                source,
                tmp_path / "other-results",
                FakeTranslator(),
                accepted_dir=accepted,
                options=_options(),
            )


def test_distinct_outputs_publish_to_distinct_private_run_namespaces(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    first = translate_queue(
        source,
        tmp_path / "first-results",
        FakeTranslator(),
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )

    second = translate_queue(
        source,
        tmp_path / "second-results",
        FakeTranslator(),
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )

    first_path = _published_accepted_paths(first)[0]
    second_path = _published_accepted_paths(second)[0]
    assert first_path != second_path
    assert first_path.is_file() and second_path.is_file()
    assert list(accepted.glob("*.jsonl")) == []


def test_committed_pending_accepted_shard_is_recovered(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    accepted_path = Path(manifest["parts"][0]["accepted"]["path"])
    pending_path = accepted_path.with_name(f".{accepted_path.name}.pending")
    accepted_path.replace(pending_path)

    completed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
    )

    assert completed["progress"]["complete"]
    assert accepted_path.is_file()
    assert not pending_path.exists()


def test_published_final_wins_over_stale_pending(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    accepted_path = Path(manifest["parts"][0]["accepted"]["path"])
    pending_path = accepted_path.with_name(f".{accepted_path.name}.pending")
    pending_path.write_text("{}\n", encoding="utf-8")

    completed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
    )

    assert completed["progress"]["complete"]
    assert accepted_path.is_file()
    assert not pending_path.exists()


def test_published_manifest_quarantines_an_unrecoverable_pending_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    accepted_path = Path(manifest["parts"][0]["accepted"]["path"])
    pending_path = accepted_path.with_name(f".{accepted_path.name}.pending")
    pending_path.write_bytes(b"foreign\n")
    _make_writable(accepted_path)
    accepted_path.unlink()

    with pytest.raises(ValueError, match="were quarantined"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )

    assert not accepted_path.exists()
    assert pending_path.exists()
    persisted = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["parts"][0]["published"] is False


def test_accepted_shard_is_not_published_before_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    original_write = queue_translation_module._atomic_write_json
    writes = 0

    def fail_part_commit(path: Path, value) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated manifest failure")
        original_write(path, value)

    monkeypatch.setattr(
        queue_translation_module,
        "_atomic_write_json",
        fail_part_commit,
    )
    with pytest.raises(OSError, match="simulated manifest failure"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
            max_rows=1,
        )

    disk_manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest["progress"]["next_part"] == 0
    assert list(accepted.glob("*.jsonl")) == []
    assert len(list(accepted.rglob("*.pending"))) == 1

    monkeypatch.setattr(
        queue_translation_module,
        "_atomic_write_json",
        original_write,
    )
    resumed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    assert resumed["progress"]["next_part"] == 1
    assert len(_published_accepted_paths(resumed)) == 1
    assert _published_accepted_paths(resumed)[0].is_file()
    assert list(accepted.glob("*.jsonl")) == []
    assert list(accepted.rglob("*.pending")) == []


def test_atomic_json_write_ignores_the_old_predictable_hardlink_name(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("preserve me\n", encoding="utf-8")
    target = tmp_path / "manifest.json"
    predictable = tmp_path / f".manifest.json.{os.getpid()}.tmp"
    os.link(victim, predictable)

    queue_translation_module._atomic_write_json(target, {"safe": True})

    assert victim.read_text(encoding="utf-8") == "preserve me\n"
    assert predictable.read_text(encoding="utf-8") == "preserve me\n"
    assert json.loads(target.read_text(encoding="utf-8")) == {"safe": True}


def test_training_materialization_never_overwrites_a_public_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    original_copy = queue_translation_module._copy_plain_file_no_replace
    collided: Path | None = None

    def collide_then_copy(private_path: Path, public_path: Path, **kwargs) -> None:
        nonlocal collided
        collided = public_path
        public_path.write_text("foreign training data\n", encoding="utf-8")
        original_copy(private_path, public_path, **kwargs)

    monkeypatch.setattr(
        queue_translation_module,
        "_copy_plain_file_no_replace",
        collide_then_copy,
    )
    with pytest.raises(FileExistsError):
        translate_queue(
            source,
            results,
            FakeTranslator(),
            accepted_dir=accepted,
            options=_options(),
            max_rows=1,
        )

    assert collided is not None
    assert collided.read_text(encoding="utf-8") == "foreign training data\n"
    persisted = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["parts"][0]["published"] is True
    assert "training" not in persisted["parts"][0]


def test_committed_part_is_not_glob_visible_when_final_publish_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    first_row = source.read_text(encoding="utf-8").splitlines()[0]
    source.write_text(first_row + "\n", encoding="utf-8")
    translator = FakeTranslator()
    original_publish = queue_translation_module._publish_no_replace

    def fail_shard_publish(pending_path: Path, accepted_path: Path) -> None:
        if accepted_path.name.endswith(".jsonl.private"):
            raise OSError("simulated final-shard publish failure")
        original_publish(pending_path, accepted_path)

    monkeypatch.setattr(
        queue_translation_module,
        "_publish_no_replace",
        fail_shard_publish,
    )
    with pytest.raises(OSError, match="final-shard publish failure"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
            max_rows=1,
        )

    disk_manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest["progress"]["next_part"] == 1
    assert disk_manifest["progress"]["complete"] is True
    assert disk_manifest["parts"][0]["published"] is False
    assert list(accepted.glob("*.jsonl")) == []
    assert len(list(accepted.rglob("*.pending"))) == 1

    monkeypatch.setattr(
        queue_translation_module,
        "_publish_no_replace",
        original_publish,
    )
    recovered = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    assert recovered["parts"][0]["published"] is True
    assert recovered["progress"]["complete"] is True
    persisted = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["parts"][0]["published"] is True
    assert persisted["progress"]["complete"] is True


def test_atomic_publish_never_overwrites_external_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    original_link = queue_translation_module.os.link

    def create_foreign_target_then_link(source_path, target_path) -> None:
        if str(target_path).endswith(".jsonl.private"):
            Path(target_path).write_text("foreign\n", encoding="utf-8")
        original_link(source_path, target_path)

    monkeypatch.setattr(
        queue_translation_module.os,
        "link",
        create_foreign_target_then_link,
    )
    with pytest.raises(FileExistsError):
        translate_queue(
            source,
            results,
            FakeTranslator(),
            accepted_dir=accepted,
            options=_options(),
            max_rows=1,
        )

    disk_manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    accepted_path = Path(disk_manifest["parts"][0]["accepted"]["path"])
    assert accepted_path.read_text(encoding="utf-8") == "foreign\n"
    assert disk_manifest["parts"][0]["published"] is False
    assert disk_manifest["progress"]["complete"] is False
    assert list(accepted.glob("*.jsonl")) == []
    assert accepted_path.parent.parent.name == ".queue-runs"


def test_manifest_run_id_cannot_escape_the_private_namespace(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    manifest_path = results / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = "../../outside"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="run_id must be 16 lowercase hexadecimal"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )

    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("accepted_name", ("results", "results/accepted"))
def test_accepted_namespace_cannot_overlap_audit_outputs(
    tmp_path: Path,
    accepted_name: str,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    _write_queue(source)

    with pytest.raises(ValueError, match="separate, non-nested directories"):
        translate_queue(
            source,
            results,
            FakeTranslator(),
            accepted_dir=tmp_path / accepted_name,
            options=_options(),
        )

    assert not results.exists()


def test_snapshot_swap_before_publication_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    original_publish = queue_translation_module._publish_no_replace

    def publish_with_swapped_snapshot(source_path: Path, target_path: Path) -> None:
        if Path(target_path).name == queue_translation_module.SOURCE_SNAPSHOT_FILENAME:
            Path(source_path).write_text('{"id":"swapped"}\n', encoding="utf-8")
        original_publish(source_path, target_path)

    monkeypatch.setattr(
        queue_translation_module,
        "_publish_no_replace",
        publish_with_swapped_snapshot,
    )
    with pytest.raises(ValueError, match="published private queue source snapshot"):
        translate_queue(
            source,
            results,
            FakeTranslator(),
            accepted_dir=accepted,
            options=_options(),
        )

    assert not (results / "manifest.json").exists()


def test_preexisting_source_snapshot_cannot_be_a_hard_link(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    results.mkdir()
    snapshot = results / queue_translation_module.SOURCE_SNAPSHOT_FILENAME
    os.link(source, snapshot)

    with pytest.raises(ValueError, match="must not have hard-link aliases"):
        translate_queue(
            source,
            results,
            FakeTranslator(),
            accepted_dir=accepted,
            options=_options(),
        )

    assert not (results / "manifest.json").exists()


def test_private_accepted_namespace_rejects_a_directory_reparse_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    original_lstat = queue_translation_module.os.lstat

    class ReparseMetadata:
        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped
            self.st_file_attributes = getattr(wrapped, "st_file_attributes", 0) | 0x400

        def __getattr__(self, name: str):
            return getattr(self._wrapped, name)

    def mark_private_root_as_reparse(path):
        observed = original_lstat(path)
        if Path(path).name == queue_translation_module.PRIVATE_ACCEPTED_DIRNAME:
            return ReparseMetadata(observed)
        return observed

    monkeypatch.setattr(queue_translation_module.os, "lstat", mark_private_root_as_reparse)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        translate_queue(
            source,
            results,
            FakeTranslator(),
            accepted_dir=accepted,
            options=_options(),
            max_rows=1,
        )


def test_owner_claim_never_overwrites_external_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    original_link = queue_translation_module.os.link

    def create_foreign_owner_then_link(source_path, target_path) -> None:
        if str(target_path).endswith(".owner.json"):
            Path(target_path).write_text("{}\n", encoding="utf-8")
        original_link(source_path, target_path)

    monkeypatch.setattr(
        queue_translation_module.os,
        "link",
        create_foreign_owner_then_link,
    )
    with pytest.raises(FileExistsError, match="already owned"):
        translate_queue(
            source,
            results,
            FakeTranslator(),
            accepted_dir=accepted,
            options=_options(),
            max_rows=1,
        )

    owner_files = list(accepted.glob("*.owner.json"))
    assert len(owner_files) == 1
    assert owner_files[0].read_text(encoding="utf-8") == "{}\n"


def test_queue_default_roundtrip_threshold_is_conservative() -> None:
    assert QueueTranslationOptions().min_roundtrip_score == 0.65


def test_han_only_candidate_cannot_pass_as_japanese(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "chinese-output",
                "source_lang": "ko",
                "target_lang": "ja",
                "source": "안녕하세요",
                "translation": None,
                "status": "pending",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    translator = FakeTranslator()
    translator.mapping[("ko", "ja", "안녕하세요")] = "你好世界"
    translator.mapping[("ja", "ko", "你好世界")] = "안녕하세요"

    manifest = translate_queue(
        source,
        tmp_path / "results",
        translator,
        accepted_dir=tmp_path / "accepted",
        options=_options(required_target_scripts=(("ja", "kana", 1),)),
    )

    assert manifest["stats"]["accepted"] == 0
    result = json.loads((tmp_path / "results" / "part-000000.jsonl").read_text(encoding="utf-8"))
    assert result["quality"]["forward"]["target_language_fraction"] == 1.0
    assert result["quality"]["forward"]["target_script_characters"] == {"han": 4}
    assert result["rejection_reasons"] == ["target_script:kana"]


def test_manifest_rejects_forged_prefix_even_with_a_recomputed_digest(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    manifest_path = results / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["accepted_shard_prefix"] = "queue_bt_shadow_"
    forged_signature = queue_translation_module._stable_digest(
        queue_translation_module._signature_configuration(manifest["configuration"])
    )
    manifest["run_signature"] = forged_signature
    manifest["run_id"] = forged_signature[:16]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    translator.calls.clear()

    with pytest.raises(ValueError, match="accepted_shard_prefix is not allowed"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )

    assert translator.calls == []
    assert list(accepted.glob("*.jsonl")) == []


def test_manifest_rejects_a_valid_but_signature_unbound_run_id(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    manifest_path = results / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = "0" * 16 if manifest["run_id"] != "0" * 16 else "1" * 16
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="run_id does not match run_signature"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )

    assert list(accepted.glob("*.jsonl")) == []


def test_teacher_approval_replays_and_rejects_manifest_attested_pilot_poison(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    pilot = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        teacher_pilot_rows=2,
    )
    result_path = results / "part-000000.jsonl"
    accepted_path = Path(pilot["parts"][0]["accepted"]["path"])
    _make_writable(result_path)
    _make_writable(accepted_path)
    result_rows = [
        json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()
    ]
    accepted_rows = [
        json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines()
    ]
    result_rows[0]["translation"] = "承認前に差し替えた文です。"
    accepted_rows[0]["translation"] = result_rows[0]["translation"]
    queue_translation_module._atomic_write_jsonl(result_path, result_rows)
    queue_translation_module._atomic_write_jsonl(accepted_path, accepted_rows)
    pilot["parts"][0]["result"] = queue_translation_module._jsonl_artifact(result_path)
    pilot["parts"][0]["accepted"] = queue_translation_module._jsonl_artifact(accepted_path)
    queue_translation_module._atomic_write_json(results / "manifest.json", pilot)
    translator.calls.clear()

    with pytest.raises(ValueError, match="cannot be reproduced"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
            teacher_pilot_rows=2,
            approve_teacher=True,
            approval_actor="reviewer",
        )

    persisted = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["teacher_review"]["approved"] is False
    assert list(accepted.glob("*.jsonl")) == []


def test_training_copy_removes_a_raced_hard_link_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_path = tmp_path / "accepted.jsonl.private"
    target_path = tmp_path / "training.jsonl.private"
    alias_path = tmp_path / "raced-alias.jsonl"
    private_path.write_text('{"safe":true}\n', encoding="utf-8", newline="\n")
    expected = queue_translation_module._jsonl_artifact(private_path)
    original_publish = queue_translation_module._publish_no_replace

    def publish_then_add_alias(pending_path: Path, published_path: Path) -> None:
        original_publish(pending_path, published_path)
        os.link(published_path, alias_path)

    monkeypatch.setattr(
        queue_translation_module,
        "_publish_no_replace",
        publish_then_add_alias,
    )

    with pytest.raises(ValueError, match="unsafe filesystem identity"):
        queue_translation_module._copy_plain_file_no_replace(
            private_path,
            target_path,
            expected=expected,
        )

    assert not target_path.exists()
    assert alias_path.read_bytes() == private_path.read_bytes()


def test_partial_resume_uses_runtime_artifacts_and_the_sqlite_id_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    rows = [
        {
            "id": f"row-{index}",
            "source_lang": "ko",
            "target_lang": "ja",
            "source": "안녕하세요.",
            "translation": None,
            "status": "pending",
        }
        for index in range(8)
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    translator = FakeTranslator()
    partial = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(shard_size=1),
        max_rows=4,
    )
    assert partial["progress"]["next_part"] == 4
    index_path = Path(partial["configuration"]["source_index"]["path"])
    with sqlite3.connect(index_path) as database:
        assert database.execute("SELECT COUNT(*) FROM queue_ids").fetchone() == (8,)

    original_artifact = queue_translation_module._jsonl_artifact
    hashed_paths: list[Path] = []

    def record_artifact(path: Path) -> dict[str, Any]:
        hashed_paths.append(Path(path))
        return original_artifact(path)

    monkeypatch.setattr(queue_translation_module, "_jsonl_artifact", record_artifact)
    translator.calls.clear()
    resumed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(shard_size=1),
        max_rows=1,
    )

    historical_names = {f"part-{index:06d}.jsonl" for index in range(4)} | {
        f"part-{index:06d}.accepted.jsonl.private" for index in range(4)
    }
    assert not historical_names & {path.name for path in hashed_paths}
    assert resumed["progress"]["completed_rows"] == 5
    assert resumed["progress"]["complete"] is False
    assert len(translator.calls) == 2


def test_teacher_review_required_is_derived_from_verified_progress(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        teacher_pilot_rows=2,
    )
    manifest["teacher_review"]["review_required"] = False
    queue_translation_module._atomic_write_json(results / "manifest.json", manifest)
    translator.calls.clear()

    with pytest.raises(ValueError, match="contradicts the verified pilot progress"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
            teacher_pilot_rows=2,
        )

    assert translator.calls == []


def test_completed_training_set_appears_as_one_atomic_top_level_file(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    rows = [
        {
            "id": f"atomic-{index}",
            "source_lang": "ko",
            "target_lang": "ja",
            "source": "안녕하세요.",
            "translation": None,
            "status": "pending",
        }
        for index in range(3)
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    translator = FakeTranslator()
    partial = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(shard_size=1),
        max_rows=2,
    )

    assert partial["progress"]["complete"] is False
    assert partial["training_set"] is None
    assert list(accepted.glob("*.jsonl")) == []

    completed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(shard_size=1),
    )
    visible = list(accepted.glob("*.jsonl"))
    assert visible == [Path(completed["training_set"]["path"])]
    assert completed["training_set"]["rows"] == 3


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("progress.next_part", True, r"progress\.next_part"),
        ("stats.processed", 1.0, r"stats\.processed"),
        ("parts.0.part", True, r"parts\[0\]\.part"),
        ("parts.0.result.rows", True, r"parts\[0\]\.result\.rows"),
    ],
)
def test_manifest_integer_controls_reject_bool_and_non_int_values(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    _write_queue(source)
    translator = FakeTranslator()
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    target: Any = manifest
    components = field.split(".")
    for component in components[:-1]:
        target = target[int(component)] if component.isdigit() else target[component]
    target[components[-1]] = value
    queue_translation_module._atomic_write_json(results / "manifest.json", manifest)

    with pytest.raises(ValueError, match=message):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )


def test_unicode_script_policy_counts_letters_not_shared_marks() -> None:
    assert script_letter_count("ーー・・", "kana") == 0
    assert script_letter_count("বাংলা ১২৩।", "bengali") == 2
    with pytest.raises(ValueError, match="too generic"):
        script_letter_count("letters", "letter")


def test_queue_accepts_an_explicit_unicode_name_script_for_an_arbitrary_language(
    tmp_path: Path,
) -> None:
    source_text = "This source sentence contains enough useful content."
    target_text = "বাংলা ভাষায় একটি দীর্ঘ অনুবাদ বাক্য লেখা হয়েছে"
    source = tmp_path / "queue.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "bengali-output",
                "source_lang": "en",
                "target_lang": "bn",
                "source": source_text,
                "translation": None,
                "status": "pending",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    translator = FakeTranslator()
    translator.translation_directions = (("en", "bn"),)
    translator.mapping[("en", "bn", source_text)] = target_text

    completed = translate_queue(
        source,
        tmp_path / "results",
        translator,
        accepted_dir=tmp_path / "accepted",
        options=_options(
            roundtrip_enabled=False,
            min_pair_score=0,
            min_target_language_fraction=0.0,
            required_target_scripts=(("bn", "bengali", 5),),
        ),
    )

    assert completed["stats"]["accepted"] == 1
    result = json.loads((tmp_path / "results" / "part-000000.jsonl").read_text(encoding="utf-8"))
    assert result["quality"]["forward"]["target_script_letters"]["bengali"] == script_letter_count(
        target_text, "bengali"
    )


def test_unknown_target_requires_an_explicit_script_policy_before_model_work(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queue.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "unknown-script",
                "source_lang": "qaa",
                "target_lang": "qab",
                "source": "source text",
                "translation": None,
                "status": "pending",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    translator = FakeTranslator()
    translator.translation_directions = (("qaa", "qab"),)

    with pytest.raises(ValueError, match="require an explicit required_target_scripts policy"):
        translate_queue(
            source,
            tmp_path / "results",
            translator,
            accepted_dir=tmp_path / "accepted",
            options=_options(roundtrip_enabled=False, required_target_scripts=()),
        )

    assert translator.calls == []


def test_no_replace_publication_flushes_after_link_and_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = tmp_path / "pending.jsonl"
    target = tmp_path / "target.jsonl"
    pending.write_text("{}\n", encoding="utf-8", newline="\n")
    flushed: list[Path] = []
    monkeypatch.setattr(
        queue_translation_module,
        "_fsync_directory",
        lambda path: flushed.append(Path(path).resolve()),
    )

    queue_translation_module._publish_no_replace(pending, target)

    assert target.read_text(encoding="utf-8") == "{}\n"
    assert not pending.exists()
    assert flushed == [tmp_path.resolve(), tmp_path.resolve()]


def test_complete_training_set_never_overwrites_a_top_level_collision(tmp_path: Path) -> None:
    source = tmp_path / "queue.jsonl"
    results = tmp_path / "results"
    accepted = tmp_path / "accepted"
    rows = [
        {
            "id": f"collision-{index}",
            "source_lang": "ko",
            "target_lang": "ja",
            "source": "안녕하세요.",
            "translation": None,
            "status": "pending",
        }
        for index in range(2)
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    translator = FakeTranslator()
    partial = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(shard_size=1),
        max_rows=1,
    )
    public_path = (
        accepted / f"{queue_translation_module.ACCEPTED_SHARD_PREFIX}{source.stem}_"
        f"{partial['run_id']}.jsonl"
    )
    public_path.write_text("foreign training rows\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete queue has a public training set"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(shard_size=1),
        )

    assert public_path.read_text(encoding="utf-8") == "foreign training rows\n"
    persisted = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["progress"]["complete"] is False
    assert persisted["training_set"] is None
