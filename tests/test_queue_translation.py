from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from sion_translate.queue_translation import (
    QueueTranslationOptions,
    translate_queue,
)


class FakeTokenizer:
    @staticmethod
    def encode(text: str) -> list[str]:
        return [character for character in text if not character.isspace()]


class FakeTranslator:
    tokenizer = FakeTokenizer()

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []
        self.mapping = {
            ("ko", "ja", "안녕하세요."): "こんにちは。",
            ("ja", "ko", "こんにちは。"): "안녕하세요.",
            ("ko", "ja", "원래 문장입니다."): "元の文です。",
            ("ja", "ko", "元の文です。"): "전혀 다른 내용입니다.",
            ("ko", "ja", "정상 문장입니다."): "正常な文です。",
            ("ja", "ko", "正常な文です。"): "정상 문장입니다.",
        }

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
            raise RuntimeError("synthetic failure")
        return [self.mapping[(source_language, target_language, text)] for text in texts]


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
        run_metadata={"model": "fake-v1"},
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

    completed = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        run_metadata={"model": "fake-v1"},
    )

    assert completed["progress"]["complete"]
    assert completed["progress"]["completed_rows"] == 5
    assert completed["stats"]["accepted"] == 2
    accepted_rows = [
        json.loads(line)
        for path in sorted(accepted.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in accepted_rows] == ["one", "two"]
    assert all(row["synthetic"] is True for row in accepted_rows)
    assert all(row["provenance"]["run_id"] == completed["run_id"] for row in accepted_rows)

    call_count = len(translator.calls)
    unchanged = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        run_metadata={"model": "fake-v1"},
    )
    assert unchanged["stats"] == completed["stats"]
    assert len(translator.calls) == call_count


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
        run_metadata={"model": "fake-v1"},
        max_rows=1,
    )

    with pytest.raises(ValueError, match="resume configuration changed"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(min_roundtrip_score=0.10),
            run_metadata={"model": "fake-v1"},
        )
    with pytest.raises(ValueError, match="resume configuration changed"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
            run_metadata={"model": "fake-v2"},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("shard_size", 0),
        ("num_beams", 0),
        ("max_new_tokens", 0),
        ("max_output_length_ratio", 0),
        ("max_output_length_margin", -1),
        ("min_roundtrip_score", 1.1),
        ("min_pair_score", 101),
        ("min_target_language_fraction", -0.1),
        ("min_japanese_kana_chars", -1),
    ],
)
def test_queue_options_validate(field: str, value: object) -> None:
    options = _options(**{field: value})
    with pytest.raises(ValueError):
        options.validate()


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

    with pytest.raises(ValueError, match="resume configuration changed"):
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
    with (results / "part-000000.jsonl").open("a", encoding="utf-8") as handle:
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
    Path(manifest["parts"][0]["accepted"]["path"]).unlink()

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
    manifest = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
        max_rows=1,
    )
    configuration_bytes = json.dumps(
        manifest["configuration"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    legacy_signature = hashlib.sha256(configuration_bytes).hexdigest()
    old_accepted_path = Path(manifest["parts"][0]["accepted"]["path"])
    legacy_run_id = legacy_signature[:16]
    legacy_accepted_path = accepted / f"bt_{source.stem}_{legacy_run_id}_000000.jsonl"
    result_path = results / "part-000000.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["run_id"] = legacy_run_id
    result_path.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    accepted_row = json.loads(old_accepted_path.read_text(encoding="utf-8"))
    accepted_row["provenance"]["run_id"] = legacy_run_id
    old_accepted_path.write_text(
        json.dumps(accepted_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    old_accepted_path.rename(legacy_accepted_path)
    manifest["run_id"] = legacy_run_id
    manifest["run_signature"] = legacy_signature
    manifest.pop("signature_version")
    manifest.pop("parts")
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    migrated = translate_queue(
        source,
        results,
        translator,
        accepted_dir=accepted,
        options=_options(),
    )

    assert migrated["progress"]["complete"]
    assert migrated["run_id"] == legacy_run_id
    assert migrated["run_signature"] != legacy_signature
    assert migrated["signature_version"] == 2
    assert len(migrated["parts"]) == migrated["progress"]["next_part"]


def test_legacy_shard_registration_rejects_broken_source_sequence(
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
    result_path = results / "part-000000.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["source_index"] = 99
    result_path.write_text(
        json.dumps(result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest.pop("parts")
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-contiguous source_index"):
        translate_queue(
            source,
            results,
            translator,
            accepted_dir=accepted,
            options=_options(),
        )


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
        options=_options(),
    )

    assert manifest["stats"]["accepted"] == 0
    result = json.loads((tmp_path / "results" / "part-000000.jsonl").read_text(encoding="utf-8"))
    assert result["quality"]["forward"]["target_language_fraction"] == 1.0
    assert result["quality"]["forward"]["target_japanese_kana_chars"] == 0
    assert result["rejection_reasons"] == ["target_japanese_kana"]
