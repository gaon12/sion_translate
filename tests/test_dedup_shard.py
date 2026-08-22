from __future__ import annotations

from collections.abc import Iterator
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "dedup_shard.py"
SPEC = importlib.util.spec_from_file_location("dedup_shard_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
DEDUP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEDUP
SPEC.loader.exec_module(DEDUP)


def test_dedup_rows_collapses_bidirectional_duplicates() -> None:
    rows = [
        {"ko": "알겠습니다.", "ja": "わかりました。"},
        {"ko": "포장마차도 줄었습니다.", "ja": "屋台も減りました。"},
        {"ko": " 알겠습니다. ", "ja": "わかりました。"},
    ]

    output = list(DEDUP.dedup_rows(rows))

    assert len(output) == 2
    assert output[0]["ko"] == "알겠습니다."
    assert output[1]["ko"] == "포장마차도 줄었습니다."


def test_dedup_rows_keeps_first_occurrence_metadata() -> None:
    rows = [
        {"ko": "가", "ja": "あ", "direction": "ko2ja"},
        {"ko": "가", "ja": "あ", "direction": "ja2ko"},
    ]

    output = list(DEDUP.dedup_rows(rows))

    assert output == [{"ko": "가", "ja": "あ", "direction": "ko2ja"}]


def test_dedup_rows_separates_different_language_sets() -> None:
    rows = [
        {"ko": "가", "ja": "あ"},
        {"ko": "가", "en": "a", "ja": "あ"},
    ]

    assert len(list(DEDUP.dedup_rows(rows))) == 2


def test_dedup_rows_passes_through_rows_without_language_text() -> None:
    rows = [{"note": "first"}, {"note": "second"}]

    assert list(DEDUP.dedup_rows(rows)) == rows


def test_main_writes_deduplicated_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "in.jsonl"
    target = tmp_path / "out.jsonl"
    source.write_text(
        '{"ko": "가", "ja": "あ"}\n{"ko": "가", "ja": "あ"}\n{"ko": "나", "ja": "い"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys, "argv", ["dedup_shard.py", "--input", str(source), "--output", str(target)]
    )

    DEDUP.main()

    report = json.loads(capsys.readouterr().out)
    assert report["input_rows"] == 3
    assert report["output_rows"] == 2
    assert report["removed_rows"] == 1
    assert target.read_text(encoding="utf-8").count("\n") == 2


def test_write_jsonl_supports_an_in_place_rewrite(tmp_path: Path) -> None:
    shard = tmp_path / "shard.jsonl"
    shard.write_text(
        '{"ko": "가", "ja": "あ"}\n'
        '{"ko": "가", "ja": "あ", "duplicate": true}\n'
        '{"ko": "나", "ja": "い"}\n',
        encoding="utf-8",
    )

    output_rows, digest = DEDUP.write_jsonl(
        shard,
        DEDUP.dedup_rows(DEDUP.read_jsonl(shard)),
    )

    output = list(DEDUP.read_jsonl(shard))
    assert output_rows == 2
    assert output == [
        {"ko": "가", "ja": "あ"},
        {"ko": "나", "ja": "い"},
    ]
    assert digest == DEDUP.sha256(shard.read_bytes()).hexdigest()
    assert not list(tmp_path.glob(f".{shard.name}.*.tmp"))


def test_main_supports_the_same_input_and_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = tmp_path / "shard.jsonl"
    shard.write_text(
        '{"ko": "가", "ja": "あ"}\n{"ko": "가", "ja": "あ"}\n{"ko": "나", "ja": "い"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dedup_shard.py", "--input", str(shard), "--output", str(shard)],
    )

    DEDUP.main()

    report = json.loads(capsys.readouterr().out)
    assert report["input_rows"] == 3
    assert report["output_rows"] == 2
    assert report["removed_rows"] == 1
    assert list(DEDUP.read_jsonl(shard)) == [
        {"ko": "가", "ja": "あ"},
        {"ko": "나", "ja": "い"},
    ]
    assert not list(tmp_path.glob(f".{shard.name}.*.tmp"))


def test_mid_write_failure_preserves_existing_output_and_cleans_staging(
    tmp_path: Path,
) -> None:
    target = tmp_path / "out.jsonl"
    original = b'{"ko": "original", "ja": "original"}\n'
    target.write_bytes(original)

    def fail_after_one_row() -> Iterator[dict[str, str]]:
        yield {"ko": "가", "ja": "あ"}
        raise RuntimeError("injected row production failure")

    with pytest.raises(RuntimeError, match="injected row production failure"):
        DEDUP.write_jsonl(target, fail_after_one_row())

    assert target.read_bytes() == original
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_validation_failure_preserves_existing_output_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.jsonl"
    original = b'{"ko": "original", "ja": "original"}\n'
    target.write_bytes(original)

    def reject_staging(*args: object, **kwargs: object) -> None:
        raise ValueError("injected validation failure")

    monkeypatch.setattr(DEDUP, "_validate_staged_jsonl", reject_staging)

    with pytest.raises(ValueError, match="injected validation failure"):
        DEDUP.write_jsonl(target, [{"ko": "가", "ja": "あ"}])

    assert target.read_bytes() == original
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_replace_failure_preserves_existing_output_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "out.jsonl"
    original = b'{"ko": "original", "ja": "original"}\n'
    target.write_bytes(original)

    def reject_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        raise PermissionError(f"injected replace failure: {source} -> {destination}")

    monkeypatch.setattr(DEDUP.os, "replace", reject_replace)

    with pytest.raises(PermissionError, match="injected replace failure"):
        DEDUP.write_jsonl(target, [{"ko": "가", "ja": "あ"}])

    assert target.read_bytes() == original
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_empty_input_cannot_erase_existing_output_without_opt_in(tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"
    original = b'{"ko": "original", "ja": "original"}\n'
    target.write_bytes(original)

    with pytest.raises(ValueError, match="refusing to publish an empty JSONL shard"):
        DEDUP.write_jsonl(target, [])

    assert target.read_bytes() == original
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_allow_empty_explicitly_publishes_an_empty_output(tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"
    target.write_text('{"ko": "old", "ja": "old"}\n', encoding="utf-8")

    output_rows, digest = DEDUP.write_jsonl(target, [], allow_empty=True)

    assert output_rows == 0
    assert digest == DEDUP.sha256(b"").hexdigest()
    assert target.read_bytes() == b""
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))
