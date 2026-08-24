from __future__ import annotations

import importlib.util
from pathlib import Path
import os
import sys
from typing import NoReturn

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_review_queue.py"
SPEC = importlib.util.spec_from_file_location("build_review_queue_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD_REVIEW_QUEUE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD_REVIEW_QUEUE
SPEC.loader.exec_module(BUILD_REVIEW_QUEUE)

REPAIR_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "apply_contamination_repairs.py"
)
REPAIR_SPEC = importlib.util.spec_from_file_location(
    "apply_contamination_repairs_test", REPAIR_SCRIPT_PATH
)
assert REPAIR_SPEC is not None and REPAIR_SPEC.loader is not None
APPLY_REPAIRS = importlib.util.module_from_spec(REPAIR_SPEC)
sys.modules[REPAIR_SPEC.name] = APPLY_REPAIRS
REPAIR_SPEC.loader.exec_module(APPLY_REPAIRS)


def test_review_queue_cli_requires_an_explicit_language_graph() -> None:
    parser = BUILD_REVIEW_QUEUE.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "parallel.jsonl", "--output", "review.jsonl"])


def test_contamination_repair_cli_requires_an_explicit_direction() -> None:
    parser = APPLY_REPAIRS.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "parallel.jsonl"])
    args = parser.parse_args(
        [
            "--input",
            "parallel.jsonl",
            "--source-language",
            "KO-kr",
            "--target-language",
            "ja-JP",
        ]
    )
    assert args.source_language == "KO-kr"
    assert args.target_language == "ja-JP"


def test_review_queue_rejects_a_partially_supported_graph_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "review.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_review_queue.py",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(output),
            "--language-pairs",
            "ko",
            "ja",
            "--language-pairs",
            "en",
            "fr",
        ],
    )

    with pytest.raises(SystemExit, match="en→fr"):
        BUILD_REVIEW_QUEUE.main()

    assert not output.exists()


def test_review_queue_refuses_to_overwrite_an_input_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "parallel.jsonl"
    original = '{"ko":"검사할 원문","ja":"検査する訳文"}\n'
    source.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_review_queue.py",
            "--input",
            str(source),
            "--output",
            str(source),
            "--language-pair",
            "ko",
            "ja",
        ],
    )

    with pytest.raises(SystemExit, match="output path must not refer to an input shard"):
        BUILD_REVIEW_QUEUE.main()

    assert source.read_text(encoding="utf-8") == original


def test_review_queue_detects_a_hard_link_to_an_input_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "parallel.jsonl"
    source.write_text('{"ko":"원문","ja":"訳文"}\n', encoding="utf-8")
    output = tmp_path / "review.jsonl"
    try:
        os.link(source, output)
    except OSError as error:
        pytest.skip(f"hard links are unavailable in this test environment: {error}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_review_queue.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--language-pair",
            "ko",
            "ja",
        ],
    )

    with pytest.raises(SystemExit, match="output path must not refer to an input shard"):
        BUILD_REVIEW_QUEUE.main()


def test_review_queue_requires_distinct_queue_and_summary_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "parallel.jsonl"
    source.write_text('{"ko":"원문","ja":"訳文"}\n', encoding="utf-8")
    output = tmp_path / "review.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_review_queue.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--summary",
            str(output),
            "--language-pair",
            "ko",
            "ja",
        ],
    )

    with pytest.raises(SystemExit, match="output and summary must use different paths"):
        BUILD_REVIEW_QUEUE.main()

    assert not output.exists()


def test_review_queue_failure_preserves_the_previous_complete_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "parallel.jsonl"
    source.write_text('{"ko":"원문","ja":"訳文"}\n', encoding="utf-8")
    output = tmp_path / "review.jsonl"
    output.write_text("previous complete queue\n", encoding="utf-8")

    def fail_after_staging(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("injected review failure")

    monkeypatch.setattr(BUILD_REVIEW_QUEUE, "assess_contamination", fail_after_staging)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_review_queue.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--language-pair",
            "ko",
            "ja",
        ],
    )

    with pytest.raises(RuntimeError, match="injected review failure"):
        BUILD_REVIEW_QUEUE.main()

    assert output.read_text(encoding="utf-8") == "previous complete queue\n"
    assert not list(tmp_path.glob(f".{output.name}.*.part"))


def test_repair_report_refuses_to_overwrite_an_input_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "parallel.jsonl"
    original = '{"ko":"검사할 원문","ja":"検査する訳文"}\n'
    source.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_contamination_repairs.py",
            "--input",
            str(source),
            "--source-language",
            "ko",
            "--target-language",
            "ja",
            "--report",
            str(source),
        ],
    )

    with pytest.raises(SystemExit, match="report path must not refer to an input shard"):
        APPLY_REPAIRS.main()

    assert source.read_text(encoding="utf-8") == original


def test_atomic_repair_writer_preserves_the_destination_when_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "parallel.jsonl"
    destination.write_text("previous complete shard\n", encoding="utf-8")

    def fail_sync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(APPLY_REPAIRS.os, "fsync", fail_sync)

    with pytest.raises(OSError, match="injected fsync failure"):
        APPLY_REPAIRS._atomic_write_text(destination, "partial replacement\n")

    assert destination.read_text(encoding="utf-8") == "previous complete shard\n"
    assert not list(tmp_path.glob(f".{destination.name}.*.part"))


def test_repair_backups_do_not_collide_when_shards_share_a_basename(tmp_path: Path) -> None:
    first = tmp_path / "first" / "parallel.jsonl"
    second = tmp_path / "second" / "parallel.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first shard\n", encoding="utf-8")
    second.write_text("second shard\n", encoding="utf-8")
    backup_root = tmp_path / "backups"

    first_backup = APPLY_REPAIRS._copy_backup(first, backup_root)
    second_backup = APPLY_REPAIRS._copy_backup(second, backup_root)

    assert first_backup != second_backup
    assert first_backup.read_text(encoding="utf-8") == "first shard\n"
    assert second_backup.read_text(encoding="utf-8") == "second shard\n"


def test_repair_backup_refuses_to_clobber_an_existing_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "parallel.jsonl"
    source.write_text("original shard\n", encoding="utf-8")
    backup_root = tmp_path / "backups"
    backup = APPLY_REPAIRS._copy_backup(source, backup_root)
    source.write_text("later shard\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable backup"):
        APPLY_REPAIRS._copy_backup(source, backup_root)

    assert backup.read_text(encoding="utf-8") == "original shard\n"
