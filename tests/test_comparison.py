"""JSONL 기반 다중 번역 시스템 비교 검증."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kjx.comparison import (
    comparison_as_markdown,
    load_comparison_cases,
    load_system_translations,
    save_comparison,
    score_systems,
    write_system_translations,
)


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _cases(path: Path):
    _write_jsonl(
        path,
        [
            {
                "id": "forward",
                "source_language": "ko",
                "target_language": "ja",
                "category": "general",
                "source": "안녕하세요",
                "reference": "こんにちは",
            },
            {
                "id": "backward",
                "source_language": "ja",
                "target_language": "ko",
                "category": "general",
                "source": "ありがとう",
                "reference": "고맙습니다",
            },
        ],
    )
    return load_comparison_cases(path)


def test_load_and_write_system_translations_in_case_order(tmp_path: Path) -> None:
    cases = _cases(tmp_path / "cases.jsonl")
    output = tmp_path / "system.jsonl"
    write_system_translations(
        output,
        cases,
        {"backward": "고맙습니다", "forward": "こんにちは"},
    )
    assert [json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()] == [
        "forward",
        "backward",
    ]
    assert load_system_translations(output, cases)["forward"] == "こんにちは"


def test_missing_or_duplicate_output_is_rejected(tmp_path: Path) -> None:
    cases = _cases(tmp_path / "cases.jsonl")
    missing = tmp_path / "missing.jsonl"
    _write_jsonl(missing, [{"id": "forward", "translation": "こんにちは"}])
    with pytest.raises(ValueError, match="번역이 없는 id"):
        load_system_translations(missing, cases)

    duplicate = tmp_path / "duplicate.jsonl"
    _write_jsonl(
        duplicate,
        [
            {"id": "forward", "translation": "こんにちは"},
            {"id": "forward", "translation": "こんにちは"},
        ],
    )
    with pytest.raises(ValueError, match="중복 id"):
        load_system_translations(duplicate, cases)


def test_score_and_report_multiple_systems(tmp_path: Path) -> None:
    cases = _cases(tmp_path / "cases.jsonl")
    systems = {
        "perfect": {"forward": "こんにちは", "backward": "고맙습니다"},
        "weak": {"forward": "さようなら", "backward": "모릅니다"},
    }
    results = score_systems(cases, systems)
    perfect = [result for result in results if result.system == "perfect"]
    assert len(perfect) == 2
    assert all(result.chrf == 100.0 for result in perfect)

    markdown = comparison_as_markdown(cases, systems, results)
    assert "| forward | general | ko-ja | perfect |" in markdown
    save_comparison(tmp_path / "comparison", cases, systems, results)
    payload = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "kjx-translation-comparison-v1"
    assert (tmp_path / "comparison.md").exists()
