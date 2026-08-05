"""JSONL 기반 다중 번역 시스템 비교 검증."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sion_translate.comparison import (
    category_results_as_markdown,
    comparison_as_markdown,
    load_comparison_cases,
    load_system_translations,
    save_comparison,
    score_system_categories,
    score_systems,
    write_system_translations,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CASE_FILES = [
    REPOSITORY_ROOT / "examples" / "comparison_cases.jsonl",
    REPOSITORY_ROOT / "examples" / "diagnostic_cases.jsonl",
    REPOSITORY_ROOT / "examples" / "expressive_cultural_cases.jsonl",
]


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize("path", SHIPPED_CASE_FILES, ids=lambda path: path.name)
def test_shipped_case_files_load_and_stay_balanced(path: Path) -> None:
    """저장소가 제공하는 진단셋이 스키마를 지키고 방향이 균형인지 확인한다."""
    cases = load_comparison_cases(path)
    assert len(cases) >= 16

    directions: dict[str, int] = {}
    for case in cases:
        directions[f"{case.source_language}-{case.target_language}"] = (
            directions.get(f"{case.source_language}-{case.target_language}", 0) + 1
        )
    # 한 방향으로 치우치면 방향별 점수를 비교할 수 없다.
    assert set(directions) == {"ko-ja", "ja-ko"}
    assert directions["ko-ja"] == directions["ja-ko"]

    # 모든 케이스에 카테고리가 있어야 카테고리별 진단이 가능하다.
    assert all(case.category != "general" for case in cases)


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
    assert payload["schema"] == "sion-translation-comparison-v1"
    assert len(payload["category_results"]) == 4
    assert (tmp_path / "comparison.md").exists()


def test_category_scoring_is_additive_and_preserves_aggregate_callers() -> None:
    cases = load_comparison_cases(REPOSITORY_ROOT / "examples" / "expressive_cultural_cases.jsonl")
    translations = {case.id: case.reference for case in cases}

    aggregate = score_systems(cases, {"perfect": translations})
    categories = score_system_categories(cases, {"perfect": translations})

    # The legacy function remains direction-only; category scoring is a separate
    # additive API so existing callers and result types do not change.
    assert len(aggregate) == 2
    assert {result.direction for result in aggregate} == {"ko-ja", "ja-ko"}
    assert len(categories) == 6
    assert {result.category for result in categories} == {
        "profanity_slang",
        "interjection_moan",
        "idiom_culture",
    }
    assert all(result.samples == 4 for result in categories)
    assert all(result.chrf == 100.0 for result in categories)
    markdown = category_results_as_markdown(categories)
    assert "| perfect | ja-ko | profanity_slang | 4 | 100.00 |" in markdown


def test_expressive_case_metadata_survives_loading_and_reporting() -> None:
    cases = load_comparison_cases(REPOSITORY_ROOT / "examples" / "expressive_cultural_cases.jsonl")
    case = next(item for item in cases if item.id == "ko-ja-profanity-slang-challenge-01")

    assert case.subcategory == "strong_profanity"
    assert case.intensity == 5
    assert case.register == "vulgar_casual"
    assert case.localization_strategy == "intensity_equivalent"


def test_invalid_comparison_intensity_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "bad-intensity",
                "source_language": "ko",
                "target_language": "ja",
                "source": "젠장",
                "reference": "ちくしょう",
                "intensity": 6,
            }
        ],
    )

    with pytest.raises(ValueError, match="intensity"):
        load_comparison_cases(path)
