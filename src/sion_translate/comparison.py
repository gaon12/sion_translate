"""JSONL 기반 번역 시스템 비교 도구.

비교 문장과 각 시스템의 번역 결과를 별도 JSONL로 유지한다. 외부 API,
로컬 모델, 사람이 복사한 결과를 모두 같은 형식으로 채점할 수 있다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from sion_translate.evaluation import (
    DirectionResult,
    number_preservation_details,
    results_as_markdown,
    score_translations,
)


@dataclass(frozen=True)
class ComparisonCase:
    """한 개의 번역 비교 문장."""

    id: str
    source_language: str
    target_language: str
    source: str
    reference: str
    category: str = "general"


def _required_text(row: object, key: str, *, location: str) -> str:
    if not isinstance(row, dict):
        raise ValueError(f"{location}: JSON object가 필요합니다")
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: '{key}'는 비어 있지 않은 문자열이어야 합니다")
    return value.strip()


def _jsonl_rows(path: str | Path) -> list[tuple[int, object]]:
    path = Path(path)
    rows: list[tuple[int, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append((line_number, json.loads(line)))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: 잘못된 JSON: {error.msg}") from error
    return rows


def load_comparison_cases(path: str | Path) -> list[ComparisonCase]:
    """비교 케이스 JSONL을 읽고 스키마와 ID 중복을 검사한다."""
    path = Path(path)
    cases: list[ComparisonCase] = []
    seen: set[str] = set()
    for line_number, row in _jsonl_rows(path):
        location = f"{path}:{line_number}"
        case_id = _required_text(row, "id", location=location)
        if case_id in seen:
            raise ValueError(f"{location}: 중복 id: {case_id}")
        seen.add(case_id)
        source_language = _required_text(row, "source_language", location=location)
        target_language = _required_text(row, "target_language", location=location)
        if source_language == target_language:
            raise ValueError(f"{location}: 원문 언어와 목표 언어가 같습니다")
        category = row.get("category", "general") if isinstance(row, dict) else "general"
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{location}: 'category'는 문자열이어야 합니다")
        cases.append(
            ComparisonCase(
                id=case_id,
                source_language=source_language,
                target_language=target_language,
                source=_required_text(row, "source", location=location),
                reference=_required_text(row, "reference", location=location),
                category=category.strip(),
            )
        )
    if not cases:
        raise ValueError(f"{path}: 비교 문장이 없습니다")
    return cases


def load_system_translations(
    path: str | Path,
    cases: Sequence[ComparisonCase],
) -> dict[str, str]:
    """시스템 출력 JSONL을 읽고 모든 케이스가 정확히 한 번 있는지 검사한다."""
    path = Path(path)
    expected = {case.id for case in cases}
    translations: dict[str, str] = {}
    for line_number, row in _jsonl_rows(path):
        location = f"{path}:{line_number}"
        case_id = _required_text(row, "id", location=location)
        if case_id not in expected:
            raise ValueError(f"{location}: 비교 케이스에 없는 id: {case_id}")
        if case_id in translations:
            raise ValueError(f"{location}: 중복 id: {case_id}")
        translations[case_id] = _required_text(row, "translation", location=location)

    missing = [case.id for case in cases if case.id not in translations]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"{path}: 번역이 없는 id {len(missing)}개: {preview}{suffix}")
    return translations


def write_system_translations(
    path: str | Path,
    cases: Sequence[ComparisonCase],
    translations: Mapping[str, str],
) -> None:
    """비교 CLI가 읽을 수 있는 순서 고정 JSONL을 쓴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [case.id for case in cases if not translations.get(case.id, "").strip()]
    if missing:
        raise ValueError(f"비어 있는 번역이 있습니다: {', '.join(missing[:5])}")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            row = {"id": case.id, "translation": translations[case.id]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_systems(
    cases: Sequence[ComparisonCase],
    systems: Mapping[str, Mapping[str, str]],
) -> list[DirectionResult]:
    """시스템별·번역 방향별 chrF/BLEU를 계산한다."""
    directions: dict[tuple[str, str], list[ComparisonCase]] = {}
    for case in cases:
        directions.setdefault((case.source_language, case.target_language), []).append(case)

    results: list[DirectionResult] = []
    for system_name, translations in systems.items():
        for (source_language, target_language), direction_cases in directions.items():
            hypotheses = [translations[case.id] for case in direction_cases]
            references = [case.reference for case in direction_cases]
            chrf, bleu, tokenize = score_translations(
                hypotheses,
                references,
                target_language=target_language,
            )
            number_result = number_preservation_details(hypotheses, references)
            results.append(
                DirectionResult(
                    system=system_name,
                    direction=f"{source_language}-{target_language}",
                    samples=len(direction_cases),
                    chrf=chrf,
                    bleu=bleu,
                    bleu_tokenize=tokenize,
                    number_f1=number_result.f1,
                    number_exact=number_result.exact,
                    number_samples=number_result.samples,
                    number_inventions=number_result.inventions,
                )
            )
    return results


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def comparison_as_markdown(
    cases: Sequence[ComparisonCase],
    systems: Mapping[str, Mapping[str, str]],
    results: Sequence[DirectionResult],
) -> str:
    """점수 표와 문장별 대조표를 한 Markdown 문서로 만든다."""
    lines = [
        "# Translation comparison",
        "",
        "> Small custom sentence sets are diagnostic, not a universal quality ranking.",
        "",
        results_as_markdown(results),
        "",
        "## Sentence-level outputs",
        "",
        "| id | category | direction | system | source | reference | translation |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in cases:
        for system_name, translations in systems.items():
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        case.id,
                        case.category,
                        f"{case.source_language}-{case.target_language}",
                        system_name,
                        case.source,
                        case.reference,
                        translations[case.id],
                    )
                )
                + " |"
            )
    return "\n".join(lines)


def save_comparison(
    output_path: str | Path,
    cases: Sequence[ComparisonCase],
    systems: Mapping[str, Mapping[str, str]],
    results: Sequence[DirectionResult],
) -> None:
    """기계용 JSON과 사람이 검토할 Markdown 결과를 저장한다."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "sion-translation-comparison-v1",
        "cases": [asdict(case) for case in cases],
        "systems": {name: dict(translations) for name, translations in systems.items()},
        "results": [asdict(result) for result in results],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_path.with_suffix(".md").write_text(
        comparison_as_markdown(cases, systems, results) + "\n",
        encoding="utf-8",
    )
