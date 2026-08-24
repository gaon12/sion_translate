"""Compare translation systems through a shared JSONL contract.

Cases and system outputs live in separate JSONL files so an external API, a
local model, and manually copied results can all use the same scoring path.
"""

# Comparison inputs are heterogeneous JSON rows validated by this module.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

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
from sion_translate.language_tags import LanguageTagError, canonicalize_language_tag


@dataclass(frozen=True)
class ComparisonCase:
    """One source/reference case in a translation comparison."""

    id: str
    source_language: str
    target_language: str
    source: str
    reference: str
    category: str = "general"
    subcategory: str = ""
    intensity: int | None = None
    register: str = ""
    localization_strategy: str = ""


@dataclass(frozen=True)
class CategoryResult:
    """One system's score for a direction/category diagnostic slice."""

    system: str
    direction: str
    category: str
    samples: int
    chrf: float
    bleu: float
    bleu_tokenize: str
    number_f1: float = 0.0
    number_exact: int = 0
    number_samples: int = 0
    number_inventions: int = 0


def _required_text(row: object, key: str, *, location: str) -> str:
    if not isinstance(row, dict):
        raise ValueError(f"{location}: expected a JSON object")
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: '{key}' must be a non-empty string")
    return value.strip()


def _optional_text(row: dict[str, object], key: str, *, location: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{location}: '{key}' must be a string")
    return value.strip()


def _optional_intensity(row: dict[str, object], *, location: str) -> int | None:
    value = row.get("intensity")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
        raise ValueError(f"{location}: 'intensity' must be an integer from 1 to 5")
    return value


def _canonical_case_language(value: str, *, field: str, location: str) -> str:
    try:
        return canonicalize_language_tag(value, field=f"{location}: '{field}'")
    except LanguageTagError as error:
        raise ValueError(str(error)) from error


def _direction_label(source_language: str, target_language: str) -> str:
    """Return a collision-free label while preserving legacy simple labels."""

    if "-" not in source_language and "-" not in target_language:
        return f"{source_language}-{target_language}"
    return f"{source_language}→{target_language}"


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
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
    return rows


def load_comparison_cases(path: str | Path) -> list[ComparisonCase]:
    """Load comparison JSONL and validate its schema and unique IDs."""
    path = Path(path)
    cases: list[ComparisonCase] = []
    seen: set[str] = set()
    for line_number, row in _jsonl_rows(path):
        location = f"{path}:{line_number}"
        case_id = _required_text(row, "id", location=location)
        if case_id in seen:
            raise ValueError(f"{location}: duplicate id: {case_id}")
        seen.add(case_id)
        source_language = _canonical_case_language(
            _required_text(row, "source_language", location=location),
            field="source_language",
            location=location,
        )
        target_language = _canonical_case_language(
            _required_text(row, "target_language", location=location),
            field="target_language",
            location=location,
        )
        if source_language == target_language:
            raise ValueError(f"{location}: source and target languages are equal")
        assert isinstance(row, dict)
        category = row.get("category", "general")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{location}: 'category' must be a string")
        cases.append(
            ComparisonCase(
                id=case_id,
                source_language=source_language,
                target_language=target_language,
                source=_required_text(row, "source", location=location),
                reference=_required_text(row, "reference", location=location),
                category=category.strip(),
                subcategory=_optional_text(row, "subcategory", location=location),
                intensity=_optional_intensity(row, location=location),
                register=_optional_text(row, "register", location=location),
                localization_strategy=_optional_text(
                    row,
                    "localization_strategy",
                    location=location,
                ),
            )
        )
    if not cases:
        raise ValueError(f"{path}: no comparison cases were found")
    return cases


def load_system_translations(
    path: str | Path,
    cases: Sequence[ComparisonCase],
) -> dict[str, str]:
    """Load system output and require every case exactly once."""
    path = Path(path)
    expected = {case.id for case in cases}
    translations: dict[str, str] = {}
    for line_number, row in _jsonl_rows(path):
        location = f"{path}:{line_number}"
        case_id = _required_text(row, "id", location=location)
        if case_id not in expected:
            raise ValueError(f"{location}: id is not in the comparison cases: {case_id}")
        if case_id in translations:
            raise ValueError(f"{location}: duplicate id: {case_id}")
        translations[case_id] = _required_text(row, "translation", location=location)

    missing = [case.id for case in cases if case.id not in translations]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"{path}: {len(missing)} IDs have no translation: {preview}{suffix}")
    return translations


def write_system_translations(
    path: str | Path,
    cases: Sequence[ComparisonCase],
    translations: Mapping[str, str],
) -> None:
    """Write stable-order JSONL that the comparison CLI can read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [case.id for case in cases if not translations.get(case.id, "").strip()]
    if missing:
        raise ValueError(f"Empty translations are not allowed: {', '.join(missing[:5])}")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            row = {"id": case.id, "translation": translations[case.id]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_systems(
    cases: Sequence[ComparisonCase],
    systems: Mapping[str, Mapping[str, str]],
) -> list[DirectionResult]:
    """Compute chrF and BLEU for each system and directed language edge."""
    directions: dict[tuple[str, str], list[ComparisonCase]] = {}
    for case in cases:
        directions.setdefault((case.source_language, case.target_language), []).append(case)

    results: list[DirectionResult] = []
    for system_name, translations in systems.items():
        for (source_language, target_language), direction_cases in directions.items():
            sources = [case.source for case in direction_cases]
            hypotheses = [translations[case.id] for case in direction_cases]
            references = [case.reference for case in direction_cases]
            chrf, bleu, tokenize = score_translations(
                hypotheses,
                references,
                target_language=target_language,
            )
            number_result = number_preservation_details(hypotheses, sources=sources)
            results.append(
                DirectionResult(
                    system=system_name,
                    direction=_direction_label(source_language, target_language),
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


def score_system_categories(
    cases: Sequence[ComparisonCase],
    systems: Mapping[str, Mapping[str, str]],
) -> list[CategoryResult]:
    """Score diagnostic slices without changing :func:`score_systems`' contract."""

    groups: dict[tuple[str, str, str], list[ComparisonCase]] = {}
    for case in cases:
        key = (case.source_language, case.target_language, case.category)
        groups.setdefault(key, []).append(case)

    results: list[CategoryResult] = []
    for system_name, translations in systems.items():
        for (source_language, target_language, category), group_cases in groups.items():
            sources = [case.source for case in group_cases]
            hypotheses = [translations[case.id] for case in group_cases]
            references = [case.reference for case in group_cases]
            chrf, bleu, tokenize = score_translations(
                hypotheses,
                references,
                target_language=target_language,
            )
            number_result = number_preservation_details(hypotheses, sources=sources)
            results.append(
                CategoryResult(
                    system=system_name,
                    direction=_direction_label(source_language, target_language),
                    category=category,
                    samples=len(group_cases),
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


def category_results_as_markdown(results: Sequence[CategoryResult]) -> str:
    """Render direction/category slices separately from aggregate direction scores."""

    lines = [
        "| system | direction | category | samples | chrF | BLEU | number F1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        number_f1 = f"{result.number_f1:.2f}" if result.number_samples else "-"
        lines.append(
            f"| {_markdown_cell(result.system)} | {_markdown_cell(result.direction)} "
            f"| {_markdown_cell(result.category)} | {result.samples} "
            f"| {result.chrf:.2f} | {result.bleu:.2f} | {number_f1} |"
        )
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def comparison_as_markdown(
    cases: Sequence[ComparisonCase],
    systems: Mapping[str, Mapping[str, str]],
    results: Sequence[DirectionResult],
    *,
    category_results: Sequence[CategoryResult] | None = None,
) -> str:
    """Create one Markdown report with scores and case-level comparisons."""
    category_results = (
        list(category_results)
        if category_results is not None
        else score_system_categories(cases, systems)
    )
    lines = [
        "# Translation comparison",
        "",
        "> Small custom sentence sets are diagnostic, not a universal quality ranking.",
        "",
        results_as_markdown(results),
        "",
        "## Category diagnostics",
        "",
        category_results_as_markdown(category_results),
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
                        _direction_label(case.source_language, case.target_language),
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
    *,
    category_results: Sequence[CategoryResult] | None = None,
) -> None:
    """Save machine-readable JSON and human-reviewable Markdown results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    category_results = (
        list(category_results)
        if category_results is not None
        else score_system_categories(cases, systems)
    )
    payload = {
        "schema": "sion-translation-comparison-v1",
        "cases": [asdict(case) for case in cases],
        "systems": {name: dict(translations) for name, translations in systems.items()},
        "results": [asdict(result) for result in results],
        "category_results": [asdict(result) for result in category_results],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_path.with_suffix(".md").write_text(
        comparison_as_markdown(
            cases,
            systems,
            results,
            category_results=category_results,
        )
        + "\n",
        encoding="utf-8",
    )
