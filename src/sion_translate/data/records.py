"""Normalize heterogeneous JSONL records into explicit parallel text pairs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


_LANGUAGE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,15}$")
_SOURCE_KEYS = ("source", "src", "input")
_TARGET_KEYS = ("target", "tgt", "reference", "translation", "output")
_CONTAINER_KEYS = ("records", "items", "pairs", "translations")


@dataclass(frozen=True, slots=True)
class ParallelText:
    language_a: str
    text_a: str
    language_b: str
    text_b: str


@dataclass(frozen=True, slots=True)
class RecordExpansion:
    pairs: tuple[ParallelText, ...]
    issues: tuple[str, ...]


def _validate_language(language: object) -> str:
    if not isinstance(language, str) or not _LANGUAGE.fullmatch(language):
        raise ValueError(
            "language keys must be 1-16 ASCII alphanumeric characters and start "
            f"with a letter; got {language!r}"
        )
    return language


def normalize_language_pairs(
    language_pair: Sequence[str] = ("ko", "ja"),
    language_pairs: Sequence[Sequence[str]] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Validate configured pairs and remove duplicate reverse edges."""

    raw_pairs: Sequence[Sequence[str]] = (
        language_pairs if language_pairs is not None else (language_pair,)
    )
    if not raw_pairs:
        raise ValueError("at least one language pair is required")
    normalized: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for raw_pair in raw_pairs:
        if len(raw_pair) != 2:
            raise ValueError(f"each language pair must contain two keys; got {raw_pair!r}")
        language_a = _validate_language(raw_pair[0])
        language_b = _validate_language(raw_pair[1])
        if language_a == language_b:
            raise ValueError(f"language pair members must be distinct; got {raw_pair!r}")
        edge = frozenset((language_a, language_b))
        if edge in seen:
            continue
        seen.add(edge)
        normalized.append((language_a, language_b))
    return tuple(normalized)


def languages_from_pairs(language_pairs: Sequence[Sequence[str]]) -> tuple[str, ...]:
    """Return languages in first-appearance order."""

    return tuple(dict.fromkeys(language for pair in language_pairs for language in pair))


def _pair_labels(language_a: str, language_b: str) -> tuple[str, ...]:
    return (
        f"{language_a}-{language_b}",
        f"{language_b}-{language_a}",
        f"{language_a}/{language_b}",
        f"{language_b}/{language_a}",
        f"{language_a}_to_{language_b}",
        f"{language_b}_to_{language_a}",
    )


def _first_value(mapping: dict, names: Sequence[str]) -> object | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def expand_parallel_record(
    record: object,
    language_pairs: Sequence[Sequence[str]],
) -> RecordExpansion:
    """Expand one JSON value into configured parallel pairs.

    Supported layouts include flat language keys, list-valued language keys,
    arrays of records, ``records/items/pairs/translation(s)`` containers, explicit
    source/target language fields, and pair-named containers such as ``ko-ja``.
    """

    configured = normalize_language_pairs(language_pairs=language_pairs)
    configured_edges = {frozenset(pair): pair for pair in configured}
    output: list[ParallelText] = []
    issues: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()

    def issue(name: str) -> None:
        if name not in issues:
            issues.append(name)

    def values(value: object) -> list[str] | None:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return list(value)
        return None

    def emit(
        language_a: str,
        value_a: object,
        language_b: str,
        value_b: object,
    ) -> None:
        edge = frozenset((language_a, language_b))
        canonical_pair = configured_edges.get(edge)
        if canonical_pair is None:
            return
        texts_a = values(value_a)
        texts_b = values(value_b)
        if texts_a is None or texts_b is None:
            issue("non_string")
            return
        if len(texts_a) != len(texts_b):
            issue("unaligned_lists")
            return
        for text_a, text_b in zip(texts_a, texts_b, strict=True):
            if not text_a.strip() or not text_b.strip():
                issue("missing_text")
                continue
            if (language_a, language_b) != canonical_pair:
                text_a, text_b = text_b, text_a
            key = (canonical_pair[0], text_a, canonical_pair[1], text_b)
            if key in seen:
                continue
            seen.add(key)
            output.append(ParallelText(*key))

    def emit_explicit(mapping: dict, context: tuple[str, str] | None) -> bool:
        source_language = mapping.get("source_language", mapping.get("src_language"))
        target_language = mapping.get("target_language", mapping.get("tgt_language"))
        if source_language is not None or target_language is not None:
            try:
                source_language = _validate_language(source_language)
                target_language = _validate_language(target_language)
            except ValueError:
                issue("invalid_language")
                return True
            source = _first_value(mapping, _SOURCE_KEYS)
            target = _first_value(mapping, _TARGET_KEYS)
            if source is None or target is None:
                issue("missing_text")
            else:
                emit(source_language, source, target_language, target)
            return True
        if context is not None:
            source = _first_value(mapping, _SOURCE_KEYS)
            target = _first_value(mapping, _TARGET_KEYS)
            if source is not None or target is not None:
                if source is None or target is None:
                    issue("missing_text")
                else:
                    emit(context[0], source, context[1], target)
                return True
        return False

    def walk(node: object, context: tuple[str, str] | None = None) -> None:
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item, context)
            return
        if not isinstance(node, dict):
            issue("invalid_record")
            return

        explicit = emit_explicit(node, context)
        if not explicit:
            for language_a, language_b in configured:
                if language_a in node or language_b in node:
                    if language_a not in node or language_b not in node:
                        issue("missing_text")
                    else:
                        emit(language_a, node[language_a], language_b, node[language_b])

        nested: dict[str, tuple[str, str]] = {}
        for language_a, language_b in configured:
            for label in _pair_labels(language_a, language_b):
                if label not in node:
                    continue
                nested[label] = (
                    (language_a, language_b)
                    if label.startswith((f"{language_a}-", f"{language_a}/", f"{language_a}_to_"))
                    else (language_b, language_a)
                )

        for key, value in node.items():
            if key in nested:
                walk(value, nested[key])
            elif key in _CONTAINER_KEYS:
                walk(value, context)
            elif key == "translation" and not explicit and isinstance(value, (dict, list, tuple)):
                # Hugging Face translation datasets conventionally wrap their
                # language map in a singular ``translation`` field.  Keep the
                # scalar field reserved for explicit source/target records.
                walk(value, context)

    walk(record)
    if not output and not issues:
        issue("missing_text")
    return RecordExpansion(tuple(output), tuple(issues))
