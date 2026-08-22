"""Normalize heterogeneous JSONL records into explicit parallel text pairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import cast

from .record_metadata import inherit_record_metadata


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
    metadata: dict[str, object] = field(default_factory=lambda: {}, hash=False)


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
        if isinstance(raw_pair, (str, bytes)) or len(raw_pair) != 2:
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


def normalize_translation_directions(
    language_pairs: Sequence[Sequence[str]],
    translation_directions: Sequence[Sequence[str]] | None = None,
    *,
    bidirectional: bool = True,
    source_only_languages: Sequence[str] = (),
) -> tuple[tuple[str, str], ...]:
    """Resolve the directed training graph over configured storage pairs.

    ``language_pairs`` identifies which two texts coexist in each physical
    parallel record.  It does not imply that both decoder directions were
    trained.  An explicit direction list can therefore mix bidirectional and
    one-way edges in one dataset; the legacy global policy is used only when
    that list is absent.
    """

    pairs = normalize_language_pairs(language_pairs=language_pairs)
    if not isinstance(bidirectional, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("bidirectional must be a boolean")
    source_only = tuple(dict.fromkeys(_validate_language(item) for item in source_only_languages))
    known_languages = set(languages_from_pairs(pairs))
    unknown_source_only = sorted(set(source_only) - known_languages)
    if unknown_source_only:
        raise ValueError(
            "source_only_languages must appear in the configured language pairs; "
            f"{unknown_source_only} do not"
        )
    for pair in pairs:
        if pair[0] in source_only and pair[1] in source_only:
            raise ValueError(
                "at most one side of a language pair may be source-only; both sides "
                f"of {list(pair)!r} are source-only"
            )

    if translation_directions is None:
        directions: list[tuple[str, str]] = []
        source_only_set = set(source_only)
        for left, right in pairs:
            if left in source_only_set:
                directions.append((left, right))
            elif right in source_only_set:
                directions.append((right, left))
            else:
                directions.append((left, right))
                if bidirectional:
                    directions.append((right, left))
        return tuple(directions)

    allowed_edges = {frozenset(pair) for pair in pairs}
    directions = []
    seen: set[tuple[str, str]] = set()
    covered_edges: set[frozenset[str]] = set()
    source_only_set = set(source_only)
    for raw_direction in translation_directions:
        if isinstance(raw_direction, (str, bytes)) or len(raw_direction) != 2:
            raise ValueError(
                "each translation direction must contain source and target language keys; "
                f"got {raw_direction!r}"
            )
        source = _validate_language(raw_direction[0])
        target = _validate_language(raw_direction[1])
        direction = (source, target)
        edge = frozenset(direction)
        if source == target or edge not in allowed_edges:
            raise ValueError(
                "translation directions must belong to configured language pairs; "
                f"got {raw_direction!r}"
            )
        if direction in seen:
            raise ValueError(f"duplicate translation direction: {raw_direction!r}")
        if target in source_only_set:
            raise ValueError(f"source-only language {target!r} cannot be a translation target")
        seen.add(direction)
        covered_edges.add(edge)
        directions.append(direction)
    missing_edges = [pair for pair in pairs if frozenset(pair) not in covered_edges]
    if missing_edges:
        raise ValueError(
            "every configured language pair needs at least one translation direction; "
            f"missing={missing_edges!r}"
        )
    return tuple(directions)


def _pair_labels(language_a: str, language_b: str) -> tuple[str, ...]:
    return (
        f"{language_a}-{language_b}",
        f"{language_b}-{language_a}",
        f"{language_a}/{language_b}",
        f"{language_b}/{language_a}",
        f"{language_a}_to_{language_b}",
        f"{language_b}_to_{language_a}",
    )


def _first_value(mapping: Mapping[object, object], names: Sequence[str]) -> object | None:
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
        if isinstance(value, (list, tuple)):
            items = cast(Sequence[object], value)
            if all(isinstance(item, str) for item in items):
                return [item for item in items if isinstance(item, str)]
        return None

    def is_container(value: object) -> bool:
        return isinstance(value, (dict, list, tuple))

    def emit(
        language_a: str,
        value_a: object,
        language_b: str,
        value_b: object,
        metadata: dict[str, object],
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
            output.append(ParallelText(*key, metadata=inherit_record_metadata({}, metadata)))

    def emit_explicit(
        mapping: Mapping[object, object],
        context: tuple[str, str] | None,
        metadata: dict[str, object],
    ) -> bool:
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
                emit(source_language, source, target_language, target, metadata)
            return True
        if context is not None:
            source = _first_value(mapping, _SOURCE_KEYS)
            target = _first_value(mapping, _TARGET_KEYS)
            if source is not None or target is not None:
                if source is None or target is None:
                    issue("missing_text")
                else:
                    emit(context[0], source, context[1], target, metadata)
                return True
        return False

    def walk(
        node: object,
        context: tuple[str, str] | None = None,
        inherited_metadata: dict[str, object] | None = None,
    ) -> None:
        if isinstance(node, (list, tuple)):
            for item in cast(Sequence[object], node):
                walk(item, context, inherited_metadata)
            return
        if not isinstance(node, dict):
            issue("invalid_record")
            return

        mapping = cast(dict[object, object], node)

        metadata = inherit_record_metadata(mapping, inherited_metadata)
        explicit = emit_explicit(mapping, context, metadata)
        if not explicit:
            for language_a, language_b in configured:
                if language_a in mapping or language_b in mapping:
                    if language_a not in mapping or language_b not in mapping:
                        issue("missing_text")
                    else:
                        emit(
                            language_a,
                            mapping[language_a],
                            language_b,
                            mapping[language_b],
                            metadata,
                        )

        nested: dict[str, tuple[str, str]] = {}
        for language_a, language_b in configured:
            for label in _pair_labels(language_a, language_b):
                if label not in mapping:
                    continue
                nested[label] = (
                    (language_a, language_b)
                    if label.startswith((f"{language_a}-", f"{language_a}/", f"{language_a}_to_"))
                    else (language_b, language_a)
                )

        for key, value in mapping.items():
            if not isinstance(key, str):
                continue
            if key in nested:
                walk(value, nested[key], metadata)
            elif key in _CONTAINER_KEYS:
                walk(value, context, metadata)
            elif key == "translation" and not explicit and is_container(value):
                # Hugging Face translation datasets conventionally wrap their
                # language map in a singular ``translation`` field.  Keep the
                # scalar field reserved for explicit source/target records.
                walk(value, context, metadata)

    walk(record)
    if not output and not issues:
        issue("missing_text")
    return RecordExpansion(tuple(output), tuple(issues))
