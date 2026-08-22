"""BCP 47 language-tag validation and canonical casing.

The project uses language tags in configuration, JSONL records, directory names,
tokenizer control symbols, and exported model metadata.  Keeping the parser here
prevents those surfaces from accepting different language identities.

This module validates the RFC 5646 syntax without consulting the IANA registry.
That is deliberate: private or newly registered languages remain usable offline,
while malformed or ambiguously cased identities are still rejected or normalized.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence, cast


_ALPHA = re.compile(r"^[A-Za-z]+$")
_ALNUM = re.compile(r"^[A-Za-z0-9]+$")
_DIGIT = re.compile(r"^[0-9]+$")

_GRANDFATHERED = (
    "art-lojban",
    "cel-gaulish",
    "en-GB-oed",
    "i-ami",
    "i-bnn",
    "i-default",
    "i-enochian",
    "i-hak",
    "i-klingon",
    "i-lux",
    "i-mingo",
    "i-navajo",
    "i-pwn",
    "i-tao",
    "i-tay",
    "i-tsu",
    "no-bok",
    "no-nyn",
    "sgn-BE-FR",
    "sgn-BE-NL",
    "sgn-CH-DE",
    "zh-guoyu",
    "zh-hakka",
    "zh-min",
    "zh-min-nan",
    "zh-xiang",
)
_GRANDFATHERED_BY_CASEFOLD = {tag.casefold(): tag for tag in _GRANDFATHERED}


class LanguageTagError(ValueError):
    """A value is not a well-formed BCP 47 language tag."""


@dataclass(frozen=True, slots=True)
class LanguageTag:
    """Parsed language identity with RFC-recommended casing."""

    canonical: str
    language: str
    extlangs: tuple[str, ...] = ()
    script: str | None = None
    region: str | None = None
    variants: tuple[str, ...] = ()
    extensions: tuple[tuple[str, tuple[str, ...]], ...] = ()
    private_use: tuple[str, ...] = ()
    grandfathered: bool = False


def _error(value: object, field: str, reason: str) -> LanguageTagError:
    return LanguageTagError(
        f"{field} must be a well-formed BCP 47 language tag; {reason}: {value!r}"
    )


def _is_alpha(value: str, length: int | range) -> bool:
    allowed = len(value) == length if isinstance(length, int) else len(value) in length
    return allowed and _ALPHA.fullmatch(value) is not None


def _is_alnum(value: str, length: range) -> bool:
    return len(value) in length and _ALNUM.fullmatch(value) is not None


def parse_language_tag(value: object, *, field: str = "language") -> LanguageTag:
    """Parse *value* as RFC 5646 syntax and return its canonical casing.

    Registry membership is intentionally not required.  For example, a new
    language subtag can be used before this package updates, and project-specific
    identities can use the standard ``x-...`` private-use form.
    """

    if not isinstance(value, str):
        raise _error(value, field, "expected a string")
    if not value or value != value.strip():
        raise _error(value, field, "empty or surrounding whitespace")
    if not value.isascii() or len(value) > 255:
        raise _error(value, field, "expected at most 255 ASCII characters")
    if "_" in value or "--" in value or value.startswith("-") or value.endswith("-"):
        raise _error(value, field, "subtags must be separated by single hyphens")

    grandfathered = _GRANDFATHERED_BY_CASEFOLD.get(value.casefold())
    if grandfathered is not None:
        primary = grandfathered.split("-", 1)[0].lower()
        return LanguageTag(
            canonical=grandfathered,
            language=primary,
            grandfathered=True,
        )

    raw = value.split("-")
    if any(not part or len(part) > 8 or _ALNUM.fullmatch(part) is None for part in raw):
        raise _error(value, field, "each subtag must contain 1-8 ASCII letters or digits")

    if raw[0].casefold() == "x":
        if len(raw) == 1:
            raise _error(value, field, "private-use prefix 'x' requires at least one subtag")
        private_use = tuple(part.lower() for part in raw[1:])
        canonical = "x-" + "-".join(private_use)
        return LanguageTag(canonical, "x", private_use=private_use)

    primary = raw[0]
    if not (
        _is_alpha(primary, range(2, 4)) or _is_alpha(primary, 4) or _is_alpha(primary, range(5, 9))
    ):
        raise _error(value, field, "primary language must contain 2-8 ASCII letters")
    language = primary.lower()
    index = 1

    extlangs: list[str] = []
    if len(primary) in (2, 3):
        while index < len(raw) and len(extlangs) < 3 and _is_alpha(raw[index], 3):
            extlangs.append(raw[index].lower())
            index += 1

    script: str | None = None
    if index < len(raw) and _is_alpha(raw[index], 4):
        script = raw[index].title()
        index += 1

    region: str | None = None
    if index < len(raw):
        candidate = raw[index]
        if _is_alpha(candidate, 2):
            region = candidate.upper()
            index += 1
        elif len(candidate) == 3 and _DIGIT.fullmatch(candidate) is not None:
            region = candidate
            index += 1

    variants: list[str] = []
    seen_variants: set[str] = set()
    while index < len(raw):
        candidate = raw[index]
        is_variant = _is_alnum(candidate, range(5, 9)) or (
            len(candidate) == 4
            and candidate[0].isdigit()
            and _ALNUM.fullmatch(candidate) is not None
        )
        if not is_variant:
            break
        normalized = candidate.lower()
        if normalized in seen_variants:
            raise _error(value, field, f"duplicate variant subtag {candidate!r}")
        seen_variants.add(normalized)
        variants.append(normalized)
        index += 1

    extensions: list[tuple[str, tuple[str, ...]]] = []
    seen_singletons: set[str] = set()
    while index < len(raw) and len(raw[index]) == 1 and raw[index].casefold() != "x":
        singleton = raw[index].lower()
        if not singleton.isalnum():
            raise _error(value, field, f"invalid extension singleton {raw[index]!r}")
        if singleton in seen_singletons:
            raise _error(value, field, f"duplicate extension singleton {singleton!r}")
        seen_singletons.add(singleton)
        index += 1
        extension_subtags: list[str] = []
        while index < len(raw) and _is_alnum(raw[index], range(2, 9)):
            extension_subtags.append(raw[index].lower())
            index += 1
        if not extension_subtags:
            raise _error(value, field, f"extension {singleton!r} requires a 2-8 character subtag")
        extensions.append((singleton, tuple(extension_subtags)))

    private_use: tuple[str, ...] = ()
    if index < len(raw) and raw[index].casefold() == "x":
        if index + 1 >= len(raw):
            raise _error(value, field, "private-use prefix 'x' requires at least one subtag")
        private_use = tuple(part.lower() for part in raw[index + 1 :])
        index = len(raw)

    if index != len(raw):
        raise _error(value, field, f"unexpected subtag {raw[index]!r}")

    # RFC 5646 canonical form orders extension sequences by singleton.  Without
    # this step, tags that differ only in extension order become two model
    # identities even though they denote the same language tag.
    extensions.sort(key=lambda item: item[0])
    canonical_parts = [language, *extlangs]
    if script is not None:
        canonical_parts.append(script)
    if region is not None:
        canonical_parts.append(region)
    canonical_parts.extend(variants)
    for singleton, extension_values in extensions:
        canonical_parts.extend((singleton, *extension_values))
    if private_use:
        canonical_parts.extend(("x", *private_use))
    canonical = "-".join(canonical_parts)
    return LanguageTag(
        canonical=canonical,
        language=language,
        extlangs=tuple(extlangs),
        script=script,
        region=region,
        variants=tuple(variants),
        extensions=tuple(extensions),
        private_use=private_use,
    )


def canonicalize_language_tag(value: object, *, field: str = "language") -> str:
    """Return one stable spelling for a well-formed language tag."""

    return parse_language_tag(value, field=field).canonical


def canonicalize_language_tags(
    values: Sequence[object],
    *,
    field: str = "languages",
    reject_duplicates: bool = True,
) -> tuple[str, ...]:
    """Canonicalize an ordered language list and handle canonical duplicates."""

    if isinstance(values, (str, bytes)):
        raise LanguageTagError(f"{field} must be a sequence of BCP 47 language tags")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        language = canonicalize_language_tag(value, field=f"{field}[{index}]")
        if language in seen:
            if reject_duplicates:
                raise LanguageTagError(
                    f"{field} must not contain duplicate language identities after "
                    f"canonicalization; duplicate={language!r}"
                )
            continue
        seen.add(language)
        normalized.append(language)
    return tuple(normalized)


def canonicalize_language_pair(
    value: object,
    *,
    field: str = "language pair",
) -> tuple[str, str]:
    """Canonicalize one ordered pair and require two distinct identities."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LanguageTagError(f"{field} must be a two-item language sequence of BCP 47 tags")
    items = cast(Sequence[object], value)
    if len(items) != 2:
        raise LanguageTagError(f"{field} must be a two-item language sequence of BCP 47 tags")
    languages = canonicalize_language_tags(
        items,
        field=field,
        reject_duplicates=True,
    )
    return languages[0], languages[1]


def is_well_formed_language_tag(value: object) -> bool:
    """Whether *value* has well-formed offline BCP 47 syntax."""

    try:
        parse_language_tag(value)
    except LanguageTagError:
        return False
    return True


__all__ = [
    "LanguageTag",
    "LanguageTagError",
    "canonicalize_language_pair",
    "canonicalize_language_tag",
    "canonicalize_language_tags",
    "is_well_formed_language_tag",
    "parse_language_tag",
]
