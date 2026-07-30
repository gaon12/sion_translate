"""Structured-span parsing and reversible slot protection.

Translation corpora contain values that should normally survive translation
verbatim: numbers, URLs, software placeholders, HTML entities, and localization
placeables.  Keeping one parser here prevents training-time protection, corpus
quality checks, and inference-time restoration from silently disagreeing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable, Sequence


_WHITESPACE = re.compile(r"\s+")
_ENTITY = re.compile(r"&(?:#\d+|#[xX][0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);")
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9_.+-])"
    r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9_.-])"
)
_URL = re.compile(r"https?://[^\s<>\"']+")
_NAMED_PERCENT = re.compile(r"%[A-Za-z_][A-Za-z0-9_]{1,}")
_PRINTF = re.compile(
    r"%(?:\d+\$)?[-+#0 ']*(?:\d+|\*)?(?:\.(?:\d+|\*))?"
    r"(?:hh|h|ll|l|j|z|t|L)?[diuoxXfFeEgGaAcspnCS]"
)
_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?:[A-Z]{2,}[A-Z0-9_-]*|"
    r"[A-Za-z][A-Za-z0-9]*(?:[_.-][A-Za-z0-9]+)+|"
    r"[A-Za-z][A-Za-z0-9_.-]*\d[A-Za-z0-9_.-]*|"
    r"\d[A-Za-z][A-Za-z0-9_.-]*)"
    r"(?![A-Za-z0-9_.-])"
)
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:[$€£¥₩₹]\s*)?[+-]?"
    r"(?:\d{1,4}(?:[.,:/]\d+)+|\d+(?:[.,]\d+)?)"
    r"(?:\s?(?:%|‰|°[CF]?|"
    r"KiB|MiB|GiB|TiB|KB|MB|GB|TB|B|"
    r"kg|mg|µg|μg|g|km|cm|mm|m|"
    r"ms|µs|μs|ns|sec|min|hrs?|days?|"
    r"kW|MW|GW|W|kWh|mAh|Ah|"
    r"Hz|kHz|MHz|GHz|Pa|kPa|MPa|"
    r"ml|mL|cl|cL|dl|dL|L|"
    r"개|명|원|엔|달러|유로|퍼센트))?"
    r"(?:\s?(?:-|–|—|~|〜)\s?"
    r"(?:[$€£¥₩₹]\s*)?[+-]?\d+(?:[.,]\d+)?"
    r"(?:\s?(?:%|‰|°[CF]?|KiB|MiB|GiB|TiB|KB|MB|GB|TB|B|"
    r"kg|mg|µg|μg|g|km|cm|mm|m|ms|sec|min|hrs?|days?))?)?"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
# These suffixes carry ordinary language meaning and should be translated
# (원→ウォン, 개→個, 명→人, ...).  Protecting the complete ``12,500원``
# surface restores the Korean suffix after Japanese generation and can even
# produce duplicated units such as ``12,500원ウォン``.  Machine-readable SI
# units (mg, mL, GB, ...) remain part of the protected span.
_TRANSLATABLE_NUMBER_SUFFIX = re.compile(
    r"(?:퍼센트|달러|유로|개|명|원|엔)$",
    re.IGNORECASE,
)

_URL_TRAILING_PUNCTUATION = ".,!?;:。！？、，；："
_URL_KOREAN_PARTICLES = tuple(
    sorted(
        {
            "으로부터",
            "에게서",
            "에서부터",
            "까지도",
            "으로는",
            "으로도",
            "에서는",
            "에게는",
            "부터",
            "까지",
            "처럼",
            "보다",
            "으로",
            "에게",
            "한테",
            "께서",
            "에서",
            "와",
            "과",
            "을",
            "를",
            "은",
            "는",
            "이",
            "가",
            "의",
            "에",
            "도",
            "만",
            "로",
        },
        key=len,
        reverse=True,
    )
)

_OPAQUE_KINDS = {"placeable", "template", "url", "email", "entity"}
_CRITICAL_KINDS = {
    "placeable",
    "template",
    "placeholder",
    "percent_placeholder",
    "printf",
    "entity",
}
_PRIORITY = {
    "html": 140,
    "placeable": 130,
    "template": 130,
    "url": 115,
    "email": 115,
    "entity": 110,
    "placeholder": 105,
    "percent_placeholder": 100,
    "printf": 100,
    "number": 90,
    "identifier": 80,
}


@dataclass(frozen=True, slots=True)
class StructuredSpan:
    """One structured surface and its exact location in the original text."""

    start: int
    end: int
    surface: str
    kind: str
    key: str


def _canonical_surface(surface: str, kind: str) -> str:
    if kind in {"placeable", "template", "placeholder", "html"}:
        return _WHITESPACE.sub(" ", surface.strip())
    return surface


def _span(start: int, end: int, surface: str, kind: str) -> StructuredSpan:
    canonical = _canonical_surface(surface, kind)
    return StructuredSpan(start, end, surface, kind, f"{kind}\0{canonical}")


def _balanced_placeables(text: str) -> Iterable[StructuredSpan]:
    """Yield balanced ``{...}`` and ``${...}`` spans, respecting quoted braces."""

    for brace_start, char in enumerate(text):
        if char != "{":
            continue
        start = brace_start - 1 if brace_start and text[brace_start - 1] == "$" else brace_start
        if start < brace_start and start > 0 and text[start - 1] == "\\":
            continue
        depth = 0
        quote = ""
        escaped = False
        for cursor in range(brace_start, len(text)):
            current = text[cursor]
            if escaped:
                escaped = False
                continue
            if current == "\\":
                escaped = True
                continue
            if quote:
                if current == quote:
                    quote = ""
                continue
            if current in {'"', "'"}:
                quote = current
                continue
            if current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    surface = text[start : cursor + 1]
                    if surface.startswith("${"):
                        kind = "template"
                    else:
                        body = surface[1:-1].strip()
                        simple = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*|\d+", body)
                        kind = "placeholder" if simple else "placeable"
                    yield _span(start, cursor + 1, surface, kind)
                    break


def _html_spans(text: str) -> Iterable[StructuredSpan]:
    """Yield HTML/XML tags without stopping at ``>`` inside quoted attributes."""

    cursor = 0
    while cursor < len(text):
        start = text.find("<", cursor)
        if start < 0:
            return
        if text.startswith("<slot_", start):
            cursor = start + 1
            continue
        next_index = start + 1
        if next_index >= len(text) or not (
            text[next_index].isalpha() or text[next_index] in {"/", "!", "?"}
        ):
            cursor = start + 1
            continue
        quote = ""
        escaped = False
        for end in range(next_index, len(text)):
            current = text[end]
            if escaped:
                escaped = False
                continue
            if current == "\\":
                escaped = True
                continue
            if quote:
                if current == quote:
                    quote = ""
                continue
            if current in {'"', "'"}:
                quote = current
            elif current == ">":
                surface = text[start : end + 1]
                yield _span(start, end + 1, surface, "html")
                cursor = end + 1
                break
        else:
            cursor = start + 1


def _trim_url(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1] in _URL_TRAILING_PUNCTUATION:
        end -= 1
    pairs = {")": "(", "]": "[", "}": "{", ">": "<"}
    while end > start and text[end - 1] in pairs:
        closer = text[end - 1]
        surface = text[start:end]
        if surface.count(closer) <= surface.count(pairs[closer]):
            break
        end -= 1
    candidate = text[start:end]
    for particle in _URL_KOREAN_PARTICLES:
        if not candidate.endswith(particle):
            continue
        prefix = candidate[: -len(particle)]
        if prefix and prefix[-1].isascii() and (prefix[-1].isalnum() or prefix[-1] in "/=_-"):
            end -= len(particle)
            break
    return end


def _regex_spans(text: str) -> Iterable[StructuredSpan]:
    for match in _ENTITY.finditer(text):
        yield _span(match.start(), match.end(), match.group(0), "entity")
    for match in _EMAIL.finditer(text):
        yield _span(match.start(), match.end(), match.group(0), "email")
    for match in _URL.finditer(text):
        end = _trim_url(text, match.start(), match.end())
        if end > match.start() + len("https://"):
            yield _span(match.start(), end, text[match.start() : end], "url")
    for match in _NAMED_PERCENT.finditer(text):
        yield _span(match.start(), match.end(), match.group(0), "percent_placeholder")
    for match in _PRINTF.finditer(text):
        yield _span(match.start(), match.end(), match.group(0), "printf")
    for match in _NUMBER.finditer(text):
        surface = match.group(0)
        suffix = _TRANSLATABLE_NUMBER_SUFFIX.search(surface)
        end = match.end() - (len(suffix.group(0)) if suffix is not None else 0)
        if end > match.start():
            yield _span(match.start(), end, text[match.start() : end], "number")
    for match in _IDENTIFIER.finditer(text):
        yield _span(match.start(), match.end(), match.group(0), "identifier")


def extract_structured_spans(text: str) -> list[StructuredSpan]:
    """Return all structured candidates, including critical spans inside HTML tags."""

    candidates = [*_balanced_placeables(text), *_html_spans(text), *_regex_spans(text)]
    unique: dict[tuple[int, int, str], StructuredSpan] = {}
    for candidate in candidates:
        unique[(candidate.start, candidate.end, candidate.kind)] = candidate
    return sorted(unique.values(), key=lambda item: (item.start, -item.end, item.kind))


def _overlaps(left: StructuredSpan, right: StructuredSpan) -> bool:
    return left.start < right.end and right.start < left.end


def _select_non_overlapping(spans: Iterable[StructuredSpan]) -> list[StructuredSpan]:
    selected: list[StructuredSpan] = []
    ordered = sorted(
        spans,
        key=lambda item: (
            -_PRIORITY[item.kind],
            -(item.end - item.start),
            item.start,
            item.key,
        ),
    )
    for candidate in ordered:
        if not any(_overlaps(candidate, claimed) for claimed in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda item: item.start)


def _signature_spans(text: str) -> list[StructuredSpan]:
    candidates = extract_structured_spans(text)
    opaque = [span for span in candidates if span.kind in _OPAQUE_KINDS]
    result: list[StructuredSpan] = []
    for candidate in candidates:
        containers = [
            parent
            for parent in opaque
            if parent is not candidate
            and parent.start <= candidate.start
            and candidate.end <= parent.end
        ]
        if containers and candidate.kind not in _CRITICAL_KINDS:
            continue
        result.append(candidate)
    return result


def structured_signature(
    text: str,
    *,
    include_numbers: bool = True,
) -> Counter[str]:
    """Return the authoritative structured-value multiset.

    ``include_numbers=False`` is used by rewards that already score numeric
    preservation separately. Keeping both modes on this parser prevents
    training, reranking, corpus checks, and reversible masking from maintaining
    competing regular-expression definitions.
    """

    return Counter(
        span.key for span in _signature_spans(text) if include_numbers or span.kind != "number"
    )


def structured_similarity(left: str, right: str) -> tuple[float, bool]:
    """Return multiset Jaccard similarity and whether a critical token differs."""

    left_counter = structured_signature(left)
    right_counter = structured_signature(right)
    keys = left_counter.keys() | right_counter.keys()
    if not keys:
        return 1.0, False
    shared = sum(min(left_counter[key], right_counter[key]) for key in keys)
    total = sum(max(left_counter[key], right_counter[key]) for key in keys)
    critical_keys = {
        span.key
        for span in [*_signature_spans(left), *_signature_spans(right)]
        if span.kind in _CRITICAL_KINDS
    }
    critical_mismatch = any(left_counter[key] != right_counter[key] for key in critical_keys)
    return shared / total, critical_mismatch


def _replace(
    text: str,
    spans: Sequence[StructuredSpan],
    key_to_slot: dict[str, str],
) -> str:
    replacements = [
        (span.start, span.end, key_to_slot[span.key]) for span in spans if span.key in key_to_slot
    ]
    result = text
    for start, end, slot in sorted(replacements, reverse=True):
        result = f"{result[:start]}{slot}{result[end:]}"
    return result


def protect_shared_structured_spans(
    left: str,
    right: str,
    *,
    slot_symbols: Sequence[str],
) -> tuple[str, str]:
    """Replace structured values shared by a parallel pair with matching slots."""

    left_candidates = extract_structured_spans(left)
    right_candidates = extract_structured_spans(right)
    shared_keys = {span.key for span in left_candidates} & {span.key for span in right_candidates}
    if not shared_keys or not slot_symbols:
        return left, right
    selected_left = _select_non_overlapping(
        span for span in left_candidates if span.key in shared_keys
    )
    selected_right = _select_non_overlapping(
        span for span in right_candidates if span.key in shared_keys
    )
    effective = {span.key for span in selected_left} & {span.key for span in selected_right}
    ordered_keys = list(dict.fromkeys(span.key for span in selected_left if span.key in effective))[
        : len(slot_symbols)
    ]
    key_to_slot = dict(zip(ordered_keys, slot_symbols[: len(ordered_keys)], strict=True))
    return (
        _replace(left, selected_left, key_to_slot),
        _replace(right, selected_right, key_to_slot),
    )


def mask_structured_spans(
    text: str,
    *,
    slot_symbols: Sequence[str],
) -> tuple[str, dict[str, str]]:
    """Mask source-side structured values and return ``slot -> exact surface``."""

    selected = _select_non_overlapping(extract_structured_spans(text))
    ordered_keys = list(dict.fromkeys(span.key for span in selected))[: len(slot_symbols)]
    key_to_slot = dict(zip(ordered_keys, slot_symbols[: len(ordered_keys)], strict=True))
    first_surface: dict[str, str] = {}
    for span in selected:
        first_surface.setdefault(span.key, span.surface)
    slot_to_surface = {slot: first_surface[key] for key, slot in key_to_slot.items()}
    return _replace(text, selected, key_to_slot), slot_to_surface
