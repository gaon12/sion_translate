"""Enforce glossary terminology during translation.

This feature maps proper names and specialized terms to required target forms,
serving the same purpose as glossary features in commercial translation APIs.
The hard-constraint path reuses the project's protected ``<slot_n>`` tokens:

1. Replace each configured source-language surface with ``<slot_k>`` and record
   the mapping from that slot to its required target-language surface.
2. Translate the protected sentence. Protect-span/TETM training teaches the
   model to retain ``<slot_n>`` tokens in its output.
3. Replace each output slot with its required target surface.

This is deterministic because the term is removed before translation and
restored from the configured mapping instead of relying on the model's lexical
choice. It requires no additional inference-time training.

Each JSON glossary entry stores surfaces by language:

    [
      {"ko": "인공지능", "ja": "人工知能"},
      {"ko": "심층학습", "ja": "深層学習"}
    ]

When both language keys exist, the entry applies in either configured direction.
"""

# Glossary files accept multiple JSON layouts validated at load time.
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sion_translate.language_tags import (
    canonicalize_language_pair,
    canonicalize_language_tag,
)
from sion_translate.scripts_registry import uses_substring_term_matching

# ---------------------------------------------------------------------------
# Soft hint format
# ---------------------------------------------------------------------------
#
# The slot mechanism above is a hard constraint applied at inference time: the
# term is cut out of the source and pasted back afterwards. That guarantees the
# surface but the model never sees the term, so it cannot inflect around it —
# ``<slot_0>를`` may need 을 rather than 를 once the surface is restored.
#
# A soft hint instead shows the model both sides and lets it produce agreeing
# morphology. It has to be trained for, which is why it only becomes available
# with a from-scratch run.
#
# The prefix uses control tokens the tokenizer already reserves, so no
# vocabulary change is needed:
#
#     <glossary> 사과 <protect> Apple <glossary> 배 <protect> Pear <seg> 원문
#
# Both the training-data builder and the serving path must produce byte-identical
# prefixes, so the format is defined once here. A mismatch between the two is the
# quiet failure mode that makes a hint feature look trained when it is not.
GLOSSARY_TOKEN = "<glossary>"
PROTECT_TOKEN = "<protect>"
SEGMENT_TOKEN = "<seg>"

_HINT_PREFIX = re.compile(
    rf"^\s*(?:{re.escape(GLOSSARY_TOKEN)}\s*(?P<body>.*?))?\s*{re.escape(SEGMENT_TOKEN)}\s*",
    re.DOTALL,
)


def format_hint_prefix(pairs: Sequence[tuple[str, str]]) -> str:
    """Render glossary hints as an encoder-input prefix.

    ``pairs`` is ``[(source_term, target_term), ...]``. Returns the empty string
    for no pairs, so callers can prepend unconditionally.
    """

    if not pairs:
        return ""
    parts: list[str] = []
    for source_term, target_term in pairs:
        source_term, target_term = source_term.strip(), target_term.strip()
        if not source_term or not target_term:
            raise ValueError(f"hint terms must be non-empty; got {(source_term, target_term)!r}")
        for term in (source_term, target_term):
            if any(token in term for token in (GLOSSARY_TOKEN, PROTECT_TOKEN, SEGMENT_TOKEN)):
                raise ValueError(f"hint term must not contain a control token: {term!r}")
        parts.append(f"{GLOSSARY_TOKEN} {source_term} {PROTECT_TOKEN} {target_term}")
    return " ".join(parts) + f" {SEGMENT_TOKEN} "


def build_hinted_source(text: str, pairs: Sequence[tuple[str, str]]) -> str:
    """``text`` with a hint prefix, or unchanged when there are no pairs."""

    prefix = format_hint_prefix(pairs)
    return f"{prefix}{text}" if prefix else text


def parse_hinted_source(text: str) -> tuple[list[tuple[str, str]], str]:
    """Inverse of :func:`build_hinted_source`.

    Returns ``(pairs, source_text)``. Text without a ``<seg>`` marker is returned
    unchanged with no pairs, so this is safe to call on ordinary input.
    """

    match = _HINT_PREFIX.match(text)
    if match is None:
        return [], text
    body = match.group("body") or ""
    remainder = text[match.end() :]
    pairs: list[tuple[str, str]] = []
    for chunk in body.split(GLOSSARY_TOKEN):
        chunk = chunk.strip()
        if not chunk:
            continue
        if PROTECT_TOKEN not in chunk:
            raise ValueError(f"hint entry is missing {PROTECT_TOKEN}: {chunk!r}")
        source_term, target_term = chunk.split(PROTECT_TOKEN, 1)
        source_term, target_term = source_term.strip(), target_term.strip()
        if not source_term or not target_term:
            raise ValueError(f"hint entry terms must be non-empty: {chunk!r}")
        if PROTECT_TOKEN in target_term:
            raise ValueError(f"hint entry contains more than one {PROTECT_TOKEN}: {chunk!r}")
        pairs.append((source_term, target_term))
    return pairs, remainder


def adherence(
    hypotheses: Sequence[str],
    required: Sequence[Sequence[str]],
    *,
    case_insensitive: bool = False,
) -> dict[str, float | int]:
    """How often required target terms actually appear in the output.

    ``required[i]`` lists the target surfaces sentence ``i`` was hinted with.
    A soft hint is not a guarantee, so this has to be measured rather than
    assumed; sentences with no requirement count as satisfied because there was
    nothing to violate.
    """

    if len(hypotheses) != len(required):
        raise ValueError(f"{len(hypotheses)} hypotheses against {len(required)} requirement lists")
    term_total = 0
    term_hits = 0
    sentence_total = 0
    sentence_hits = 0
    for hypothesis, terms in zip(hypotheses, required, strict=True):
        haystack = hypothesis.casefold() if case_insensitive else hypothesis
        present = 0
        for term in terms:
            needle = term.casefold() if case_insensitive else term
            term_total += 1
            if needle and needle in haystack:
                term_hits += 1
                present += 1
        sentence_total += 1
        if present == len(terms):
            sentence_hits += 1
    return {
        "terms": term_total,
        "term_hits": term_hits,
        "term_rate": term_hits / term_total if term_total else 1.0,
        "sentences": sentence_total,
        "sentence_hits": sentence_hits,
        "sentence_rate": sentence_hits / sentence_total if sentence_total else 1.0,
    }


def rank_terms_for_hinting(
    pairs: Sequence[tuple[str, str]],
    corpus_counts: Counter[str] | None = None,
) -> list[tuple[str, str]]:
    """Order term pairs by how much a hint is likely to help.

    Rarer source terms and longer target surfaces come first: a common word the
    model already translates correctly gains nothing from a hint, while a rare
    proper noun or a multi-word target is exactly where it fails. Ties fall back
    to the source surface so the order is deterministic.
    """

    counts = corpus_counts or Counter()

    def key(pair: tuple[str, str]) -> tuple[int, int, str]:
        source_term, target_term = pair
        return (counts.get(source_term, 0), -len(target_term), source_term)

    return sorted(pairs, key=key)


@dataclass(frozen=True)
class Glossary:
    """Store terminology with one surface per language.

    entries: [{"ko": "인공지능", "ja": "人工知能"}, ...]
    """

    entries: tuple[dict[str, str], ...]

    def __post_init__(self) -> None:
        normalized_entries: list[dict[str, str]] = []
        for entry_index, entry in enumerate(self.entries):
            normalized: dict[str, str] = {}
            original_keys: dict[str, str] = {}
            for raw_language, surface in entry.items():
                language = canonicalize_language_tag(
                    raw_language,
                    field=f"glossary entry[{entry_index}] language",
                )
                if language in normalized:
                    raise ValueError(
                        "glossary entry contains duplicate language aliases after BCP 47 "
                        f"canonicalization: {original_keys[language]!r}, {raw_language!r}"
                    )
                original_keys[language] = raw_language
                normalized[language] = surface
            normalized_entries.append(normalized)
        object.__setattr__(self, "entries", tuple(normalized_entries))

    def __len__(self) -> int:
        return len(self.entries)

    def for_direction(self, source_language: str, target_language: str) -> list[tuple[str, str]]:
        """Return ``(source, target)`` surfaces ordered by source length.

        Replacing long terms first prevents a shorter term from consuming part
        of a longer one, such as "인공지능" inside "인공지능학회".
        """
        source_language, target_language = canonicalize_language_pair(
            (source_language, target_language),
            field="glossary direction",
        )
        pairs: list[tuple[str, str]] = []
        for entry in self.entries:
            source = entry.get(source_language)
            target = entry.get(target_language)
            if source and target:
                pairs.append((source.strip(), target.strip()))
        # Keep the first mapping for a duplicate source and sort longest first.
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for source, target in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
            if source and source not in seen:
                seen.add(source)
                unique.append((source, target))
        return unique


def load_glossary(path: str | Path) -> Glossary:
    """Load a language-keyed JSON glossary.

    The canonical format is a list such as
    ``[{"ko": "...", "ja": "..."}, ...]``. A plain source-to-target mapping
    omits language identity and is therefore not accepted at this boundary.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError('glossary must be a list in the form [{"ko": ..., "ja": ...}, ...]')
    entries: list[dict[str, str]] = []
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("each glossary entry must map language tags to term surfaces")
        cleaned = {
            str(key): str(value)
            for key, value in row.items()
            if isinstance(value, str) and value.strip()
        }
        if len(cleaned) >= 2:
            entries.append(cleaned)
    return Glossary(tuple(entries))


def _match_positions(text: str, term: str, language: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` positions where ``term`` occurs in ``text``.

    Languages without word boundaries, including CJK scripts, use substring
    matching. Other languages respect word boundaries so English "cat", for
    example, does not match "category".
    """
    if not term:
        return []
    substring_matching = uses_substring_term_matching(language)
    if substring_matching:
        pattern = re.escape(term)
    else:
        # Match case-insensitively only where adjacent characters are not word characters.
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
    flags = 0 if substring_matching else re.IGNORECASE
    return [(match.start(), match.end()) for match in re.finditer(pattern, text, flags)]


def apply_source_placeholders(
    text: str,
    glossary: Glossary,
    *,
    source_language: str,
    target_language: str,
    slot_symbols: Sequence[str],
) -> tuple[str, dict[str, str]]:
    """Replace source glossary terms with protected slot tokens.

    Return ``(protected text, {slot token: target surface})``. Repeated
    occurrences of one term reuse a slot. Terms beyond the available slot count
    remain unchanged and are not enforced.
    """
    directional = glossary.for_direction(source_language, target_language)
    if not directional:
        return text, {}

    # Track spans claimed by longer terms to prevent overlapping replacements.
    claimed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < c_end and c_start < end for c_start, c_end in claimed)

    # Assign one slot per source term and reuse it for repeated occurrences.
    term_to_slot: dict[str, str] = {}
    slot_to_target: dict[str, str] = {}
    replacements: list[tuple[int, int, str]] = []  # (start, end, slot token)

    for source_term, target_term in directional:
        for start, end in _match_positions(text, source_term, source_language):
            if overlaps(start, end):
                continue
            slot = term_to_slot.get(source_term)
            if slot is None:
                if len(term_to_slot) >= len(slot_symbols):
                    # Leave this term unconstrained when no protected slot remains.
                    break
                slot = slot_symbols[len(term_to_slot)]
                term_to_slot[source_term] = slot
                slot_to_target[slot] = target_term
            claimed.append((start, end))
            replacements.append((start, end, slot))

    if not replacements:
        return text, {}

    # Replace from right to left so earlier character offsets remain stable.
    replacements.sort(key=lambda item: item[0], reverse=True)
    result = text
    for start, end, slot in replacements:
        # Surround the slot with spaces to keep adjacent text from changing tokenization.
        result = f"{result[:start]} {slot} {result[end:]}"
    return result, slot_to_target


def restore_targets(translated: str, slot_to_target: dict[str, str]) -> tuple[str, list[str]]:
    """Restore target surfaces into protected slots in a translated sentence.

    Return ``(restored text, missing target surfaces)``. A missing target means
    the model failed to preserve its slot; callers can warn or apply an explicit
    fallback policy.
    """
    if not slot_to_target:
        return translated, []
    result = translated
    missing: list[str] = []
    for slot, target in slot_to_target.items():
        if slot in result:
            result = result.replace(slot, target)
        else:
            missing.append(target)
    # Collapse extra whitespace left by the spaces surrounding protected slots.
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result, missing
