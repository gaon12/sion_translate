"""Detect function morphemes left stranded by a deleted placeholder.

Visual-novel scripts interpolate the player's and heroine's names at runtime, so
the exported text carries a variable where the name belongs. When an extractor
drops the variable instead of substituting it, the grammar around it survives and
the noun does not::

    늦은 밤, 의 숲속에서 사냥꾼이 집으로 돌아갈 준비를 해.
    深夜、 の森の中、猟師が家に帰る準備してる.

Both sides lose the same noun, so the pair still looks like a translation of
itself and every similarity check passes it. Training on it teaches the model to
emit the hole. The detectable signature is grammatical rather than semantic: a
case particle needs a host noun, so a particle with nothing in front of it means
the host was deleted.

How "nothing in front of it" is spelled depends on the writing system, so the
rule follows :data:`sion_translate.scripts_registry.SPACELESS_SCRIPTS` rather
than the language name:

*space-using scripts*
    The stranded particle becomes its own whitespace-delimited token. Korean
    never writes a particle as a free word, so ``의`` alone is conclusive.

*spaceless scripts*
    There are no tokens to inspect, so position is the signal: whitespace
    immediately before a particle. Segmenter spacing would fire on that too, so
    the text is first passed through ``collapse_spurious_spaces``; what remains
    is whitespace the segmenter cannot explain.

Known limitation: a hole between two spaceless characters (``恋人の が``) is
removed by the collapse along with genuine segmenter spacing, so it is not
detected here. Those rows are caught instead by the spurious-space density check
in ``recover_shard.py``, which drops targets carrying isolated spacing.

The tables are deliberately language-specific and live apart from
:mod:`sion_translate.scripts_registry`, which stays generic. Languages absent
from the table are simply not checked, so adding a pair never requires touching
the callers. The lists stay conservative: a particle that is also a free-standing
word is left out, because a false positive deletes a good row. Korean drops
``이`` (the determiner in ``이 사람``), ``가``/``와`` (imperatives of 가다/오다),
``과`` (a lesson), ``은`` (silver), and ``도``/``만`` (a province, ten thousand).
"""

from __future__ import annotations

import re
import unicodedata

from sion_translate.scripts_registry import (
    LANGUAGE_SCRIPTS,
    SPACELESS_SCRIPTS,
    collapse_spurious_spaces,
    resolve_scripts,
)

# Function morphemes that cannot host themselves. Keyed by the language tag used
# in ``data.language_pairs``.
ORPHAN_FUNCTION_TOKENS: dict[str, frozenset[str]] = {
    "ko": frozenset(
        {
            "의",
            "는",
            "을",
            "를",
            "에",
            "에서",
            "에게",
            "에게서",
            "한테",
            "께",
            "께서",
            "으로",
            "이랑",
            "부터",
            "까지",
            "처럼",
            "이라서",
            "이라고",
            "이라는",
        }
    ),
    "ja": frozenset(
        {
            "は",
            "が",
            "を",
            "に",
            "へ",
            "と",
            "の",
            "から",
            "まで",
            "より",
            "のは",
            "には",
            "とは",
            "では",
            "への",
            "からは",
        }
    ),
}
# 한본어 mixes both writing systems, so either language's particle may be the
# stranded one. Derived rather than duplicated so the two tables stay the source
# of truth.
ORPHAN_FUNCTION_TOKENS["kj"] = ORPHAN_FUNCTION_TOKENS["ko"] | ORPHAN_FUNCTION_TOKENS["ja"]

# Categories trimmed from a token before the lookup, so that ``의,`` and ``の。``
# are still recognised as stranded: punctuation, symbols, separators, controls.
_TRIMMED_CATEGORIES = frozenset({"P", "S", "Z", "C"})

_WHITESPACE_RUN = re.compile(r"\s+")


def _strip_punctuation(token: str) -> str:
    """Trim leading and trailing punctuation, symbols and whitespace.

    Unicode categories are used rather than a literal character class so that
    full-width and CJK punctuation is covered without enumerating it.
    """

    start = 0
    end = len(token)
    while start < end and unicodedata.category(token[start])[0] in _TRIMMED_CATEGORIES:
        start += 1
    while end > start and unicodedata.category(token[end - 1])[0] in _TRIMMED_CATEGORIES:
        end -= 1
    return token[start:end]


def known_languages() -> tuple[str, ...]:
    return tuple(sorted(ORPHAN_FUNCTION_TOKENS))


def _particles(language: str) -> frozenset[str]:
    return ORPHAN_FUNCTION_TOKENS.get(str(language).strip().lower(), frozenset())


def _writes_with_spaces(language: str) -> bool:
    """True when any script this language uses separates words with spaces."""

    name = str(language).strip().lower()
    if name not in LANGUAGE_SCRIPTS:
        return True
    return bool(resolve_scripts([name]) - SPACELESS_SCRIPTS)


def _spaceless_scripts_used(language: str) -> bool:
    name = str(language).strip().lower()
    if name not in LANGUAGE_SCRIPTS:
        return False
    return bool(resolve_scripts([name]) & SPACELESS_SCRIPTS)


def orphan_function_tokens(text: str, language: str) -> tuple[str, ...]:
    """Whitespace-delimited tokens of ``text`` that are bare function morphemes.

    Only meaningful for a language whose script separates words with spaces.
    """

    particles = _particles(language)
    if not particles or not text:
        return ()
    found: list[str] = []
    for raw in text.split():
        token = _strip_punctuation(raw)
        if token and token in particles:
            found.append(token)
    return tuple(found)


def stranded_function_markers(text: str, language: str) -> tuple[str, ...]:
    """Particles preceded by whitespace a segmenter cannot account for.

    For a spaceless script, any surviving space before a particle is a hole. The
    collapse runs first so ordinary morpheme segmentation does not fire.
    """

    particles = _particles(language)
    if not particles or not text:
        return ()
    collapsed = collapse_spurious_spaces(text)
    lengths = sorted({len(particle) for particle in particles}, reverse=True)
    found: list[str] = []
    for match in _WHITESPACE_RUN.finditer(collapsed):
        start = match.end()
        for length in lengths:
            candidate = collapsed[start : start + length]
            if candidate in particles:
                found.append(candidate)
                break
    return tuple(found)


def placeholder_hole_markers(text: str, language: str) -> tuple[str, ...]:
    """Every stranded function morpheme in ``text``, by whichever rule applies."""

    markers: list[str] = []
    if _writes_with_spaces(language):
        markers.extend(orphan_function_tokens(text, language))
    if _spaceless_scripts_used(language):
        markers.extend(stranded_function_markers(text, language))
    return tuple(markers)


def has_placeholder_hole(text: str, language: str) -> bool:
    return bool(placeholder_hole_markers(text, language))


__all__ = [
    "ORPHAN_FUNCTION_TOKENS",
    "has_placeholder_hole",
    "known_languages",
    "orphan_function_tokens",
    "placeholder_hole_markers",
    "stranded_function_markers",
]
