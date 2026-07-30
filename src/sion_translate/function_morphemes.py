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

A stranded particle has two possible causes, and they need opposite treatment:

*the host noun was deleted*
    ``늦은 밤, 의 숲속에서`` - nothing can host the particle, so the row is
    unrecoverable and must be dropped.

*the particle was merely spaced off its host*
    ``금요일 오전 아홉 시 에 깨워줘`` - the noun is right there. Joining restores
    correct orthography, and dropping would throw away a good row. data37 has 629
    of these and no deleted nouns at all.

The two are told apart by what precedes the particle: a host that ends in
punctuation, or no preceding token at all, means the noun is gone.
:func:`rejoin_orphan_particles` performs the repair, and
:func:`placeholder_hole_markers` reports only the orphans that cannot be repaired.

Known limitations, both pinned by tests rather than papered over:

* A hole between two spaceless characters (``恋人の が``) is removed by the
  collapse along with genuine segmenter spacing, so it is not detected here.
  Those rows are caught by the spurious-space density check in
  ``recover_shard.py`` instead.
* Telling ``와서 에`` (a verb form cannot host a case particle, so the noun is
  gone) from ``시 에`` (a noun that can) needs part-of-speech information. The
  join heuristic treats both as repairable, so a small number of damaged rows
  survive prepare and have to be caught by the similarity filter.

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

# A particle is never followed by one of these, so if the character after a
# candidate match is here, the match is really the head of a longer word:
# ``はっきり`` is an adverb, not the particle ``は`` plus a noun.
_NON_INITIAL_KANA = frozenset("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮーｰ゛゜々")


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


def _ends_with_punctuation(token: str) -> bool:
    return bool(token) and unicodedata.category(token[-1])[0] in _TRIMMED_CATEGORIES


def orphan_function_tokens(text: str, language: str) -> tuple[str, ...]:
    """Whitespace-delimited tokens of ``text`` that are bare function morphemes.

    Only meaningful for a language whose script separates words with spaces. Both
    repairable and unrepairable orphans are reported; use
    :func:`rejoin_orphan_particles` or :func:`orphan_hole_tokens` to separate them.
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


def rejoin_orphan_particles(text: str, language: str) -> tuple[str, int]:
    """Reattach particles that were merely spaced off a host that is still there.

    Returns the repaired text and how many particles were joined. A particle
    whose preceding token ends in punctuation, or which starts the text, is left
    alone: there is nothing to attach it to, which is the signature of a deleted
    noun rather than a spacing slip.

    This is a repair for orthography that uses inter-word spaces. A spaceless
    script has no correct spacing to restore - every space in it is already the
    business of ``collapse_spurious_spaces`` - so those languages are returned
    unchanged rather than double-counted by both repairs.
    """

    particles = _particles(language)
    if not particles or not text or not _writes_with_spaces(language):
        return text, 0
    tokens = text.split()
    if not tokens:
        return text, 0
    rebuilt: list[str] = []
    joined = 0
    for raw in tokens:
        stripped = _strip_punctuation(raw)
        if (
            stripped
            and stripped in particles
            and rebuilt
            and not _ends_with_punctuation(rebuilt[-1])
        ):
            rebuilt[-1] = rebuilt[-1] + raw
            joined += 1
            continue
        rebuilt.append(raw)
    return " ".join(rebuilt), joined


def orphan_hole_tokens(text: str, language: str) -> tuple[str, ...]:
    """Orphan particles that no repair can rescue, so the row must be dropped."""

    particles = _particles(language)
    if not particles or not text:
        return ()
    tokens = text.split()
    found: list[str] = []
    previous: str | None = None
    for raw in tokens:
        stripped = _strip_punctuation(raw)
        if stripped and stripped in particles:
            if previous is None or _ends_with_punctuation(previous):
                found.append(stripped)
        previous = raw
    return tuple(found)


def stranded_function_markers(text: str, language: str) -> tuple[str, ...]:
    """Particles preceded by whitespace a segmenter cannot account for.

    For a spaceless script, any surviving space before a particle is a hole. The
    collapse runs first so ordinary morpheme segmentation does not fire, and a
    match is rejected when the next character shows it is really the head of a
    longer word.
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
            if candidate not in particles:
                continue
            following = collapsed[start + length : start + length + 1]
            if following and following in _NON_INITIAL_KANA:
                break
            found.append(candidate)
            break
    return tuple(found)


def placeholder_hole_markers(text: str, language: str) -> tuple[str, ...]:
    """Stranded morphemes in ``text`` that indicate an unrecoverable deletion."""

    markers: list[str] = []
    if _writes_with_spaces(language):
        markers.extend(orphan_hole_tokens(text, language))
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
    "orphan_hole_tokens",
    "placeholder_hole_markers",
    "rejoin_orphan_particles",
    "stranded_function_markers",
]
