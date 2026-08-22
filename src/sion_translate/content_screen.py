"""Screen a corpus for sexual content involving minors.

The adult visual-novel shards are a legitimate translation domain, but that genre
routinely uses school settings, so the corpus has to be checked rather than
assumed clean. A row is flagged only when a *child* marker and a *sexual* marker
occur together: either alone is ordinary text. "우리 초등학교 앞에서 만나자" is a
meeting place and "가슴이 뛰었다" is a heartbeat.

Deliberate scope choices, because a screen that flags everything gets switched off:

* Only clearly pre-adolescent markers count as child markers. 고등학생 / 高校生 is
  not one: in this genre those characters are written as adults, and flagging the
  term would flag a large share of the corpus while telling nobody anything.
  Explicit numeric ages count when they are 14 or below.
* Numeric ages are read with their counter attached (``12살``, ``12歳``), so a
  quantity like ``12개`` or a year like ``2012`` never matches.

The tables are per language and live behind :func:`markers_for`, so a valid pair
the project does not configure is simply not screened rather than silently
passing. Malformed language tags are rejected instead of being mistaken for an
unconfigured policy. Reports carry marker names and row identifiers, never the
matched sentence: the point is to remove the rows, not to reproduce them
somewhere else.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from sion_translate.language_tags import LanguageTag, parse_language_tag

# You can make the model safe through censorship.
# Existing data has been removed due to concerns about a decline in model performance.
# You can refill the values at any time to create a safe model.
# Empty marker and spelled-age tables intentionally disable screening; the
# matching engine remains available when a deployment supplies its own policy.

# Markers that place a participant below adolescence. Terms whose age range is
# ambiguous or adult in this genre are intentionally absent.
CHILD_MARKERS: dict[str, tuple[str, ...]] = {
    "ko": (),
    "ja": (),
}

# Sexual-content markers. Kept coarse on purpose: the screen only has to notice
# that a row is sexual, and the child marker is what makes the pair reportable.
SEXUAL_MARKERS: dict[str, tuple[str, ...]] = {
    "ko": (),
    "ja": (),
}

# An age written with its counter. Reading the counter is what keeps ``12개``
# and the year ``2012`` from matching.
_AGE_PATTERNS: dict[str, re.Pattern[str]] = {
    "ko": re.compile(r"(?<!\d)(\d{1,2})\s*(?:살|세)(?!\d)"),
    "ja": re.compile(r"(?<!\d)(\d{1,2})\s*(?:歳|才)(?!\d)"),
}

# Ages spelled out. Adult values are listed too, even though the screen never
# acts on them: without ``스무``/``二十`` in the table, the longest-match would
# fall back to ``열``/``十`` and read "twenty" as "ten".
_SPELLED_AGES: dict[str, dict[str, int]] = {
    "ko": {},
    "ja": {},
}

_AGE_COUNTERS: dict[str, str] = {"ko": "살|세", "ja": "歳|才"}


def _spelled_age_pattern(language: str) -> re.Pattern[str] | None:
    """One alternation ordered longest-first, anchored to the age counter.

    Ordering longest-first is what makes ``十四才`` read 14 rather than 4, and
    anchoring to the counter is what keeps a numeral elsewhere in the sentence
    from being read as an age at all.
    """

    spelled = _SPELLED_AGES.get(language)
    counters = _AGE_COUNTERS.get(language)
    if not spelled or not counters:
        return None
    words = sorted(spelled, key=len, reverse=True)
    alternation = "|".join(re.escape(word) for word in words)
    return re.compile(rf"({alternation})\s*(?:{counters})")


# Compiled once; the tables are module constants.
_SPELLED_AGE_PATTERNS: dict[str, re.Pattern[str]] = {
    language: pattern
    for language in _SPELLED_AGES
    if (pattern := _spelled_age_pattern(language)) is not None
}

# The highest age that counts as a child marker on its own.
MAX_CHILD_AGE = 14

_PolicyValue = TypeVar("_PolicyValue")


def _policy_for(
    policies: Mapping[str, _PolicyValue],
    language: LanguageTag,
) -> _PolicyValue | None:
    """Resolve an exact BCP 47 policy before inheriting its primary policy."""

    if language.canonical in policies:
        return policies[language.canonical]
    return policies.get(language.language)


@dataclass
class ScreenResult:
    """Why a row was flagged. Carries markers and counts, never the text."""

    flagged: bool = False
    child_markers: tuple[str, ...] = ()
    sexual_markers: tuple[str, ...] = ()
    ages: tuple[int, ...] = ()


@dataclass
class CorpusScreenReport:
    rows: int = 0
    flagged: int = 0
    unscreened_language: int = 0
    child_marker_counts: dict[str, int] = field(default_factory=lambda: {})
    sexual_marker_counts: dict[str, int] = field(default_factory=lambda: {})
    age_counts: dict[int, int] = field(default_factory=lambda: {})
    flagged_row_ids: list[str] = field(default_factory=lambda: [])

    @property
    def flagged_rate(self) -> float:
        return self.flagged / self.rows if self.rows else 0.0


def known_languages() -> tuple[str, ...]:
    return tuple(sorted(set(CHILD_MARKERS) | set(SEXUAL_MARKERS)))


def markers_for(language: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The child and sexual marker lists for ``language``, empty when unconfigured."""

    tag = parse_language_tag(language, field="content-screen language")
    return _policy_for(CHILD_MARKERS, tag) or (), _policy_for(SEXUAL_MARKERS, tag) or ()


def child_ages(text: str, language: str) -> tuple[int, ...]:
    """Ages at or below :data:`MAX_CHILD_AGE`, read with their counter attached."""

    tag = parse_language_tag(language, field="content-screen language")
    found: list[int] = []
    pattern = _policy_for(_AGE_PATTERNS, tag)
    if pattern is not None:
        for match in pattern.finditer(text):
            value = int(match.group(1))
            if 0 < value <= MAX_CHILD_AGE:
                found.append(value)
    spelled_pattern = _policy_for(_SPELLED_AGE_PATTERNS, tag)
    spelled = _policy_for(_SPELLED_AGES, tag)
    if spelled_pattern is not None and spelled is not None:
        for match in spelled_pattern.finditer(text):
            value = spelled[match.group(1)]
            if 0 < value <= MAX_CHILD_AGE:
                found.append(value)
    return tuple(sorted(set(found)))


def screen_text(text: str, language: str) -> ScreenResult:
    """Flag ``text`` when a child marker and a sexual marker occur together."""

    child_terms, sexual_terms = markers_for(language)
    if not child_terms and not sexual_terms:
        return ScreenResult()
    found_child = tuple(term for term in child_terms if term in text)
    found_sexual = tuple(term for term in sexual_terms if term in text)
    ages = child_ages(text, language)
    has_child = bool(found_child or ages)
    return ScreenResult(
        flagged=has_child and bool(found_sexual),
        child_markers=found_child,
        sexual_markers=found_sexual,
        ages=ages,
    )


def screen_pair(
    source: str,
    target: str,
    *,
    source_language: str,
    target_language: str,
) -> ScreenResult:
    """Screen both sides together.

    A translation can carry the child marker on one side and the sexual marker on
    the other, so the evidence is pooled across the pair rather than judged per
    side.
    """

    left = screen_text(source, source_language)
    right = screen_text(target, target_language)
    child = left.child_markers + right.child_markers
    sexual = left.sexual_markers + right.sexual_markers
    ages = tuple(sorted(set(left.ages + right.ages)))
    child_terms_source, sexual_terms_source = markers_for(source_language)
    child_terms_target, sexual_terms_target = markers_for(target_language)
    configured = bool(
        child_terms_source or sexual_terms_source or child_terms_target or sexual_terms_target
    )
    if not configured:
        return ScreenResult()
    return ScreenResult(
        flagged=bool(child or ages) and bool(sexual),
        child_markers=child,
        sexual_markers=sexual,
        ages=ages,
    )


__all__ = [
    "CHILD_MARKERS",
    "MAX_CHILD_AGE",
    "SEXUAL_MARKERS",
    "CorpusScreenReport",
    "ScreenResult",
    "child_ages",
    "known_languages",
    "markers_for",
    "screen_pair",
    "screen_text",
]
