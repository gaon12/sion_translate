"""Detect semantically contaminated targets and repair only deterministic cases.

``data.quality.assess_pair`` checks length, script ratios, and repetition. It
cannot detect a target whose meaning is wrong but whose surface statistics are
normal. A row translating `씨발` as `種まき` (seed planting), for example,
passes with ``accepted=True, score=100`` and would otherwise be learned as gold.

This module detects three classes:

1. **Known literal mistranslations**, such as the measured `씨발` to `種まき`
   mapping. There is no normal context for that pair, so its precision is near
   1.0. This is the only automatically repairable class.
2. **Literal idiom translations**, where a Korean idiom appears as a literal
   Japanese rendering.
3. **Lost profanity intensity**, where the source contains profanity but the
   target contains no vulgarity or emphasis marker. This heuristic can find
   unseen contamination but has lower precision.

Manual review of the queue from 8.97 million rows showed very different rule
precision:

- All 89 sampled ``known_literal_mistranslation`` rows were true contamination.
- Most of the 155 ``literal_idiom`` rows were false positives. Phrases such as
  `붕어빵을 입에 물고` and `펭수네 붕어빵` refer to actual food, where `たい焼き`
  is correct.
- Most of the 84 ``dog_prefix_literal`` rows were false positives. Removing
  whitespace makes `개 소리가 들려` (a dog's sound is heard) indistinguishable
  from `개소리` (nonsense).

Two safeguards follow from that review. Dog-prefix profanity is searched in a
normalization that preserves spaces. Ambiguous idioms such as `붕어빵` also
require a contextual idiomatic-sense marker from
:attr:`LiteralIdiom.korean_sense_markers`; without one, the literal food sense
is assumed.

Literal idioms and lost intensity are flagged but not rewritten. Choosing new
Japanese wording is contextual translation and requires human review.
:func:`repair_pair` acts only when the replacement is context-independent
because the literal artifact itself cannot be valid there. It returns a repair
only after verifying that the rewritten row no longer triggers detection.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sion_translate.language_tags import LanguageTagError, parse_language_tag

# Korean profanity surfaces are matched after normalization for spacing and repetition.
KOREAN_PROFANITY = (
    "씨발",
    "시발",
    "씨팔",
    "좆",
    "존나",
    "졸라",
    "개새끼",
    # Do not include bare `새끼`, which also means a young animal and caused
    # measured false positives for `새끼 개` and `새끼 고양이`. Include only
    # profanity forms. Since ``normalize`` removes spaces, `이 새끼` still matches.
    "새끼야",
    "새끼들아",
    "이새끼",
    "저새끼",
    "그새끼",
    "병신",
    "지랄",
    "닥쳐",
    "미친놈",
    "미친년",
    "썅",
    "빌어먹을",
    "엿먹",
)

# Any Japanese marker indicates that intensity survived. This list need not be
# exhaustive: the heuristic flags only their absence, so broad coverage reduces false positives.
JAPANESE_VULGAR_MARKERS = (
    "くそ",
    "クソ",
    "糞",
    "ちくしょう",
    "畜生",
    "ふざけ",
    "うざ",
    "きも",
    "黙れ",
    "死ね",
    "野郎",
    "馬鹿",
    "ばか",
    "バカ",
    "あほ",
    "アホ",
    "てめえ",
    "てめー",
    "やがる",
    "しやがれ",
    "ぶっ",
    "クソッ",
    "ちっ",
    "最悪",
    "むかつく",
)

# Measured ``(Korean source, literal Japanese target)`` pairs whose co-occurrence
# is almost certainly contamination.
KNOWN_LITERAL_MISTRANSLATIONS: tuple[tuple[str, str], ...] = (
    ("씨발", "種まき"),
    ("시발", "種まき"),
    ("씨발", "種の足"),
    ("시발", "種の足"),
    ("좆", "種"),
)


@dataclass(frozen=True)
class LiteralIdiom:
    """Describe an idiom and evidence that the source uses its idiomatic sense.

    An empty ``korean_sense_markers`` means the expression is only idiomatic;
    there is no natural literal reading of `발 벗고 나서`. Otherwise, at least
    one marker must appear in the source. `붕어빵` is ambiguous between a
    resemblance idiom and food, and `たい焼き` is correct for the food sense.
    """

    korean: str
    japanese_literal: str
    korean_sense_markers: tuple[str, ...] = ()


# Idioms paired with Japanese literal renderings that indicate localization failure.
KNOWN_LITERAL_IDIOMS: tuple[LiteralIdiom, ...] = (
    LiteralIdiom("같은 값이면 다홍치마", "紅スカート"),
    # This is idiomatic only in the resemblance sense; without a marker it means food.
    LiteralIdiom(
        "붕어빵",
        "たい焼き",
        korean_sense_markers=(
            "닮",
            "판박이",
            "똑같",
            "빼닮",
            "랑 붕어빵",
            "와 붕어빵",
            "과 붕어빵",
        ),
    ),
    LiteralIdiom("식은 죽 먹기", "冷めた粥"),
    LiteralIdiom("발 벗고 나서", "足を脱いで"),
    LiteralIdiom("눈코 뜰 새", "目と鼻を開ける"),
    LiteralIdiom("바가지를 쓰", "ひさごをかぶ"),
    LiteralIdiom("미역국을 먹", "わかめスープを飲"),
    LiteralIdiom("국수를 먹", "そうめんを食べ"),
)

# Detect dog-prefix profanity rendered as the animal `犬`. Because `개` often
# denotes an actual dog, only attached prefix forms are evidence. Space-preserving
# matching keeps `개 소리가 들려` distinct from `개소리`.
DOG_PREFIX_PROFANITY = ("개새끼", "개자식", "개놈", "개년", "개소리", "개수작")

_REPEATED = re.compile(r"(.)\1{2,}")
_PUNCTUATION = r"[.,!?~·・…\-_*]+"


def normalize(text: str) -> str:
    """Normalize repetition and inter-character separators for comparison.

    Whitespace is removed to catch separated profanity such as `씨 발`. This
    loses distinctions where spacing changes meaning, such as `개 소리` versus
    `개소리`; those rules must use :func:`spaced_normalize`.

    The normalizer cannot restore forms stretched by inserting a vowel, such as
    `씨이이발`. Repetition collapse handles only adjacent copies of the same
    character; inserted syllables require an explicit lexicon entry.
    """

    return re.sub(r"\s+", "", spaced_normalize(text))


def spaced_normalize(text: str) -> str:
    """Normalize like :func:`normalize` while retaining collapsed whitespace.

    Use this for rules where spacing changes meaning. Removing whitespace caused
    the measured false positive that treated `개 소리가 들려` as `개소리`.
    """

    folded = unicodedata.normalize("NFKC", text)
    folded = _REPEATED.sub(r"\1", folded)
    folded = re.sub(_PUNCTUATION, "", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _normalized_pairs(
    pairs: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str, str, str], ...]:
    """Precompute source/target patterns and their comparison forms.

    Patterns must use the same normalization as text. Otherwise an idiom with
    spaces can never match: `같은 값이면 다홍치마` becomes
    `같은값이면다홍치마` in normalized text.
    """

    return tuple((source, target, normalize(source), normalize(target)) for source, target in pairs)


@dataclass(frozen=True)
class ContaminationFinding:
    """Describe one suspected row; ``confidence`` orders human review."""

    rule: str
    reason: str
    confidence: float
    evidence: tuple[str, ...] = field(default_factory=tuple)


def _contains_any(haystack: str, needles: Iterable[str]) -> tuple[str, ...]:
    return tuple(needle for needle in needles if needle in haystack)


_NORMALIZED_MISTRANSLATIONS = _normalized_pairs(KNOWN_LITERAL_MISTRANSLATIONS)
_NORMALIZED_IDIOMS: tuple[tuple[LiteralIdiom, str, str, tuple[str, ...]], ...] = tuple(
    (
        idiom,
        normalize(idiom.korean),
        normalize(idiom.japanese_literal),
        tuple(normalize(marker) for marker in idiom.korean_sense_markers),
    )
    for idiom in KNOWN_LITERAL_IDIOMS
)
# Match in the space-preserving form described above ``DOG_PREFIX_PROFANITY``.
_SPACED_DOG = tuple(spaced_normalize(item) for item in DOG_PREFIX_PROFANITY)
_NORMALIZED_PROFANITY = tuple(normalize(item) for item in KOREAN_PROFANITY)
_NORMALIZED_VULGAR = tuple(normalize(item) for item in JAPANESE_VULGAR_MARKERS)


def assess_contamination(
    source: str,
    target: str,
    *,
    source_language: str,
    target_language: str,
) -> list[ContaminationFinding]:
    """Return all contamination evidence for one pair, or an empty list.

    Rules currently cover only ko-to-ja. Other directions return an empty list,
    so callers must use :func:`supported_direction` and must not report an
    unsupported direction as clean.
    """

    if not supported_direction(source_language, target_language):
        return []

    findings: list[ContaminationFinding] = []
    flat_source = normalize(source)
    flat_target = normalize(target)

    for korean, japanese, flat_korean, flat_japanese in _NORMALIZED_MISTRANSLATIONS:
        if flat_korean in flat_source and flat_japanese in flat_target:
            findings.append(
                ContaminationFinding(
                    rule="known_literal_mistranslation",
                    reason=f"profanity {korean!r} was rendered literally as {japanese!r}",
                    confidence=0.95,
                    evidence=(korean, japanese),
                )
            )

    for idiom, flat_korean, flat_japanese, flat_markers in _NORMALIZED_IDIOMS:
        if flat_korean not in flat_source or flat_japanese not in flat_target:
            continue
        # Marker-gated expressions also have common literal senses. Without a
        # marker they refer to the object and a literal translation is correct.
        if flat_markers and not _contains_any(flat_source, flat_markers):
            continue
        findings.append(
            ContaminationFinding(
                rule="literal_idiom",
                reason=(
                    f"idiom {idiom.korean!r} was rendered literally as {idiom.japanese_literal!r}"
                ),
                confidence=0.85,
                evidence=(idiom.korean, idiom.japanese_literal),
            )
        )

    dog = _contains_any(spaced_normalize(source), _SPACED_DOG)
    if dog and "犬" in flat_target:
        findings.append(
            ContaminationFinding(
                rule="dog_prefix_literal",
                reason=f"prefix profanity {dog[0]!r} was rendered as the animal '犬'",
                confidence=0.75,
                evidence=dog + ("犬",),
            )
        )

    # This lexicon-independent rule finds unseen contamination at the lowest precision.
    profanity = _contains_any(flat_source, _NORMALIZED_PROFANITY)
    if profanity and not _contains_any(flat_target, _NORMALIZED_VULGAR):
        findings.append(
            ContaminationFinding(
                rule="profanity_intensity_lost",
                reason=(
                    f"the target has no vulgarity or emphasis marker corresponding to "
                    f"source profanity {profanity[0]!r}"
                ),
                confidence=0.45,
                evidence=profanity,
            )
        )
    return findings


# Literal artifacts and their deterministic replacements. Each key is erroneous
# merely by occurring in this context. `種まき` cannot validly correspond to
# `씨발`, so its replacement does not depend on surrounding context.
#
# `くそ` preserves both intensity and grammatical use. It works as an
# exclamation (`このくそ`) and as an intensifying prefix (`くそ暑い`), covering
# both positions where the literal artifact appeared.
#
# A single-character mapping such as `좆` to `種` is intentionally absent.
# `種` is a common noun, so replacement could corrupt valid text. Those rows
# remain in the human-review queue.
LITERAL_ARTIFACT_REPAIRS: dict[str, str] = {
    "種まき": "くそ",
    "種の足": "くそ",
}

# Only this rule permits automatic repair; all others require contextual human translation.
REPAIRABLE_RULES = frozenset({"known_literal_mistranslation"})


@dataclass(frozen=True)
class ContaminationRepair:
    """Store one repair together with its original target for reversibility."""

    target: str
    original_target: str
    rule: str
    replacements: tuple[tuple[str, str], ...]

    @property
    def changed(self) -> bool:
        return self.target != self.original_target


def repair_pair(
    source: str,
    target: str,
    *,
    source_language: str,
    target_language: str,
) -> ContaminationRepair | None:
    """Repair only deterministic contamination; otherwise return ``None``.

    ``None`` means "not safely repairable by a rule," not "clean." Callers must
    assess contamination separately and keep unrepaired findings in the human
    review queue.

    The repaired target is assessed again. If any finding remains, the repair
    is rejected because publishing a partial correction that evades one rule
    would be worse than preserving the original for review.
    """

    findings = assess_contamination(
        source, target, source_language=source_language, target_language=target_language
    )
    # At least one repairable finding is required. Other findings do not block
    # the attempt: a literal artifact that displaced profanity also triggers
    # lost intensity, so rejecting multiple findings would eliminate every
    # repairable row. Post-repair validation is the authoritative gate.
    if not any(finding.rule in REPAIRABLE_RULES for finding in findings):
        return None

    repaired = target
    applied: list[tuple[str, str]] = []
    for artifact, replacement in LITERAL_ARTIFACT_REPAIRS.items():
        if artifact in repaired:
            repaired = repaired.replace(artifact, replacement)
            applied.append((artifact, replacement))
    if not applied or repaired == target:
        return None

    # Reject any repair that still triggers a finding and leave it for review.
    if assess_contamination(
        source, repaired, source_language=source_language, target_language=target_language
    ):
        return None

    return ContaminationRepair(
        target=repaired,
        original_target=target,
        rule="known_literal_mistranslation",
        replacements=tuple(applied),
    )


def supported_direction(source_language: str, target_language: str) -> bool:
    """Return whether this module has rules for the requested direction.

    Reporting an unsupported direction as contamination-free would make the
    audit misleading.
    """

    try:
        source_primary = parse_language_tag(
            source_language, field="contamination source language"
        ).language
        target_primary = parse_language_tag(
            target_language, field="contamination target language"
        ).language
    except LanguageTagError:
        return False
    return (source_primary, target_primary) == ("ko", "ja")


def rank_findings(findings: Sequence[ContaminationFinding]) -> ContaminationFinding | None:
    """Return the highest-confidence finding for review prioritization."""

    return max(findings, key=lambda finding: finding.confidence, default=None)


__all__ = [
    "DOG_PREFIX_PROFANITY",
    "JAPANESE_VULGAR_MARKERS",
    "KNOWN_LITERAL_IDIOMS",
    "KNOWN_LITERAL_MISTRANSLATIONS",
    "KOREAN_PROFANITY",
    "LITERAL_ARTIFACT_REPAIRS",
    "REPAIRABLE_RULES",
    "ContaminationFinding",
    "ContaminationRepair",
    "LiteralIdiom",
    "assess_contamination",
    "normalize",
    "rank_findings",
    "repair_pair",
    "spaced_normalize",
    "supported_direction",
]
