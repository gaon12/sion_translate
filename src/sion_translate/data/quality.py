from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import re
import unicodedata

from sion_translate.scripts_registry import primary_language, script_of, scripts_for_language
from sion_translate.splitting import normalized_split_key
from sion_translate.structured import structured_similarity


_WHITESPACE = re.compile(r"\s+")

_QUALITY_PENALTIES = {
    "too_short": 45,
    "length_ratio": 25,
    "identical_text": 50,
    "ko_script_mismatch": 35,
    "ja_script_mismatch": 35,
    "control_characters": 40,
    "excessive_repetition": 40,
    "structured_span_mismatch": 10,
    "ja_no_kana": 10,
}

_EXPRESSIVE_QUALITY_PROFILE = "expressive_v1"
_EXPRESSIVE_ALLOWED_REJECTIONS = frozenset({"too_short", "excessive_repetition"})


@dataclass(frozen=True)
class QualityPolicy:
    """Conservative, language-aware filters for arbitrary raw parallel pairs.

    These checks intentionally target obvious corpus damage rather than trying to
    judge translation fluency. Borderline pairs remain in the corpus with a lower
    quality score so source-level sampling can handle them without silently
    discarding useful domain data.
    """

    min_chars_per_side: int = 2
    max_length_ratio: float = 5.0
    min_language_fraction: float = 0.10
    min_language_check_chars: int = 4
    long_ja_kana_warning_chars: int = 12
    reject_identical: bool = True
    reject_script_mismatch: bool = True
    reject_controls: bool = True
    reject_repetition: bool = True

    def validate(self) -> None:
        if self.min_chars_per_side < 1:
            raise ValueError("min_chars_per_side must be positive")
        if self.max_length_ratio <= 1.0:
            raise ValueError("max_length_ratio must be greater than 1")
        if not 0.0 <= self.min_language_fraction <= 1.0:
            raise ValueError("min_language_fraction must be in [0, 1]")
        if self.min_language_check_chars < 1:
            raise ValueError("min_language_check_chars must be positive")
        if self.long_ja_kana_warning_chars < 1:
            raise ValueError("long_ja_kana_warning_chars must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PairAssessment:
    accepted: bool
    score: int
    rejection_reasons: tuple[str, ...]
    warning_reasons: tuple[str, ...]
    ko_chars: int
    ja_chars: int
    length_ratio: float
    ko_language_fraction: float | None
    ja_language_fraction: float | None


def apply_record_quality_profile(
    assessment: PairAssessment,
    profile: object,
) -> PairAssessment:
    """Apply a narrow, explicit exception for curated expressive records.

    One-character reactions and deliberately prolonged cries are valid language,
    but indistinguishable from noise to a generic length/repetition filter. The
    profile waives only those two reasons; script mismatches, controls, identical
    pairs, structural corruption, and extreme length ratios remain protected.
    """

    if profile != _EXPRESSIVE_QUALITY_PROFILE:
        return assessment
    kept = tuple(
        reason
        for reason in assessment.rejection_reasons
        if reason not in _EXPRESSIVE_ALLOWED_REJECTIONS
    )
    removed = set(assessment.rejection_reasons) - set(kept)
    if not removed:
        return assessment
    restored_score = min(
        100,
        assessment.score + sum(_QUALITY_PENALTIES[reason] for reason in removed),
    )
    return replace(
        assessment,
        accepted=not kept,
        score=restored_score,
        rejection_reasons=kept,
    )


def canonical_text(text: str) -> str:
    """Normalize storage text without changing width or compatibility forms."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text).strip())


def dedup_key(text: str) -> str:
    """Use stronger compatibility normalization only for dedup/split keys."""

    return normalized_split_key(text)


def _visible_length(text: str) -> int:
    return sum(not char.isspace() for char in text)


def _is_han(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def _is_kana(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0xFF66 <= codepoint <= 0xFF9D
    )


def japanese_kana_count(text: str) -> int:
    """Count hiragana, katakana, extensions, and half-width kana."""

    return sum(_is_kana(char) for char in text)


def language_fraction(text: str, language: str) -> float | None:
    """Return the share of letters in the tag's checkable writing systems.

    ``None`` means the tag has no known script profile, so callers must omit
    the check instead of recording a fictitious perfect score. Arbitrary
    languages can opt in without code changes by using an explicit supported
    ISO 15924 script subtag such as ``sr-Latn`` or ``az-Arab``.
    """

    scripts = scripts_for_language(language)
    if scripts is None:
        return None
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    signal = sum(script_of(char) in scripts for char in letters)
    return signal / len(letters)


def _comparison_key(text: str) -> str:
    return "".join(char.casefold() for char in dedup_key(text) if char.isalnum())


def _has_control_characters(text: str) -> bool:
    for char in text:
        category = unicodedata.category(char)
        if category == "Cs" or (category == "Cc" and char not in "\t\r\n"):
            return True
    return False


def _has_excessive_repetition(text: str) -> bool:
    surface = [char for char in text if not char.isspace()]
    if len(surface) < 12:
        return False
    most_common = Counter(surface).most_common(1)[0][1]
    if most_common / len(surface) >= 0.70:
        return True
    compact = "".join(surface)
    return re.search(r"(.{1,8})\1{4,}", compact) is not None


def assess_pair(
    ko: str,
    ja: str,
    policy: QualityPolicy | None = None,
    *,
    languages: tuple[str, str] | list[str],
) -> PairAssessment:
    """Assess one parallel pair under explicit source and target identities.

    Languages without a known script profile skip only the script-fraction
    check. The legacy ``ko`` and ``ja`` variable names mean the first and second
    sides internally; they do not restrict the accepted language graph.
    """
    policy = policy or QualityPolicy()
    policy.validate()
    language_a, language_b = languages
    ko = canonical_text(ko)
    ja = canonical_text(ja)
    ko_chars = _visible_length(ko)
    ja_chars = _visible_length(ja)
    shorter = max(1, min(ko_chars, ja_chars))
    ratio = max(ko_chars, ja_chars) / shorter
    ko_fraction = language_fraction(ko, language_a)
    ja_fraction = language_fraction(ja, language_b)
    rejections: list[str] = []
    warnings: list[str] = []

    if ko_chars < policy.min_chars_per_side or ja_chars < policy.min_chars_per_side:
        rejections.append("too_short")
    if ratio > policy.max_length_ratio:
        rejections.append("length_ratio")
    elif ratio > min(2.5, policy.max_length_ratio):
        warnings.append("length_ratio")

    ko_comparison = _comparison_key(ko)
    ja_comparison = _comparison_key(ja)
    if policy.reject_identical and ko_comparison and ko_comparison == ja_comparison:
        rejections.append("identical_text")

    ko_letters = sum(char.isalpha() for char in ko)
    ja_letters = sum(char.isalpha() for char in ja)
    if (
        policy.reject_script_mismatch
        and ko_letters >= policy.min_language_check_chars
        and ko_fraction is not None
        and ko_fraction < policy.min_language_fraction
    ):
        rejections.append("ko_script_mismatch")
    if (
        policy.reject_script_mismatch
        and ja_letters >= policy.min_language_check_chars
        and ja_fraction is not None
        and ja_fraction < policy.min_language_fraction
    ):
        rejections.append("ja_script_mismatch")
    # A long Japanese side with Han characters but no kana can indicate Chinese
    # contamination. Region and script variants inherit the primary language.
    for text, language, letter_count in (
        (ko, language_a, ko_letters),
        (ja, language_b, ja_letters),
    ):
        if primary_language(language) != "ja":
            continue
        kana = japanese_kana_count(text)
        han = sum(_is_han(char) for char in text)
        if letter_count >= policy.long_ja_kana_warning_chars and kana == 0 and han >= 4:
            warnings.append("ja_no_kana")

    if policy.reject_controls and (_has_control_characters(ko) or _has_control_characters(ja)):
        rejections.append("control_characters")
    if policy.reject_repetition and (
        _has_excessive_repetition(ko) or _has_excessive_repetition(ja)
    ):
        rejections.append("excessive_repetition")

    structured_score, critical_mismatch = structured_similarity(ko, ja)
    if critical_mismatch:
        rejections.append("structured_span_mismatch")
    elif structured_score < 0.5:
        warnings.append("structured_span_mismatch")

    score = max(
        0,
        100
        - sum(_QUALITY_PENALTIES[reason] for reason in rejections)
        - sum(_QUALITY_PENALTIES[reason] for reason in warnings),
    )
    return PairAssessment(
        accepted=not rejections,
        score=score,
        rejection_reasons=tuple(dict.fromkeys(rejections)),
        warning_reasons=tuple(dict.fromkeys(warnings)),
        ko_chars=ko_chars,
        ja_chars=ja_chars,
        length_ratio=ratio,
        ko_language_fraction=ko_fraction,
        ja_language_fraction=ja_fraction,
    )
