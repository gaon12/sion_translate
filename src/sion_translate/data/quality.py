from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
import unicodedata

from sion_translate.splitting import normalized_split_key
from sion_translate.structured import structured_similarity


_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class QualityPolicy:
    """Conservative, language-aware filters for raw Korean-Japanese pairs.

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
    ko_language_fraction: float
    ja_language_fraction: float


def canonical_text(text: str) -> str:
    """Normalize storage text without changing width or compatibility forms."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text).strip())


def dedup_key(text: str) -> str:
    """Use stronger compatibility normalization only for dedup/split keys."""

    return normalized_split_key(text)


def _visible_length(text: str) -> int:
    return sum(not char.isspace() for char in text)


def _is_hangul(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xAC00 <= codepoint <= 0xD7A3
    )


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


def language_fraction(text: str, language: str) -> float:
    """해당 언어 문자가 차지하는 비율. 문자 기반 판별이 가능한 언어(ko/ja)만
    실제로 검사하고, 그 외 언어는 1.0 을 돌려 검사를 통과시킵니다
    (라틴 문자 언어끼리는 문자만으로 언어를 구분할 수 없기 때문)."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    if language == "ko":
        signal = sum(_is_hangul(char) for char in letters)
    elif language == "ja":
        signal = sum(_is_kana(char) or _is_han(char) for char in letters)
    else:
        return 1.0
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
    languages: tuple[str, str] | list[str] = ("ko", "ja"),
) -> PairAssessment:
    """번역쌍 품질 평가. ``languages`` 로 다른 언어쌍(en-de 등)도 검사할 수
    있으며, 문자 기반 판별이 불가능한 언어는 script 검사만 건너뜁니다.
    (필드 이름의 ko/ja 는 '첫 번째/두 번째 언어'라는 의미의 내부 명칭입니다.)"""
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
        and ko_fraction < policy.min_language_fraction
    ):
        rejections.append("ko_script_mismatch")
    if (
        policy.reject_script_mismatch
        and ja_letters >= policy.min_language_check_chars
        and ja_fraction < policy.min_language_fraction
    ):
        rejections.append("ja_script_mismatch")
    # 일본어 특화 경고: 긴 문장에 가나가 전혀 없으면 중국어 혼입 의심.
    if language_b == "ja":
        ja_kana = japanese_kana_count(ja)
        ja_han = sum(_is_han(char) for char in ja)
        if ja_letters >= policy.long_ja_kana_warning_chars and ja_kana == 0 and ja_han >= 4:
            warnings.append("ja_no_kana")

    if policy.reject_controls and (_has_control_characters(ko) or _has_control_characters(ja)):
        rejections.append("control_characters")
    if policy.reject_repetition and (
        _has_excessive_repetition(ko) or _has_excessive_repetition(ja)
    ):
        rejections.append("excessive_repetition")

    structured_score, critical_mismatch = structured_similarity(ko, ja)
    if critical_mismatch or structured_score < 0.5:
        warnings.append("structured_span_mismatch")

    penalties = {
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
    score = max(
        0,
        100
        - sum(penalties[reason] for reason in rejections)
        - sum(penalties[reason] for reason in warnings),
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
