"""오염된 정답쌍을 사람 검수 queue 로 뽑아낸다.

``data.quality.assess_pair`` 는 길이·문자 비율·반복을 봅니다. 그것으로는
**의미가 틀린 정답**을 잡을 수 없습니다. `씨발` 을 `種まき`(씨 뿌리기)로 옮긴
행은 길이도 문자 비율도 정상이라 ``accepted=True, score=100`` 으로 통과합니다.
모델은 그것을 정답으로 배웁니다.

여기서 하는 일은 **삭제가 아니라 표시**입니다. 자동 삭제는 두 가지 이유로
틀립니다. 첫째, 아래 규칙은 전부 휴리스틱이라 정상 번역도 걸립니다 — `개` 가
실제로 동물인 문장, `붕어빵` 이 실제로 음식인 문장이 있습니다. 둘째, 오염된
행의 가치는 삭제가 아니라 **재번역**에 있습니다.

## 무엇을 잡는가

세 가지이고, 세 번째가 가장 일반적입니다.

1. **알려진 직역 매핑**: 실측으로 확인된 대응(`씨발` → `種まき`)입니다. 정밀도가
   높지만 목록에 있는 것만 잡습니다.
2. **관용구 직역**: 한국어 관용구가 있는데 일본어가 축자 번역인 경우입니다.
3. **욕설 강도 소실**: 원문에 욕설이 있는데 번역문에 어떤 비속·강조 표지도
   없는 경우입니다. 목록에 의존하지 않아 새 오염도 잡습니다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# 한국어 욕설·비속 표현. 자소 분리와 반복을 견디도록 정규화 후 검사합니다.
KOREAN_PROFANITY = (
    "씨발",
    "시발",
    "씨팔",
    "좆",
    "존나",
    "졸라",
    "개새끼",
    "새끼",
    "병신",
    "지랄",
    "닥쳐",
    "미친놈",
    "미친년",
    "썅",
    "빌어먹을",
    "엿먹",
)

# 일본어 쪽에 하나라도 있으면 "강도가 살아 있다"고 봅니다. 완전한 목록일
# 필요는 없습니다 — 없을 때만 의심하는 용도이므로 넓게 잡을수록 오탐이 줍니다.
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

# 실측으로 확인된 직역 대응. `(한국어, 일본어 직역)` 이고, 둘이 같은 행에
# 있으면 거의 확실한 오염입니다.
KNOWN_LITERAL_MISTRANSLATIONS: tuple[tuple[str, str], ...] = (
    ("씨발", "種まき"),
    ("시발", "種まき"),
    ("씨발", "種の足"),
    ("시발", "種の足"),
    ("좆", "種"),
)

# 관용구와 그 축자 번역. 일본어에 축자 번역이 그대로 나오면 현지화 실패입니다.
KNOWN_LITERAL_IDIOMS: tuple[tuple[str, str], ...] = (
    ("같은 값이면 다홍치마", "紅スカート"),
    ("붕어빵", "たい焼き"),
    ("식은 죽 먹기", "冷めた粥"),
    ("발 벗고 나서", "足を脱いで"),
    ("눈코 뜰 새", "目と鼻を開ける"),
    ("바가지를 쓰", "ひさごをかぶ"),
    ("미역국을 먹", "わかめスープを飲"),
    ("국수를 먹", "そうめんを食べ"),
)

# 개-접두 욕설이 동물 `犬` 으로 옮겨진 경우. `개` 가 실제 동물인 문장이 많아
# 단독으로는 근거가 약하므로, 접두사로 쓰인 형태만 봅니다.
DOG_PREFIX_PROFANITY = ("개새끼", "개자식", "개놈", "개년", "개소리", "개수작")

_REPEATED = re.compile(r"(.)\1{2,}")


def normalize(text: str) -> str:
    """반복 문자와 자간 기호를 걷어낸 비교용 표현.

    한계: `씨이이발` 처럼 **모음을 끼워 늘인** 형태는 되돌리지 못합니다.
    반복 축약은 같은 문자가 이어질 때만 동작하고, 늘임은 새 음절을 넣는
    변형이기 때문입니다. 그런 형태는 목록에 직접 넣어야 잡힙니다.
    """

    folded = unicodedata.normalize("NFKC", text)
    folded = _REPEATED.sub(r"\1", folded)
    return re.sub(r"[\s.,!?~·・…\-_*]+", "", folded)


def _normalized_pairs(
    pairs: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str, str, str], ...]:
    """``(원문 패턴, 번역 패턴)`` 을 비교용 형태와 함께 미리 계산한다.

    패턴을 텍스트와 **같은 방식으로** 정규화하지 않으면 공백이 든 관용구가
    영영 걸리지 않습니다 — `같은 값이면 다홍치마` 는 정규화된 텍스트에서
    `같은값이면다홍치마` 이므로 원문 그대로는 일치하지 않습니다.
    """

    return tuple((source, target, normalize(source), normalize(target)) for source, target in pairs)


@dataclass(frozen=True)
class ContaminationFinding:
    """의심되는 행 하나. ``confidence`` 는 검수 순서를 정하는 용도입니다."""

    rule: str
    reason: str
    confidence: float
    evidence: tuple[str, ...] = field(default_factory=tuple)


def _contains_any(haystack: str, needles: Iterable[str]) -> tuple[str, ...]:
    return tuple(needle for needle in needles if needle in haystack)


_NORMALIZED_MISTRANSLATIONS = _normalized_pairs(KNOWN_LITERAL_MISTRANSLATIONS)
_NORMALIZED_IDIOMS = _normalized_pairs(KNOWN_LITERAL_IDIOMS)
_NORMALIZED_DOG = tuple(normalize(item) for item in DOG_PREFIX_PROFANITY)
_NORMALIZED_PROFANITY = tuple(normalize(item) for item in KOREAN_PROFANITY)
_NORMALIZED_VULGAR = tuple(normalize(item) for item in JAPANESE_VULGAR_MARKERS)


def assess_contamination(
    source: str,
    target: str,
    *,
    source_language: str = "ko",
    target_language: str = "ja",
) -> list[ContaminationFinding]:
    """한 쌍에 대한 의심 근거를 전부 돌려준다 (없으면 빈 목록).

    ko→ja 규칙만 있습니다. 다른 언어쌍은 빈 목록을 돌려주므로, 규칙 없는
    언어쌍이 조용히 "깨끗하다"고 보고되지 않도록 호출자가 언어를 확인해야
    합니다 — ``supported_direction`` 을 쓰십시오.
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
                    reason=f"욕설 {korean!r} 이 축자 번역 {japanese!r} 으로 옮겨졌습니다",
                    confidence=0.95,
                    evidence=(korean, japanese),
                )
            )

    for korean, japanese, flat_korean, flat_japanese in _NORMALIZED_IDIOMS:
        if flat_korean in flat_source and flat_japanese in flat_target:
            findings.append(
                ContaminationFinding(
                    rule="literal_idiom",
                    reason=f"관용구 {korean!r} 이 축자 번역 {japanese!r} 으로 옮겨졌습니다",
                    confidence=0.85,
                    evidence=(korean, japanese),
                )
            )

    dog = _contains_any(flat_source, _NORMALIZED_DOG)
    if dog and "犬" in flat_target:
        findings.append(
            ContaminationFinding(
                rule="dog_prefix_literal",
                reason=f"접두 욕설 {dog[0]!r} 이 동물 '犬' 으로 옮겨졌습니다",
                confidence=0.75,
                evidence=dog + ("犬",),
            )
        )

    # 목록에 의존하지 않는 규칙. 새 오염도 잡히지만 정밀도는 가장 낮습니다.
    profanity = _contains_any(flat_source, _NORMALIZED_PROFANITY)
    if profanity and not _contains_any(flat_target, _NORMALIZED_VULGAR):
        findings.append(
            ContaminationFinding(
                rule="profanity_intensity_lost",
                reason=(
                    f"원문의 욕설 {profanity[0]!r} 에 대응하는 비속·강조 표지가 번역문에 없습니다"
                ),
                confidence=0.45,
                evidence=profanity,
            )
        )
    return findings


def supported_direction(source_language: str, target_language: str) -> bool:
    """이 모듈이 규칙을 가진 방향인지.

    규칙이 없는 방향을 "오염 없음" 으로 보고하면 감사가 있으나 마나 합니다.
    """

    return (str(source_language), str(target_language)) == ("ko", "ja")


def rank_findings(findings: Sequence[ContaminationFinding]) -> ContaminationFinding | None:
    """검수 우선순위를 정할 대표 근거 (가장 확신이 높은 것)."""

    return max(findings, key=lambda finding: finding.confidence, default=None)


__all__ = [
    "DOG_PREFIX_PROFANITY",
    "JAPANESE_VULGAR_MARKERS",
    "KNOWN_LITERAL_IDIOMS",
    "KNOWN_LITERAL_MISTRANSLATIONS",
    "KOREAN_PROFANITY",
    "ContaminationFinding",
    "assess_contamination",
    "normalize",
    "rank_findings",
    "supported_direction",
]
