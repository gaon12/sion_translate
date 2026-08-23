"""오염된 정답쌍을 찾아내고, 확정적인 것만 고친다.

``data.quality.assess_pair`` 는 길이·문자 비율·반복을 봅니다. 그것으로는
**의미가 틀린 정답**을 잡을 수 없습니다. `씨발` 을 `種まき`(씨 뿌리기)로 옮긴
행은 길이도 문자 비율도 정상이라 ``accepted=True, score=100`` 으로 통과합니다.
모델은 그것을 정답으로 배웁니다.

## 무엇을 잡는가

1. **알려진 직역 매핑**: 실측으로 확인된 대응(`씨발` → `種まき`)입니다.
   `種まき` 가 `씨발` 자리에 오는 정상적인 문맥은 없으므로 정밀도가 사실상
   1.0 이고, **자동 수정 대상은 이 규칙 하나뿐**입니다.
2. **관용구 직역**: 한국어 관용구가 있는데 일본어가 축자 번역인 경우입니다.
3. **욕설 강도 소실**: 원문에 욕설이 있는데 번역문에 어떤 비속·강조 표지도
   없는 경우입니다. 목록에 의존하지 않아 새 오염도 잡습니다.

## 규칙마다 정밀도가 다르다 — 실측

897만 행을 훑어 나온 queue 를 직접 읽고 정밀도를 확인했습니다. 세 규칙이
전혀 다른 물건이었습니다.

* `known_literal_mistranslation` (89행) — 표본 전부가 진짜 오염이었습니다.
* `literal_idiom` (155행) — **대부분 오탐**이었습니다. `붕어빵을 입에 물고`,
  `펭수네 붕어빵` 은 전부 진짜 음식이고 `たい焼き` 가 맞는 번역입니다.
* `dog_prefix_literal` (84행) — **대부분 오탐**이었습니다. `개 소리가 들려`
  (개가 짖는 소리)가 공백을 지운 뒤 `개소리`(헛소리)와 같아졌기 때문입니다.

그래서 두 가지를 바꿨습니다. 첫째, `개`-접두 욕설은 **공백을 남긴 형태**에서
찾습니다 — `개 소리` 와 `개소리` 는 다른 말이고, 공백을 지우는 순간 그 구별이
사라집니다. 둘째, `붕어빵` 처럼 관용·축자 두 뜻이 다 살아 있는 표현은 관용
의미를 가리키는 **문맥 표지**를 함께 요구합니다 (:class:`LiteralIdiom` 의
``korean_sense_markers``). 표지가 없으면 음식으로 봅니다.

## 왜 대부분은 여전히 고치지 않는가

관용구 직역과 욕설 강도 소실은 **표시만** 합니다. 고치려면 대체할 일본어를
새로 써야 하는데, 그것은 규칙이 아니라 번역이고 사람이 할 일입니다.
:func:`repair_pair` 는 대체어가 문맥과 무관하게 결정되는 경우 — 축자 산물이
그 자리에 있다는 것 자체가 오류인 경우 — 에만 동작하고, 고친 뒤 그 행이
탐지에 더 이상 걸리지 않는지 **검증한 다음에야** 결과를 돌려줍니다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sion_translate.language_tags import LanguageTagError, parse_language_tag

# 한국어 욕설·비속 표현. 자소 분리와 반복을 견디도록 정규화 후 검사합니다.
KOREAN_PROFANITY = (
    "씨발",
    "시발",
    "씨팔",
    "좆",
    "존나",
    "졸라",
    "개새끼",
    # 맨 `새끼` 는 넣지 않습니다. 새끼 짐승을 뜻하는 평범한 말이라
    # `새끼 개`, `새끼 고양이` 가 전부 걸립니다 (실측 오탐). 욕설로 쓰일 때의
    # 형태만 등재합니다. `normalize` 가 공백을 지우므로 `이 새끼` 도 잡힙니다.
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


@dataclass(frozen=True)
class LiteralIdiom:
    """관용구 하나와, 그것이 관용 의미로 쓰였다고 볼 근거.

    ``korean_sense_markers`` 가 비어 있으면 그 표현은 관용 의미로만 쓰입니다 —
    `발 벗고 나서` 를 축자로 읽을 문맥은 없습니다. 비어 있지 않으면 표지 중
    하나가 원문에 함께 있어야 관용구로 봅니다. `붕어빵` 이 그런 경우로, 닮음을
    뜻하는 관용 용법과 밀가루 음식이라는 축자 용법이 둘 다 흔하고, 후자에서는
    `たい焼き` 가 **맞는 번역**입니다.
    """

    korean: str
    japanese_literal: str
    korean_sense_markers: tuple[str, ...] = ()


# 관용구와 그 축자 번역. 일본어에 축자 번역이 그대로 나오면 현지화 실패입니다.
KNOWN_LITERAL_IDIOMS: tuple[LiteralIdiom, ...] = (
    LiteralIdiom("같은 값이면 다홍치마", "紅スカート"),
    # 닮음을 뜻할 때만 관용구입니다. 표지가 없으면 먹는 붕어빵입니다.
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

# 개-접두 욕설이 동물 `犬` 으로 옮겨진 경우. `개` 가 실제 동물인 문장이 많아
# 단독으로는 근거가 약하므로, 접두사로 쓰인 형태만 봅니다. 공백을 남긴
# 형태에서 찾는 것이 핵심입니다 — `개 소리가 들려` 는 개가 짖는 소리이고,
# 공백을 지우면 `개소리`(헛소리)와 구별할 수 없게 됩니다.
DOG_PREFIX_PROFANITY = ("개새끼", "개자식", "개놈", "개년", "개소리", "개수작")

_REPEATED = re.compile(r"(.)\1{2,}")
_PUNCTUATION = r"[.,!?~·・…\-_*]+"


def normalize(text: str) -> str:
    """반복 문자와 자간 기호를 걷어낸 비교용 표현.

    공백까지 지웁니다. `씨 발` 처럼 자간을 벌려 쓴 욕설을 잡기 위해서입니다.
    그 대가로 띄어쓰기가 뜻을 가르는 쌍(`개 소리` 대 `개소리`)은 구별하지
    못하므로, 그런 규칙은 :func:`spaced_normalize` 를 씁니다.

    한계: `씨이이발` 처럼 **모음을 끼워 늘인** 형태는 되돌리지 못합니다.
    반복 축약은 같은 문자가 이어질 때만 동작하고, 늘임은 새 음절을 넣는
    변형이기 때문입니다. 그런 형태는 목록에 직접 넣어야 잡힙니다.
    """

    return re.sub(r"\s+", "", spaced_normalize(text))


def spaced_normalize(text: str) -> str:
    """:func:`normalize` 와 같되 **띄어쓰기를 하나로 줄여 남긴다**.

    띄어쓰기가 의미를 가르는 규칙에 씁니다. 실측에서 `개 소리가 들려` 가
    `개소리`(헛소리)로 잘못 걸린 것이 공백을 지웠기 때문이었습니다.
    """

    folded = unicodedata.normalize("NFKC", text)
    folded = _REPEATED.sub(r"\1", folded)
    folded = re.sub(_PUNCTUATION, "", folded)
    return re.sub(r"\s+", " ", folded).strip()


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
_NORMALIZED_IDIOMS: tuple[tuple[LiteralIdiom, str, str, tuple[str, ...]], ...] = tuple(
    (
        idiom,
        normalize(idiom.korean),
        normalize(idiom.japanese_literal),
        tuple(normalize(marker) for marker in idiom.korean_sense_markers),
    )
    for idiom in KNOWN_LITERAL_IDIOMS
)
# 공백을 남긴 형태에서 찾습니다. 위 DOG_PREFIX_PROFANITY 주석을 보십시오.
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

    for idiom, flat_korean, flat_japanese, flat_markers in _NORMALIZED_IDIOMS:
        if flat_korean not in flat_source or flat_japanese not in flat_target:
            continue
        # 표지를 요구하는 표현은 축자 의미로도 흔하게 쓰입니다. 표지가 없으면
        # 관용구가 아니라 그냥 그 사물이고, 축자 번역이 맞는 번역입니다.
        if flat_markers and not _contains_any(flat_source, flat_markers):
            continue
        findings.append(
            ContaminationFinding(
                rule="literal_idiom",
                reason=(
                    f"관용구 {idiom.korean!r} 이 축자 번역 "
                    f"{idiom.japanese_literal!r} 으로 옮겨졌습니다"
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


# 축자 산물과 그 자리에 있어야 할 표현. 여기 있는 열쇠는 **그 자리에 있다는
# 것 자체가 오류**인 문자열입니다. `種まき` 가 `씨발` 옆에 오는 정상적인 문맥은
# 없으므로 대체어가 문맥에 의존하지 않고, 그래서 규칙으로 고칠 수 있습니다.
#
# `くそ` 를 고른 이유는 강도와 품사가 둘 다 맞기 때문입니다. 감탄사로도
# (`このくそ`), 강조 접두사로도 (`くそ暑い`) 쓰이므로 `種まき` 가 나타나던
# 두 자리를 모두 감당합니다.
#
# 여기 없는 것: `좆`→`種` 같은 한 글자 대응입니다. `種` 는 흔한 보통명사라
# 문장 안의 멀쩡한 `種` 까지 바꿔 버립니다. 그런 행은 사람 검수로 갑니다.
LITERAL_ARTIFACT_REPAIRS: dict[str, str] = {
    "種まき": "くそ",
    "種の足": "くそ",
}

# 자동 수정을 허용하는 규칙. 나머지는 대체어를 새로 써야 하므로 사람이 합니다.
REPAIRABLE_RULES = frozenset({"known_literal_mistranslation"})


@dataclass(frozen=True)
class ContaminationRepair:
    """고친 결과 하나. 되돌릴 수 있도록 원본을 함께 들고 있습니다."""

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
    """확정적으로 고칠 수 있는 오염만 고친다. 아니면 ``None``.

    ``None`` 은 "깨끗하다"가 아니라 **"규칙으로 못 고친다"** 입니다. 호출자는
    :func:`assess_contamination` 으로 오염 여부를 따로 판단해서, 못 고친 행을
    사람 검수 queue 에 남겨야 합니다.

    고친 뒤 :func:`assess_contamination` 을 다시 돌려 그 행이 더 이상 걸리지
    않는지 확인합니다. 확인에 실패하면 고치지 않은 것으로 봅니다 — 절반만
    고쳐 놓고 탐지에서 사라지는 것이 원래 상태보다 나쁩니다.
    """

    findings = assess_contamination(
        source, target, source_language=source_language, target_language=target_language
    )
    # 고칠 근거가 하나는 있어야 합니다. 다른 규칙이 함께 걸린 것은 막지
    # 않습니다 — 축자 산물이 욕설을 밀어낸 행은 **반드시** `강도 소실` 도 같이
    # 띄우므로, 그것을 이유로 거절하면 고칠 수 있는 행이 하나도 남지 않습니다.
    # 진짜 관문은 아래의 사후 검증입니다.
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

    # 고친 결과가 깨끗한지 확인합니다. 남아 있으면 손대지 않은 것으로 처리해
    # 사람 검수로 보냅니다.
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
    """이 모듈이 규칙을 가진 방향인지.

    규칙이 없는 방향을 "오염 없음" 으로 보고하면 감사가 있으나 마나 합니다.
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
    """검수 우선순위를 정할 대표 근거 (가장 확신이 높은 것)."""

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
