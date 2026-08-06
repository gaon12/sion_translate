"""오염된 정답쌍 탐지.

``assess_pair`` 는 길이·문자 비율·반복만 봅니다. 의미가 틀린 정답은
그 검사를 전부 통과하므로 별도의 규칙이 필요합니다.
"""

from __future__ import annotations

import pytest

from sion_translate.contamination import (
    ContaminationFinding,
    assess_contamination,
    normalize,
    rank_findings,
    supported_direction,
)
from sion_translate.data.quality import QualityPolicy, assess_pair


def test_the_documented_contamination_passes_the_quality_filter() -> None:
    """이 모듈이 존재하는 이유.

    실제 원천에서 확인된 오역이 기존 품질 필터를 만점으로 통과합니다.
    """
    accepted = assess_pair("씨발 진짜 짜증나네", "種まき 本当にうざい", QualityPolicy())
    assert accepted.accepted


def test_the_known_profanity_mistranslation_is_flagged() -> None:
    findings = assess_contamination("씨발 진짜 짜증나네", "種まき 本当にうざい")
    leader = rank_findings(findings)
    assert leader is not None
    assert leader.rule == "known_literal_mistranslation"
    assert leader.confidence > 0.9


def test_spacing_and_repetition_do_not_hide_the_mistranslation() -> None:
    """자간·문장부호·반복 문자는 걷어내고 비교한다."""
    findings = assess_contamination("씨 발!!! 아 진짜", "種 ま き ああ本当に")
    assert any(finding.rule == "known_literal_mistranslation" for finding in findings)


def test_a_vowel_stretched_insult_is_a_known_limitation() -> None:
    """`씨이이발` 은 모음을 끼워 넣은 변형이라 반복 축약으로 되돌릴 수 없다.

    잡으려면 목록에 직접 넣어야 합니다. 잡히는 척하지 않도록 고정해 둡니다.
    """
    assert "씨발" not in normalize("씨이이발")
    assert assess_contamination("씨이이발", "種まき") == []


def test_a_literally_translated_idiom_is_flagged() -> None:
    findings = assess_contamination(
        "같은 값이면 다홍치마라고 하잖아",
        "同じ値段なら紅スカートと言うでしょう",
    )
    assert any(finding.rule == "literal_idiom" for finding in findings)


def test_the_lookalike_idiom_is_flagged() -> None:
    """닮았다는 뜻의 `붕어빵` 이 음식 `たい焼き` 로 옮겨진 경우."""
    findings = assess_contamination("아빠랑 붕어빵이네", "パパとたい焼きだね")
    assert any(finding.rule == "literal_idiom" for finding in findings)


def test_a_dog_prefix_insult_translated_as_the_animal_is_flagged() -> None:
    findings = assess_contamination("이 개새끼야", "この犬野郎")
    assert any(finding.rule == "dog_prefix_literal" for finding in findings)


def test_profanity_losing_its_intensity_is_flagged_without_a_lookup_table() -> None:
    """목록에 없는 새 오염도 잡아야 한다."""
    findings = assess_contamination("존나 짜증나", "とても嫌です")
    assert any(finding.rule == "profanity_intensity_lost" for finding in findings)


def test_a_correctly_localized_insult_is_not_flagged() -> None:
    """비속 표지가 살아 있으면 오염이 아니다. 오탐이 많으면 queue 가 무용해진다."""
    assert assess_contamination("씨발 진짜 짜증나네", "くそ、本当にむかつく") == []
    assert assess_contamination("이 개새끼야", "このクソ野郎") == []


def test_ordinary_pairs_are_not_flagged() -> None:
    assert assess_contamination("오늘 날씨가 좋네요", "今日は天気がいいですね") == []
    assert assess_contamination("붕어빵 두 개 주세요", "たい焼きを二つください") != []  # 동음이의


def test_an_unsupported_direction_returns_nothing_and_says_so() -> None:
    """규칙 없는 방향을 '오염 없음' 으로 보고하면 감사가 무용해진다."""
    assert (
        assess_contamination("hello", "bonjour", source_language="en", target_language="fr") == []
    )
    assert not supported_direction("en", "fr")
    assert supported_direction("ko", "ja")


def test_normalize_strips_spacing_and_repetition() -> None:
    assert normalize("씨 발!!!") == "씨발"
    assert normalize("아아아 진짜") == "아진짜"


def test_ranking_picks_the_most_confident_reason() -> None:
    findings = [
        ContaminationFinding("low", "약함", 0.2),
        ContaminationFinding("high", "강함", 0.9),
    ]
    assert rank_findings(findings).rule == "high"
    assert rank_findings([]) is None


@pytest.mark.parametrize(
    "source, target",
    [
        ("씨발", "種まき"),
        ("시발", "種の足"),
        ("같은 값이면 다홍치마", "紅スカート"),
    ],
)
def test_every_documented_case_is_covered(source: str, target: str) -> None:
    """로스트에 기록된 실제 사례가 전부 잡혀야 한다."""
    assert assess_contamination(source, target)
