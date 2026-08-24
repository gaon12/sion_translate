"""Verify detection and conservative repair of contaminated target pairs.

``assess_pair`` checks length, script ratios, and repetition. A semantically
wrong target can pass all of those checks and therefore needs separate rules.
"""

from __future__ import annotations

import pytest

from sion_translate.contamination import (
    ContaminationFinding,
    ContaminationRepair,
    assess_contamination as _assess_contamination,
    normalize,
    rank_findings,
    repair_pair as _repair_pair,
    spaced_normalize,
    supported_direction,
)
from sion_translate.data.quality import QualityPolicy, assess_pair


def assess_contamination(
    source: str,
    target: str,
    *,
    source_language: str = "ko",
    target_language: str = "ja",
) -> list[ContaminationFinding]:
    return _assess_contamination(
        source,
        target,
        source_language=source_language,
        target_language=target_language,
    )


def repair_pair(source: str, target: str) -> ContaminationRepair | None:
    return _repair_pair(
        source,
        target,
        source_language="ko",
        target_language="ja",
    )


def test_contamination_apis_require_explicit_language_identity() -> None:
    with pytest.raises(TypeError, match="source_language"):
        _assess_contamination("source", "target")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="source_language"):
        _repair_pair("source", "target")  # type: ignore[call-arg]


def test_the_documented_contamination_passes_the_quality_filter() -> None:
    """Demonstrate why a semantic contamination detector is necessary.

    A mistranslation measured in the source corpus passes the old surface filter
    with a perfect score.
    """
    accepted = assess_pair(
        "씨발 진짜 짜증나네",
        "種まき 本当にうざい",
        QualityPolicy(),
        languages=("ko", "ja"),
    )
    assert accepted.accepted


def test_the_known_profanity_mistranslation_is_flagged() -> None:
    findings = assess_contamination("씨발 진짜 짜증나네", "種まき 本当にうざい")
    leader = rank_findings(findings)
    assert leader is not None
    assert leader.rule == "known_literal_mistranslation"
    assert leader.confidence > 0.9


def test_spacing_and_repetition_do_not_hide_the_mistranslation() -> None:
    """Normalize spacing, punctuation, and repeated characters before matching."""
    findings = assess_contamination("씨 발!!! 아 진짜", "種 ま き ああ本当に")
    assert any(finding.rule == "known_literal_mistranslation" for finding in findings)


def test_a_vowel_stretched_insult_is_a_known_limitation() -> None:
    """Document that inserted-vowel stretching is outside normalizer coverage.

    Detecting `씨이이발` requires a direct lexicon entry. This regression test
    prevents the system from pretending that repetition collapse handles it.
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
    """Flag `붕어빵` as literal only when context indicates resemblance."""
    findings = assess_contamination("아빠랑 붕어빵이네", "パパとたい焼きだね")
    assert any(finding.rule == "literal_idiom" for finding in findings)

    findings = assess_contamination(
        "남동생과 붕어빵처럼 똑 닮았다", "弟とたい焼きのようにそっくりだ"
    )
    assert any(finding.rule == "literal_idiom" for finding in findings)


def test_the_food_sense_of_the_lookalike_word_is_not_flagged() -> None:
    """Prevent measured false positives for literal food references.

    Most of the 155 queued ``literal_idiom`` rows referred to real food. Without
    a resemblance marker, `たい焼き` is valid and must not trigger the idiom rule.
    """

    for source, target in (
        ("붕어빵 두 개 주세요", "たい焼きを二つください"),
        ("김명수가 붕어빵을 입에 물고 있다", "キム・ミョンスがたい焼きをくわえている"),
        ("펭수네 붕어빵 가게", "ペンスネたい焼き店"),
    ):
        assert assess_contamination(source, target) == [], source


def test_a_dog_prefix_insult_translated_as_the_animal_is_flagged() -> None:
    findings = assess_contamination("이 개새끼야", "この犬野郎")
    assert any(finding.rule == "dog_prefix_literal" for finding in findings)

    findings = assess_contamination("개소리 하지 마", "犬のこと言わないで")
    assert any(finding.rule == "dog_prefix_literal" for finding in findings)


def test_a_spaced_animal_reference_is_not_a_dog_prefix_insult() -> None:
    """Keep the measured dog-sound false positive distinct from prefix profanity."""

    assert (
        assess_contamination(
            "길마다 개 소리가 들려 웃음을 자아냈다", "犬の声が聞こえて笑いを誘った"
        )
        == []
    )
    assert assess_contamination("어미 개에 딸린 새끼 개의 형국", "母犬にくっつく子犬の形") == []


def test_profanity_losing_its_intensity_is_flagged_without_a_lookup_table() -> None:
    """Catch unseen contamination through the generic intensity-loss rule."""
    findings = assess_contamination("존나 짜증나", "とても嫌です")
    assert any(finding.rule == "profanity_intensity_lost" for finding in findings)


def test_a_correctly_localized_insult_is_not_flagged() -> None:
    """Do not flag a target that retains an appropriate vulgarity marker."""
    assert assess_contamination("씨발 진짜 짜증나네", "くそ、本当にむかつく") == []
    assert assess_contamination("이 개새끼야", "このクソ野郎") == []


def test_ordinary_pairs_are_not_flagged() -> None:
    assert assess_contamination("오늘 날씨가 좋네요", "今日は天気がいいですね") == []


def test_an_unsupported_direction_returns_nothing_and_says_so() -> None:
    """Make unsupported directions explicit instead of reporting them as clean."""
    assert (
        assess_contamination("hello", "bonjour", source_language="en", target_language="fr") == []
    )
    assert not supported_direction("en", "fr")
    assert supported_direction("ko", "ja")


def test_language_variants_inherit_the_supported_contamination_direction() -> None:
    assert supported_direction("KO-kr", "ja-jp")
    findings = assess_contamination(
        "씨발 진짜 짜증나네",
        "種まき 本当にうざい",
        source_language="KO-kr",
        target_language="ja-jp",
    )
    assert any(finding.rule == "known_literal_mistranslation" for finding in findings)


def test_invalid_language_tags_are_not_supported_for_contamination() -> None:
    assert not supported_direction("ko_KR", "ja-JP")


def test_normalize_strips_spacing_and_repetition() -> None:
    assert normalize("씨 발!!!") == "씨발"
    assert normalize("아아아 진짜") == "아진짜"


def test_spaced_normalize_keeps_the_boundary_that_carries_meaning() -> None:
    assert spaced_normalize("개 소리!!!") == "개 소리"
    assert spaced_normalize("개소리") == "개소리"
    assert spaced_normalize("개  소리") != spaced_normalize("개소리")


def test_the_known_literal_artifact_is_repaired_and_verified() -> None:
    """Repair a context-independent literal artifact and validate the result."""

    repair = repair_pair("씨발 진짜 짜증나네", "種まき 本当にうざい")
    assert repair is not None
    assert repair.changed
    assert repair.target == "くそ 本当にうざい"
    assert repair.original_target == "種まき 本当にうざい"
    assert ("種まき", "くそ") in repair.replacements
    # ``repair_pair`` itself must ensure the repaired result no longer triggers.
    assert assess_contamination("씨발 진짜 짜증나네", repair.target) == []


def test_every_literal_artifact_occurrence_is_repaired() -> None:
    repair = repair_pair("이 씨발, 어떤 씨발년이", "この種まき、ある種まきは")
    assert repair is not None
    assert "種まき" not in repair.target


def test_a_pair_needing_a_new_translation_is_not_repaired() -> None:
    """Leave contextual idiom and intensity corrections to human translators."""

    assert repair_pair("아빠랑 붕어빵이네", "パパとたい焼きだね") is None
    assert repair_pair("존나 짜증나", "とても嫌です") is None
    assert repair_pair("개소리 하지 마", "犬のこと言わないで") is None


def test_a_clean_pair_is_not_repaired() -> None:
    """Treat ``None`` as not repairable, not as proof that a row is clean."""

    assert repair_pair("오늘 날씨가 좋네요", "今日は天気がいいですね") is None
    assert repair_pair("씨발 진짜 짜증나네", "くそ、本当にむかつく") is None


def test_ranking_picks_the_most_confident_reason() -> None:
    findings = [
        ContaminationFinding("low", "약함", 0.2),
        ContaminationFinding("high", "강함", 0.9),
    ]
    leader = rank_findings(findings)
    assert leader is not None
    assert leader.rule == "high"
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
    """Keep every measured roast example covered by a detection rule."""
    assert assess_contamination(source, target)
