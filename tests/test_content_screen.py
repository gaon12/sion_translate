"""The screen must fire on the conjunction and stay silent on either half alone.

A screen that flags ordinary sentences gets turned off, and one that never fires
is decoration. These tests pin both failure modes.
"""

from __future__ import annotations

import pytest

import sion_translate.content_screen as content_screen
from sion_translate.content_screen import (
    MAX_CHILD_AGE,
    child_ages,
    known_languages,
    markers_for,
    screen_pair,
    screen_text,
)


pytestmark = pytest.mark.usefixtures("configured_content_screen")


def test_a_child_marker_alone_is_not_flagged() -> None:
    for text in [
        "초등학교 앞에서 만나자",
        "유치원 버스가 도착했다",
        "어린이 요금은 절반입니다",
        "그 아이는 열두 살이다",
    ]:
        result = screen_text(text, "ko")
        assert not result.flagged, text
        assert result.child_markers or result.ages, text


def test_a_sexual_marker_alone_is_not_flagged() -> None:
    for text in [
        "둘은 그날 밤 섹스를 했다",
        "알몸으로 욕실에서 나왔다",
        "변태 같은 소리 하지 마",
    ]:
        result = screen_text(text, "ko")
        assert not result.flagged, text
        assert result.sexual_markers, text


def test_the_conjunction_is_flagged() -> None:
    result = screen_text("초등학생과 성관계를 했다", "ko")
    assert result.flagged
    assert "초등학생" in result.child_markers
    assert "성관계" in result.sexual_markers


def test_the_conjunction_is_flagged_in_japanese() -> None:
    result = screen_text("小学生と性交した", "ja")
    assert result.flagged
    assert "小学生" in result.child_markers
    assert "性交" in result.sexual_markers


def test_evidence_is_pooled_across_the_pair() -> None:
    # The child marker is on one side and the sexual marker on the other, which a
    # per-side check would miss entirely.
    result = screen_pair(
        "그 초등학생이 방으로 들어왔다",
        "その子が挿入をせがんだ",
        source_language="ko",
        target_language="ja",
    )
    assert result.flagged
    assert result.child_markers == ("초등학생",)
    assert result.sexual_markers == ("挿入",)


def test_high_school_is_deliberately_not_a_child_marker() -> None:
    # Written as adults in this genre; flagging it would flag a large share of the
    # corpus and say nothing useful.
    for term in ("고등학생", "고교생", "중학생"):
        assert term not in content_screen.CHILD_MARKERS["ko"], term
    for term in ("高校生", "中学生", "JK"):
        assert term not in content_screen.CHILD_MARKERS["ja"], term
    assert not screen_text("고등학생 커플이 섹스를 했다", "ko").flagged


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("열두 살 생일", (12,)),
        ("12살이었다", (12,)),
        ("14세 아동", (14,)),
        ("여섯 살 무렵", (6,)),
    ],
)
def test_korean_ages_are_read_with_their_counter(text: str, expected: tuple[int, ...]) -> None:
    assert child_ages(text, "ko") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("十二歳の誕生日", (12,)),
        ("12歳だった", (12,)),
        ("十四才", (14,)),
    ],
)
def test_japanese_ages_are_read_with_their_counter(text: str, expected: tuple[int, ...]) -> None:
    assert child_ages(text, "ja") == expected


def test_a_bare_number_is_not_an_age() -> None:
    # Quantities and years must never be mistaken for ages.
    assert child_ages("사과 12개를 샀다", "ko") == ()
    assert child_ages("2012년에 태어났다", "ko") == ()
    assert child_ages("12개の林檎", "ja") == ()
    assert child_ages("2012年", "ja") == ()
    assert child_ages("가격은 12원", "ko") == ()


def test_an_adult_age_is_not_a_child_marker() -> None:
    assert child_ages("스무 살이다", "ko") == ()
    assert child_ages("18살이다", "ko") == ()
    assert child_ages("二十歳", "ja") == ()
    assert not screen_text("18살이고 성관계에 동의했다", "ko").flagged


def test_the_age_cutoff_is_the_documented_one() -> None:
    assert MAX_CHILD_AGE == 14
    assert child_ages(f"{MAX_CHILD_AGE}살", "ko") == (MAX_CHILD_AGE,)
    assert child_ages(f"{MAX_CHILD_AGE + 1}살", "ko") == ()


def test_an_age_plus_a_sexual_marker_is_flagged() -> None:
    result = screen_text("12살 아이에게 삽입했다", "ko")
    assert result.flagged
    assert result.ages == (12,)


def test_an_unconfigured_language_is_not_screened() -> None:
    result = screen_text("anything at all", "en")
    assert not result.flagged
    assert result.child_markers == ()
    assert markers_for("en") == ((), ())
    pair = screen_pair("a", "b", source_language="en", target_language="de")
    assert not pair.flagged


def test_language_tags_are_case_and_space_insensitive() -> None:
    assert screen_text("초등학생과 성관계", " KO ").flagged


def test_empty_input_is_safe() -> None:
    assert not screen_text("", "ko").flagged
    assert not screen_pair("", "", source_language="ko", target_language="ja").flagged


def test_empty_policy_tables_disable_screening(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(content_screen, "CHILD_MARKERS", {"ko": (), "ja": ()})
    monkeypatch.setattr(content_screen, "SEXUAL_MARKERS", {"ko": (), "ja": ()})

    assert not screen_text("초등학생과 성관계", "ko").flagged
    assert not screen_pair(
        "초등학생과 성관계",
        "小学生と性交した",
        source_language="ko",
        target_language="ja",
    ).flagged
    assert markers_for("ko") == ((), ())


def test_both_marker_tables_cover_the_configured_pair() -> None:
    assert known_languages() == ("ja", "ko")
    for language in known_languages():
        child, sexual = markers_for(language)
        assert child, language
        assert sexual, language
        assert len(set(child)) == len(child), language
        assert len(set(sexual)) == len(sexual), language
        # No empty strings, which would match every row.
        assert all(term for term in child + sexual), language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Longest-match: 十四 must win over 四, and 열두 over 두.
        ("十四才", (14,)),
        ("열두 살", (12,)),
        ("열네 살", (14,)),
        # Adult numerals are in the table so they cannot decay to a child age:
        # without 二十, this would read as 十 = 10.
        ("二十歳", ()),
        ("스무 살", ()),
        ("스물한 살", ()),
        ("十九歳", ()),
        ("열아홉 살", ()),
        ("三十歳", ()),
    ],
)
def test_spelled_ages_use_longest_match(text: str, expected: tuple[int, ...]) -> None:
    language = "ja" if any("\u4e00" <= char <= "\u9fff" for char in text) else "ko"
    assert child_ages(text, language) == expected


def test_a_numeral_without_a_counter_is_not_an_age() -> None:
    # 열 alone is "ten" or "open"; only the counter makes it an age.
    assert child_ages("열 개를 샀다", "ko") == ()
    assert child_ages("十個買った", "ja") == ()
    assert child_ages("문을 열었다", "ko") == ()
