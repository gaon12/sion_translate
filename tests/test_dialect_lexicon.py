"""A wrong dialect rule teaches wrong grammar, so refusing must be reliable.

방언 is the worst diagnostic category (chrF 9.89 against 3,409 rows), and the fix
is more data - but only if the data is right. These tests pin the rewrites that
must happen, and, more importantly, the ones that must not: every case here was a
real defect caught while probing the generator against the corpus.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "dialect_lexicon.py"
SPEC = importlib.util.spec_from_file_location("dialect_lexicon_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
LEXICON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LEXICON
SPEC.loader.exec_module(LEXICON)


def rewrite(language: str, code: str, text: str) -> str | None:
    profile = LEXICON.profile(language, code)
    assert profile is not None, code
    result = LEXICON.to_dialect(text, profile)
    return None if result is None else result[0]


def test_both_languages_have_many_regions() -> None:
    ko = [profile.code for profile in LEXICON.profiles_for("ko")]
    ja = [profile.code for profile in LEXICON.profiles_for("ja")]
    assert len(ko) >= 8, ko
    assert len(ja) >= 14, ja
    assert len(set(ko)) == len(ko)
    assert len(set(ja)) == len(ja)


def test_every_profile_is_internally_consistent() -> None:
    for profile in LEXICON.all_profiles():
        assert profile.code and profile.label and profile.language
        assert profile.endings, profile.code
        for standard, dialect in profile.endings + profile.wh_endings:
            assert standard, profile.code
            assert dialect, profile.code
        for standard, dialect in profile.vocabulary:
            assert standard != dialect, (profile.code, standard)


def test_an_unconfigured_language_yields_nothing() -> None:
    assert LEXICON.profiles_for("en") == ()
    assert LEXICON.profile("en", "cockney") is None


# --- the yes/no versus wh contrast, the headline feature of the 동남 varieties ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("밥 먹었니?", "밥 먹었나?"),
        ("지금 가니?", "지금 가나?"),
    ],
)
def test_a_yes_no_question_takes_na(text: str, expected: str) -> None:
    assert rewrite("ko", "gyeongsang", text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("뭐 먹었니?", "뭐 먹었노?"),
        ("어디 가니?", "어디 가노?"),
        ("왜 그러니?", "왜 그러노?"),
        ("누가 왔니?", "누가 왔노?"),
    ],
)
def test_a_wh_question_takes_no(text: str, expected: str) -> None:
    assert rewrite("ko", "gyeongsang", text) == expected


def test_a_question_mark_alone_does_not_select_the_wh_form() -> None:
    # This was the bug: keying on "?" made every question take -노, which erases
    # the contrast the variety is known for.
    assert rewrite("ko", "gyeongsang", "밥 먹었니?") == "밥 먹었나?"


# --- Korean rewrites that must refuse ---


@pytest.mark.parametrize(
    "text",
    [
        "순찰 횟수를 늘리다니…",
        "그런 말을 하다니",
        "이렇게 될 줄 알았더니",
    ],
)
def test_an_exclamative_is_not_an_interrogative(text: str) -> None:
    # ~다니 is exclamative. Treating it as the question ~니 gives 늘리다냐.
    assert rewrite("ko", "jeonnam", text) is None


def test_jeju_drops_the_stem_vowel_rather_than_stacking_it() -> None:
    # 망가졌어수다 was wrong; 제주 attaches 수다 to the stem.
    assert rewrite("ko", "jeju", "수확한 물자가 망가졌어요") == "수확한 물자가 망가졌수다"
    assert rewrite("ko", "jeju", "좋아요") == "좋수다"
    assert rewrite("ko", "jeju", "일괄 판매를 종료하시겠습니까?") == "일괄 판매를 종료하시겠수과?"


@pytest.mark.parametrize(
    ("code", "text", "expected"),
    [
        ("chungcheong", "요즘 자주 수다를 떨어요.", "요즘 자주 수다를 떨어유."),
        ("gangwon", "6년 동안 따라다녀서 사귀게 됐습니다.", "6년 동안 따라다녀서 사귀게 됐슴다."),
        ("pyeongan", "꽃무늬 셔츠입니다.", "꽃무늬 셔츠입네다."),
        ("hamgyeong", "알림이 왔어요?", "알림이 왔슴메?"),
        ("jeonnam", "과찬이야", "과찬이여"),
        ("jeonbuk", "여기 있는 게 다 뚱카롱이에요?", "여기 있는 게 다 뚱카롱이지라?"),
    ],
)
def test_each_korean_region_rewrites_its_own_way(code: str, text: str, expected: str) -> None:
    assert rewrite("ko", code, text) == expected


def test_the_korean_regions_do_not_all_produce_the_same_text() -> None:
    # Two profiles that emit identical strings are one profile pretending to be
    # two, which is why 경남/경북 were merged.
    text = "지금 어디 가요?"
    outputs = {
        profile.code: rewrite("ko", profile.code, text) for profile in LEXICON.profiles_for("ko")
    }
    produced = [value for value in outputs.values() if value is not None]
    assert len(set(produced)) > 1, outputs


# --- Japanese: the ない rule is the dangerous one ---


@pytest.mark.parametrize(
    ("code", "text", "expected"),
    [
        ("kansai", "わからない。", "わからへん。"),
        ("hakata", "行かない。", "行かん。"),
        ("kansai", "私は働いています。", "私は働いとります。"),
        ("hakata", "彼らは長く会っていない。", "彼らは長く会っとらん。"),
        ("kansai", "彼らは長く会っていない。", "彼らは長く会っとらへん。"),
    ],
)
def test_a_verb_negative_becomes_the_dialect_negative(code: str, text: str, expected: str) -> None:
    assert rewrite("ja", code, text) == expected


@pytest.mark.parametrize(
    ("code", "text"),
    [
        # Existence negative after a particle. 一つもん and 写真がん are not Japanese.
        ("kumamoto", "はんこが一つもない。"),
        ("tosa", "その次は写真がない。"),
        ("kansai", "時間がない。"),
        # Adjectives that merely end in ない.
        ("kansai", "少ない。"),
        ("kansai", "それは危ない。"),
        # じゃない is standard casual, not this variety's form.
        ("hiroshima", "一応全部そろってるじゃない"),
    ],
)
def test_nai_that_is_not_a_verb_negative_refuses(code: str, text: str) -> None:
    assert rewrite("ja", code, text) is None


def test_a_more_specific_rule_still_fires_over_its_own_block() -> None:
    # kansai blocks じゃない for the bare ない rule but keeps the explicit
    # ("じゃない", "やない") rule. Blocking globally would kill both.
    assert rewrite("ja", "kansai", "これはええんじゃない") == "これはええんやない"


@pytest.mark.parametrize("code", ["kansai", "kyoto", "hakata", "hiroshima", "kagoshima"])
def test_a_non_copula_da_refuses(code: str) -> None:
    # まだ is not a copula; rewriting it gives まや.
    assert rewrite("ja", code, "まだ。") is None


def test_dame_da_becomes_one_word_rather_than_stacking() -> None:
    # A だめ -> あかん vocabulary swap plus the copula rule produced あかんや.
    assert rewrite("ja", "kansai", "これは本当にだめだ。") == "これはほんまにあかん。"
    assert rewrite("ja", "kansai", "それはだめ。") == "それはあかん。"


@pytest.mark.parametrize(
    ("code", "text", "expected"),
    [
        ("hiroshima", "悲しいゲームだね", "悲しいゲームじゃのう"),
        ("hakata", "悲しいゲームだね", "悲しいゲームばいね"),
        ("kagoshima", "悲しいゲームだね", "悲しいゲームじゃね"),
        ("nagoya", "二つとも増えるんだ。", "二つとも増えるんだがや。"),
        ("okinawa", "人生を捧げた理由だ。", "人生を捧げた理由やっさ。"),
        ("hokkaido", "これが問題だよ。", "これが問題だべ。"),
        ("tsugaru", "いいんだよ。", "いいんだでば。"),
        ("kyoto", "有名だよ。", "有名やで。"),
    ],
)
def test_each_japanese_region_rewrites_its_own_way(code: str, text: str, expected: str) -> None:
    assert rewrite("ja", code, text) == expected


def test_the_japanese_regions_do_not_all_produce_the_same_text() -> None:
    text = "これは面白いゲームだね"
    outputs = {
        profile.code: rewrite("ja", profile.code, text) for profile in LEXICON.profiles_for("ja")
    }
    produced = [value for value in outputs.values() if value is not None]
    assert len(set(produced)) >= 4, outputs


# --- mechanics ---


def test_an_unmatched_sentence_is_refused_rather_than_passed_through() -> None:
    # Emitting the input unchanged would label standard text as dialect.
    assert rewrite("ko", "jeju", "그렇구나") is None
    assert rewrite("ja", "kansai", "食べた") is None


def test_an_ending_the_variety_shares_with_the_standard_refuses() -> None:
    # 전남 keeps 했습니다, and matching it must stop a shorter rule from firing.
    assert rewrite("ko", "jeonnam", "확인했습니다") is None


def test_trailing_punctuation_survives_the_rewrite() -> None:
    assert rewrite("ko", "chungcheong", "맛있어요!!") == "맛있어유!!"
    assert rewrite("ja", "kansai", "働いています……") == "働いとります……"
    assert rewrite("ko", "chungcheong", "맛있어요") == "맛있어유"


def test_split_trailing_separates_punctuation_only() -> None:
    assert LEXICON.split_trailing("먹었니?") == ("먹었니", "?")
    assert LEXICON.split_trailing("そうや。") == ("そうや", "。")
    assert LEXICON.split_trailing("없다") == ("없다", "")
    assert LEXICON.split_trailing("") == ("", "")


def test_an_empty_stem_refuses() -> None:
    # The whole sentence being the ending leaves nothing to attach to.
    assert rewrite("ko", "chungcheong", "해") is None


def test_interrogative_detection_is_korean_only() -> None:
    assert LEXICON.is_interrogative("뭐 먹었니", "ko")
    assert not LEXICON.is_interrogative("밥 먹었니", "ko")
    # Japanese has no such contrast, so the wh table is never selected.
    assert not LEXICON.is_interrogative("何を食べた", "ja")
