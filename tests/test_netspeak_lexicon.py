"""A one-sided net-speak rewrite teaches register loss, so pairing is the rule.

Internet register differs from dialect: a dialect row keeps one side standard,
because the goal is to understand dialect input. Here both sides must change, or
the pair teaches the model to drop the register it is supposed to learn. Every
refusal below was a real defect found while probing the generator on the corpus.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "netspeak_lexicon.py"
SPEC = importlib.util.spec_from_file_location("netspeak_lexicon_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
LEXICON = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LEXICON
SPEC.loader.exec_module(LEXICON)


def rewrite(code: str, ko: str, ja: str, variant: int = 0):
    style = LEXICON.style(code)
    assert style is not None, code
    return LEXICON.to_netspeak(ko, ja, style, variant=variant)


# --- formality must be judged over the whole sentence ---


@pytest.mark.parametrize(
    ("ko", "ja"),
    [
        # `합니다` is 합+니+다; the pattern `ㅂ니다` is a compatibility jamo and
        # never matched it, so this polite sentence used to get ㄹㅇ attached.
        ("너무 오랫동안 기다리게 해서 정말 죄송합니다", "随分と待たせちゃって本当にごめんね"),
        # Polite mid-sentence, casual at the end: an end-anchored test passes it.
        ("귀여워 좋아요. 귀엽네.", "可愛い。いいですね。可愛いな。"),
        ("진짜 위험해요", "マジで危ないんですよ"),
        ("확인했습니다", "確認しました"),
        ("조금 기다려 주세요", "少し待ってください"),
    ],
)
def test_a_polite_pair_is_refused(ko: str, ja: str) -> None:
    assert not LEXICON.is_casual_pair(ko, ja)
    assert rewrite("laughter", ko, ja) is None


def test_a_dictionary_form_ending_is_not_chat() -> None:
    # A quest objective is not something anyone types in a chat window.
    assert not LEXICON.is_casual_pair("메인 스토리 누적 20 회 클리어", "累計でメインストーリーを20回クリアする")


@pytest.mark.parametrize(
    ("ko", "ja"),
    [
        ("감사 인사를 전했다", "ありがとうと伝えた"),
        ("밥 먹었다", "ご飯食べた"),
        ("그렇게 말하였다", "そう言った"),
    ],
)
def test_narrative_prose_is_refused(ko: str, ja: str) -> None:
    # Appending ㅋㅋ to reported prose is not a register anyone writes.
    assert not LEXICON.is_casual_pair(ko, ja)


@pytest.mark.parametrize(
    ("ko", "ja"),
    [
        ("이거 진짜 대박이다", "これ本当にやばい"),
        ("너무 맛있어", "とてもうまい"),
        ("그건 좀 어려워", "それはちょっと難しい"),
        ("오늘 날씨 좋다", "今日は天気いい"),
        ("축하해", "おめでとう"),
    ],
)
def test_a_casual_pair_is_accepted(ko: str, ja: str) -> None:
    assert LEXICON.is_casual_pair(ko, ja)


# --- pairing ---


def test_both_sides_change_or_the_row_is_refused() -> None:
    result = rewrite("laughter", "이거 재밌어", "これ面白い")
    assert result is not None
    ko, ja, fired = result
    assert ko != "이거 재밌어"
    assert ja != "これ面白い"
    assert "laughter" in fired


def test_a_substitution_needs_both_standards_present() -> None:
    # Korean has 진짜 but the Japanese has no 本当, so nothing fires: rewriting
    # only the Korean would be the register mismatch this module exists to stop.
    assert rewrite("abbreviation", "이거 진짜 좋아", "これいい") is None
    assert rewrite("abbreviation", "이거 진짜 좋아", "これ本当にいい") is not None


def test_the_longest_japanese_standard_wins() -> None:
    # 本当 inside 本当に would leave マジに, which is not Japanese.
    result = rewrite("abbreviation", "이거 진짜 대박이다", "これ本当にやばい")
    assert result is not None
    assert result[1] == "これマジでやばい"
    assert result[0] == "이거 ㄹㅇ 대박이다"


# --- interjections need interjection position ---


def test_an_interjection_fires_when_it_stands_alone() -> None:
    assert rewrite("abbreviation", "고마워", "ありがとう")[:2] == ("ㄱㅅ", "あり")
    assert rewrite("abbreviation", "축하해", "おめでとう")[:2] == ("ㅊㅋ", "おめ")


def test_a_noun_phrase_is_not_an_interjection() -> None:
    # 감사 인사 is a noun phrase; ㄱㅅ 인사를 전했다 is nonsense.
    assert not LEXICON._is_interjection_position("감사 인사를 전했다", "감사")
    assert not LEXICON._is_interjection_position("축하의 말을 했어", "축하")
    assert LEXICON._is_interjection_position("감사", "감사")
    assert LEXICON._is_interjection_position("고마워 진짜", "고마워")


# --- intensifiers ---


def test_an_intensifier_attaches_to_a_paired_adjective() -> None:
    result = rewrite("intensifier", "오늘 날씨 좋아", "今日は天気いい")
    assert result is not None
    assert result[0] == "오늘 날씨 개좋아"
    assert result[1] == "今日は天気ガチいい"


def test_an_intensifier_needs_the_adjective_on_both_sides() -> None:
    assert rewrite("intensifier", "오늘 날씨 좋아", "いい天気だね") is not None
    assert rewrite("intensifier", "오늘 뭐 하지", "今日何しよう") is None


def test_the_intensifier_list_excludes_the_fusing_prefix() -> None:
    # 존 fuses into 존맛/존예, so 존 + 맛있어 gives 존맛있어, which is not a word.
    prefixes = {ko for ko, _ in LEXICON.INTENSIFIERS}
    assert "존" not in prefixes
    assert prefixes == {"개", "핵"}


# --- lament carries sentiment, so it cannot be picked by hash ---


def test_a_lament_marker_needs_a_negative_cue() -> None:
    # `오늘 날씨 좋다ㅠㅠ` reads as the opposite of what it says.
    assert rewrite("lament", "오늘 날씨 좋다", "今日は天気いい") is None
    result = rewrite("lament", "그건 좀 어려워", "それはちょっと難しい")
    assert result is not None
    assert result[0].endswith(("ㅠㅠ", "ㅜㅜ"))
    assert "lament" in result[2]


def test_laughter_never_carries_a_crying_marker() -> None:
    style = LEXICON.style("laughter")
    markers = {ko for ko, _ in style.laughter}
    assert "ㅠㅠ" not in markers
    assert "ㅋㅋ" in markers


# --- mechanics ---


def test_a_marker_goes_before_trailing_punctuation() -> None:
    assert LEXICON.append_marker("대박!", "ㅋㅋ") == "대박ㅋㅋ!"
    assert LEXICON.append_marker("やばい。", "w") == "やばいw。"
    assert LEXICON.append_marker("대박", "ㅋㅋ") == "대박ㅋㅋ"


def test_a_collapsed_abbreviation_gets_no_laughter() -> None:
    # 축하해 -> ㅊㅋ, and appending ㅋㅋ gives ㅊㅋㅋㅋ, which reads as neither.
    result = rewrite("abbreviation_laughter", "축하해", "おめでとう")
    assert result is not None
    assert result[0] == "ㅊㅋ"
    assert "laughter" not in result[2]


def test_the_variant_selects_which_marker_is_used() -> None:
    markers = set()
    for variant in range(len(LEXICON.LAUGHTER)):
        result = rewrite("laughter", "이거 재밌어", "これ面白い", variant=variant)
        assert result is not None
        markers.add(result[0])
    assert len(markers) > 1


def test_empty_input_is_safe() -> None:
    assert rewrite("laughter", "", "") is None
    assert rewrite("laughter", "재밌어", "") is None


def test_every_style_is_internally_consistent() -> None:
    assert LEXICON.known_styles()
    for style in LEXICON.STYLES:
        assert style.code and style.label
        assert style.laughter or style.substitutions or style.intensifiers or style.lament
        for entry in style.substitutions + style.interjections:
            ko_standard, ko_net, ja_standard, ja_net = entry
            assert ko_standard and ko_net and ja_standard and ja_net
            # Both sides must actually change, or the entry creates a mismatch.
            assert ko_standard != ko_net, entry
            assert ja_standard != ja_net, entry


def test_an_unknown_style_is_none() -> None:
    assert LEXICON.style("keyboard-smash") is None
