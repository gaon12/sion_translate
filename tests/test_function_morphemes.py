"""A stranded particle is the only reliable trace of a deleted name placeholder.

The damage is symmetric - both sides lose the same noun - so the pair still
scores as a valid translation and every similarity threshold passes it. These
tests pin the real damaged rows found in data25 and data50, pin the words the
detector must not touch (a false positive silently deletes a good row), and pin
the one case the detector knowingly cannot reach.
"""

from __future__ import annotations

import pytest

from sion_translate.function_morphemes import (
    ORPHAN_FUNCTION_TOKENS,
    has_placeholder_hole,
    known_languages,
    orphan_function_tokens,
    placeholder_hole_markers,
    stranded_function_markers,
)


@pytest.mark.parametrize(
    "text",
    [
        "늦은 밤, 의 숲속에서 사냥꾼이 집으로 돌아갈 준비를 해.",
        "아침에 친구가 와서 에 뭔가 이상하다고 했어.",
        "이상하네, 언제부터인지 으로 가는 다리가 끊어졌어.",
        "(일제히) 의 보물을 찾아라!",
    ],
)
def test_real_damaged_korean_rows_are_detected(text: str) -> None:
    assert has_placeholder_hole(text, "ko")


@pytest.mark.parametrize(
    "text",
    [
        "小さくうなずいた後、 が のほうによってくる。",
        "深夜、 の森の中、猟師が家に帰る準備してる.",
        "１時間後、 と の姿はベッドの上にあった。",
        "（全員） の宝探しだ!",
    ],
)
def test_real_damaged_japanese_rows_are_detected(text: str) -> None:
    assert has_placeholder_hole(text, "ja")


@pytest.mark.parametrize(
    "text",
    [
        "늦은 밤의 숲속에서 사냥꾼이 집으로 돌아갈 준비를 해.",
        "이 사람이 그 유명한 번역가야.",
        "그 도시까지 가는 표를 두 장 주세요.",
        "만 원짜리 지폐를 들고 있었다.",
        "도 단위 행정 구역이 개편되었다.",
        "가! 지금 당장 가라고!",
        "와! 진짜 대박이다!",
        "은 세공품을 하나 샀다.",
        "과 대표가 인사를 했다.",
    ],
)
def test_healthy_korean_is_left_alone(text: str) -> None:
    assert placeholder_hole_markers(text, "ko") == ()


def test_ambiguous_korean_particles_are_deliberately_absent() -> None:
    # Each of these is also a free-standing word, so including it would delete
    # good rows - worse than keeping a rare damaged one.
    for token in ("이", "가", "와", "과", "은", "도", "만", "뿐", "밖에"):
        assert token not in ORPHAN_FUNCTION_TOKENS["ko"], token


def test_segmenter_spacing_is_not_mistaken_for_a_hole() -> None:
    # Fully segmented Japanese has a space before every particle. The collapse
    # runs first, so none of them are reported.
    assert stranded_function_markers("お互い の 体温 が 伝わり 、 緊張 が ほぐれた 。", "ja") == ()
    assert stranded_function_markers("甘い 香り が 鼻先 を かすめる", "ja") == ()


def test_interpolated_item_names_are_not_mistaken_for_a_hole() -> None:
    # The noun is present; the spaces around it come from string interpolation.
    assert placeholder_hole_markers("こんなの 伝説の鉄原石 なわけねぇだろ!", "ja") == ()
    assert placeholder_hole_markers("お前が見つけたこれ、 伝説の鉄原石 だ!", "ja") == ()


def test_a_space_after_punctuation_alone_is_not_a_hole() -> None:
    # Japanese does space after a question mark; only a particle makes it a hole.
    assert placeholder_hole_markers("え？ 何？", "ja") == ()
    assert placeholder_hole_markers("鉄 3 個", "ja") == ()


def test_unsegmented_japanese_has_no_markers() -> None:
    assert placeholder_hole_markers("恋人がいつものように声をかけてきた。", "ja") == ()


def test_the_undetectable_case_is_pinned_as_undetectable() -> None:
    # `恋人の が` sits between two spaceless characters, so the collapse removes
    # it exactly as it removes segmenter spacing. recover_shard.py's density
    # check is what drops this row; claiming coverage here would be a lie.
    assert placeholder_hole_markers("仕事時間が終わってから、恋人の がいつものように", "ja") == ()


def test_punctuation_around_a_stranded_particle_is_ignored() -> None:
    assert orphan_function_tokens("친구가 왔다. 의, 집으로 갔다.", "ko") == ("의",)
    assert placeholder_hole_markers("「 は」と言った", "ja") == ("は",)
    assert placeholder_hole_markers("(의) 보물", "ko") == ("의",)


def test_every_stranded_token_is_reported() -> None:
    assert placeholder_hole_markers("그건 의 것이고 에 두었다", "ko") == ("의", "에")
    # This row has two holes but only the one after punctuation survives the
    # collapse. Reporting one marker is enough to drop the row, which is the
    # decision the caller actually makes.
    assert placeholder_hole_markers("小さくうなずいた後、 が のほうに", "ja") == ("が",)


def test_the_longest_matching_particle_wins() -> None:
    assert stranded_function_markers("それ、 からは違う", "ja") == ("からは",)
    assert stranded_function_markers("それ、 には無い", "ja") == ("には",)


def test_hanboneo_accepts_either_languages_particles() -> None:
    assert ORPHAN_FUNCTION_TOKENS["kj"] == (
        ORPHAN_FUNCTION_TOKENS["ko"] | ORPHAN_FUNCTION_TOKENS["ja"]
    )
    assert has_placeholder_hole("엌ㅋㅋ 의 유리와 튼튼데스네", "kj")
    assert has_placeholder_hole("엌ㅋㅋ、 の 유리와 튼튼데스네", "kj")


def test_an_unconfigured_language_is_never_penalised() -> None:
    assert placeholder_hole_markers("the of a to", "en") == ()
    assert not has_placeholder_hole("의 에 으로", "en")


def test_language_tags_are_case_and_space_insensitive() -> None:
    assert placeholder_hole_markers("의 숲", " KO ") == ("의",)


def test_empty_input_is_safe() -> None:
    assert placeholder_hole_markers("", "ko") == ()
    assert placeholder_hole_markers("   ", "ko") == ()
    assert placeholder_hole_markers("...", "ko") == ()
    assert placeholder_hole_markers("", "ja") == ()


def test_listing_helper_covers_the_configured_pairs() -> None:
    assert known_languages() == ("ja", "kj", "ko")
