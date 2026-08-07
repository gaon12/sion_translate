"""MinHash split keys group near-duplicates that exact keys let through.

``normalized_split_key`` is exact-match, so two rows differing by one particle
land in independent splits. That is how a holdout stops carrying information.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from sion_translate.splitting import (
    SHINGLE_SIZE,
    SIGNATURE_LENGTH,
    approximate_split_key,
    character_shingles,
    choose_split_for_key,
    comparison_key,
    minhash_signature,
    normalized_split_key,
)


def test_comparison_key_drops_punctuation_where_the_split_key_keeps_it() -> None:
    """두 키는 서로 다른 질문에 답한다.

    dedup 은 "이 두 행이 같은 행인가" 이고 구두점은 유효한 차이입니다.
    누출 감사는 "이 문장이 이미 코퍼스에 있는가" 이고 거기서는 아닙니다.
    """

    assert normalized_split_key("김칫국부터 마시지 마.") != normalized_split_key(
        "김칫국부터 마시지 마…"
    )
    assert comparison_key("김칫국부터 마시지 마.") == comparison_key("김칫국부터 마시지 마…")


def test_comparison_key_drops_by_category_not_by_a_listed_alphabet() -> None:
    """목록에 없는 기호가 비교를 조용히 무력화하면 안 된다."""

    plain = comparison_key("최고")
    for decorated in ("최고!", "「최고」", "최고～", "최고 ✨", "최고\u200b"):
        assert comparison_key(decorated) == plain, decorated

    # 글자·숫자는 살아남는다. 지우면 서로 다른 문장이 같아진다.
    assert comparison_key("가격은 1,000원") == "가격은1000원"
    assert comparison_key("ＡＢＣ") == "ABC"

    # 자모 웃음은 기호가 아니라 글자다. 이것이 지워지면 `ㅋㅋㅋ` 하나로
    # 이루어진 감탄사 challenge 가 빈 문자열이 되어 아무 행에나 일치한다.
    # NFKC 가 호환 자모를 결합 자모로 접으므로 표기는 바뀌지만, 코퍼스와
    # holdout 이 같은 변환을 거치므로 비교는 성립한다.
    assert len(comparison_key("ㅋㅋㅋ")) == 3
    assert comparison_key("ㅋㅋㅋ!") == comparison_key("ㅋㅋㅋ")
    assert comparison_key("ㅠㅠ") == comparison_key("ㅠㅠ…")


def test_shingles_ignore_whitespace_and_normalize() -> None:
    assert character_shingles("가나다라마바") == ["가나다라마", "나다라마바"]
    assert character_shingles("가나 다라마바") == character_shingles("가나다라마바")
    assert character_shingles("ＡＢＣＤＥ") == ["ABCDE"]
    assert character_shingles("") == []
    assert character_shingles("가나다") == ["가나다"]


def test_shingle_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="shingle size must be positive"):
        character_shingles("가나다라마", 0)


def test_signature_is_deterministic_and_the_requested_length() -> None:
    text = "지역 도서관은 지난달에 모은 자료를 다시 검토했다."

    first = minhash_signature(text)
    second = minhash_signature(text)

    assert first == second
    assert len(first) == SIGNATURE_LENGTH
    assert len(minhash_signature(text, num_perm=3)) == 3


def test_signature_permutations_are_independent() -> None:
    signature = minhash_signature("지역 도서관은 지난달에 모은 자료를 다시 검토했다.", num_perm=8)

    # Independent hash families must not collapse onto one value.
    assert len(set(signature)) > 1


def test_empty_text_yields_a_zero_signature() -> None:
    assert minhash_signature("   ") == (0,) * SIGNATURE_LENGTH


def test_num_perm_must_be_positive() -> None:
    with pytest.raises(ValueError, match="num_perm must be positive"):
        minhash_signature("가나다라마바", num_perm=0)


def test_whitespace_only_differences_share_a_key() -> None:
    # normalized_split_key already collapses runs of whitespace, so the
    # difference has to be a space that is present in one and absent in the
    # other for the exact keys to diverge.
    spaced = "지역 도서관은 자료를 다시 검토했다."
    unspaced = "지역도서관은 자료를 다시 검토했다."

    assert normalized_split_key(spaced) != normalized_split_key(unspaced)
    assert approximate_split_key(spaced) == approximate_split_key(unspaced)


def test_short_text_falls_back_to_the_exact_key() -> None:
    key = approximate_split_key("가나다")

    assert key.startswith("exact\0")
    assert approximate_split_key("가나다") != approximate_split_key("라마바")


def test_long_text_uses_the_minhash_key() -> None:
    key = approximate_split_key("지역 도서관은 자료를 다시 검토했다.")

    assert key.startswith("minhash\0")
    assert len(key.split("\0")[1]) == SIGNATURE_LENGTH * 16


def test_high_similarity_edits_are_grouped_far_more_often_than_by_exact_keys() -> None:
    """Small edits keep Jaccard high, so the default must group most of them.

    Recall is probabilistic, not guaranteed, so this asserts a rate over many
    variants rather than a single collision.
    """

    base = (
        "지역 도서관은 예산이 처음 계획보다 줄었지만 지난달에 모은 자료를 다시 "
        "검토했다. 이는 결정 과정을 누구나 확인할 수 있게 하기 위해서였다."
    )
    replacements = [
        ("검토했다", "검토하였다"),
        ("이는", "그것은"),
        ("줄었지만", "줄었으나"),
        ("모은", "모아 둔"),
        ("확인할", "살펴볼"),
        ("과정을", "절차를"),
        ("자료를", "자료들을"),
        ("도서관은", "도서관에서는"),
    ]
    variants = [base.replace(old, new) for old, new in replacements]

    exact_hits = sum(normalized_split_key(base) == normalized_split_key(v) for v in variants)
    minhash_hits = sum(approximate_split_key(base) == approximate_split_key(v) for v in variants)

    assert exact_hits == 0
    assert minhash_hits >= len(variants) // 2


def test_unrelated_texts_do_not_share_a_key() -> None:
    first = "지역 도서관은 지난달에 모은 자료를 다시 검토했다."
    second = "이 약은 하루 두 번 식후 삼십 분에 복용하세요."

    assert approximate_split_key(first) != approximate_split_key(second)


_SUBJECTS = ("도서관", "박물관", "기록관", "미술관", "연구소", "학교", "시청", "우체국")
_OBJECTS = ("자료", "목록", "사진", "지도", "장부", "원고", "표본", "도면")
_VERBS = (
    "검토했다",
    "정리했다",
    "공개했다",
    "보관했다",
    "복원했다",
    "분류했다",
    "폐기했다",
    "이관했다",
)
_ADVERBS = (
    "지난달에",
    "이번 주에",
    "작년부터",
    "어제까지",
    "올해 초에",
    "다음 달에",
    "방금",
    "며칠 전에",
)


def varied_corpus(count: int, seed: int = 5) -> list[str]:
    """Sentences built from random words, so Jaccard between any two is low.

    A corpus that shares a frame is the template case, and no synthetic fixture
    reproduces the diversity of the real corpus. Proportion behaviour on real
    data is recorded in ``minhash_signature``; this fixture exercises the
    property a unit test can actually pin down.
    """

    # A 32-word vocabulary is not enough: 12-word sentences drawn from it share
    # most of their 5-grams. The real corpus has tens of thousands of distinct
    # words, so the fixture needs a comparably large one.
    syllables = "가나다라마바사아자차카타파하거너더러머버서어저처커터"
    generator = random.Random(seed)
    vocabulary = sorted(
        {"".join(generator.choice(syllables) for _ in range(3)) for _ in range(4_000)}
    )
    return [" ".join(generator.choices(vocabulary, k=12)) for _ in range(count)]


def test_distinct_sentences_receive_distinct_keys() -> None:
    """The key space must not collapse on a corpus without a shared frame."""

    texts = varied_corpus(4_000)
    unique_texts = len(set(texts))

    keys = {approximate_split_key(text) for text in texts}

    assert unique_texts >= 3_900
    assert len(keys) / unique_texts >= 0.99


def test_split_proportions_hold_on_a_varied_corpus() -> None:
    """Measured on 300,000 real data29 sources the default lands 99.02 / 0.45 /
    0.53 against a requested 99.0 / 0.5 / 0.5. This is the unit-scale guard."""

    counts: Counter[str] = Counter(
        choose_split_for_key(approximate_split_key(text), 0.02, 0.02)
        for text in varied_corpus(20_000)
    )
    total = sum(counts.values())

    assert 0.010 <= counts["validation"] / total <= 0.035
    assert 0.010 <= counts["test"] / total <= 0.035


def test_a_single_frame_corpus_collapses_the_key_space() -> None:
    """A documented limit, not a bug.

    When every row shares one frame, the minimum shingle usually comes from the
    shared part, so the whole file lands on few keys and the split shares stop
    matching the request. That is why template collapse is handled by capping
    frame reuse in scripts/data/resample_generated_shards.py rather than here.
    """

    syllables = "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허"
    texts = [
        f"{syllables[index % 26]}{syllables[(index // 26) % 26]} 마을의 기록관은 자료를 다시 검토했다."
        for index in range(20_000)
    ]

    distinct_exact = len({normalized_split_key(text) for text in texts})
    distinct_minhash = len({approximate_split_key(text) for text in texts})

    assert distinct_exact == 676
    assert distinct_minhash < distinct_exact


def test_more_permutations_group_less(monkeypatch: pytest.MonkeyPatch) -> None:
    """num_perm is the recall knob: collision probability is J ** num_perm."""

    base = (
        "지역 도서관은 예산이 처음 계획보다 줄었지만 지난달에 모은 자료를 다시 "
        "검토했다. 이는 결정 과정을 누구나 확인할 수 있게 하기 위해서였다."
    )
    replacements = ["검토하였다", "검토해 두었다", "검토하기로 했다", "다시 검토했다고 한다"]
    variants = [base.replace("검토했다", value) for value in replacements]

    def grouped(num_perm: int) -> int:
        return sum(
            approximate_split_key(base, num_perm=num_perm)
            == approximate_split_key(v, num_perm=num_perm)
            for v in variants
        )

    assert grouped(1) >= grouped(8)


def test_documented_constants() -> None:
    assert SHINGLE_SIZE == 5
    assert SIGNATURE_LENGTH == 1
