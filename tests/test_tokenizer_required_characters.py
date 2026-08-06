"""Byte fallback on a content character is a regression, not a graceful degrade.

The shipped tokenizer renders 넼 as three ``<0x..>`` pieces because the 한본어
corpus did not exist when it was trained. Two things caused that to survive: the
corpus was sampled down to 22.2%, and nothing forced a frequent character into
the vocabulary. These tests pin the second fix and keep it language-generic.
"""

from __future__ import annotations

from collections import Counter

import pytest

from sion_translate.tokenizer import required_characters_from_counts


def test_a_frequent_character_is_required() -> None:
    counts = Counter({"넼": 1233, "엌": 1178, "슥": 338})
    required = required_characters_from_counts(counts, min_occurrences=25)
    assert set(required) == {"넼", "엌", "슥"}


def test_an_incidental_character_is_left_to_byte_fallback() -> None:
    # Below the floor a character is genuinely rare, and byte fallback is the
    # right answer - that is what byte fallback exists for.
    counts = Counter({"넼": 1233, "콬": 5, "鸚": 12})
    required = required_characters_from_counts(counts, min_occurrences=25)
    assert required == ["넼"]


def test_whitespace_is_never_required() -> None:
    # SentencePiece handles whitespace through its own meta symbol; reserving it
    # as a required character would waste a slot and confuse the segmenter.
    counts = Counter({" ": 10**6, "\n": 10**6, "　": 10**6, "가": 100})
    assert required_characters_from_counts(counts, min_occurrences=25) == ["가"]


def test_the_result_is_sorted_so_the_build_is_reproducible() -> None:
    counts = Counter({"힣": 100, "가": 100, "あ": 100, "漢": 100, "Z": 100})
    required = required_characters_from_counts(counts, min_occurrences=25)
    assert required == sorted(required)


def test_the_threshold_is_honoured_exactly() -> None:
    counts = Counter({"가": 25, "나": 24})
    required = required_characters_from_counts(counts, min_occurrences=25)
    assert required == ["가"]


def test_nothing_here_names_a_script() -> None:
    # A new language pair must get the same protection without a code change, so
    # the rule is frequency alone.
    counts = Counter({"Ж": 500, "ก": 500, "א": 500, "ñ": 500})
    required = required_characters_from_counts(counts, min_occurrences=25)
    assert set(required) == {"Ж", "ก", "א", "ñ"}


def test_an_empty_corpus_requires_nothing() -> None:
    assert required_characters_from_counts(Counter(), min_occurrences=25) == []


def test_a_non_positive_threshold_is_rejected() -> None:
    # Zero would require every character ever seen, filling the vocabulary with
    # noise. The caller disables the feature at the train_tokenizer level.
    with pytest.raises(ValueError, match="min_occurrences must be positive"):
        required_characters_from_counts(Counter({"가": 1}), min_occurrences=0)


def test_the_vocab_floor_is_reported_before_the_corpus_scan(tmp_path, monkeypatch) -> None:
    """SentencePiece only complains after reading the corpus; say it sooner.

    required_chars plus the control symbols plus 256 byte-fallback pieces have to
    fit inside vocab_size. Hitting that limit after a full pass over a nine
    million row corpus wastes the whole scan, and the SentencePiece message does
    not name the setting to change.
    """

    import json

    from sion_translate import tokenizer as tokenizer_module

    shard = tmp_path / "mini.jsonl"
    with shard.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(50):
            handle.write(
                json.dumps(
                    {"ko": f"문장 {index} 입니다", "ja": f"文 {index} です"}, ensure_ascii=False
                )
                + "\n"
            )

    trained: list[object] = []
    monkeypatch.setattr(
        tokenizer_module.spm.SentencePieceTrainer,
        "train",
        lambda **kwargs: trained.append(kwargs),
    )

    with pytest.raises(ValueError, match="required_character_min_occurrences"):
        tokenizer_module.train_tokenizer(
            [str(shard)],
            tmp_path / "out",
            vocab_size=300,
            required_character_min_occurrences=1,
        )
    assert not trained, "training must not start once the floor is known to fail"


def _tiny_shard(tmp_path):
    import json

    shard = tmp_path / "shard.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for index in range(50):
            handle.write(
                json.dumps(
                    {"ko": f"문장 {index} 입니다", "ja": f"文 {index} です"}, ensure_ascii=False
                )
                + "\n"
            )
    return shard


def test_full_coverage_is_rejected_because_it_disables_the_other_two_mechanisms(
    tmp_path,
    monkeypatch,
) -> None:
    """``character_coverage=1.0`` 은 required_chars 와 byte fallback 을 동시에 무력화한다.

    코퍼스의 모든 문자를 어휘에 넣으므로 (a) required_chars 가 이미 포함된
    부분집합이 되고, (b) byte fallback 256 조각이 발화할 대상을 잃고,
    (c) GPU 시간 앞의 byte fallback 비율 관문이 정의상 통과합니다.
    실측: 8,978,338 레코드 코퍼스에서 distinct 문자 10,760개 중 4,275개가
    25회 미만입니다.
    """
    import sion_translate.tokenizer as tokenizer_module

    trained: list[object] = []
    monkeypatch.setattr(
        tokenizer_module.spm.SentencePieceTrainer,
        "train",
        lambda **kwargs: trained.append(kwargs),
    )

    with pytest.raises(ValueError, match="byte fallback unreachable"):
        tokenizer_module.train_tokenizer(
            [str(_tiny_shard(tmp_path))],
            tmp_path / "out",
            vocab_size=300,
            character_coverage=1.0,
        )
    assert not trained, "training must not start on a self-defeating coverage setting"


def test_full_coverage_is_allowed_when_the_frequency_floor_is_opted_out(
    tmp_path,
    monkeypatch,
) -> None:
    """두 장치 중 하나만 쓰겠다는 명시적 선택은 막지 않는다."""
    import sion_translate.tokenizer as tokenizer_module

    trained: list[dict] = []
    monkeypatch.setattr(
        tokenizer_module.spm.SentencePieceTrainer,
        "train",
        lambda **kwargs: trained.append(kwargs),
    )
    monkeypatch.setattr(tokenizer_module, "write_token_features", lambda *a, **k: None)
    monkeypatch.setattr(tokenizer_module, "write_tokenizer_metadata", lambda *a, **k: None)

    tokenizer_module.train_tokenizer(
        [str(_tiny_shard(tmp_path))],
        tmp_path / "out",
        vocab_size=300,
        character_coverage=1.0,
        required_character_min_occurrences=0,
    )
    assert trained and trained[0]["character_coverage"] == 1.0
    assert trained[0]["required_chars"] == ""


def test_the_default_coverage_leaves_a_tail_for_byte_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    import sion_translate.tokenizer as tokenizer_module

    trained: list[dict] = []
    monkeypatch.setattr(
        tokenizer_module.spm.SentencePieceTrainer,
        "train",
        lambda **kwargs: trained.append(kwargs),
    )
    monkeypatch.setattr(tokenizer_module, "write_token_features", lambda *a, **k: None)
    monkeypatch.setattr(tokenizer_module, "write_tokenizer_metadata", lambda *a, **k: None)

    tokenizer_module.train_tokenizer(
        [str(_tiny_shard(tmp_path))],
        tmp_path / "out",
        vocab_size=600,
    )
    assert trained
    assert trained[0]["character_coverage"] < 1.0
    assert trained[0]["byte_fallback"] is True


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.5])
def test_coverage_outside_the_unit_interval_is_rejected(tmp_path, coverage) -> None:
    import sion_translate.tokenizer as tokenizer_module

    with pytest.raises(ValueError, match="character_coverage"):
        tokenizer_module.train_tokenizer(
            [str(_tiny_shard(tmp_path))],
            tmp_path / "out",
            vocab_size=300,
            character_coverage=coverage,
        )
