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
