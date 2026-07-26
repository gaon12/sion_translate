"""문장 이어붙이기 증강 검증."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sion_translate.concat import (
    build_concatenations,
    read_pairs,
    write_concatenations,
)
from sion_translate.data.prepare import DEFAULT_TRAIN_ONLY_PREFIXES


def _pairs(count: int = 40) -> list[tuple[str, str]]:
    """공백 없는 문장. 이어붙인 결과를 공백으로 다시 쪼개 검증할 수 있게 한다."""
    return [
        (f"한국어문장{index}번입니다.", f"日本語の文{index}番です。") for index in range(count)
    ]


def test_concatenation_keeps_both_sides_aligned_and_ordered() -> None:
    pairs = _pairs()
    examples, stats = build_concatenations(
        pairs, count=50, min_sentences=2, max_sentences=4, seed=1
    )
    assert len(examples) == 50
    assert stats.written == 50

    index_by_source = {source: position for position, (source, _) in enumerate(pairs)}
    for joined_source, joined_target in examples:
        source_parts = joined_source.split(" ")
        target_parts = joined_target.split(" ")
        # 양쪽 문장 수가 같아야 "빠뜨리지 않고 번역"이 정답이 된다.
        assert len(source_parts) == len(target_parts)
        # 양쪽 순서가 같아야 정렬 유지가 정답이 된다.
        chosen = [index_by_source[part] for part in source_parts]
        assert target_parts == [pairs[position][1] for position in chosen]


def test_concatenation_never_reuses_a_pair_within_one_example() -> None:
    examples, _ = build_concatenations(_pairs(), count=100, seed=7)
    for joined_source, _ in examples:
        parts = joined_source.split(" ")
        assert len(parts) == len(set(parts))


def test_sentence_count_histogram_respects_bounds() -> None:
    _, stats = build_concatenations(
        _pairs(), count=200, min_sentences=3, max_sentences=5, seed=3
    )
    assert set(stats.sentences_per_example) <= {3, 4, 5}
    assert sum(stats.sentences_per_example.values()) == stats.written


def test_seg_separator_marks_boundaries_explicitly() -> None:
    examples, _ = build_concatenations(
        _pairs(), count=10, min_sentences=2, max_sentences=2, separator="seg", seed=5
    )
    for joined_source, joined_target in examples:
        assert joined_source.count("<seg>") == 1
        assert joined_target.count("<seg>") == 1


def test_examples_over_the_length_budget_are_dropped() -> None:
    long_pairs = [(f"{'가' * 300}{index}", f"{'あ' * 300}{index}") for index in range(10)]
    examples, stats = build_concatenations(long_pairs, count=20, max_chars=400, seed=2)
    # 2문장만 이어붙여도 600자를 넘으므로 하나도 만들 수 없다.
    assert examples == []
    assert stats.written == 0
    assert stats.skipped_too_long > 0


def test_token_budget_uses_the_supplied_counter() -> None:
    pairs = _pairs(10)
    calls: list[str] = []

    def count_tokens(text: str) -> int:
        calls.append(text)
        return len(text)  # 글자 수를 토큰 수로 가정

    _, stats = build_concatenations(
        pairs, count=5, max_chars=10_000, max_tokens=5, count_tokens=count_tokens, seed=4
    )
    assert calls, "count_tokens 가 호출되어야 한다"
    assert stats.written == 0  # 토큰 상한 5 를 넘으므로 전부 버려진다


def test_invalid_arguments_are_rejected() -> None:
    pairs = _pairs(10)
    with pytest.raises(ValueError, match="min_sentences 는 2 이상"):
        build_concatenations(pairs, count=1, min_sentences=1)
    with pytest.raises(ValueError, match="max_sentences 는 min_sentences 이상"):
        build_concatenations(pairs, count=1, min_sentences=4, max_sentences=3)
    with pytest.raises(ValueError, match="separator"):
        build_concatenations(pairs, count=1, separator="pipe")
    with pytest.raises(ValueError, match="쌍이 1개뿐"):
        build_concatenations(pairs[:1], count=1)


def test_same_seed_reproduces_the_same_examples() -> None:
    first, _ = build_concatenations(_pairs(), count=20, seed=11)
    second, _ = build_concatenations(_pairs(), count=20, seed=11)
    third, _ = build_concatenations(_pairs(), count=20, seed=12)
    assert first == second
    assert first != third


def test_read_pairs_skips_malformed_and_incomplete_rows(tmp_path: Path) -> None:
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"ko": "정상", "ja": "正常"}, ensure_ascii=False),
                "{ this is not json",
                json.dumps({"ko": "", "ja": "빈 원문"}, ensure_ascii=False),
                json.dumps({"ko": "번역 없음"}, ensure_ascii=False),
                json.dumps({"ko": 123, "ja": "문자열 아님"}, ensure_ascii=False),
                json.dumps({"ko": "  둘째  ", "ja": "  二番目  "}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert list(read_pairs([source])) == [("정상", "正常"), ("둘째", "二番目")]


def test_written_file_round_trips_and_uses_a_train_only_prefix(tmp_path: Path) -> None:
    examples, _ = build_concatenations(_pairs(), count=5, seed=9)
    output = tmp_path / "concat_multi.jsonl"
    assert write_concatenations(output, examples) == 5
    # 기본 train-only 접두어에 걸려야 holdout 으로 새지 않는다.
    assert output.name.startswith(DEFAULT_TRAIN_ONLY_PREFIXES)
    assert list(read_pairs([output])) == examples
