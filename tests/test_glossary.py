"""글로서리(용어 강제) 로직 검증."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sion_translate.glossary import (
    Glossary,
    apply_source_placeholders,
    load_glossary,
    restore_targets,
)
from sion_translate.tokenizer import SLOT_SYMBOLS


def make_glossary() -> Glossary:
    return Glossary(
        (
            {"ko": "인공지능", "ja": "人工知能"},
            {"ko": "심층학습", "ja": "深層学習"},
            {"ko": "인공지능학회", "ja": "人工知能学会"},
        )
    )


def test_directional_pairs_sorted_longest_first() -> None:
    glossary = make_glossary()
    pairs = glossary.for_direction("ko", "ja")
    # 긴 용어("인공지능학회")가 짧은 용어("인공지능")보다 먼저 와야 한다.
    assert pairs[0][0] == "인공지능학회"
    # 역방향도 동작해야 한다.
    reverse = glossary.for_direction("ja", "ko")
    assert ("人工知能", "인공지능") in reverse


def test_placeholder_and_restore_round_trip() -> None:
    glossary = make_glossary()
    text = "오늘 인공지능 강의를 들었다."
    masked, slot_map = apply_source_placeholders(
        text,
        glossary,
        source_language="ko",
        target_language="ja",
        slot_symbols=SLOT_SYMBOLS,
    )
    assert "인공지능" not in masked  # 용어가 slot 으로 치환됨
    assert "<slot_0>" in masked
    assert slot_map["<slot_0>"] == "人工知能"
    # 모델이 slot 을 보존했다고 가정한 출력 → 복원
    model_output = masked.replace("오늘", "今日").replace("강의를 들었다.", "講義を受けた。")
    restored, missing = restore_targets(model_output, slot_map)
    assert "人工知能" in restored
    assert "<slot_0>" not in restored
    assert missing == []


def test_longest_match_wins_no_overlap() -> None:
    glossary = make_glossary()
    text = "인공지능학회 발표"
    masked, slot_map = apply_source_placeholders(
        text,
        glossary,
        source_language="ko",
        target_language="ja",
        slot_symbols=SLOT_SYMBOLS,
    )
    # "인공지능학회" 전체가 하나의 slot 으로 잡혀야 하며, 안쪽 "인공지능"이
    # 별도로 또 치환되면 안 된다 (slot 은 정확히 1개).
    assert len(slot_map) == 1
    assert list(slot_map.values())[0] == "人工知能学会"


def test_repeated_term_shares_one_slot() -> None:
    glossary = Glossary(({"ko": "인공지능", "ja": "人工知能"},))
    text = "인공지능 그리고 또 인공지능"
    masked, slot_map = apply_source_placeholders(
        text,
        glossary,
        source_language="ko",
        target_language="ja",
        slot_symbols=SLOT_SYMBOLS,
    )
    # 같은 용어가 두 번 나와도 slot 은 하나만 쓰고, 두 자리 모두 그 slot 이 된다.
    assert len(slot_map) == 1
    assert masked.count("<slot_0>") == 2


def test_missing_slot_reported_when_model_drops_it() -> None:
    slot_map = {"<slot_0>": "人工知能"}
    # 모델이 slot 을 흘려서 출력에 없는 경우
    restored, missing = restore_targets("今日は良い天気", slot_map)
    assert missing == ["人工知能"]
    assert restored == "今日は良い天気"


def test_word_boundary_for_latin_languages() -> None:
    glossary = Glossary(({"en": "cat", "de": "Katze"},))
    # "category" 의 일부인 "cat" 은 매칭되면 안 된다 (단어 경계 존중).
    masked, slot_map = apply_source_placeholders(
        "the category is broad",
        glossary,
        source_language="en",
        target_language="de",
        slot_symbols=SLOT_SYMBOLS,
    )
    assert slot_map == {}
    # 독립 단어 "cat" 은 매칭된다.
    masked2, slot_map2 = apply_source_placeholders(
        "the cat sleeps",
        glossary,
        source_language="en",
        target_language="de",
        slot_symbols=SLOT_SYMBOLS,
    )
    assert slot_map2 == {"<slot_0>": "Katze"}


def test_slot_budget_is_respected() -> None:
    # 용어가 slot 개수보다 많으면 초과분은 강제하지 않는다.
    entries = tuple(
        {"ko": f"용어{i}", "ja": f"用語{i}"} for i in range(len(SLOT_SYMBOLS) + 5)
    )
    text = " ".join(entry["ko"] for entry in entries)
    _, slot_map = apply_source_placeholders(
        text,
        Glossary(entries),
        source_language="ko",
        target_language="ja",
        slot_symbols=SLOT_SYMBOLS,
    )
    assert len(slot_map) == len(SLOT_SYMBOLS)


def test_load_glossary_from_file(tmp_path: Path) -> None:
    path = tmp_path / "glossary.json"
    path.write_text(
        json.dumps(
            [
                {"ko": "인공지능", "ja": "人工知能"},
                {"ko": "no-target"},  # 표면형이 하나뿐 → 무시됨
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    glossary = load_glossary(path)
    assert len(glossary) == 1
    assert glossary.for_direction("ko", "ja") == [("인공지능", "人工知能")]


def test_load_glossary_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"인공지능": "人工知能"}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="리스트"):
        load_glossary(path)


def test_no_glossary_terms_returns_text_unchanged() -> None:
    glossary = Glossary(({"ko": "인공지능", "ja": "人工知能"},))
    text = "오늘 날씨가 좋다"  # 글로서리 용어 없음
    masked, slot_map = apply_source_placeholders(
        text,
        glossary,
        source_language="ko",
        target_language="ja",
        slot_symbols=SLOT_SYMBOLS,
    )
    assert masked == text
    assert slot_map == {}
