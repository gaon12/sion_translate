"""Verify hard glossary constraints and target-surface restoration."""

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
    # The longer term must precede the shorter term that it contains.
    assert pairs[0][0] == "인공지능학회"
    # The same entry must work in the reverse configured direction.
    reverse = glossary.for_direction("ja", "ko")
    assert ("人工知能", "인공지능") in reverse


def test_glossary_canonicalizes_bcp47_entry_and_lookup_aliases() -> None:
    glossary = Glossary(({"pt-br": "gato", "EN": "cat"},))

    assert glossary.entries == ({"pt-BR": "gato", "en": "cat"},)
    assert glossary.for_direction("PT-br", "en") == [("gato", "cat")]


def test_glossary_rejects_duplicate_canonical_language_aliases() -> None:
    with pytest.raises(ValueError, match="duplicate language aliases"):
        Glossary(({"pt-br": "gato", "pt-BR": "felino", "en": "cat"},))


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
    assert "인공지능" not in masked  # The protected slot replaces the source term.
    assert "<slot_0>" in masked
    assert slot_map["<slot_0>"] == "人工知能"
    # Restore a simulated model output that preserved the protected slot.
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
    # The full long term must occupy one slot; its nested short term must not
    # receive a second replacement.
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
    # Repeated occurrences share one slot and both positions use it.
    assert len(slot_map) == 1
    assert masked.count("<slot_0>") == 2


def test_missing_slot_reported_when_model_drops_it() -> None:
    slot_map = {"<slot_0>": "人工知能"}
    # Report a required target when the model drops its slot.
    restored, missing = restore_targets("今日は良い天気", slot_map)
    assert missing == ["人工知能"]
    assert restored == "今日は良い天気"


def test_word_boundary_for_latin_languages() -> None:
    glossary = Glossary(({"en": "cat", "de": "Katze"},))
    # Respect word boundaries: "cat" must not match inside "category".
    masked, slot_map = apply_source_placeholders(
        "the category is broad",
        glossary,
        source_language="en",
        target_language="de",
        slot_symbols=SLOT_SYMBOLS,
    )
    assert slot_map == {}
    # A standalone "cat" does match.
    masked2, slot_map2 = apply_source_placeholders(
        "the cat sleeps",
        glossary,
        source_language="en",
        target_language="de",
        slot_symbols=SLOT_SYMBOLS,
    )
    assert slot_map2 == {"<slot_0>": "Katze"}


@pytest.mark.parametrize(
    ("source_language", "text", "term"),
    [
        ("ko-KR", "인공지능학회", "인공지능"),
        ("ja-JP", "人工知能学会", "人工知能"),
        ("zh-Hant", "人工智慧學會", "人工智慧"),
        ("th-TH", "แมวกำลังนอน", "แมว"),
    ],
)
def test_glossary_substring_boundaries_follow_script_profiles(
    source_language: str,
    text: str,
    term: str,
) -> None:
    glossary = Glossary(({source_language: term, "en": "term"},))
    masked, slot_map = apply_source_placeholders(
        text,
        glossary,
        source_language=source_language,
        target_language="en",
        slot_symbols=SLOT_SYMBOLS,
    )

    assert "<slot_0>" in masked
    assert slot_map == {"<slot_0>": "term"}


def test_slot_budget_is_respected() -> None:
    # Terms beyond the slot capacity remain unconstrained.
    entries = tuple({"ko": f"용어{i}", "ja": f"用語{i}"} for i in range(len(SLOT_SYMBOLS) + 5))
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
                {"ko": "no-target"},  # Ignore an entry with only one language surface.
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
    with pytest.raises(ValueError, match="must be a list"):
        load_glossary(path)


def test_no_glossary_terms_returns_text_unchanged() -> None:
    glossary = Glossary(({"ko": "인공지능", "ja": "人工知能"},))
    text = "오늘 날씨가 좋다"  # No glossary term occurs in this sentence.
    masked, slot_map = apply_source_placeholders(
        text,
        glossary,
        source_language="ko",
        target_language="ja",
        slot_symbols=SLOT_SYMBOLS,
    )
    assert masked == text
    assert slot_map == {}
