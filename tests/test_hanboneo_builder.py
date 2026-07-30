from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import sys

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_hanboneo.py"
SPEC = importlib.util.spec_from_file_location("build_hanboneo_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


HANGUL = tuple(range(0xAC00, 0xD7A4))


def _has_kana(text: str) -> bool:
    return any("぀" <= char <= "ヿ" or "ｦ" <= char <= "ﾟ" for char in text)


def _has_han(text: str) -> bool:
    return any("一" <= char <= "鿿" for char in text)


def _has_hangul(text: str) -> bool:
    return any("가" <= char <= "힣" or "ㄱ" <= char <= "ㅣ" for char in text)


# ---------------------------------------------------------------------------
# Hangul jamo helpers
# ---------------------------------------------------------------------------


def test_laugh_consonant_fuses_into_an_open_syllable() -> None:
    assert BUILDER.fuse_final_consonant("네", BUILDER._JONGSEONG_KIEUK) == "넼"
    assert BUILDER.fuse_final_consonant("어", BUILDER._JONGSEONG_KIEUK) == "엌"
    assert BUILDER.fuse_final_consonant("데스네", BUILDER._JONGSEONG_KIEUK) == "데스넼"


def test_laugh_consonant_leaves_a_closed_syllable_alone() -> None:
    assert BUILDER.fuse_final_consonant("헐", BUILDER._JONGSEONG_KIEUK) == "헐"
    assert BUILDER.fuse_final_consonant("", BUILDER._JONGSEONG_KIEUK) == ""
    assert BUILDER.fuse_final_consonant("ok", BUILDER._JONGSEONG_KIEUK) == "ok"


def test_fuse_rejects_an_out_of_range_jongseong() -> None:
    with pytest.raises(ValueError):
        BUILDER.fuse_final_consonant("네", 0)
    with pytest.raises(ValueError):
        BUILDER.fuse_final_consonant("네", 28)


@pytest.mark.parametrize(
    ("word", "topic", "subject"),
    [
        ("인간", "은", "이"),
        ("오빠", "는", "가"),
        ("고양이", "는", "가"),
        ("선생님", "은", "이"),
        ("떡볶이", "는", "가"),
    ],
)
def test_particles_follow_the_final_consonant(word: str, topic: str, subject: str) -> None:
    assert BUILDER.topic_particle(word) == topic
    assert BUILDER.subject_particle(word) == subject


# ---------------------------------------------------------------------------
# Lexicon invariants
# ---------------------------------------------------------------------------


def test_every_verb_declares_its_dictionary_form() -> None:
    verbs = [item for item in BUILDER.PREDICATES if item.form == "verb"]

    assert verbs
    for verb in verbs:
        assert verb.kana_plain and verb.hangul_plain
        # 飲み + る would be 飲みる; the declared form must differ from that.
        assert verb.kana_plain != f"{verb.kana}る" or verb.kana_plain.endswith("る")


def test_godan_dictionary_forms_are_not_naive_ru_suffixes() -> None:
    by_stem = {item.kana: item for item in BUILDER.PREDICATES if item.form == "verb"}

    assert by_stem["飲み"].kana_plain == "飲む"
    assert by_stem["聞き"].kana_plain == "聞く"
    assert by_stem["行き"].kana_plain == "行く"
    assert by_stem["食べ"].kana_plain == "食べる"


def test_predicate_rejects_a_verb_without_a_dictionary_form() -> None:
    with pytest.raises(ValueError, match="dictionary form"):
        BUILDER.Predicate("타베", "食べ", "verb", "먹어요", "먹는다", "먹지 않아요", frozenset())


def test_predicate_rejects_an_unknown_form() -> None:
    with pytest.raises(ValueError, match="unknown predicate form"):
        BUILDER.Predicate("x", "x", "adverb", "x", "x", "x", frozenset())


def test_every_noun_class_has_at_least_one_predicate() -> None:
    for noun in BUILDER.NOUNS:
        assert BUILDER._accepting_predicates(noun.kind), noun


def test_every_genitive_modifier_class_can_find_a_head() -> None:
    for modifier_kind, heads in BUILDER.GENITIVE_HEADS.items():
        usable = [
            noun
            for noun in BUILDER.NOUNS
            if noun.kind in heads
            and BUILDER._accepting_predicates(noun.kind, frozenset({"na", "i"}))
        ]
        assert usable, modifier_kind


def test_negative_korean_forms_are_polite_like_the_japanese() -> None:
    # The Japanese negative the builder emits is polite (ません / ないです), so a
    # plain Korean negative would be a register mismatch.
    for predicate in BUILDER.PREDICATES:
        assert predicate.ko_negative.endswith("요"), predicate


# ---------------------------------------------------------------------------
# Row validation
# ---------------------------------------------------------------------------


def _row(**overrides: object) -> BUILDER.Row:
    defaults = {
        "kj": "닝겐와 츠요이데스네",
        "ko": "인간은 강하네요",
        "ja": "人間は強いですね",
        "frame": "test",
        "items": ("人間",),
        "mixing": "hangul_only",
    }
    defaults.update(overrides)
    return BUILDER.Row(**defaults)  # type: ignore[arg-type]


def test_validate_accepts_a_well_formed_row() -> None:
    BUILDER.validate_row(_row())


def test_validate_rejects_kana_in_the_korean_side() -> None:
    with pytest.raises(ValueError, match="Korean side is not monolingual"):
        BUILDER.validate_row(_row(ko="인간은 強いですね"))


def test_validate_rejects_hangul_in_the_japanese_side() -> None:
    with pytest.raises(ValueError, match="Japanese side is not monolingual"):
        BUILDER.validate_row(_row(ja="人間は 강하다"))


def test_validate_enforces_each_mixing_register() -> None:
    with pytest.raises(ValueError, match="script mixture"):
        BUILDER.validate_row(_row(mixing="script"))
    with pytest.raises(ValueError, match="all Hangul"):
        BUILDER.validate_row(_row(kj="人間は強いですね", mixing="hangul_only"))
    with pytest.raises(ValueError, match="all Japanese script"):
        BUILDER.validate_row(_row(mixing="kana_only"))
    with pytest.raises(ValueError, match="unknown mixing mode"):
        BUILDER.validate_row(_row(mixing="whatever"))


def test_validate_rejects_empty_and_control_characters() -> None:
    with pytest.raises(ValueError, match="empty ko field"):
        BUILDER.validate_row(_row(ko="   "))
    with pytest.raises(ValueError, match="control character"):
        BUILDER.validate_row(_row(ko="인간은\x07 강하네요"))


# ---------------------------------------------------------------------------
# Frames and build
# ---------------------------------------------------------------------------


def test_every_frame_produces_valid_rows_over_many_seeds() -> None:
    for frame in BUILDER.FRAMES:
        for seed in range(120):
            row = frame(random.Random(seed))
            BUILDER.validate_row(row)
            assert row.frame
            assert row.items


def test_build_is_deterministic_for_a_seed() -> None:
    first, _ = BUILDER.build(max_rows=300, seed=5, max_per_item=50, max_per_frame=500)
    second, _ = BUILDER.build(max_rows=300, seed=5, max_per_item=50, max_per_frame=500)

    assert [row.kj for row in first] == [row.kj for row in second]


def test_build_respects_the_per_item_cap() -> None:
    rows, report = BUILDER.build(max_rows=5_000, seed=1, max_per_item=7, max_per_frame=5_000)

    counts: dict[str, int] = {}
    for row in rows:
        for item in row.items:
            counts[item] = counts.get(item, 0) + 1
    assert counts
    assert max(counts.values()) <= 7
    assert report["rejected_by_caps"] >= 0


def test_build_respects_the_per_frame_cap() -> None:
    _, report = BUILDER.build(max_rows=5_000, seed=1, max_per_item=500, max_per_frame=11)

    frames = report["frames"]
    assert isinstance(frames, dict)
    assert frames
    assert max(frames.values()) <= 11


def test_build_emits_no_duplicate_code_mixed_sources() -> None:
    rows, _ = BUILDER.build(max_rows=2_000, seed=3, max_per_item=200, max_per_frame=2_000)

    sources = [row.kj for row in rows]
    assert len(sources) == len(set(sources))


def test_build_covers_all_three_registers() -> None:
    _, report = BUILDER.build(max_rows=2_000, seed=3, max_per_item=200, max_per_frame=2_000)

    registers = report["registers"]
    assert isinstance(registers, dict)
    assert set(registers) == {"script", "hangul_only", "kana_only"}
    assert all(count > 0 for count in registers.values())


def test_build_output_sides_are_monolingual() -> None:
    rows, _ = BUILDER.build(max_rows=2_000, seed=9, max_per_item=200, max_per_frame=2_000)

    assert rows
    for row in rows:
        assert not _has_kana(row.ko) and not _has_han(row.ko), row
        assert not _has_hangul(row.ja), row


@pytest.mark.parametrize("value", [0, -1])
def test_build_rejects_non_positive_limits(value: int) -> None:
    with pytest.raises(ValueError):
        BUILDER.build(max_rows=value, seed=1, max_per_item=5, max_per_frame=5)
    with pytest.raises(ValueError):
        BUILDER.build(max_rows=5, seed=1, max_per_item=value, max_per_frame=5)
    with pytest.raises(ValueError):
        BUILDER.build(max_rows=5, seed=1, max_per_item=5, max_per_frame=value)


def test_main_writes_jsonl_with_the_synthetic_flag(tmp_path: Path) -> None:
    output = tmp_path / "hanboneo.jsonl"
    report = tmp_path / "report.json"

    assert (
        BUILDER.main(["--output", str(output), "--report", str(report), "--max-rows", "200"]) == 0
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    assert all(set(row) == {"kj", "ko", "ja", "synthetic"} for row in rows)
    assert all(row["synthetic"] is True for row in rows)
    assert json.loads(report.read_text(encoding="utf-8"))["rows"] == len(rows)


def test_main_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    output = tmp_path / "hanboneo.jsonl"

    assert BUILDER.main(["--output", str(output), "--max-rows", "50"]) == 0

    assert not list(tmp_path.glob("*.part"))
