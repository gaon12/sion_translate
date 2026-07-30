from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import sys

import pytest

from sion_translate.scripts_registry import scripts_in


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SPEC = importlib.util.spec_from_file_location(
    "build_hanboneo_test", SCRIPTS_DIR / "build_hanboneo.py"
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)

LEXICON = importlib.import_module("hanboneo_lexicon")


# ---------------------------------------------------------------------------
# Hangul jamo helpers and particle selection
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
        ("소주", "는", "가"),
    ],
)
def test_particles_follow_the_final_consonant(word: str, topic: str, subject: str) -> None:
    assert BUILDER.attach_particle(word, "은/는") == word + topic
    assert BUILDER.attach_particle(word, "이/가") == word + subject


def test_hangulpy_and_the_fallback_agree_on_the_pairs_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional dependency must not change the output of these frames."""

    words = ["인간", "오빠", "고양이", "선생님", "떡볶이", "소주", "마음", "학교", "손", "노래"]
    with_library = {word: BUILDER.attach_particle(word, "은/는") for word in words}

    monkeypatch.setattr(BUILDER, "_hangulpy_josa", None)
    monkeypatch.setattr(BUILDER, "_hangulpy_has_batchim", None)
    without_library = {word: BUILDER.attach_particle(word, "은/는") for word in words}

    assert with_library == without_library


def test_fallback_particle_selection_works_without_hangulpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BUILDER, "_hangulpy_josa", None)
    monkeypatch.setattr(BUILDER, "_hangulpy_has_batchim", None)

    assert BUILDER.attach_particle("오빠", "이/가") == "오빠가"
    assert BUILDER.attach_particle("인간", "이/가") == "인간이"
    assert BUILDER.has_final_consonant("인간") is True
    assert BUILDER.has_final_consonant("오빠") is False
    assert BUILDER.has_final_consonant("") is False


# ---------------------------------------------------------------------------
# Register classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "register"),
    [
        ("체고카요", "hangul_only"),
        ("やばいンデ", "kana_only"),
        ("チンチャそれな", "kana_only"),
        ("오늘 スケジュール 어때", "script"),
        ("마지 スケジュール", "script"),
    ],
)
def test_register_is_derived_from_the_scripts_present(text: str, register: str) -> None:
    assert BUILDER.register_of(text) == register


def test_register_rejects_text_with_neither_writing_system() -> None:
    with pytest.raises(ValueError, match="neither writing system"):
        BUILDER.register_of("12345 !?")


# ---------------------------------------------------------------------------
# Lexicon invariants
# ---------------------------------------------------------------------------


def test_every_verb_declares_its_dictionary_form() -> None:
    verbs = [item for item in LEXICON.PREDICATES if item.form == "verb"]

    assert verbs
    for verb in verbs:
        assert verb.kana_plain and verb.hangul_plain


def test_godan_dictionary_forms_are_not_naive_ru_suffixes() -> None:
    by_stem = {item.kana: item for item in LEXICON.PREDICATES if item.form == "verb"}

    assert by_stem["飲み"].kana_plain == "飲む"
    assert by_stem["聞き"].kana_plain == "聞く"
    assert by_stem["行き"].kana_plain == "行く"
    assert by_stem["食べ"].kana_plain == "食べる"


def test_predicate_rejects_a_verb_without_a_dictionary_form() -> None:
    with pytest.raises(ValueError, match="dictionary form"):
        LEXICON.Predicate("타베", "食べ", "verb", "먹어요", "먹는다", "먹지 않아요", frozenset())


def test_predicate_rejects_an_unknown_form() -> None:
    with pytest.raises(ValueError, match="unknown predicate form"):
        LEXICON.Predicate("x", "x", "adverb", "x", "x", "x", frozenset())


def test_every_noun_class_has_at_least_one_predicate() -> None:
    for noun in LEXICON.NOUNS:
        assert BUILDER._accepting(noun.kind), noun
    for noun in LEXICON.KOREAN_CONTENT_NOUNS:
        assert BUILDER._accepting(noun.kind), noun


def test_every_genitive_modifier_class_can_find_a_head() -> None:
    for table in (LEXICON.NOUNS, LEXICON.KOREAN_CONTENT_NOUNS):
        for modifier_kind, heads in LEXICON.GENITIVE_HEADS.items():
            if not any(noun.kind == modifier_kind for noun in table):
                continue
            usable = [
                noun
                for noun in table
                if noun.kind in heads and BUILDER._accepting(noun.kind, frozenset({"na", "i"}))
            ]
            assert usable, (modifier_kind, table[0].__class__.__name__)


def test_negative_korean_forms_are_polite_like_the_japanese() -> None:
    for predicate in LEXICON.PREDICATES:
        assert predicate.ko_negative.endswith("요"), predicate


def test_hada_ending_is_restricted_to_i_adjectives() -> None:
    """~하다 keeps the bare Japanese stem: 早い is a sentence, 有名 is not."""

    hada = next(item for item in LEXICON.KOREAN_ENDINGS if item.hangul == "하다")

    assert hada.accepts == frozenset({"i"})


def test_blend_sides_are_monolingual_and_the_mixture_is_not() -> None:
    for blend in LEXICON.BLENDS:
        assert scripts_in(blend.ko) <= {"hangul"}, blend
        assert scripts_in(blend.ja) <= {"kana", "han", "latin"}, blend
        assert scripts_in(blend.kj) & {"hangul", "kana", "han"}, blend
        assert blend.note, blend


def test_blends_are_unique() -> None:
    surfaces = [blend.kj for blend in LEXICON.BLENDS]

    assert len(surfaces) == len(set(surfaces))


@pytest.mark.parametrize(
    ("mixed", "korean", "japanese"),
    [
        ("아랏소데스", "알겠습니다", "分かりました"),
        ("키요이", "귀여워", "かわいい"),
        ("마지코마워", "정말 고마워", "まじありがとう"),
        ("체고카요", "최고냐고", "最高かよ"),
        ("친챠소레나", "진짜 그거지", "ほんとそれな"),
        ("테바이", "대박임", "やばい"),
        ("やばいンデ", "굉장한데", "やばいけど"),
        ("부라더 다메요", "형 안 돼", "ブラザーだめよ"),
        ("아나타와 햄스터데스까", "당신은 햄스터입니까", "あなたはハムスターですか"),
    ],
)
def test_the_examples_from_the_brief_are_present(mixed: str, korean: str, japanese: str) -> None:
    by_surface = {blend.kj: blend for blend in LEXICON.BLENDS}

    assert mixed in by_surface, mixed
    assert by_surface[mixed].ko == korean
    assert by_surface[mixed].ja == japanese


# ---------------------------------------------------------------------------
# Row validation
# ---------------------------------------------------------------------------


def _row(**overrides: object) -> BUILDER.Row:
    defaults = {
        "mixed": "닝겐와 츠요이데스네",
        "first": "인간은 강하네요",
        "second": "人間は強いですね",
        "frame": "test",
        "items": ("人間",),
        "mixing": "hangul_only",
    }
    defaults.update(overrides)
    return BUILDER.Row(**defaults)  # type: ignore[arg-type]


def test_validate_accepts_a_well_formed_row() -> None:
    BUILDER.validate_row(_row())


def test_validate_names_the_intruding_script() -> None:
    with pytest.raises(ValueError, match=r"Korean side is not monolingual.*kana"):
        BUILDER.validate_row(_row(first="인간은 強いですね"))
    with pytest.raises(ValueError, match=r"Japanese side is not monolingual.*hangul"):
        BUILDER.validate_row(_row(second="人間は 강하다"))


def test_validate_enforces_each_mixing_register() -> None:
    with pytest.raises(ValueError, match="script mixture"):
        BUILDER.validate_row(_row(mixing="script"))
    with pytest.raises(ValueError, match="all Hangul"):
        BUILDER.validate_row(_row(mixed="人間は強いですね", mixing="hangul_only"))
    with pytest.raises(ValueError, match="all Japanese script"):
        BUILDER.validate_row(_row(mixing="kana_only"))
    with pytest.raises(ValueError, match="unknown mixing mode"):
        BUILDER.validate_row(_row(mixing="whatever"))


def test_validate_rejects_empty_and_control_characters() -> None:
    with pytest.raises(ValueError, match="empty first field"):
        BUILDER.validate_row(_row(first="   "))
    with pytest.raises(ValueError, match="control character"):
        BUILDER.validate_row(_row(first="인간은\x07 강하네요"))


def test_validate_allows_latin_in_the_japanese_side() -> None:
    """Fullwidth ｗ laughter is Latin and is normal in Japanese text."""

    BUILDER.validate_row(_row(second="人間は強いですねｗｗ"))


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

    assert [row.mixed for row in first] == [row.mixed for row in second]


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

    sources = [row.mixed for row in rows]
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
        assert scripts_in(row.first) <= {"hangul"}, row
        assert scripts_in(row.second) <= {"kana", "han", "latin"}, row


def test_build_reports_whether_hangulpy_was_used() -> None:
    _, report = BUILDER.build(max_rows=50, seed=1, max_per_item=50, max_per_frame=50)

    assert report["hangulpy"] is BUILDER.HANGULPY_AVAILABLE


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


def test_json_keys_are_configurable(tmp_path: Path) -> None:
    """The stack is language-generic, so the field names must not be fixed."""

    output = tmp_path / "custom.jsonl"

    assert (
        BUILDER.main(
            [
                "--output",
                str(output),
                "--max-rows",
                "50",
                "--mixed-key",
                "mixed",
                "--first-key",
                "korean",
                "--second-key",
                "japanese",
            ]
        )
        == 0
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    assert all(set(row) == {"mixed", "korean", "japanese", "synthetic"} for row in rows)


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    assert (
        BUILDER.main(
            ["--output", str(tmp_path / "x.jsonl"), "--max-rows", "10", "--first-key", "kj"]
        )
        == 2
    )


def test_main_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    output = tmp_path / "hanboneo.jsonl"

    assert BUILDER.main(["--output", str(output), "--max-rows", "50"]) == 0

    assert not list(tmp_path.glob("*.part"))
