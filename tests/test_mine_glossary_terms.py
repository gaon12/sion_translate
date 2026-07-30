"""Dice alone accepts table artifacts, so the mutual-best test does the real work.

A character stat table puts the same Korean label in every row that also holds two
different Japanese labels, so both score dice 1.0 and at most one can be right.
Every rejection pinned here was a wrong pair the miner actually produced.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "mine_glossary_terms.py"
SPEC = importlib.util.spec_from_file_location("mine_glossary_terms_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MINE
SPEC.loader.exec_module(MINE)


# Aligned context pairs. Both sides must vary, or an invariant word on either
# side co-occurs with the term just as reliably as its real partner does and the
# ambiguity filter rejects the lot - which is the filter working, not failing.
CONTEXT = [
    ("광산", "鉱山"),
    ("기지", "基地"),
    ("항구", "港湾"),
    ("연구소", "研究所"),
    ("격납고", "格納庫"),
    ("전초", "前哨"),
    ("본부", "本部"),
    ("공장", "工場"),
]


def rows_around(ko_term: str, ja_term: str) -> list[tuple[str, str]]:
    """Sentences where only ``ko_term``/``ja_term`` is constant on both sides."""

    return [(f"{ko} {ko_term}", f"{ja} {ja_term}") for ko, ja in CONTEXT]


def write_shard(path: Path, rows: list[tuple[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for ko, ja in rows:
            handle.write(json.dumps({"ko": ko, "ja": ja}, ensure_ascii=False) + "\n")
    return path


def read_terms(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(source: Path, output: Path, **overrides: object) -> object:
    arguments: dict[str, object] = {
        "source_key": "ko",
        "target_key": "ja",
        "min_dice": 0.8,
        "min_count": 3,
        "max_length_ratio": 3.0,
        "max_source_chars": 120,
    }
    arguments.update(overrides)
    return MINE.mine([source], output, **arguments)


def test_a_consistently_paired_term_is_mined(tmp_path: Path) -> None:
    rows = rows_around("에테르", "エーテル")
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "terms.json"
    run(source, output)
    mined = {(entry["ko"], entry["ja"]) for entry in read_terms(output)}
    assert ("에테르", "エーテル") in mined


def test_a_particle_does_not_split_a_term(tmp_path: Path) -> None:
    # 에테르를 / 에테르가 / 에테르는 are one term, not three.
    rows = [
        ("에테르를 회수했다", "エーテルを回収した"),
        ("에테르가 부족하다", "エーテルが足りない"),
        ("에테르는 위험하다", "エーテルは危険だ"),
        ("에테르에 노출됐다", "エーテルに晒された"),
    ]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "terms.json"
    run(source, output, min_count=3)
    mined = {entry["ko"] for entry in read_terms(output)}
    assert "에테르" in mined
    assert not any(term.startswith("에테르") and term != "에테르" for term in mined)


def test_strip_particle_leaves_a_two_syllable_stem() -> None:
    assert MINE.strip_particle("에테르를") == "에테르"
    assert MINE.strip_particle("유령선은") == "유령선"
    assert MINE.strip_particle("초구에서") == "초구"
    # Too short to strip: the result would not be a term.
    assert MINE.strip_particle("가는") == "가는"
    assert MINE.strip_particle("리나") == "리나"


def test_a_table_artifact_is_rejected_by_mutual_best_match(tmp_path: Path) -> None:
    # 계획력 appears with both 生理的耐 and 戦場機動 in every stat card, so both
    # score dice 1.0. Only one can be a term, and neither is.
    rows = [
        (f"계획력 {index} 강화력 {index}", f"生理的耐{index} 戦場機動{index}") for index in range(6)
    ]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "terms.json"
    result = run(source, output)
    mined = {entry["ko"] for entry in read_terms(output)}
    # Whatever survives, the same Korean term must not map to two Japanese ones.
    assert len(mined) == len(read_terms(output))
    assert result.rejected_ambiguous + result.rejected_not_mutual > 0


def test_a_one_to_many_mapping_is_dropped_entirely(tmp_path: Path) -> None:
    # 설계국 cannot be both 製造局 and 中央設計.
    rows = []
    for index in range(6):
        rows.append((f"설계국 보고서 {index}", f"製造局 報告書{index}"))
        rows.append((f"설계국 명령서 {index}", f"中央設計 命令書{index}"))
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "terms.json"
    result = run(source, output, min_dice=0.5)
    mined = {entry["ko"] for entry in read_terms(output)}
    assert "설계국" not in mined
    assert result.rejected_ambiguous > 0


def test_a_lopsided_length_ratio_is_rejected(tmp_path: Path) -> None:
    # 간조 is two syllables against four kanji: a ratio of 2.0.
    rows = rows_around("간조", "勘定奉行")
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "terms.json"
    result = run(source, output, max_length_ratio=1.5)
    mined = {entry["ko"] for entry in read_terms(output)}
    assert "간조" not in mined
    assert result.rejected_length_ratio > 0


def test_a_conjugated_form_is_never_a_term() -> None:
    # `보유한다 -> 無敵効果` and `해내면 -> 保全作業` were both mined before this.
    for word in ("보유한다", "해내면", "돌려주리", "확인했다", "가능하고", "필요합니다"):
        assert MINE.is_conjugated(word), word
    for word in ("에테르", "유령선", "증명사진", "한파", "오르골"):
        assert not MINE.is_conjugated(word), word
    # The list is explicit, so nouns that merely look verbal survive it. That is
    # deliberate: a blanket rule on `면` would take 라면 and 냉면 with it.
    assert not MINE.is_conjugated("라면")
    assert not MINE.is_conjugated("냉면")


def test_conjugated_forms_are_excluded_from_candidates() -> None:
    found = MINE.source_terms("무적 효과를 보유한다")
    assert "보유한다" not in found
    assert "무적" in found


def test_stopwords_are_excluded() -> None:
    found = MINE.source_terms("그리고 우리는 에테르를 봤다")
    assert "그리고" not in found
    assert "우리" not in found
    assert "에테르" in found


def test_hiragana_is_not_a_candidate_term() -> None:
    # Hiragana is grammar, so mining it would produce particles as terms.
    found = MINE.target_terms("エーテルをかいしゅうした")
    assert "エーテル" in found
    assert not any("か" <= char <= "ん" for term in found for char in term)


def test_a_rare_coincidence_cannot_score_perfectly(tmp_path: Path) -> None:
    # Two words appearing together once would otherwise reach dice 1.0.
    rows = [("우연 조합", "偶然 組合")] + [
        (f"다른 문장 {index}", f"別の文{index}") for index in range(10)
    ]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "terms.json"
    run(source, output, min_count=3)
    mined = {entry["ko"] for entry in read_terms(output)}
    assert "우연" not in mined


def test_markup_rows_are_skipped(tmp_path: Path) -> None:
    rows = [(f"{{name}} {ko}", f"{{name}} {ja}") for ko, ja in CONTEXT]
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "terms.json"
    result = run(source, output)
    assert result.rows_used == 0
    assert result.terms_written == 0


def test_the_output_is_sorted_and_deterministic(tmp_path: Path) -> None:
    rows = rows_around("에테르", "エーテル")
    source = write_shard(tmp_path / "in.jsonl", rows)
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    run(source, first)
    run(source, second)
    assert read_terms(first) == read_terms(second)
    dice_values = [entry["dice"] for entry in read_terms(first)]
    assert dice_values == sorted(dice_values, reverse=True)


def test_each_entry_carries_its_evidence(tmp_path: Path) -> None:
    rows = rows_around("에테르", "エーテル")
    source = write_shard(tmp_path / "in.jsonl", rows)
    output = tmp_path / "terms.json"
    run(source, output)
    entry = read_terms(output)[0]
    assert 0.0 < float(entry["dice"]) <= 1.0
    assert int(entry["count"]) >= 3
    assert entry["ko"] and entry["ja"]
