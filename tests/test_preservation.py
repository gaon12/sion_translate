"""Preservation checks target the defects chrF and digit-F1 miss."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sion_translate.cli import check_preservation
from sion_translate.preservation import (
    UNIT_CLASSES,
    check_corpus,
    check_pair,
    format_report,
    iter_texts,
    sign_markers,
    unit_tokens,
)


# ---------------------------------------------------------------------------
# Sign extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("허용 오차는 ±0.05mm", ["±"]),
        ("측정값은 -2.5mg 감소", ["-"]),
        ("증가분은 +2.5mg", ["+"]),
        ("변화는 −2.5mg", ["-"]),  # U+2212 folds onto ASCII
        ("변화는 －2.5mg", ["-"]),  # fullwidth folds too
        ("변화는 ＋2.5mg", ["+"]),
        ("오차 없음", []),
    ],
)
def test_sign_markers_are_extracted_and_folded(text: str, expected: list[str]) -> None:
    assert sign_markers(text) == expected


def test_hyphens_inside_identifiers_are_not_signs() -> None:
    """110-482-937561 has no polarity, and neither does an ISO date."""

    assert sign_markers("계좌번호는 110-482-937561 입니다.") == []
    assert sign_markers("기간은 2026-07-30 까지") == []


def test_plus_minus_is_distinct_from_minus() -> None:
    """Losing ± is a different error from flipping a sign, so they must differ."""

    assert sign_markers("±0.05") != sign_markers("-0.05")


# ---------------------------------------------------------------------------
# Unit extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("250mg씩", ["milligram"]),
        ("62.5kg 기준", ["kilogram"]),
        ("0.0037mg/L 이하", ["milligram_per_litre"]),
        ("35%에서 62.5%로", ["percent", "percent"]),
        ("36.8℃", ["celsius"]),
        ("총액 1,286,400원", ["won"]),
        ("総額1,286,400ウォン", ["won"]),
        ("허용 오차 0.05mm", ["millimetre"]),
    ],
)
def test_units_map_to_canonical_classes(text: str, expected: list[str]) -> None:
    assert unit_tokens(text) == expected


def test_translated_currency_is_not_a_violation() -> None:
    """원 and ウォン are the same unit, so a correct translation must compare equal."""

    result = check_pair("총액은 1,286,400원이다.", "総額は1,286,400ウォンだ。")

    assert result["unit_ok"] is True


def test_grammar_after_a_number_is_not_a_unit() -> None:
    """An earlier version read any CJK run after a digit and flagged copulas."""

    result = check_pair("계좌번호는 110-482-937561 입니다.", "口座番号は110-482-937561です。")

    assert unit_tokens("110-482-937561 입니다.") == []
    assert result["unit_ok"] is True


def test_spelled_out_source_numbers_do_not_create_unit_findings() -> None:
    result = check_pair("이 약은 250mg씩 하루 두 번 복용한다.", "この薬は250mgずつ1日2回服用する。")

    assert result["unit_ok"] is True


def test_a_split_compound_unit_is_caught() -> None:
    """mg/L -> mg + L keeps every digit, so only the unit check sees it."""

    result = check_pair(
        "농도는 0.0037mg/L 이하로 유지하세요.", "濃度は0.0037mg 分の 1 L以下に保って。"
    )

    assert result["number_ok"] is False  # the inserted 1 changes the values
    assert result["unit_ok"] is False
    assert unit_tokens("0.0037mg/L") == ["milligram_per_litre"]


def test_latin_units_inside_longer_words_are_ignored() -> None:
    """A unit letter inside a longer Latin word is not a unit."""

    assert unit_tokens("level comment gram config") == []
    assert unit_tokens("무게는 5 g 이다") == ["gram"]
    assert unit_tokens("5 mb 파일") == ["megabyte"]
    assert unit_tokens("5 mgmt 는 단위가 아니다") == []


def test_a_unit_must_follow_a_value() -> None:
    """Korean grammar reuses currency syllables, so bare surfaces are not units.

    The pilot corpus produced two false unit findings this way: the 원 in 원조
    read as won and the 엔 in 초창기엔 read as yen.
    """

    assert unit_tokens("원조 위저드리에서") == []
    assert unit_tokens("초창기엔 비중이 높았다") == []
    assert unit_tokens("병원에 갔다") == []
    # With a value in front, the same surfaces are units again.
    assert unit_tokens("1,286,400원") == ["won"]
    assert unit_tokens("500엔") == ["yen"]
    assert unit_tokens("무게 5 kg") == ["kilogram"]


def test_unit_table_has_no_conflicting_surfaces() -> None:
    assert UNIT_CLASSES
    # _register raises on a conflict, so reaching here means the table is
    # consistent; assert the invariant explicitly for a reader.
    assert len(set(UNIT_CLASSES)) == len(UNIT_CLASSES)


# ---------------------------------------------------------------------------
# Sentence and corpus level
# ---------------------------------------------------------------------------


def test_sign_loss_is_caught_when_digits_survive() -> None:
    """±0.05mm -> 0.05mm is invisible to chrF and to digit-F1."""

    result = check_pair(
        "허용 오차는 ±0.05mm 이내여야 한다.", "許容誤差は0.05mm以内でなければならない。"
    )

    assert result["sign_ok"] is False
    assert result["number_ok"] is True
    assert result["unit_ok"] is True


def test_script_leakage_is_caught_and_named() -> None:
    result = check_pair("인간은 강하다.", "人間は 강하다.", target_scripts=("ja",))

    assert result["script_ok"] is False
    assert result["foreign_scripts"] == ["hangul"]


def test_script_check_is_skipped_without_target_scripts() -> None:
    result = check_pair("인간은 강하다.", "人間は 강하다.")

    assert result["script_ok"] is True
    assert result["foreign_scripts"] == []


def test_code_mixed_source_is_fine_against_a_monolingual_target() -> None:
    result = check_pair(
        "오늘 スケジュール 어때", "今日のスケジュールはどう", target_scripts=("ja",)
    )

    assert result["script_ok"] is True


def test_corpus_aggregates_and_keeps_examples() -> None:
    sources = [
        "허용 오차는 ±0.05mm 이내여야 한다.",
        "농도는 0.0037mg/L 이하로 유지하세요.",
        "측정값은 -2.5mg 감소했다.",
        "총액은 1,286,400원이다.",
    ]
    hypotheses = [
        "許容誤差は0.05mm以内でなければならない。",
        "濃度は0.0037mg 分の 1 L以下に保って。",
        "測定値は-2.5mg減少した。",
        "総額は1,286,400ウォンだ。",
    ]

    counts = check_corpus(sources, hypotheses, target_scripts=("ja", "latin"), examples=2)

    assert counts.sentences == 4
    assert counts.sign_violations == 1
    assert counts.unit_violations == 1
    assert counts.script_violations == 0
    assert len(counts.examples) == 2
    assert all("failed" in example for example in counts.examples)


def test_corpus_rejects_mismatched_lengths_and_bad_arguments() -> None:
    with pytest.raises(ValueError, match="against"):
        check_corpus(["a"], ["b", "c"])
    with pytest.raises(ValueError, match="examples must be non-negative"):
        check_corpus(["a"], ["b"], examples=-1)
    with pytest.raises(ValueError, match="unknown script or language"):
        check_corpus(["a"], ["b"], target_scripts=("klingon",))


def test_empty_corpus_reports_nothing_rather_than_dividing_by_zero() -> None:
    counts = check_corpus([], [])

    assert counts.sentences == 0
    assert format_report(counts) == "preservation: no sentences"


def test_report_names_every_check() -> None:
    counts = check_corpus(["±1mg"], ["1mg"], target_scripts=("ja",))

    report = format_report(counts, title="probe")

    assert "probe" in report
    for name in ("number", "sign", "unit", "script"):
        assert name in report


def test_iter_texts_picks_the_first_present_key() -> None:
    rows = [{"ko": "가"}, {"source": "나"}]

    assert iter_texts(rows, "source", "ko") == ["가", "나"]
    with pytest.raises(ValueError, match="at least one key"):
        iter_texts(rows)
    with pytest.raises(ValueError, match="missing all of"):
        iter_texts([{"other": "다"}], "source", "ko")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_cli_reports_and_writes_json(tmp_path: Path, capsys) -> None:
    path = write_jsonl(
        tmp_path / "out.jsonl",
        [
            {"source": "허용 오차는 ±0.05mm 이내다.", "translation": "許容誤差は0.05mm以内だ。"},
            {"source": "측정값은 -2.5mg 감소했다.", "translation": "測定値は-2.5mg減少した。"},
        ],
    )
    report = tmp_path / "report.json"

    assert (
        check_preservation.main(["--target-scripts", "ja,latin", "--json", str(report), str(path)])
        == 0
    )

    printed = capsys.readouterr().out
    assert "out.jsonl" in printed
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload[0]["sentences"] == 2
    assert payload[0]["sign_violations"] == 1


def test_cli_fails_when_a_threshold_is_exceeded(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "out.jsonl",
        [{"source": "오차는 ±0.05mm", "translation": "誤差は0.05mm"}],
    )

    assert check_preservation.main(["--max-violation-rate", "0.5", str(path)]) == 1
    assert check_preservation.main(["--max-violation-rate", "1.0", str(path)]) == 0


def test_cli_accepts_custom_keys(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "hanboneo.jsonl",
        [{"kj": "체고카요", "ko": "최고냐고", "ja": "最高かよ"}],
    )

    assert (
        check_preservation.main(
            ["--source-key", "kj", "--target-key", "ko", "--target-scripts", "ko", str(path)]
        )
        == 0
    )


def test_cli_rejects_bad_input(tmp_path: Path) -> None:
    good = write_jsonl(tmp_path / "a.jsonl", [{"source": "가", "translation": "あ"}])
    broken = tmp_path / "b.jsonl"
    broken.write_text("not json\n", encoding="utf-8")
    missing_field = write_jsonl(tmp_path / "c.jsonl", [{"other": "가"}])

    assert check_preservation.main([str(broken)]) == 2
    assert check_preservation.main([str(missing_field)]) == 2
    assert check_preservation.main(["--examples", "-1", str(good)]) == 2
    assert check_preservation.main(["--max-violation-rate", "2", str(good)]) == 2
    assert check_preservation.main([str(tmp_path / "missing.jsonl")]) == 2
