"""Soft glossary hints: format, and the training rows built from them.

The slot mechanism enforces a term by removing it from the source, which
guarantees the surface but leaves the model unable to inflect around it. A soft
hint shows both sides. It only works if the model was trained on the same prefix
the serving path produces, so the format has exactly one definition and this
asserts round-tripping.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from sion_translate.glossary import (
    GLOSSARY_TOKEN,
    PROTECT_TOKEN,
    SEGMENT_TOKEN,
    Glossary,
    adherence,
    build_hinted_source,
    format_hint_prefix,
    parse_hinted_source,
    rank_terms_for_hinting,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_glossary_hints.py"
SPEC = importlib.util.spec_from_file_location("build_glossary_hints_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


# ---------------------------------------------------------------------------
# Hint format
# ---------------------------------------------------------------------------


def test_prefix_uses_only_reserved_control_tokens() -> None:
    from sion_translate.tokenizer import SHARED_CONTROL_SYMBOLS

    for token in (GLOSSARY_TOKEN, PROTECT_TOKEN, SEGMENT_TOKEN):
        assert token in SHARED_CONTROL_SYMBOLS


def test_prefix_round_trips() -> None:
    pairs = [("사과", "Apple"), ("배", "Pear")]

    hinted = build_hinted_source("나는 사과와 배를 먹었다.", pairs)
    parsed, text = parse_hinted_source(hinted)

    assert parsed == pairs
    assert text == "나는 사과와 배를 먹었다."


def test_no_pairs_leaves_the_source_untouched() -> None:
    assert format_hint_prefix([]) == ""
    assert build_hinted_source("원문", []) == "원문"
    assert parse_hinted_source("원문") == ([], "원문")


def test_parse_is_safe_on_ordinary_text() -> None:
    """Serving calls this on arbitrary input, so it must not raise."""

    assert parse_hinted_source("따옴표 <protect> 없는 문장") == ([], "따옴표 <protect> 없는 문장")


def test_terms_containing_control_tokens_are_rejected() -> None:
    with pytest.raises(ValueError, match="control token"):
        format_hint_prefix([(f"사과 {SEGMENT_TOKEN}", "Apple")])
    with pytest.raises(ValueError, match="control token"):
        format_hint_prefix([("사과", f"{GLOSSARY_TOKEN} Apple")])


def test_empty_terms_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        format_hint_prefix([("", "Apple")])
    with pytest.raises(ValueError, match="non-empty"):
        format_hint_prefix([("사과", "   ")])


def test_malformed_hint_entry_is_reported() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_hinted_source(f"{GLOSSARY_TOKEN} 사과 {SEGMENT_TOKEN} 원문")


# ---------------------------------------------------------------------------
# Adherence
# ---------------------------------------------------------------------------


def test_adherence_counts_terms_and_sentences() -> None:
    result = adherence(
        ["リンゴを食べた", "ナシを食べた", "何かを食べた"],
        [["リンゴ"], ["ナシ", "食べ"], ["パエトーン"]],
    )

    assert result["terms"] == 4
    assert result["term_hits"] == 3
    assert result["sentences"] == 3
    assert result["sentence_hits"] == 2


def test_adherence_treats_no_requirement_as_satisfied() -> None:
    result = adherence(["何でもいい"], [[]])

    assert result["sentence_hits"] == 1
    assert result["term_rate"] == 1.0


def test_adherence_can_ignore_case() -> None:
    strict = adherence(["apple pie"], [["Apple"]])
    relaxed = adherence(["apple pie"], [["Apple"]], case_insensitive=True)

    assert strict["term_hits"] == 0
    assert relaxed["term_hits"] == 1


def test_adherence_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="against"):
        adherence(["a"], [["x"], ["y"]])


def test_rare_and_long_terms_rank_first() -> None:
    from collections import Counter

    pairs = [("흔한말", "よくある"), ("파에톤", "パエトーン"), ("보통", "ふつう")]
    counts = Counter({"흔한말": 5000, "보통": 900, "파에톤": 3})

    assert rank_terms_for_hinting(pairs, counts)[0] == ("파에톤", "パエトーン")


def test_ranking_is_deterministic_without_counts() -> None:
    pairs = [("나", "ナ"), ("가", "ガ")]

    assert rank_terms_for_hinting(pairs) == rank_terms_for_hinting(pairs)


# ---------------------------------------------------------------------------
# Term matching
# ---------------------------------------------------------------------------


def test_trie_finds_every_term() -> None:
    trie = BUILDER.TermTrie(["미나", "루미나", "광장"])

    assert trie.find_all("루미나 광장에서") == {"미나", "루미나", "광장"}
    assert trie.find_all("아무것도 없음") == set()


def test_undelimited_matches_are_rejected() -> None:
    """미나 inside 루미나 would hint the model to translate a fragment."""

    assert BUILDER.is_delimited("루미나 광장에서", "루미나") is True
    assert BUILDER.is_delimited("루미나 광장에서", "미나") is False
    assert BUILDER.is_delimited("미나 광장에서", "미나") is True
    assert BUILDER.is_delimited("「미나」가 왔다", "미나") is True
    assert BUILDER.is_delimited("없는말", "있는말") is False
    assert BUILDER.is_delimited("아무말", "") is False


def test_a_script_change_counts_as_a_boundary() -> None:
    """Japanese marks word boundaries by script change, so the rule follows it.

    ルミナ followed by 広場 is katakana against kanji, which is a real boundary in
    Japanese orthography, so it is accepted. A katakana term inside a longer
    katakana run is not.
    """

    assert BUILDER.is_delimited("ルミナ広場", "ルミナ") is True
    assert BUILDER.is_delimited("ルミナ 広場", "ルミナ") is True
    assert BUILDER.is_delimited("ルミナスクエア", "スクエア") is False
    assert BUILDER.is_delimited("ルミナスクエア", "ルミナ") is False
    assert BUILDER.is_delimited("5ミナ", "ミナ") is True


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def write_corpus(path: Path, rows: list[tuple[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for source, target in rows:
            handle.write(json.dumps({"ko": source, "ja": target}, ensure_ascii=False) + "\n")
    return path


GLOSSARY = Glossary(({"ko": "파에톤", "ja": "パエトーン"}, {"ko": "광장", "ja": "スクエア"}))


def test_build_hints_only_rows_where_both_surfaces_occur(tmp_path: Path) -> None:
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        [
            ("파에톤 이 왔다.", "パエトーン が来た。"),  # both present
            ("파에톤 이 왔다고 한다.", "あの人が来たらしい。"),  # target term absent
            ("아무도 오지 않았다.", "誰も来なかった。"),  # source term absent
        ],
    )

    rows, report, _ = BUILDER.build(
        [corpus],
        GLOSSARY,
        source_key="ko",
        target_key="ja",
        rate=1.0,
        max_hints_per_row=3,
        max_per_term=50,
        seed=1,
    )

    assert report.corpus_rows == 3
    assert report.eligible_rows == 1
    assert len(rows) == 1
    parsed, text = parse_hinted_source(str(rows[0]["ko"]))
    assert parsed == [("파에톤", "パエトーン")]
    assert text == "파에톤 이 왔다."
    assert rows[0]["synthetic"] is True
    assert rows[0]["glossary_terms"] == [["파에톤", "パエトーン"]]


def test_build_rejects_undelimited_matches_and_reports_them(tmp_path: Path) -> None:
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        [("루미나광장에서 만났다.", "ルミナスクエアで会った。")] * 4,
    )

    rows, report, _ = BUILDER.build(
        [corpus],
        GLOSSARY,
        source_key="ko",
        target_key="ja",
        rate=1.0,
        max_hints_per_row=3,
        max_per_term=50,
        seed=1,
    )

    assert rows == []
    assert report.rejected_undelimited == 4


def test_build_is_deterministic_and_rate_bounded(tmp_path: Path) -> None:
    rows_in = [
        (f"파에톤 이 {index}번 왔다.", f"パエトーン が{index}回来た。") for index in range(400)
    ]
    corpus = write_corpus(tmp_path / "corpus.jsonl", rows_in)
    kwargs = dict(
        source_key="ko", target_key="ja", max_hints_per_row=3, max_per_term=10_000, seed=4
    )

    first, report, _ = BUILDER.build([corpus], GLOSSARY, rate=0.25, **kwargs)
    second, _, _ = BUILDER.build([corpus], GLOSSARY, rate=0.25, **kwargs)
    everything, _, _ = BUILDER.build([corpus], GLOSSARY, rate=1.0, **kwargs)

    assert [row["ko"] for row in first] == [row["ko"] for row in second]
    assert report.eligible_rows == 400
    assert 0.20 * 400 <= len(first) <= 0.31 * 400
    assert len(everything) == 400


def test_build_respects_the_per_term_cap(tmp_path: Path) -> None:
    rows_in = [
        (f"파에톤 이 {index}번 왔다.", f"パエトーン が{index}回来た。") for index in range(200)
    ]
    corpus = write_corpus(tmp_path / "corpus.jsonl", rows_in)

    rows, report, _ = BUILDER.build(
        [corpus],
        GLOSSARY,
        source_key="ko",
        target_key="ja",
        rate=1.0,
        max_hints_per_row=3,
        max_per_term=7,
        seed=4,
    )

    assert len(rows) == 7
    assert report.most_hinted_terms[0] == ("파에톤", 7)


def test_build_rejects_invalid_arguments(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path / "corpus.jsonl", [("파에톤 이 왔다.", "パエトーン が来た。")])
    kwargs = dict(source_key="ko", target_key="ja", seed=1)

    for bad in ({"rate": 0.0}, {"rate": 1.5}):
        with pytest.raises(ValueError, match="rate must be"):
            BUILDER.build([corpus], GLOSSARY, max_hints_per_row=1, max_per_term=1, **bad, **kwargs)
    with pytest.raises(ValueError, match="max_hints_per_row"):
        BUILDER.build([corpus], GLOSSARY, rate=1.0, max_hints_per_row=0, max_per_term=1, **kwargs)
    with pytest.raises(ValueError, match="max_per_term"):
        BUILDER.build([corpus], GLOSSARY, rate=1.0, max_hints_per_row=1, max_per_term=0, **kwargs)
    with pytest.raises(ValueError, match="no ko->en entries"):
        BUILDER.build(
            [corpus],
            GLOSSARY,
            source_key="ko",
            target_key="en",
            rate=1.0,
            max_hints_per_row=1,
            max_per_term=1,
            seed=1,
        )


def test_glossary_from_corpus_skips_sentence_length_rows(tmp_path: Path) -> None:
    path = write_corpus(
        tmp_path / "terms.jsonl",
        [
            ("파에톤", "パエトーン"),
            ("가", "ガ"),  # too short
            ("이 문장은 용어가 아니라 문장이므로 제외되어야 한다", "これは文なので除外"),
            ("파에톤", "パエトーン2"),  # duplicate source, first wins
        ],
    )

    glossary = BUILDER.glossary_from_corpus(path, "ko", "ja")

    assert glossary.for_direction("ko", "ja") == [("파에톤", "パエトーン")]


def test_cli_writes_rows_and_a_report(tmp_path: Path) -> None:
    corpus = write_corpus(
        tmp_path / "corpus.jsonl",
        [("파에톤 이 왔다.", "パエトーン が来た。")] * 20,
    )
    terms = write_corpus(tmp_path / "terms.jsonl", [("파에톤", "パエトーン")])
    output = tmp_path / "hints.jsonl"
    report = tmp_path / "report.json"

    assert (
        BUILDER.main(
            [
                "--corpus",
                str(corpus),
                "--terms-from-corpus",
                str(terms),
                "--rate",
                "1.0",
                "--output",
                str(output),
                "--report",
                str(report),
            ]
        )
        == 0
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    assert all(SEGMENT_TOKEN in row["ko"] for row in rows)
    assert json.loads(report.read_text(encoding="utf-8"))["rows_out"] == len(rows)
    assert not list(tmp_path.glob("*.part"))


def test_cli_reports_failure_when_nothing_is_hintable(tmp_path: Path) -> None:
    corpus = write_corpus(tmp_path / "corpus.jsonl", [("아무도 없다.", "誰もいない。")])
    terms = write_corpus(tmp_path / "terms.jsonl", [("파에톤", "パエトーン")])

    assert (
        BUILDER.main(
            [
                "--corpus",
                str(corpus),
                "--terms-from-corpus",
                str(terms),
                "--output",
                str(tmp_path / "out.jsonl"),
            ]
        )
        == 2
    )
