"""Audit challenge-sentence leakage into the training corpus.

The original leakage guard worked only **within** the 30 seed pairs. It did not
check whether the 12 challenge sentences already occurred in the source corpus
of roughly 8.97 million rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sion_translate.holdout_audit import (
    HoldoutItem,
    audit_holdout_leakage as audit_with_pairs,
    containment,
    load_holdout_items as load_with_pairs,
    summarize,
)


LANGUAGE_PAIRS = (("ko", "ja"),)


def load_holdout_items(paths):
    return load_with_pairs(paths, language_pairs=LANGUAGE_PAIRS)


def audit_holdout_leakage(items, corpus_paths, **kwargs):
    return audit_with_pairs(
        items,
        corpus_paths,
        language_pairs=LANGUAGE_PAIRS,
        **kwargs,
    )


def _shard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _challenge(path, cases):
    return _shard(path, cases)


def test_both_sides_of_a_challenge_case_are_audited(tmp_path) -> None:
    """A reference-side match is also a leak because the model has generated it."""
    path = _challenge(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "c1",
                "source": "김칫국부터 마시지 마.",
                "reference": "取らぬ狸の皮算用をするな。",
                "source_language": "ko",
                "target_language": "ja",
                "category": "idiom_culture",
            }
        ],
    )
    items = load_holdout_items([path])
    assert {item.identifier for item in items} == {"c1#source", "c1#reference"}
    assert {item.language for item in items} == {"ko", "ja"}


def test_arbitrary_bcp47_pair_and_nested_corpus_are_audited(tmp_path) -> None:
    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "regional",
                "source": "Esta é uma frase de auditoria.",
                "reference": "這是一個稽核句子。",
                "source_language": "PT-br",
                "target_language": "zh-hant",
            }
        ],
    )
    corpus = _shard(
        tmp_path / "nested.jsonl",
        [
            {
                "records": [
                    {
                        "source_language": "pt-BR",
                        "target_language": "zh-Hant",
                        "source": "Esta é uma frase de auditoria.",
                        "target": "這是一個稽核句子。",
                    }
                ]
            }
        ],
    )
    pairs = (("pt-br", "ZH-hant"),)

    items = load_with_pairs([challenge], language_pairs=pairs)
    findings = audit_with_pairs(items, [corpus], language_pairs=pairs)

    assert {item.language for item in items} == {"pt-BR", "zh-Hant"}
    assert len([finding for finding in findings if finding.leaked]) == 2


def test_holdout_rows_outside_the_configured_graph_are_rejected(tmp_path) -> None:
    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "wrong-graph",
                "source": "This case cannot be silently skipped.",
                "source_language": "en",
            }
        ],
    )

    with pytest.raises(ValueError, match="outside the configured language_pairs graph"):
        load_holdout_items([challenge])


def test_holdout_text_without_language_identity_is_rejected(tmp_path) -> None:
    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [{"id": "missing-identity", "source": "언어 표지가 없는 문장입니다."}],
    )

    with pytest.raises(ValueError, match="requires source_language"):
        load_holdout_items([challenge])


def test_an_exact_duplicate_in_the_corpus_is_found(tmp_path) -> None:
    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "c1",
                "source": "세 살 버릇 여든까지 간다",
                "reference": "三つ子の魂百まで",
                "source_language": "ko",
                "target_language": "ja",
            }
        ],
    )
    corpus = _shard(
        tmp_path / "data" / "shard.jsonl",
        [
            {"ko": "전혀 다른 문장입니다", "ja": "全く違う文です"},
            {"ko": "세 살 버릇 여든까지 간다", "ja": "三つ子の魂百まで"},
        ],
    )
    findings = audit_holdout_leakage(load_holdout_items([challenge]), [corpus])
    leaked = [finding for finding in findings if finding.leaked]

    assert len(leaked) == 2  # Both the source and reference are audited.
    assert all(finding.worst.exact for finding in leaked)
    assert all(finding.worst.line == 2 for finding in leaked)


def test_trailing_punctuation_does_not_hide_a_verbatim_leak(tmp_path) -> None:
    """A punctuation-only difference must not hide a measured verbatim leak.

    `김칫국부터 마시지 마.` occurs in full as `김칫국부터 마시지 마…` at
    `data29.jsonl:185527`. The deduplication key preserves punctuation, so the two
    forms were counted as different sentences and the old gate reported zero
    exact leaks.
    """

    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "c1",
                "source": "김칫국부터 마시지 마.",
                "source_language": "ko",
                "target_language": "ja",
            }
        ],
    )
    corpus = _shard(
        tmp_path / "data" / "shard.jsonl",
        [{"ko": "김칫국부터 마시지 마…", "ja": "取らぬ狸の皮算用をするな。"}],
    )
    findings = audit_holdout_leakage(load_holdout_items([challenge]), [corpus])
    leaked = [finding for finding in findings if finding.leaked]

    assert len(leaked) == 1
    assert leaked[0].worst.exact, "Punctuation-only variants must count as exact leaks"


def test_a_near_duplicate_is_found_where_exact_matching_would_miss_it(tmp_path) -> None:
    """Similarity detection catches a row that differs by one Korean particle."""
    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "c1",
                "source": "세 살 버릇 여든까지 간다",
                "source_language": "ko",
                "target_language": "ja",
            }
        ],
    )
    corpus = _shard(
        tmp_path / "data" / "shard.jsonl",
        [{"ko": "세 살 버릇이 여든까지 간다고 하죠", "ja": "三つ子の魂百までと言いますね"}],
    )
    findings = audit_holdout_leakage(load_holdout_items([challenge]), [corpus])
    leaked = [finding for finding in findings if finding.leaked]

    assert len(leaked) == 1
    assert not leaked[0].worst.exact
    assert leaked[0].worst.similarity >= 0.7


def test_an_unrelated_corpus_reports_no_leak(tmp_path) -> None:
    """Unrelated sentences must not create false positives that erode trust."""
    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "c1",
                "source": "김칫국부터 마시지 마.",
                "source_language": "ko",
                "target_language": "ja",
            }
        ],
    )
    corpus = _shard(
        tmp_path / "data" / "shard.jsonl",
        [
            {"ko": "오늘 날씨가 정말 좋습니다", "ja": "今日は本当にいい天気です"},
            {"ko": "회의는 세 시에 시작합니다", "ja": "会議は三時に始まります"},
        ],
    )
    findings = audit_holdout_leakage(load_holdout_items([challenge]), [corpus])
    assert not any(finding.leaked for finding in findings)


def test_a_different_language_field_is_never_compared(tmp_path) -> None:
    """A Korean challenge item must never be compared with a Japanese field."""
    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "c1",
                "source": "가나다라마바사",
                "source_language": "ko",
                "target_language": "ja",
            }
        ],
    )
    corpus = _shard(tmp_path / "data" / "shard.jsonl", [{"ja": "가나다라마바사"}])
    findings = audit_holdout_leakage(load_holdout_items([challenge]), [corpus])
    assert not any(finding.leaked for finding in findings)


def test_matches_are_capped_per_item(tmp_path) -> None:
    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "c1",
                "source": "같은 문장이 여러 번 나옵니다",
                "source_language": "ko",
                "target_language": "ja",
            }
        ],
    )
    corpus = _shard(
        tmp_path / "data" / "shard.jsonl",
        [{"ko": "같은 문장이 여러 번 나옵니다", "ja": "同じ"} for _ in range(20)],
    )
    findings = audit_holdout_leakage(
        load_holdout_items([challenge]), [corpus], maximum_matches_per_item=3
    )
    assert max(len(finding.matches) for finding in findings) == 3


def test_the_summary_reports_the_leak_rate_and_a_warning(tmp_path) -> None:
    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [
            {
                "id": "c1",
                "source": "누출되는 문장입니다",
                "source_language": "ko",
                "target_language": "ja",
                "category": "idiom_culture",
            },
            {
                "id": "c2",
                "source": "완전히 무관한 다른 표현",
                "source_language": "ko",
                "target_language": "ja",
                "category": "profanity",
            },
        ],
    )
    corpus = _shard(tmp_path / "data" / "shard.jsonl", [{"ko": "누출되는 문장입니다", "ja": "x"}])
    summary = summarize(audit_holdout_leakage(load_holdout_items([challenge]), [corpus]))

    assert summary["leaked_items"] == 1
    assert summary["exact_leaked_items"] == 1
    assert summary["by_category"] == {"idiom_culture": 1}
    # The report must state that a leaked set cannot be used as a quality benchmark.
    assert "benchmark" in summary["note"]


def test_containment_answers_is_the_holdout_inside_the_corpus_line() -> None:
    """Containment answers whether the holdout appears inside a corpus line.

    A typical leak embeds an idiom in a longer sentence. Its score must not fall
    merely because the corpus sentence contains additional text.
    """
    idiom = "김칫국부터 마시지 마"
    assert containment(idiom, idiom) == 1.0
    assert containment(idiom, "야 그러니까 김칫국부터 마시지 마 진짜") > 0.8
    assert containment(idiom, "오늘 날씨가 정말 좋습니다") == 0.0
    # The score is asymmetric because it asks whether the short side is in the long side.
    assert containment("야 그러니까 김칫국부터 마시지 마 진짜", idiom) < 0.8


def test_an_empty_holdout_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="challenge 문장이 없습니다"):
        audit_holdout_leakage([], [])


def test_empty_corpus_and_nonpositive_match_cap_are_refused() -> None:
    item = HoldoutItem("x", "ko", "가나다")
    with pytest.raises(ValueError, match="학습 코퍼스가 없습니다"):
        audit_holdout_leakage([item], [])
    with pytest.raises(ValueError, match="maximum_matches_per_item"):
        audit_holdout_leakage([item], [Path("unused.jsonl")], maximum_matches_per_item=0)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_a_threshold_outside_the_unit_interval_is_refused(threshold) -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        audit_holdout_leakage(
            [HoldoutItem("x", "ko", "가나다")], [], similarity_threshold=threshold
        )


def test_the_match_cap_keeps_the_worst_leaks_not_the_first_ones(tmp_path) -> None:
    """The match cap keeps the most severe leaks instead of the first matches.

    A measured case for `호랑이도 제 말 하면 온다더니.` scored 0.91 in data12
    and 1.00 in data9. Lexical file ordering visited data12 first, so a first-N
    cap discarded the exact match. A safety gate must not under-report leakage.
    """
    idiom = "호랑이도 제 말 하면 온다더니"
    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [{"id": "c1", "source": idiom, "source_language": "ko", "target_language": "ja"}],
    )
    # The first file contains only **partial** matches. A line containing the
    # complete idiom has containment 1.0 regardless of its length; that intended
    # asymmetry cannot distinguish "first N" from "worst N" in this regression.
    weak = _shard(
        tmp_path / "data" / "a_first.jsonl",
        [{"ko": f"호랑이도 제 말 하면 좋겠다는 생각 {index}", "ja": "x"} for index in range(6)],
    )
    strong = _shard(tmp_path / "data" / "z_last.jsonl", [{"ko": idiom, "ja": "x"}])

    findings = audit_holdout_leakage(
        load_holdout_items([challenge]), [weak, strong], maximum_matches_per_item=3
    )
    leaked = [finding for finding in findings if finding.leaked][0]

    assert len(leaked.matches) == 3
    assert leaked.worst.similarity == 1.0
    assert "z_last" in leaked.worst.file
    # Severity ordering lets a reviewer process the most important rows first.
    assert leaked.matches == sorted(leaked.matches, key=lambda m: -m.similarity)
