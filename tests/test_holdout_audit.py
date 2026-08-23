"""challenge 문장의 학습 코퍼스 누출 감사.

기존 누출 방지는 seed 30쌍 **내부**에서만 동작합니다. 그 12개 challenge
문장이 897만 행짜리 원천 코퍼스에 이미 있는지는 아무도 확인하지 않았습니다.
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
    """정답 쪽이 코퍼스에 있어도 누출이다 — 모델이 그 문장을 생성해 본 것이다."""
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

    assert len(leaked) == 2  # 원문과 정답 양쪽
    assert all(finding.worst.exact for finding in leaked)
    assert all(finding.worst.line == 2 for finding in leaked)


def test_trailing_punctuation_does_not_hide_a_verbatim_leak(tmp_path) -> None:
    """실측 사례. 마침표 하나가 완전일치 집계를 0 으로 만들고 있었다.

    `김칫국부터 마시지 마.` 는 `data29.jsonl:185527` 에 `김칫국부터 마시지 마…`
    로 통째로 들어 있습니다. dedup 키는 구두점을 남기므로 둘이 다른 문장으로
    집계됐고, 관문은 "완전일치 누출 0개" 라고 보고했습니다.
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
    assert leaked[0].worst.exact, "구두점만 다른 행은 완전일치로 잡혀야 한다"


def test_a_near_duplicate_is_found_where_exact_matching_would_miss_it(tmp_path) -> None:
    """조사 하나 다른 행은 완전일치로 잡히지 않는다. 그래서 MinHash 를 쓴다."""
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
    """오탐이 많으면 이 관문은 무시당한다."""
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
    """한국어 challenge 를 일본어 행과 비교하면 안 된다."""
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
    # 누출된 집합을 품질 benchmark 로 쓰면 안 된다는 사실이 보고서에 남아야 한다.
    assert "benchmark" in summary["note"]


def test_containment_answers_is_the_holdout_inside_the_corpus_line() -> None:
    """Jaccard 가 아닌 이유: 누출의 전형은 긴 문장 안에 관용구가 들어 있는 것.

    상대 문장이 길다는 이유로 점수가 떨어지면 그 누출을 놓칩니다.
    """
    idiom = "김칫국부터 마시지 마"
    assert containment(idiom, idiom) == 1.0
    assert containment(idiom, "야 그러니까 김칫국부터 마시지 마 진짜") > 0.8
    assert containment(idiom, "오늘 날씨가 정말 좋습니다") == 0.0
    # 비대칭이어야 한다 — 짧은 쪽이 긴 쪽에 들어 있는지를 묻는 것이다.
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
    """스캔 순서대로 앞의 N 개를 남기면 더 심한 누출을 버린다.

    실제로 `호랑이도 제 말 하면 온다더니.` 는 data12 에서 0.91, data9 에서
    1.00 인데 문자열 정렬상 data12 가 먼저라 1.00 이 상한에 걸려 사라졌습니다.
    안전 관문이 누출을 과소보고하는 방향이라 허용할 수 없습니다.
    """
    idiom = "호랑이도 제 말 하면 온다더니"
    challenge = _challenge(
        tmp_path / "cases.jsonl",
        [{"id": "c1", "source": idiom, "source_language": "ko", "target_language": "ja"}],
    )
    # 먼저 스캔되는 파일에는 **부분** 일치만 둡니다. 관용구를 통째로 담으면
    # 문장이 아무리 길어도 containment 는 1.0 입니다 — 비대칭이 설계 의도라
    # 그것으로는 "앞의 N 개"와 "가장 나쁜 N 개"를 구분할 수 없습니다.
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
    # 보고서가 심한 순으로 정렬되어 있어야 검수가 위에서부터 유효하다.
    assert leaked.matches == sorted(leaked.matches, key=lambda m: -m.similarity)
