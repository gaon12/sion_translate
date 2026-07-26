"""후보 재순위(MBR / QE) 검증."""

from __future__ import annotations

import pytest

from sion_translate.rerank import mbr_scores, qe_components, qe_score, select


def test_qe_penalises_altered_numbers_against_the_source() -> None:
    source = "1회 250mg씩, 48시간 간격으로 복용하세요."
    faithful = "1回250mgずつ、48時間間隔で服用してください。"
    corrupted = "1回1200mgずつ、48時間間隔で服用してください。"

    assert qe_components(source, faithful)["number"] == pytest.approx(1.0)
    assert qe_components(source, corrupted)["number"] < 1.0
    assert qe_score(source, faithful)[0] > qe_score(source, corrupted)[0]


def test_qe_penalises_broken_identifiers() -> None:
    source = "config.json의 retry_limit을 늘려 주세요."
    faithful = "config.jsonのretry_limitを増やしてください。"
    broken = "config.jsonのretry_limyを増やしてください。"

    assert qe_components(source, faithful)["structured"] > (
        qe_components(source, broken)["structured"]
    )


def test_qe_penalises_repetition_collapse_and_source_copy() -> None:
    source = "두 개의 사랑이 진화한다."
    collapsed = "에휴. 에휴. 에휴. 에휴. 에휴. 에휴. 에휴. 에휴."
    healthy = "二つの恋が進化する。"

    assert qe_score(source, collapsed)[0] < qe_score(source, healthy)[0]
    # 원문을 그대로 돌려주는 것은 번역이 아니다.
    assert qe_score(source, source)[0] < qe_score(source, healthy)[0]
    assert qe_score(source, "")[0] == 0.0


def test_qe_length_score_penalises_omission_and_runaway() -> None:
    source = "전철이 늦지 않았다면 제시간에 도착했을 텐데, 역에 도착했을 때는 접수가 끝나 있었다."
    complete = "電車が遅れていなければ間に合ったはずだが、駅に着いた時には受付が終わっていた。"
    truncated = "電車が遅れた。"

    assert qe_components(source, complete)["length"] > (
        qe_components(source, truncated)["length"]
    )


def test_mbr_prefers_the_consensus_candidate() -> None:
    # 세 후보가 서로 비슷하고 하나가 동떨어져 있으면 다수 쪽이 뽑혀야 한다.
    candidates = [
        "会議は3時に始まります。",
        "会議は3時に始まります",
        "会議は三時に始まります。",
        "まったく関係のない文章です。",
    ]
    scores = mbr_scores(candidates)
    assert scores.index(max(scores)) != 3
    assert scores[3] == min(scores)


def test_mbr_with_a_single_candidate_is_neutral() -> None:
    assert mbr_scores(["하나뿐인 후보"]) == [1.0]


def test_select_keeps_the_first_candidate_on_a_tie() -> None:
    # 첫 후보를 beam 결과로 두므로, 동점이면 기존 동작이 유지되어야 한다.
    result = select("같은 문장", ["同じ文", "同じ文"], strategy="mbr+qe")
    assert result.chosen_index == 0


def test_select_recovers_a_faithful_candidate_from_a_corrupted_beam() -> None:
    source = "합계 금액은 38,720엔입니다."
    # 첫 후보(beam)가 금액을 바꿔 썼고, 샘플 후보 하나가 값을 지켰다.
    candidates = ["合計金額は38,000円です。", "合計金額は38,720円です。"]
    result = select(source, candidates, strategy="qe", target_language="ja")
    assert result.chosen_index == 1
    assert result.text == "合計金額は38,720円です。"


def test_select_none_returns_the_first_candidate_untouched() -> None:
    result = select("원문", ["첫 후보", "둘째 후보"], strategy="none")
    assert result.chosen_index == 0
    assert result.text == "첫 후보"


def test_select_rejects_unknown_strategy_and_empty_candidates() -> None:
    with pytest.raises(ValueError, match="알 수 없는 재순위 방식"):
        select("원문", ["후보"], strategy="bogus")
    with pytest.raises(ValueError, match="후보가 비어 있습니다"):
        select("원문", [], strategy="mbr")


def test_select_records_every_candidate_and_score() -> None:
    result = select("원문입니다", ["첫째", "둘째", "셋째"], strategy="mbr+qe")
    assert len(result.candidates) == 3
    assert len(result.scores) == 3
    assert result.text == result.candidates[result.chosen_index]
