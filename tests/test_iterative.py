"""Verify iterative sequence revision and dynamic stopping."""

from __future__ import annotations

from typing import Sequence

import pytest

from sion_translate.iterative import refine, refine_batch, summarize

GOOD = "合計金額は38,720円です。"
BAD = "合計金額は38,000円です。"
SOURCE = "합계 금액은 38,720엔입니다."


def test_already_good_translation_is_not_revised() -> None:
    calls: list[str] = []

    def revise(source: str, draft: str) -> str:
        calls.append(draft)
        return draft

    result = refine(SOURCE, GOOD, revise, target_language="ja", accept_score=0.9)
    assert calls == [], "a sentence above the threshold must not be revised"
    assert result.stop_reason == "accept_score"
    assert result.revisions_used == 0
    assert result.text == GOOD


def test_bad_translation_is_revised_until_accepted() -> None:
    def revise(source: str, draft: str) -> str:
        return GOOD  # Correct the draft in one pass.

    result = refine(SOURCE, BAD, revise, target_language="ja", accept_score=0.9)
    assert result.revisions_used == 1
    assert result.stop_reason == "accept_score"
    assert result.text == GOOD


def test_a_worse_revision_is_never_returned() -> None:
    def revise(source: str, draft: str) -> str:
        return "まったく無関係で数字のない文。"

    result = refine(SOURCE, BAD, revise, target_language="ja", accept_score=0.99)
    # Retain the initial draft because it scored higher than the revision.
    assert result.text == BAD
    assert len(result.rounds) > 1


def test_stalled_improvement_stops_early() -> None:
    calls = {"count": 0}

    def revise(source: str, draft: str) -> str:
        calls["count"] += 1
        return draft  # No change means zero gain.

    result = refine(
        SOURCE,
        BAD,
        revise,
        target_language="ja",
        accept_score=0.99,
        min_gain=0.01,
        max_rounds=5,
    )
    assert result.stop_reason == "min_gain"
    assert calls["count"] == 1, "stop remaining rounds after progress stalls"


def test_max_rounds_caps_the_work() -> None:
    """Stop at the hard bound even while the score continues to improve.

    The draft never restores the source number, so it cannot reach
    ``accept_score``. Its length approaches the source and the score keeps
    improving enough to avoid ``min_gain``; only ``max_rounds`` can stop it.
    """
    calls = {"count": 0}
    texts = ["合計", "合計金額", "合計金額は38,000円"]

    def revise(source: str, draft: str) -> str:
        text = texts[min(calls["count"], len(texts) - 1)]
        calls["count"] += 1
        return text

    result = refine(
        SOURCE,
        "合",
        revise,
        target_language="ja",
        accept_score=1.0,
        min_gain=0.0,
        max_rounds=2,
    )
    assert calls["count"] == 2
    assert result.stop_reason == "max_rounds"


def test_zero_max_rounds_is_translation_only() -> None:
    def revise(source: str, draft: str) -> str:
        raise AssertionError("revision must not run when max_rounds=0")

    result = refine(SOURCE, BAD, revise, accept_score=0.99, max_rounds=0)
    assert result.text == BAD
    assert result.revisions_used == 0


def test_invalid_arguments_are_rejected() -> None:
    def revise(source: str, draft: str) -> str:
        return draft

    with pytest.raises(ValueError, match="max_rounds"):
        refine(SOURCE, BAD, revise, max_rounds=-1)
    with pytest.raises(ValueError, match="accept_score"):
        refine(SOURCE, BAD, revise, accept_score=1.5)
    with pytest.raises(ValueError, match="min_gain"):
        refine(SOURCE, BAD, revise, min_gain=-0.1)


def test_batch_only_revises_the_sentences_that_need_it() -> None:
    sources = [SOURCE, "안녕하세요.", SOURCE]
    initials = [BAD, "こんにちは。", BAD]
    seen_batches: list[list[str]] = []

    def revise_batch(batch_sources: Sequence[str], drafts: Sequence[str]) -> list[str]:
        seen_batches.append(list(drafts))
        return [GOOD] * len(drafts)

    results = refine_batch(sources, initials, revise_batch, target_language="ja", accept_score=0.9)
    # The already acceptable second sentence must not enter the revision batch.
    assert seen_batches == [[BAD, BAD]]
    assert results[1].revisions_used == 0
    assert results[0].text == GOOD
    assert results[2].text == GOOD


def test_batch_stops_when_nothing_is_pending() -> None:
    calls = {"count": 0}

    def revise_batch(sources: Sequence[str], drafts: Sequence[str]) -> list[str]:
        calls["count"] += 1
        return [GOOD] * len(drafts)

    refine_batch(
        [SOURCE],
        [BAD],
        revise_batch,
        target_language="ja",
        accept_score=0.9,
        max_rounds=5,
    )
    assert calls["count"] == 1, "stop remaining rounds after reaching the threshold"


def test_batch_rejects_mismatched_lengths() -> None:
    def revise_batch(sources: Sequence[str], drafts: Sequence[str]) -> list[str]:
        return []

    with pytest.raises(ValueError, match="does not match"):
        refine_batch([SOURCE], [BAD, GOOD], revise_batch)


def test_batch_rejects_a_reviser_that_returns_the_wrong_count() -> None:
    def revise_batch(sources: Sequence[str], drafts: Sequence[str]) -> list[str]:
        return []  # Return fewer revisions than requested.

    with pytest.raises(ValueError, match="revision returned"):
        refine_batch([SOURCE], [BAD], revise_batch, accept_score=0.99)


def test_summary_reports_the_saved_work() -> None:
    sources = [SOURCE, "안녕하세요."]
    initials = [BAD, "こんにちは。"]

    def revise_batch(batch_sources: Sequence[str], drafts: Sequence[str]) -> list[str]:
        return [GOOD] * len(drafts)

    results = refine_batch(sources, initials, revise_batch, target_language="ja", accept_score=0.9)
    summary = summarize(results)
    assert summary["sentences"] == 2
    assert summary["unrevised_sentences"] == 1
    assert summary["revisions_total"] == 1
    assert summary["stop_reasons"]["accept_score"] == 2
    assert summarize([])["sentences"] == 0
