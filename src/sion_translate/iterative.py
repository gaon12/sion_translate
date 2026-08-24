"""Iterative sequence-revision loop with evaluator-controlled stopping.

Easy sentences often need one pass. Revising only difficult sentences allows
compute to vary by sentence, which is cheaper than applying a fixed number of
passes everywhere and can be more accurate than always stopping after one:

    translate -> evaluate -> revise if below threshold -> evaluate -> ...

All three stopping conditions are necessary:

``accept_score``
    Stop when the draft is already good enough. Without this guard, revision
    can damage an acceptable translation.

``min_gain``
    Stop when another revision improves the score by less than this amount.
    Remaining passes are wasteful after progress stalls.

``max_rounds``
    Set a hard bound because improvements and regressions can alternate.

The loop retains the best-scoring text seen so far and returns it at the end.
A revision round that lowers the score is recorded but cannot become the final
output.

Evaluation uses reference-free QE from :mod:`sion_translate.rerank`. Because
no gold target exists at inference time, it compares against the source for
number and identifier preservation, target-language use, length, and repeated
generation collapse.
"""

# Iterative-run state is loaded from a JSON manifest.
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from sion_translate.rerank import qe_score


@dataclass
class Round:
    """Record one revision round so the stopping decision remains auditable."""

    index: int
    text: str
    score: float
    accepted: bool


@dataclass
class IterativeResult:
    """Store the iterative revision result for one sentence."""

    text: str
    score: float
    rounds: list[Round] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def revisions_used(self) -> int:
        """Return the number of revision passes, excluding initial translation."""
        return max(0, len(self.rounds) - 1)


def refine(
    source: str,
    initial: str,
    revise: Callable[[str, str], str],
    *,
    target_language: str | None = None,
    accept_score: float = 0.95,
    min_gain: float = 0.01,
    max_rounds: int = 3,
) -> IterativeResult:
    """Revise one sentence until a configured stopping condition is met.

    ``revise(source, draft)`` returns one revised translation. It commonly
    wraps ``Translator.revise`` for a single sentence.
    """
    if max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    if not 0.0 <= accept_score <= 1.0:
        raise ValueError("accept_score must be between 0 and 1")
    if min_gain < 0:
        raise ValueError("min_gain must be non-negative")

    score, _ = qe_score(source, initial, target_language=target_language)
    best_text, best_score = initial, score
    rounds = [Round(0, initial, score, score >= accept_score)]

    if score >= accept_score:
        return IterativeResult(best_text, best_score, rounds, "accept_score")
    if max_rounds == 0:
        return IterativeResult(best_text, best_score, rounds, "max_rounds")

    stop_reason = "max_rounds"
    for index in range(1, max_rounds + 1):
        candidate = revise(source, best_text)
        candidate_score, _ = qe_score(source, candidate, target_language=target_language)
        accepted = candidate_score >= accept_score
        rounds.append(Round(index, candidate, candidate_score, accepted))

        gain = candidate_score - best_score
        if candidate_score > best_score:
            # Rejecting worse rounds prevents the final output from regressing.
            best_text, best_score = candidate, candidate_score
        if accepted:
            stop_reason = "accept_score"
            break
        if gain < min_gain:
            stop_reason = "min_gain"
            break

    return IterativeResult(best_text, best_score, rounds, stop_reason)


def refine_batch(
    sources: Sequence[str],
    initials: Sequence[str],
    revise_batch: Callable[[Sequence[str], Sequence[str]], list[str]],
    *,
    target_language: str | None = None,
    accept_score: float = 0.95,
    min_gain: float = 0.01,
    max_rounds: int = 3,
) -> list[IterativeResult]:
    """Revise a batch while forwarding only unfinished sentences to each round.

    Processing sentences individually loses batching efficiency, while forcing
    every sentence through the same number of passes defeats dynamic stopping.
    Grouping only unfinished sentences at each round preserves both benefits.
    """
    if len(sources) != len(initials):
        raise ValueError(
            f"source count {len(sources)} does not match initial-draft count {len(initials)}"
        )
    if max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")

    results: list[IterativeResult] = []
    for source, initial in zip(sources, initials, strict=True):
        score, _ = qe_score(source, initial, target_language=target_language)
        accepted = score >= accept_score
        results.append(
            IterativeResult(
                initial,
                score,
                [Round(0, initial, score, accepted)],
                "accept_score" if accepted else "max_rounds",
            )
        )

    # Carry only sentences that have not reached the acceptance threshold.
    pending = [
        index for index, result in enumerate(results) if result.stop_reason != "accept_score"
    ]
    for round_index in range(1, max_rounds + 1):
        if not pending:
            break
        revised = revise_batch(
            [sources[index] for index in pending],
            [results[index].text for index in pending],
        )
        if len(revised) != len(pending):
            raise ValueError(
                f"revision returned {len(revised)} results for {len(pending)} requests"
            )
        still_pending: list[int] = []
        for index, candidate in zip(pending, revised, strict=True):
            result = results[index]
            candidate_score, _ = qe_score(
                sources[index], candidate, target_language=target_language
            )
            accepted = candidate_score >= accept_score
            result.rounds.append(Round(round_index, candidate, candidate_score, accepted))
            gain = candidate_score - result.score
            if candidate_score > result.score:
                result.text, result.score = candidate, candidate_score
            if accepted:
                result.stop_reason = "accept_score"
                continue
            if gain < min_gain:
                result.stop_reason = "min_gain"
                continue
            still_pending.append(index)
        pending = still_pending
    return results


def summarize(results: Sequence[IterativeResult]) -> dict[str, object]:
    """Summarize whether dynamic stopping reduced revision work."""
    total = len(results)
    if not total:
        return {"sentences": 0}
    revisions = [result.revisions_used for result in results]
    reasons: dict[str, int] = {}
    for result in results:
        reasons[result.stop_reason] = reasons.get(result.stop_reason, 0) + 1
    return {
        "sentences": total,
        "revisions_total": sum(revisions),
        "revisions_per_sentence": sum(revisions) / total,
        "unrevised_sentences": sum(1 for count in revisions if count == 0),
        "mean_score": sum(result.score for result in results) / total,
        "stop_reasons": dict(sorted(reasons.items())),
    }
