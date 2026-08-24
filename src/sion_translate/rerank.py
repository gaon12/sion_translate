"""Sequence-level reranking after generating multiple complete translations.

This optional inference-only feature tests whether extra decoding compute can
improve quality without retraining. It provides two selection criteria and a
combined mode. It is distinct from the model's trained full-vocabulary
per-token distribution refinement.

**MBR (Minimum Bayes Risk)**
    Compare each candidate with every other candidate and select the highest
    average similarity. This assumes that agreement among candidates is useful
    without requiring a reference. It cannot help when every candidate repeats
    the same confident mistranslation.

**QE (reference-free quality estimation)**
    Compare each candidate with the source instead of a gold target. Numbers,
    URLs, and identifiers should survive translation, so a candidate that
    changes them can be penalized. This uses the reference-free parts of the
    post-training reward and excludes reference-dependent chrF and token F1.

MBR favors consensus and fluency, while QE favors source preservation. The
default ``mbr+qe`` strategy averages them.

The following measurements used 40 holdout sentences and a deployed export.
MBR was the useful component; QE alone should not be used:

    configuration            ko-ja chrF   ja-ko chrF   number exact
    beam4 baseline                59.81        49.87   19/20, 15/20
    mbr,    n=7,  T=0.7          60.39        48.82   18/20, 16/20
    qe,     n=7,  T=0.7          54.33        46.56   19/20, 16/20
    mbr+qe, n=7,  T=0.7          60.51        49.29   19/20, 16/20
    mbr+qe, n=7,  T=0.3          60.53        50.36   19/20, 16/20
    mbr+qe, n=15, T=0.5          58.69        52.06   19/20, 16/20

QE alone lost 5.5 chrF on ko-ja. Reference-free signals cannot reliably reject
fluent nonsense; a poor candidate with coincidentally correct numbers can
outrank a sound one. MBR consensus suppresses that outlier in the combined mode.

The gain was small and configuration-sensitive: about +0.7/+0.5 at T=0.3 for
28% more compute. With only 20 sentences per direction, this difference may be
noise and must be confirmed on a larger evaluation set before adoption.

This method did not solve number mistranslation: exact matches moved only from
15/20 to 16/20. If the tokenizer does not represent digits robustly, every
candidate can misread them the same way and reranking has no correct candidate
to select. That failure requires tokenizer retraining.
"""

# Metric component dictionaries are assembled dynamically.
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sion_translate.data.quality import canonical_text, language_fraction
from sion_translate.evaluation import (
    has_excessive_repetition,
    multiset_f1,
    numeric_tokens,
    structured_tokens,
)

STRATEGIES = ("none", "mbr", "qe", "mbr+qe")

# Preservation signals receive the largest QE weights because number and
# structured-token corruption is the model's largest observed defect and can
# be checked reliably against the source without a reference.
DEFAULT_QE_WEIGHTS: dict[str, float] = {
    "number": 0.40,
    "structured": 0.20,
    "language": 0.20,
    "length": 0.20,
}

# Repetition collapse and source copying are direct penalties, not average components.
REPETITION_PENALTY = 0.50
COPY_PENALTY = 0.50


@dataclass
class RerankResult:
    """Store one reranking decision and the evidence behind it."""

    text: str
    chosen_index: int
    candidates: list[str]
    scores: list[float]
    components: list[dict[str, float]] = field(default_factory=list)


def _length_score(source: str, hypothesis: str) -> float:
    """Score target/source length ratio, penalizing omission and overgeneration."""
    import math

    source_length = sum(not char.isspace() for char in source)
    hypothesis_length = sum(not char.isspace() for char in hypothesis)
    return math.exp(-abs(math.log((hypothesis_length + 1.0) / (source_length + 1.0))))


def qe_components(
    source: str,
    hypothesis: str,
    *,
    target_language: str | None = None,
) -> dict[str, float]:
    """Return reference-free QE components in [0, 1] using only the source."""
    letters = sum(char.isalpha() for char in hypothesis)
    components = {
        # Numbers must remain consistent with the source after translation.
        "number": multiset_f1(numeric_tokens(source), numeric_tokens(hypothesis)),
        "structured": multiset_f1(structured_tokens(source), structured_tokens(hypothesis)),
        "length": _length_score(source, hypothesis),
    }
    # Omit this component for short text or languages without a script profile.
    # Reporting 1.0 would incorrectly give an unchecked result a perfect score.
    language = (
        language_fraction(hypothesis, target_language)
        if target_language is not None and letters >= 4
        else None
    )
    if language is not None:
        components["language"] = language
    return components


def qe_score(
    source: str,
    hypothesis: str,
    *,
    target_language: str | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Return ``(QE score, components)`` with the score clamped to [0, 1]."""
    weights = weights or DEFAULT_QE_WEIGHTS
    components = qe_components(source, hypothesis, target_language=target_language)
    active_weights = {name: weight for name, weight in weights.items() if name in components}
    total = sum(active_weights.values())
    if total <= 0:
        raise ValueError("the sum of active QE weights must be positive")
    score = sum(active_weights[name] * components[name] for name in active_weights) / total
    if not hypothesis.strip():
        return 0.0, components
    if has_excessive_repetition(hypothesis):
        score -= REPETITION_PENALTY
    if canonical_text(source).casefold() == canonical_text(hypothesis).casefold():
        # Returning the source unchanged is not a translation.
        score -= COPY_PENALTY
    return max(0.0, min(1.0, score)), components


def mbr_scores(candidates: Sequence[str]) -> list[float]:
    """Return expected utility per candidate as mean pairwise chrF in [0, 1].

    A singleton candidate receives 1.0 because it has no comparison target.
    """
    if len(candidates) <= 1:
        return [1.0] * len(candidates)
    from sacrebleu.metrics.chrf import CHRF

    chrf = CHRF(word_order=0)
    scores: list[float] = []
    for index, candidate in enumerate(candidates):
        others = [other for position, other in enumerate(candidates) if position != index]
        if not candidate.strip():
            scores.append(0.0)
            continue
        total = sum(chrf.sentence_score(candidate, [other]).score for other in others)
        scores.append(total / len(others) / 100.0)
    return scores


def select(
    source: str,
    candidates: Sequence[str],
    *,
    strategy: str = "mbr+qe",
    target_language: str | None = None,
    qe_weights: dict[str, float] | None = None,
) -> RerankResult:
    """Select one candidate, retaining the earliest candidate on a tie.

    If the caller places the beam result first, a tied reranking decision keeps
    the existing beam behavior.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown reranking strategy: {strategy} (available: {', '.join(STRATEGIES)})"
        )
    if not candidates:
        raise ValueError("candidate list must not be empty")

    pool = list(candidates)
    components: list[dict[str, float]] = []
    if strategy == "none":
        return RerankResult(pool[0], 0, pool, [1.0] + [0.0] * (len(pool) - 1), components)

    quality: list[float] = []
    if strategy in ("qe", "mbr+qe"):
        for candidate in pool:
            score, parts = qe_score(
                source, candidate, target_language=target_language, weights=qe_weights
            )
            quality.append(score)
            components.append(parts)

    consensus = mbr_scores(pool) if strategy in ("mbr", "mbr+qe") else []

    if strategy == "mbr":
        scores = consensus
    elif strategy == "qe":
        scores = quality
    else:
        scores = [(a + b) / 2.0 for a, b in zip(consensus, quality, strict=True)]

    best = max(range(len(pool)), key=lambda index: scores[index])
    return RerankResult(pool[best], best, pool, scores, components)
