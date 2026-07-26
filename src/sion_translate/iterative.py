"""평가기가 멈출 시점을 정하는 반복 수정 루프.

쉬운 문장은 한 번에 맞습니다. 어려운 문장만 여러 번 고치게 하면, 문장마다 계산량을
다르게 쓸 수 있습니다. 모든 문장에 같은 횟수를 쓰는 것보다 싸고, 한 번만 하는 것보다
정확합니다.

    번역 → 평가 → (기준 미달이면) 수정 → 평가 → …

세 가지 종료 조건을 씁니다. 셋 다 필요합니다.

``accept_score``
    이미 충분히 좋으면 더 고치지 않습니다. 이것이 없으면 잘된 문장을 계속 건드려
    오히려 망가뜨립니다.

``min_gain``
    한 번 더 고쳐서 점수가 이만큼도 오르지 않으면 멈춥니다. 개선이 정체되면 남은
    반복은 낭비입니다.

``max_rounds``
    개선과 악화가 번갈아 나타날 수 있으므로 상한을 둡니다.

**점수가 떨어지면 되돌립니다.** 각 라운드에서 지금까지 가장 좋았던 결과를 들고
있다가 마지막에 그것을 돌려줍니다. 수정 모델이 문장을 더 나쁘게 만든 라운드가
최종 출력이 되는 일은 없습니다.

평가는 참조 없는 QE(``sion_translate.rerank``)입니다. 실제 정답이 없으므로 원문과
대조해 숫자·식별자 보존, 목표 언어, 길이, 반복 붕괴를 봅니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from sion_translate.rerank import qe_score


@dataclass
class Round:
    """한 라운드의 기록. 왜 멈췄는지 추적할 수 있습니다."""

    index: int
    text: str
    score: float
    accepted: bool


@dataclass
class IterativeResult:
    """한 문장의 반복 수정 결과."""

    text: str
    score: float
    rounds: list[Round] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def revisions_used(self) -> int:
        """실제로 수행한 수정 횟수 (첫 번역은 세지 않습니다)."""
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
    """한 문장을 조건이 만족될 때까지 고칩니다.

    ``revise(source, draft)`` 는 고친 번역 하나를 돌려주는 호출 가능 객체입니다.
    보통 ``Translator.revise`` 를 한 문장에 대해 감싼 것입니다.
    """
    if max_rounds < 0:
        raise ValueError("max_rounds 는 0 이상이어야 합니다")
    if not 0.0 <= accept_score <= 1.0:
        raise ValueError("accept_score 는 0~1 이어야 합니다")
    if min_gain < 0:
        raise ValueError("min_gain 은 0 이상이어야 합니다")

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
            # 나빠진 라운드는 채택하지 않으므로 최종 출력이 후퇴하지 않습니다.
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
    """여러 문장을 함께 고칩니다. 남은 문장만 다음 라운드로 넘깁니다.

    한 문장씩 처리하면 배치 이득을 잃고, 전부 같은 횟수를 돌리면 동적 종료의
    의미가 없습니다. 라운드마다 아직 끝나지 않은 문장만 모아 한 번에 수정하므로
    둘을 모두 얻습니다.
    """
    if len(sources) != len(initials):
        raise ValueError(f"원문 {len(sources)}개와 초안 {len(initials)}개의 수가 다릅니다")
    if max_rounds < 0:
        raise ValueError("max_rounds 는 0 이상이어야 합니다")

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

    # 아직 기준에 못 미친 문장의 인덱스만 들고 갑니다.
    pending = [
        index
        for index, result in enumerate(results)
        if result.stop_reason != "accept_score"
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
                f"수정 결과 {len(revised)}개가 요청한 {len(pending)}개와 다릅니다"
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
    """동적 종료가 실제로 계산량을 아꼈는지 확인할 요약."""
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
