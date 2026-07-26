"""후보 번역을 생성한 뒤 하나를 고르는 재순위 로직.

모델을 재학습하지 않고 추론 계산량만 늘려 품질을 올릴 수 있는지 확인하기 위한
장치입니다. 두 가지 선택 기준을 제공하고, 둘을 합해 쓸 수도 있습니다.

**MBR (Minimum Bayes Risk)**
    후보 하나를 나머지 후보 전체와 비교해 평균 유사도가 가장 높은 것을 고릅니다.
    "다수의 후보가 동의하는 번역이 맞을 가능성이 높다"는 가정이며, 정답이
    필요하지 않습니다. 모델이 자신 있게 틀린 경우(모든 후보가 같은 오역)에는
    도움이 되지 않습니다.

**QE (Quality Estimation, 참조 없음)**
    정답 대신 **원문**과 대조합니다. 숫자·URL·식별자는 번역 후에도 보존되어야
    하므로, 원문의 값과 다른 값을 쓴 후보를 감점할 수 있습니다. 사후학습 보상과
    같은 신호를 쓰지만 참조가 필요한 항목(chrF, token F1)은 제외합니다.

두 기준의 성격이 다릅니다. MBR 은 유창성·일관성 쪽으로, QE 는 보존성 쪽으로
후보를 밀어냅니다. 기본값 ``mbr+qe`` 는 둘을 평균합니다.

**측정 결과 (홀드아웃 40문장, 배포된 export).** 유효 성분은 MBR 이고, QE 단독은
쓰지 마십시오.

    설정                    ko-ja chrF   ja-ko chrF   숫자 일치
    beam4 (기준)                 59.81        49.87   19/20, 15/20
    mbr,    n=7,  T=0.7          60.39        48.82   18/20, 16/20
    qe,     n=7,  T=0.7          54.33        46.56   19/20, 16/20
    mbr+qe, n=7,  T=0.7          60.51        49.29   19/20, 16/20
    mbr+qe, n=7,  T=0.3          60.53        50.36   19/20, 16/20
    mbr+qe, n=15, T=0.5          58.69        52.06   19/20, 16/20

QE 단독은 ko-ja 에서 5.5점을 잃습니다. 참조 없는 신호만으로는 "유창한 쓰레기"를
걸러내지 못하고, 숫자가 우연히 맞은 엉터리 후보가 멀쩡한 후보보다 높게 나옵니다.
MBR 의 합의 조건이 그 이상치를 눌러 주기 때문에 합쳐야 쓸 수 있습니다.

이득은 작고 설정에 민감합니다 (T=0.3 에서 +0.7 / +0.5, 계산량 +28%). 방향당
20문장이므로 이 차이는 잡음과 구분되지 않습니다 — 채택 전에 더 큰 셋으로
확인하십시오.

**숫자 오역은 이 방법으로 고쳐지지 않습니다.** 숫자 일치가 15/20 에서 16/20 으로
한 문장 움직였을 뿐입니다. 토크나이저가 숫자를 자릿수로 보지 못하면 후보 전체가
같은 방식으로 자릿수를 잘못 읽으므로, 고를 만한 올바른 후보가 애초에 생기지
않습니다. 그쪽은 토크나이저 재학습으로만 해결됩니다.
"""

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

# QE 항목별 가중치. 보존성 신호(숫자/구조 문자열)를 가장 크게 둡니다 — 이 프로젝트
# 모델의 관측된 최대 결함이고, 참조 없이도 원문만으로 확실하게 판정할 수 있는
# 항목이기 때문입니다.
DEFAULT_QE_WEIGHTS: dict[str, float] = {
    "number": 0.40,
    "structured": 0.20,
    "language": 0.20,
    "length": 0.20,
}

# 생성 붕괴와 원문 복사는 가중 평균이 아니라 직접 감점합니다.
REPETITION_PENALTY = 0.50
COPY_PENALTY = 0.50


@dataclass
class RerankResult:
    """한 문장의 재순위 결과. 어느 후보가 왜 뽑혔는지 추적할 수 있습니다."""

    text: str
    chosen_index: int
    candidates: list[str]
    scores: list[float]
    components: list[dict[str, float]] = field(default_factory=list)


def _length_score(source: str, hypothesis: str) -> float:
    """원문 대비 길이비. 누락과 폭주를 모두 1에서 멀어지게 만듭니다."""
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
    """참조 없이 원문만으로 계산한 QE 세부 항목 (각 0~1)."""
    letters = sum(char.isalpha() for char in hypothesis)
    return {
        # 원문의 숫자가 번역문에 그대로 남아야 합니다. 값이 바뀌면 여기서 떨어집니다.
        "number": multiset_f1(numeric_tokens(source), numeric_tokens(hypothesis)),
        "structured": multiset_f1(structured_tokens(source), structured_tokens(hypothesis)),
        # 짧은 문장에서는 문자 종류만으로 언어를 판정하기 어려워 건너뜁니다.
        "language": (
            language_fraction(hypothesis, target_language)
            if target_language is not None and letters >= 4
            else 1.0
        ),
        "length": _length_score(source, hypothesis),
    }


def qe_score(
    source: str,
    hypothesis: str,
    *,
    target_language: str | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """(QE 점수, 세부 항목). 점수는 0~1 로 자릅니다."""
    weights = weights or DEFAULT_QE_WEIGHTS
    components = qe_components(source, hypothesis, target_language=target_language)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("QE 가중치의 합이 0보다 커야 합니다")
    score = sum(weights.get(name, 0.0) * value for name, value in components.items()) / total
    if not hypothesis.strip():
        return 0.0, components
    if has_excessive_repetition(hypothesis):
        score -= REPETITION_PENALTY
    if canonical_text(source).casefold() == canonical_text(hypothesis).casefold():
        # 원문을 그대로 돌려주는 것은 번역이 아닙니다.
        score -= COPY_PENALTY
    return max(0.0, min(1.0, score)), components


def mbr_scores(candidates: Sequence[str]) -> list[float]:
    """후보별 기대 유용도 (자신을 제외한 나머지와의 평균 chrF, 0~1).

    후보가 하나면 비교 대상이 없으므로 모두 1.0 입니다.
    """
    if len(candidates) <= 1:
        return [1.0] * len(candidates)
    from sacrebleu.metrics import CHRF

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
    """후보 중 하나를 고릅니다. 동점이면 먼저 온 후보를 유지합니다.

    호출자가 첫 번째 후보를 beam 결과로 두면, 동점일 때 beam 결과가 유지되므로
    재순위가 기존 동작보다 나빠지지 않습니다.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"알 수 없는 재순위 방식: {strategy} (가능: {', '.join(STRATEGIES)})")
    if not candidates:
        raise ValueError("후보가 비어 있습니다")

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
