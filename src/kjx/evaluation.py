"""번역 품질 평가 공용 로직.

고정된 평가셋에 대해 chrF / BLEU 를 계산합니다. `kjx-evaluate` CLI 가
사용하며, 같은 평가셋에 대한 외부 서비스(DeepL/Google/Papago 등) 출력과의
비교도 지원합니다.

지표 선택 이유:
- **chrF** (문자 n-gram F-score): 토큰화가 필요 없어 한국어/일본어처럼
  띄어쓰기 규칙이 다른 언어에서도 공정합니다. 주 지표로 사용합니다.
- **BLEU**: 관례상 함께 보고합니다. 한/일/중은 단어 분리가 모호하므로
  문자 단위(tokenize="char")로, 그 외 언어는 표준 13a 토큰화로 계산합니다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from kjx.data import IndexedParallelDataset
from kjx.tokenizer import KJTokenizer

# 문자 단위 BLEU 를 쓰는 언어 (공백 기반 토큰화가 무의미한 언어)
CHARACTER_LEVEL_LANGUAGES = {"ko", "ja", "zh"}


@dataclass
class DirectionResult:
    """한 방향(예: ko→ja)에 대한 한 시스템의 평가 결과."""

    system: str          # "kjx" 또는 --compare 로 넘긴 외부 시스템 이름
    direction: str       # "ko-ja" 형식
    samples: int
    chrf: float          # 주 지표 (0~100, 높을수록 좋음)
    bleu: float
    bleu_tokenize: str   # BLEU 토큰화 방식 (재현성 기록용)


def score_translations(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    target_language: str,
) -> tuple[float, float, str]:
    """(chrF, BLEU, BLEU 토큰화 방식) 을 반환합니다."""
    if len(hypotheses) != len(references):
        raise ValueError(
            f"번역문 {len(hypotheses)}개와 정답 {len(references)}개의 수가 다릅니다"
        )
    from sacrebleu.metrics import BLEU, CHRF

    tokenize = "char" if target_language in CHARACTER_LEVEL_LANGUAGES else "13a"
    chrf = CHRF().corpus_score(list(hypotheses), [list(references)]).score
    bleu = BLEU(tokenize=tokenize).corpus_score(list(hypotheses), [list(references)]).score
    return chrf, bleu, tokenize


def load_split_pairs(
    dataset_dir: str | Path,
    split: str,
    tokenizer: KJTokenizer,
    *,
    max_samples_per_direction: int,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """준비된 데이터셋의 holdout split 을 (원문, 정답) 텍스트 쌍으로 되돌립니다.

    반환: {(원문 언어, 목표 언어): [(원문, 정답), ...]}
    shard 에는 토큰 id 만 저장되어 있으므로 토크나이저로 디코딩합니다.
    """
    dataset = IndexedParallelDataset(dataset_dir, split, bidirectional=True)
    pairs: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        direction = (item["src_language"], item["target_language"])
        bucket = pairs.setdefault(direction, [])
        if len(bucket) >= max_samples_per_direction:
            # 두 방향 모두 상한에 도달하면 더 읽을 필요가 없습니다.
            if all(
                len(existing) >= max_samples_per_direction
                for existing in pairs.values()
            ) and len(pairs) == 2:
                break
            continue
        bucket.append(
            (tokenizer.decode(item["src"].tolist()), tokenizer.decode(item["tgt"].tolist()))
        )
    return pairs


def load_benchmark_pairs(
    paths: Sequence[str | Path],
    language_pair: Sequence[str],
    *,
    max_samples_per_direction: int,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """외부 벤치마크 JSONL(FLORES 변환본 등)을 평가쌍으로 읽습니다.

    형식은 학습 데이터와 같습니다: 한 줄에 {"ko": ..., "ja": ...}.
    양방향 모두 평가셋으로 씁니다.
    """
    key_a, key_b = language_pair
    forward: list[tuple[str, str]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text_a, text_b = row.get(key_a), row.get(key_b)
                if isinstance(text_a, str) and isinstance(text_b, str) and text_a and text_b:
                    forward.append((text_a, text_b))
                if len(forward) >= max_samples_per_direction:
                    break
    return {
        (key_a, key_b): forward,
        (key_b, key_a): [(b, a) for a, b in forward],
    }


def results_as_markdown(results: Sequence[DirectionResult]) -> str:
    """비교 표를 사람이 읽기 좋은 markdown 으로 만듭니다."""
    lines = [
        "| system | direction | samples | chrF | BLEU |",
        "|---|---|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.system} | {result.direction} | {result.samples} "
            f"| {result.chrf:.2f} | {result.bleu:.2f} |"
        )
    return "\n".join(lines)


def save_results(
    results: Sequence[DirectionResult],
    output_path: str | Path,
    *,
    metadata: dict,
) -> None:
    """결과를 JSON(기계용)과 Markdown(사람용)으로 저장합니다."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": [asdict(result) for result in results]}
    output_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_path.with_suffix(".md").write_text(
        results_as_markdown(results) + "\n", encoding="utf-8"
    )
