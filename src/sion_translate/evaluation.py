"""번역 품질 평가 공용 로직.

고정된 평가셋에 대해 chrF / BLEU 를 계산합니다. `sion-evaluate` CLI 가
사용하며, 같은 평가셋에 대한 외부 서비스(DeepL/Google/Papago 등) 출력과의
비교도 지원합니다.

지표 선택 이유:
- **chrF** (문자 n-gram F-score): 토큰화가 필요 없어 한국어/일본어처럼
  띄어쓰기 규칙이 다른 언어에서도 공정합니다. 주 지표로 사용합니다.
- **BLEU**: 관례상 함께 보고합니다. 한/일/중은 단어 분리가 모호하므로
  문자 단위(tokenize="char")로, 그 외 언어는 표준 13a 토큰화로 계산합니다.
- **숫자 보존**: chrF/BLEU 는 문자 n-gram 이 대부분 겹치면 높은 점수를 주므로
  ``250mg`` → ``1200mg`` 처럼 값 하나만 바뀐 오역을 거의 벌하지 않습니다.
  금액·용량·날짜는 한 글자만 틀려도 문장 전체가 쓸 수 없게 되므로 따로
  집계합니다. 재현율(누락)과 정밀도(환각)를 함께 보기 위해 F1 을 쓰고,
  "숫자가 하나라도 틀린 문장이 몇 %인지"를 exact 비율로 함께 보고합니다.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from sion_translate.data import IndexedParallelDataset
from sion_translate.data.records import expand_parallel_record, normalize_language_pairs
from sion_translate.tokenizer import SionTokenizer

# 문자 단위 BLEU 를 쓰는 언어 (공백 기반 토큰화가 무의미한 언어)
CHARACTER_LEVEL_LANGUAGES = {"ko", "ja", "zh"}

# 금액/용량/날짜/버전 등 "값" 으로 취급할 숫자열. 사후학습 보상과 같은 정의를
# 써야 학습이 최적화하는 대상과 평가가 재는 대상이 어긋나지 않습니다.
#
# 경계 조건은 ASCII 영숫자/밑줄로만 판정합니다. 파이썬 ``\w`` 는 한글과 가나까지
# 포함하므로 ``(?![\w])`` 로 막으면 ``4월``, ``1회``, ``5개`` 처럼 조사·단위가 붙은
# 한 자리 숫자가 전부 값에서 빠집니다 — 날짜와 복용 횟수가 바로 그 형태입니다.
# 반대로 ``utf8``, ``config2``, ``HTTP429`` 의 숫자는 값이 아니라 식별자의 일부라서
# 앞뒤 ASCII 문자로 걸러냅니다. ``250mg`` 은 첫 분기가 ``250`` 을 잡아냅니다.
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?\d[\d,.:/%+\-]*\d|(?<![A-Za-z0-9_])[-+]?\d(?![0-9])"
)

# URL, 이메일, 코드 식별자처럼 번역하지 않고 그대로 옮겨야 하는 문자열.
STRUCTURED_PATTERN = re.compile(
    r"https?://[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"(?<![A-Za-z0-9])(?:[A-Z]{2,}[A-Z0-9_-]*|[A-Za-z]+[-_][A-Za-z0-9_-]+)"
    r"(?![A-Za-z0-9])"
)


def normalized_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    """NFKC 정규화 후 패턴에 걸린 표면형 목록. 전각 숫자도 반각과 같게 봅니다."""
    normalized = unicodedata.normalize("NFKC", text)
    return [match.group(0).casefold().rstrip(".,;:!?") for match in pattern.finditer(normalized)]


def multiset_f1(expected: Sequence[object], actual: Sequence[object]) -> float:
    """중복을 보존하는 F1. 둘 다 비었으면 위반이 없으므로 1입니다."""
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    if not expected_counts and not actual_counts:
        return 1.0
    if not expected_counts or not actual_counts:
        return 0.0
    overlap = sum((expected_counts & actual_counts).values())
    precision = overlap / sum(actual_counts.values())
    recall = overlap / sum(expected_counts.values())
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def structured_tokens(text: str) -> list[str]:
    """URL·이메일·식별자처럼 그대로 옮겨야 하는 문자열 목록."""
    return normalized_matches(STRUCTURED_PATTERN, text)


def has_excessive_repetition(text: str) -> bool:
    """한 문자나 짧은 구절이 병적으로 반복되는 생성 붕괴를 판정합니다."""
    surface = [char for char in text if not char.isspace()]
    if len(surface) < 12:
        return False
    if Counter(surface).most_common(1)[0][1] / len(surface) >= 0.70:
        return True
    return re.search(r"(.{1,8})\1{4,}", "".join(surface)) is not None


def numeric_tokens(text: str) -> list[str]:
    """문장에서 값으로 볼 숫자열을 뽑아냅니다. 콤마는 자리 구분자로 무시합니다.

    ``38,720`` 과 ``38720`` 은 같은 값이고, 한쪽 표기만 다른 것을 오역으로
    세면 지표가 표기 관습을 벌하게 됩니다.
    """
    return [token.replace(",", "") for token in normalized_matches(NUMBER_PATTERN, text)]


def number_preservation(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> tuple[float, int]:
    """(숫자 F1 평균 ×100, 숫자가 모두 일치한 문장 수) 를 돌려줍니다.

    숫자가 없는 문장쌍은 위반이 있을 수 없으므로 F1 1.0, 일치로 셉니다.
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"번역문 {len(hypotheses)}개와 정답 {len(references)}개의 수가 다릅니다")
    if not references:
        return 0.0, 0
    scores: list[float] = []
    exact = 0
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        expected = numeric_tokens(reference)
        actual = numeric_tokens(hypothesis)
        scores.append(multiset_f1(expected, actual))
        if Counter(expected) == Counter(actual):
            exact += 1
    return 100.0 * sum(scores) / len(scores), exact


@dataclass
class DirectionResult:
    """한 방향(예: ko→ja)에 대한 한 시스템의 평가 결과."""

    system: str  # "sion" 또는 --compare 로 넘긴 외부 시스템 이름
    direction: str  # "ko-ja" 형식
    samples: int
    chrf: float  # 주 지표 (0~100, 높을수록 좋음)
    bleu: float
    bleu_tokenize: str  # BLEU 토큰화 방식 (재현성 기록용)
    number_f1: float = 0.0  # 숫자 보존 F1 평균 (0~100)
    number_exact: int = 0  # 숫자가 모두 일치한 문장 수


def score_translations(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    target_language: str,
) -> tuple[float, float, str]:
    """(chrF, BLEU, BLEU 토큰화 방식) 을 반환합니다."""
    if len(hypotheses) != len(references):
        raise ValueError(f"번역문 {len(hypotheses)}개와 정답 {len(references)}개의 수가 다릅니다")
    from sacrebleu.metrics import BLEU, CHRF

    tokenize = "char" if target_language in CHARACTER_LEVEL_LANGUAGES else "13a"
    chrf = CHRF().corpus_score(list(hypotheses), [list(references)]).score
    bleu = BLEU(tokenize=tokenize).corpus_score(list(hypotheses), [list(references)]).score
    return chrf, bleu, tokenize


def load_split_pairs(
    dataset_dir: str | Path,
    split: str,
    tokenizer: SionTokenizer,
    *,
    max_samples_per_direction: int,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """준비된 데이터셋의 holdout split 을 (원문, 정답) 텍스트 쌍으로 되돌립니다.

    반환: {(원문 언어, 목표 언어): [(원문, 정답), ...]}
    shard 에는 토큰 id 만 저장되어 있으므로 토크나이저로 디코딩합니다.
    """
    dataset = IndexedParallelDataset(dataset_dir, split, bidirectional=True)
    # Source-only languages (한본어 kj) are never a target, so the reachable
    # direction count is smaller than 2x the pair count and the early exit
    # below has to use the real number or it never fires.
    expected_directions = dataset.direction_count
    pairs: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        direction = (item["src_language"], item["target_language"])
        bucket = pairs.setdefault(direction, [])
        if len(bucket) >= max_samples_per_direction:
            # 두 방향 모두 상한에 도달하면 더 읽을 필요가 없습니다.
            if (
                all(len(existing) >= max_samples_per_direction for existing in pairs.values())
                and len(pairs) == expected_directions
            ):
                break
            continue
        bucket.append(
            (tokenizer.decode(item["src"].tolist()), tokenizer.decode(item["tgt"].tolist()))
        )
    return pairs


def load_benchmark_pairs(
    paths: Sequence[str | Path],
    language_pair: Sequence[str] | Sequence[Sequence[str]],
    *,
    max_samples_per_direction: int,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """외부 벤치마크 JSONL(FLORES 변환본 등)을 평가쌍으로 읽습니다.

    형식은 학습 데이터와 같습니다: 한 줄에 {"ko": ..., "ja": ...}.
    양방향 모두 평가셋으로 씁니다.
    """
    if language_pair and isinstance(language_pair[0], str):
        language_pairs = normalize_language_pairs(language_pair)  # type: ignore[arg-type]
    else:
        language_pairs = normalize_language_pairs(
            language_pairs=language_pair  # type: ignore[arg-type]
        )
    output: dict[tuple[str, str], list[tuple[str, str]]] = {
        direction: [] for pair in language_pairs for direction in (pair, (pair[1], pair[0]))
    }
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                expansion = expand_parallel_record(row, language_pairs)
                for pair in expansion.pairs:
                    forward = output[(pair.language_a, pair.language_b)]
                    reverse = output[(pair.language_b, pair.language_a)]
                    if len(forward) < max_samples_per_direction:
                        forward.append((pair.text_a, pair.text_b))
                    if len(reverse) < max_samples_per_direction:
                        reverse.append((pair.text_b, pair.text_a))
                if all(len(samples) >= max_samples_per_direction for samples in output.values()):
                    break
    return output


def results_as_markdown(results: Sequence[DirectionResult]) -> str:
    """비교 표를 사람이 읽기 좋은 markdown 으로 만듭니다."""
    lines = [
        "| system | direction | samples | chrF | BLEU | 숫자 F1 | 숫자 일치 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        exact = f"{result.number_exact}/{result.samples}" if result.samples else "-"
        lines.append(
            f"| {result.system} | {result.direction} | {result.samples} "
            f"| {result.chrf:.2f} | {result.bleu:.2f} "
            f"| {result.number_f1:.2f} | {exact} |"
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
    output_path.with_suffix(".md").write_text(results_as_markdown(results) + "\n", encoding="utf-8")
