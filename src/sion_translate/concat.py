"""서로 무관한 번역쌍을 이어붙여 긴 다문장 학습 예제를 만든다.

문장 단위로만 학습한 모델은 여러 문장을 한 번에 받으면 뒤쪽을 빠뜨리거나, 두
문장을 하나로 합쳐 버리거나, 중간에서 생성을 멈추는 일이 있습니다. 원문 코퍼스가
전부 한 문장짜리여도 이어붙이기만으로 다음을 지도할 수 있습니다.

- 긴 토큰 시퀀스 처리: 입력·출력 길이가 문장 하나보다 몇 배 길어집니다.
- 문장 누락 방지: 원문 문장 수와 번역문 문장 수가 같아야 정답이 됩니다.
- 앞뒤 정렬 유지: 두 언어에서 같은 순서로 이어붙이므로 순서를 바꾸면 틀립니다.
- 긴 출력 생성: 조기 종료(EOS)가 정답이 아닌 예제가 생깁니다.
- 문장 경계 인식: 구분자가 경계 신호가 됩니다.

**무관한 문장을 일부러 씁니다.** 문맥이 이어지는 문단을 쓰면 모델이 앞 문장의
내용으로 뒤 문장을 추측할 수 있어, 실제로는 "빠뜨리지 않고 다 옮기는" 능력이
아니라 문맥 예측을 배우게 됩니다. 서로 관계가 없으면 각 문장을 실제로 읽어야만
정답을 맞출 수 있습니다.

산출 파일은 ``concat_`` 으로 시작하므로, ``prepare_dataset`` 의
``train_only_prefixes`` 에 그 접두어를 넣으면 합성 예제가 validation/test 로
새지 않습니다.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence, cast

DEFAULT_LANGUAGE_PAIR = ("ko", "ja")

# <seg> 은 토크나이저가 이미 예약해 둔 제어 토큰입니다. 공백 구분자는 사용자가
# 여러 문장을 그냥 붙여 넣는 실제 사용 형태에 가깝고, <seg> 는 경계를 명시적으로
# 지도합니다.
SEPARATORS = {"space": " ", "seg": " <seg> "}


@dataclass
class ConcatStats:
    """이어붙이기 결과 요약."""

    source_pairs: int
    written: int
    skipped_too_long: int
    sentences_per_example: dict[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_pairs": self.source_pairs,
            "written": self.written,
            "skipped_too_long": self.skipped_too_long,
            "sentences_per_example": {
                str(count): total for count, total in sorted(self.sentences_per_example.items())
            },
        }


def read_pairs(
    paths: Sequence[str | Path],
    language_pair: Sequence[str] = DEFAULT_LANGUAGE_PAIR,
) -> Iterator[tuple[str, str]]:
    """JSONL 에서 (원문, 번역) 쌍을 읽습니다. 잘못된 줄은 건너뜁니다."""
    key_a, key_b = language_pair
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                mapping = cast(dict[object, object], row)
                text_a, text_b = mapping.get(key_a), mapping.get(key_b)
                if isinstance(text_a, str) and isinstance(text_b, str):
                    text_a, text_b = text_a.strip(), text_b.strip()
                    if text_a and text_b:
                        yield text_a, text_b


def _too_long(
    joined_a: str,
    joined_b: str,
    *,
    max_tokens: int | None,
    max_chars: int,
    count_tokens: Callable[[str], int] | None,
) -> bool:
    if len(joined_a) > max_chars or len(joined_b) > max_chars:
        return True
    if max_tokens is not None and count_tokens is not None:
        if count_tokens(joined_a) > max_tokens or count_tokens(joined_b) > max_tokens:
            return True
    return False


def build_concatenations(
    pairs: Sequence[tuple[str, str]],
    *,
    count: int,
    min_sentences: int = 2,
    max_sentences: int = 4,
    separator: str = "space",
    max_chars: int = 480,
    max_tokens: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
    seed: int = 20260726,
) -> tuple[list[tuple[str, str]], ConcatStats]:
    """무관한 쌍을 묶어 다문장 예제를 만듭니다.

    같은 예제 안에서는 쌍을 재사용하지 않으므로 한 문장이 자기 자신과 이어붙는
    일은 없습니다. 길이 상한을 넘는 조합은 버립니다 — 학습 shard 가 어차피
    잘라낼 예제를 만들어 둘 이유가 없습니다.
    """
    if count < 0:
        raise ValueError("count 는 0 이상이어야 합니다")
    if min_sentences < 2:
        raise ValueError("min_sentences 는 2 이상이어야 합니다 (이어붙이기의 목적)")
    if max_sentences < min_sentences:
        raise ValueError("max_sentences 는 min_sentences 이상이어야 합니다")
    if separator not in SEPARATORS:
        raise ValueError(f"separator 는 {sorted(SEPARATORS)} 중 하나여야 합니다")
    if len(pairs) < min_sentences:
        raise ValueError(f"쌍이 {len(pairs)}개뿐이라 {min_sentences}문장을 만들 수 없습니다")

    joiner = SEPARATORS[separator]
    rng = random.Random(seed)
    built: list[tuple[str, str]] = []
    histogram: dict[int, int] = {}
    skipped = 0
    # 무한 루프 방지: 길이 상한이 빡빡하면 목표 수를 못 채울 수 있습니다.
    attempts = 0
    attempt_budget = max(count * 20, 100)
    while len(built) < count and attempts < attempt_budget:
        attempts += 1
        wanted = rng.randint(min_sentences, min(max_sentences, len(pairs)))
        chosen = rng.sample(range(len(pairs)), wanted)
        joined_a = joiner.join(pairs[index][0] for index in chosen)
        joined_b = joiner.join(pairs[index][1] for index in chosen)
        if _too_long(
            joined_a,
            joined_b,
            max_tokens=max_tokens,
            max_chars=max_chars,
            count_tokens=count_tokens,
        ):
            skipped += 1
            continue
        built.append((joined_a, joined_b))
        histogram[wanted] = histogram.get(wanted, 0) + 1
    return built, ConcatStats(len(pairs), len(built), skipped, histogram)


def write_concatenations(
    output_path: str | Path,
    examples: Iterable[tuple[str, str]],
    language_pair: Sequence[str] = DEFAULT_LANGUAGE_PAIR,
) -> int:
    """``prepare_dataset`` 가 읽는 형식으로 씁니다."""
    key_a, key_b = language_pair
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for text_a, text_b in examples:
            handle.write(json.dumps({key_a: text_a, key_b: text_b}, ensure_ascii=False) + "\n")
            written += 1
    return written
