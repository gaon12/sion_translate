"""무관한 번역쌍을 이어붙여 다문장 학습 예제를 만드는 CLI.

    sion-concat --input "data/*.jsonl" --output data/concat_multi.jsonl --count 200000

산출 파일 이름이 ``concat_`` 으로 시작하면 ``sion-prepare-data`` 의
``--train-only-prefix`` 기본값에 걸려 train split 에만 들어갑니다. 합성 예제로
holdout 점수를 올리는 일을 막기 위한 것입니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sion_translate.concat import (
    SEPARATORS,
    build_concatenations,
    read_pairs,
    write_concatenations,
)
from sion_translate.console import configure_stdio
from sion_translate.data.prepare import DEFAULT_TRAIN_ONLY_PREFIXES, expand_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concatenate unrelated pairs into multi-sentence training examples"
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL 파일 또는 glob 패턴")
    parser.add_argument("--output", required=True, help="산출 JSONL 경로")
    parser.add_argument("--count", type=int, required=True, help="만들 예제 수")
    parser.add_argument("--min-sentences", type=int, default=2)
    parser.add_argument("--max-sentences", type=int, default=4)
    parser.add_argument(
        "--separator",
        default="space",
        choices=sorted(SEPARATORS),
        help="space=공백으로 이어붙임(실사용 입력에 가까움), seg=<seg> 제어 토큰으로 경계 명시",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=480,
        help="한쪽 최대 글자 수 (기본 480). 학습 shard 가 잘라낼 예제를 미리 버립니다",
    )
    parser.add_argument(
        "--tokenizer",
        help="SentencePiece 모델 경로. 주면 --max-tokens 를 토큰 수로 정확히 셉니다",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="한쪽 최대 토큰 수 (--tokenizer 와 함께 사용; 보통 학습의 max_tokens_per_side)",
    )
    parser.add_argument(
        "--language-pair",
        nargs=2,
        default=["ko", "ja"],
        metavar=("LANG_A", "LANG_B"),
        help="JSONL 키 이름 (기본: ko ja)",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    if args.max_tokens is not None and not args.tokenizer:
        raise SystemExit("--max-tokens 를 쓰려면 --tokenizer 도 지정해야 합니다")

    count_tokens = None
    if args.tokenizer:
        from sion_translate.tokenizer import SionTokenizer

        tokenizer = SionTokenizer(args.tokenizer)

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

    paths = expand_inputs(args.input)
    if not paths:
        raise SystemExit(f"입력 JSONL 을 찾지 못했습니다: {args.input}")

    pairs = list(read_pairs(paths, args.language_pair))
    if not pairs:
        raise SystemExit("읽을 수 있는 번역쌍이 없습니다")

    examples, stats = build_concatenations(
        pairs,
        count=args.count,
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        separator=args.separator,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
        count_tokens=count_tokens,
        seed=args.seed,
    )
    written = write_concatenations(args.output, examples, args.language_pair)

    output = Path(args.output)
    if not output.name.startswith(DEFAULT_TRAIN_ONLY_PREFIXES):
        print(
            f"[sion] 주의: {output.name} 은 "
            f"{' / '.join(DEFAULT_TRAIN_ONLY_PREFIXES)} 로 시작하지 않습니다. "
            "이대로면 합성 예제가 validation/test 로 들어가 holdout 점수를 부풀립니다. "
            "파일 이름을 바꾸거나 sion-prepare-data 의 --train-only-prefix 를 맞추십시오."
        )
    print(f"[sion] {written}개 예제를 {output} 에 썼습니다")
    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
