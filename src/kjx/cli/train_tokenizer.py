from __future__ import annotations

import argparse

from kjx.tokenizer import train_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the from-scratch KJ-X tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="JSONL files or glob patterns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=48000)
    parser.add_argument("--input-sentence-size", type=int, default=4_000_000)
    parser.add_argument("--seed-sentencepiece-size", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=None, help="전처리 프로세스 수 (기본: 자동)")
    parser.add_argument(
        "--threads", type=int, default=None, help="SentencePiece 스레드 수 (기본: 자동)"
    )
    parser.add_argument("--validation-fraction", type=float, default=0.005)
    parser.add_argument("--test-fraction", type=float, default=0.005)
    parser.add_argument(
        "--language-pair",
        nargs=2,
        default=["ko", "ja"],
        metavar=("LANG_A", "LANG_B"),
        help="JSONL 키 이름 (기본: ko ja)",
    )
    parser.add_argument(
        "--no-split-digits",
        dest="split_digits",
        action="store_false",
        help=(
            "숫자를 한 자리씩 분리하지 않음 (권장하지 않음). "
            "끄면 금액·용량·날짜가 다른 값으로 바뀌는 오역이 늘어납니다."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = train_tokenizer(
        args.input,
        args.output_dir,
        vocab_size=args.vocab_size,
        input_sentence_size=args.input_sentence_size,
        seed_sentencepiece_size=args.seed_sentencepiece_size,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        language_pair=args.language_pair,
        num_workers=args.workers,
        num_threads=args.threads,
        split_digits=args.split_digits,
    )
    print(model_path)


if __name__ == "__main__":
    main()
