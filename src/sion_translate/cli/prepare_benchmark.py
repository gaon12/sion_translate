"""FLORES-200 표준 평가셋을 sion-evaluate 형식(JSONL)으로 변환하는 CLI.

FLORES-200 은 공개 논문·상용 번역 서비스가 공통으로 쓰는 표준 평가셋이라,
여기에 우리 점수를 맞추면 외부 숫자와 '절대 비교'가 가능합니다.

권장 (오프라인): FLORES-200 배포판을 내려받아 압축을 푼 뒤 그 폴더를 지정.
    # 배포판 예: https://tinyurl.com/flores200dataset  (언어별 텍스트 파일 모음)
    sion-prepare-benchmark --flores-dir flores200_dataset --split devtest

또는 (Hugging Face datasets 설치 시):
    sion-prepare-benchmark --hf --split devtest

결과는 benchmarks/flores_<a>_<b>.<split>.jsonl 로 저장되며, 바로
    sion-evaluate --benchmark benchmarks/flores_ko_ja.devtest.jsonl
로 평가할 수 있습니다. 언어쌍은 sion_translate.yaml 의 language_pair 를 따릅니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sion_translate.benchmark import (
    flores_code,
    pairs_from_hf_datasets,
    pairs_from_local_flores,
    write_jsonl,
)
from sion_translate.config import config_from_raw, load_raw_config
from sion_translate.console import configure_stdio

DEFAULT_CONFIG_FILE = "sion_translate.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert FLORES-200 to sion-evaluate JSONL")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--flores-dir", help="압축 해제한 FLORES-200 폴더 (오프라인)")
    source.add_argument("--hf", action="store_true", help="Hugging Face datasets 로 내려받기")
    parser.add_argument(
        "--split",
        default="devtest",
        choices=("dev", "devtest"),
        help="FLORES split (기본: devtest — 표준 평가용)",
    )
    parser.add_argument(
        "--flores-code",
        action="append",
        default=[],
        metavar="LANG=CODE",
        help="언어별 FLORES 코드 수동 지정 (예: ko=kor_Hang). 기본 코드가 있으면 생략 가능",
    )
    parser.add_argument(
        "--output", help="출력 JSONL 경로 (기본: benchmarks/flores_<a>_<b>.<split>.jsonl)"
    )
    parser.add_argument(
        "--language-pair",
        nargs=2,
        metavar=("LANG_A", "LANG_B"),
        help="다국어 설정에서 변환할 한 언어쌍",
    )
    parser.add_argument("--config", help=f"설정 파일 (기본: {DEFAULT_CONFIG_FILE})")
    return parser


def log(message: str) -> None:
    print(f"[sion] {message}", flush=True)


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    config_path = args.config or (
        DEFAULT_CONFIG_FILE if Path(DEFAULT_CONFIG_FILE).exists() else None
    )
    config = config_from_raw(load_raw_config(config_path) if config_path else {})
    configured_pairs = config.data.configured_language_pairs()
    if args.language_pair:
        pair = tuple(args.language_pair)
        if frozenset(pair) not in {frozenset(item) for item in configured_pairs}:
            raise SystemExit(
                f"설정에 없는 --language-pair 입니다: {pair} (지원: {configured_pairs})"
            )
    elif len(configured_pairs) == 1:
        pair = configured_pairs[0]
    else:
        raise SystemExit(
            "다국어 설정에서는 --language-pair LANG_A LANG_B를 지정하세요 "
            f"(지원: {configured_pairs})"
        )

    # --flores-code ko=kor_Hang ... 파싱
    overrides: dict[str, str] = {}
    for spec in args.flores_code:
        language, _, code = spec.partition("=")
        if not code:
            raise SystemExit(f"--flores-code 형식은 언어=코드 입니다: {spec}")
        overrides[language] = code
    # 코드가 모두 확인되는지 먼저 검사 (친절한 에러를 위해)
    codes = {language: flores_code(language, overrides.get(language)) for language in pair}

    if args.hf:
        log(f"Hugging Face datasets 에서 FLORES {args.split} 로드 ({pair[0]}↔{pair[1]})")
        pairs = pairs_from_hf_datasets(pair, split=args.split, code_overrides=overrides)
    else:
        if not args.flores_dir:
            raise SystemExit(
                "--flores-dir <폴더> 또는 --hf 를 지정하세요. "
                "FLORES-200 배포판을 내려받아 압축을 푼 폴더를 --flores-dir 로 주면 됩니다."
            )
        log(
            f"로컬 FLORES {args.split} 로드: {args.flores_dir} ({codes[pair[0]]}, {codes[pair[1]]})"
        )
        pairs = pairs_from_local_flores(
            args.flores_dir, pair, split=args.split, code_overrides=overrides
        )

    output = args.output or f"benchmarks/flores_{pair[0]}_{pair[1]}.{args.split}.jsonl"
    count = write_jsonl(pairs, output)
    log(f"{count:,}개 병렬쌍 저장: {output}")
    log(f"평가: sion-evaluate --benchmark {output}")


if __name__ == "__main__":
    main()
