"""번역 CLI — 학습된 모델로 양방향 번역을 수행합니다.

    kjx-translate --to ja "안녕하세요"        # 한국어 → 일본어
    kjx-translate --to ko "こんにちは"         # 일본어 → 한국어
    cat input.txt | kjx-translate --to ja     # 파일/파이프 입력 (줄 단위)

모델은 지정하지 않으면 runs/… 의 exports 에서 자동으로 찾습니다
(best 의 EMA 가중치 우선 — 보통 가장 품질이 좋습니다).
언어쌍은 토크나이저에서 자동 인식되므로 en-de 모델이면 --to de 처럼 씁니다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kjx.config import config_from_raw, load_raw_config
from kjx.console import configure_stdio
from kjx.glossary import load_glossary
from kjx.inference import Translator, find_exported_model

DEFAULT_CONFIG_FILE = "kjx.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate with a trained KJ-X model")
    parser.add_argument("text", nargs="*", help="번역할 문장 (없으면 표준 입력에서 줄 단위로 읽음)")
    parser.add_argument("--to", dest="target", help="목표 언어 (기본: 언어쌍의 두 번째, ko-ja 면 ja)")
    parser.add_argument("--model", help="내보낸 모델 경로 (기본: exports 에서 자동 탐색)")
    parser.add_argument(
        "--int8",
        action="store_true",
        help="INT8 양자화 모델 사용 (CPU 전용, 용량·메모리 절감. 속도는 빨라지지 않음)",
    )
    parser.add_argument("--num-beams", type=int, default=4, help="beam 수 (1=greedy, 기본 4)")
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--glossary",
        help="용어집 JSON 경로 (지정한 용어를 정해진 대응어로 강제; 기본: 설정의 data.glossary)",
    )
    parser.add_argument(
        "--no-glossary",
        action="store_true",
        help="설정에 글로서리가 있어도 이번에는 사용하지 않음",
    )
    parser.add_argument("--config", help=f"설정 파일 (기본: {DEFAULT_CONFIG_FILE})")
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    # 설정에서 토크나이저 위치와 출력 디렉터리를 알아냅니다.
    config_path = args.config or (
        DEFAULT_CONFIG_FILE if Path(DEFAULT_CONFIG_FILE).exists() else None
    )
    config = config_from_raw(load_raw_config(config_path) if config_path else {})

    model_path = args.model or find_exported_model(
        config.training.output_dir, int8=args.int8
    )
    translator = Translator(model_path, config.data.tokenizer_model)

    # 목표 언어: 지정하지 않으면 언어쌍의 두 번째 언어 (ko-ja 면 ja).
    target = args.target or config.data.language_pair[1]
    if target not in translator.languages:
        raise SystemExit(
            f"--to {target} 는 이 모델이 지원하지 않습니다 (지원: {sorted(translator.languages)})"
        )

    # 글로서리: --glossary > 설정 data.glossary. --no-glossary 면 끔.
    glossary = None
    glossary_path = None if args.no_glossary else (args.glossary or config.data.glossary)
    if glossary_path:
        glossary = load_glossary(glossary_path)
        print(
            f"[KJ-X] 글로서리 적용: {glossary_path} ({len(glossary)}개 용어)",
            file=sys.stderr,
            flush=True,
        )

    lines = args.text if args.text else [line.rstrip("\n") for line in sys.stdin]
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise SystemExit("번역할 문장이 없습니다.")

    print(f"[KJ-X] 모델: {model_path} → {target} 로 번역", file=sys.stderr, flush=True)
    for translated in translator.translate(
        lines,
        target_language=target,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        max_new_tokens=args.max_new_tokens,
        glossary=glossary,
    ):
        print(translated, flush=True)


if __name__ == "__main__":
    main()
