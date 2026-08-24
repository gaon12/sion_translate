"""번역 CLI — 학습된 모델로 양방향 번역을 수행합니다.

    sion-translate --to ja "안녕하세요"        # 한국어 → 일본어
    sion-translate --to ko "こんにちは"         # 일본어 → 한국어
    cat input.txt | sion-translate --to ja     # 파일/파이프 입력 (줄 단위)

모델은 지정하지 않으면 runs/… 의 exports 에서 자동으로 찾습니다
(best 의 EMA 가중치 우선 — 보통 가장 품질이 좋습니다).
언어쌍은 토크나이저에서 자동 인식되므로 en-de 모델이면 --to de 처럼 씁니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from sion_translate.config import config_from_raw, load_raw_config
from sion_translate.console import configure_stdio
from sion_translate.generation import (
    DEFAULT_LENGTH_PENALTY,
    DEFAULT_MAX_OUTPUT_LENGTH_RATIO,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_NUM_BEAMS,
)
from sion_translate.glossary import load_glossary
from sion_translate.inference import Translator, find_exported_model
from sion_translate.iterative import refine_batch, summarize
from sion_translate.language_tags import (
    LanguageTagError,
    canonicalize_language_pair,
    canonicalize_language_tag,
)
from sion_translate.rerank import STRATEGIES as RERANK_STRATEGIES

DEFAULT_CONFIG_FILE = "sion_translate.yaml"


def _canonical_cli_language(value: str | None, *, option: str) -> str | None:
    if value is None:
        return None
    try:
        return canonicalize_language_tag(value, field=option)
    except LanguageTagError as error:
        raise SystemExit(str(error)) from error


def _canonical_model_directions(
    trained_directions: Sequence[Sequence[str]],
) -> tuple[tuple[str, str], ...]:
    directions: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_direction in enumerate(trained_directions):
        try:
            direction = canonicalize_language_pair(
                raw_direction,
                field=f"model translation_directions[{index}]",
            )
        except LanguageTagError as error:
            raise SystemExit(str(error)) from error
        if direction in seen:
            raise SystemExit(
                "모델 translation_directions에 BCP 47 정규화 후 중복인 방향이 "
                f"있습니다: {direction[0]}→{direction[1]}"
            )
        seen.add(direction)
        directions.append(direction)
    return tuple(directions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate with a trained sion_translate model")
    parser.add_argument("text", nargs="*", help="번역할 문장 (없으면 표준 입력에서 줄 단위로 읽음)")
    parser.add_argument(
        "--from",
        dest="source",
        help="원문 언어 (다국어 모델에서는 필수)",
    )
    parser.add_argument(
        "--to",
        dest="target",
        help="목표 언어 (모델에 학습 방향이 정확히 하나일 때만 생략 가능)",
    )
    parser.add_argument("--model", help="내보낸 모델 경로 (기본: exports 에서 자동 탐색)")
    parser.add_argument(
        "--int8",
        action="store_true",
        help="INT8 양자화 모델 사용 (CPU 전용, 용량·메모리 절감. 속도는 빨라지지 않음)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=DEFAULT_NUM_BEAMS,
        help="number of beams; use 1 for greedy decoding (default: 4)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=0,
        help=(
            "beam 결과에 더해 뽑을 확률적 후보 수 (0=재순위 없음). "
            "재학습 없이 추론 계산량만 늘려 품질을 올리는 경로입니다."
        ),
    )
    parser.add_argument(
        "--rerank",
        default="mbr+qe",
        choices=RERANK_STRATEGIES,
        help="후보 선택 방식 (기본 mbr+qe). --candidates 가 1 이상일 때만 적용",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="후보 샘플링 온도 (기본 0.3 — 홀드아웃에서 0.7 보다 나았음)",
    )
    parser.add_argument("--top-k", type=int, default=0, help="후보 샘플링 top-k (0=제한 없음)")
    parser.add_argument(
        "--revise-rounds",
        type=int,
        default=0,
        help=(
            "초안 수정 최대 반복 횟수 (0=끔). 쉬운 문장은 한 번만 번역하고 "
            "기준 미달 문장만 다시 고칩니다. sion-revise-data 로 만든 데이터로 "
            "학습한 모델에서만 의미가 있습니다"
        ),
    )
    parser.add_argument(
        "--accept-score",
        type=float,
        default=0.95,
        help="QE 점수가 이 값 이상이면 수정하지 않음 (기본 0.95)",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.01,
        help="한 라운드의 QE 개선이 이 값 미만이면 중단 (기본 0.01)",
    )
    parser.add_argument("--length-penalty", type=float, default=DEFAULT_LENGTH_PENALTY)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--reasoning-level",
        type=int,
        choices=range(10),
        default=None,
        metavar="0-9",
        help=(
            "내부 검증/후보정제 endpoint (0=직접 번역, 1-9=설정된 반복의 "
            "단조 단계; 반복 1이면 1-9가 동일). 생략하면 학습·검증과 같은 "
            "checkpoint 기본 endpoint를 사용합니다"
        ),
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=DEFAULT_NO_REPEAT_NGRAM_SIZE,
        help="forbid repeated n-grams of this size; use 0 to disable (default: 4)",
    )
    parser.add_argument(
        "--max-output-length-ratio",
        type=float,
        default=DEFAULT_MAX_OUTPUT_LENGTH_RATIO,
        help="maximum output/source token ratio, plus a separate 16-token margin",
    )
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


def resolve_translation_target(
    requested: str | None,
    source_language: str | None,
    trained_directions: Sequence[Sequence[str]],
) -> str:
    """Compatibility wrapper returning the target of an authenticated direction."""

    return resolve_translation_direction(
        requested,
        source_language,
        trained_directions,
    )[1]


def resolve_translation_direction(
    requested_target: str | None,
    requested_source: str | None,
    trained_directions: Sequence[Sequence[str]],
) -> tuple[str, str]:
    """Resolve both endpoints, including a uniquely implied missing endpoint."""

    directions = _canonical_model_directions(trained_directions)
    if not directions:
        raise SystemExit("모델에 인증된 translation_directions가 없습니다")
    source = _canonical_cli_language(requested_source, option="--from")
    target = _canonical_cli_language(requested_target, option="--to")
    supported = ", ".join(f"{edge_source}→{edge_target}" for edge_source, edge_target in directions)

    if source is None and target is None:
        if len(directions) != 1:
            raise SystemExit(
                "모델에 학습 방향이 여러 개입니다. --from LANG 또는 --to LANG을 "
                f"지정하세요 (지원: {supported})"
            )
        return directions[0]
    if source is not None and target is not None:
        if (source, target) in set(directions):
            return source, target
        raise SystemExit(f"{source}→{target} 는 학습되지 않은 방향입니다 (지원: {supported})")
    if source is not None:
        outgoing = [direction for direction in directions if direction[0] == source]
        if not outgoing:
            raise SystemExit(
                f"--from {source} 에서 출발하는 학습 방향이 없습니다 (지원: {supported})"
            )
        if len(outgoing) > 1:
            choices = ", ".join(
                f"{edge_source}→{edge_target}" for edge_source, edge_target in outgoing
            )
            raise SystemExit(
                f"--from {source} 에서 갈 수 있는 target이 여러 개입니다. "
                f"--to LANG을 지정하세요 (지원: {choices})"
            )
        return outgoing[0]

    assert target is not None
    incoming = [direction for direction in directions if direction[1] == target]
    if not incoming:
        raise SystemExit(f"--to {target} 는 학습된 target이 아닙니다 (지원: {supported})")
    if len(incoming) > 1:
        choices = ", ".join(f"{edge_source}→{edge_target}" for edge_source, edge_target in incoming)
        raise SystemExit(
            f"--to {target} 로 들어오는 source가 여러 개입니다. "
            f"--from LANG을 지정하세요 (지원: {choices})"
        )
    return incoming[0]


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    # 설정에서 토크나이저 위치와 출력 디렉터리를 알아냅니다.
    config_path = args.config or (
        DEFAULT_CONFIG_FILE if Path(DEFAULT_CONFIG_FILE).exists() else None
    )
    config = config_from_raw(load_raw_config(config_path) if config_path else {})

    model_path = args.model or find_exported_model(config.training.output_dir, int8=args.int8)
    translator = Translator(model_path, config.data.tokenizer_model)

    source_language, target = resolve_translation_direction(
        args.target,
        args.source,
        translator.translation_directions,
    )

    # 글로서리: --glossary > 설정 data.glossary. --no-glossary 면 끔.
    glossary = None
    glossary_path = None if args.no_glossary else (args.glossary or config.data.glossary)
    if glossary_path:
        glossary = load_glossary(glossary_path)
        print(
            f"[sion] 글로서리 적용: {glossary_path} ({len(glossary)}개 용어)",
            file=sys.stderr,
            flush=True,
        )

    lines = args.text if args.text else [line.rstrip("\n") for line in sys.stdin]
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise SystemExit("번역할 문장이 없습니다.")

    print(f"[sion] 모델: {model_path} → {target} 로 번역", file=sys.stderr, flush=True)
    if args.candidates > 0:
        print(
            f"[sion] 후보 {args.candidates + 1}개(beam 1 + 샘플 {args.candidates})를 "
            f"{args.rerank} 로 재순위",
            file=sys.stderr,
            flush=True,
        )
    translations = translator.translate(
        lines,
        source_language=source_language,
        target_language=target,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        max_new_tokens=args.max_new_tokens,
        glossary=glossary,
        num_candidates=args.candidates,
        rerank=args.rerank,
        temperature=args.temperature,
        top_k=args.top_k,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_output_length_ratio=args.max_output_length_ratio,
        reasoning_level=args.reasoning_level,
    )

    if args.revise_rounds > 0:
        if translator.tokenizer.draft_id is None:
            raise SystemExit(
                "이 토크나이저에는 <draft> 제어 토큰이 없어 --revise-rounds 를 쓸 수 "
                "없습니다. sion-train-tokenizer 로 토크나이저를 다시 학습하고, "
                "sion-revise-data 로 만든 데이터를 학습에 포함하십시오."
            )

        def revise_batch(sources: Sequence[str], drafts: Sequence[str]) -> list[str]:
            return translator.revise(
                sources,
                drafts,
                source_language=source_language,
                target_language=target,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                max_new_tokens=args.max_new_tokens,
                reasoning_level=args.reasoning_level,
            )

        results = refine_batch(
            lines,
            translations,
            revise_batch,
            target_language=target,
            accept_score=args.accept_score,
            min_gain=args.min_gain,
            max_rounds=args.revise_rounds,
        )
        translations = [result.text for result in results]
        print(
            f"[sion] 반복 수정: {json.dumps(summarize(results), ensure_ascii=False)}",
            file=sys.stderr,
            flush=True,
        )

    for translated in translations:
        print(translated, flush=True)


if __name__ == "__main__":
    main()
