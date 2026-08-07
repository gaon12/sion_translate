"""번역 품질 평가 CLI — 고정 평가셋에서 chrF/BLEU 를 측정합니다.

    sion-evaluate                          # 자체 test split 양방향 평가
    sion-evaluate --benchmark flores.jsonl # 외부 벤치마크(JSONL) 평가
    sion-evaluate --direction ko-ja \
      --compare deepl=deepl_out.txt \
      --compare google=google_out.txt     # 외부 서비스 출력과 비교

외부 서비스 비교 방법:
    1. sion-evaluate --direction ko-ja --export-sources src.txt 로
       평가셋 원문을 파일로 뽑습니다.
    2. 그 원문을 DeepL/Google/Papago 등에 넣어 번역 결과를
       한 줄에 한 문장씩 파일로 저장합니다.
    3. --compare 서비스이름=결과파일 로 넘기면 같은 정답에 대해
       같은 지표로 나란히 채점됩니다.

결과는 터미널 표 + reports/evaluation-*.json/.md 로 저장됩니다.
"""

# CLI registry callables are discovered dynamically.
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sion_translate.config import config_from_raw, load_raw_config
from sion_translate.console import configure_stdio
from sion_translate.glossary import load_glossary
from sion_translate.evaluation import (
    DirectionResult,
    load_benchmark_pairs,
    load_split_pairs,
    number_preservation_details,
    results_as_markdown,
    save_results,
    score_translations,
)
from sion_translate.inference import Translator, find_exported_model

DEFAULT_CONFIG_FILE = "sion_translate.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate translation quality (chrF/BLEU)")
    parser.add_argument("--split", default="test", help="평가할 데이터셋 split (기본: test)")
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="외부 벤치마크 JSONL 경로 (지정하면 split 대신 사용; 여러 번 지정 가능)",
    )
    parser.add_argument(
        "--direction",
        default="both",
        help="평가 방향: both(기본) 또는 ko-ja 처럼 '원문-목표' 형식",
    )
    parser.add_argument("--max-samples", type=int, default=500, help="방향당 최대 평가 문장 수")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", help="내보낸 모델 경로 (기본: exports 자동 탐색)")
    parser.add_argument("--int8", action="store_true", help="INT8 양자화 모델 평가")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="외부 시스템 출력 파일(한 줄=한 번역). --direction 지정 필요",
    )
    parser.add_argument(
        "--export-sources",
        help="평가셋 원문을 이 파일로 저장 (외부 서비스에 넣을 입력 추출용). --direction 지정 필요",
    )
    parser.add_argument("--output", help="결과 저장 경로 (기본: reports/evaluation-<시각>)")
    parser.add_argument(
        "--glossary",
        help="용어집 JSON 경로 (지정 시 sion_translate 번역에 용어 강제 적용; 기본: 설정의 data.glossary)",
    )
    parser.add_argument(
        "--no-glossary",
        action="store_true",
        help="설정에 글로서리가 있어도 평가에서는 사용하지 않음",
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
    configured_edges = {frozenset(pair) for pair in configured_pairs}

    # ── 평가 방향 결정 ──────────────────────────────────────────────────
    if args.direction == "both":
        directions = [
            direction for pair in configured_pairs for direction in (pair, (pair[1], pair[0]))
        ]
    else:
        source, _, target = args.direction.partition("-")
        if frozenset((source, target)) not in configured_edges:
            valid = ", ".join(f"{left}-{right}/{right}-{left}" for left, right in configured_pairs)
            raise SystemExit(f"--direction 은 both 또는 다음 중 하나여야 합니다: {valid}")
        directions = [(source, target)]
    if (args.compare or args.export_sources) and len(directions) != 1:
        raise SystemExit(
            "--compare / --export-sources 는 --direction 을 한 방향으로 지정해야 합니다"
        )

    # ── 글로서리 (선택) ─────────────────────────────────────────────────
    glossary = None
    glossary_path = None if args.no_glossary else (args.glossary or config.data.glossary)
    if glossary_path:
        glossary = load_glossary(glossary_path)
        log(f"글로서리 적용: {glossary_path} ({len(glossary)}개 용어)")

    # ── 평가쌍 로드 ─────────────────────────────────────────────────────
    model_path = args.model or find_exported_model(config.training.output_dir, int8=args.int8)
    translator = Translator(model_path, config.data.tokenizer_model)
    if args.benchmark:
        log(f"벤치마크 로드: {', '.join(args.benchmark)}")
        pairs = load_benchmark_pairs(
            args.benchmark,
            configured_pairs,
            max_samples_per_direction=args.max_samples,
        )
        eval_set_name = ";".join(args.benchmark)
    else:
        log(f"자체 {args.split} split 로드 (학습에 전혀 노출되지 않은 holdout)")
        pairs = load_split_pairs(
            config.data.dataset_dir,
            args.split,
            translator.tokenizer,
            max_samples_per_direction=args.max_samples,
        )
        eval_set_name = f"dataset:{args.split}"

    # ── (선택) 외부 서비스용 원문 추출 ──────────────────────────────────
    if args.export_sources:
        direction = directions[0]
        sources = [source for source, _ in pairs.get(direction, [])]
        Path(args.export_sources).write_text("\n".join(sources) + "\n", encoding="utf-8")
        log(
            f"원문 {len(sources)}문장 저장: {args.export_sources} "
            f"(외부 서비스 번역 후 --compare 로 넘기세요)"
        )

    # ── 평가 실행 ───────────────────────────────────────────────────────
    results: list[DirectionResult] = []
    for source_language, target_language in directions:
        samples = pairs.get((source_language, target_language), [])
        if not samples:
            log(f"{source_language}-{target_language}: 평가쌍이 없어 건너뜁니다")
            continue
        sources = [source for source, _ in samples]
        references = [reference for _, reference in samples]
        direction_name = f"{source_language}-{target_language}"

        log(f"{direction_name}: {len(samples)}문장 번역 중 (beam {args.num_beams})...")
        started = time.perf_counter()
        hypotheses = translator.translate(
            sources,
            source_language=source_language,
            target_language=target_language,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            glossary=glossary,
        )
        elapsed = time.perf_counter() - started
        chrf, bleu, tokenize = score_translations(
            hypotheses, references, target_language=target_language
        )
        number_result = number_preservation_details(hypotheses, sources=sources)
        results.append(
            DirectionResult(
                system="sion",
                direction=direction_name,
                samples=len(samples),
                chrf=chrf,
                bleu=bleu,
                bleu_tokenize=tokenize,
                number_f1=number_result.f1,
                number_exact=number_result.exact,
                number_samples=number_result.samples,
                number_inventions=number_result.inventions,
            )
        )
        number_summary = (
            f"숫자 F1 {number_result.f1:.2f} "
            f"(일치 {number_result.exact}/{number_result.samples}, "
            f"환각 {number_result.inventions})"
            if number_result.samples
            else "숫자 문장 없음"
        )
        log(
            f"{direction_name}: chrF {chrf:.2f} / BLEU {bleu:.2f} / "
            f"{number_summary} ({elapsed:.0f}초)"
        )

        # 외부 시스템 출력 채점 (같은 정답, 같은 지표)
        for spec in args.compare:
            name, _, file_path = spec.partition("=")
            if not file_path:
                raise SystemExit(f"--compare 형식은 이름=파일 입니다: {spec}")
            lines = [
                line.rstrip("\n")
                for line in Path(file_path).read_text(encoding="utf-8").splitlines()
            ]
            if len(lines) < len(references):
                raise SystemExit(
                    f"{file_path}: 번역 {len(lines)}줄 < 평가쌍 {len(references)}개 — "
                    "줄 수가 평가셋과 일치해야 합니다"
                )
            hypotheses = lines[: len(references)]
            chrf, bleu, tokenize = score_translations(
                hypotheses, references, target_language=target_language
            )
            number_result = number_preservation_details(hypotheses, sources=sources)
            results.append(
                DirectionResult(
                    system=name,
                    direction=direction_name,
                    samples=len(references),
                    chrf=chrf,
                    bleu=bleu,
                    bleu_tokenize=tokenize,
                    number_f1=number_result.f1,
                    number_exact=number_result.exact,
                    number_samples=number_result.samples,
                    number_inventions=number_result.inventions,
                )
            )

    if not results:
        raise SystemExit("평가할 데이터가 없습니다.")

    # ── 결과 출력·저장 ──────────────────────────────────────────────────
    print()
    print(results_as_markdown(results))
    output = args.output or f"reports/evaluation-{time.strftime('%Y%m%d-%H%M%S')}"
    save_results(
        results,
        output,
        metadata={
            "model": str(model_path),
            "eval_set": eval_set_name,
            "num_beams": args.num_beams,
            "max_samples": args.max_samples,
            "language_pairs": [list(pair) for pair in configured_pairs],
        },
    )
    log(f"저장: {output}.json / {output}.md")


if __name__ == "__main__":
    main()
