"""비교 문장 JSONL을 KJ-X 또는 공개 baseline으로 번역한다."""

from __future__ import annotations

import argparse
from collections import defaultdict

from kjx.baselines import HF_BASELINES, translate_with_hf_baseline
from kjx.comparison import ComparisonCase, load_comparison_cases, write_system_translations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate comparison cases to output JSONL")
    parser.add_argument("--cases", required=True, help="비교 문장 JSONL")
    parser.add_argument("--output", required=True, help="번역 출력 JSONL")
    parser.add_argument("--backend", required=True, choices=("kjx", *HF_BASELINES))
    parser.add_argument("--model", help="KJ-X export(.pt); backend=kjx에서 필수")
    parser.add_argument("--tokenizer", help="KJ-X SentencePiece(.model); backend=kjx에서 필수")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 등")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def _translate_with_kjx(
    cases: list[ComparisonCase],
    *,
    model_path: str,
    tokenizer_path: str,
    device: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
) -> dict[str, str]:
    from kjx.inference import Translator

    resolved_device = None if device == "auto" else device
    translator = Translator(model_path, tokenizer_path, device=resolved_device)
    grouped: dict[tuple[str, str], list[ComparisonCase]] = defaultdict(list)
    for case in cases:
        grouped[(case.source_language, case.target_language)].append(case)

    translations: dict[str, str] = {}
    for (_, target_language), direction_cases in grouped.items():
        decoded = translator.translate(
            [case.source for case in direction_cases],
            target_language=target_language,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        translations.update(
            {case.id: text for case, text in zip(direction_cases, decoded, strict=True)}
        )
    return translations


def main() -> None:
    args = build_parser().parse_args()
    try:
        cases = load_comparison_cases(args.cases)
        if args.backend == "kjx":
            if not args.model or not args.tokenizer:
                raise ValueError("backend=kjx에는 --model과 --tokenizer가 필요합니다")
            translations = _translate_with_kjx(
                cases,
                model_path=args.model,
                tokenizer_path=args.tokenizer,
                device=args.device,
                batch_size=args.batch_size,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            translations = translate_with_hf_baseline(
                cases,
                backend=args.backend,
                device=args.device,
                batch_size=args.batch_size,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
        write_system_translations(args.output, cases, translations)
    except (OSError, UnicodeError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"저장: {args.output} ({len(cases)}문장)")


if __name__ == "__main__":
    main()
