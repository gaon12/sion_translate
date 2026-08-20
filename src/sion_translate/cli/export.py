"""Convert a stable FP32/EMA export into deployment/storage formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sion_translate.console import configure_stdio
from sion_translate.training.export import (
    DEFAULT_CONVERSION_FORMATS,
    SUPPORTED_FORMATS,
    convert_export,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a Sion FP32/EMA state-dict export")
    parser.add_argument("source", help="model.pt 또는 model_ema.pt")
    parser.add_argument(
        "--output",
        required=True,
        help="변환 파일과 export_manifest.json을 저장할 폴더",
    )
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_CONVERSION_FORMATS),
        help=("쉼표로 구분한 형식 (기본: " + ",".join(DEFAULT_CONVERSION_FORMATS) + ")"),
    )
    parser.add_argument("--tokenizer", help="호환성 SHA256을 기록할 tokenizer model")
    parser.add_argument(
        "--token-features",
        help="MorphoScript token_features.npz (HF checkpoint에도 복사·검증)",
    )
    parser.add_argument(
        "--language-pair",
        nargs=2,
        action="append",
        dest="language_pairs",
        metavar=("SOURCE", "TARGET"),
        help=("내보낼 언어 방향. 여러 번 지정 가능: --language-pair ko ja --language-pair en ru"),
    )
    parser.add_argument(
        "--unidirectional",
        action="store_true",
        help="각 --language-pair의 SOURCE→TARGET 방향만 학습된 것으로 기록",
    )
    parser.add_argument(
        "--release-name",
        help="metadata가 없는 구형 export의 배포 이름(예: sion 또는 sion_translate)",
    )
    parser.add_argument(
        "--release-version",
        help="metadata가 없는 구형 export의 모델 세대(예: 1.0); 추측하지 않음",
    )
    release_capability = parser.add_mutually_exclusive_group()
    release_capability.add_argument(
        "--translation-capable",
        dest="translation_capable",
        action="store_true",
        help="metadata가 없는 구형 export가 번역 학습을 마쳤음을 명시",
    )
    release_capability.add_argument(
        "--foundation-only",
        dest="translation_capable",
        action="store_false",
        help="metadata가 없는 구형 export가 번역 불가 foundation 가중치임을 명시",
    )
    parser.set_defaults(translation_capable=None)
    capability = parser.add_mutually_exclusive_group()
    capability.add_argument(
        "--revision-trained",
        action="store_true",
        help="revision 학습 예제가 포함됐음을 metadata에 기록",
    )
    capability.add_argument(
        "--no-revision-trained",
        action="store_true",
        help="revision 학습 예제가 없음을 metadata에 기록",
    )
    parser.add_argument(
        "--int4-backend",
        choices=("auto", "torchao", "packed"),
        default="auto",
        help="auto는 TorchAO 실패 시 portable packed INT4로 전환",
    )
    parser.add_argument(
        "--llama-quantize",
        help=("하위 호환 인자(현재 무시됨). GGUF는 내장 deterministic K-quant로 생성"),
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    formats = tuple(value.strip().lower() for value in args.formats.split(",") if value.strip())
    unknown = sorted(set(formats) - set(SUPPORTED_FORMATS))
    if unknown:
        raise SystemExit(f"지원하지 않는 형식: {unknown} (지원: {', '.join(SUPPORTED_FORMATS)})")
    revision_trained = (
        True if args.revision_trained else False if args.no_revision_trained else None
    )
    manifest = convert_export(
        args.source,
        args.output,
        formats=formats,
        tokenizer_path=args.tokenizer,
        token_features_path=args.token_features,
        language_pairs=args.language_pairs,
        bidirectional=not args.unidirectional,
        revision_trained=revision_trained,
        release_name=args.release_name,
        release_version=args.release_version,
        translation_capable=args.translation_capable,
        int4_backend=args.int4_backend,
        llama_quantize=args.llama_quantize,
    )
    print(json.dumps(manifest["formats"], ensure_ascii=False, indent=2))
    failures = [
        name for name, result in manifest["formats"].items() if result.get("status") != "ok"
    ]
    print(f"[sion] manifest: {Path(args.output) / 'export_manifest.json'}")
    if failures:
        raise SystemExit(f"내보내기 실패 형식: {', '.join(failures)}")


if __name__ == "__main__":
    main()
