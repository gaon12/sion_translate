"""초안 수정 학습 데이터를 만드는 CLI.

    sion-revise-data --input "data/*.jsonl" --output data/revise_synthetic.jsonl

정답 번역을 실제 관측된 오류 유형으로 망가뜨려 초안을 만들고,
``원문 <draft> 초안 -> 정답 번역`` 예제를 씁니다. 학습된 모델이 필요하지 않으므로
사전학습 전에도 만들 수 있습니다.

이미 학습된 모델의 실제 출력을 초안으로 쓰려면 ``--drafts`` 로 초안 JSONL 을 주십시오
(``{"draft": ...}`` 를 한 줄에 하나씩, 입력과 같은 순서). 그 편이 분포가 정확하지만
모델이 먼저 있어야 합니다.
"""

# CLI result serializers are attached dynamically.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sion_translate.concat import read_records
from sion_translate.console import configure_stdio
from sion_translate.data.prepare import DEFAULT_TRAIN_ONLY_PREFIXES
from sion_translate.language_tags import canonicalize_language_pair
from sion_translate.revision import (
    DEFAULT_CORRUPTIONS,
    RevisionExample,
    RevisionStats,
    build_revision_examples,
    serialize_revision_input,
    write_revision_examples,
)
from sion_translate.tokenizer import expand_inputs


# 이 비율을 넘으면 손상이 대부분 통하지 않았다는 뜻이므로 경고합니다.
UNCHANGED_WARNING_RATIO = 0.40


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build source+draft -> target revision training data"
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL 파일 또는 glob 패턴")
    parser.add_argument("--output", required=True, help="산출 JSONL 경로")
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="사용할 최대 쌍 수 (기본: 전체)",
    )
    parser.add_argument(
        "--drafts",
        help=(
            '모델이 만든 초안 JSONL ({"draft": ...} 한 줄에 하나, 입력과 같은 순서). '
            "주면 합성 손상 대신 이 초안을 씁니다"
        ),
    )
    for kind, default in DEFAULT_CORRUPTIONS.items():
        parser.add_argument(
            f"--weight-{kind.replace('_', '-')}",
            type=float,
            default=default,
            dest=f"weight_{kind}",
            help=f"{kind} 손상 비중 (기본 {default})",
        )
    parser.add_argument(
        "--language-pair",
        nargs=2,
        required=True,
        metavar=("LANG_A", "LANG_B"),
        help="JSONL 언어 키와 revision 방향 (SOURCE TARGET)",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    return parser


def _load_drafts(path: str | Path) -> list[str]:
    drafts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            draft = row.get("draft")
            if not isinstance(draft, str) or not draft.strip():
                raise SystemExit(f"{path}:{number}: 'draft' 가 비어 있지 않은 문자열이어야 합니다")
            drafts.append(draft.strip())
    return drafts


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    paths = expand_inputs(args.input)
    if not paths:
        raise SystemExit(f"입력 JSONL 을 찾지 못했습니다: {args.input}")
    language_pair = canonicalize_language_pair(
        args.language_pair,
        field="revision CLI language_pair",
    )
    records = list(read_records(paths, language_pair))
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("읽을 수 있는 번역쌍이 없습니다")
    incompatible = [
        record.source_identifier
        for record in records
        if record.metadata.get("training_direction") is not None
        and record.metadata.get("training_direction") != list(language_pair)
    ]
    if incompatible:
        raise SystemExit(
            "입력 training_direction이 요청한 revision 방향과 다릅니다: "
            f"requested={language_pair!r}, first={incompatible[0]}"
        )
    pairs = [(record.text_a, record.text_b) for record in records]

    if args.drafts:
        drafts = _load_drafts(args.drafts)
        if len(drafts) != len(pairs):
            raise SystemExit(
                f"초안 {len(drafts)}개와 번역쌍 {len(pairs)}개의 수가 다릅니다 — "
                "같은 순서, 같은 개수여야 합니다"
            )
        raw_examples = [
            (serialize_revision_input(source, draft), target)
            for (source, target), draft in zip(pairs, drafts, strict=True)
        ]
        unchanged = sum(
            1 for (_, target), draft in zip(pairs, drafts, strict=True) if draft == target
        )
        stats = RevisionStats(
            len(raw_examples),
            {"model_draft": len(raw_examples)},
            unchanged,
        )
    else:
        weights = {kind: getattr(args, f"weight_{kind}") for kind in DEFAULT_CORRUPTIONS}
        weights = {kind: value for kind, value in weights.items() if value > 0}
        raw_examples, stats = build_revision_examples(pairs, weights=weights, seed=args.seed)

    examples = [
        RevisionExample(
            serialized_source=serialized,
            target=target,
            metadata=record.metadata,
            source_identifier=record.source_identifier,
        )
        for (serialized, target), record in zip(raw_examples, records, strict=True)
    ]

    written = write_revision_examples(args.output, examples, language_pair)
    output = Path(args.output)
    if not output.name.startswith(DEFAULT_TRAIN_ONLY_PREFIXES):
        print(
            f"[sion] 주의: {output.name} 은 "
            f"{' / '.join(DEFAULT_TRAIN_ONLY_PREFIXES)} 로 시작하지 않습니다. "
            "이대로면 합성 예제가 validation/test 로 들어가 holdout 점수를 부풀립니다."
        )
    print(f"[sion] {written}개 예제를 {output} 에 썼습니다")
    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))

    if written and stats.unchanged / written > UNCHANGED_WARNING_RATIO:
        share = 100.0 * stats.unchanged / written
        print(
            f"[sion] 주의: 초안의 {share:.0f}% 가 정답과 같습니다. "
            "짧은 단문 코퍼스에서는 손상이 적용되지 않는 경우가 많습니다 — "
            "숫자가 없으면 number 가, 절이 하나면 drop_clause 와 swap 이 그대로 둡니다. "
            "이 상태로 학습하면 수정 모델이 '고치지 않기'만 배웁니다. "
            "장문이 있는 입력을 쓰거나, 학습된 모델의 실제 출력을 --drafts 로 주십시오."
        )


if __name__ == "__main__":
    main()
