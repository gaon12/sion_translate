"""오염 의심 정답쌍을 사람 검수 queue 로 뽑는다.

**아무것도 지우지 않습니다.** 원천별로 묶고 확신도 순으로 정렬한 JSONL 을
낼 뿐입니다. 자동 삭제가 틀린 이유는 로스트에 적힌 그대로입니다 — 규칙이
휴리스틱이라 정상 번역도 걸리고(`개` 가 실제 동물인 문장), 오염된 행의 가치는
삭제가 아니라 재번역에 있습니다.

    python scripts/data/build_review_queue.py \
        --input "data/*.jsonl" --output reports/review_queue.jsonl

출력 한 줄이 검수 대상 하나이고, 원본 파일과 행 번호를 그대로 들고 있어
고친 뒤 되돌려 넣을 수 있습니다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sion_translate.console import configure_stdio
from sion_translate.contamination import (
    assess_contamination,
    rank_findings,
    supported_direction,
)
from sion_translate.data.quality import canonical_text
from sion_translate.data.records import expand_parallel_record, normalize_language_pairs
from sion_translate.tokenizer import expand_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="오염 의심 정답쌍을 사람 검수 queue 로 뽑습니다 (삭제하지 않음)"
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL 파일 또는 glob")
    parser.add_argument("--output", required=True, help="검수 queue JSONL 경로")
    parser.add_argument("--language-pair", nargs=2, default=["ko", "ja"])
    parser.add_argument("--language-pairs", nargs=2, action="append")
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.0,
        help="이 확신도 미만은 queue 에서 제외 (기본 0: 전부 포함)",
    )
    parser.add_argument(
        "--summary",
        help="원천별·규칙별 집계 JSON 경로 (기본: stdout 요약만)",
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    pairs = normalize_language_pairs(args.language_pair, args.language_pairs)
    if not any(supported_direction(*pair) for pair in pairs):
        raise SystemExit(
            "이 도구는 ko→ja 규칙만 가지고 있습니다. 규칙 없는 방향을 "
            "'오염 없음' 으로 보고하지 않기 위해 여기서 중단합니다."
        )

    paths = expand_inputs(args.input)
    if not paths:
        raise SystemExit(f"입력과 일치하는 JSONL 이 없습니다: {args.input}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    by_rule: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    scanned = 0
    queued = 0

    with output.open("w", encoding="utf-8") as sink:
        for path in paths:
            with path.open("rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    try:
                        row = json.loads(raw_line.decode("utf-8-sig"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    for pair in expand_parallel_record(row, pairs).pairs:
                        if not supported_direction(pair.language_a, pair.language_b):
                            continue
                        scanned += 1
                        source = canonical_text(pair.text_a)
                        target = canonical_text(pair.text_b)
                        findings = assess_contamination(source, target)
                        leader = rank_findings(findings)
                        if leader is None or leader.confidence < args.minimum_confidence:
                            continue
                        queued += 1
                        by_rule[leader.rule] += 1
                        by_source[path.name] += 1
                        sink.write(
                            json.dumps(
                                {
                                    "file": str(path),
                                    "line": line_number,
                                    "source_language": pair.language_a,
                                    "target_language": pair.language_b,
                                    "source": source,
                                    "target": target,
                                    "confidence": leader.confidence,
                                    "rule": leader.rule,
                                    "reason": leader.reason,
                                    "all_rules": [
                                        {
                                            "rule": finding.rule,
                                            "reason": finding.reason,
                                            "confidence": finding.confidence,
                                            "evidence": list(finding.evidence),
                                        }
                                        for finding in findings
                                    ],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

    summary = {
        "scanned_pairs": scanned,
        "queued_pairs": queued,
        "queued_rate": (queued / scanned) if scanned else 0.0,
        "by_rule": dict(by_rule.most_common()),
        "by_source": dict(by_source.most_common()),
        "note": "이 목록은 자동 삭제 기준이 아니라 사람 검수·재번역 대상입니다.",
    }
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"검사한 쌍 {scanned:,} / 검수 대상 {queued:,} ({summary['queued_rate']:.3%})")
    for rule, count in by_rule.most_common():
        print(f"  {rule}: {count:,}")
    print(f"queue: {output}")


if __name__ == "__main__":
    main()
