"""expressive challenge 문장이 학습 코퍼스에 이미 있는지 감사한다.

    python scripts/data/audit_holdout_leakage.py \
        --holdout examples/expressive_cultural_cases.jsonl \
        --corpus "data/*.jsonl" --output reports/holdout_leakage.json

누출된 항목이 있으면 종료 코드가 0이 아닙니다. 독립 holdout 이 아닌 것을
품질 benchmark 로 인용하는 일을 막기 위한 관문입니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sion_translate.console import configure_stdio
from sion_translate.holdout_audit import (
    DEFAULT_SIMILARITY_THRESHOLD,
    audit_holdout_leakage,
    load_holdout_items,
    summarize,
)
from sion_translate.tokenizer import expand_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="challenge 문장의 학습 코퍼스 누출 감사")
    parser.add_argument("--holdout", nargs="+", required=True, help="challenge JSONL")
    parser.add_argument("--corpus", nargs="+", required=True, help="학습 JSONL 또는 glob")
    parser.add_argument("--language", nargs="+", default=["ko", "ja"])
    parser.add_argument("--similarity", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--max-matches", type=int, default=5)
    parser.add_argument("--output", help="JSON 보고서 경로")
    parser.add_argument(
        "--allow-leaks",
        action="store_true",
        help="누출이 있어도 0 으로 종료 (조사용)",
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    items = load_holdout_items(args.holdout, languages=args.language)
    if not items:
        raise SystemExit(f"challenge 문장을 읽지 못했습니다: {args.holdout}")
    corpus = expand_inputs(args.corpus)
    if not corpus:
        raise SystemExit(f"학습 코퍼스와 일치하는 JSONL 이 없습니다: {args.corpus}")

    print(f"challenge {len(items)}개를 코퍼스 {len(corpus)}개 파일과 대조합니다.", flush=True)
    findings = audit_holdout_leakage(
        items,
        corpus,
        similarity_threshold=args.similarity,
        maximum_matches_per_item=args.max_matches,
        languages=args.language,
    )
    summary = summarize(findings)
    report = {
        "summary": summary,
        "similarity_threshold": args.similarity,
        "corpus_files": [str(path) for path in corpus],
        "findings": [
            {
                "id": finding.item.identifier,
                "language": finding.item.language,
                "category": finding.item.category,
                "text": finding.item.text,
                "matches": [
                    {
                        "file": match.file,
                        "line": match.line,
                        "similarity": round(match.similarity, 4),
                        "exact": match.exact,
                        "text": match.text,
                    }
                    for match in sorted(finding.matches, key=lambda m: -m.similarity)
                ],
            }
            for finding in findings
            if finding.leaked
        ],
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"감사 {summary['audited_items']}개 / 누출 {summary['leaked_items']}개 "
        f"({summary['leak_rate']:.1%}), 그중 완전일치 {summary['exact_leaked_items']}개"
    )
    for finding in findings:
        worst = finding.worst
        if worst is None:
            continue
        print(
            f"  [{worst.similarity:.2f}{' 완전일치' if worst.exact else ''}] {finding.item.identifier}"
        )
        print(f"    holdout: {finding.item.text[:60]}")
        print(f"    corpus : {Path(worst.file).name}:{worst.line}  {worst.text[:60]}")
    if summary["leaked_items"] and not args.allow_leaks:
        raise SystemExit(
            "누출된 challenge 문장이 있습니다. 이 집합은 독립 holdout 이 아니므로 "
            "품질 benchmark 로 인용하지 마십시오 — 회귀 smoke set 으로만 쓰거나, "
            "누출된 항목을 교체하십시오."
        )


if __name__ == "__main__":
    main()
