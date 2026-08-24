"""Audit whether expressive challenge sentences already occur in the training corpus.

    python scripts/data/audit_holdout_leakage.py \
        --holdout examples/expressive_cultural_cases.jsonl \
        --corpus "data/*.jsonl" --language-pair ko ja \
        --output reports/holdout_leakage.json

The command exits with a nonzero status when it finds a leak. This safety gate
prevents a non-independent holdout from being cited as a quality benchmark.
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
    parser = argparse.ArgumentParser(
        description="Audit challenge-sentence leakage into a training corpus"
    )
    parser.add_argument("--holdout", nargs="+", required=True, help="challenge JSONL files")
    parser.add_argument("--corpus", nargs="+", required=True, help="training JSONL files or globs")
    parser.add_argument(
        "--language-pair",
        nargs=2,
        action="append",
        required=True,
        metavar=("LANG_A", "LANG_B"),
    )
    parser.add_argument("--similarity", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--max-matches", type=int, default=5)
    parser.add_argument("--output", help="path for the JSON report")
    parser.add_argument(
        "--allow-leaks",
        action="store_true",
        help="exit successfully even when leaks are found (investigation only)",
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    items = load_holdout_items(args.holdout, language_pairs=args.language_pair)
    if not items:
        raise SystemExit(f"No challenge sentences could be read from: {args.holdout}")
    corpus = expand_inputs(args.corpus)
    if not corpus:
        raise SystemExit(f"No training JSONL files matched: {args.corpus}")

    print(
        f"Comparing {len(items)} challenge item(s) against {len(corpus)} corpus file(s).",
        flush=True,
    )
    findings = audit_holdout_leakage(
        items,
        corpus,
        similarity_threshold=args.similarity,
        maximum_matches_per_item=args.max_matches,
        language_pairs=args.language_pair,
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
        f"Audited items: {summary['audited_items']} / leaked items: "
        f"{summary['leaked_items']} ({summary['leak_rate']:.1%}) / exact leaks: "
        f"{summary['exact_leaked_items']}"
    )
    for finding in findings:
        worst = finding.worst
        if worst is None:
            continue
        print(
            f"  [{worst.similarity:.2f}{' exact' if worst.exact else ''}] {finding.item.identifier}"
        )
        print(f"    holdout: {finding.item.text[:60]}")
        print(f"    corpus : {Path(worst.file).name}:{worst.line}  {worst.text[:60]}")
    if summary["leaked_items"] and not args.allow_leaks:
        raise SystemExit(
            "Leaked challenge sentences were found. This set is not an independent "
            "holdout and must not be cited as a quality benchmark. Use it only as a "
            "regression smoke set, or replace the leaked items."
        )


if __name__ == "__main__":
    main()
