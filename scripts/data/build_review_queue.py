"""Build a human review queue of translation pairs that may be contaminated.

**This command never deletes corpus rows.** It emits review records in stable
input order and reports counts grouped by source and rule. Each row places its
highest-confidence finding first. Automatic deletion is unsafe because the
heuristics also catch valid translations, such as sentences where `개` really
means an animal. A contaminated row is also usually worth retranslating rather
than discarding.

    python scripts/data/build_review_queue.py \
        --input "data/*.jsonl" --output reports/review_queue.jsonl \
        --language-pair ko ja

Each output line is one review item. It retains the original file and line
number so a reviewer can repair the pair and place it back in the corpus.
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
        description=(
            "Build a human review queue of potentially contaminated translation "
            "pairs without deleting corpus rows"
        )
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL files or globs")
    parser.add_argument("--output", required=True, help="path for the review queue JSONL")
    language_group = parser.add_mutually_exclusive_group(required=True)
    language_group.add_argument("--language-pair", nargs=2)
    language_group.add_argument("--language-pairs", nargs=2, action="append")
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.0,
        help="exclude findings below this confidence (default: 0, include all)",
    )
    parser.add_argument(
        "--summary",
        help="path for counts by source and rule (default: print a summary only)",
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    pairs = normalize_language_pairs(args.language_pair, args.language_pairs)
    unsupported = [pair for pair in pairs if not supported_direction(*pair)]
    if unsupported:
        rendered = ", ".join(f"{source}→{target}" for source, target in unsupported)
        raise SystemExit(
            "This tool has rules only for ko->ja. It is stopping instead of silently "
            f"excluding unsupported directions: {rendered}"
        )

    paths = expand_inputs(args.input)
    if not paths:
        raise SystemExit(f"No JSONL files matched the requested inputs: {args.input}")

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
                        scanned += 1
                        source = canonical_text(pair.text_a)
                        target = canonical_text(pair.text_b)
                        findings = assess_contamination(
                            source,
                            target,
                            source_language=pair.language_a,
                            target_language=pair.language_b,
                        )
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
        "note": (
            "This list is a queue for human review and retranslation, not a rule "
            "for automatic deletion."
        ),
    }
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"Scanned pairs: {scanned:,} / queued for review: {queued:,} ({summary['queued_rate']:.3%})"
    )
    for rule, count in by_rule.most_common():
        print(f"  {rule}: {count:,}")
    print(f"Review queue: {output}")


if __name__ == "__main__":
    main()
