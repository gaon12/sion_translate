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
from collections.abc import Generator
from contextlib import contextmanager
import json
import os
from collections import Counter
from pathlib import Path
import tempfile
from typing import cast, TextIO

from sion_translate.console import configure_stdio
from sion_translate.contamination import (
    assess_contamination,
    rank_findings,
    supported_direction,
)
from sion_translate.data.quality import canonical_text
from sion_translate.data.records import expand_parallel_record, normalize_language_pairs
from sion_translate.tokenizer import expand_inputs


def _paths_alias(left: Path, right: Path) -> bool:
    """Return whether two paths resolve to the same file, including hard links."""

    try:
        if left.resolve() == right.resolve():
            return True
    except OSError:
        pass
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _fsync_directory(path: Path) -> None:
    """Persist a directory replacement on platforms that permit directory fsync."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _atomic_text_sink(path: Path) -> Generator[TextIO, None, None]:
    """Yield a private sibling file and publish it only after a complete sync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete=False,
        ) as sink:
            temporary = Path(sink.name)
            yield cast(TextIO, sink)
            sink.flush()
            os.fsync(sink.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    with _atomic_text_sink(path) as sink:
        sink.write(text)


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
    summary_path = Path(args.summary) if args.summary else None
    for label, candidate in (("output", output), ("summary", summary_path)):
        if candidate is not None and any(_paths_alias(candidate, path) for path in paths):
            raise SystemExit(
                f"The {label} path must not refer to an input shard. Choose a separate "
                f"destination: {candidate}"
            )
    if summary_path is not None and _paths_alias(output, summary_path):
        raise SystemExit("The review queue output and summary must use different paths.")

    by_rule: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    scanned = 0
    queued = 0

    with _atomic_text_sink(output) as sink:
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
    if summary_path is not None:
        _atomic_write_text(
            summary_path,
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
    print(
        f"Scanned pairs: {scanned:,} / queued for review: {queued:,} ({summary['queued_rate']:.3%})"
    )
    for rule, count in by_rule.most_common():
        print(f"  {rule}: {count:,}")
    print(f"Review queue: {output}")


if __name__ == "__main__":
    main()
