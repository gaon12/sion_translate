#!/usr/bin/env python3
"""Build glossary-hint training examples from a parallel corpus and a term list.

The slot mechanism in :mod:`sion_translate.glossary` enforces a term at inference
time by cutting it out of the source and pasting it back. That guarantees the
surface but the model never sees the term, so it cannot inflect around it. A soft
hint shows both sides and lets the model produce agreeing morphology, which is
only possible if it was trained to read the hint.

This emits those training rows. For each corpus row it finds term pairs whose
source surface appears in the source *and* whose target surface appears in the
target, then rewrites the source with a hint prefix:

    <glossary> 사과 <protect> Apple <seg> 나는 오늘 사과를 먹었다.

Requiring the target surface to be present is the point. A hint whose target does
not occur in the reference would teach the model that hints can be ignored, which
is the opposite of the intent.

The term list can be an explicit glossary JSON or another parallel shard used as
one. data41.jsonl is 44,475 place and person name pairs, which is exactly a
proper-noun glossary, and proper nouns are where the released checkpoint fails
worst.

Hint rate is a parameter rather than a constant because it is the thing to ablate:
too low and the model ignores hints, too high and unhinted translation degrades.

Usage::

    python scripts/data/build_glossary_hints.py \\
        --corpus "data/data9.jsonl" --terms-from-corpus data/data41.jsonl \\
        --rate 0.10 --output data/synthetic_glossary_hints.jsonl

    python scripts/data/build_glossary_hints.py \\
        --corpus data/data29.jsonl --glossary glossary_input.json \\
        --rate 0.15 --report report.json --output out.jsonl

Exit codes: 0 rows written, 2 bad input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

from sion_translate.data.quality import canonical_text
from sion_translate.glossary import (
    Glossary,
    build_hinted_source,
    load_glossary,
    rank_terms_for_hinting,
)
from sion_translate.scripts_registry import script_of, scripts_in


@dataclass
class BuildReport:
    corpus_rows: int = 0
    unreadable_rows: int = 0
    glossary_terms: int = 0
    eligible_rows: int = 0
    rejected_undelimited: int = 0
    hinted_rows: int = 0
    rows_out: int = 0
    hints_per_row: dict[int, int] = field(default_factory=dict)
    most_hinted_terms: list[tuple[str, int]] = field(default_factory=list)
    rate: float = 0.0
    max_hints_per_row: int = 0
    max_per_term: int = 0
    seed: int = 0


def iter_rows(path: Path, source_key: str, target_key: str):
    """Yield canonicalized (source, target), or (None, None) when unreadable."""

    with path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                yield None, None
                continue
            if not isinstance(row, dict):
                yield None, None
                continue
            source = row.get(source_key)
            target = row.get(target_key)
            if not isinstance(source, str) or not isinstance(target, str):
                yield None, None
                continue
            yield canonical_text(source), canonical_text(target)


def glossary_from_corpus(
    path: Path,
    source_key: str,
    target_key: str,
    *,
    min_length: int = 2,
    max_length: int = 24,
) -> Glossary:
    """Treat a short-entry parallel shard as a glossary.

    Rows long enough to be sentences are skipped: a glossary entry is a term, and
    substring-matching a whole sentence would hint the answer.
    """

    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, target in iter_rows(path, source_key, target_key):
        if source is None or target is None:
            continue
        if not (min_length <= len(source) <= max_length):
            continue
        if not (min_length <= len(target) <= max_length):
            continue
        if source in seen:
            continue
        seen.add(source)
        entries.append({source_key: source, target_key: target})
    return Glossary(tuple(entries))


class TermTrie:
    """Multi-pattern substring matcher over a term list.

    A term-by-term ``in`` scan is O(rows x terms), which is 315,251 x 44,475
    substring searches for data29 against data41 and does not finish. Walking a
    trie from each position is effectively linear in the text, because for real
    sentences the walk stops after one or two characters.
    """

    __slots__ = ("_root",)

    def __init__(self, terms: Iterable[str]) -> None:
        self._root: dict[str, object] = {}
        for term in terms:
            if not term:
                continue
            node = self._root
            for char in term:
                node = node.setdefault(char, {})  # type: ignore[assignment]
            node[""] = term  # type: ignore[index]

    def find_all(self, text: str) -> set[str]:
        """Every term occurring anywhere in ``text``."""

        found: set[str] = set()
        root = self._root
        for start in range(len(text)):
            node = root
            for index in range(start, len(text)):
                node = node.get(text[index])  # type: ignore[assignment]
                if node is None:
                    break
                term = node.get("")  # type: ignore[union-attr]
                if term is not None:
                    found.add(term)  # type: ignore[arg-type]
        return found


def is_delimited(text: str, term: str) -> bool:
    """True when ``term`` occurs in ``text`` on its own rather than inside a word.

    Korean and Japanese do not delimit words with spaces, so a plain substring
    search matches 미나 inside 루미나 and produces a hint telling the model to
    translate a fragment. Requiring the match to be bounded by a different script
    class, punctuation, or a string edge rejects that.

    This trades recall for precision on purpose: a missed hint only costs one
    training example, while a wrong hint teaches the model to mistranslate.
    """

    if not term:
        return False
    start = 0
    inner = script_of(term[0])
    inner_end = script_of(term[-1])
    while True:
        index = text.find(term, start)
        if index < 0:
            return False
        before = text[index - 1] if index else ""
        after = text[index + len(term)] if index + len(term) < len(text) else ""
        left_ok = not before or script_of(before) != inner
        right_ok = not after or script_of(after) != inner_end
        if left_ok and right_ok:
            return True
        start = index + 1


def _stable_fraction(text: str, seed: int) -> float:
    """A deterministic value in [0, 1) for a row, independent of file order."""

    digest = hashlib.blake2b(f"{seed}\0{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def build(
    corpus_paths: list[Path],
    glossary: Glossary,
    *,
    source_key: str,
    target_key: str,
    rate: float,
    max_hints_per_row: int,
    max_per_term: int,
    seed: int,
    examples: int = 5,
) -> tuple[list[dict[str, object]], BuildReport, list[dict[str, str]]]:
    """Emit hinted rows for a fraction of the corpus."""

    if not 0.0 < rate <= 1.0:
        raise ValueError("rate must be in (0, 1]")
    if max_hints_per_row < 1:
        raise ValueError("max_hints_per_row must be positive")
    if max_per_term < 1:
        raise ValueError("max_per_term must be positive")

    directional = glossary.for_direction(source_key, target_key)
    report = BuildReport(
        glossary_terms=len(directional),
        rate=rate,
        max_hints_per_row=max_hints_per_row,
        max_per_term=max_per_term,
        seed=seed,
    )
    if not directional:
        raise ValueError(f"glossary has no {source_key}->{target_key} entries")

    targets_for = dict(directional)
    source_trie = TermTrie(targets_for)
    target_trie = TermTrie(targets_for.values())

    # Count source-term frequency across the corpus so rare terms rank first.
    frequency: Counter[str] = Counter()
    rows: list[tuple[str, str, set[str], set[str]]] = []
    for path in corpus_paths:
        for source, target in iter_rows(path, source_key, target_key):
            report.corpus_rows += 1
            if source is None or target is None:
                report.unreadable_rows += 1
                continue
            source_hits = source_trie.find_all(source)
            target_hits = target_trie.find_all(target) if source_hits else set()
            rows.append((source, target, source_hits, target_hits))
            frequency.update(source_hits)

    term_usage: Counter[str] = Counter()
    hint_counts: Counter[int] = Counter()
    output: list[dict[str, object]] = []
    samples: list[dict[str, str]] = []

    for source, target, source_hits, target_hits in rows:
        # A pair is usable only when the target surface is actually in the
        # reference; otherwise the hint teaches the model to ignore hints.
        usable = [
            (source_term, targets_for[source_term])
            for source_term in sorted(source_hits)
            if targets_for[source_term] in target_hits
            and is_delimited(source, source_term)
            and is_delimited(target, targets_for[source_term])
        ]
        if not usable:
            if source_hits and target_hits:
                report.rejected_undelimited += 1
            continue
        report.eligible_rows += 1
        if _stable_fraction(source, seed) >= rate:
            continue
        ranked = rank_terms_for_hinting(usable, frequency)
        selected: list[tuple[str, str]] = []
        for pair in ranked:
            if len(selected) >= max_hints_per_row:
                break
            if term_usage[pair[0]] >= max_per_term:
                continue
            selected.append(pair)
        if not selected:
            continue
        for pair in selected:
            term_usage[pair[0]] += 1
        hinted = build_hinted_source(source, selected)
        row: dict[str, object] = {
            source_key: hinted,
            target_key: target,
            "synthetic": True,
            "glossary_terms": [list(pair) for pair in selected],
        }
        output.append(row)
        report.hinted_rows += 1
        hint_counts[len(selected)] += 1
        if len(samples) < examples:
            samples.append({"source": hinted, "target": target})

    report.rows_out = len(output)
    report.hints_per_row = dict(sorted(hint_counts.items()))
    report.most_hinted_terms = term_usage.most_common(5)
    return output, report, samples


def write_rows(rows: list[dict[str, object]], output: Path) -> None:
    """Write atomically so an interrupted build leaves no partial shard."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus", action="append", required=True, help="parallel JSONL, repeatable"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--glossary", help="glossary JSON")
    source.add_argument("--terms-from-corpus", help="parallel shard of short entries used as terms")
    parser.add_argument("--output", required=True, help="destination JSONL")
    parser.add_argument("--report", help="write the build report JSON here")
    parser.add_argument("--source-key", default="ko")
    parser.add_argument("--target-key", default="ja")
    parser.add_argument(
        "--rate",
        type=float,
        default=0.10,
        help="fraction of eligible rows to hint; the value to ablate (0.05-0.20)",
    )
    parser.add_argument("--max-hints-per-row", type=int, default=3)
    parser.add_argument(
        "--max-per-term",
        type=int,
        default=50,
        help="cap on how often one term may be hinted, so frequent terms do not dominate",
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--examples", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.glossary:
            glossary = load_glossary(args.glossary)
        else:
            glossary = glossary_from_corpus(
                Path(args.terms_from_corpus), args.source_key, args.target_key
            )
        rows, report, samples = build(
            [Path(path) for path in args.corpus],
            glossary,
            source_key=args.source_key,
            target_key=args.target_key,
            rate=args.rate,
            max_hints_per_row=args.max_hints_per_row,
            max_per_term=args.max_per_term,
            seed=args.seed,
            examples=max(0, args.examples),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"build failed: {error}", file=sys.stderr)
        return 2
    if not rows:
        print("build produced no hinted rows", file=sys.stderr)
        return 2

    output = Path(args.output)
    write_rows(rows, output)
    payload = asdict(report)
    payload["output"] = str(output)
    payload["examples"] = samples
    if args.report:
        Path(args.report).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"{output}: {report.rows_out:,} hinted rows from {report.eligible_rows:,} eligible "
        f"of {report.corpus_rows:,} (rate {report.rate:.2f}), "
        f"{report.glossary_terms:,} glossary terms"
    )
    for sample in samples[:2]:
        print(f"    {sample['source']}")
        print(f"      -> {sample['target']}")
    unexpected = {name for row in rows for name in scripts_in(str(row[args.target_key]))}
    print(f"    target scripts present: {sorted(unexpected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
