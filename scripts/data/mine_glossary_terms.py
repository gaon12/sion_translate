#!/usr/bin/env python3
"""Mine an aligned term glossary from a parallel corpus.

``build_glossary_hints.py`` needs a term list whose domain matches the corpus it
hints. Running it with data41 (Jeju place names) against the game corpus yielded
1,761 eligible rows out of 784,827 and 410 after capping, because place names in
Jeju do not appear in game dialogue. Terms mined from the same shards match by
construction.

Alignment is scored with the Dice coefficient over sentence co-occurrence::

    dice = 2 * rows_with_both / (rows_with_ko + rows_with_ja)

which is 1.0 for a pair that never appears apart. Dice alone is not enough: a
character stat table puts 계획력 in every row that also holds 生理的耐 *and*
戦場機動, so both score 1.0 and at most one can be right. Two further tests do the
real work:

``mutual best match``
    The Korean term's highest-scoring Japanese partner must have that same Korean
    term as *its* highest-scoring partner. A table artifact fails this because
    the Japanese column has its own best partner elsewhere.

``one-to-one``
    A Korean term with several Japanese partners above the threshold is dropped
    outright. ``설계국`` mapping to both ``製造局`` and ``中央設計`` cannot be a
    term pair; one of them is positional noise.

Usage::

    python scripts/data/mine_glossary_terms.py \
        --output glossary_mined.json --report reports/glossary-mining.json \
        --min-dice 0.85 --min-count 4 \
        data/data29.jsonl data/data33.jsonl data/data54.jsonl

Exit codes: 0 written, 2 bad input.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

# Katakana and kanji runs are the Japanese candidates: a loanword, a name or a
# compound noun. Hiragana is excluded because it is mostly grammar.
_KATAKANA = re.compile(r"[ァ-ヺー]{2,}")
_KANJI = re.compile(r"[一-鿿]{2,4}")
_HANGUL = re.compile(r"[가-힣]{2,}")
_MARKUP = re.compile(r"[{}<>\[\]|]|\$[A-Za-z_]|%[sd]\b")

# Korean particles glue to a noun, so the same term appears with different tails
# and would otherwise be counted as different terms. Longest first.
PARTICLES: tuple[str, ...] = (
    "으로써",
    "에서는",
    "에게서",
    "이라고",
    "으로부터",
    "라고",
    "으로",
    "에서",
    "에게",
    "한테",
    "까지",
    "부터",
    "이나",
    "처럼",
    "보다",
    "만큼",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "와",
    "과",
    "도",
    "로",
    "만",
)

# Conjugated endings. A glossary term is a noun that appears verbatim, so a verb
# or adjective form cannot be one - `보유한다 -> 無敵効果` and `해내면 -> 保全作業`
# are the shape of that failure. Matching the ending is enough because Korean
# conjugation is suffixal.
CONJUGATED_ENDINGS: tuple[str, ...] = (
    "한다",
    "된다",
    "진다",
    "인다",
    "간다",
    "온다",
    "났다",
    "했다",
    "었다",
    "았다",
    "린다",
    "른다",
    "친다",
    "웠다",
    "이다",
    "하면",
    "되면",
    "지면",
    "으면",
    "하고",
    "하며",
    "하여",
    "해서",
    "해도",
    "니다",
    "세요",
    "겠다",
    "는다",
    "준다",
    "본다",
    "주리",
    "리라",
    "느냐",
    "는가",
    "구나",
    "군요",
    "네요",
    "어요",
    "아요",
    "지만",
    "면서",
    "니까",
    "거야",
    "잖아",
    "는데",
    # Conditional -(으)면 after a verb stem. The bare `면` is not listed because
    # 라면 and 냉면 are nouns; only the stems that actually conjugate are.
    "내면",
    "가면",
    "오면",
    "보면",
    "주면",
    "받으면",
    "이라면",
)


def is_conjugated(word: str) -> bool:
    """True when ``word`` looks like a verb or adjective form rather than a noun."""

    return any(word.endswith(ending) for ending in CONJUGATED_ENDINGS)


# The list above is explicit rather than derived, so it is necessarily
# incomplete: Korean conjugation is suffixal but the ending set is large, and a
# blanket rule would take nouns with it (라면, 냉면, 바다 all end like verbs).
# What survives is caught by the mutual-best test instead.


# Words that carry no terminology and would otherwise dominate the counts.
STOPWORDS: frozenset[str] = frozenset(
    {
        "그리고",
        "하지만",
        "그래서",
        "그러나",
        "때문에",
        "이것",
        "저것",
        "그것",
        "여기",
        "거기",
        "저기",
        "우리",
        "당신",
        "자신",
        "지금",
        "오늘",
        "내일",
        "어제",
        "정말",
        "진짜",
        "조금",
        "많이",
        "다시",
        "아직",
        "이제",
        "함께",
        "먼저",
        "이런",
        "저런",
        "그런",
        "무슨",
        "어떤",
    }
)


@dataclass
class MiningResult:
    inputs: list[str] = field(default_factory=list)
    output: str = ""
    rows_read: int = 0
    rows_used: int = 0
    distinct_source_terms: int = 0
    distinct_target_terms: int = 0
    candidate_pairs: int = 0
    above_threshold: int = 0
    rejected_not_mutual: int = 0
    rejected_ambiguous: int = 0
    rejected_length_ratio: int = 0
    terms_written: int = 0
    min_dice: float = 0.0
    min_count: int = 0
    samples: list[dict[str, Any]] = field(default_factory=list)
    rejected_samples: list[dict[str, Any]] = field(default_factory=list)


def strip_particle(word: str) -> str:
    """Remove a trailing Korean particle, leaving at least two syllables."""

    for particle in PARTICLES:
        if len(word) > len(particle) + 1 and word.endswith(particle):
            return word[: -len(particle)]
    return word


def source_terms(text: str) -> set[str]:
    found = {strip_particle(word) for word in _HANGUL.findall(text)}
    return {
        word
        for word in found
        if len(word) >= 2 and word not in STOPWORDS and not is_conjugated(word)
    }


def target_terms(text: str) -> set[str]:
    return set(_KATAKANA.findall(text)) | set(_KANJI.findall(text))


def read_rows(
    paths: Sequence[Path], *, source_key: str, target_key: str
) -> Iterable[tuple[str, str]]:
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                source = row.get(source_key)
                target = row.get(target_key)
                if isinstance(source, str) and isinstance(target, str):
                    yield source, target


def mine(
    paths: Sequence[Path],
    output: Path,
    *,
    source_key: str,
    target_key: str,
    min_dice: float,
    min_count: int,
    max_length_ratio: float,
    max_source_chars: int,
) -> MiningResult:
    result = MiningResult(
        inputs=[str(path) for path in paths],
        output=str(output),
        min_dice=min_dice,
        min_count=min_count,
    )

    source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    joint: defaultdict[tuple[str, str], int] = defaultdict(int)

    for source, target in read_rows(paths, source_key=source_key, target_key=target_key):
        result.rows_read += 1
        if not source or not target or len(source) > max_source_chars:
            continue
        if _MARKUP.search(source) or _MARKUP.search(target):
            continue
        found_target = target_terms(target)
        if not found_target:
            continue
        found_source = source_terms(source)
        if not found_source:
            continue
        result.rows_used += 1
        for term in found_source:
            source_counts[term] += 1
        for term in found_target:
            target_counts[term] += 1
        for source_term in found_source:
            for target_term in found_target:
                joint[(source_term, target_term)] += 1

    result.distinct_source_terms = len(source_counts)
    result.distinct_target_terms = len(target_counts)
    result.candidate_pairs = len(joint)

    scored: list[tuple[float, int, str, str]] = []
    for (source_term, target_term), count in joint.items():
        if count < min_count:
            continue
        dice = 2 * count / (source_counts[source_term] + target_counts[target_term])
        if dice < min_dice:
            continue
        scored.append((dice, count, source_term, target_term))
    result.above_threshold = len(scored)

    # Best partner in each direction, needed for the mutual-best-match test.
    best_for_source: dict[str, tuple[float, str]] = {}
    best_for_target: dict[str, tuple[float, str]] = {}
    partners_per_source: defaultdict[str, list[str]] = defaultdict(list)
    for dice, _, source_term, target_term in scored:
        partners_per_source[source_term].append(target_term)
        if dice > best_for_source.get(source_term, (0.0, ""))[0]:
            best_for_source[source_term] = (dice, target_term)
        if dice > best_for_target.get(target_term, (0.0, ""))[0]:
            best_for_target[target_term] = (dice, source_term)

    entries: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for dice, count, source_term, target_term in sorted(scored, reverse=True):
        if source_term in seen_sources:
            continue
        # A Korean term with several Japanese partners over the threshold is a
        # table artifact: 설계국 cannot be both 製造局 and 中央設計.
        if len(partners_per_source[source_term]) > 1:
            result.rejected_ambiguous += 1
            if len(result.rejected_samples) < 12:
                result.rejected_samples.append(
                    {
                        "reason": "ambiguous",
                        source_key: source_term,
                        "candidates": partners_per_source[source_term][:4],
                    }
                )
            seen_sources.add(source_term)
            continue
        # Mutual best match. A stat-table column has its own best partner
        # elsewhere, so it fails this even at dice 1.0.
        if best_for_target.get(target_term, (0.0, ""))[1] != source_term:
            result.rejected_not_mutual += 1
            if len(result.rejected_samples) < 12:
                result.rejected_samples.append(
                    {
                        "reason": "not_mutual",
                        source_key: source_term,
                        target_key: target_term,
                        "target_prefers": best_for_target.get(target_term, (0.0, ""))[1],
                    }
                )
            continue
        ratio = max(len(source_term), len(target_term)) / max(
            1, min(len(source_term), len(target_term))
        )
        if ratio > max_length_ratio:
            result.rejected_length_ratio += 1
            if len(result.rejected_samples) < 12:
                result.rejected_samples.append(
                    {
                        "reason": "length_ratio",
                        source_key: source_term,
                        target_key: target_term,
                        "ratio": round(ratio, 2),
                    }
                )
            continue
        seen_sources.add(source_term)
        entries.append(
            {
                source_key: source_term,
                target_key: target_term,
                "dice": round(dice, 4),
                "count": count,
            }
        )
        if len(result.samples) < 40:
            result.samples.append(entries[-1])

    entries.sort(key=lambda entry: (-float(entry["dice"]), str(entry[source_key])))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)

    result.terms_written = len(entries)
    return result


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--source-key", default="ko")
    parser.add_argument("--target-key", default="ja")
    parser.add_argument(
        "--min-dice",
        type=float,
        default=0.85,
        help="co-occurrence score floor; 1.0 means the two never appear apart",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=4,
        help="rows a pair must co-occur in, so a single coincidence cannot score 1.0",
    )
    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=3.0,
        help="reject a pair whose sides differ wildly in length, e.g. 간조 against 勘定奉行",
    )
    parser.add_argument("--max-source-chars", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        print(f"not a file: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        result = mine(
            args.inputs,
            args.output,
            source_key=args.source_key,
            target_key=args.target_key,
            min_dice=args.min_dice,
            min_count=args.min_count,
            max_length_ratio=args.max_length_ratio,
            max_source_chars=args.max_source_chars,
        )
    except (OSError, ValueError) as error:
        print(f"cannot mine ({error})", file=sys.stderr)
        return 2

    print(
        f"{result.rows_read:,} rows read, {result.rows_used:,} used -> "
        f"{result.terms_written:,} terms"
    )
    print(
        f"  {result.candidate_pairs:,} candidate pairs, "
        f"{result.above_threshold:,} above dice {result.min_dice}"
    )
    print(
        f"  rejected: ambiguous={result.rejected_ambiguous:,} "
        f"notMutual={result.rejected_not_mutual:,} "
        f"lengthRatio={result.rejected_length_ratio:,}"
    )
    for entry in result.samples[:15]:
        print(
            f"      {entry['dice']:.3f} x{entry['count']:<5} "
            f"{entry[args.source_key]:20} -> {entry[args.target_key]}"
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
