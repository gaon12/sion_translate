#!/usr/bin/env python3
"""Generate parallel rows for the domains the corpus has no coverage for.

법률 holds 100 rows and 전자상거래 / IT 기술문서 / 행정 민원 / 관광·교통 hold
none. Those domains are also the ones whose real text is formulaic, so a template
matches the real distribution here - unlike 문학 or 방언, where the same approach
cost data44/data45 90% of their rows to restatement.

The guard against becoming that failure is the skeleton cap. Rows are grouped by
the audit's own skeleton definition (quoted spans blanked, digits collapsed) and
at most ``--max-per-skeleton`` rows are kept per group. A frame that varies only
in digits therefore contributes a handful of rows, not thousands, and
``formulaic_lexicon.validate`` refuses any frame with fewer than two word slots
in the first place.

Usage::

    python scripts/data/build_formulaic_corpus.py \
        --output data/synthetic_formulaic.jsonl \
        --report reports/formulaic-build.json \
        --max-per-skeleton 3

Exit codes: 0 written, 2 bad input or an invalid lexicon.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

_LEXICON_PATH = Path(__file__).resolve().parent / "formulaic_lexicon.py"
_SPEC = importlib.util.spec_from_file_location("formulaic_lexicon", _LEXICON_PATH)
assert _SPEC is not None and _SPEC.loader is not None
lexicon = importlib.util.module_from_spec(_SPEC)
sys.modules["formulaic_lexicon"] = lexicon
_SPEC.loader.exec_module(lexicon)

_DIGITS = re.compile(r"\d+")
_QUOTED = re.compile(r"[\"'“”「」『』]([^\"'“”「」『』]{1,60})[\"'“”「」『』]")


@dataclass
class BuildResult:
    output: str = ""
    rows_written: int = 0
    combinations_considered: int = 0
    skipped_skeleton_cap: int = 0
    skipped_duplicate: int = 0
    skipped_unresolved: int = 0
    max_per_skeleton: int = 0
    per_domain: dict[str, int] = field(default_factory=dict)
    distinct_skeletons: dict[str, int] = field(default_factory=dict)
    samples: list[dict[str, str]] = field(default_factory=list)


def frame_skeleton(text: str) -> str:
    """The audit's skeleton, so the cap here and the audit downstream agree."""

    return _DIGITS.sub("#", _QUOTED.sub("<Q>", text))


def rank_key(seed: str, *parts: str) -> str:
    payload = "\x00".join((seed,) + parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def assignments(
    frame: lexicon.Frame,
    *,
    limit: int,
    seed: str,
) -> list[dict[str, tuple[str, str]]]:
    """Slot assignments for ``frame``, deterministically ordered and bounded.

    The full product of every slot is far larger than the cap needs, so it is
    ranked by a seeded hash and truncated. Ranking rather than slicing the raw
    product keeps the kept subset from being biased toward whichever value
    happens to come first in each table.
    """

    names = frame.slots()
    tables = [lexicon.slot_values(name) for name in names]
    if not names or any(not table for table in tables):
        return []
    combinations = itertools.product(*tables)
    ordered = sorted(
        (dict(zip(names, values, strict=True)) for values in combinations),
        key=lambda mapping: rank_key(seed, *(value[0] for value in mapping.values())),
    )
    return ordered[:limit]


def build(
    output: Path,
    *,
    source_key: str,
    target_key: str,
    max_per_skeleton: int,
    max_combinations: int,
    seed: str,
    domains: Sequence[str],
) -> BuildResult:
    problems = lexicon.validate()
    if problems:
        raise ValueError("lexicon is invalid: " + "; ".join(problems[:5]))

    result = BuildResult(output=str(output), max_per_skeleton=max_per_skeleton)
    selected = [
        candidate for candidate in lexicon.DOMAINS if not domains or candidate.code in domains
    ]
    if not selected:
        raise ValueError(f"no domains selected; known: {', '.join(lexicon.known_domains())}")

    per_domain: Counter[str] = Counter()
    skeletons: defaultdict[str, set[str]] = defaultdict(set)
    per_skeleton: defaultdict[tuple[str, str], int] = defaultdict(int)
    written: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []

    for spec in selected:
        for index, frame in enumerate(spec.frames):
            for assignment in assignments(frame, limit=max_combinations, seed=seed):
                result.combinations_considered += 1
                source, target = lexicon.fill(frame, assignment)
                if lexicon.unresolved_slots(source) or lexicon.unresolved_slots(target):
                    result.skipped_unresolved += 1
                    continue
                skeleton = frame_skeleton(source)
                key = (spec.code, skeleton)
                if per_skeleton[key] >= max_per_skeleton:
                    result.skipped_skeleton_cap += 1
                    continue
                pair = (source, target)
                if pair in written:
                    result.skipped_duplicate += 1
                    continue
                written.add(pair)
                per_skeleton[key] += 1
                per_domain[spec.code] += 1
                skeletons[spec.code].add(skeleton)
                rows.append(
                    {
                        source_key: source,
                        target_key: target,
                        "domain": spec.code,
                        "domain_label": spec.label,
                        "frame_index": index,
                        "synthetic": True,
                    }
                )
                if len(result.samples) < 60 and per_skeleton[key] == 1 and index < 3:
                    result.samples.append(
                        {"domain": spec.code, source_key: source, target_key: target}
                    )

    rows.sort(key=lambda row: rank_key(seed, str(row["domain"]), str(row[source_key])))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)

    result.rows_written = len(rows)
    result.per_domain = dict(per_domain.most_common())
    result.distinct_skeletons = {code: len(values) for code, values in sorted(skeletons.items())}
    return result


def domain_list(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    known = set(lexicon.known_domains())
    unknown = [name for name in names if name not in known]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown domain(s) {unknown}; known: {', '.join(sorted(known))}"
        )
    return names


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--source-key", default="ko")
    parser.add_argument("--target-key", default="ja")
    parser.add_argument(
        "--max-per-skeleton",
        type=int,
        default=3,
        help=(
            "rows kept per sentence skeleton, using the audit's definition. "
            "This is the cap that stops a digit-only frame from filling the shard"
        ),
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=400,
        help="slot assignments considered per frame, ranked by a seeded hash",
    )
    parser.add_argument("--seed", default="sion-formulaic-v1")
    parser.add_argument("--domains", type=domain_list, default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build(
            args.output,
            source_key=args.source_key,
            target_key=args.target_key,
            max_per_skeleton=args.max_per_skeleton,
            max_combinations=args.max_combinations,
            seed=args.seed,
            domains=args.domains,
        )
    except (OSError, ValueError) as error:
        print(f"cannot build ({error})", file=sys.stderr)
        return 2

    print(
        f"{result.combinations_considered:,} combinations -> {result.rows_written:,} rows "
        f"(skeleton cap {result.max_per_skeleton})"
    )
    print(
        f"  skipped: skeletonCap={result.skipped_skeleton_cap:,} "
        f"dup={result.skipped_duplicate:,} unresolved={result.skipped_unresolved:,}"
    )
    for code, count in result.per_domain.items():
        distinct = result.distinct_skeletons.get(code, 0)
        print(f"      {code:16} {count:6,} rows over {distinct:5,} skeletons")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
