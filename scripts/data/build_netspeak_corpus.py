#!/usr/bin/env python3
"""Generate an internet-register corpus by rewriting both sides of a pair.

The corpus holds no internet register at all and 구어 scores chrF 30.45. It does
hold 2.4M spoken rows, but transcribed speech is standard-language and
sentence-complete: ``ㅋㅋ``, ``ㄹㅇ``, ``w`` and ``草`` appear nowhere.

Unlike ``build_dialect_corpus.py``, this writes an ordinary ``ko``/``ja`` pair
with no new language tag. Producing casual Japanese from casual Korean is the
wanted capability, not a hazard, so both directions are fine. What *is* a hazard
is a one-sided rewrite, so :func:`netspeak_lexicon.to_netspeak` refuses unless
both sides changed - see that module for why.

Rows carry the style that produced them::

    {"ko": "이거 ㄹㅇ 대박이다ㅋㅋ", "ja": "これマジでやばいw",
     "net_style": "abbreviation_laughter", "net_rules": "substitution laughter",
     "synthetic": true}

Usage::

    python scripts/data/build_netspeak_corpus.py \
        --output data/synthetic_netspeak.jsonl \
        --report reports/netspeak-build.json \
        --max-per-style 6000 --max-per-frame 3 \
        data/data29.jsonl data/data12.jsonl data/data33.jsonl data/data10.jsonl

Exit codes: 0 written, 2 bad input.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

from sion_translate.data.quality import canonical_text
from sion_translate.scripts_registry import has_foreign_script

_LEXICON_PATH = Path(__file__).resolve().parent / "netspeak_lexicon.py"
_SPEC = importlib.util.spec_from_file_location("netspeak_lexicon", _LEXICON_PATH)
assert _SPEC is not None and _SPEC.loader is not None
lexicon = importlib.util.module_from_spec(_SPEC)
sys.modules["netspeak_lexicon"] = lexicon
_SPEC.loader.exec_module(lexicon)

_DIGITS = re.compile(r"\d+")
_QUOTED = re.compile(r"[\"'“”「」『』]([^\"'“”「」『』]{1,60})[\"'“”「」『』]")
# Interface scaffolding is not something anyone types in a chat window.
_MARKUP = re.compile(r"[{}<>\[\]|]|\$[A-Za-z_]|%[sd]\b|·|【|｜")


@dataclass
class BuildResult:
    inputs: list[str] = field(default_factory=list)
    output: str = ""
    pairs_read: int = 0
    pairs_eligible: int = 0
    rows_written: int = 0
    skipped_markup: int = 0
    skipped_length: int = 0
    skipped_script: int = 0
    skipped_not_casual: int = 0
    skipped_no_rule: int = 0
    skipped_style_cap: int = 0
    skipped_frame_cap: int = 0
    skipped_duplicate: int = 0
    per_style: dict[str, int] = field(default_factory=dict)
    per_rule: dict[str, int] = field(default_factory=dict)
    samples: list[dict[str, str]] = field(default_factory=list)


def frame(text: str) -> str:
    """Sentence skeleton, using the audit's definition so the caps agree."""

    return _DIGITS.sub("#", _QUOTED.sub("<Q>", text))


def rank_key(seed: str, *parts: str) -> str:
    payload = "\x00".join((seed,) + parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def variant_of(seed: str, *parts: str) -> int:
    """Deterministic small integer, used to pick which marker a row gets."""

    return int(rank_key(seed, *parts)[:8], 16)


def read_pairs(
    paths: Sequence[Path],
    *,
    source_key: str,
    target_key: str,
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
                    yield source.strip(), target.strip()


def build(
    paths: Sequence[Path],
    output: Path,
    *,
    source_key: str,
    target_key: str,
    source_language: str,
    target_language: str,
    min_chars: int,
    max_chars: int,
    max_per_style: int,
    max_per_frame: int,
    seed: str,
    styles: Sequence[str],
) -> BuildResult:
    result = BuildResult(inputs=[str(path) for path in paths], output=str(output))

    selected = [candidate for candidate in lexicon.STYLES if not styles or candidate.code in styles]
    if not selected:
        raise ValueError(f"no styles selected; known: {', '.join(lexicon.known_styles())}")

    candidates: list[tuple[str, str, str, tuple[str, ...]]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for source, target in read_pairs(paths, source_key=source_key, target_key=target_key):
        result.pairs_read += 1
        if not source or not target:
            continue
        if _MARKUP.search(source) or _MARKUP.search(target):
            result.skipped_markup += 1
            continue
        if not (min_chars <= len(source) <= max_chars and min_chars <= len(target) <= max_chars):
            result.skipped_length += 1
            continue
        if has_foreign_script(source, [source_language]) or has_foreign_script(
            target, [target_language]
        ):
            result.skipped_script += 1
            continue
        if not lexicon.is_casual_pair(source, target):
            result.skipped_not_casual += 1
            continue
        key = (canonical_text(source), canonical_text(target))
        if key in seen_pairs:
            result.skipped_duplicate += 1
            continue
        seen_pairs.add(key)
        result.pairs_eligible += 1

        for style_ in selected:
            variant = variant_of(seed, style_.code, source)
            rewritten = lexicon.to_netspeak(source, target, style_, variant=variant)
            if rewritten is None:
                result.skipped_no_rule += 1
                continue
            new_source, new_target, fired = rewritten
            candidates.append((style_.code, new_source, new_target, fired))

    candidates.sort(key=lambda item: rank_key(seed, item[0], item[1]))

    per_style: Counter[str] = Counter()
    per_rule: Counter[str] = Counter()
    per_frame: defaultdict[tuple[str, str], int] = defaultdict(int)
    written: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []

    for code, source, target, fired in candidates:
        if per_style[code] >= max_per_style:
            result.skipped_style_cap += 1
            continue
        frame_key = (code, frame(source))
        if per_frame[frame_key] >= max_per_frame:
            result.skipped_frame_cap += 1
            continue
        # Two styles can converge on the same output, for instance when both fall
        # back to laughter. Keep the pair once.
        pair = (canonical_text(source), canonical_text(target))
        if pair in written:
            result.skipped_duplicate += 1
            continue
        written.add(pair)
        per_style[code] += 1
        per_frame[frame_key] += 1
        for rule in fired:
            per_rule[rule] += 1
        rows.append(
            {
                source_key: source,
                target_key: target,
                "net_style": code,
                "net_rules": " ".join(fired),
                "synthetic": True,
            }
        )
        if len(result.samples) < 40 and per_style[code] <= 3:
            result.samples.append({"style": code, source_key: source, target_key: target})

    rows.sort(key=lambda row: rank_key(seed, str(row["net_style"]), str(row[source_key])))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)

    result.rows_written = len(rows)
    result.per_style = dict(per_style.most_common())
    result.per_rule = dict(per_rule.most_common())
    return result


def style_list(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    known = set(lexicon.known_styles())
    unknown = [name for name in names if name not in known]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown style(s) {unknown}; known: {', '.join(sorted(known))}"
        )
    return names


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
    parser.add_argument("--source-language", default="ko")
    parser.add_argument("--target-language", default="ja")
    parser.add_argument("--min-chars", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=45)
    parser.add_argument("--max-per-style", type=int, default=6000)
    parser.add_argument("--max-per-frame", type=int, default=3)
    parser.add_argument("--seed", default="sion-netspeak-v1")
    parser.add_argument("--styles", type=style_list, default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        print(f"not a file: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        result = build(
            args.inputs,
            args.output,
            source_key=args.source_key,
            target_key=args.target_key,
            source_language=args.source_language,
            target_language=args.target_language,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            max_per_style=args.max_per_style,
            max_per_frame=args.max_per_frame,
            seed=args.seed,
            styles=args.styles,
        )
    except (OSError, ValueError) as error:
        print(f"cannot build ({error})", file=sys.stderr)
        return 2

    print(
        f"{result.pairs_read:,} pairs read, {result.pairs_eligible:,} casual "
        f"-> {result.rows_written:,} rows"
    )
    print(
        f"  skipped: markup={result.skipped_markup:,} length={result.skipped_length:,} "
        f"script={result.skipped_script:,} notCasual={result.skipped_not_casual:,} "
        f"noRule={result.skipped_no_rule:,} styleCap={result.skipped_style_cap:,} "
        f"frameCap={result.skipped_frame_cap:,} dup={result.skipped_duplicate:,}"
    )
    for code, count in result.per_style.items():
        print(f"      {code:24} {count:6,}")
    print(f"  rules fired: {result.per_rule}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
