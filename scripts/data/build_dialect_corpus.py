#!/usr/bin/env python3
"""Generate a regional-dialect corpus from standard parallel pairs.

방언 is the worst category in the diagnostic - chrF 9.89 against 3,409 rows - and
the ranking of categories by score is the ranking by row count, so this is a
coverage problem. Regional sentence endings are a closed, systematic class, so
they can be generated deterministically rather than collected.

Each output row keeps the standard pair and adds the dialect rendering of one
side::

    {"kd": "밥 먹었나?", "ko": "밥 먹었니?", "ja": "ご飯食べた?",
     "dialect_language": "ko", "dialect_region": "gyeongsang", "synthetic": true}

``kd``/``jd`` are source-only languages, like ``kj`` for 한본어. Registering them
in ``data.source_only_languages`` trains kd->ko and kd->ja while never training
ko->kd, because the measured failure is *understanding* dialect input; teaching
the model to answer a standard prompt in dialect is a different feature and needs
a tag scheme rather than a silent direction.

Dialect is a spoken register, so the source pool must be speech. Feeding it
formal written prose produces sentences no speaker would say (``航空会社は、毎年約
20,000 部の手紙を顧客に送っとります``), which is why ``--input`` should point at
transcription or dialogue shards.

One file is written per dialect language, named after ``--output``:
``data/synthetic_dialect.jsonl`` yields ``synthetic_dialect_ko.jsonl`` (``kd``)
and ``synthetic_dialect_ja.jsonl`` (``jd``). A single mixed file would leave
every row missing one dialect field, and the preservation gate needs one
source key per file.

Usage::

    python scripts/data/build_dialect_corpus.py \
        --output data/synthetic_dialect.jsonl \
        --report reports/dialect-build.json \
        --max-per-region 4000 --max-per-frame 3 \
        data/data9.jsonl data/data10.jsonl

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

_LEXICON_PATH = Path(__file__).resolve().parent / "dialect_lexicon.py"
_SPEC = importlib.util.spec_from_file_location("dialect_lexicon", _LEXICON_PATH)
assert _SPEC is not None and _SPEC.loader is not None
lexicon = importlib.util.module_from_spec(_SPEC)
sys.modules["dialect_lexicon"] = lexicon
_SPEC.loader.exec_module(lexicon)

# Field names for the dialect rendering, per source language.
DIALECT_KEYS: dict[str, str] = {"ko": "kd", "ja": "jd"}

_DIGITS = re.compile(r"\d+")
_QUOTED = re.compile(r"[\"'“”「」『』]([^\"'“”「」『』]{1,60})[\"'“”「」『』]")
# Markup and interface scaffolding. A UI string is not speech, so dialect must
# not be applied to it.
_MARKUP = re.compile(r"[{}<>\[\]|]|\$[A-Za-z_]|%[sd]\b|·|【|｜")


@dataclass
class BuildResult:
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    pairs_read: int = 0
    pairs_eligible: int = 0
    rows_written: int = 0
    skipped_markup: int = 0
    skipped_length: int = 0
    skipped_script: int = 0
    skipped_no_rule: int = 0
    skipped_frame_cap: int = 0
    skipped_region_cap: int = 0
    skipped_duplicate: int = 0
    per_region: dict[str, int] = field(default_factory=dict)
    per_language: dict[str, int] = field(default_factory=dict)
    endings_used: dict[str, int] = field(default_factory=dict)
    samples: list[dict[str, str]] = field(default_factory=list)


def frame(text: str) -> str:
    """Sentence skeleton: quoted spans blanked, digits collapsed.

    The same definition ``audit_generated_shards.py`` uses, so the frame cap here
    and the audit downstream agree on what counts as a repeated sentence.
    """

    return _DIGITS.sub("#", _QUOTED.sub("<Q>", text))


def rank_key(seed: str, *parts: str) -> str:
    payload = "\x00".join((seed,) + parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


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
    max_per_region: int,
    max_per_frame: int,
    seed: str,
    regions: Sequence[str],
) -> BuildResult:
    result = BuildResult(inputs=[str(path) for path in paths])

    languages = {source_language: source_key, target_language: target_key}
    selected: list[lexicon.DialectProfile] = []
    for language in languages:
        for profile in lexicon.profiles_for(language):
            if regions and profile.code not in regions:
                continue
            selected.append(profile)
    if not selected:
        raise ValueError(
            f"no dialect profiles for languages {sorted(languages)}; "
            f"configured: {lexicon.known_languages()}"
        )

    # Candidates are gathered first and chosen afterwards, so the caps apply to a
    # ranked pool rather than to whatever happened to come first in the file.
    candidates: list[tuple[str, lexicon.DialectProfile, str, str, str, str]] = []
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
        key = (canonical_text(source), canonical_text(target))
        if key in seen_pairs:
            result.skipped_duplicate += 1
            continue
        seen_pairs.add(key)
        result.pairs_eligible += 1

        for profile in selected:
            text = source if profile.language == source_language else target
            rewritten = lexicon.to_dialect(text, profile)
            if rewritten is None:
                result.skipped_no_rule += 1
                continue
            dialect, ending = rewritten
            candidates.append((profile.code, profile, dialect, ending, source, target))

    candidates.sort(key=lambda item: rank_key(seed, item[0], item[2]))

    per_region: Counter[str] = Counter()
    per_frame: defaultdict[tuple[str, str], int] = defaultdict(int)
    endings: Counter[str] = Counter()
    per_language: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for code, profile, dialect, ending, source, target in candidates:
        if per_region[code] >= max_per_region:
            result.skipped_region_cap += 1
            continue
        frame_key = (code, frame(dialect))
        if per_frame[frame_key] >= max_per_frame:
            result.skipped_frame_cap += 1
            continue
        per_region[code] += 1
        per_frame[frame_key] += 1
        endings[ending] += 1
        per_language[profile.language] += 1
        row = {
            DIALECT_KEYS[profile.language]: dialect,
            source_key: source,
            target_key: target,
            "dialect_language": profile.language,
            "dialect_region": code,
            "dialect_label": profile.label,
            "synthetic": True,
        }
        rows.append(row)
        if len(result.samples) < 40 and per_region[code] <= 2:
            result.samples.append(
                {
                    "region": code,
                    "label": profile.label,
                    "standard": source if profile.language == source_language else target,
                    "dialect": dialect,
                }
            )

    rows.sort(
        key=lambda row: rank_key(seed, row["dialect_region"], str(row.get("kd") or row.get("jd")))
    )

    # One file per dialect language. A single mixed file would leave every row
    # missing one of the two dialect fields, which no downstream check can read:
    # the preservation gate needs one source key for the whole file.
    output.parent.mkdir(parents=True, exist_ok=True)
    by_language: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[str(row["dialect_language"])].append(row)

    for language, language_rows in sorted(by_language.items()):
        path = output.with_name(f"{output.stem}_{language}{output.suffix}")
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in language_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(path)
        result.outputs.append(str(path))

    result.rows_written = len(rows)
    result.per_region = dict(per_region.most_common())
    result.per_language = dict(per_language.most_common())
    result.endings_used = dict(endings.most_common(40))
    return result


def region_list(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    known = {profile.code for profile in lexicon.all_profiles()}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown region(s) {unknown}; known: {', '.join(sorted(known))}"
        )
    return names


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="standard parallel JSONL shards")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--source-key", default="ko")
    parser.add_argument("--target-key", default="ja")
    parser.add_argument("--source-language", default="ko")
    parser.add_argument("--target-language", default="ja")
    parser.add_argument("--min-chars", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=60)
    parser.add_argument(
        "--max-per-region",
        type=int,
        default=4000,
        help="cap per region so a variety with common endings cannot dominate",
    )
    parser.add_argument(
        "--max-per-frame",
        type=int,
        default=3,
        help="cap rows sharing a sentence skeleton, matching the audit definition",
    )
    parser.add_argument("--seed", default="sion-dialect-v1")
    parser.add_argument(
        "--regions",
        type=region_list,
        default=[],
        help="restrict to these region codes, comma separated. omit for all",
    )
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
            max_per_region=args.max_per_region,
            max_per_frame=args.max_per_frame,
            seed=args.seed,
            regions=args.regions,
        )
    except (OSError, ValueError) as error:
        print(f"cannot build ({error})", file=sys.stderr)
        return 2

    print(
        f"{result.pairs_read:,} pairs read, {result.pairs_eligible:,} eligible "
        f"-> {result.rows_written:,} rows"
    )
    for written in result.outputs:
        print(f"  wrote {written}")
    print(
        f"  skipped: markup={result.skipped_markup:,} length={result.skipped_length:,} "
        f"script={result.skipped_script:,} noRule={result.skipped_no_rule:,} "
        f"frameCap={result.skipped_frame_cap:,} regionCap={result.skipped_region_cap:,}"
    )
    for language, count in result.per_language.items():
        print(f"  {language}: {count:,}")
    for code, count in result.per_region.items():
        print(f"      {code:12} {count:6,}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
