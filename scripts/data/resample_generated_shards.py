#!/usr/bin/env python3
"""Cap how many rows a generated shard may contribute per sentence frame.

``audit_generated_shards.py`` reports that data44/45/48/50/51 restate a small
number of frames thousands of times. Deleting those shards throws away the
lexical content they do carry; keeping them whole lets one frame dominate the
sampler and lets near-duplicates cross the train/holdout boundary.

This tool keeps at most ``--max-per-skeleton`` rows for each sentence frame,
where a frame is the sentence with quoted spans replaced by ``<Q>`` and digits
by ``#`` - the same definition the audit uses. Selection is deterministic: rows
are ranked by a seeded hash of the pair so the same input always yields the same
output, independent of file order.

It also drops rows whose target carries the wrong script, which is how data46
and data51 leak Hangul into Japanese.

Usage::

    python scripts/data/resample_generated_shards.py \
        --max-per-skeleton 8 --output-dir data/resampled data/data44.jsonl

    python scripts/data/resample_generated_shards.py \
        --max-per-skeleton 8 --in-place --report r.json data/data4[3458].jsonl

Exit codes: 0 written, 2 bad input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sys

from sion_translate.data.quality import canonical_text


_QUOTED = re.compile(r"[\"“‘'][^\"”’']{1,120}[\"”’']")
_DIGITS = re.compile(r"\d")
_HANGUL = re.compile(r"[가-힣ㄱ-ㅣ]")
_KANA = re.compile(r"[぀-ヿｦ-ﾟ]")
_HAN = re.compile(r"[一-鿿]")


@dataclass
class ShardResult:
    path: str
    output: str
    rows_in: int = 0
    rows_out: int = 0
    unreadable: int = 0
    dropped_foreign_script: int = 0
    dropped_duplicate: int = 0
    dropped_over_cap: int = 0
    dropped_over_span_cap: int = 0
    skeletons_in: int = 0
    skeletons_out: int = 0
    largest_frame_in: int = 0
    largest_frame_out: int = 0
    quoted_spans_in: int = 0
    largest_span_in: int = 0
    dropped_skeleton_examples: list[tuple[str, int]] = field(default_factory=list)
    dropped_span_examples: list[tuple[str, int]] = field(default_factory=list)


def skeleton(text: str) -> str:
    """Blank quoted spans and digits, matching audit_generated_shards.skeleton."""

    return _DIGITS.sub("#", _QUOTED.sub("<Q>", text))


def target_is_foreign(text: str, target_language: str) -> bool:
    if target_language == "ja":
        return bool(_HANGUL.search(text))
    if target_language == "ko":
        return bool(_KANA.search(text) or _HAN.search(text))
    if target_language == "none":
        return False
    raise ValueError(f"target_language must be ja, ko or none; got {target_language!r}")


def _rank(source: str, target: str, seed: int) -> bytes:
    """A stable per-row ordering key, independent of position in the file."""

    payload = f"{seed}\0{source}\0{target}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def resample_shard(
    path: Path,
    output: Path,
    *,
    max_per_skeleton: int,
    max_per_quoted_span: int | None = None,
    source_key: str = "ko",
    target_key: str = "ja",
    target_language: str = "ja",
    seed: int = 20260730,
) -> ShardResult:
    """Write ``output`` under a per-frame and an optional per-quoted-span cap.

    The two axes degenerate independently. data44 restated 709 rows of a single
    frame, which the frame cap fixes. data48 varies its frames but draws on nine
    distinct quoted spans for 6,171 uses, which only the span cap fixes.
    """

    if max_per_skeleton < 1:
        raise ValueError("max_per_skeleton must be positive")
    if max_per_quoted_span is not None and max_per_quoted_span < 1:
        raise ValueError("max_per_quoted_span must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)
    target_is_foreign("", target_language)

    result = ShardResult(path=str(path), output=str(output))
    candidates: list[tuple[bytes, str, str, tuple[str, ...]]] = []
    seen_pairs: set[bytes] = set()
    skeleton_counts: Counter[str] = Counter()
    span_counts: Counter[str] = Counter()

    with path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            result.rows_in += 1
            try:
                row = json.loads(raw_line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                result.unreadable += 1
                continue
            if not isinstance(row, dict):
                result.unreadable += 1
                continue
            source = row.get(source_key)
            target = row.get(target_key)
            if not isinstance(source, str) or not isinstance(target, str):
                result.unreadable += 1
                continue
            source, target = canonical_text(source), canonical_text(target)
            if not source or not target:
                result.unreadable += 1
                continue
            if target_is_foreign(target, target_language):
                result.dropped_foreign_script += 1
                continue
            digest = _rank(source, target, seed)
            if digest in seen_pairs:
                result.dropped_duplicate += 1
                continue
            seen_pairs.add(digest)
            frame = skeleton(source)
            spans = tuple(_QUOTED.findall(source))
            skeleton_counts[frame] += 1
            for span in spans:
                span_counts[span] += 1
            candidates.append(
                (
                    digest,
                    json.dumps({source_key: source, target_key: target}, ensure_ascii=False),
                    frame,
                    spans,
                )
            )

    result.skeletons_in = len(skeleton_counts)
    result.largest_frame_in = max(skeleton_counts.values(), default=0)
    result.quoted_spans_in = len(span_counts)
    result.largest_span_in = max(span_counts.values(), default=0)

    # Greedy selection in rank order applies both caps uniformly. The rank is a
    # hash of the pair, so the result does not depend on position in the file.
    candidates.sort(key=lambda item: item[0])
    kept: list[str] = []
    kept_frames: Counter[str] = Counter()
    kept_spans: Counter[str] = Counter()
    over_frame_cap: Counter[str] = Counter()
    over_span_cap: Counter[str] = Counter()
    for _, line, frame, spans in candidates:
        if kept_frames[frame] >= max_per_skeleton:
            over_frame_cap[frame] += 1
            result.dropped_over_cap += 1
            continue
        if max_per_quoted_span is not None:
            saturated = [span for span in spans if kept_spans[span] >= max_per_quoted_span]
            if saturated:
                over_span_cap[saturated[0]] += 1
                result.dropped_over_span_cap += 1
                continue
        kept.append(line)
        kept_frames[frame] += 1
        for span in spans:
            kept_spans[span] += 1

    result.rows_out = len(kept)
    result.skeletons_out = len(kept_frames)
    result.largest_frame_out = max(kept_frames.values(), default=0)
    result.dropped_skeleton_examples = [
        (frame[:120], count) for frame, count in over_frame_cap.most_common(3)
    ]
    result.dropped_span_examples = [
        (span[:120], count) for span, count in over_span_cap.most_common(3)
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for line in kept:
            handle.write(line + "\n")
    temporary.replace(output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="JSONL shards to resample")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", help="write resampled shards here")
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite each input after keeping a .orig copy",
    )
    parser.add_argument("--max-per-skeleton", type=int, default=8)
    parser.add_argument(
        "--max-per-quoted-span",
        type=int,
        help="cap reuse of one quoted span; omit to leave spans uncapped",
    )
    parser.add_argument("--source-key", default="ko")
    parser.add_argument("--target-key", default="ja")
    parser.add_argument("--target-language", default="ja", choices=("ja", "ko", "none"))
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--report", help="write the resampling report JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_per_skeleton < 1:
        print("--max-per-skeleton must be positive", file=sys.stderr)
        return 2
    if args.max_per_quoted_span is not None and args.max_per_quoted_span < 1:
        print("--max-per-quoted-span must be positive", file=sys.stderr)
        return 2

    results: list[ShardResult] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if args.in_place:
            backup = path.with_suffix(path.suffix + ".orig")
            if not backup.exists():
                try:
                    backup.write_bytes(path.read_bytes())
                except OSError as error:
                    print(f"{raw_path}: cannot back up ({error})", file=sys.stderr)
                    return 2
            output = path
            source = backup
        else:
            output = Path(args.output_dir) / path.name
            source = path
        try:
            results.append(
                resample_shard(
                    source,
                    output,
                    max_per_skeleton=args.max_per_skeleton,
                    max_per_quoted_span=args.max_per_quoted_span,
                    source_key=args.source_key,
                    target_key=args.target_key,
                    target_language=args.target_language,
                    seed=args.seed,
                )
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"{raw_path}: cannot resample ({error})", file=sys.stderr)
            return 2

    for result in results:
        print(
            f"{Path(result.path).name:26} {result.rows_in:>8,} -> {result.rows_out:>8,} rows  "
            f"frames {result.skeletons_in:>6,}  "
            f"largest frame {result.largest_frame_in:>6,} -> {result.largest_frame_out:>4,}  "
            f"dropped frame={result.dropped_over_cap:,} "
            f"span={result.dropped_over_span_cap:,} "
            f"script={result.dropped_foreign_script:,} dup={result.dropped_duplicate:,}"
        )
        for frame, count in result.dropped_skeleton_examples:
            print(f"      frame -{count:<6,} {frame!r}")
        for span, count in result.dropped_span_examples:
            print(f"      span  -{count:<6,} {span!r}")

    if args.report:
        Path(args.report).write_text(
            json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
