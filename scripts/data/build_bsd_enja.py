#!/usr/bin/env python3
"""Convert the Business Scene Dialogue corpus into one English-Japanese shard.

BSD is written and translated by people, one business scene at a time: meetings,
phone calls, presentations, training, face-to-face and general chatting. Half the
scenarios were written in Japanese first and half in English, so neither side is
a translation artefact of the other throughout.

The corpus fills a register this corpus has nothing else for. The English-
Japanese half is web crawl, film subtitles, encyclopedia text and software
strings; business conversation appears nowhere in it.

**One row per conversation, not per turn.** The turns of a scene are emitted as
list-valued ``en``/``ja`` keys, which the preparer expands into one pair per turn
while assigning the whole record to a single split. Emitting turns as separate
rows would scatter one conversation across train, validation and test, which is
the leakage the data plan explicitly forbids for conversational data.

**Licence: CC BY-NC-SA.** Non-commercial only, and share-alike. That is stricter
than the rest of this corpus, so it is recorded in the manifest and must be
carried into any redistribution or model release built on it.

Usage::

    python scripts/data/build_bsd_enja.py --input ~/BSD --output data/data81_enja.jsonl

Exit codes: 0 success, 2 bad input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


DEFAULT_SPLITS = ("train", "dev", "test")
LICENCE = "CC BY-NC-SA (non-commercial, share-alike)"


def load_conversations(
    root: Path, splits: Sequence[str]
) -> Iterator[tuple[str, dict[str, object]]]:
    for split in splits:
        path = root / f"{split}.json"
        if not path.is_file():
            raise ValueError(f"{path} is missing; expected the BSD repository layout")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, list):
            raise ValueError(f"{path} must hold a JSON array of conversations")
        for conversation in document:
            if isinstance(conversation, dict):
                yield split, conversation


def build_row(conversation: dict[str, object]) -> dict[str, object] | None:
    """Return one record holding every turn of a scene, or ``None`` if unusable."""

    turns = conversation.get("conversation")
    if not isinstance(turns, list):
        return None
    english: list[str] = []
    japanese: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        source = turn.get("en_sentence")
        target = turn.get("ja_sentence")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if not source.strip() or not target.strip():
            continue
        english.append(source.strip())
        japanese.append(target.strip())
    if not english:
        return None
    row: dict[str, object] = {"en": english, "ja": japanese}
    for field, key in (("tag", "scene"), ("title", "title"), ("id", "conversation_id")):
        value = conversation.get(field)
        if isinstance(value, str) and value.strip():
            row[key] = value.strip()
    original = conversation.get("original_language")
    if isinstance(original, str) and original.strip():
        row["original_language"] = original.strip()
    return row


def write_shard(path: Path, rows: Iterator[dict[str, object]]) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                payload = json.dumps(row, ensure_ascii=False) + "\n"
                encoded = payload.encode("utf-8")
                handle.write(payload)
                digest.update(encoded)
                size += len(encoded)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if count == 0:
            raise ValueError("refusing to publish an empty shard")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return count, size, digest.hexdigest()


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="BSD repository directory")
    parser.add_argument("--output", type=Path, required=True, help="shard to write")
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help=f"BSD split to include, repeatable (default: {', '.join(DEFAULT_SPLITS)})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.input.expanduser()
    if not root.is_dir():
        print(f"input directory not found: {root}", file=sys.stderr)
        return 2
    splits = tuple(args.split) or DEFAULT_SPLITS

    scenes: Counter[str] = Counter()
    per_split: Counter[str] = Counter()
    turns = 0
    skipped = 0

    def rows() -> Iterator[dict[str, object]]:
        nonlocal turns, skipped
        for split, conversation in load_conversations(root, splits):
            row = build_row(conversation)
            if row is None:
                skipped += 1
                continue
            turns += len(row["en"])
            per_split[split] += 1
            scene = row.get("scene")
            if isinstance(scene, str):
                scenes[scene] += 1
            yield row

    try:
        count, size, digest = write_shard(args.output, rows())
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2

    manifest = {
        "languages": ["en", "ja"],
        "source_entry": "Business Scene Dialogue corpus (BSD)",
        "source_url": "https://github.com/tsuruoka-lab/BSD",
        "license": LICENCE,
        "strategy": "one record per conversation so every turn shares a split",
        "splits_included": list(splits),
        "conversations": count,
        "turns": turns,
        "skipped_conversations": skipped,
        "row_count": count,
        "output_bytes": size,
        "sha256": digest,
        "output": str(args.output.resolve()),
        "conversations_per_split": dict(per_split),
        "scenes": dict(scenes.most_common()),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "conversations": count,
                "turns": turns,
                "license": LICENCE,
                "scenes": dict(scenes.most_common()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
