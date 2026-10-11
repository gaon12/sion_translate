#!/usr/bin/env python3
"""Add Wuthering Waves 2.2 English voice transcripts to ko/ja data13.

The existing ko/ja rows are mapped to the Japanese/Korean archive intersection
in Japanese archive order. English is then joined by the same normalized
speaker-directory and voice-file ID. No text similarity or row-offset heuristic
is used.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any


_LANGUAGE_PREFIX_RE = re.compile(r"^(?:ja|ko|en)_", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
SOURCE = "AI-Hobbyist WutheringWaves_Datasets 2.2 JP/KR/EN voice transcripts"
SOURCE_REVISION = "WutheringWaves2.2:JP+KR+EN:2025-04-22"


def canonical_member_id(member: str) -> str:
    """Normalize an archive/local .lab path to a language-neutral voice ID."""

    normalized = member.replace("\\", "/")
    parts = normalized.split("/", 1)
    if len(parts) != 2 or not parts[1].lower().endswith(".lab"):
        raise ValueError(f"invalid Wuthering Waves lab member: {member}")
    relative = PurePosixPath(parts[1])
    filename = _LANGUAGE_PREFIX_RE.sub("", relative.name, count=1)
    if filename == relative.name:
        raise ValueError(f"missing language prefix in lab member: {member}")
    return str(relative.with_name(filename))


def load_manifest(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"manifest must be a JSON string array: {path}")
    identifiers = [canonical_member_id(item) for item in value]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"manifest contains duplicate normalized IDs: {path}")
    return identifiers


def load_english(english_dir: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for directory, _, filenames in os.walk(english_dir):
        for filename in filenames:
            if not filename.lower().endswith(".lab"):
                continue
            path = Path(directory) / filename
            relative = path.relative_to(english_dir).as_posix()
            identifier = canonical_member_id(f"English/{relative}")
            if identifier in transcripts:
                raise ValueError(f"duplicate English normalized ID: {identifier}")
            text = _SPACE_RE.sub(" ", path.read_text(encoding="utf-8-sig")).strip()
            if not text:
                raise ValueError(f"empty English transcript: {path}")
            transcripts[identifier] = text
    return transcripts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("ko"), str):
                raise ValueError(f"invalid ko/ja JSONL row {line_number}")
            if not isinstance(row.get("ja"), str):
                raise ValueError(f"invalid ko/ja JSONL row {line_number}")
            rows.append(row)
    return rows


def add_english_rows(
    rows: list[dict[str, Any]],
    japanese_ids: list[str],
    korean_ids: list[str],
    english: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join existing rows with English by the exact three-way voice ID."""

    korean_set = set(korean_ids)
    bilingual_ids = [identifier for identifier in japanese_ids if identifier in korean_set]
    if len(bilingual_ids) != len(rows):
        raise ValueError(
            "existing row count does not match JP/KR archive intersection: "
            f"rows={len(rows)} intersection={len(bilingual_ids)}"
        )

    output: list[dict[str, Any]] = []
    for row, identifier in zip(rows, bilingual_ids, strict=True):
        english_text = english.get(identifier)
        if english_text is None:
            continue
        relative = PurePosixPath(identifier)
        output.append(
            {
                "ko": row["ko"],
                "en": english_text,
                "ja": row["ja"],
                "source": SOURCE,
                "resource_id": identifier,
                "document_id": f"wutheringwaves2.2/{relative.parent}",
                "family_id": f"wutheringwaves2.2:{identifier}",
                "domain": "game",
                "original_direction": "parallel_voice_id",
                "source_revision": SOURCE_REVISION,
            }
        )

    stats = {
        "japanese_ids": len(japanese_ids),
        "korean_ids": len(korean_ids),
        "english_ids": len(english),
        "jp_kr_intersection": len(bilingual_ids),
        "three_way_intersection": len(output),
        "jp_kr_without_english": len(bilingual_ids) - len(output),
    }
    return output, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> tuple[int, int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    digest = sha256()
    size = 0
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            handle.write(encoded)
            digest.update(encoded)
            size += len(encoded)
    temporary.replace(path)
    return len(rows), size, digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--english-dir", type=Path, required=True)
    parser.add_argument("--jp-manifest", type=Path, required=True)
    parser.add_argument("--kr-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input.resolve())
    japanese_ids = load_manifest(args.jp_manifest.resolve())
    korean_ids = load_manifest(args.kr_manifest.resolve())
    english = load_english(args.english_dir.resolve())
    output, stats = add_english_rows(rows, japanese_ids, korean_ids, english)
    count, size, digest = write_jsonl(args.output.resolve(), output)
    print(
        json.dumps(
            {
                **stats,
                "output_rows": count,
                "output_bytes": size,
                "sha256": digest,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
