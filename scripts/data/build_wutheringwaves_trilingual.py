#!/usr/bin/env python3
"""Rebuild Wuthering Waves ko/en/ja rows from exact shared voice IDs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any


LANGUAGE_PREFIX_RE = re.compile(r"^(?:ja|ko|en)_", flags=re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
SOURCE = "AI-Hobbyist WutheringWaves_Datasets 2.2 JP/KR/EN voice transcripts"
SOURCE_REVISION = "WutheringWaves2.2:JP+KR+EN:2025-04-22"


def canonical_id(path: Path, root: Path) -> str:
    relative = PurePosixPath(path.relative_to(root).as_posix())
    filename = LANGUAGE_PREFIX_RE.sub("", relative.name, count=1)
    if filename == relative.name:
        raise ValueError(f"missing language prefix: {path}")
    return str(relative.with_name(filename))


def load_language(root: Path) -> tuple[list[str], dict[str, str]]:
    ordered: list[str] = []
    texts: dict[str, str] = {}
    for directory, directories, filenames in os.walk(root):
        directories.sort()
        for filename in sorted(filenames):
            if not filename.lower().endswith(".lab"):
                continue
            path = Path(directory) / filename
            identifier = canonical_id(path, root)
            if identifier in texts:
                raise ValueError(f"duplicate normalized ID: {identifier}")
            text = SPACE_RE.sub(" ", path.read_text(encoding="utf-8-sig")).strip()
            if not text:
                raise ValueError(f"empty transcript: {path}")
            ordered.append(identifier)
            texts[identifier] = text
    return ordered, texts


def build(
    japanese_root: Path, korean_root: Path, english_root: Path, output: Path
) -> dict[str, Any]:
    japanese_order, japanese = load_language(japanese_root)
    _, korean = load_language(korean_root)
    _, english = load_language(english_root)
    shared = [
        identifier
        for identifier in japanese_order
        if identifier in korean and identifier in english
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    digest = sha256()
    output_bytes = 0
    with temporary.open("wb") as handle:
        for identifier in shared:
            relative = PurePosixPath(identifier)
            row = {
                "ko": korean[identifier],
                "en": english[identifier],
                "ja": japanese[identifier],
                "source": SOURCE,
                "resource_id": identifier,
                "document_id": f"wutheringwaves2.2/{relative.parent}",
                "family_id": f"wutheringwaves2.2:{identifier}",
                "domain": "game",
                "original_direction": "parallel_voice_id",
                "source_revision": SOURCE_REVISION,
            }
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            handle.write(encoded)
            digest.update(encoded)
            output_bytes += len(encoded)
    temporary.replace(output)
    return {
        "japanese_ids": len(japanese),
        "korean_ids": len(korean),
        "english_ids": len(english),
        "three_way_intersection": len(shared),
        "output_rows": len(shared),
        "output_bytes": output_bytes,
        "sha256": digest.hexdigest(),
        "output": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--japanese-root", type=Path, required=True)
    parser.add_argument("--korean-root", type=Path, required=True)
    parser.add_argument("--english-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.japanese_root.resolve(),
        args.korean_root.resolve(),
        args.english_root.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
