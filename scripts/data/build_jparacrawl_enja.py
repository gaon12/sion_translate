#!/usr/bin/env python3
"""Build a quality-filtered, evenly sampled JParaCrawl English-Japanese shard."""

from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import re
from typing import Iterator
import zipfile


SPACE_RE = re.compile(r"\s+")
LATIN_RE = re.compile(r"[A-Za-z]")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def clean_pair(en: str, ja: str) -> tuple[str, str] | None:
    en = SPACE_RE.sub(" ", en).strip()
    ja = SPACE_RE.sub(" ", ja).strip()
    if not (4 <= len(en) <= 2_000 and 2 <= len(ja) <= 2_000):
        return None
    if not LATIN_RE.search(en) or not JAPANESE_RE.search(ja):
        return None
    shorter, longer = sorted((len(en), len(ja)))
    if longer / shorter > 5:
        return None
    return en, ja


def members(archive: zipfile.ZipFile) -> tuple[str, str]:
    names = archive.namelist()
    english = [name for name in names if name.endswith(".en")]
    japanese = [name for name in names if name.endswith(".ja")]
    if len(english) != 1 or len(japanese) != 1:
        raise ValueError(f"unexpected archive members: en={english}, ja={japanese}")
    return english[0], japanese[0]


def pairs(archive_path: Path) -> Iterator[tuple[str, str] | None]:
    with zipfile.ZipFile(archive_path) as archive:
        en_member, ja_member = members(archive)
        with (
            io.TextIOWrapper(
                archive.open(en_member), encoding="utf-8", errors="replace"
            ) as en_file,
            io.TextIOWrapper(
                archive.open(ja_member), encoding="utf-8", errors="replace"
            ) as ja_file,
        ):
            sentinel = object()
            en_iterator = iter(en_file)
            ja_iterator = iter(ja_file)
            while True:
                en = next(en_iterator, sentinel)
                ja = next(ja_iterator, sentinel)
                if en is sentinel and ja is sentinel:
                    return
                if en is sentinel or ja is sentinel:
                    raise ValueError("unaligned JParaCrawl line counts")
                yield clean_pair(en, ja)


def build(archive: Path, output: Path, target: int) -> dict[str, object]:
    scanned = 0
    valid = 0
    for pair in pairs(archive):
        scanned += 1
        if pair is not None:
            valid += 1
    if target > valid:
        raise ValueError(f"target {target:,} exceeds {valid:,} valid rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    digest = sha256()
    output_bytes = 0
    selected = 0
    valid_index = 0
    next_selected = 0 if target else None
    with temporary.open("wb") as destination:
        for pair in pairs(archive):
            if pair is None:
                continue
            if next_selected is not None and valid_index == next_selected:
                encoded = (
                    json.dumps(
                        {"en": pair[0], "ja": pair[1]}, ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                ).encode("utf-8")
                destination.write(encoded)
                digest.update(encoded)
                output_bytes += len(encoded)
                selected += 1
                next_selected = (selected * valid) // target if selected < target else None
            valid_index += 1
    if selected != target:
        raise AssertionError(f"selected {selected:,}, expected {target:,}")
    temporary.replace(output)
    report = {
        "source": "JParaCrawl v3.0 via OPUS mirror",
        "source_url": "https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/",
        "mirror_url": "https://object.pouta.csc.fi/OPUS-JParaCrawl/v3.0/moses/en-ja.txt.zip",
        "license": "research and development use only; commercial use requires contacting NTT",
        "languages": ["en", "ja"],
        "strategy": "script and length quality filters plus deterministic even sampling",
        "scanned_rows": scanned,
        "valid_rows": valid,
        "rejected_rows": scanned - valid,
        "row_count": selected,
        "output_bytes": output_bytes,
        "sha256": digest.hexdigest(),
        "output": str(output.resolve()),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/data65_enja.jsonl"))
    parser.add_argument("--target", type=int, default=3_743_113)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.archive.resolve(), args.output.resolve(), args.target),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
