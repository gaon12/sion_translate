#!/usr/bin/env python3
"""Fetch and normalize diverse OPUS English-Japanese corpora."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import blake2b, sha256
import io
import json
from pathlib import Path
import re
from typing import Any
from urllib.request import urlretrieve
import zipfile


SPACE_RE = re.compile(r"\s+")
LATIN_RE = re.compile(r"[A-Za-z]")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


@dataclass(frozen=True)
class Corpus:
    name: str
    url: str
    domain: str
    license: str


CORPORA = (
    Corpus(
        "kftt",
        "https://object.pouta.csc.fi/OPUS-KFTT/v1.0/moses/en-ja.txt.zip",
        "professional_encyclopedia",
        "CC BY-SA 3.0",
    ),
    Corpus(
        "wikimedia",
        "https://object.pouta.csc.fi/OPUS-wikimedia/v20260327/moses/en-ja.txt.zip",
        "encyclopedia",
        "Wikimedia source licenses",
    ),
    Corpus(
        "wikimatrix",
        "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/en-ja.txt.zip",
        "encyclopedia_mined",
        "Wikipedia source licenses",
    ),
    Corpus(
        "tatoeba",
        "https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/en-ja.txt.zip",
        "community_conversation",
        "Tatoeba sentence licenses",
    ),
    Corpus(
        "kde4",
        "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-ja.txt.zip",
        "software_localization",
        "upstream KDE licenses",
    ),
    Corpus(
        "mdn",
        "https://object.pouta.csc.fi/OPUS-MDN_Web_Docs/v2023-09-25/moses/en-ja.txt.zip",
        "web_technology",
        "CC BY-SA 2.5 / upstream MDN terms",
    ),
)


def clean_pair(en: str, ja: str) -> tuple[str, str] | None:
    en = SPACE_RE.sub(" ", en).strip()
    ja = SPACE_RE.sub(" ", ja).strip()
    if not (2 <= len(en) <= 6_000 and 1 <= len(ja) <= 6_000):
        return None
    if not LATIN_RE.search(en) or not JAPANESE_RE.search(ja):
        return None
    shorter, longer = sorted((len(en), len(ja)))
    if shorter < 4 and longer > 80:
        return None
    if longer / max(shorter, 1) > 8:
        return None
    return en, ja


def language_members(archive: zipfile.ZipFile) -> tuple[str, str]:
    files = [name for name in archive.namelist() if not name.endswith("/")]
    english = [name for name in files if name.lower().endswith(".en")]
    japanese = [name for name in files if name.lower().endswith(".ja")]
    if len(english) != 1 or len(japanese) != 1:
        raise ValueError(f"expected one .en and .ja member, got en={english}, ja={japanese}")
    return english[0], japanese[0]


def import_corpus(
    corpus: Corpus,
    cache_dir: Path,
    output_dir: Path,
    seen: set[bytes],
) -> dict[str, Any]:
    archive_path = cache_dir / f"{corpus.name}.zip"
    if not archive_path.exists():
        urlretrieve(corpus.url, archive_path)
    output = output_dir / f"enja_opus_{corpus.name}.jsonl"
    temporary = output.with_suffix(output.suffix + ".part")
    digest = sha256()
    rows = 0
    rejected = 0
    duplicates = 0
    bytes_written = 0

    with zipfile.ZipFile(archive_path) as archive:
        en_member, ja_member = language_members(archive)
        with (
            io.TextIOWrapper(
                archive.open(en_member), encoding="utf-8", errors="replace"
            ) as en_file,
            io.TextIOWrapper(
                archive.open(ja_member), encoding="utf-8", errors="replace"
            ) as ja_file,
            temporary.open("wb") as destination,
        ):
            sentinel = object()
            en_iterator = iter(en_file)
            ja_iterator = iter(ja_file)
            while True:
                en = next(en_iterator, sentinel)
                ja = next(ja_iterator, sentinel)
                if en is sentinel and ja is sentinel:
                    break
                if en is sentinel or ja is sentinel:
                    raise ValueError(f"unaligned line counts in {corpus.name}")
                pair = clean_pair(en, ja)
                if pair is None:
                    rejected += 1
                    continue
                key = blake2b((pair[0] + "\0" + pair[1]).encode("utf-8"), digest_size=16).digest()
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                encoded = (
                    json.dumps(
                        {"en": pair[0], "ja": pair[1]}, ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                ).encode("utf-8")
                destination.write(encoded)
                digest.update(encoded)
                rows += 1
                bytes_written += len(encoded)
    temporary.replace(output)
    archive_path.unlink()
    return {
        **asdict(corpus),
        "output": str(output.resolve()),
        "rows": rows,
        "rejected": rejected,
        "cross_source_duplicates": duplicates,
        "bytes": bytes_written,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".codex_work/opus_enja"))
    parser.add_argument("--manifest", type=Path, default=Path("data/enja_opus.manifest.json"))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    seen: set[bytes] = set()
    reports = [import_corpus(corpus, cache_dir, output_dir, seen) for corpus in CORPORA]
    manifest = {
        "languages": ["en", "ja"],
        "source_catalog": "https://opus.nlpl.eu/corpora",
        "rows": sum(report["rows"] for report in reports),
        "corpora": reports,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
