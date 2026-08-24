"""Convert NIKL Modu Corpus ZIP archives into foundation-corpus JSONL.

Archive folder names vary by release, but each JSON document uses the same
structure::

    {"id": ..., "metadata": {...},
     "document": [{"id": ..., "utterance"|"paragraph"|"sentence": [{"form": "..."}]}]}

The converter writes **one utterance or paragraph per line**. Writing a whole
document as one line forces foundation preparation to rediscover sentence
boundaries after length filtering can already truncate or reject it. A measured
document-level conversion lost 25.9% of its characters this way.

Empty ``form`` values are skipped. Spoken-language archives contain real entries
that have only a note such as "배경 음악 있음" and no utterance.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterator

from sion_translate.console import configure_stdio

# Different releases label their segments as utterances, paragraphs, or sentences.
SEGMENT_KEYS = ("utterance", "paragraph", "sentence")

# Remove spoken-transcription notation so the model does not learn it as content.
_TRANSCRIPTION_NOISE = re.compile(
    r"&[a-zA-Z-]+\d*&"  # De-identification tags such as &name& and &address&.
    r"|\([^)]{0,40}\)/\([^)]{0,40}\)"  # Paired (spelling)/(pronunciation) notation.
    r"|[{}<>@#*~]"
)
_WHITESPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    """Remove transcription notation and normalize whitespace."""

    text = unicodedata.normalize("NFC", text)
    text = _TRANSCRIPTION_NOISE.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def iter_segments(payload: object) -> Iterator[str]:
    """Yield utterance, paragraph, or sentence text from one JSON document."""

    if not isinstance(payload, dict):
        return
    documents = payload.get("document")
    if not isinstance(documents, list):
        documents = [payload]
    for document in documents:
        if not isinstance(document, dict):
            continue
        for key in SEGMENT_KEYS:
            segments = document.get(key)
            if not isinstance(segments, list):
                continue
            for segment in segments:
                if isinstance(segment, dict):
                    form = segment.get("form")
                elif isinstance(segment, str):
                    form = segment
                else:
                    form = None
                if isinstance(form, str) and form.strip():
                    yield form


def extract_archive(
    archive: Path,
    sink,
    *,
    minimum_characters: int,
    maximum_characters: int,
    seen: set[int] | None,
) -> Counter:
    stats: Counter = Counter()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if not member.lower().endswith(".json"):
                continue
            try:
                payload = json.loads(bundle.read(member).decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                stats["unreadable_members"] += 1
                continue
            stats["members"] += 1
            for form in iter_segments(payload):
                stats["segments"] += 1
                text = clean(form)
                if len(text) < minimum_characters:
                    stats["too_short"] += 1
                    continue
                if len(text) > maximum_characters:
                    stats["too_long"] += 1
                    continue
                if seen is not None:
                    digest = hash(text)
                    if digest in seen:
                        stats["duplicate"] += 1
                        continue
                    seen.add(digest)
                sink.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                stats["written"] += 1
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert NIKL Modu Corpus ZIP archives to foundation JSONL"
    )
    parser.add_argument("--archive", nargs="+", required=True, help="input ZIP paths")
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument("--minimum-characters", type=int, default=8)
    parser.add_argument("--maximum-characters", type=int, default=4000)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="disable deduplication (default: deduplicate within this conversion)",
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[int] | None = None if args.keep_duplicates else set()
    totals: Counter = Counter()
    with output.open("w", encoding="utf-8") as sink:
        for name in args.archive:
            archive = Path(name)
            if not archive.is_file():
                print(f"  skipped missing archive: {archive}", flush=True)
                continue
            stats = extract_archive(
                archive,
                sink,
                minimum_characters=args.minimum_characters,
                maximum_characters=args.maximum_characters,
                seen=seen,
            )
            totals.update(stats)
            print(
                f"  {archive.name}: segments {stats['segments']:,} -> "
                f"written {stats['written']:,} (too short {stats['too_short']:,} / "
                f"too long {stats['too_long']:,} / duplicates {stats['duplicate']:,})",
                flush=True,
            )
    print(
        f"{output}: {totals['written']:,} total rows, {output.stat().st_size / 1e9:.2f} GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
