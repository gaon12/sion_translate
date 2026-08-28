#!/usr/bin/env python3
"""Convert an AI-Hub Korean-English parallel distribution into a training shard.

AI-Hub ships these corpora as a directory tree of ZIP archives, each holding one
large JSON document whose ``data`` array carries the sentence records. Every
record contains **three** renderings, and only two of them are human work:

``ko`` / ``en``
    The reviewed human translation pair. This is what the shard keeps.
``mt``
    A machine translation of the source side, included by AI-Hub so that
    post-editing effort can be studied. Training on it would teach this model to
    imitate whichever engine produced it, so this script never reads the field.

``ko_original`` / ``en_original`` record the side that was written first. They
duplicate the reviewed text often enough to be useless as a second pair, and
where they differ they are the pre-review draft, so they are dropped as well.

Each row keeps its ``domain``, ``subdomain`` and ``style`` labels. The corpus
previously had to be classified by reading samples, because shards recorded only
a source URL and no per-row provenance; these three fields are the cheapest
possible fix for the shards this script writes.

Usage::

    python scripts/data/build_aihub_koen.py \\
        --input "~/Downloads/025.일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터" \\
        --output data/data79_koen.jsonl

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
import zipfile


# AI-Hub wraps the records in a single object; a few distributions ship a bare
# array instead, so both shapes are accepted.
RECORD_CONTAINER_KEYS = ("data", "records", "items")
LABEL_FIELDS = ("domain", "subdomain", "style")
# Never read: `mt` is machine output and the `_original` fields are pre-review.
EXCLUDED_FIELDS = frozenset({"mt", "ko_original", "en_original"})


def iter_archive_documents(root: Path) -> Iterator[tuple[Path, str, object]]:
    """Yield ``(archive, member, parsed JSON)`` for every JSON in the tree."""

    archives = sorted(root.rglob("*.zip")) if root.is_dir() else [root]
    if not archives:
        raise ValueError(f"no ZIP archives found under {root}")
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if name.lower().endswith(".json")]
            if not members:
                raise ValueError(f"{archive} contains no JSON member")
            for member in members:
                with bundle.open(member) as handle:
                    yield archive, member, json.load(handle)


def records_of(document: object) -> list[dict[str, object]]:
    if isinstance(document, list):
        rows = document
    elif isinstance(document, dict):
        for key in RECORD_CONTAINER_KEYS:
            value = document.get(key)
            if isinstance(value, list):
                rows = value
                break
        else:
            raise ValueError(
                "JSON object has no record array; expected one of "
                + ", ".join(RECORD_CONTAINER_KEYS)
            )
    else:
        raise ValueError("JSON document must be an object or an array")
    return [row for row in rows if isinstance(row, dict)]


def build_row(record: dict[str, object], *, keep_labels: bool) -> dict[str, str] | None:
    """Return the human ko/en pair with its labels, or ``None`` when unusable."""

    korean = record.get("ko")
    english = record.get("en")
    if not isinstance(korean, str) or not isinstance(english, str):
        return None
    if not korean.strip() or not english.strip():
        return None
    row: dict[str, str] = {"ko": korean.strip(), "en": english.strip()}
    if keep_labels:
        for field in LABEL_FIELDS:
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                row[field] = value.strip()
    return row


def write_shard(
    path: Path,
    rows: Iterator[dict[str, str]],
) -> tuple[int, int, str]:
    """Durably build the shard and publish it atomically."""

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
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="AI-Hub distribution directory (searched for *.zip) or one archive",
    )
    parser.add_argument("--output", type=Path, required=True, help="shard to write")
    parser.add_argument(
        "--source-entry",
        default="",
        help="registry description recorded in the manifest",
    )
    parser.add_argument("--source-url", default="", help="dataset URL recorded in the manifest")
    parser.add_argument(
        "--drop-labels",
        action="store_true",
        help="omit the domain/subdomain/style fields from each row",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.input.expanduser()
    if not root.exists():
        print(f"input not found: {root}", file=sys.stderr)
        return 2

    domains: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    archives: list[dict[str, object]] = []
    scanned = 0
    unusable = 0

    def rows() -> Iterator[dict[str, str]]:
        nonlocal scanned, unusable
        for archive, member, document in iter_archive_documents(root):
            records = records_of(document)
            emitted = 0
            for record in records:
                scanned += 1
                row = build_row(record, keep_labels=not args.drop_labels)
                if row is None:
                    unusable += 1
                    continue
                label = str(record.get("domain") or "")
                if label:
                    domains[label] += 1
                style = str(record.get("style") or "")
                if style:
                    styles[style] += 1
                emitted += 1
                yield row
            archives.append(
                {
                    "archive": archive.name,
                    "member": member,
                    "records": len(records),
                    "written": emitted,
                }
            )

    try:
        count, size, digest = write_shard(args.output, rows())
    except (ValueError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2

    manifest = {
        "languages": ["ko", "en"],
        "source_entry": args.source_entry,
        "source_url": args.source_url,
        "strategy": "human ko/en pair only; machine `mt` and pre-review `_original` dropped",
        "excluded_fields": sorted(EXCLUDED_FIELDS),
        "scanned_records": scanned,
        "unusable_records": unusable,
        "row_count": count,
        "output_bytes": size,
        "sha256": digest,
        "output": str(args.output.resolve()),
        "archives": archives,
        "domains": dict(domains.most_common()),
        "styles": dict(styles.most_common()),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scanned_records": scanned,
                "row_count": count,
                "unusable_records": unusable,
                "sha256": digest,
                "domains": dict(domains.most_common(8)),
                "styles": dict(styles),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
