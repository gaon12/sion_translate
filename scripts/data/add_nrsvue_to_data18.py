#!/usr/bin/env python3
"""Add Bible.com NRSVue English verses to an existing ko/ja data18 JSONL.

The existing row set is authoritative. English is joined only by the exact
``resource_id`` (Bible.com ``data-usfm``), and the output is not replaced unless
every row has one unique NRSVue verse with expected Latin text.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from rebuild_verified_parallel import _crawl_bible_version


_LATIN_RE = re.compile(r"[A-Za-z]")
SOURCE = "Bible.com RNKSV 142 + NRSVUE 3523 + 新共同訳 1819"
SOURCE_REVISION = "bible.com:142+3523+1819:2026-08-18"


def add_english_rows(
    rows: Iterable[dict[str, Any]],
    english_verses: dict[str, str],
) -> list[dict[str, Any]]:
    """Return rows with NRSVue text inserted between ko and ja.

    Bible.com occasionally exposes one English span for multiple numbered
    verses (currently JAS.1.7+JAS.1.8). In that case the matching ko/ja rows are
    merged rather than duplicating or heuristically splitting the English.
    """

    source_rows = list(rows)
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    missing_ids: list[str] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    row_indexes: dict[str, int] = {}
    for index, row in enumerate(source_rows):
        verse_id = row.get("resource_id")
        if not isinstance(verse_id, str) or not verse_id:
            raise ValueError("data18 row is missing a string resource_id")
        if verse_id in seen_ids:
            raise ValueError(f"duplicate resource_id in data18: {verse_id}")
        seen_ids.add(verse_id)
        if not isinstance(row.get("ko"), str) or not isinstance(row.get("ja"), str):
            raise ValueError(f"data18 row is missing ko/ja text: {verse_id}")
        rows_by_id[verse_id] = row
        row_indexes[verse_id] = index

    compound_by_component: dict[str, tuple[str, list[str], str]] = {}
    for english_id, english in english_verses.items():
        if "+" not in english_id:
            continue
        components = english_id.split("+")
        for component in components:
            if component in compound_by_component:
                raise ValueError(f"ambiguous compound NRSVue verse component: {component}")
            compound_by_component[component] = (english_id, components, english)

    consumed_ids: set[str] = set()
    for row in source_rows:
        verse_id = str(row["resource_id"])
        if verse_id in consumed_ids:
            continue
        english = english_verses.get(verse_id)
        merged_row = row
        output_id = verse_id
        if not english:
            compound = compound_by_component.get(verse_id)
            if compound is not None:
                output_id, components, english = compound
                component_rows = [rows_by_id.get(component) for component in components]
                if any(component_row is None for component_row in component_rows):
                    missing_ids.extend(
                        component
                        for component, value in zip(components, component_rows, strict=True)
                        if value is None
                    )
                    continue
                indexes = [row_indexes[component] for component in components]
                if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
                    raise ValueError(f"compound NRSVue verses are not contiguous: {output_id}")
                if verse_id != components[0]:
                    continue
                merged_row = {
                    **row,
                    "ko": " ".join(str(value["ko"]) for value in component_rows if value),
                    "ja": "".join(str(value["ja"]) for value in component_rows if value),
                    "resource_id": output_id,
                    "family_id": f"bible:{output_id}",
                }
                consumed_ids.update(components)
        if not english or not _LATIN_RE.search(english):
            missing_ids.append(verse_id)
            continue
        updated = {
            "ko": merged_row["ko"],
            "en": english,
            "ja": merged_row["ja"],
            **{key: value for key, value in merged_row.items() if key not in {"ko", "en", "ja"}},
        }
        updated["source"] = SOURCE
        updated["source_revision"] = SOURCE_REVISION
        output.append(updated)
    if missing_ids:
        sample = ", ".join(missing_ids[:20])
        raise ValueError(f"NRSVue exact-ID join missed {len(missing_ids)} rows: {sample}")
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    digest = sha256()
    count = 0
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            handle.write(encoded)
            digest.update(encoded)
            count += 1
    temporary.replace(path)
    return count, digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/source_cache/remediation_20260818"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.input.resolve())
    english_verses, _, chapters = _crawl_bible_version("en", args.cache_dir.resolve())
    output_rows = add_english_rows(source_rows, english_verses)
    count, digest = write_jsonl(args.output.resolve(), output_rows)
    print(
        json.dumps(
            {
                "input_rows": len(source_rows),
                "output_rows": count,
                "nrsvue_chapters": chapters,
                "nrsvue_verse_ids": len(english_verses),
                "sha256": digest,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
