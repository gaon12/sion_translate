#!/usr/bin/env python3
"""Bring ``data/data.tsv`` and the shard manifests back in line with the files.

Every builder writes a manifest recording the row count, byte size and SHA-256
of the shard it produced, and ``data/data.tsv`` is the registry that carries the
same three numbers for the whole corpus. Nothing recomputes them afterwards, so
any operation that rewrites a shard -- corpus-wide deduplication, a refilter, a
repair -- silently turns the provenance record into fiction while leaving the
training pipeline working perfectly. This script recomputes those numbers from
the files that are actually on disk.

It changes only measurements. Sources, URLs, licences and notes are the record
of where data came from, and no measurement can reconstruct them, so they are
left exactly as they are apart from an optional ``--note`` appended to the rows
whose numbers moved.

``data.tsv`` rows are matched to files by the basename of ``original_file``,
because the ``file`` column uses the language-group layout of the published
registry rather than the flat layout of the training root. A row whose file is
gone is reported and, with ``--mark-missing``, has its measurements zeroed and
the reason recorded, which is what an archived shard looks like afterwards.

Usage::

    python scripts/data/refresh_shard_registry.py --check
    python scripts/data/refresh_shard_registry.py \\
        --note "코퍼스 전역 중복 제거(2026-08-28)"

Exit codes: 0 success (or, with ``--check``, everything already matches),
1 ``--check`` found stale numbers, 2 bad input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


DEFAULT_REGISTRY = Path("data/data.tsv")
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_MANIFEST_GLOB = "*manifest.json"

# Manifests were written by different builders and do not agree on names.
COUNT_KEYS = ("row_count", "rows", "rows_out")
BYTE_KEYS = ("output_bytes", "bytes")
DIGEST_KEYS = ("sha256",)


def measure(path: Path) -> tuple[int, int, str]:
    """Return ``(rows, bytes, sha256)`` for one JSONL shard."""

    digest = hashlib.sha256()
    rows = 0
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            rows += block.count(b"\n")
            size += len(block)
    return rows, size, digest.hexdigest()


def resolve_output(value: object, data_root: Path) -> Path | None:
    """Map a manifest ``output`` field onto the shard in this working tree.

    Manifests store absolute paths from the machine that built them, so only the
    file name is portable.
    """

    if not isinstance(value, str) or not value:
        return None
    name = Path(value.replace("\\", "/")).name
    candidate = data_root / name
    return candidate if candidate.is_file() else None


def update_entry(entry: dict[str, Any], path: Path) -> dict[str, tuple[Any, Any]]:
    """Refresh whichever measurement keys this manifest entry happens to use."""

    rows, size, digest = measure(path)
    changes: dict[str, tuple[Any, Any]] = {}
    for keys, value in ((COUNT_KEYS, rows), (BYTE_KEYS, size), (DIGEST_KEYS, digest)):
        for key in keys:
            if key in entry and entry[key] != value:
                changes[key] = (entry[key], value)
                entry[key] = value
    return changes


def refresh_manifest(path: Path, data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the updated manifest and a per-shard record of what moved."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    changes: dict[str, Any] = {}

    shard = resolve_output(manifest.get("output"), data_root)
    if shard is not None:
        moved = update_entry(manifest, shard)
        if moved:
            changes[shard.name] = moved

    corpora = manifest.get("corpora")
    if isinstance(corpora, list):
        total = 0
        for member in corpora:
            if not isinstance(member, dict):
                continue
            member_shard = resolve_output(member.get("output"), data_root)
            if member_shard is None:
                continue
            moved = update_entry(member, member_shard)
            if moved:
                changes[member_shard.name] = moved
            for key in COUNT_KEYS:
                if key in member:
                    total += int(member[key])
                    break
        # The aggregate manifest carries no output of its own; its total is the
        # sum of the members it lists.
        if shard is None:
            for key in COUNT_KEYS:
                if key in manifest and manifest[key] != total:
                    changes[f"{path.name}:{key}"] = (manifest[key], total)
                    manifest[key] = total
                    break

    return manifest, changes


def refresh_registry(
    path: Path,
    data_root: Path,
    *,
    note: str | None,
    mark_missing: bool,
) -> tuple[list[str], list[dict[str, Any]], list[str], list[str]]:
    """Recompute the registry rows, returning the new file and what changed."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for column in ("original_file", "rows", "bytes", "sha256"):
        if column not in fieldnames:
            raise ValueError(f"{path} has no {column!r} column")

    changed: list[dict[str, Any]] = []
    missing: list[str] = []
    unchanged: list[str] = []
    for row in rows:
        source = row.get("original_file") or ""
        name = Path(source.replace("\\", "/")).name
        shard = data_root / name
        if not shard.is_file():
            missing.append(name or "(no original_file)")
            if mark_missing and name:
                row["rows"] = "0"
                row["bytes"] = "0"
                row["sha256"] = ""
                if note:
                    row["notes"] = f"{row.get('notes', '')}; {note}".lstrip("; ")
            continue
        measured_rows, measured_bytes, digest = measure(shard)
        before = (row["rows"], row["bytes"], row["sha256"])
        after = (str(measured_rows), str(measured_bytes), digest)
        if before == after:
            unchanged.append(name)
            continue
        row["rows"], row["bytes"], row["sha256"] = after
        if note:
            row["notes"] = f"{row.get('notes', '')}; {note}".lstrip("; ")
        changed.append(
            {
                "file": name,
                "rows": [int(before[0] or 0), measured_rows],
                "bytes": [int(before[1] or 0), measured_bytes],
            }
        )

    lines: list[str] = ["\t".join(fieldnames)]
    for row in rows:
        lines.append("\t".join((row.get(name) or "").replace("\t", " ") for name in fieldnames))
    return lines, changed, missing, unchanged


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--manifest-glob",
        default=DEFAULT_MANIFEST_GLOB,
        help=f"manifests to refresh under --data-root (default: {DEFAULT_MANIFEST_GLOB})",
    )
    parser.add_argument(
        "--note",
        help="appended to the notes of every registry row whose numbers moved",
    )
    parser.add_argument(
        "--mark-missing",
        action="store_true",
        help="zero the measurements of a registry row whose shard is gone",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit nonzero without writing anything",
    )
    parser.add_argument("--report", type=Path, help="write the JSON summary here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.registry.is_file():
        print(f"registry not found: {args.registry}", file=sys.stderr)
        return 2
    if not args.data_root.is_dir():
        print(f"data root not found: {args.data_root}", file=sys.stderr)
        return 2

    try:
        lines, changed, missing, unchanged = refresh_registry(
            args.registry,
            args.data_root,
            note=None if args.check else args.note,
            mark_missing=args.mark_missing,
        )
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2

    manifest_changes: dict[str, Any] = {}
    manifest_payloads: dict[Path, dict[str, Any]] = {}
    for manifest_path in sorted(args.data_root.glob(args.manifest_glob)):
        manifest, changes = refresh_manifest(manifest_path, args.data_root)
        manifest_payloads[manifest_path] = manifest
        if changes:
            manifest_changes[manifest_path.name] = changes

    if not args.check:
        args.registry.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for manifest_path, manifest in manifest_payloads.items():
            if manifest_path.name in manifest_changes:
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    report = {
        "registry": str(args.registry),
        "registry_rows_updated": len(changed),
        "registry_rows_unchanged": len(unchanged),
        "registry_rows_missing_file": missing,
        "registry_changes": changed,
        "manifest_changes": manifest_changes,
        "written": not args.check,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.check and (changed or manifest_changes):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
