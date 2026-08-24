#!/usr/bin/env python3
"""Apply only contamination repairs whose replacements are deterministic.

A row that translates `씨발` as `種まき` can be repaired by a rule because the
replacement does not depend on context. Literal idioms and lost profanity
intensity require a person to write a new Japanese translation, so they cannot
be repaired safely by a fixed rule. This tool handles only the deterministic
case and leaves the rest in the human review queue built by
``build_review_queue.py``.

    # Preview every proposed change without writing anything.
    python scripts/data/apply_contamination_repairs.py --input "data/*.jsonl" \
        --source-language ko --target-language ja

    # Apply the proposed changes and save a detailed report.
    python scripts/data/apply_contamination_repairs.py --input "data/*.jsonl" \
        --source-language ko --target-language ja \
        --apply --report reports/contamination_repairs.json

Without ``--apply``, the command does not write any files. Preview mode is the
default because applying a repair modifies the corpus in place.

The original shard is preserved under
``data/excluded/contamination_repair_<date>/``. The report counts every repair
and includes original and repaired text for a bounded review sample. Its
``backup_files`` mapping identifies the exact immutable snapshot for each input;
an existing snapshot is never overwritten. Restore that snapshot to undo an
applied repair.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import date
from hashlib import sha256
from pathlib import Path
import tempfile
from typing import cast

from sion_translate.console import configure_stdio
from sion_translate.contamination import repair_pair, supported_direction
from sion_translate.data.quality import canonical_text
from sion_translate.tokenizer import expand_inputs

# Keep the report compact enough for a person to inspect. The review queue holds
# the complete set of candidates.
SAMPLE_LIMIT = 20


def _paths_alias(left: Path, right: Path) -> bool:
    """Return whether two paths resolve to the same file, including hard links."""

    try:
        if left.resolve() == right.resolve():
            return True
    except OSError:
        pass
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _fsync_directory(path: Path) -> None:
    """Persist a directory replacement on platforms that permit directory fsync."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(
    path: Path,
    text: str,
    *,
    preserve_metadata_from: Path | None = None,
) -> None:
    """Write complete text beside its destination, sync it, and replace atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete=False,
        ) as sink:
            temporary = Path(sink.name)
            sink.write(text)
            sink.flush()
            os.fsync(sink.fileno())
        if preserve_metadata_from is not None:
            shutil.copystat(preserve_metadata_from, temporary)
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _copy_backup(source: Path, backup_root: Path) -> Path:
    """Publish one immutable backup without colliding on duplicate basenames."""

    identity = sha256(os.path.normcase(str(source.resolve())).encode("utf-8")).hexdigest()[:32]
    directory = backup_root / identity
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name
    temporary: Path | None = None
    try:
        with source.open("rb") as input_handle:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=directory,
                prefix=f".{source.name}.",
                suffix=".part",
                delete=False,
            ) as sink:
                temporary = Path(sink.name)
                shutil.copyfileobj(input_handle, sink)
                sink.flush()
                os.fsync(sink.fileno())
        shutil.copystat(source, temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to overwrite an existing immutable backup: {destination}"
            ) from error
        _fsync_directory(directory)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Repair only deterministic contamination (default: preview without writing)")
    )
    parser.add_argument("--input", nargs="+", required=True, help="JSONL files or globs")
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="modify the input files; omit this option to preview proposed changes",
    )
    parser.add_argument(
        "--backup-root",
        default=None,
        help=(
            "directory that preserves original shards "
            "(default: data/excluded/contamination_repair_<date>)"
        ),
    )
    parser.add_argument("--report", help="path for the JSON report")
    return parser


def _default_backup_root() -> Path:
    return Path("data/excluded") / f"contamination_repair_{date.today():%Y%m%d}"


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()

    source_language = args.source_language
    target_language = args.target_language
    if not supported_direction(source_language, target_language):
        raise SystemExit(
            f"No repair rules exist for {source_language}->{target_language}. "
            "The command is stopping instead of reporting an unsupported direction "
            "as having nothing to repair."
        )

    paths = expand_inputs(args.input)
    if not paths:
        raise SystemExit(f"No JSONL files matched the requested inputs: {args.input}")
    report_path = Path(args.report) if args.report else None
    if report_path is not None and any(_paths_alias(report_path, path) for path in paths):
        raise SystemExit(
            "The report path must not refer to an input shard. Choose a separate "
            f"destination: {report_path}"
        )

    backup_root = Path(args.backup_root) if args.backup_root else _default_backup_root()
    by_file: Counter[str] = Counter()
    backup_files: dict[str, str] = {}
    samples: list[dict[str, object]] = []
    scanned = 0
    repaired_rows = 0

    for path in paths:
        original_lines = path.read_text(encoding="utf-8-sig").splitlines()
        rewritten: list[str] = []
        changed_here = 0

        for line_number, raw_line in enumerate(original_lines, start=1):
            stripped = raw_line.strip()
            if not stripped:
                rewritten.append(raw_line)
                continue
            try:
                payload: object = json.loads(stripped)
            except json.JSONDecodeError:
                rewritten.append(raw_line)
                continue
            if not isinstance(payload, dict):
                rewritten.append(raw_line)
                continue
            row = cast(dict[str, object], payload)

            source = row.get(source_language)
            target = row.get(target_language)
            if not isinstance(source, str) or not isinstance(target, str):
                rewritten.append(raw_line)
                continue

            scanned += 1
            repair = repair_pair(
                canonical_text(source),
                canonical_text(target),
                source_language=source_language,
                target_language=target_language,
            )
            if repair is None:
                rewritten.append(raw_line)
                continue

            changed_here += 1
            repaired_rows += 1
            by_file[path.name] += 1
            if len(samples) < SAMPLE_LIMIT:
                samples.append(
                    {
                        "file": path.name,
                        "line": line_number,
                        "source": canonical_text(source),
                        "before": repair.original_target,
                        "after": repair.target,
                        "replacements": [list(pair) for pair in repair.replacements],
                    }
                )
            row[target_language] = repair.target
            # Mark the row as an additional recovery aid. This identifies every
            # rule-repaired row even if its preserved original is later lost.
            row["contamination_repaired"] = True
            rewritten.append(json.dumps(row, ensure_ascii=False))

        if changed_here and args.apply:
            backup_path = _copy_backup(path, backup_root)
            backup_files[str(path)] = str(backup_path)
            _atomic_write_text(
                path,
                "\n".join(rewritten) + "\n",
                preserve_metadata_from=path,
            )

    report = {
        "applied": bool(args.apply),
        "scanned_pairs": scanned,
        "repaired_rows": repaired_rows,
        "by_file": dict(by_file.most_common()),
        "backup_root": str(backup_root) if args.apply and repaired_rows else None,
        "backup_files": backup_files,
        "samples": samples,
        "note": (
            "This report counts only rows repaired by deterministic rules. Literal "
            "idioms and lost profanity intensity remain in the build_review_queue.py "
            "output because a person must write their replacements."
        ),
    }
    if report_path is not None:
        _atomic_write_text(
            report_path,
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )

    print(f"Scanned pairs: {scanned:,} / repaired rows: {repaired_rows:,}")
    for name, count in by_file.most_common():
        print(f"  {name}: {count:,}")
    if not args.apply:
        print("\nPreview complete. No files were written; pass --apply to apply the repairs.")
    elif repaired_rows:
        print(f"\nOriginal shards preserved in: {backup_root}")


if __name__ == "__main__":
    main()
