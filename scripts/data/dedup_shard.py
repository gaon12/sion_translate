#!/usr/bin/env python3
"""Drop repeated translation pairs from a parallel JSONL shard.

Identity is the tuple of populated ko/en/ja texts after whitespace stripping,
so a shard split into ko->ja and ja->ko directions collapses back to one row
per pair. The first occurrence wins and input order is preserved; every other
key on that row is kept untouched.

Rows whose populated language set differs are never merged, because a ko/ja row
and a ko/en/ja row carry different training signal even when their Korean side
matches.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any


LANGUAGES = ("ko", "en", "ja")


def pair_identity(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return the stripped language fields that identify a translation pair."""

    identity: list[tuple[str, str]] = []
    for language in LANGUAGES:
        value = row.get(language)
        if isinstance(value, str) and value.strip():
            identity.append((language, value.strip()))
    return tuple(identity)


def dedup_rows(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield the first row for each distinct translation pair, in input order.

    A row with no populated language field cannot be identified as a pair, so it
    is passed through rather than silently collapsed into one arbitrary row.
    """

    seen: set[tuple[tuple[str, str], ...]] = set()
    for row in rows:
        identity = pair_identity(row)
        if not identity:
            yield row
            continue
        if identity in seen:
            continue
        seen.add(identity)
        yield row


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is not valid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            yield row


def _validate_staged_jsonl(
    path: Path,
    *,
    expected_count: int,
    expected_digest: str,
    expected_bytes: int,
) -> None:
    """Verify the complete staged artifact before publishing it."""

    digest = sha256()
    count = 0
    with path.open("rb") as handle:
        for number, payload in enumerate(handle, start=1):
            digest.update(payload)
            if not payload.endswith(b"\n"):
                raise ValueError(f"{path}:{number} is missing its terminating newline")
            try:
                row = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{number} is not valid UTF-8 JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            count += 1

    actual_bytes = path.stat().st_size
    actual_digest = digest.hexdigest()
    if count != expected_count:
        raise ValueError(
            f"staged JSONL row count mismatch: expected {expected_count}, found {count}"
        )
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"staged JSONL size mismatch: expected {expected_bytes}, found {actual_bytes}"
        )
    if actual_digest != expected_digest:
        raise ValueError(
            f"staged JSONL SHA-256 mismatch: expected {expected_digest}, found {actual_digest}"
        )


def _fsync_directory(path: Path) -> None:
    """Persist a published directory entry on platforms that support it."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    allow_empty: bool = False,
) -> tuple[int, str]:
    """Durably build and atomically replace a JSONL shard.

    The staging file is a unique sibling of ``path``. This keeps an existing
    output intact if row production, serialization, validation, or publication
    fails, and lets ``rows`` safely read from ``path`` for an in-place rewrite.
    """

    digest = sha256()
    count = 0
    byte_count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                payload = json.dumps(row, ensure_ascii=False) + "\n"
                encoded = payload.encode("utf-8")
                handle.write(payload)
                digest.update(encoded)
                byte_count += len(encoded)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())

        staged_digest = digest.hexdigest()
        if count == 0 and not allow_empty:
            raise ValueError(
                "refusing to publish an empty JSONL shard; "
                "pass --allow-empty only when an empty output is intentional"
            )
        _validate_staged_jsonl(
            temporary_path,
            expected_count=count,
            expected_digest=staged_digest,
            expected_bytes=byte_count,
        )
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
        return count, staged_digest
    except BaseException as error:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(f"also failed to remove staging file {temporary_path}: {cleanup_error}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="allow an empty input to atomically replace the output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    target = args.output.resolve()
    input_rows = 0

    def counted_source_rows() -> Iterator[dict[str, Any]]:
        nonlocal input_rows
        for row in read_jsonl(source):
            input_rows += 1
            yield row

    output_rows, digest = write_jsonl(
        target,
        dedup_rows(counted_source_rows()),
        allow_empty=args.allow_empty,
    )
    print(
        json.dumps(
            {
                "input_rows": input_rows,
                "output_rows": output_rows,
                "removed_rows": input_rows - output_rows,
                "sha256": digest,
                "bytes": target.stat().st_size,
                "output": str(target),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
