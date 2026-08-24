#!/usr/bin/env python3
"""Cap how many rows a generated shard may contribute per sentence frame.

``audit_generated_shards.py`` reports that data44/45/48/50/51 restate a small
number of frames thousands of times. Deleting those shards throws away the
lexical content they do carry; keeping them whole lets one frame dominate the
sampler and lets near-duplicates cross the train/holdout boundary.

This tool keeps at most ``--max-per-skeleton`` rows for each sentence frame,
where a frame is the sentence with quoted spans replaced by ``<Q>`` and digits
by ``#`` - the same definition the audit uses. Selection is deterministic: rows
are ranked by a seeded hash of the pair so the same input always yields the same
output, independent of file order.

It also drops rows whose target carries the wrong script, which is how data46
and data51 leak Hangul into Japanese.

Usage::

    python scripts/data/resample_generated_shards.py \
        --source-key de --target-key fr --max-per-skeleton 8 \
        --output-dir data/resampled data/generated_de_fr.jsonl

    python scripts/data/resample_generated_shards.py \
        --source-key sr-Latn --target-key ar --max-per-skeleton 8 \
        --in-place --report r.json data/generated_sr_ar*.jsonl

``--in-place`` preserves each original under ``--backup-dir`` (by default
``<input dir>/excluded/resampled_original/``) and resamples from there, so
re-running with a tighter cap does not compound on an already-reduced file.
That matters because the generator is not available to regenerate from.

Exit codes: 0 written, 2 bad input.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Sequence, cast
import uuid

from sion_translate.data.quality import canonical_text
from sion_translate.scripts_registry import (
    has_foreign_script,
    resolve_scripts,
)


_QUOTED = re.compile(r"[\"“‘'][^\"”’']{1,120}[\"”’']")
_DIGITS = re.compile(r"\d")


@dataclass
class ShardResult:
    path: str
    output: str
    rows_in: int = 0
    rows_out: int = 0
    unreadable: int = 0
    dropped_foreign_script: int = 0
    dropped_duplicate: int = 0
    dropped_over_cap: int = 0
    dropped_over_span_cap: int = 0
    skeletons_in: int = 0
    skeletons_out: int = 0
    largest_frame_in: int = 0
    largest_frame_out: int = 0
    quoted_spans_in: int = 0
    largest_span_in: int = 0
    dropped_skeleton_examples: list[tuple[str, int]] = field(default_factory=list[tuple[str, int]])
    dropped_span_examples: list[tuple[str, int]] = field(default_factory=list[tuple[str, int]])


@dataclass(frozen=True)
class _ShardPlan:
    input: Path
    source: Path
    reported_source: Path
    output: Path
    backup: Path | None
    create_backup: bool


def validate_record_keys(source_key: str, target_key: str) -> None:
    """Require two explicit, distinct JSON object keys."""

    for option, key in (("source_key", source_key), ("target_key", target_key)):
        if (
            not isinstance(key, str)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not key
            or key != key.strip()
        ):
            raise ValueError(f"{option} must be a non-empty key without surrounding whitespace")
    if source_key == target_key:
        raise ValueError("source_key and target_key must be distinct")


def _path_identity(path: Path) -> str:
    resolved = str(path.resolve(strict=False))
    return resolved.casefold() if sys.platform == "win32" else resolved


def _validate_output_path(path: Path) -> None:
    if path.exists() and not path.is_file():
        raise ValueError(f"output path is not a regular file: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise ValueError(f"output parent is not a directory: {path.parent}")


def _validate_resample_options(
    *,
    source_key: str,
    target_key: str,
    max_per_skeleton: int,
    max_per_quoted_span: int | None,
    target_scripts: Sequence[str],
) -> None:
    validate_record_keys(source_key, target_key)
    if max_per_skeleton < 1:
        raise ValueError("max_per_skeleton must be positive")
    if max_per_quoted_span is not None and max_per_quoted_span < 1:
        raise ValueError("max_per_quoted_span must be positive")
    resolve_scripts(target_scripts)


def _stage_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _stage_lines(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line + "\n")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _stage_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source, temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def script_list(value: str) -> tuple[str, ...]:
    """Parse a comma-separated script/language list.

    A comma-separated single value is used rather than nargs="+", which would
    swallow the positional shard paths.
    """

    names = tuple(part.strip() for part in value.split(",") if part.strip())
    resolve_scripts(names)
    return names


def skeleton(text: str) -> str:
    """Blank quoted spans and digits, matching audit_generated_shards.skeleton."""

    return _DIGITS.sub("#", _QUOTED.sub("<Q>", text))


def target_is_foreign(text: str, target_scripts: Sequence[str]) -> bool:
    """True when the target uses a script its language does not permit."""

    return has_foreign_script(text, target_scripts)


def _rank(source: str, target: str, seed: int) -> bytes:
    """A stable per-row ordering key, independent of position in the file."""

    payload = f"{seed}\0{source}\0{target}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def _build_resampled_shard(
    path: Path,
    output: Path,
    *,
    max_per_skeleton: int,
    max_per_quoted_span: int | None = None,
    source_key: str,
    target_key: str,
    target_scripts: Sequence[str] = (),
    seed: int = 20260730,
) -> tuple[ShardResult, list[str]]:
    """Build rows under per-frame and optional per-quoted-span caps without writing.

    The two axes degenerate independently. data44 restated 709 rows of a single
    frame, which the frame cap fixes. data48 varies its frames but draws on nine
    distinct quoted spans for 6,171 uses, which only the span cap fixes.
    """

    _validate_resample_options(
        source_key=source_key,
        target_key=target_key,
        max_per_skeleton=max_per_skeleton,
        max_per_quoted_span=max_per_quoted_span,
        target_scripts=target_scripts,
    )
    if not path.is_file():
        raise FileNotFoundError(path)

    result = ShardResult(path=str(path), output=str(output))
    candidates: list[tuple[bytes, str, str, tuple[str, ...]]] = []
    seen_pairs: set[bytes] = set()
    skeleton_counts: Counter[str] = Counter()
    span_counts: Counter[str] = Counter()

    with path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            result.rows_in += 1
            try:
                raw_row: object = json.loads(raw_line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                result.unreadable += 1
                continue
            if not isinstance(raw_row, dict):
                result.unreadable += 1
                continue
            row = cast(dict[object, object], raw_row)
            source = row.get(source_key)
            target = row.get(target_key)
            if not isinstance(source, str) or not isinstance(target, str):
                result.unreadable += 1
                continue
            source, target = canonical_text(source), canonical_text(target)
            if not source or not target:
                result.unreadable += 1
                continue
            if target_is_foreign(target, target_scripts):
                result.dropped_foreign_script += 1
                continue
            digest = _rank(source, target, seed)
            if digest in seen_pairs:
                result.dropped_duplicate += 1
                continue
            seen_pairs.add(digest)
            frame = skeleton(source)
            spans = tuple(_QUOTED.findall(source))
            skeleton_counts[frame] += 1
            for span in spans:
                span_counts[span] += 1
            candidates.append(
                (
                    digest,
                    json.dumps({source_key: source, target_key: target}, ensure_ascii=False),
                    frame,
                    spans,
                )
            )

    result.skeletons_in = len(skeleton_counts)
    result.largest_frame_in = max(skeleton_counts.values(), default=0)
    result.quoted_spans_in = len(span_counts)
    result.largest_span_in = max(span_counts.values(), default=0)

    # Greedy selection in rank order applies both caps uniformly. The rank is a
    # hash of the pair, so the result does not depend on position in the file.
    candidates.sort(key=lambda item: item[0])
    kept: list[str] = []
    kept_frames: Counter[str] = Counter()
    kept_spans: Counter[str] = Counter()
    over_frame_cap: Counter[str] = Counter()
    over_span_cap: Counter[str] = Counter()
    for _, line, frame, spans in candidates:
        if kept_frames[frame] >= max_per_skeleton:
            over_frame_cap[frame] += 1
            result.dropped_over_cap += 1
            continue
        if max_per_quoted_span is not None:
            saturated = [span for span in spans if kept_spans[span] >= max_per_quoted_span]
            if saturated:
                over_span_cap[saturated[0]] += 1
                result.dropped_over_span_cap += 1
                continue
        kept.append(line)
        kept_frames[frame] += 1
        for span in spans:
            kept_spans[span] += 1

    result.rows_out = len(kept)
    result.skeletons_out = len(kept_frames)
    result.largest_frame_out = max(kept_frames.values(), default=0)
    result.dropped_skeleton_examples = [
        (frame[:120], count) for frame, count in over_frame_cap.most_common(3)
    ]
    result.dropped_span_examples = [
        (span[:120], count) for span, count in over_span_cap.most_common(3)
    ]

    return result, kept


def resample_shard(
    path: Path,
    output: Path,
    *,
    max_per_skeleton: int,
    source_key: str,
    target_key: str,
    max_per_quoted_span: int | None = None,
    target_scripts: Sequence[str] = (),
    seed: int = 20260730,
) -> ShardResult:
    """Atomically write one resampled shard after validating its full contract."""

    _validate_resample_options(
        source_key=source_key,
        target_key=target_key,
        max_per_skeleton=max_per_skeleton,
        max_per_quoted_span=max_per_quoted_span,
        target_scripts=target_scripts,
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    _validate_output_path(output)
    if _path_identity(path) == _path_identity(output):
        raise ValueError("input and output paths must be distinct; use CLI --in-place")
    result, kept = _build_resampled_shard(
        path,
        output,
        max_per_skeleton=max_per_skeleton,
        max_per_quoted_span=max_per_quoted_span,
        source_key=source_key,
        target_key=target_key,
        target_scripts=target_scripts,
        seed=seed,
    )
    temporary = _stage_lines(output, kept)
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Resample generated shards").splitlines()[0]
    )
    parser.add_argument("paths", nargs="+", help="JSONL shards to resample")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", help="write resampled shards here")
    destination.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "overwrite each input, preserving the original under --backup-dir. "
            "Re-running with a different cap resamples the preserved original "
            "rather than the already-reduced file, so caps never compound"
        ),
    )
    parser.add_argument(
        "--backup-dir",
        help=(
            "where --in-place preserves originals "
            "(default: <input dir>/excluded/<input stem prefix>_original)"
        ),
    )
    parser.add_argument("--max-per-skeleton", type=int, default=8)
    parser.add_argument(
        "--max-per-quoted-span",
        type=int,
        help="cap reuse of one quoted span; omit to leave spans uncapped",
    )
    parser.add_argument(
        "--source-key",
        required=True,
        help="JSON key holding the source text (required; no language default)",
    )
    parser.add_argument(
        "--target-key",
        required=True,
        help="JSON key holding the target text (required; no language default)",
    )
    parser.add_argument(
        "--target-scripts",
        type=script_list,
        default=(),
        metavar="LIST",
        help=(
            "writing systems the target may use: script names or language "
            "shorthands, comma separated (de / ar / latin,cyrillic). "
            "omit to keep every row regardless of script"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--report", help="write the resampling report JSON here")
    return parser


def _preflight_cli(args: argparse.Namespace) -> tuple[list[_ShardPlan], Path | None]:
    """Resolve every path and option before any backup or output is created."""

    _validate_resample_options(
        source_key=args.source_key,
        target_key=args.target_key,
        max_per_skeleton=args.max_per_skeleton,
        max_per_quoted_span=args.max_per_quoted_span,
        target_scripts=args.target_scripts,
    )
    if args.backup_dir and not args.in_place:
        raise ValueError("--backup-dir only applies to --in-place")

    inputs = [Path(raw_path) for raw_path in cast(list[str], args.paths)]
    input_identities: set[str] = set()
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        identity = _path_identity(path)
        if identity in input_identities:
            raise ValueError(f"duplicate input shard: {path}")
        input_identities.add(identity)

    plans: list[_ShardPlan] = []
    output_identities: set[str] = set()
    backup_identities: set[str] = set()
    directory_identities: set[str] = set()
    output_root: Path | None = None
    if not args.in_place:
        output_root = Path(cast(str, args.output_dir))
        if output_root.exists() and not output_root.is_dir():
            raise ValueError(f"--output-dir is not a directory: {output_root}")
        directory_identities.add(_path_identity(output_root))

    for path in inputs:
        if args.in_place:
            backup_dir = (
                Path(args.backup_dir)
                if args.backup_dir
                else path.parent / "excluded" / "resampled_original"
            )
            if backup_dir.exists() and not backup_dir.is_dir():
                raise ValueError(f"--backup-dir is not a directory: {backup_dir}")
            directory_identities.add(_path_identity(backup_dir))
            backup = backup_dir / path.name
            _validate_output_path(backup)
            backup_identity = _path_identity(backup)
            if backup_identity in input_identities:
                raise ValueError(f"backup path collides with an input/output shard: {backup}")
            if backup_identity in backup_identities:
                raise ValueError(f"multiple inputs map to the same backup path: {backup}")
            backup_identities.add(backup_identity)
            create_backup = not backup.exists()
            source = path if create_backup else backup
            output = path
            reported_source = backup
        else:
            assert output_root is not None
            output = output_root / path.name
            backup = None
            create_backup = False
            source = path
            reported_source = path

        _validate_output_path(output)
        output_identity = _path_identity(output)
        if output_identity in output_identities:
            raise ValueError(f"multiple inputs map to the same output path: {output}")
        if not args.in_place and output_identity in input_identities:
            raise ValueError(f"output path collides with an input shard: {output}")
        output_identities.add(output_identity)
        plans.append(
            _ShardPlan(
                input=path,
                source=source,
                reported_source=reported_source,
                output=output,
                backup=backup,
                create_backup=create_backup,
            )
        )

    report_path = Path(args.report) if args.report else None
    if report_path is not None:
        _validate_output_path(report_path)
        report_identity = _path_identity(report_path)
        occupied = input_identities | output_identities | backup_identities | directory_identities
        if report_identity in occupied:
            raise ValueError(f"report path collides with a shard or backup: {report_path}")
    return plans, report_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plans, report_path = _preflight_cli(args)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"cannot resample ({error})", file=sys.stderr)
        return 2

    prepared: list[tuple[_ShardPlan, ShardResult, list[str]]] = []
    try:
        for plan in plans:
            result, kept = _build_resampled_shard(
                plan.source,
                plan.output,
                max_per_skeleton=args.max_per_skeleton,
                max_per_quoted_span=args.max_per_quoted_span,
                source_key=args.source_key,
                target_key=args.target_key,
                target_scripts=args.target_scripts,
                seed=args.seed,
            )
            result.path = str(plan.reported_source)
            result.output = str(plan.output)
            prepared.append((plan, result, kept))
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"cannot resample ({error})", file=sys.stderr)
        return 2

    staged: list[Path] = []
    staged_backups: list[tuple[Path, Path]] = []
    staged_outputs: list[tuple[Path, Path]] = []
    staged_report: tuple[Path, Path] | None = None
    try:
        for plan, _result, _kept in prepared:
            if plan.create_backup:
                assert plan.backup is not None
                if plan.backup.exists():
                    raise FileExistsError(f"backup appeared after preflight: {plan.backup}")
                temporary = _stage_copy(plan.input, plan.backup)
                staged.append(temporary)
                staged_backups.append((temporary, plan.backup))
        for plan, _result, kept in prepared:
            temporary = _stage_lines(plan.output, kept)
            staged.append(temporary)
            staged_outputs.append((temporary, plan.output))
        if report_path is not None:
            report_text = (
                json.dumps(
                    [asdict(result) for _plan, result, _kept in prepared],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            temporary = _stage_text(report_path, report_text)
            staged.append(temporary)
            staged_report = (temporary, report_path)

        for temporary, destination in staged_backups:
            temporary.replace(destination)
        for temporary, destination in staged_outputs:
            temporary.replace(destination)
        if staged_report is not None:
            staged_report[0].replace(staged_report[1])
    except OSError as error:
        print(f"cannot commit resampled outputs ({error})", file=sys.stderr)
        return 2
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)

    results = [result for _plan, result, _kept in prepared]

    for result in results:
        print(
            f"{Path(result.path).name:26} {result.rows_in:>8,} -> {result.rows_out:>8,} rows  "
            f"frames {result.skeletons_in:>6,}  "
            f"largest frame {result.largest_frame_in:>6,} -> {result.largest_frame_out:>4,}  "
            f"dropped frame={result.dropped_over_cap:,} "
            f"span={result.dropped_over_span_cap:,} "
            f"script={result.dropped_foreign_script:,} dup={result.dropped_duplicate:,}"
        )
        for frame, count in result.dropped_skeleton_examples:
            print(f"      frame -{count:<6,} {frame!r}")
        for span, count in result.dropped_span_examples:
            print(f"      span  -{count:<6,} {span!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
