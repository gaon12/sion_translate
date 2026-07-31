"""Build and verify a self-contained, reproducible GPU training bundle.

The bundle is intentionally assembled from a narrow set of sources:

* regular files at stage 0 in Git's index, except generated/runtime paths;
* immediate ``data/*.jsonl`` corpus files; and
* regular files below ``data/evaluation_only``.

Nothing else in the working tree is eligible.  In particular, stale artifacts,
checkpoints, virtual environments, caches, and ``data/excluded`` cannot enter
the archive just because they happen to exist beside the source tree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
from typing import BinaryIO, Iterable
import unicodedata
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "sion_translate"
MANIFEST_NAME = "PACKAGE_MANIFEST.json"
CHECKSUMS_NAME = "SHA256SUMS"
FORMAT_VERSION = 1
COPY_BUFFER_SIZE = 8 * 1024 * 1024
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MAX_METADATA_SIZE = 64 * 1024 * 1024

EXCLUDED_TOP_LEVEL = {
    ".git",
    ".venv",
    ".agents",
    ".codex",
    "artifacts",
    "build",
    "cache",
    "caches",
    "checkpoints",
    "comparison_outputs",
    "dist",
    "env",
    "exports",
    "models",
    "runs",
    "translation_queue",
    "venv",
}
EXCLUDED_PATH_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "cache",
    "caches",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
ALLOWED_ORIGINS = {"git-index", "data-jsonl", "evaluation-only"}
REGULAR_GIT_MODES = {"100644", "100755"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40,64}")


class BundleError(RuntimeError):
    """Raised when a bundle cannot be built or fails integrity validation."""


@dataclass(frozen=True)
class SourceEntry:
    """One source file selected for the bundle."""

    relative_path: PurePosixPath
    source_path: Path
    origin: str
    mode: str


@dataclass(frozen=True)
class FileRecord:
    """Integrity metadata for one payload file."""

    path: str
    size: int
    sha256: str
    origin: str
    mode: str

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "origin": self.origin,
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True)
class VerificationResult:
    """Summary returned by archive and extracted-tree verification."""

    file_count: int
    total_bytes: int
    git_commit: str
    git_tree: str


@dataclass(frozen=True)
class BuildResult:
    """Summary returned after an archive is atomically published."""

    output_path: Path
    archive_sha256: str
    file_count: int
    total_bytes: int
    git_commit: str
    git_tree: str


def _run_git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BundleError(f"git {' '.join(arguments)} failed: {detail or 'unknown error'}")
    return completed.stdout


def _ensure_clean_tracked_tree(root: Path) -> None:
    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=no",
    )
    if not status:
        return

    records = [item.decode("utf-8", errors="replace") for item in status.split(b"\0") if item]
    preview = ", ".join(records[:5])
    if len(records) > 5:
        preview += f", ... ({len(records)} entries)"
    raise BundleError(
        "tracked files are not clean; commit or restore them before packaging"
        + (f": {preview}" if preview else "")
    )


def _git_identity(root: Path) -> tuple[str, str]:
    commit = _run_git(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    tree = _run_git(root, "rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    if not GIT_OBJECT_PATTERN.fullmatch(commit) or not GIT_OBJECT_PATTERN.fullmatch(tree):
        raise BundleError("Git returned an invalid commit or tree object id")
    return commit, tree


def _validated_relative_path(raw_path: str) -> PurePosixPath:
    if not raw_path or "\\" in raw_path or "\r" in raw_path or "\n" in raw_path:
        raise BundleError(f"unsupported bundle path: {raw_path!r}")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BundleError(f"unsafe bundle path: {raw_path!r}")
    if path.as_posix() != raw_path:
        raise BundleError(f"non-canonical bundle path: {raw_path!r}")
    return path


def _is_excluded_tracked_path(path: PurePosixPath) -> bool:
    if path.parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    if any(part in EXCLUDED_PATH_PARTS for part in path.parts):
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return path.parts[:2] == ("data", "excluded")


def _portable_path_key(path: PurePosixPath) -> str:
    """Normalize names the way common case-insensitive extractors do."""

    return unicodedata.normalize("NFC", path.as_posix()).casefold()


def _tracked_stage_zero_entries(root: Path) -> list[SourceEntry]:
    output = _run_git(root, "ls-files", "--stage", "-z")
    entries: list[SourceEntry] = []
    for raw_record in output.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode_raw, _object_id, stage_raw = metadata.split(b" ", 2)
            mode = mode_raw.decode("ascii")
            stage = stage_raw.decode("ascii")
            path_text = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise BundleError("could not parse a stage-0 Git index entry") from error
        if stage != "0":
            continue

        relative_path = _validated_relative_path(path_text)
        if _is_excluded_tracked_path(relative_path):
            continue
        if mode not in REGULAR_GIT_MODES:
            raise BundleError(
                f"tracked path {path_text!r} has unsupported Git mode {mode}; "
                "bundles accept regular files only"
            )
        entries.append(
            SourceEntry(
                relative_path=relative_path,
                source_path=root.joinpath(*relative_path.parts),
                origin="git-index",
                mode=mode,
            )
        )
    return entries


def _assert_regular_source(path: Path, relative_path: PurePosixPath) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise BundleError(f"selected source file is missing: {relative_path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise BundleError(f"selected source is not a regular file: {relative_path}")


def _collect_sources(root: Path) -> list[SourceEntry]:
    selected: dict[PurePosixPath, SourceEntry] = {}
    portable_paths = {
        _portable_path_key(PurePosixPath(MANIFEST_NAME)): MANIFEST_NAME,
        _portable_path_key(PurePosixPath(CHECKSUMS_NAME)): CHECKSUMS_NAME,
    }

    def add(entry: SourceEntry) -> None:
        _assert_regular_source(entry.source_path, entry.relative_path)
        previous = selected.get(entry.relative_path)
        if previous is not None:
            if previous.source_path.resolve() != entry.source_path.resolve():
                raise BundleError(f"multiple sources map to {entry.relative_path}")
            return
        portable_key = _portable_path_key(entry.relative_path)
        collision = portable_paths.get(portable_key)
        if collision is not None:
            raise BundleError(
                f"portable path collision between {collision!r} and "
                f"{entry.relative_path.as_posix()!r}"
            )
        portable_paths[portable_key] = entry.relative_path.as_posix()
        selected[entry.relative_path] = entry

    for entry in _tracked_stage_zero_entries(root):
        add(entry)

    data_root = root / "data"
    if data_root.is_dir():
        for source_path in sorted(data_root.glob("*.jsonl"), key=lambda path: path.name):
            if not source_path.name.endswith(".jsonl"):
                continue
            relative_path = _validated_relative_path(source_path.relative_to(root).as_posix())
            add(
                SourceEntry(
                    relative_path=relative_path,
                    source_path=source_path,
                    origin="data-jsonl",
                    mode="100644",
                )
            )

    evaluation_root = data_root / "evaluation_only"
    if evaluation_root.is_dir():
        evaluation_paths = sorted(
            evaluation_root.rglob("*"),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        for source_path in evaluation_paths:
            if source_path.is_symlink():
                relative = source_path.relative_to(root).as_posix()
                raise BundleError(f"evaluation-only source may not be a symlink: {relative}")
            if source_path.is_dir():
                continue
            relative_path = _validated_relative_path(source_path.relative_to(root).as_posix())
            add(
                SourceEntry(
                    relative_path=relative_path,
                    source_path=source_path,
                    origin="evaluation-only",
                    mode="100644",
                )
            )

    entries = [selected[path] for path in sorted(selected, key=lambda item: item.as_posix())]
    if not any(
        len(entry.relative_path.parts) == 2
        and entry.relative_path.parts[0] == "data"
        and entry.relative_path.name.endswith(".jsonl")
        for entry in entries
    ):
        raise BundleError("no immediate data/*.jsonl training corpus files were selected")
    if not any(entry.relative_path.parts[:2] == ("data", "evaluation_only") for entry in entries):
        raise BundleError("data/evaluation_only is missing or contains no regular files")
    return entries


def _zip_info(relative_path: str, mode: str) -> zipfile.ZipInfo:
    canonical_path = _validated_relative_path(relative_path).as_posix()
    info = zipfile.ZipInfo(
        filename=f"{ARCHIVE_ROOT}/{canonical_path}",
        date_time=ZIP_TIMESTAMP,
    )
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = int(mode, 8) << 16
    info.flag_bits |= 0x800
    # ZipFile.open has no public per-entry compression-level argument.
    info._compresslevel = 6
    return info


def _copy_and_hash(source: BinaryIO, destination: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = source.read(COPY_BUFFER_SIZE)
        if not chunk:
            break
        destination.write(chunk)
        digest.update(chunk)
        total += len(chunk)
    return total, digest.hexdigest()


def _write_source(
    archive: zipfile.ZipFile,
    entry: SourceEntry,
) -> FileRecord:
    before = entry.source_path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise BundleError(f"selected source is no longer a regular file: {entry.relative_path}")

    info = _zip_info(entry.relative_path.as_posix(), entry.mode)
    with (
        entry.source_path.open("rb") as source,
        archive.open(
            info,
            mode="w",
            force_zip64=True,
        ) as destination,
    ):
        size, digest = _copy_and_hash(source, destination)

    after = entry.source_path.stat()
    if (
        size != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
        or after.st_ino != before.st_ino
        or after.st_dev != before.st_dev
    ):
        raise BundleError(f"source changed while it was packaged: {entry.relative_path}")
    return FileRecord(
        path=entry.relative_path.as_posix(),
        size=size,
        sha256=digest,
        origin=entry.origin,
        mode=entry.mode,
    )


def _write_bytes(
    archive: zipfile.ZipFile,
    relative_path: str,
    content: bytes,
) -> str:
    digest = hashlib.sha256(content).hexdigest()
    info = _zip_info(relative_path, "100644")
    with archive.open(info, mode="w", force_zip64=True) as destination:
        destination.write(content)
    return digest


def _manifest_bytes(
    commit: str,
    tree: str,
    records: list[FileRecord],
) -> bytes:
    manifest = {
        "archive_root": ARCHIVE_ROOT,
        "files": [record.as_dict() for record in records],
        "format_version": FORMAT_VERSION,
        "git": {
            "commit": commit,
            "tree": tree,
        },
        "payload": {
            "file_count": len(records),
            "total_bytes": sum(record.size for record in records),
        },
        "zip_metadata": {
            "compression": "deflate",
            "timestamp": "1980-01-01T00:00:00Z",
            "zip64": True,
        },
    }
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _checksums_bytes(
    records: Iterable[FileRecord],
    manifest_sha256: str,
) -> bytes:
    lines = [f"{record.sha256}  {record.path}\n" for record in records]
    lines.append(f"{manifest_sha256}  {MANIFEST_NAME}\n")
    return "".join(lines).encode("utf-8")


def _write_archive(
    destination: Path,
    sources: list[SourceEntry],
    commit: str,
    tree: str,
) -> None:
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
        strict_timestamps=True,
    ) as archive:
        records = [_write_source(archive, entry) for entry in sources]
        manifest = _manifest_bytes(commit, tree, records)
        manifest_sha256 = _write_bytes(archive, MANIFEST_NAME, manifest)
        _write_bytes(
            archive,
            CHECKSUMS_NAME,
            _checksums_bytes(records, manifest_sha256),
        )


def _fsync_file(path: Path) -> None:
    # Windows requires a writable descriptor for FlushFileBuffers, which backs
    # os.fsync there.  No bytes are changed.
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_stream(source: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = source.read(COPY_BUFFER_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def _hash_file(path: Path) -> tuple[int, str]:
    with path.open("rb") as source:
        return _hash_stream(source)


def _read_limited(source: BinaryIO, size: int, name: str) -> bytes:
    if size > MAX_METADATA_SIZE:
        raise BundleError(f"{name} is unreasonably large ({size} bytes)")
    content = source.read(MAX_METADATA_SIZE + 1)
    if len(content) > MAX_METADATA_SIZE:
        raise BundleError(f"{name} exceeds the metadata size limit")
    return content


def _parse_manifest(content: bytes) -> tuple[dict[str, object], list[FileRecord]]:
    try:
        raw = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"{MANIFEST_NAME} is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise BundleError(f"{MANIFEST_NAME} must contain a JSON object")
    if raw.get("format_version") != FORMAT_VERSION:
        raise BundleError("unsupported package manifest format version")
    if raw.get("archive_root") != ARCHIVE_ROOT:
        raise BundleError(f"manifest archive_root must be {ARCHIVE_ROOT!r}")

    git_identity = raw.get("git")
    if not isinstance(git_identity, dict):
        raise BundleError("manifest git identity is missing")
    commit = git_identity.get("commit")
    tree = git_identity.get("tree")
    if not isinstance(commit, str) or not GIT_OBJECT_PATTERN.fullmatch(commit):
        raise BundleError("manifest Git commit is invalid")
    if not isinstance(tree, str) or not GIT_OBJECT_PATTERN.fullmatch(tree):
        raise BundleError("manifest Git tree is invalid")

    raw_files = raw.get("files")
    if not isinstance(raw_files, list):
        raise BundleError("manifest files must be a list")
    records: list[FileRecord] = []
    for raw_record in raw_files:
        if not isinstance(raw_record, dict):
            raise BundleError("manifest contains a non-object file record")
        path = raw_record.get("path")
        size = raw_record.get("size")
        digest = raw_record.get("sha256")
        origin = raw_record.get("origin")
        mode = raw_record.get("mode")
        if not isinstance(path, str):
            raise BundleError("manifest file path is invalid")
        _validated_relative_path(path)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BundleError(f"manifest size is invalid for {path}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise BundleError(f"manifest sha256 is invalid for {path}")
        if origin not in ALLOWED_ORIGINS:
            raise BundleError(f"manifest origin is invalid for {path}")
        if mode not in REGULAR_GIT_MODES:
            raise BundleError(f"manifest mode is invalid for {path}")
        records.append(
            FileRecord(
                path=path,
                size=size,
                sha256=digest,
                origin=origin,
                mode=mode,
            )
        )

    paths = [record.path for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise BundleError("manifest file paths must be unique and sorted")
    if MANIFEST_NAME in paths or CHECKSUMS_NAME in paths:
        raise BundleError("generated metadata may not appear as a payload file")

    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise BundleError("manifest payload summary is missing")
    if payload.get("file_count") != len(records):
        raise BundleError("manifest payload file_count does not match its files")
    if payload.get("total_bytes") != sum(record.size for record in records):
        raise BundleError("manifest payload total_bytes does not match its files")
    return raw, records


def _parse_checksums(content: bytes) -> dict[str, str]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BundleError(f"{CHECKSUMS_NAME} is not valid UTF-8") from error
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise BundleError(f"{CHECKSUMS_NAME}:{line_number}: blank lines are not allowed")
        if len(line) < 67 or line[64:66] != "  ":
            raise BundleError(f"{CHECKSUMS_NAME}:{line_number}: malformed checksum line")
        digest = line[:64]
        path = line[66:]
        if not SHA256_PATTERN.fullmatch(digest):
            raise BundleError(f"{CHECKSUMS_NAME}:{line_number}: invalid SHA-256")
        _validated_relative_path(path)
        if path in checksums:
            raise BundleError(f"{CHECKSUMS_NAME}: duplicate path {path!r}")
        checksums[path] = digest
    return checksums


def _validate_checksums(
    records: list[FileRecord],
    checksums: dict[str, str],
    manifest_sha256: str,
) -> None:
    expected_paths = {record.path for record in records} | {MANIFEST_NAME}
    if set(checksums) != expected_paths:
        missing = sorted(expected_paths - set(checksums))
        extra = sorted(set(checksums) - expected_paths)
        raise BundleError(f"{CHECKSUMS_NAME} path set mismatch; missing={missing}, extra={extra}")
    for record in records:
        if checksums[record.path] != record.sha256:
            raise BundleError(f"{CHECKSUMS_NAME} disagrees with manifest for {record.path}")
    if checksums[MANIFEST_NAME] != manifest_sha256:
        raise BundleError(f"{CHECKSUMS_NAME} contains the wrong manifest hash")


def _validate_zip_member_name(name: str) -> tuple[str, PurePosixPath]:
    if "\\" in name or "\r" in name or "\n" in name:
        raise BundleError(f"unsafe ZIP member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or len(path.parts) < 2:
        raise BundleError(f"unsafe ZIP member name: {name!r}")
    if path.parts[0] != ARCHIVE_ROOT:
        raise BundleError(f"ZIP member is outside the {ARCHIVE_ROOT!r} root: {name!r}")
    relative = _validated_relative_path(PurePosixPath(*path.parts[1:]).as_posix())
    return path.parts[0], relative


def _zip_mode(info: zipfile.ZipInfo) -> str:
    unix_mode = (info.external_attr >> 16) & 0o177777
    return f"{unix_mode:o}"


def verify_archive(archive_path: Path | str) -> VerificationResult:
    """Verify member safety, manifest metadata, and all archive payload hashes."""

    path = Path(archive_path).resolve()
    if not path.is_file():
        raise BundleError(f"archive does not exist: {path}")

    try:
        with zipfile.ZipFile(path, mode="r", allowZip64=True) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise BundleError("ZIP contains duplicate member names")
            by_relative_path: dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                if info.is_dir():
                    raise BundleError(
                        f"ZIP contains an unexpected directory entry: {info.filename}"
                    )
                if info.flag_bits & 0x1:
                    raise BundleError(f"encrypted ZIP members are not supported: {info.filename}")
                _root, relative = _validate_zip_member_name(info.filename)
                by_relative_path[relative.as_posix()] = info

            if MANIFEST_NAME not in by_relative_path or CHECKSUMS_NAME not in by_relative_path:
                raise BundleError("ZIP is missing package integrity metadata")
            manifest_info = by_relative_path[MANIFEST_NAME]
            checksums_info = by_relative_path[CHECKSUMS_NAME]
            with archive.open(manifest_info, mode="r") as source:
                manifest_content = _read_limited(
                    source,
                    manifest_info.file_size,
                    MANIFEST_NAME,
                )
            raw_manifest, records = _parse_manifest(manifest_content)
            with archive.open(checksums_info, mode="r") as source:
                checksums_content = _read_limited(
                    source,
                    checksums_info.file_size,
                    CHECKSUMS_NAME,
                )
            checksums = _parse_checksums(checksums_content)
            _validate_checksums(
                records,
                checksums,
                hashlib.sha256(manifest_content).hexdigest(),
            )

            expected_members = {record.path for record in records} | {
                MANIFEST_NAME,
                CHECKSUMS_NAME,
            }
            if set(by_relative_path) != expected_members:
                missing = sorted(expected_members - set(by_relative_path))
                extra = sorted(set(by_relative_path) - expected_members)
                raise BundleError(f"ZIP member set mismatch; missing={missing}, extra={extra}")

            for record in records:
                info = by_relative_path[record.path]
                if _zip_mode(info) != record.mode:
                    raise BundleError(f"ZIP mode mismatch for {record.path}")
                if info.file_size != record.size:
                    raise BundleError(f"ZIP size mismatch for {record.path}")
                with archive.open(info, mode="r") as source:
                    size, digest = _hash_stream(source)
                if size != record.size or digest != record.sha256:
                    raise BundleError(f"ZIP payload hash mismatch for {record.path}")
            if _zip_mode(manifest_info) != "100644":
                raise BundleError(f"ZIP mode mismatch for {MANIFEST_NAME}")
            if _zip_mode(checksums_info) != "100644":
                raise BundleError(f"ZIP mode mismatch for {CHECKSUMS_NAME}")
    except zipfile.BadZipFile as error:
        raise BundleError(f"invalid ZIP archive: {error}") from error

    git_identity = raw_manifest["git"]
    payload = raw_manifest["payload"]
    assert isinstance(git_identity, dict)
    assert isinstance(payload, dict)
    return VerificationResult(
        file_count=len(records),
        total_bytes=sum(record.size for record in records),
        git_commit=str(git_identity["commit"]),
        git_tree=str(git_identity["tree"]),
    )


def _read_tree_metadata(path: Path, name: str) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise BundleError(f"{name} is not a regular file")
    with path.open("rb") as source:
        return _read_limited(source, metadata.st_size, name)


def _resolve_tree_root(path: Path) -> Path:
    if path.is_symlink():
        raise BundleError("package tree root may not be a symlink")
    candidate = path.resolve()
    if (candidate / MANIFEST_NAME).is_file():
        return candidate
    nested = candidate / ARCHIVE_ROOT
    if (nested / MANIFEST_NAME).is_file():
        return nested
    raise BundleError(f"could not find {MANIFEST_NAME} in {candidate} or its {ARCHIVE_ROOT} child")


def verify_tree(tree_path: Path | str) -> VerificationResult:
    """Verify an extracted package tree against its embedded integrity metadata."""

    root = _resolve_tree_root(Path(tree_path))
    manifest_path = root / MANIFEST_NAME
    checksums_path = root / CHECKSUMS_NAME
    if not checksums_path.exists():
        raise BundleError(f"package tree is missing {CHECKSUMS_NAME}")

    manifest_content = _read_tree_metadata(manifest_path, MANIFEST_NAME)
    raw_manifest, records = _parse_manifest(manifest_content)
    checksums_content = _read_tree_metadata(checksums_path, CHECKSUMS_NAME)
    checksums = _parse_checksums(checksums_content)
    _validate_checksums(
        records,
        checksums,
        hashlib.sha256(manifest_content).hexdigest(),
    )

    actual_files: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        metadata = candidate.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise BundleError(f"package tree contains a non-regular path: {relative}")
        actual_files.add(_validated_relative_path(relative).as_posix())

    expected_files = {record.path for record in records} | {MANIFEST_NAME, CHECKSUMS_NAME}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise BundleError(f"package tree file set mismatch; missing={missing}, extra={extra}")

    for record in records:
        source_path = root.joinpath(*PurePosixPath(record.path).parts)
        size, digest = _hash_file(source_path)
        if size != record.size or digest != record.sha256:
            raise BundleError(f"package tree payload hash mismatch for {record.path}")

    git_identity = raw_manifest["git"]
    assert isinstance(git_identity, dict)
    return VerificationResult(
        file_count=len(records),
        total_bytes=sum(record.size for record in records),
        git_commit=str(git_identity["commit"]),
        git_tree=str(git_identity["tree"]),
    )


def build_bundle(
    repository_root: Path | str = REPOSITORY_ROOT,
    output_path: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> BuildResult:
    """Build, verify, and atomically publish a deterministic GPU bundle."""

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise BundleError(f"repository root does not exist: {root}")
    output = (
        Path(output_path).resolve()
        if output_path is not None
        else (root / f"{ARCHIVE_ROOT}.zip").resolve()
    )
    if output.suffix.lower() != ".zip":
        raise BundleError("bundle output must use the .zip extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise BundleError(f"bundle output already exists; pass --overwrite to replace it: {output}")

    _ensure_clean_tracked_tree(root)
    commit, tree = _git_identity(root)
    sources = _collect_sources(root)
    if not sources:
        raise BundleError("the bundle source allowlist selected no files")
    if any(source.source_path.resolve() == output for source in sources):
        raise BundleError("bundle output may not overwrite a selected source file")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_archive(temporary_path, sources, commit, tree)
        _fsync_file(temporary_path)
        verification = verify_archive(temporary_path)
        _ensure_clean_tracked_tree(root)
        if _git_identity(root) != (commit, tree):
            raise BundleError("Git HEAD changed while the bundle was being built")
        if output.exists() and not overwrite:
            raise BundleError(
                f"bundle output appeared while building; refusing to replace it: {output}"
            )
        os.replace(temporary_path, output)
        _fsync_directory(output.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    _archive_size, archive_sha256 = _hash_file(output)
    return BuildResult(
        output_path=output,
        archive_sha256=archive_sha256,
        file_count=verification.file_count,
        total_bytes=verification.total_bytes,
        git_commit=verification.git_commit,
        git_tree=verification.git_tree,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the self-contained sion_translate GPU bundle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build and atomically publish the ZIP")
    build_parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="clean Git repository root (default: repository containing this script)",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        help="output ZIP path (default: ROOT/sion_translate.zip)",
    )
    build_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing output ZIP",
    )

    archive_parser = subparsers.add_parser(
        "verify-archive",
        help="verify an existing bundle ZIP",
    )
    archive_parser.add_argument("archive", type=Path)

    tree_parser = subparsers.add_parser(
        "verify-tree",
        help="verify an extracted bundle directory",
    )
    tree_parser.add_argument("tree", type=Path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    parser = _argument_parser()
    parsed = parser.parse_args(arguments)
    try:
        if parsed.command == "build":
            result = build_bundle(
                parsed.root,
                parsed.output,
                overwrite=parsed.overwrite,
            )
            print(f"bundle: {result.output_path}")
            print(f"sha256: {result.archive_sha256}")
            print(f"git commit: {result.git_commit}")
            print(f"payload: {result.file_count} files, {result.total_bytes:,} uncompressed bytes")
        elif parsed.command == "verify-archive":
            result = verify_archive(parsed.archive)
            print(
                f"verified archive: {result.file_count} files, "
                f"{result.total_bytes:,} bytes, commit {result.git_commit}"
            )
        else:
            result = verify_tree(parsed.tree)
            print(
                f"verified tree: {result.file_count} files, "
                f"{result.total_bytes:,} bytes, commit {result.git_commit}"
            )
    except (BundleError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
