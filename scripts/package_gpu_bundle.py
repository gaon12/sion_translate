"""Build and verify a self-contained, reproducible GPU training bundle.

The bundle is intentionally assembled from a narrow set of sources:

* regular files at stage 0 in Git's index, except generated/runtime paths;
* immediate ``data/*.jsonl`` corpus files; and
* regular files below ``data/evaluation_only``.

Nothing else in the working tree is eligible.  In particular, stale artifacts,
checkpoints, virtual environments, caches, and ``data/excluded`` cannot enter
the archive just because they happen to exist beside the source tree.

Four further trees can be added, but only when asked for by name:

``--with-monolingual-corpus``
    The configured ``foundation.corpus_dir`` input, restricted to the
    configured foundation languages.  Unconfigured language directories are
    deliberately excluded.  The corpus can be tens of gigabytes, which is
    exactly why it is not a default.

``--with-tokenizer``
    ``artifacts/tokenizer``.  Training a tokenizer is CPU and RAM work that
    does not touch the GPU, so doing it beforehand and shipping the result
    keeps a rented GPU from idling through it.  ``cli.train`` reuses whatever
    tokenizer already exists, so a complete directory here means the server
    skips that step entirely.  The set is checked for completeness before it
    ships: a partial tokenizer would fail on the server, after the upload.

``--with-dataset``
    ``artifacts/dataset``, the tokenized training shards.  Preparing them is
    more CPU work the GPU cannot help with.  It requires ``--with-tokenizer``,
    because the shards are token ids and mean nothing without the tokenizer
    that produced them -- shipping them alone only wastes the upload.

``--with-foundation-dataset``
    ``artifacts/foundation_dataset``, the prepared denoising/reasoning shards.
    Like the translation dataset, it requires the tokenizer whose token ids it
    stores.  Its manifest inventory is authenticated before any archive is
    published.

All four are recorded in the manifest with their own origin, so
``verify-archive`` and ``verify-tree`` cover them like everything else.
Shipping fourteen gigabytes of corpus outside the manifest would leave the
largest part of the payload with no integrity check at all.

``--prepared-only`` selects the tokenizer and both applicable indexed datasets
as one coherent upload unit, while omitting parallel and monolingual source
corpora even when those files are tracked. The authenticated artifact contracts
remain in the archive, so the GPU host can train without repeating CPU-heavy
tokenization and preparation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable, IO, Iterable, cast
import unicodedata
import zipfile

import numpy as np
from numpy.typing import NDArray
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "sion_translate"
MANIFEST_NAME = "PACKAGE_MANIFEST.json"
CHECKSUMS_NAME = "SHA256SUMS"
FORMAT_VERSION = 2
TRAINING_CONTRACT_SCHEMA = "sion-gpu-training-contract-v1"
TOKENIZER_TRAINING_SCHEMA = "sion-tokenizer-training-v4"
TOKENIZER_INPUT_TRAVERSAL_POLICY = "portable-input-order-v1"
DEFAULT_CONFIG_PATH = "sion_translate.yaml"
DATASET_FINGERPRINT_SCHEMA = "sion-dataset-fingerprint-v2"
DATASET_ARTIFACT_INVENTORY_SCHEMA = "sion-indexed-artifact-inventory-v1"
TRANSLATION_DATASET_FORMAT = "sion-indexed-parallel-v6"
FOUNDATION_DATASET_FORMAT = "sion-foundation-indexed-v3"
FOUNDATION_RELEASE_NAME = "sion"
FOUNDATION_PREPROCESSING_SCHEMA = "foundation-mixed-objectives-v6"
FOUNDATION_SOURCE_IDENTITY_SCHEMA = "corpus-relative-posix-sha256-v1"
FOUNDATION_TOKENIZER_IDENTITY_SCHEMA = "content-sha256-v1"
FOUNDATION_TARGET_STORAGE = "row-shared-source-v1"
FOUNDATION_STORAGE_SIDES = ["src", "tgt"]
FOUNDATION_INDEX_DTYPE = np.dtype(
    [
        ("src_offset", "<u8"),
        ("src_length", "<u4"),
        ("tgt_offset", "<u8"),
        ("tgt_length", "<u4"),
        ("src_register", "u1"),
        ("tgt_register", "u1"),
        ("src_language_id", "<u2"),
        ("tgt_language_id", "<u2"),
        ("source_id", "<u2"),
        ("quality_score", "u1"),
        ("synthetic", "u1"),
        ("forward_only", "u1"),
        ("target_shared", "u1"),
    ]
)
PREPARE_COMPLETION_SCHEMA = "sion-prepare-completion-v1"
TOKENIZER_ROOT_PATH = "artifacts/tokenizer"
TOKENIZER_MODEL_PATH = "artifacts/tokenizer/sion.model"
TOKENIZER_VOCAB_PATH = "artifacts/tokenizer/sion.vocab"
TOKENIZER_METADATA_PATH = "artifacts/tokenizer/tokenizer_metadata.json"
TOKENIZER_FEATURES_PATH = "artifacts/tokenizer/token_features.npz"
TRANSLATION_DATASET_ROOT_PATH = "artifacts/dataset"
DATASET_RAW_FINGERPRINT_PATH = "artifacts/dataset/raw_fingerprint.json"
DATASET_MANIFEST_PATH = "artifacts/dataset/manifest.json"
DATASET_COMPLETION_PATH = "artifacts/dataset/.sion-prepare-complete.json"
FOUNDATION_DATASET_ROOT_PATH = "artifacts/foundation_dataset"
FOUNDATION_DATASET_MANIFEST_PATH = "artifacts/foundation_dataset/manifest.json"
COPY_BUFFER_SIZE = 8 * 1024 * 1024
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MAX_METADATA_SIZE = 64 * 1024 * 1024
MIN_FREE_DISK_RESERVE = 512 * 1024 * 1024
FREE_DISK_RESERVE_DIVISOR = 50

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
ALLOWED_ORIGINS = {
    "git-index",
    "data-jsonl",
    "evaluation-only",
    # Opt-in only. These live under directories the default allowlist refuses,
    # and each is large enough that shipping one by accident is a real cost.
    "monolingual-corpus",
    "tokenizer",
    "dataset",
    "foundation-dataset",
}
ARTIFACT_ORIGIN_ROOTS = {
    "tokenizer": TOKENIZER_ROOT_PATH,
    "dataset": TRANSLATION_DATASET_ROOT_PATH,
    "foundation-dataset": FOUNDATION_DATASET_ROOT_PATH,
}

# Monolingual corpus files the foundation stage can actually read. Mirrors
# ``sion_translate.data.monolingual.ALLOWED_SUFFIXES``: anything else in that
# tree is a stray download, and silently shipping it wastes gigabytes.
MONOLINGUAL_SUFFIXES = {".txt", ".jsonl"}

# A tokenizer is only useful to the training pipeline as a complete set. The
# model alone loads, but ``tokenizer_policy_problem`` then cannot read the
# digit policy or the language tags and the run stops after the upload.
REQUIRED_TOKENIZER_FILES = {
    "sion.model",
    "sion.vocab",
    "token_features.npz",
    "tokenizer_metadata.json",
}
REGULAR_GIT_MODES = {"100644", "100755"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40,64}")
WINDOWS_RESERVED_NAMES = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
WINDOWS_FORBIDDEN_CHARACTERS = frozenset('<>:"|?*')


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
class OmittedSourceFreshness:
    """Hashes and metadata for prepared-artifact sources omitted from the ZIP."""

    identities: dict[str, tuple[int, str]]
    raw_parallel_paths: frozenset[str]
    monolingual_paths: frozenset[str]
    metadata: tuple[tuple[SourceEntry, tuple[int, ...]], ...]


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


@dataclass(frozen=True)
class MonolingualSelection:
    """Repository-relative corpus root and languages authorized by one config."""

    config_path: PurePosixPath
    corpus_root: PurePosixPath
    languages: tuple[str, ...]
    require_all_languages: bool


@dataclass(frozen=True)
class ConfigSelection:
    """Validated project configuration selected for this exact bundle."""

    config_path: PurePosixPath
    config: object


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
    if unicodedata.normalize("NFC", raw_path) != raw_path:
        raise BundleError(f"bundle path is not NFC-normalized: {raw_path!r}")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BundleError(f"unsafe bundle path: {raw_path!r}")
    if path.as_posix() != raw_path:
        raise BundleError(f"non-canonical bundle path: {raw_path!r}")
    for part in path.parts:
        if part.endswith((" ", ".")):
            raise BundleError(f"bundle path has a Windows-ambiguous segment: {raw_path!r}")
        if any(ord(character) < 32 for character in part) or any(
            character in WINDOWS_FORBIDDEN_CHARACTERS for character in part
        ):
            raise BundleError(f"bundle path has a Windows-unsafe segment: {raw_path!r}")
        device_name = part.split(".", 1)[0].casefold()
        if device_name in WINDOWS_RESERVED_NAMES:
            raise BundleError(f"bundle path uses a reserved Windows name: {raw_path!r}")
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


def _repository_relative_path(root: Path, raw_path: object, *, field: str) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path:
        raise BundleError(f"bundle config {field} must be a non-empty path string")
    configured = Path(raw_path)
    if configured.is_absolute():
        raise BundleError(f"bundle config {field} must be repository-relative")
    candidate = (root / configured).resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise BundleError(f"bundle config {field} escapes the repository") from error
    return _validated_relative_path(relative.as_posix())


def _load_validated_project_config(path: Path) -> object:
    """Load the same validated configuration contract used by training."""

    source_root = REPOSITORY_ROOT / "src"
    source_root_text = str(source_root)
    if source_root_text not in sys.path:
        # Running this file directly places scripts/, not src/, on sys.path.
        # Add the repository's source tree so packaging and training share the
        # exact BCP 47 canonicalization and configuration validation rules.
        sys.path.insert(0, source_root_text)
    try:
        from sion_translate.config import load_config

        return load_config(path)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise BundleError(f"could not validate bundle config: {path}: {error}") from error


def _canonical_language(value: object, *, field: str) -> str:
    source_root = REPOSITORY_ROOT / "src"
    source_root_text = str(source_root)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)
    try:
        from sion_translate.language_tags import canonicalize_language_tag

        return canonicalize_language_tag(value, field=field)
    except ValueError as error:
        raise BundleError(str(error)) from error


def _load_config_selection(
    root: Path,
    config_path: Path | str | None,
) -> ConfigSelection:
    raw_config_path = Path(config_path) if config_path is not None else Path(DEFAULT_CONFIG_PATH)
    if raw_config_path.is_absolute():
        resolved_candidate = raw_config_path.resolve(strict=False)
        try:
            candidate_relative = resolved_candidate.relative_to(root)
        except ValueError as error:
            raise BundleError("bundle config path escapes the repository") from error
        config_relative = _validated_relative_path(candidate_relative.as_posix())
    else:
        config_relative = _repository_relative_path(root, str(raw_config_path), field="config path")
    resolved_config = root.joinpath(*config_relative.parts)
    _assert_regular_source(root, resolved_config, config_relative)
    config_object = _load_validated_project_config(resolved_config)
    from sion_translate.config import AppConfig

    if not isinstance(config_object, AppConfig):
        raise BundleError("validated bundle config returned an unexpected object")
    return ConfigSelection(config_path=config_relative, config=config_object)


def _load_monolingual_selection(
    root: Path,
    config_path: Path | str | None,
) -> MonolingualSelection:
    config_selection = _load_config_selection(root, config_path)
    config = config_selection.config
    from sion_translate.config import AppConfig

    if not isinstance(config, AppConfig):
        raise BundleError("validated bundle config returned an unexpected object")
    if not config.foundation.enabled:
        raise BundleError(
            "--with-monolingual-corpus conflicts with foundation.enabled=false in the config"
        )
    selected_languages = config.foundation_languages()
    if not selected_languages:
        raise BundleError("bundle config selects no foundation languages after exclusions")
    corpus_root = _repository_relative_path(
        root,
        config.foundation.corpus_dir,
        field="foundation.corpus_dir",
    )
    return MonolingualSelection(
        config_path=config_selection.config_path,
        corpus_root=corpus_root,
        languages=selected_languages,
        require_all_languages=config.foundation.require_all_languages,
    )


def _tracked_stage_zero_entries(
    root: Path,
    *,
    reserved_roots: tuple[PurePosixPath, ...] = (),
) -> list[SourceEntry]:
    """Collect ordinary tracked files outside separately classified data trees."""

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
        if any(
            relative_path == reserved_root or reserved_root in relative_path.parents
            for reserved_root in reserved_roots
        ):
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


def _metadata_is_link_like(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = int(getattr(metadata, "st_file_attributes", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(reparse_flag and file_attributes & reparse_flag)


def _assert_regular_source(
    root: Path,
    path: Path,
    relative_path: PurePosixPath,
) -> os.stat_result:
    """Reject link-like ancestors and return a stable leaf metadata snapshot."""

    current = root
    for index, part in enumerate(relative_path.parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise BundleError(f"selected source file is missing: {relative_path}") from error
        if _metadata_is_link_like(metadata):
            raise BundleError(
                "selected source path may not contain a symlink, junction, or reparse point: "
                f"{relative_path}"
            )
        leaf = index == len(relative_path.parts) - 1
        if not leaf and not stat.S_ISDIR(metadata.st_mode):
            raise BundleError(f"selected source parent is not a directory: {relative_path}")
        if leaf:
            if not stat.S_ISREG(metadata.st_mode):
                raise BundleError(f"selected source is not a regular file: {relative_path}")
            if current != path:
                raise BundleError(f"selected source path is not canonical: {relative_path}")
            return metadata
    raise BundleError(f"selected source path is empty: {relative_path}")


def _tree_root_is_directory(
    root: Path,
    tree_root: Path,
    origin: str,
) -> bool:
    try:
        relative = PurePosixPath(tree_root.relative_to(root).as_posix())
    except ValueError as error:
        raise BundleError(f"{origin} root escapes the repository: {tree_root}") from error
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return False
        if _metadata_is_link_like(metadata):
            raise BundleError(
                f"{origin} root may not contain a symlink, junction, or reparse point: {relative}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            return False
    return True


def _walk_regular_tree(tree_root: Path, origin: str) -> list[Path]:
    """Walk without following link-like directories on any supported platform."""

    files: list[Path] = []
    pending = [tree_root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise BundleError(
                f"could not inspect {origin} directory {directory}: {error}"
            ) from error
        child_directories: list[Path] = []
        for child in children:
            child_path = Path(child.path)
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as error:
                raise BundleError(
                    f"could not inspect {origin} source {child_path}: {error}"
                ) from error
            if _metadata_is_link_like(metadata):
                raise BundleError(
                    f"{origin} may not contain a symlink, junction, or reparse point: {child_path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                child_directories.append(child_path)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(child_path)
            else:
                raise BundleError(f"{origin} contains a non-regular path: {child_path}")
        pending.extend(reversed(child_directories))
    return sorted(files, key=lambda path: path.as_posix())


def _collect_tree(
    root: Path,
    tree_root: Path,
    origin: str,
    *,
    suffixes: set[str] | None = None,
) -> list[SourceEntry]:
    """Regular files below ``tree_root``, refusing symlinks.

    Symlinks are refused rather than followed because these trees are not in
    Git and nothing else has checked them: a link pointing outside the
    repository would silently pull an arbitrary file into a bundle whose whole
    promise is that its contents are accounted for.
    """

    entries: list[SourceEntry] = []
    if not _tree_root_is_directory(root, tree_root, origin):
        return entries
    for source_path in _walk_regular_tree(tree_root, origin):
        if suffixes is not None and source_path.suffix.lower() not in suffixes:
            continue
        relative_path = _validated_relative_path(source_path.relative_to(root).as_posix())
        entries.append(
            SourceEntry(
                relative_path=relative_path,
                source_path=source_path,
                origin=origin,
                mode="100644",
            )
        )
    return entries


def _collect_configured_parallel_sources(
    root: Path,
    raw_root_relative: PurePosixPath,
) -> list[SourceEntry]:
    """Collect the immediate JSONL files used by translation preparation."""

    data_root = root.joinpath(*raw_root_relative.parts)
    if not _tree_root_is_directory(root, data_root, "data corpus"):
        return []
    entries: list[SourceEntry] = []
    for source_path in sorted(data_root.glob("*.jsonl"), key=lambda path: path.name):
        relative_path = _validated_relative_path(source_path.relative_to(root).as_posix())
        entries.append(
            SourceEntry(
                relative_path=relative_path,
                source_path=source_path,
                origin="data-jsonl",
                mode="100644",
            )
        )
    return entries


def _collect_configured_monolingual_sources(
    root: Path,
    selection: MonolingualSelection,
) -> list[SourceEntry]:
    corpus_root = root.joinpath(*selection.corpus_root.parts)
    if not _tree_root_is_directory(root, corpus_root, "monolingual-corpus"):
        raise BundleError(
            "--with-monolingual-corpus was requested but the configured corpus directory "
            f"does not exist: {selection.corpus_root}"
        )

    configured = {language: language for language in selection.languages}
    matching_directories: dict[str, Path] = {}
    for candidate in sorted(corpus_root.iterdir(), key=lambda path: path.name.casefold()):
        try:
            key = _canonical_language(candidate.name, field="monolingual corpus directory")
        except BundleError:
            continue
        if key not in configured:
            continue
        metadata = candidate.lstat()
        if _metadata_is_link_like(metadata):
            raise BundleError(
                "configured monolingual language directory may not be a symlink, junction, "
                f"or reparse point: {candidate.relative_to(root)}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            continue
        previous = matching_directories.get(key)
        if previous is not None:
            raise BundleError(
                "multiple monolingual directories match one configured language: "
                f"{previous.relative_to(root)} and {candidate.relative_to(root)}"
            )
        matching_directories[key] = candidate

    missing = [language for language in selection.languages if language not in matching_directories]
    if missing and selection.require_all_languages:
        raise BundleError(
            "configured foundation languages have no monolingual corpus directory: "
            f"{', '.join(missing)}"
        )

    entries: list[SourceEntry] = []
    languages_without_files: list[str] = []
    for language in selection.languages:
        directory = matching_directories.get(language)
        if directory is None:
            continue
        language_entries = _collect_tree(
            root,
            directory,
            "monolingual-corpus",
            suffixes=MONOLINGUAL_SUFFIXES,
        )
        if not language_entries:
            languages_without_files.append(language)
            continue
        entries.extend(language_entries)
    if languages_without_files and selection.require_all_languages:
        raise BundleError(
            "configured foundation languages have no readable monolingual files: "
            f"{', '.join(languages_without_files)}"
        )
    if not entries:
        raise BundleError(
            "--with-monolingual-corpus selected no readable files for the configured "
            f"foundation languages: {', '.join(selection.languages)}"
        )
    return entries


def _collect_omitted_source_freshness(
    root: Path,
    config_selection: ConfigSelection,
    *,
    parallel: bool,
    monolingual: bool,
) -> OmittedSourceFreshness | None:
    """Authenticate local sources that prepared artifacts replace in the ZIP."""

    if not parallel and not monolingual:
        return None
    from sion_translate.config import AppConfig

    config = config_selection.config
    if not isinstance(config, AppConfig):
        raise BundleError("validated bundle config returned an unexpected object")
    entries: list[SourceEntry] = []
    if parallel:
        raw_root = _repository_relative_path(root, config.data.raw_dir, field="data.raw_dir")
        parallel_entries = _collect_configured_parallel_sources(root, raw_root)
        if not parallel_entries:
            raise BundleError(
                "prepared artifacts cannot be freshness-checked because the configured "
                f"{raw_root.as_posix()}/*.jsonl sources are absent"
            )
        entries.extend(parallel_entries)
    if monolingual:
        selection = _load_monolingual_selection(
            root,
            config_selection.config_path.as_posix(),
        )
        entries.extend(_collect_configured_monolingual_sources(root, selection))

    identities, metadata = _hash_omitted_sources(root, entries)
    return OmittedSourceFreshness(
        identities=identities,
        raw_parallel_paths=frozenset(
            entry.relative_path.as_posix() for entry in entries if entry.origin == "data-jsonl"
        ),
        monolingual_paths=frozenset(
            entry.relative_path.as_posix()
            for entry in entries
            if entry.origin == "monolingual-corpus"
        ),
        metadata=metadata,
    )


def _parse_json_object(content: bytes, name: str) -> dict[str, object]:
    try:
        decoded = cast(object, json.loads(content.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise BundleError(f"{name} must contain a JSON object")
    return cast(dict[str, object], decoded)


def _fingerprint_tokenizer_hash(fingerprint: dict[str, object], name: str) -> str:
    if fingerprint.get("schema") != DATASET_FINGERPRINT_SCHEMA:
        raise BundleError(f"{name} does not use the current {DATASET_FINGERPRINT_SCHEMA} schema")
    recorded_hash = fingerprint.get("tokenizer_sha256")
    if not isinstance(recorded_hash, str) or not SHA256_PATTERN.fullmatch(recorded_hash):
        raise BundleError(f"{name} has no valid tokenizer_sha256")
    return recorded_hash


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_tokenizer_contract(
    payload_paths: set[str],
    identities: Mapping[str, tuple[int, str]],
    read_payload: Callable[[str], bytes],
) -> str | None:
    tokenizer_prefix = f"{TOKENIZER_ROOT_PATH}/"
    tokenizer_paths = {path for path in payload_paths if path.startswith(tokenizer_prefix)}
    if not tokenizer_paths:
        return None
    expected = {
        TOKENIZER_MODEL_PATH,
        TOKENIZER_VOCAB_PATH,
        TOKENIZER_METADATA_PATH,
        TOKENIZER_FEATURES_PATH,
    }
    if tokenizer_paths != expected:
        missing = sorted(expected - tokenizer_paths)
        unexpected = sorted(tokenizer_paths - expected)
        raise BundleError(
            "tokenizer artifact inventory differs from the complete contract; "
            f"missing={missing}, unexpected={unexpected}"
        )

    metadata = _parse_json_object(
        read_payload(TOKENIZER_METADATA_PATH),
        TOKENIZER_METADATA_PATH,
    )
    version = metadata.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 2:
        raise BundleError("tokenizer metadata must use version 2 or newer")
    if metadata.get("split_digits") is not True:
        raise BundleError("tokenizer metadata must require split_digits=true")
    if metadata.get("model_file") != "sion.model":
        raise BundleError("tokenizer metadata model_file must be 'sion.model'")
    if metadata.get("vocab_file") != "sion.vocab":
        raise BundleError("tokenizer metadata vocab_file must be 'sion.vocab'")
    if metadata.get("token_features_file") != "token_features.npz":
        raise BundleError("tokenizer metadata must require the token_features.npz sidecar")

    checks = (
        (TOKENIZER_MODEL_PATH, "model_sha256", None),
        (TOKENIZER_VOCAB_PATH, "vocab_sha256", None),
        (TOKENIZER_FEATURES_PATH, "token_features_sha256", "token_features_size"),
    )
    for path, hash_field, size_field in checks:
        size, digest = identities[path]
        recorded_digest = metadata.get(hash_field)
        if (
            not isinstance(recorded_digest, str)
            or not SHA256_PATTERN.fullmatch(recorded_digest)
            or recorded_digest != digest
        ):
            raise BundleError(f"tokenizer metadata {hash_field} does not match {path}")
        if size_field is not None:
            recorded_size = metadata.get(size_field)
            if (
                isinstance(recorded_size, bool)
                or not isinstance(recorded_size, int)
                or recorded_size != size
            ):
                raise BundleError(f"tokenizer metadata {size_field} does not match {path}")
    raw_training_contract = metadata.get("training_contract")
    if not isinstance(raw_training_contract, Mapping):
        raise BundleError("tokenizer metadata has no authenticated training_contract")
    training_contract = dict(cast(Mapping[str, object], raw_training_contract))
    if training_contract.get("schema") != TOKENIZER_TRAINING_SCHEMA:
        raise BundleError(f"tokenizer training contract must use {TOKENIZER_TRAINING_SCHEMA}")
    if training_contract.get("input_traversal_policy") != TOKENIZER_INPUT_TRAVERSAL_POLICY:
        raise BundleError("tokenizer training contract has an obsolete input traversal policy")
    training_contract_sha256 = hashlib.sha256(_canonical_json_bytes(training_contract)).hexdigest()
    if metadata.get("training_contract_sha256") != training_contract_sha256:
        raise BundleError("tokenizer training contract digest does not match its payload")
    return identities[TOKENIZER_MODEL_PATH][1]


def _validated_artifact_inventory(
    manifest: Mapping[str, object],
    *,
    dataset_root: str,
) -> dict[str, tuple[int, str]]:
    raw_inventory = manifest.get("artifact_inventory")
    if not isinstance(raw_inventory, Mapping):
        raise BundleError(f"{dataset_root}/manifest.json has no artifact inventory")
    inventory = cast(Mapping[object, object], raw_inventory)
    if set(inventory) != {"schema", "files"}:
        raise BundleError(f"{dataset_root} artifact inventory has unexpected fields")
    if inventory.get("schema") != DATASET_ARTIFACT_INVENTORY_SCHEMA:
        raise BundleError(f"{dataset_root} artifact inventory has an unsupported schema")
    raw_files = inventory.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise BundleError(f"{dataset_root} artifact inventory has no files")

    files: dict[str, tuple[int, str]] = {}
    ordered_paths: list[str] = []
    for raw_entry in cast(list[object], raw_files):
        if not isinstance(raw_entry, Mapping):
            raise BundleError(f"{dataset_root} artifact inventory contains a non-object entry")
        entry = cast(Mapping[object, object], raw_entry)
        if set(entry) != {"path", "size", "sha256"}:
            raise BundleError(f"{dataset_root} artifact inventory entry has unexpected fields")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            raise BundleError(f"{dataset_root} artifact inventory path must be a string")
        relative = _validated_relative_path(raw_path)
        if relative.parts[0] not in {"train", "validation", "test"}:
            raise BundleError(f"{dataset_root} artifact inventory path is outside a split")
        size = entry.get("size")
        digest = entry.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleError(f"{dataset_root} artifact inventory size is invalid for {raw_path}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise BundleError(
                f"{dataset_root} artifact inventory SHA-256 is invalid for {raw_path}"
            )
        if raw_path in files:
            raise BundleError(f"{dataset_root} artifact inventory repeats {raw_path}")
        files[raw_path] = (size, digest)
        ordered_paths.append(raw_path)
    if ordered_paths != sorted(ordered_paths):
        raise BundleError(f"{dataset_root} artifact inventory paths must be sorted")
    return files


def _foundation_preprocessing_options(manifest: Mapping[str, object]) -> dict[str, object]:
    raw_options = manifest.get("preprocessing_options")
    if not isinstance(raw_options, Mapping):
        raise BundleError("foundation manifest has no valid preprocessing_options object")
    option_values = cast(Mapping[object, object], raw_options)
    if any(not isinstance(key, str) for key in option_values):
        raise BundleError("foundation preprocessing_options keys must be strings")
    options = {cast(str, key): value for key, value in option_values.items()}
    expected_fields = {
        "deduplicate",
        "deduplication_backend",
        "maximum_characters",
        "max_tokens",
        "max_target_tokens",
        "minimum_characters",
        "reasoning_sample_share",
        "shard_size",
        "validation_fraction",
    }
    if set(options) != expected_fields:
        raise BundleError("foundation preprocessing_options fields do not match v6")

    deduplicate = options["deduplicate"]
    backend = options["deduplication_backend"]
    if not isinstance(deduplicate, bool):
        raise BundleError("foundation deduplicate option must be a boolean")
    expected_backend = "sqlite-blake2b-128-v1" if deduplicate else "disabled"
    if backend != expected_backend:
        raise BundleError("foundation deduplication backend contradicts its enabled flag")

    integer_fields = (
        "minimum_characters",
        "maximum_characters",
        "max_tokens",
        "max_target_tokens",
        "shard_size",
    )
    integers: dict[str, int] = {}
    for field in integer_fields:
        value = options[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BundleError(f"foundation preprocessing option {field} must be positive")
        integers[field] = value
    if integers["maximum_characters"] <= integers["minimum_characters"]:
        raise BundleError("foundation maximum_characters must exceed minimum_characters")
    if integers["max_target_tokens"] < 6:
        raise BundleError("foundation max_target_tokens is too small for trace markers")

    reasoning_share = options["reasoning_sample_share"]
    if (
        isinstance(reasoning_share, bool)
        or not isinstance(reasoning_share, (int, float))
        or not math.isfinite(float(reasoning_share))
        or not 0.0 <= float(reasoning_share) <= 0.10
    ):
        raise BundleError("foundation preprocessing option reasoning_sample_share is invalid")
    validation_fraction = options["validation_fraction"]
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(float(validation_fraction))
        or not 0.0 < float(validation_fraction) < 0.5
    ):
        raise BundleError("foundation preprocessing option validation_fraction is invalid")
    return options


def _foundation_source_contract(
    manifest: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    raw_languages = manifest.get("languages")
    language_values = cast(list[object], raw_languages) if isinstance(raw_languages, list) else []
    if (
        not isinstance(raw_languages, list)
        or not raw_languages
        or any(not isinstance(language, str) or not language for language in language_values)
    ):
        raise BundleError("foundation manifest languages must be a non-empty string list")
    languages = cast(list[str], language_values)
    if len(languages) != len(set(languages)):
        raise BundleError("foundation manifest languages must be unique")
    expected_language_to_id = {language: index for index, language in enumerate(languages)}
    if manifest.get("language_to_id") != expected_language_to_id:
        raise BundleError("foundation manifest language_to_id is not contiguous")
    if manifest.get("language_pairs") != [[language, language] for language in languages]:
        raise BundleError("foundation language pairs must be self-denoising pairs")
    if manifest.get("source_only_languages") != []:
        raise BundleError("foundation manifest may not declare source-only languages")

    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise BundleError("foundation manifest has no source tasks")
    tasks: list[str] = []
    records: list[int] = []
    language_ids: list[int] = []
    source_identities: list[dict[str, object]] = []
    expected_source_fields = {
        "id",
        "language",
        "logical_path",
        "name",
        "records",
        "sha256",
        "size_bytes",
        "task",
    }
    for source_id, raw_source in enumerate(cast(list[object], raw_sources)):
        if not isinstance(raw_source, Mapping):
            raise BundleError("foundation source metadata must contain objects")
        source_values = cast(Mapping[object, object], raw_source)
        if any(not isinstance(key, str) for key in source_values):
            raise BundleError("foundation source metadata keys must be strings")
        source = {cast(str, key): value for key, value in source_values.items()}
        if set(source) != expected_source_fields or source.get("id") != source_id:
            raise BundleError("foundation source ids and fields do not match v3")
        language = source.get("language")
        if not isinstance(language, str) or language not in expected_language_to_id:
            raise BundleError(f"foundation source {source_id} has an invalid language")
        task = source.get("task")
        if task not in {"denoising", "reasoning"}:
            raise BundleError(f"foundation source {source_id} has an invalid task")
        record_count = source.get("records")
        size_bytes = source.get("size_bytes")
        digest = source.get("sha256")
        logical_path = source.get("logical_path")
        name = source.get("name")
        if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 0:
            raise BundleError(f"foundation source {source_id} has an invalid record count")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise BundleError(f"foundation source {source_id} has an invalid byte size")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise BundleError(f"foundation source {source_id} has an invalid SHA-256")
        if not isinstance(logical_path, str):
            raise BundleError(f"foundation source {source_id} has an invalid logical path")
        validated_logical_path = _validated_relative_path(logical_path)
        if not isinstance(name, str) or name != validated_logical_path.name:
            raise BundleError(f"foundation source {source_id} name disagrees with its path")
        tasks.append(cast(str, task))
        records.append(record_count)
        language_ids.append(expected_language_to_id[language])
        source_identities.append(
            {
                "language": language,
                "logical_path": logical_path,
                "sha256": digest,
                "size_bytes": size_bytes,
                "task": task,
            }
        )

    sources_digest = hashlib.sha256(_canonical_json_bytes(source_identities)).hexdigest()
    if manifest.get("sources_sha256") != sources_digest:
        raise BundleError("foundation aggregate source identity is invalid")
    return tuple(tasks), tuple(records), tuple(language_ids)


def _foundation_shard_groups(
    inventory: Mapping[str, tuple[int, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    groups: dict[tuple[str, str], dict[str, str]] = {}
    pattern = re.compile(r"(?P<prefix>[0-9]{5,})\.(?P<kind>idx\.npy|src\.bin|tgt\.bin)")
    for raw_path in inventory:
        relative = PurePosixPath(raw_path)
        if len(relative.parts) != 2 or relative.parts[0] not in {"train", "validation"}:
            raise BundleError(f"foundation inventory contains a non-shard path: {raw_path}")
        match = pattern.fullmatch(relative.name)
        if match is None:
            raise BundleError(f"foundation inventory has an invalid shard name: {raw_path}")
        key = (relative.parts[0], match.group("prefix"))
        groups.setdefault(key, {})[match.group("kind")] = raw_path

    expected_kinds = {"idx.npy", "src.bin", "tgt.bin"}
    for (split, prefix), members in groups.items():
        if set(members) != expected_kinds:
            raise BundleError(
                f"foundation shard {split}/{prefix} is incomplete: "
                f"missing={sorted(expected_kinds - set(members))}"
            )
    if not any(split == "train" for split, _prefix in groups):
        raise BundleError("foundation inventory has no training shards")
    for split in ("train", "validation"):
        prefixes = sorted(prefix for candidate, prefix in groups if candidate == split)
        expected = [f"{index:05d}" for index in range(len(prefixes))]
        if prefixes != expected:
            raise BundleError(f"foundation {split} shard sequence is not contiguous")
    return groups


def _validate_foundation_sampling_contract(
    manifest: Mapping[str, object],
    source_tasks: tuple[str, ...],
    source_records: tuple[int, ...],
    source_language_ids: tuple[int, ...],
    preprocessing_options: Mapping[str, object],
) -> None:
    languages = cast(list[str], manifest["languages"])
    expected_counts = {language: 0 for language in languages}
    for record_count, language_id in zip(source_records, source_language_ids, strict=True):
        expected_counts[languages[language_id]] += record_count

    raw_sampling = manifest.get("language_sampling")
    if not isinstance(raw_sampling, Mapping):
        raise BundleError("foundation manifest has no usable language_sampling policy")
    sampling_values = cast(Mapping[str, object], raw_sampling)
    if set(sampling_values) != {"alpha", "minimum_share", "weights", "counts", "warnings"}:
        raise BundleError("foundation language_sampling fields are invalid")
    alpha = sampling_values.get("alpha")
    minimum_share = sampling_values.get("minimum_share")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or not 0.0 < float(alpha) <= 1.0
    ):
        raise BundleError("foundation language_sampling alpha is invalid")
    if (
        isinstance(minimum_share, bool)
        or not isinstance(minimum_share, (int, float))
        or not math.isfinite(float(minimum_share))
        or not 0.0 <= float(minimum_share) < 1.0
    ):
        raise BundleError("foundation language_sampling minimum_share is invalid")
    if sampling_values.get("counts") != expected_counts:
        raise BundleError("foundation language_sampling counts disagree with source records")

    raw_weights = sampling_values.get("weights")
    if not isinstance(raw_weights, Mapping):
        raise BundleError("foundation language_sampling weights must be an object")
    weight_values = cast(Mapping[str, object], raw_weights)
    if set(weight_values) != set(languages):
        raise BundleError("foundation language_sampling weight languages are invalid")
    scaled = {
        language: math.pow(float(count), float(alpha))
        for language, count in expected_counts.items()
        if count > 0
    }
    scaled_total = sum(scaled.values())
    for language in languages:
        raw_weight = weight_values.get(language)
        expected_weight = scaled.get(language, 0.0) / scaled_total if scaled_total > 0.0 else 0.0
        if (
            isinstance(raw_weight, bool)
            or not isinstance(raw_weight, (int, float))
            or not math.isfinite(float(raw_weight))
            or not math.isclose(
                float(raw_weight),
                expected_weight,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise BundleError(f"foundation language_sampling weight is invalid for {language}")
    raw_warnings = sampling_values.get("warnings")
    if not isinstance(raw_warnings, list) or any(
        not isinstance(warning, str) for warning in cast(list[object], raw_warnings)
    ):
        raise BundleError("foundation language_sampling warnings must be strings")

    raw_reasoning = manifest.get("reasoning")
    if not isinstance(raw_reasoning, Mapping):
        raise BundleError("foundation manifest has no usable reasoning policy")
    reasoning_values = cast(Mapping[str, object], raw_reasoning)
    if set(reasoning_values) != {
        "contract",
        "languages",
        "records",
        "sample_share",
        "trace_symbols",
    }:
        raise BundleError("foundation reasoning policy fields are invalid")
    reasoning_languages = list(
        dict.fromkeys(
            languages[language_id]
            for task, language_id in zip(source_tasks, source_language_ids, strict=True)
            if task == "reasoning"
        )
    )
    reasoning_records = sum(
        record_count
        for task, record_count in zip(source_tasks, source_records, strict=True)
        if task == "reasoning"
    )
    if (
        reasoning_values.get("contract") != "prompt-to-delimited-trace-v1"
        or reasoning_values.get("languages") != reasoning_languages
        or reasoning_values.get("records") != reasoning_records
        or reasoning_values.get("sample_share") != preprocessing_options["reasoning_sample_share"]
        or reasoning_values.get("trace_symbols") != ["<think>", "</think>", "<answer>", "</answer>"]
    ):
        raise BundleError("foundation reasoning policy contradicts its source tasks")
    expected_objective = (
        "span-corruption-denoising+structured-reasoning"
        if reasoning_records > 0
        else "span-corruption-denoising"
    )
    if manifest.get("objective") != expected_objective:
        raise BundleError("foundation objective contradicts its indexed reasoning rows")


def _load_foundation_index(content: bytes, path: str) -> NDArray[np.void]:
    try:
        raw_loaded: object = np.load(io.BytesIO(content), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise BundleError(f"foundation index cannot be read: {path}") from error
    if not isinstance(raw_loaded, np.ndarray):
        raise BundleError(f"foundation index is not an array: {path}")
    loaded = cast(NDArray[np.void], raw_loaded)
    if loaded.ndim != 1 or loaded.dtype != FOUNDATION_INDEX_DTYPE:
        raise BundleError(f"foundation index dtype or shape is invalid: {path}")
    if len(loaded) == 0:
        raise BundleError(f"foundation index shard is empty: {path}")
    return loaded


def _validate_foundation_dataset_contract(
    *,
    manifest: Mapping[str, object],
    inventory: Mapping[str, tuple[int, str]],
    identities: Mapping[str, tuple[int, str]],
    read_payload: Callable[[str], bytes],
    tokenizer_sha256: str,
) -> None:
    if manifest.get("stage") != "foundation":
        raise BundleError("foundation manifest has an invalid stage marker")
    release_name = manifest.get("release_name")
    if (
        not isinstance(release_name, str)
        or not release_name
        or release_name != release_name.strip()
        or not release_name.isascii()
    ):
        raise BundleError("foundation manifest release_name must be normalized non-empty ASCII")
    if manifest.get("preprocessing_schema") != FOUNDATION_PREPROCESSING_SCHEMA:
        raise BundleError(f"foundation manifest does not use {FOUNDATION_PREPROCESSING_SCHEMA}")
    preprocessing_options = _foundation_preprocessing_options(manifest)
    if manifest.get("storage_sides") != FOUNDATION_STORAGE_SIDES:
        raise BundleError("foundation storage_sides marker is invalid")
    if manifest.get("target_storage") != FOUNDATION_TARGET_STORAGE:
        raise BundleError("foundation target_storage marker is invalid")
    expected_dtype = json.loads(json.dumps(FOUNDATION_INDEX_DTYPE.descr))
    if manifest.get("index_dtype") != expected_dtype:
        raise BundleError("foundation index_dtype marker does not match shared-target v3")
    if manifest.get("source_identity_schema") != FOUNDATION_SOURCE_IDENTITY_SCHEMA:
        raise BundleError("foundation source identity schema is obsolete")

    tokenizer_identity = manifest.get("tokenizer_identity")
    model_size, _model_digest = identities[TOKENIZER_MODEL_PATH]
    expected_tokenizer_identity = {
        "schema": FOUNDATION_TOKENIZER_IDENTITY_SCHEMA,
        "size_bytes": model_size,
        "sha256": tokenizer_sha256,
    }
    if tokenizer_identity != expected_tokenizer_identity:
        raise BundleError("foundation tokenizer_identity does not authenticate sion.model")
    if manifest.get("tokenizer_model") != "sion.model":
        raise BundleError("foundation tokenizer_model must be the portable sion.model name")
    if manifest.get("fingerprint") != {"tokenizer_sha256": tokenizer_sha256}:
        raise BundleError("foundation fingerprint disagrees with its tokenizer identity")

    source_tasks, expected_source_records, source_language_ids = _foundation_source_contract(
        manifest
    )
    _validate_foundation_sampling_contract(
        manifest,
        source_tasks,
        expected_source_records,
        source_language_ids,
        preprocessing_options,
    )
    shard_groups = _foundation_shard_groups(inventory)
    observed_source_records = np.zeros(len(source_tasks), dtype=np.uint64)
    observed_split_records = {"train": 0, "validation": 0}
    dataset_prefix = f"{FOUNDATION_DATASET_ROOT_PATH}/"
    shard_size = cast(int, preprocessing_options["shard_size"])

    for (split, _prefix), members in sorted(shard_groups.items()):
        index_relative = members["idx.npy"]
        index_path = f"{dataset_prefix}{index_relative}"
        index = _load_foundation_index(read_payload(index_path), index_path)
        if len(index) > shard_size:
            raise BundleError(f"foundation shard exceeds configured shard_size: {index_path}")

        source_ids = np.asarray(index["source_id"], dtype=np.int64)
        if int(source_ids.min()) < 0 or int(source_ids.max()) >= len(source_tasks):
            raise BundleError(f"foundation source_id is out of range: {index_path}")
        observed_source_records += np.bincount(
            source_ids,
            minlength=len(source_tasks),
        )[: len(source_tasks)].astype(np.uint64)
        observed_split_records[split] += len(index)

        source_languages = np.asarray(index["src_language_id"], dtype=np.int64)
        target_languages = np.asarray(index["tgt_language_id"], dtype=np.int64)
        expected_languages = np.asarray(source_language_ids, dtype=np.int64)[source_ids]
        if not np.array_equal(source_languages, expected_languages) or not np.array_equal(
            target_languages,
            expected_languages,
        ):
            raise BundleError(f"foundation index language ids disagree with sources: {index_path}")
        if not bool((np.asarray(index["forward_only"], dtype=np.uint8) == 1).all()):
            raise BundleError(f"foundation index contains a reverse-enabled row: {index_path}")

        target_shared = np.asarray(index["target_shared"], dtype=np.uint8)
        if not bool(np.isin(target_shared, (0, 1)).all()):
            raise BundleError(f"foundation target_shared flag is invalid: {index_path}")
        shared_mask = target_shared.astype(np.bool_)
        expected_shared = np.fromiter(
            (source_tasks[int(source_id)] == "denoising" for source_id in source_ids.tolist()),
            dtype=np.bool_,
            count=len(source_ids),
        )
        if not np.array_equal(shared_mask, expected_shared):
            raise BundleError(f"foundation target aliases disagree with source tasks: {index_path}")

        source_offsets = np.asarray(index["src_offset"], dtype=np.uint64)
        source_lengths = np.asarray(index["src_length"], dtype=np.uint64)
        target_offsets = np.asarray(index["tgt_offset"], dtype=np.uint64)
        target_lengths = np.asarray(index["tgt_length"], dtype=np.uint64)
        expected_source_offsets = np.concatenate(
            (np.zeros(1, dtype=np.uint64), np.cumsum(source_lengths[:-1], dtype=np.uint64))
        )
        stored_target_lengths = np.where(shared_mask, 0, target_lengths)
        expected_target_offsets = np.concatenate(
            (
                np.zeros(1, dtype=np.uint64),
                np.cumsum(stored_target_lengths[:-1], dtype=np.uint64),
            )
        )
        if not np.array_equal(source_offsets, expected_source_offsets) or not np.array_equal(
            target_offsets,
            expected_target_offsets,
        ):
            raise BundleError(f"foundation token offsets are not contiguous: {index_path}")
        if bool(shared_mask.any()) and (
            not np.array_equal(source_lengths[shared_mask], target_lengths[shared_mask])
            or not np.array_equal(
                np.asarray(index["src_register"])[shared_mask],
                np.asarray(index["tgt_register"])[shared_mask],
            )
            or not bool((np.asarray(index["synthetic"], dtype=np.uint8)[shared_mask] == 0).all())
        ):
            raise BundleError(f"foundation shared targets contradict source rows: {index_path}")

        source_relative = members["src.bin"]
        target_relative = members["tgt.bin"]
        expected_source_size = int(source_lengths.sum(dtype=np.uint64)) * 4
        expected_target_size = int(stored_target_lengths.sum(dtype=np.uint64)) * 4
        if inventory[source_relative][0] != expected_source_size:
            raise BundleError(f"foundation source payload size is invalid: {source_relative}")
        if inventory[target_relative][0] != expected_target_size:
            raise BundleError(f"foundation target payload size is invalid: {target_relative}")

    if observed_source_records.tolist() != list(expected_source_records):
        raise BundleError("foundation source record counts disagree with indexed rows")
    stats = manifest.get("stats")
    if not isinstance(stats, Mapping):
        raise BundleError("foundation manifest has no valid stats object")
    stat_values = cast(Mapping[str, object], stats)
    for split, field in (("train", "train_records"), ("validation", "validation_records")):
        value = stat_values.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != observed_split_records[split]
        ):
            raise BundleError(f"foundation {field} disagrees with indexed rows")


def _validate_dataset_contract(
    *,
    payload_paths: set[str],
    identities: Mapping[str, tuple[int, str]],
    read_payload: Callable[[str], bytes],
    dataset_root: str,
    manifest_path: str,
    expected_format: str,
    tokenizer_sha256: str | None,
    translation: bool,
) -> None:
    prefix = f"{dataset_root}/"
    dataset_paths = {path for path in payload_paths if path.startswith(prefix)}
    if not dataset_paths:
        return
    if tokenizer_sha256 is None:
        raise BundleError(f"{dataset_root} requires the complete tokenizer in the same payload")
    if manifest_path not in dataset_paths:
        raise BundleError(f"{dataset_root} is missing manifest.json")

    manifest = _parse_json_object(read_payload(manifest_path), manifest_path)
    if manifest.get("format") != expected_format:
        raise BundleError(f"{manifest_path} does not use {expected_format}")
    inventory = _validated_artifact_inventory(manifest, dataset_root=dataset_root)

    sidecars = {manifest_path}
    if translation:
        sidecars.update({DATASET_RAW_FINGERPRINT_PATH, DATASET_COMPLETION_PATH})
    missing_sidecars = sorted(sidecars - dataset_paths)
    if missing_sidecars:
        raise BundleError(f"{dataset_root} is missing required sidecars: {missing_sidecars}")

    indexed_paths = dataset_paths - sidecars
    expected_indexed_paths = {f"{dataset_root}/{path}" for path in inventory}
    if indexed_paths != expected_indexed_paths:
        missing = sorted(expected_indexed_paths - indexed_paths)
        unexpected = sorted(indexed_paths - expected_indexed_paths)
        raise BundleError(
            f"{dataset_root} file set differs from its artifact inventory; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative, expected_identity in inventory.items():
        path = f"{dataset_root}/{relative}"
        if identities[path] != expected_identity:
            raise BundleError(f"{dataset_root} artifact identity mismatch for {relative}")

    if translation:
        raw_fingerprint = _parse_json_object(
            read_payload(DATASET_RAW_FINGERPRINT_PATH),
            DATASET_RAW_FINGERPRINT_PATH,
        )
        manifest_fingerprint = manifest.get("fingerprint")
        if not isinstance(manifest_fingerprint, dict):
            raise BundleError(f"{manifest_path} has no valid fingerprint object")
        if raw_fingerprint != manifest_fingerprint:
            raise BundleError("prepared dataset fingerprint sidecar disagrees with its manifest")
        recorded_hash = _fingerprint_tokenizer_hash(raw_fingerprint, DATASET_RAW_FINGERPRINT_PATH)
        completion = _parse_json_object(
            read_payload(DATASET_COMPLETION_PATH),
            DATASET_COMPLETION_PATH,
        )
        expected_completion = {
            "schema": PREPARE_COMPLETION_SCHEMA,
            "manifest_sha256": identities[manifest_path][1],
            "raw_fingerprint_sha256": identities[DATASET_RAW_FINGERPRINT_PATH][1],
            "artifact_inventory_sha256": hashlib.sha256(
                _canonical_json_bytes(manifest["artifact_inventory"])
            ).hexdigest(),
        }
        if completion != expected_completion:
            raise BundleError(
                "prepared dataset completion marker does not authenticate its sidecars"
            )
    else:
        recorded_hash = manifest.get("tokenizer_sha256")
        if not isinstance(recorded_hash, str) or not SHA256_PATTERN.fullmatch(recorded_hash):
            raise BundleError(f"{manifest_path} has no valid tokenizer_sha256")
        if recorded_hash != tokenizer_sha256:
            raise BundleError(
                f"{dataset_root} tokenizer mismatch: manifest records {recorded_hash}, "
                f"but {TOKENIZER_MODEL_PATH} is {tokenizer_sha256}"
            )
        _validate_foundation_dataset_contract(
            manifest=manifest,
            inventory=inventory,
            identities=identities,
            read_payload=read_payload,
            tokenizer_sha256=tokenizer_sha256,
        )

    if recorded_hash != tokenizer_sha256:
        raise BundleError(
            f"{dataset_root} tokenizer mismatch: manifest records {recorded_hash}, "
            f"but {TOKENIZER_MODEL_PATH} is {tokenizer_sha256}"
        )


def _validate_artifact_contracts(
    payload_paths: set[str],
    identities: Mapping[str, tuple[int, str]],
    read_payload: Callable[[str], bytes],
) -> None:
    tokenizer_sha256 = _validate_tokenizer_contract(payload_paths, identities, read_payload)
    _validate_dataset_contract(
        payload_paths=payload_paths,
        identities=identities,
        read_payload=read_payload,
        dataset_root=TRANSLATION_DATASET_ROOT_PATH,
        manifest_path=DATASET_MANIFEST_PATH,
        expected_format=TRANSLATION_DATASET_FORMAT,
        tokenizer_sha256=tokenizer_sha256,
        translation=True,
    )
    _validate_dataset_contract(
        payload_paths=payload_paths,
        identities=identities,
        read_payload=read_payload,
        dataset_root=FOUNDATION_DATASET_ROOT_PATH,
        manifest_path=FOUNDATION_DATASET_MANIFEST_PATH,
        expected_format=FOUNDATION_DATASET_FORMAT,
        tokenizer_sha256=tokenizer_sha256,
        translation=False,
    )


def _validated_config_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BundleError(f"bundle config {field} must be a non-empty path string")
    path = Path(value)
    if path.is_absolute():
        raise BundleError(f"bundle config {field} must be repository-relative")
    return _validated_relative_path(path.as_posix()).as_posix()


def _validated_config_from_payload(content: bytes, name: str) -> object:
    try:
        decoded = cast(object, yaml.safe_load(content.decode("utf-8")))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise BundleError(f"{name} is not valid UTF-8 YAML") from error
    if decoded is None:
        raw: dict[str, object] = {}
    elif isinstance(decoded, dict) and all(isinstance(key, str) for key in decoded):
        raw = cast(dict[str, object], decoded)
    else:
        raise BundleError(f"{name} must contain a YAML mapping")
    source_root = REPOSITORY_ROOT / "src"
    source_root_text = str(source_root)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)
    try:
        from sion_translate.config import config_from_raw

        return config_from_raw(raw)
    except (TypeError, ValueError) as error:
        raise BundleError(f"could not validate bundled training config {name}: {error}") from error


def _training_contract_payload(
    config_path: PurePosixPath,
    config_sha256: str,
    config: object,
    *,
    raw_parallel_data_included: bool,
) -> dict[str, object]:
    from sion_translate.config import AppConfig

    if not isinstance(config, AppConfig):
        raise BundleError("validated bundle config returned an unexpected object")
    if config_path != PurePosixPath(DEFAULT_CONFIG_PATH):
        raise BundleError(
            "GPU bundles require sion_translate.yaml as the selected config so the "
            "default sion-train command cannot silently load a different file"
        )
    paths = {
        "raw_dir": _validated_config_relative_path(config.data.raw_dir, field="data.raw_dir"),
        "tokenizer_model": _validated_config_relative_path(
            config.data.tokenizer_model,
            field="data.tokenizer_model",
        ),
        "tokenizer_features": _validated_config_relative_path(
            config.data.tokenizer_features,
            field="data.tokenizer_features",
        ),
        "translation_dataset": _validated_config_relative_path(
            config.data.dataset_dir,
            field="data.dataset_dir",
        ),
        "foundation_dataset": _validated_config_relative_path(
            config.foundation.dataset_dir,
            field="foundation.dataset_dir",
        ),
    }
    expected_paths = {
        "raw_dir": "data",
        "tokenizer_model": TOKENIZER_MODEL_PATH,
        "tokenizer_features": TOKENIZER_FEATURES_PATH,
        "translation_dataset": TRANSLATION_DATASET_ROOT_PATH,
        "foundation_dataset": FOUNDATION_DATASET_ROOT_PATH,
    }
    if paths != expected_paths:
        mismatches = {
            field: {"config": paths[field], "bundle": expected}
            for field, expected in expected_paths.items()
            if paths[field] != expected
        }
        raise BundleError(
            "GPU bundle artifact collection currently requires the canonical repository paths; "
            f"configured path mismatches={mismatches}"
        )
    return {
        "schema": TRAINING_CONTRACT_SCHEMA,
        "config_path": config_path.as_posix(),
        "config_sha256": config_sha256,
        "raw_parallel_data_included": raw_parallel_data_included,
        "language_pairs": [list(pair) for pair in config.data.configured_language_pairs()],
        "translation_directions": [
            list(direction) for direction in config.data.configured_translation_directions()
        ],
        "source_only_languages": list(config.data.configured_source_only_languages()),
        "foundation_enabled": config.foundation.enabled,
        "foundation_languages": list(config.foundation_languages()),
        "paths": paths,
    }


def _validated_tokenizer_source_records(
    raw_sources: object,
) -> dict[tuple[str, str], tuple[int, str, str | None]]:
    """Normalize the source identities authenticated by tokenizer training."""

    if not isinstance(raw_sources, list) or not raw_sources:
        raise BundleError("tokenizer training contract has no source identities")
    records: dict[tuple[str, str], tuple[int, str, str | None]] = {}
    for raw_source in cast(list[object], raw_sources):
        if not isinstance(raw_source, Mapping):
            raise BundleError("tokenizer training contract contains a non-object source")
        source = cast(Mapping[object, object], raw_source)
        role = source.get("role")
        if role not in {"parallel", "monolingual"}:
            raise BundleError("tokenizer training contract source role is invalid")
        expected_fields = {"role", "path", "size", "sha256"}
        if role == "monolingual":
            expected_fields.add("language")
        if set(source) != expected_fields:
            raise BundleError("tokenizer training contract source fields are invalid")
        raw_path = source.get("path")
        if not isinstance(raw_path, str):
            raise BundleError("tokenizer training contract source path must be a string")
        path = _validated_relative_path(raw_path).as_posix()
        size = source.get("size")
        digest = source.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleError(f"tokenizer training source size is invalid: {path}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise BundleError(f"tokenizer training source SHA-256 is invalid: {path}")
        language: str | None = None
        if role == "monolingual":
            language = _canonical_language(
                source.get("language"),
                field=f"tokenizer source language for {path}",
            )
        key = (role, path)
        if key in records:
            raise BundleError(f"tokenizer training contract repeats source {role}:{path}")
        records[key] = (size, digest, language)
    return records


def _validated_fingerprint_file_identities(
    raw_files: object,
) -> dict[str, tuple[int, str]]:
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise BundleError("prepared translation fingerprint has no raw file inventory")
    identities: dict[str, tuple[int, str]] = {}
    for raw_name, raw_identity in cast(Mapping[object, object], raw_files).items():
        if not isinstance(raw_name, str):
            raise BundleError("prepared translation fingerprint filename must be a string")
        name = _validated_relative_path(raw_name)
        if len(name.parts) != 1:
            raise BundleError("prepared translation fingerprint filenames must be basenames")
        if not isinstance(raw_identity, Mapping):
            raise BundleError(f"prepared translation fingerprint identity is invalid: {raw_name}")
        identity = cast(Mapping[object, object], raw_identity)
        if set(identity) != {"size", "sha256"}:
            raise BundleError(
                f"prepared translation fingerprint identity fields are invalid: {raw_name}"
            )
        size = identity.get("size")
        digest = identity.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleError(f"prepared translation fingerprint size is invalid: {raw_name}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise BundleError(f"prepared translation fingerprint SHA-256 is invalid: {raw_name}")
        identities[name.as_posix()] = (size, digest)
    return identities


def _validated_foundation_source_identities(
    manifest: Mapping[str, object],
) -> dict[str, tuple[int, str, str]]:
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise BundleError("prepared foundation dataset has no source identities")
    identities: dict[str, tuple[int, str, str]] = {}
    for raw_source in cast(list[object], raw_sources):
        if not isinstance(raw_source, Mapping):
            raise BundleError("prepared foundation dataset contains a non-object source")
        source = cast(Mapping[object, object], raw_source)
        raw_path = source.get("logical_path")
        if not isinstance(raw_path, str):
            raise BundleError("prepared foundation source logical_path must be a string")
        path = _validated_relative_path(raw_path).as_posix()
        size = source.get("size_bytes")
        digest = source.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleError(f"prepared foundation source size is invalid: {path}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise BundleError(f"prepared foundation source SHA-256 is invalid: {path}")
        language = _canonical_language(
            source.get("language"),
            field=f"prepared foundation source language for {path}",
        )
        if path in identities:
            raise BundleError(f"prepared foundation dataset repeats source {path}")
        identities[path] = (size, digest, language)
    return identities


def _validate_config_artifact_semantics(
    config: object,
    *,
    payload_paths: set[str],
    identities: Mapping[str, tuple[int, str]],
    read_payload: Callable[[str], bytes],
    raw_parallel_paths: set[str],
    monolingual_paths: set[str],
) -> None:
    from sion_translate.config import AppConfig
    from sion_translate.data.prepare import prepare_preprocessing_options
    from sion_translate.fingerprint import PREPROCESSING_SCHEMA

    if not isinstance(config, AppConfig):
        raise BundleError("validated bundle config returned an unexpected object")
    expected_pairs = [list(pair) for pair in config.data.configured_language_pairs()]
    expected_directions = [
        list(direction) for direction in config.data.configured_translation_directions()
    ]
    expected_source_only = list(config.data.configured_source_only_languages())
    translation_languages = [
        language for pair in config.data.configured_language_pairs() for language in pair
    ]
    expected_denoise_languages = list(
        dict.fromkeys([*translation_languages, *config.foundation_languages()])
    )
    tokenizer_source_records: dict[tuple[str, str], tuple[int, str, str | None]] = {}
    tokenizer_reasoning_languages: list[str] = []
    translation_source_identities: dict[str, tuple[int, str]] = {}
    foundation_source_identities: dict[str, tuple[int, str, str]] = {}
    foundation_reasoning_languages: list[str] = []

    if TOKENIZER_METADATA_PATH in payload_paths:
        tokenizer_metadata = _parse_json_object(
            read_payload(TOKENIZER_METADATA_PATH),
            TOKENIZER_METADATA_PATH,
        )
        tokenizer_expected = {
            "language_pairs": expected_pairs,
            "translation_directions": expected_directions,
            "denoise_languages": expected_denoise_languages,
        }
        for field, expected in tokenizer_expected.items():
            if tokenizer_metadata.get(field) != expected:
                raise BundleError(
                    f"tokenizer metadata {field} disagrees with the selected training config"
                )
        raw_reasoning_languages = tokenizer_metadata.get("reasoning_languages")
        if not isinstance(raw_reasoning_languages, list) or any(
            not isinstance(language, str)
            for language in cast(list[object], raw_reasoning_languages)
        ):
            raise BundleError("tokenizer metadata reasoning_languages are invalid")
        tokenizer_reasoning_languages = cast(list[str], raw_reasoning_languages)
        raw_tokenizer_contract = tokenizer_metadata.get("training_contract")
        if not isinstance(raw_tokenizer_contract, Mapping):
            raise BundleError("tokenizer metadata has no authenticated training_contract")
        tokenizer_contract = cast(Mapping[object, object], raw_tokenizer_contract)
        tokenizer_contract_expected: dict[str, object] = {
            "language_pairs": expected_pairs,
            "translation_directions": expected_directions,
            "denoise_languages": expected_denoise_languages,
            "reasoning_languages": tokenizer_reasoning_languages,
            "approximate_split": config.data.approximate_split,
            "source_only_languages": expected_source_only,
            "train_only_prefixes": list(config.data.configured_synthetic_prefixes()),
            "split_digits": True,
            "monolingual_sample_ratio": config.foundation.tokenizer_sample_ratio,
        }
        for field, expected in tokenizer_contract_expected.items():
            if tokenizer_contract.get(field) != expected:
                raise BundleError(
                    f"tokenizer training contract {field} disagrees with the selected config"
                )
        tokenizer_source_records = _validated_tokenizer_source_records(
            tokenizer_contract.get("sources")
        )

    if DATASET_MANIFEST_PATH in payload_paths:
        manifest = _parse_json_object(
            read_payload(DATASET_MANIFEST_PATH),
            DATASET_MANIFEST_PATH,
        )
        expected_options = prepare_preprocessing_options(
            approximate_split=config.data.approximate_split,
            source_only_languages=config.data.configured_source_only_languages(),
            translation_directions=config.data.configured_translation_directions(),
            train_only_prefixes=config.data.configured_synthetic_prefixes(),
            managed_augmentation_prefix=config.data.synthetic_prefix,
            synthetic_sampling_weight=config.data.synthetic_sampling_weight,
            language_pair_count=len(config.data.configured_language_pairs()),
        )
        expected_manifest_fields: dict[str, object] = {
            "language_pairs": expected_pairs,
            "translation_directions": expected_directions,
            "source_only_languages": expected_source_only,
            "preprocessing_schema": PREPROCESSING_SCHEMA,
            "preprocessing_options": expected_options,
        }
        for field, expected in expected_manifest_fields.items():
            if manifest.get(field) != expected:
                raise BundleError(
                    f"prepared translation dataset {field} disagrees with the selected training config"
                )
        fingerprint = manifest.get("fingerprint")
        if not isinstance(fingerprint, Mapping):
            raise BundleError("prepared translation dataset has no valid fingerprint")
        fingerprint_values = cast(Mapping[object, object], fingerprint)
        if fingerprint_values.get("language_pairs") != expected_pairs:
            raise BundleError(
                "prepared translation fingerprint language_pairs disagree with config"
            )
        if fingerprint_values.get("preprocessing_schema") != PREPROCESSING_SCHEMA:
            raise BundleError("prepared translation fingerprint preprocessing schema is stale")
        if fingerprint_values.get("preprocessing_options") != expected_options:
            raise BundleError(
                "prepared translation fingerprint preprocessing options disagree with config"
            )
        translation_source_identities = _validated_fingerprint_file_identities(
            fingerprint_values.get("files")
        )
        if raw_parallel_paths:
            expected_raw_files = {
                PurePosixPath(path).name: identities[path] for path in sorted(raw_parallel_paths)
            }
            if translation_source_identities != expected_raw_files:
                raise BundleError(
                    "current raw parallel corpus differs from the prepared dataset fingerprint"
                )

    if FOUNDATION_DATASET_MANIFEST_PATH in payload_paths:
        if not config.foundation.enabled:
            raise BundleError(
                "a prepared foundation dataset cannot ship with foundation.enabled=false"
            )
        foundation_manifest = _parse_json_object(
            read_payload(FOUNDATION_DATASET_MANIFEST_PATH),
            FOUNDATION_DATASET_MANIFEST_PATH,
        )
        expected_foundation_options = {
            "deduplicate": config.foundation.deduplicate,
            "deduplication_backend": (
                "sqlite-blake2b-128-v1" if config.foundation.deduplicate else "disabled"
            ),
            "maximum_characters": config.foundation.maximum_characters,
            "max_tokens": config.data.max_source_length - 2,
            "max_target_tokens": config.data.max_target_length - 1,
            "minimum_characters": config.foundation.minimum_characters,
            "reasoning_sample_share": config.foundation.reasoning_sample_share,
            "shard_size": config.foundation.shard_size,
            "validation_fraction": config.foundation.validation_fraction,
        }
        expected_foundation_fields: dict[str, object] = {
            "release_name": config.foundation.release_name,
            "preprocessing_options": expected_foundation_options,
        }
        for field, expected in expected_foundation_fields.items():
            if foundation_manifest.get(field) != expected:
                raise BundleError(
                    f"prepared foundation dataset {field} disagrees with the selected training config"
                )
        raw_foundation_languages = foundation_manifest.get("languages")
        if not isinstance(raw_foundation_languages, list) or not raw_foundation_languages:
            raise BundleError("prepared foundation dataset has no language list")
        foundation_languages = [
            _canonical_language(
                language,
                field=f"prepared foundation languages[{index}]",
            )
            for index, language in enumerate(cast(list[object], raw_foundation_languages))
        ]
        if foundation_languages != raw_foundation_languages or len(foundation_languages) != len(
            set(foundation_languages)
        ):
            raise BundleError("prepared foundation languages must be canonical and unique")
        configured_foundation_languages = config.foundation_languages()
        prepared_language_set = set(foundation_languages)
        if tuple(foundation_languages) != tuple(
            language
            for language in configured_foundation_languages
            if language in prepared_language_set
        ):
            raise BundleError(
                "prepared foundation languages are not an ordered subset of the config"
            )
        if (
            config.foundation.require_all_languages
            and tuple(foundation_languages) != configured_foundation_languages
        ):
            raise BundleError(
                "foundation.require_all_languages=true, but the prepared dataset "
                "does not cover every configured language"
            )
        sampling = foundation_manifest.get("language_sampling")
        if not isinstance(sampling, Mapping):
            raise BundleError("prepared foundation dataset has no language_sampling contract")
        if sampling.get("alpha") != config.foundation.language_sampling_alpha:
            raise BundleError("foundation language sampling alpha disagrees with config")
        if sampling.get("minimum_share") != config.foundation.minimum_language_share:
            raise BundleError("foundation minimum language share disagrees with config")
        foundation_source_identities = _validated_foundation_source_identities(foundation_manifest)
        raw_reasoning = foundation_manifest.get("reasoning")
        if not isinstance(raw_reasoning, Mapping):
            raise BundleError("prepared foundation dataset has no reasoning contract")
        raw_foundation_reasoning_languages = raw_reasoning.get("languages")
        if not isinstance(raw_foundation_reasoning_languages, list) or any(
            not isinstance(language, str)
            for language in cast(list[object], raw_foundation_reasoning_languages)
        ):
            raise BundleError("prepared foundation reasoning languages are invalid")
        foundation_reasoning_languages = cast(
            list[str],
            raw_foundation_reasoning_languages,
        )

    parallel_tokenizer_sources = {
        path: (size, digest)
        for (role, path), (size, digest, _language) in tokenizer_source_records.items()
        if role == "parallel"
    }
    expected_parallel_sources = translation_source_identities or {
        PurePosixPath(path).name: identities[path] for path in sorted(raw_parallel_paths)
    }
    if tokenizer_source_records and parallel_tokenizer_sources != expected_parallel_sources:
        raise BundleError(
            "tokenizer parallel-source provenance differs from the prepared or included corpus"
        )

    tokenizer_monolingual_sources = {
        path: (size, digest, cast(str, language))
        for (role, path), (size, digest, language) in tokenizer_source_records.items()
        if role == "monolingual"
    }
    if foundation_source_identities:
        if tokenizer_monolingual_sources != foundation_source_identities:
            raise BundleError(
                "tokenizer monolingual-source provenance differs from the foundation dataset"
            )
        if tokenizer_reasoning_languages != foundation_reasoning_languages:
            raise BundleError("tokenizer reasoning languages differ from the foundation dataset")
    if monolingual_paths:
        corpus_root = PurePosixPath(config.foundation.corpus_dir)
        included_monolingual_sources: dict[str, tuple[int, str, str]] = {}
        for path in sorted(monolingual_paths):
            try:
                logical_path = PurePosixPath(path).relative_to(corpus_root)
            except ValueError as error:
                raise BundleError(
                    f"included monolingual source is outside {corpus_root}: {path}"
                ) from error
            language = _canonical_language(
                logical_path.parts[0],
                field=f"included monolingual source language for {logical_path}",
            )
            included_monolingual_sources[logical_path.as_posix()] = (
                identities[path][0],
                identities[path][1],
                language,
            )
        if foundation_source_identities and (
            included_monolingual_sources != foundation_source_identities
        ):
            raise BundleError(
                "current monolingual corpus differs from the prepared foundation dataset"
            )
        if tokenizer_source_records and (
            included_monolingual_sources != tokenizer_monolingual_sources
        ):
            raise BundleError(
                "current monolingual corpus differs from tokenizer training provenance"
            )


def _validate_training_contract(
    contract: object,
    *,
    payload_paths: set[str],
    identities: Mapping[str, tuple[int, str]],
    origins: Mapping[str, str],
    read_payload: Callable[[str], bytes],
    omitted_source_freshness: OmittedSourceFreshness | None = None,
) -> None:
    if not isinstance(contract, Mapping):
        raise BundleError("package manifest training_contract is missing")
    contract_values = cast(Mapping[object, object], contract)
    config_path_value = contract_values.get("config_path")
    if not isinstance(config_path_value, str):
        raise BundleError("training contract config_path must be a string")
    config_path = _validated_relative_path(config_path_value)
    config_key = config_path.as_posix()
    if config_key not in payload_paths or origins.get(config_key) != "git-index":
        raise BundleError("selected training config is not an authenticated Git payload file")
    config_size, config_sha256 = identities[config_key]
    config_content = read_payload(config_key)
    if len(config_content) != config_size:
        raise BundleError("selected training config size changed during validation")
    config = _validated_config_from_payload(config_content, config_key)
    payload_raw_parallel_paths = {
        path for path, origin in origins.items() if origin == "data-jsonl"
    }
    raw_parallel_paths = set(payload_raw_parallel_paths)
    monolingual_paths = {path for path, origin in origins.items() if origin == "monolingual-corpus"}
    semantic_identities = dict(identities)
    if omitted_source_freshness is not None:
        collisions = set(semantic_identities).intersection(omitted_source_freshness.identities)
        if collisions:
            raise BundleError(
                f"omitted-source freshness paths collide with bundle payloads: {sorted(collisions)}"
            )
        semantic_identities.update(omitted_source_freshness.identities)
        raw_parallel_paths.update(omitted_source_freshness.raw_parallel_paths)
        monolingual_paths.update(omitted_source_freshness.monolingual_paths)
    expected_contract = _training_contract_payload(
        config_path,
        config_sha256,
        config,
        raw_parallel_data_included=bool(payload_raw_parallel_paths),
    )
    if dict(contract_values) != expected_contract:
        raise BundleError("package training contract disagrees with its selected config or payload")
    _validate_config_artifact_semantics(
        config,
        payload_paths=payload_paths,
        identities=semantic_identities,
        read_payload=read_payload,
        raw_parallel_paths=raw_parallel_paths,
        monolingual_paths=monolingual_paths,
    )


def _collect_sources(
    root: Path,
    *,
    config_path: Path | str | None = None,
    include_monolingual_corpus: bool = False,
    include_tokenizer: bool = False,
    include_dataset: bool = False,
    include_foundation_dataset: bool = False,
    include_raw_parallel_data: bool = True,
) -> list[SourceEntry]:
    config_selection = _load_config_selection(root, config_path)
    from sion_translate.config import AppConfig

    config = config_selection.config
    if not isinstance(config, AppConfig):
        raise BundleError("validated bundle config returned an unexpected object")
    raw_root_relative = _repository_relative_path(
        root,
        config.data.raw_dir,
        field="data.raw_dir",
    )
    evaluation_root_relative = raw_root_relative / "evaluation_only"
    monolingual_root_relative = _repository_relative_path(
        root,
        config.foundation.corpus_dir,
        field="foundation.corpus_dir",
    )
    selected: dict[PurePosixPath, SourceEntry] = {}
    portable_paths = {
        _portable_path_key(PurePosixPath(MANIFEST_NAME)): MANIFEST_NAME,
        _portable_path_key(PurePosixPath(CHECKSUMS_NAME)): CHECKSUMS_NAME,
    }

    def add(entry: SourceEntry) -> None:
        _assert_regular_source(root, entry.source_path, entry.relative_path)
        previous = selected.get(entry.relative_path)
        if previous is not None:
            if previous.source_path.resolve() != entry.source_path.resolve():
                raise BundleError(f"multiple sources map to {entry.relative_path}")
            if previous.origin == "git-index" and entry.origin != "git-index":
                selected[entry.relative_path] = entry
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

    reserved_roots = [evaluation_root_relative, monolingual_root_relative]
    if not include_raw_parallel_data:
        # Exclude the complete configured raw tree from generic Git payload
        # collection. Evaluation inputs are added back with their own role.
        reserved_roots.append(raw_root_relative)
    for entry in _tracked_stage_zero_entries(
        root,
        reserved_roots=tuple(reserved_roots),
    ):
        add(entry)
    if config_selection.config_path not in selected:
        raise BundleError(
            "the selected training config is not a tracked bundle file: "
            f"{config_selection.config_path}"
        )

    data_root = root.joinpath(*raw_root_relative.parts)
    if include_raw_parallel_data:
        for entry in _collect_configured_parallel_sources(root, raw_root_relative):
            add(entry)

    evaluation_root = data_root / "evaluation_only"
    for entry in _collect_tree(root, evaluation_root, "evaluation-only"):
        add(entry)

    if include_monolingual_corpus:
        selection = _load_monolingual_selection(
            root,
            config_selection.config_path.as_posix(),
        )
        if selection.config_path not in selected:
            raise BundleError(
                "the config used to select monolingual corpora is not a tracked bundle file: "
                f"{selection.config_path}"
            )
        for entry in _collect_configured_monolingual_sources(root, selection):
            add(entry)

    if include_tokenizer:
        tokenizer_root = root / "artifacts" / "tokenizer"
        tokenizer_entries = _collect_tree(root, tokenizer_root, "tokenizer")
        if not tokenizer_entries:
            raise BundleError(
                f"--with-tokenizer was requested but {tokenizer_root} does not exist "
                "or holds no files"
            )
        present = {entry.relative_path.name for entry in tokenizer_entries}
        missing = sorted(REQUIRED_TOKENIZER_FILES - present)
        if missing:
            # Shipping a partial tokenizer moves the failure to the GPU server,
            # after the upload and after the environment is paid for.
            raise BundleError(
                "the tokenizer directory is incomplete; refusing to ship it. "
                f"missing: {', '.join(missing)}"
            )
        for entry in tokenizer_entries:
            add(entry)

    if include_dataset:
        dataset_root = root.joinpath(*PurePosixPath(TRANSLATION_DATASET_ROOT_PATH).parts)
        dataset_entries = _collect_tree(root, dataset_root, "dataset")
        if not dataset_entries:
            raise BundleError(
                f"--with-dataset was requested but {dataset_root} does not exist or holds no files"
            )
        if not include_tokenizer:
            # The dataset is token ids. Without the tokenizer that produced them
            # the server rebuilds it anyway, so shipping gigabytes of it alone is
            # wasted upload.
            raise BundleError(
                "--with-dataset requires --with-tokenizer: a prepared dataset is "
                "only reusable together with the tokenizer whose ids it holds"
            )
        for entry in dataset_entries:
            add(entry)

    if include_foundation_dataset:
        foundation_root = root.joinpath(*PurePosixPath(FOUNDATION_DATASET_ROOT_PATH).parts)
        foundation_entries = _collect_tree(
            root,
            foundation_root,
            "foundation-dataset",
        )
        if not foundation_entries:
            raise BundleError(
                "--with-foundation-dataset was requested but "
                f"{foundation_root} does not exist or holds no files"
            )
        if not include_tokenizer:
            raise BundleError(
                "--with-foundation-dataset requires --with-tokenizer because its token ids "
                "must remain bound to the tokenizer that produced them"
            )
        for entry in foundation_entries:
            add(entry)

    entries = [selected[path] for path in sorted(selected, key=lambda item: item.as_posix())]
    if include_raw_parallel_data and not any(entry.origin == "data-jsonl" for entry in entries):
        raise BundleError(
            f"no immediate {raw_root_relative.as_posix()}/*.jsonl training corpus files "
            "were selected"
        )
    if not any(entry.origin == "evaluation-only" for entry in entries):
        raise BundleError(
            f"{evaluation_root_relative.as_posix()} is missing or contains no regular files"
        )
    return entries


def _training_contract_from_sources(
    sources: list[SourceEntry],
    config_path: PurePosixPath,
) -> tuple[
    dict[str, object],
    dict[str, tuple[int, str]],
    dict[str, str],
    Callable[[str], bytes],
]:
    entries = {entry.relative_path.as_posix(): entry for entry in sources}
    config_key = config_path.as_posix()
    if config_key not in entries:
        raise BundleError(f"selected training config is absent from bundle sources: {config_key}")
    identities = {path: _hash_file(entry.source_path) for path, entry in entries.items()}
    origins = {path: entry.origin for path, entry in entries.items()}

    def read_payload(relative_path: str) -> bytes:
        entry = entries.get(relative_path)
        if entry is None:
            raise BundleError(f"training contract references an absent payload: {relative_path}")
        size = identities[relative_path][0]
        with entry.source_path.open("rb") as source:
            return _read_limited(source, size, relative_path)

    config = _validated_config_from_payload(read_payload(config_key), config_key)
    raw_parallel_data_included = any(origin == "data-jsonl" for origin in origins.values())
    contract = _training_contract_payload(
        config_path,
        identities[config_key][1],
        config,
        raw_parallel_data_included=raw_parallel_data_included,
    )
    return contract, identities, origins, read_payload


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
    info._compresslevel = 6  # pyright: ignore[reportAttributeAccessIssue]
    return info


def _copy_and_hash(source: IO[bytes], destination: IO[bytes]) -> tuple[int, str]:
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


def _source_root(entry: SourceEntry) -> Path:
    root = entry.source_path
    for _part in entry.relative_path.parts:
        root = root.parent
    return root


def _source_metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _hash_omitted_sources(
    root: Path,
    entries: Iterable[SourceEntry],
) -> tuple[dict[str, tuple[int, str]], tuple[tuple[SourceEntry, tuple[int, ...]], ...]]:
    """Hash omitted preparation inputs and prove each hash came from stable bytes."""

    identities: dict[str, tuple[int, str]] = {}
    snapshots: list[tuple[SourceEntry, tuple[int, ...]]] = []
    for entry in entries:
        key = entry.relative_path.as_posix()
        if key in identities:
            raise BundleError(f"duplicate omitted preparation source: {key}")
        before = _assert_regular_source(root, entry.source_path, entry.relative_path)
        identity = _source_metadata_identity(before)
        identities[key] = _hash_file(entry.source_path)
        after = _assert_regular_source(root, entry.source_path, entry.relative_path)
        if _source_metadata_identity(after) != identity:
            raise BundleError(f"omitted preparation source changed while hashing: {key}")
        snapshots.append((entry, identity))
    return identities, tuple(snapshots)


def _assert_omitted_sources_unchanged(freshness: OmittedSourceFreshness | None) -> None:
    """Reject a bundle if an omitted source changed after freshness validation."""

    if freshness is None:
        return
    for entry, expected in freshness.metadata:
        current = _assert_regular_source(
            _source_root(entry),
            entry.source_path,
            entry.relative_path,
        )
        if _source_metadata_identity(current) != expected:
            raise BundleError(
                "omitted preparation source changed after bundle preflight: "
                f"{entry.relative_path.as_posix()}"
            )


def _write_source(
    archive: zipfile.ZipFile,
    entry: SourceEntry,
    *,
    expected_identity: tuple[int, ...] | None = None,
) -> FileRecord:
    root = _source_root(entry)
    before = _assert_regular_source(root, entry.source_path, entry.relative_path)
    if expected_identity is not None and _source_metadata_identity(before) != expected_identity:
        raise BundleError(f"source changed after bundle preflight: {entry.relative_path}")

    info = _zip_info(entry.relative_path.as_posix(), entry.mode)
    with entry.source_path.open("rb") as source:
        opened = os.fstat(source.fileno())
        if _metadata_is_link_like(opened) or not stat.S_ISREG(opened.st_mode):
            raise BundleError(
                f"selected source became link-like while it was opened: {entry.relative_path}"
            )
        if _source_metadata_identity(opened) != _source_metadata_identity(before):
            raise BundleError(f"source changed before it was packaged: {entry.relative_path}")
        with archive.open(
            info,
            mode="w",
            force_zip64=True,
        ) as destination:
            size, digest = _copy_and_hash(source, destination)
        after_handle = os.fstat(source.fileno())

    after = _assert_regular_source(root, entry.source_path, entry.relative_path)
    if (
        size != before.st_size
        or _source_metadata_identity(after_handle) != _source_metadata_identity(before)
        or _source_metadata_identity(after) != _source_metadata_identity(before)
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
    training_contract: Mapping[str, object],
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
        "training_contract": dict(training_contract),
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
    *,
    expected_source_identities: Mapping[str, tuple[int, ...]] | None = None,
    training_contract: Mapping[str, object] | None = None,
    config_path: PurePosixPath | None = None,
) -> None:
    if training_contract is None:
        training_contract, _identities, _origins, _reader = _training_contract_from_sources(
            sources,
            config_path or PurePosixPath(DEFAULT_CONFIG_PATH),
        )
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
        strict_timestamps=True,
    ) as archive:
        records = [
            _write_source(
                archive,
                entry,
                expected_identity=(
                    expected_source_identities.get(entry.relative_path.as_posix())
                    if expected_source_identities is not None
                    else None
                ),
            )
            for entry in sources
        ]
        manifest = _manifest_bytes(commit, tree, records, training_contract)
        manifest_sha256 = _write_bytes(archive, MANIFEST_NAME, manifest)
        _write_bytes(
            archive,
            CHECKSUMS_NAME,
            _checksums_bytes(records, manifest_sha256),
        )


def _deflate_size_bound(size: int) -> int:
    """Return zlib's conservative upper bound for one deflated payload."""

    return size + (size >> 12) + (size >> 14) + (size >> 25) + 13


def _zip_member_size_bound(relative_path: str, size: int) -> int:
    encoded_name_size = len(f"{ARCHIVE_ROOT}/{relative_path}".encode("utf-8"))
    # Account for local and central headers, ZIP64 fields, and a possible data
    # descriptor. Seekable output usually needs less, but the preflight must
    # remain safe if zipfile changes that implementation detail.
    metadata_bound = 30 + 20 + encoded_name_size + 24 + 46 + 28 + encoded_name_size
    return _deflate_size_bound(size) + metadata_bound


def _estimated_archive_size_bound(
    sources: list[SourceEntry],
    commit: str,
    tree: str,
    training_contract: Mapping[str, object],
) -> tuple[int, dict[str, tuple[int, ...]]]:
    records: list[FileRecord] = []
    source_identities: dict[str, tuple[int, ...]] = {}
    total = 0
    for entry in sources:
        metadata = _assert_regular_source(
            _source_root(entry),
            entry.source_path,
            entry.relative_path,
        )
        source_identities[entry.relative_path.as_posix()] = _source_metadata_identity(metadata)
        record = FileRecord(
            path=entry.relative_path.as_posix(),
            size=metadata.st_size,
            sha256="0" * 64,
            origin=entry.origin,
            mode=entry.mode,
        )
        records.append(record)
        total += _zip_member_size_bound(record.path, record.size)

    manifest = _manifest_bytes(commit, tree, records, training_contract)
    checksums = _checksums_bytes(records, "0" * 64)
    total += _zip_member_size_bound(MANIFEST_NAME, len(manifest))
    total += _zip_member_size_bound(CHECKSUMS_NAME, len(checksums))
    # End-of-central-directory, ZIP64 locator, and a small implementation
    # cushion. The per-member bounds above dominate this fixed allowance.
    return total + 4096, source_identities


def _format_byte_count(value: int) -> str:
    gibibytes = value / (1024**3)
    return f"{gibibytes:.2f} GiB ({value:,} bytes)"


def _ensure_free_disk_for_archive(
    output_parent: Path,
    sources: list[SourceEntry],
    commit: str,
    tree: str,
    training_contract: Mapping[str, object],
) -> tuple[int, dict[str, tuple[int, ...]]]:
    archive_bound, source_identities = _estimated_archive_size_bound(
        sources,
        commit,
        tree,
        training_contract,
    )
    reserve = max(MIN_FREE_DISK_RESERVE, archive_bound // FREE_DISK_RESERVE_DIVISOR)
    required = archive_bound + reserve
    try:
        free = shutil.disk_usage(output_parent).free
    except OSError as error:
        raise BundleError(
            f"could not inspect free disk space at {output_parent}: {error}"
        ) from error
    if free < required:
        raise BundleError(
            "insufficient free disk space for a safely staged GPU bundle: "
            f"need at least {_format_byte_count(required)}, "
            f"but only {_format_byte_count(free)} is free"
        )
    return archive_bound, source_identities


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


def _publish_archive(temporary_path: Path, output: Path, *, overwrite: bool) -> None:
    """Publish atomically while making the non-overwrite promise race-free."""

    if overwrite:
        os.replace(temporary_path, output)
        return
    try:
        # Both files are in the same directory, so the hard-link publication is
        # atomic and cannot replace a destination created by another process.
        os.link(temporary_path, output)
    except FileExistsError as error:
        raise BundleError(
            f"bundle output appeared while building; refusing to replace it: {output}"
        ) from error
    except OSError as error:
        raise BundleError(
            f"could not atomically publish bundle without overwrite: {error}"
        ) from error
    temporary_path.unlink()


def _hash_stream(source: IO[bytes]) -> tuple[int, str]:
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


def _read_limited(source: IO[bytes], size: int, name: str) -> bytes:
    if size > MAX_METADATA_SIZE:
        raise BundleError(f"{name} is unreasonably large ({size} bytes)")
    content = source.read(MAX_METADATA_SIZE + 1)
    if len(content) > MAX_METADATA_SIZE:
        raise BundleError(f"{name} exceeds the metadata size limit")
    return content


def _parse_manifest(content: bytes) -> tuple[dict[str, object], list[FileRecord]]:
    try:
        decoded = cast(object, json.loads(content.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"{MANIFEST_NAME} is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise BundleError(f"{MANIFEST_NAME} must contain a JSON object")
    raw = cast(dict[str, object], decoded)
    expected_top_level = {
        "archive_root",
        "files",
        "format_version",
        "git",
        "payload",
        "training_contract",
        "zip_metadata",
    }
    if set(raw) != expected_top_level:
        raise BundleError(f"package manifest fields do not match format version {FORMAT_VERSION}")
    format_version = raw.get("format_version")
    if isinstance(format_version, bool) or format_version != FORMAT_VERSION:
        raise BundleError("unsupported package manifest format version")
    if raw.get("archive_root") != ARCHIVE_ROOT:
        raise BundleError(f"manifest archive_root must be {ARCHIVE_ROOT!r}")

    git_identity = raw.get("git")
    if not isinstance(git_identity, dict):
        raise BundleError("manifest git identity is missing")
    git_values = cast(dict[object, object], git_identity)
    if set(git_values) != {"commit", "tree"}:
        raise BundleError("manifest Git identity fields are invalid")
    commit = git_values.get("commit")
    tree = git_values.get("tree")
    if not isinstance(commit, str) or not GIT_OBJECT_PATTERN.fullmatch(commit):
        raise BundleError("manifest Git commit is invalid")
    if not isinstance(tree, str) or not GIT_OBJECT_PATTERN.fullmatch(tree):
        raise BundleError("manifest Git tree is invalid")

    raw_files = raw.get("files")
    if not isinstance(raw_files, list):
        raise BundleError("manifest files must be a list")
    records: list[FileRecord] = []
    for raw_record in cast(list[object], raw_files):
        if not isinstance(raw_record, dict):
            raise BundleError("manifest contains a non-object file record")
        record_values = cast(dict[object, object], raw_record)
        if set(record_values) != {"mode", "origin", "path", "sha256", "size"}:
            raise BundleError("manifest file record fields are invalid")
        path = record_values.get("path")
        size = record_values.get("size")
        digest = record_values.get("sha256")
        origin = record_values.get("origin")
        mode = record_values.get("mode")
        if not isinstance(path, str):
            raise BundleError("manifest file path is invalid")
        _validated_relative_path(path)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BundleError(f"manifest size is invalid for {path}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise BundleError(f"manifest sha256 is invalid for {path}")
        if not isinstance(origin, str) or origin not in ALLOWED_ORIGINS:
            raise BundleError(f"manifest origin is invalid for {path}")
        if not isinstance(mode, str) or mode not in REGULAR_GIT_MODES:
            raise BundleError(f"manifest mode is invalid for {path}")
        expected_artifact_origin = next(
            (
                candidate_origin
                for candidate_origin, root_path in ARTIFACT_ORIGIN_ROOTS.items()
                if path.startswith(f"{root_path}/")
            ),
            None,
        )
        if expected_artifact_origin is not None and origin != expected_artifact_origin:
            raise BundleError(f"manifest origin does not match the artifact root for {path}")
        artifact_root = ARTIFACT_ORIGIN_ROOTS.get(origin)
        if artifact_root is not None and not path.startswith(f"{artifact_root}/"):
            raise BundleError(f"manifest {origin} file is outside its artifact root: {path}")
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
    portable_paths = [_portable_path_key(PurePosixPath(path)) for path in paths]
    if len(portable_paths) != len(set(portable_paths)):
        raise BundleError("manifest file paths collide on portable filesystems")
    if MANIFEST_NAME in paths or CHECKSUMS_NAME in paths:
        raise BundleError("generated metadata may not appear as a payload file")

    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise BundleError("manifest payload summary is missing")
    payload_values = cast(dict[object, object], payload)
    if set(payload_values) != {"file_count", "total_bytes"}:
        raise BundleError("manifest payload summary fields are invalid")
    if payload_values.get("file_count") != len(records):
        raise BundleError("manifest payload file_count does not match its files")
    if payload_values.get("total_bytes") != sum(record.size for record in records):
        raise BundleError("manifest payload total_bytes does not match its files")
    if raw.get("zip_metadata") != {
        "compression": "deflate",
        "timestamp": "1980-01-01T00:00:00Z",
        "zip64": True,
    }:
        raise BundleError("manifest ZIP metadata does not match the deterministic contract")
    return raw, records


def _parse_checksums(content: bytes) -> dict[str, str]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BundleError(f"{CHECKSUMS_NAME} is not valid UTF-8") from error
    checksums: dict[str, str] = {}
    portable_paths: dict[str, str] = {}
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
        portable_key = _portable_path_key(PurePosixPath(path))
        collision = portable_paths.get(portable_key)
        if collision is not None:
            raise BundleError(
                f"{CHECKSUMS_NAME}: portable path collision between {collision!r} and {path!r}"
            )
        portable_paths[portable_key] = path
        checksums[path] = digest
    return checksums


def _validate_checksums(
    records: list[FileRecord],
    checksums: dict[str, str],
    manifest_sha256: str,
) -> None:
    expected_order = [record.path for record in records] + [MANIFEST_NAME]
    expected_paths = set(expected_order)
    if set(checksums) != expected_paths:
        missing = sorted(expected_paths - set(checksums))
        extra = sorted(set(checksums) - expected_paths)
        raise BundleError(f"{CHECKSUMS_NAME} path set mismatch; missing={missing}, extra={extra}")
    if list(checksums) != expected_order:
        raise BundleError(f"{CHECKSUMS_NAME} paths are not in deterministic manifest order")
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
            if archive.comment:
                raise BundleError("ZIP archive comments are not part of the deterministic contract")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise BundleError("ZIP contains duplicate member names")
            by_relative_path: dict[str, zipfile.ZipInfo] = {}
            portable_members: dict[str, str] = {}
            for info in infos:
                if info.is_dir():
                    raise BundleError(
                        f"ZIP contains an unexpected directory entry: {info.filename}"
                    )
                if info.flag_bits & 0x1:
                    raise BundleError(f"encrypted ZIP members are not supported: {info.filename}")
                if info.comment:
                    raise BundleError(f"ZIP member comments are not supported: {info.filename}")
                if info.compress_type != zipfile.ZIP_DEFLATED:
                    raise BundleError(
                        f"ZIP member uses an unsupported compression method: {info.filename}"
                    )
                if info.date_time != ZIP_TIMESTAMP:
                    raise BundleError(
                        f"ZIP member has a non-reproducible timestamp: {info.filename}"
                    )
                if info.create_system != 3:
                    raise BundleError(
                        f"ZIP member has an unsupported creator system: {info.filename}"
                    )
                if info.create_version != 45 or info.extract_version != 45:
                    raise BundleError(
                        f"ZIP member does not use deterministic ZIP64 headers: {info.filename}"
                    )
                _root, relative = _validate_zip_member_name(info.filename)
                portable_key = _portable_path_key(relative)
                collision = portable_members.get(portable_key)
                if collision is not None:
                    raise BundleError(
                        "ZIP contains a portable member-name collision between "
                        f"{collision!r} and {info.filename!r}"
                    )
                portable_members[portable_key] = info.filename
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
            expected_member_order = [
                *(f"{ARCHIVE_ROOT}/{record.path}" for record in records),
                f"{ARCHIVE_ROOT}/{MANIFEST_NAME}",
                f"{ARCHIVE_ROOT}/{CHECKSUMS_NAME}",
            ]
            if names != expected_member_order:
                raise BundleError("ZIP members are not in deterministic manifest order")
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

            records_by_path = {record.path: record for record in records}

            def read_payload(relative_path: str) -> bytes:
                info = by_relative_path[relative_path]
                with archive.open(info, mode="r") as source:
                    return _read_limited(
                        source,
                        info.file_size,
                        relative_path,
                    )

            _validate_artifact_contracts(
                set(records_by_path),
                {record.path: (record.size, record.sha256) for record in records},
                read_payload,
            )
            _validate_training_contract(
                raw_manifest.get("training_contract"),
                payload_paths=set(records_by_path),
                identities={record.path: (record.size, record.sha256) for record in records},
                origins={record.path: record.origin for record in records},
                read_payload=read_payload,
            )
            if _zip_mode(manifest_info) != "100644":
                raise BundleError(f"ZIP mode mismatch for {MANIFEST_NAME}")
            if _zip_mode(checksums_info) != "100644":
                raise BundleError(f"ZIP mode mismatch for {CHECKSUMS_NAME}")
    except zipfile.BadZipFile as error:
        raise BundleError(f"invalid ZIP archive: {error}") from error

    git_identity = raw_manifest["git"]
    assert isinstance(git_identity, dict)
    git_values = cast(dict[str, object], git_identity)
    return VerificationResult(
        file_count=len(records),
        total_bytes=sum(record.size for record in records),
        git_commit=cast(str, git_values["commit"]),
        git_tree=cast(str, git_values["tree"]),
    )


def _read_tree_metadata(path: Path, name: str) -> bytes:
    metadata = path.lstat()
    if _metadata_is_link_like(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise BundleError(f"{name} is not a regular file")
    with path.open("rb") as source:
        return _read_limited(source, metadata.st_size, name)


def _resolve_tree_root(path: Path) -> Path:
    try:
        root_metadata = path.lstat()
    except FileNotFoundError as error:
        raise BundleError(f"package tree root does not exist: {path}") from error
    if _metadata_is_link_like(root_metadata):
        raise BundleError("package tree root may not be a symlink, junction, or reparse point")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise BundleError(f"package tree root is not a directory: {path}")
    candidate = path.resolve()
    if (candidate / MANIFEST_NAME).is_file():
        return candidate
    nested = candidate / ARCHIVE_ROOT
    try:
        nested_metadata = nested.lstat()
    except FileNotFoundError:
        nested_metadata = None
    if nested_metadata is not None and _metadata_is_link_like(nested_metadata):
        raise BundleError("nested package tree root may not be link-like")
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
    for candidate in _walk_regular_tree(root, "package tree"):
        relative = candidate.relative_to(root).as_posix()
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

    records_by_path = {record.path: record for record in records}

    def read_payload(relative_path: str) -> bytes:
        source_path = root.joinpath(*PurePosixPath(relative_path).parts)
        return _read_tree_metadata(source_path, relative_path)

    _validate_artifact_contracts(
        set(records_by_path),
        {record.path: (record.size, record.sha256) for record in records},
        read_payload,
    )
    _validate_training_contract(
        raw_manifest.get("training_contract"),
        payload_paths=set(records_by_path),
        identities={record.path: (record.size, record.sha256) for record in records},
        origins={record.path: record.origin for record in records},
        read_payload=read_payload,
    )

    git_identity = raw_manifest["git"]
    assert isinstance(git_identity, dict)
    git_values = cast(dict[str, object], git_identity)
    return VerificationResult(
        file_count=len(records),
        total_bytes=sum(record.size for record in records),
        git_commit=cast(str, git_values["commit"]),
        git_tree=cast(str, git_values["tree"]),
    )


def _resolved_output_path(raw_path: Path) -> Path:
    """Resolve the parent while refusing an existing link-like output leaf."""

    absolute = Path(os.path.abspath(raw_path))
    try:
        metadata = absolute.lstat()
    except FileNotFoundError:
        metadata = None
    if metadata is not None:
        if _metadata_is_link_like(metadata):
            raise BundleError(
                f"bundle output may not be a symlink, junction, or reparse point: {absolute}"
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise BundleError(f"bundle output exists but is not a regular file: {absolute}")
    return absolute.parent.resolve(strict=False) / absolute.name


def build_bundle(
    repository_root: Path | str = REPOSITORY_ROOT,
    output_path: Path | str | None = None,
    *,
    overwrite: bool = False,
    config_path: Path | str | None = None,
    include_monolingual_corpus: bool = False,
    include_tokenizer: bool = False,
    include_dataset: bool = False,
    include_foundation_dataset: bool = False,
    prepared_only: bool = False,
) -> BuildResult:
    """Build, verify, and atomically publish a deterministic GPU bundle."""

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise BundleError(f"repository root does not exist: {root}")
    output = _resolved_output_path(
        Path(output_path) if output_path is not None else root / f"{ARCHIVE_ROOT}.zip"
    )
    if output.suffix.lower() != ".zip":
        raise BundleError("bundle output must use the .zip extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise BundleError(f"bundle output already exists; pass --overwrite to replace it: {output}")

    _ensure_clean_tracked_tree(root)
    commit, tree = _git_identity(root)
    config_selection = _load_config_selection(root, config_path)
    if prepared_only:
        if include_monolingual_corpus:
            raise BundleError(
                "--prepared-only cannot include the monolingual corpus; the mode exists "
                "to avoid uploading raw preparation inputs"
            )
        from sion_translate.config import AppConfig

        config = config_selection.config
        if not isinstance(config, AppConfig):
            raise BundleError("validated bundle config returned an unexpected object")
        include_tokenizer = True
        include_dataset = True
        include_foundation_dataset = config.foundation.enabled
    sources = _collect_sources(
        root,
        config_path=config_path,
        include_monolingual_corpus=include_monolingual_corpus,
        include_tokenizer=include_tokenizer,
        include_dataset=include_dataset,
        include_foundation_dataset=include_foundation_dataset,
        include_raw_parallel_data=not prepared_only,
    )
    if not sources:
        raise BundleError("the bundle source allowlist selected no files")
    omitted_source_freshness = _collect_omitted_source_freshness(
        root,
        config_selection,
        parallel=prepared_only and include_dataset,
        monolingual=include_foundation_dataset and not include_monolingual_corpus,
    )
    training_contract, identities, origins, read_payload = _training_contract_from_sources(
        sources,
        config_selection.config_path,
    )
    _validate_artifact_contracts(set(identities), identities, read_payload)
    _validate_training_contract(
        training_contract,
        payload_paths=set(identities),
        identities=identities,
        origins=origins,
        read_payload=read_payload,
        omitted_source_freshness=omitted_source_freshness,
    )
    if any(source.source_path.resolve() == output for source in sources):
        raise BundleError("bundle output may not overwrite a selected source file")
    archive_size_bound, source_identities = _ensure_free_disk_for_archive(
        output.parent,
        sources,
        commit,
        tree,
        training_contract,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_archive(
            temporary_path,
            sources,
            commit,
            tree,
            expected_source_identities=source_identities,
            training_contract=training_contract,
            config_path=config_selection.config_path,
        )
        _fsync_file(temporary_path)
        verification = verify_archive(temporary_path)
        temporary_size, temporary_sha256 = _hash_file(temporary_path)
        if temporary_size > archive_size_bound:
            raise BundleError(
                "temporary ZIP exceeded the conservative disk-space estimate; "
                "refusing to publish it"
            )
        _assert_omitted_sources_unchanged(omitted_source_freshness)
        _ensure_clean_tracked_tree(root)
        if _git_identity(root) != (commit, tree):
            raise BundleError("Git HEAD changed while the bundle was being built")
        _publish_archive(temporary_path, output, overwrite=overwrite)
        _fsync_directory(output.parent)
        archive_size, archive_sha256 = _hash_file(output)
        if (archive_size, archive_sha256) != (temporary_size, temporary_sha256):
            raise BundleError("published bundle bytes differ from the verified temporary ZIP")
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

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
    build_parser.add_argument(
        "--config",
        type=Path,
        help=(
            "tracked YAML configuration authenticated by the bundle. Format version 2 "
            "requires ROOT/sion_translate.yaml so plain sion-train uses the same file."
        ),
    )
    build_parser.add_argument(
        "--prepared-only",
        action="store_true",
        help=(
            "ship the tokenizer and all configured prepared datasets but omit raw "
            "parallel and monolingual corpora, including tracked corpus files"
        ),
    )
    build_parser.add_argument(
        "--with-monolingual-corpus",
        action="store_true",
        help=(
            "also ship data/corpus (foundation preparation input). Without it, the "
            "foundation stage requires authenticated prepared foundation shards or is skipped."
        ),
    )
    build_parser.add_argument(
        "--with-tokenizer",
        action="store_true",
        help=(
            "also ship artifacts/tokenizer so the server reuses it instead of "
            "spending hours training one. The directory must be complete."
        ),
    )
    build_parser.add_argument(
        "--with-dataset",
        action="store_true",
        help=(
            "also ship artifacts/dataset, the tokenized training shards. "
            "Requires --with-tokenizer, because the ids only mean anything "
            "alongside the tokenizer that produced them."
        ),
    )
    build_parser.add_argument(
        "--with-foundation-dataset",
        action="store_true",
        help=(
            "also ship artifacts/foundation_dataset, the prepared foundation shards. "
            "Requires --with-tokenizer and authenticates the manifest inventory."
        ),
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
                config_path=parsed.config,
                include_monolingual_corpus=parsed.with_monolingual_corpus,
                include_tokenizer=parsed.with_tokenizer,
                include_dataset=parsed.with_dataset,
                include_foundation_dataset=parsed.with_foundation_dataset,
                prepared_only=parsed.prepared_only,
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
