"""Validate the immutable inputs carried by an extracted GPU bundle.

The archive builder performs the complete, format-specific validation before an
archive is published.  This module provides the smaller runtime boundary needed
by both ``easy_run.py`` and the training CLI: authenticate the embedded manifest,
bind the selected configuration to it, verify packaged payload bytes on demand,
and refuse any effective training source inventory that differs from the bundle.

The source comparison is intentionally based on manifest paths and hashes rather
than language names.  It therefore supports any configured language graph and
cannot accidentally turn a Korean/English/Japanese example into a runtime rule.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, cast

from sion_translate.fingerprint import DatasetFingerprint, file_sha256


MANIFEST_NAME = "PACKAGE_MANIFEST.json"
CHECKSUMS_NAME = "SHA256SUMS"
ARCHIVE_ROOT = "sion_translate"
FORMAT_VERSION = 2
TRAINING_CONTRACT_SCHEMA = "sion-gpu-training-contract-v2"
DEPENDENCY_CONTRACT_SCHEMA = "sion-gpu-dependency-environment-v1"
DEFAULT_CONFIG_PATH = "sion_translate.yaml"
GIT_CHECKOUT_SENTINEL = "src/sion_translate/bundle_contract.py"

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40,64}")
REGULAR_MODES = frozenset({"100644", "100755"})
ALLOWED_ORIGINS = frozenset(
    {
        "git-index",
        "data-jsonl",
        "evaluation-only",
        "monolingual-corpus",
        "tokenizer",
        "dataset",
        "foundation-dataset",
    }
)
ARTIFACT_ORIGIN_ROOTS = {
    "tokenizer": "artifacts/tokenizer",
    "dataset": "artifacts/dataset",
    "foundation-dataset": "artifacts/foundation_dataset",
}
CANONICAL_PATHS = {
    "raw_dir": "data",
    "tokenizer_model": "artifacts/tokenizer/sion.model",
    "tokenizer_features": "artifacts/tokenizer/token_features.npz",
    "translation_dataset": "artifacts/dataset",
    "foundation_dataset": "artifacts/foundation_dataset",
}
EXPECTED_DEPENDENCY_TARGET = {
    "machine": "x86_64",
    "manylinux": "2_28",
    "os": "linux",
    "python_implementation": "cpython",
    "python_version": "3.11",
    "torch_backend": "cu128",
}
EXPECTED_RUNTIME_VERSIONS = {
    "numpy": "2.4.6",
    "sentencepiece": "0.2.1",
    "torch": "2.10.0+cu128",
    "torchao": "0.17.0+cu128",
    "transformers": "5.16.1",
}
TOKENIZER_TRAINING_SCHEMA = "sion-tokenizer-training-v4"
TOKENIZER_INPUT_TRAVERSAL_POLICY = "portable-input-order-v1"
MAX_METADATA_BYTES = 64 * 1024 * 1024
WINDOWS_REPARSE_POINT = 0x400
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

# Installation and training create these paths after the extracted tree has
# passed strict ``verify-tree``.  Runtime verification still hashes every
# packaged payload, but it ignores these known mutable namespaces when looking
# for extra files.
RUNTIME_MUTABLE_TOP_LEVEL = frozenset(
    {
        ".git",
        ".venv",
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
)
RUNTIME_CACHE_PARTS = frozenset({".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"})
RUNTIME_LOCK_FILENAMES = frozenset({".sion_artifacts.lock", ".sion_training_run.lock"})
RUNTIME_EDITABLE_METADATA_ROOTS = (PurePosixPath("src/sion_translate.egg-info"),)


class BundleContractError(RuntimeError):
    """Raised when an extracted bundle no longer satisfies its runtime contract."""


@dataclass(frozen=True)
class BundleFileRecord:
    """One regular payload file authenticated by the package manifest."""

    path: str
    size: int
    sha256: str
    origin: str
    mode: str


@dataclass(frozen=True)
class EmbeddedTrainingContract:
    """The validated immutable training policy embedded in one package tree."""

    root: Path
    manifest_sha256: str
    records: tuple[BundleFileRecord, ...]
    config_path: str
    config_sha256: str
    config_content: bytes
    raw_parallel_data_included: bool
    foundation_enabled: bool
    dependency_environment: Mapping[str, Any]

    def records_for_origin(self, origin: str) -> tuple[BundleFileRecord, ...]:
        return tuple(record for record in self.records if record.origin == origin)

    @property
    def records_by_path(self) -> dict[str, BundleFileRecord]:
        return {record.path: record for record in self.records}


def _portable_path_key(path: PurePosixPath) -> str:
    return unicodedata.normalize("NFC", path.as_posix()).casefold()


def _metadata_is_link_like(metadata: os.stat_result) -> bool:
    """Recognize POSIX links and Windows junction/reparse-point aliases."""

    return bool(
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_reparse_tag", 0)
        or getattr(metadata, "st_file_attributes", 0) & WINDOWS_REPARSE_POINT
    )


def _entry_exists(path: Path, *, label: str) -> bool:
    """Check directory-entry presence without following a dangling link."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise BundleContractError(f"cannot inspect {label}: {path}") from error
    return True


def _looks_like_project_source_tree(root: Path) -> bool:
    """Recognize a source-layout project after its bundle metadata was stripped."""

    return _entry_exists(
        root / "src" / "sion_translate",
        label="project source package",
    )


def _is_valid_project_git_checkout(root: Path) -> bool:
    """Accept only a real rooted Git checkout with the runtime verifier tracked."""

    git_command = shutil.which("git")
    if git_command is None:
        return False
    git_executable = Path(git_command).resolve()
    try:
        git_executable.relative_to(root)
    except ValueError:
        pass
    else:
        return False
    common_arguments = [
        str(git_executable),
        "--no-replace-objects",
        "-C",
        str(root),
    ]
    environment = {
        name: value for name, value in os.environ.items() if not name.upper().startswith("GIT_")
    }
    environment["GIT_NO_LAZY_FETCH"] = "1"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        identity = subprocess.run(
            [*common_arguments, "rev-parse", "--show-toplevel", "HEAD^{commit}"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=5,
            env=environment,
        )
        if identity.returncode != 0:
            return False
        lines = identity.stdout.splitlines()
        if (
            len(lines) != 2
            or Path(lines[0]).resolve() != root
            or GIT_OBJECT_PATTERN.fullmatch(lines[1]) is None
        ):
            return False
        committed = subprocess.run(
            [
                *common_arguments,
                "ls-tree",
                "-z",
                "HEAD",
                "--",
                GIT_CHECKOUT_SENTINEL,
            ],
            check=False,
            capture_output=True,
            timeout=5,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return False
    if committed.returncode != 0 or not committed.stdout.endswith(b"\0"):
        return False
    try:
        metadata, path = committed.stdout[:-1].split(b"\t", 1)
        mode, object_type, object_id = metadata.split(b" ", 2)
        decoded_object_id = object_id.decode("ascii")
    except (ValueError, UnicodeError):
        return False
    return bool(
        mode in {b"100644", b"100755"}
        and object_type == b"blob"
        and GIT_OBJECT_PATTERN.fullmatch(decoded_object_id)
        and path == GIT_CHECKOUT_SENTINEL.encode("ascii")
    )


def _validated_relative_path(value: str) -> PurePosixPath:
    if not value or "\\" in value or "\r" in value or "\n" in value:
        raise BundleContractError(f"bundle path is not canonical POSIX text: {value!r}")
    if unicodedata.normalize("NFC", value) != value:
        raise BundleContractError(f"bundle path is not NFC-normalized: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise BundleContractError(f"bundle path is not canonical and relative: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise BundleContractError(f"bundle path contains an unsafe component: {value!r}")
    for part in path.parts:
        if part.endswith((" ", ".")):
            raise BundleContractError(f"bundle path has a Windows-ambiguous segment: {value!r}")
        if any(ord(character) < 32 for character in part) or any(
            character in WINDOWS_FORBIDDEN_CHARACTERS for character in part
        ):
            raise BundleContractError(f"bundle path has a Windows-unsafe segment: {value!r}")
        device_name = part.split(".", 1)[0].casefold()
        if device_name in WINDOWS_RESERVED_NAMES:
            raise BundleContractError(f"bundle path uses a reserved Windows name: {value!r}")
    return path


def _read_regular_file(path: Path, label: str, *, limit: int | None = None) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise BundleContractError(f"the extracted GPU bundle is missing {label}: {path}") from error
    if _metadata_is_link_like(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise BundleContractError(f"the extracted GPU bundle {label} is not a regular file: {path}")
    if limit is not None and metadata.st_size > limit:
        raise BundleContractError(f"the extracted GPU bundle {label} is unreasonably large")
    try:
        return path.read_bytes()
    except OSError as error:
        raise BundleContractError(
            f"cannot read the extracted GPU bundle {label}: {path}"
        ) from error


def _parse_file_records(raw_files: object) -> tuple[BundleFileRecord, ...]:
    if not isinstance(raw_files, list):
        raise BundleContractError("the GPU bundle manifest files field must be a list")
    records: list[BundleFileRecord] = []
    for raw_record in cast(list[object], raw_files):
        if not isinstance(raw_record, dict):
            raise BundleContractError("the GPU bundle manifest contains a non-object file record")
        values = cast(dict[object, object], raw_record)
        if set(values) != {"mode", "origin", "path", "sha256", "size"}:
            raise BundleContractError("the GPU bundle manifest file-record fields are invalid")
        path = values.get("path")
        size = values.get("size")
        digest = values.get("sha256")
        origin = values.get("origin")
        mode = values.get("mode")
        if not isinstance(path, str):
            raise BundleContractError("the GPU bundle manifest contains a non-string path")
        _validated_relative_path(path)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleContractError(f"the GPU bundle manifest size is invalid for {path}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise BundleContractError(f"the GPU bundle manifest SHA-256 is invalid for {path}")
        if not isinstance(origin, str) or origin not in ALLOWED_ORIGINS:
            raise BundleContractError(f"the GPU bundle manifest origin is invalid for {path}")
        if not isinstance(mode, str) or mode not in REGULAR_MODES:
            raise BundleContractError(f"the GPU bundle manifest mode is invalid for {path}")
        artifact_root = ARTIFACT_ORIGIN_ROOTS.get(origin)
        if artifact_root is not None and not path.startswith(f"{artifact_root}/"):
            raise BundleContractError(
                f"the GPU bundle {origin} record is outside {artifact_root}: {path}"
            )
        expected_artifact_origin = next(
            (
                candidate_origin
                for candidate_origin, root in ARTIFACT_ORIGIN_ROOTS.items()
                if path.startswith(f"{root}/")
            ),
            None,
        )
        if expected_artifact_origin is not None and origin != expected_artifact_origin:
            raise BundleContractError(
                f"the GPU bundle origin does not match the artifact path: {path}"
            )
        records.append(BundleFileRecord(path, size, digest, origin, mode))

    paths = [record.path for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise BundleContractError("the GPU bundle manifest paths must be unique and sorted")
    portable = [_portable_path_key(PurePosixPath(path)) for path in paths]
    if len(portable) != len(set(portable)):
        raise BundleContractError("the GPU bundle manifest has a portable path collision")
    if MANIFEST_NAME in paths or CHECKSUMS_NAME in paths:
        raise BundleContractError("package metadata may not be listed as a payload file")
    return tuple(records)


def _parse_checksums(content: bytes) -> dict[str, str]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BundleContractError(f"{CHECKSUMS_NAME} is not valid UTF-8") from error
    checksums: dict[str, str] = {}
    portable_paths: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if len(line) < 67 or line[64:66] != "  ":
            raise BundleContractError(f"{CHECKSUMS_NAME}:{line_number} is malformed")
        digest = line[:64]
        raw_path = line[66:]
        if not SHA256_PATTERN.fullmatch(digest):
            raise BundleContractError(f"{CHECKSUMS_NAME}:{line_number} has an invalid digest")
        path = _validated_relative_path(raw_path).as_posix()
        portable = _portable_path_key(PurePosixPath(path))
        if path in checksums or portable in portable_paths:
            raise BundleContractError(f"{CHECKSUMS_NAME} repeats or collides at {path!r}")
        checksums[path] = digest
        portable_paths.add(portable)
    return checksums


def _validate_dependency_reference(
    raw_reference: object,
    *,
    label: str,
    records_by_path: Mapping[str, BundleFileRecord],
) -> None:
    if not isinstance(raw_reference, dict):
        raise BundleContractError(f"the GPU dependency {label} reference must be an object")
    reference = cast(dict[object, object], raw_reference)
    if set(reference) != {"path", "sha256", "size"}:
        raise BundleContractError(f"the GPU dependency {label} reference fields are invalid")
    path = reference.get("path")
    size = reference.get("size")
    digest = reference.get("sha256")
    if not isinstance(path, str):
        raise BundleContractError(f"the GPU dependency {label} path is invalid")
    record = records_by_path.get(_validated_relative_path(path).as_posix())
    if (
        record is None
        or record.origin != "git-index"
        or record.size != size
        or record.sha256 != digest
    ):
        raise BundleContractError(
            f"the GPU dependency {label} reference is not bound to its Git payload"
        )


def _validate_dependency_environment(
    raw_environment: object,
    records_by_path: Mapping[str, BundleFileRecord],
) -> Mapping[str, Any]:
    if not isinstance(raw_environment, dict):
        raise BundleContractError("the GPU bundle has no authenticated dependency environment")
    environment = cast(dict[str, Any], raw_environment)
    expected_fields = {
        "schema",
        "generator",
        "target",
        "inputs",
        "normalization",
        "provenance",
        "lock",
        "venv_command",
        "compile_command",
        "sync_command",
        "project_install_command",
        "resolved_runtime_versions",
    }
    if set(environment) != expected_fields:
        raise BundleContractError("the GPU dependency-environment fields are invalid")
    if environment.get("schema") != DEPENDENCY_CONTRACT_SCHEMA:
        raise BundleContractError("the GPU dependency-environment schema is unsupported")
    if environment.get("target") != EXPECTED_DEPENDENCY_TARGET:
        raise BundleContractError(
            "the GPU dependency target is not CPython 3.11/Linux x86_64/CUDA 12.8"
        )
    if environment.get("resolved_runtime_versions") != EXPECTED_RUNTIME_VERSIONS:
        raise BundleContractError("the GPU dependency runtime versions are not the reviewed set")
    raw_generator = environment.get("generator")
    if not isinstance(raw_generator, dict):
        raise BundleContractError("the GPU dependency generator identity is invalid")
    generator = cast(dict[str, object], raw_generator)
    if set(generator) != {"name", "version"}:
        raise BundleContractError("the GPU dependency generator identity is invalid")
    if generator.get("name") != "uv" or not isinstance(generator.get("version"), str):
        raise BundleContractError("the GPU dependency environment was not generated by uv")
    raw_inputs = environment.get("inputs")
    if not isinstance(raw_inputs, dict) or not raw_inputs:
        raise BundleContractError("the GPU dependency input identities are missing")
    inputs = cast(dict[object, object], raw_inputs)
    for path, raw_identity in inputs.items():
        if not isinstance(path, str) or not isinstance(raw_identity, dict):
            raise BundleContractError("the GPU dependency input identity is invalid")
        identity = cast(dict[object, object], raw_identity)
        if set(identity) != {"sha256", "size"}:
            raise BundleContractError("the GPU dependency input identity fields are invalid")
        _validate_dependency_reference(
            {"path": path, "sha256": identity.get("sha256"), "size": identity.get("size")},
            label=f"input {path}",
            records_by_path=records_by_path,
        )
    _validate_dependency_reference(
        environment.get("normalization"),
        label="normalization",
        records_by_path=records_by_path,
    )
    _validate_dependency_reference(
        environment.get("provenance"),
        label="provenance",
        records_by_path=records_by_path,
    )
    raw_lock = environment.get("lock")
    if not isinstance(raw_lock, dict):
        raise BundleContractError("the GPU dependency lock reference is invalid")
    lock = cast(dict[str, Any], raw_lock)
    if set(lock) != {
        "format",
        "lock_version",
        "package_count",
        "path",
        "sha256",
        "size",
        "wheel_count",
    }:
        raise BundleContractError("the GPU dependency lock fields are invalid")
    if (
        lock.get("format") != "pep751"
        or lock.get("lock_version") != "1.0"
        or isinstance(lock.get("package_count"), bool)
        or not isinstance(lock.get("package_count"), int)
        or cast(int, lock.get("package_count")) <= 0
        or isinstance(lock.get("wheel_count"), bool)
        or not isinstance(lock.get("wheel_count"), int)
        or cast(int, lock.get("wheel_count")) <= 0
    ):
        raise BundleContractError("the GPU dependency lock metadata is invalid")
    _validate_dependency_reference(
        {field: lock[field] for field in ("path", "sha256", "size")},
        label="lock",
        records_by_path=records_by_path,
    )
    for command_name in (
        "venv_command",
        "compile_command",
        "sync_command",
        "project_install_command",
    ):
        raw_command = environment.get(command_name)
        if not isinstance(raw_command, list) or not raw_command:
            raise BundleContractError(f"the GPU dependency {command_name} is invalid")
        command = cast(list[object], raw_command)
        if not all(isinstance(argument, str) and argument for argument in command):
            raise BundleContractError(f"the GPU dependency {command_name} is invalid")
    return environment


def _validate_string_sequence(value: object, label: str) -> None:
    if not isinstance(value, list):
        raise BundleContractError(f"the GPU training contract {label} is invalid")
    items = cast(list[object], value)
    if not all(isinstance(item, str) and item.strip() == item and item for item in items):
        raise BundleContractError(f"the GPU training contract {label} is invalid")


def _validate_pair_sequence(value: object, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise BundleContractError(f"the GPU training contract {label} is invalid")
    for item in cast(list[object], value):
        values = cast(list[object], item) if isinstance(item, list) else []
        if (
            not isinstance(item, list)
            or len(values) != 2
            or not all(isinstance(tag, str) and tag.strip() == tag and tag for tag in values)
        ):
            raise BundleContractError(f"the GPU training contract {label} is invalid")


def load_embedded_training_contract(
    root: str | Path = ".",
    *,
    require_project_identity: bool = False,
) -> EmbeddedTrainingContract | None:
    """Load and validate an extracted format-2 training contract, if present.

    A metadata-free source tree is accepted only as an existing operator-owned
    Git checkout. This prevents accidental or partial bundle stripping; it is
    not an out-of-tree signature against an account owner who deliberately
    creates and commits a new repository after deleting the bundle contract.
    """

    unresolved_root = Path(root)
    try:
        root_metadata = unresolved_root.lstat()
    except FileNotFoundError as error:
        if require_project_identity:
            raise BundleContractError(
                f"the required project root does not exist: {unresolved_root}"
            ) from error
        return None
    if _metadata_is_link_like(root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
        raise BundleContractError("the extracted GPU bundle root must be a real directory")
    bundle_root = unresolved_root.resolve()
    manifest_path = bundle_root / MANIFEST_NAME
    checksums_path = bundle_root / CHECKSUMS_NAME
    manifest_exists = _entry_exists(manifest_path, label=MANIFEST_NAME)
    checksums_exists = _entry_exists(checksums_path, label=CHECKSUMS_NAME)
    if not manifest_exists and not checksums_exists:
        # Source-layout callers are either authenticated bundles or real Git
        # checkouts. A directory merely named .git is not enough: require a
        # resolvable HEAD rooted here and the verifier itself in the index.
        if (
            require_project_identity or _looks_like_project_source_tree(bundle_root)
        ) and not _is_valid_project_git_checkout(bundle_root):
            raise BundleContractError(
                "the source-layout project has no GPU bundle integrity metadata and is "
                "not a valid Git checkout; re-extract the reviewed bundle"
            )
        return None
    if not manifest_exists or not checksums_exists:
        raise BundleContractError(
            "the extracted GPU bundle has incomplete integrity metadata; "
            f"{MANIFEST_NAME} present={manifest_exists}, {CHECKSUMS_NAME} present={checksums_exists}"
        )
    manifest_content = _read_regular_file(
        manifest_path,
        MANIFEST_NAME,
        limit=MAX_METADATA_BYTES,
    )
    try:
        raw_manifest: object = json.loads(manifest_content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleContractError(f"{MANIFEST_NAME} is not valid UTF-8 JSON") from error
    if not isinstance(raw_manifest, dict):
        raise BundleContractError(f"{MANIFEST_NAME} must contain a JSON object")
    manifest = cast(dict[str, Any], raw_manifest)
    expected_manifest_fields = {
        "archive_root",
        "files",
        "format_version",
        "git",
        "payload",
        "training_contract",
        "zip_metadata",
    }
    if set(manifest) != expected_manifest_fields:
        raise BundleContractError("the embedded manifest fields do not match format 2")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise BundleContractError("the embedded package manifest format is unsupported")
    if manifest.get("archive_root") != ARCHIVE_ROOT:
        raise BundleContractError("the embedded package archive root is invalid")
    raw_git_identity = manifest.get("git")
    if not isinstance(raw_git_identity, dict):
        raise BundleContractError("the embedded package Git identity is invalid")
    git_identity = cast(dict[str, object], raw_git_identity)
    if set(git_identity) != {"commit", "tree"}:
        raise BundleContractError("the embedded package Git identity is invalid")
    if not all(
        isinstance(git_identity.get(field), str)
        and GIT_OBJECT_PATTERN.fullmatch(cast(str, git_identity.get(field)))
        for field in ("commit", "tree")
    ):
        raise BundleContractError("the embedded package Git object IDs are invalid")

    records = _parse_file_records(manifest.get("files"))
    raw_payload = manifest.get("payload")
    if not isinstance(raw_payload, dict):
        raise BundleContractError("the embedded package payload summary is invalid")
    payload = cast(dict[str, object], raw_payload)
    if set(payload) != {"file_count", "total_bytes"}:
        raise BundleContractError("the embedded package payload summary is invalid")
    if payload.get("file_count") != len(records) or payload.get("total_bytes") != sum(
        record.size for record in records
    ):
        raise BundleContractError("the embedded package payload summary does not match its files")
    if manifest.get("zip_metadata") != {
        "compression": "deflate",
        "timestamp": "1980-01-01T00:00:00Z",
        "zip64": True,
    }:
        raise BundleContractError("the embedded package ZIP policy is invalid")

    manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
    checksums_content = _read_regular_file(
        checksums_path,
        CHECKSUMS_NAME,
        limit=MAX_METADATA_BYTES,
    )
    checksums = _parse_checksums(checksums_content)
    expected_checksum_paths = [record.path for record in records] + [MANIFEST_NAME]
    if list(checksums) != expected_checksum_paths:
        raise BundleContractError(f"{CHECKSUMS_NAME} does not match deterministic manifest order")
    for record in records:
        if checksums.get(record.path) != record.sha256:
            raise BundleContractError(f"{CHECKSUMS_NAME} disagrees with {record.path}")
    if checksums.get(MANIFEST_NAME) != manifest_sha256:
        raise BundleContractError(f"{CHECKSUMS_NAME} contains the wrong manifest digest")

    raw_contract = manifest.get("training_contract")
    if not isinstance(raw_contract, dict):
        raise BundleContractError("the embedded GPU bundle has no training contract")
    contract = cast(dict[str, Any], raw_contract)
    expected_contract_fields = {
        "schema",
        "config_path",
        "config_sha256",
        "raw_parallel_data_included",
        "language_pairs",
        "translation_directions",
        "source_only_languages",
        "foundation_enabled",
        "foundation_languages",
        "paths",
        "dependency_environment",
    }
    if set(contract) != expected_contract_fields:
        raise BundleContractError("the embedded GPU training-contract fields are invalid")
    if contract.get("schema") != TRAINING_CONTRACT_SCHEMA:
        raise BundleContractError("the embedded GPU training-contract schema is unsupported")
    config_path = contract.get("config_path")
    config_sha256 = contract.get("config_sha256")
    if (
        config_path != DEFAULT_CONFIG_PATH
        or not isinstance(config_sha256, str)
        or not SHA256_PATTERN.fullmatch(config_sha256)
    ):
        raise BundleContractError(
            "the embedded GPU bundle does not authenticate its default config"
        )
    raw_included = contract.get("raw_parallel_data_included")
    foundation_enabled = contract.get("foundation_enabled")
    if not isinstance(raw_included, bool) or not isinstance(foundation_enabled, bool):
        raise BundleContractError("the embedded GPU training stage flags are invalid")
    _validate_pair_sequence(contract.get("language_pairs"), "language_pairs")
    _validate_pair_sequence(contract.get("translation_directions"), "translation_directions")
    _validate_string_sequence(contract.get("source_only_languages"), "source_only_languages")
    _validate_string_sequence(contract.get("foundation_languages"), "foundation_languages")
    if contract.get("paths") != CANONICAL_PATHS:
        raise BundleContractError("the embedded GPU bundle does not use canonical runtime paths")

    records_by_path = {record.path: record for record in records}
    config_record = records_by_path.get(DEFAULT_CONFIG_PATH)
    if (
        config_record is None
        or config_record.origin != "git-index"
        or config_record.sha256 != config_sha256
    ):
        raise BundleContractError("the selected GPU config is not an authenticated Git payload")
    config_content = _read_regular_file(bundle_root / DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_PATH)
    if len(config_content) != config_record.size or hashlib.sha256(config_content).hexdigest() != (
        config_record.sha256
    ):
        raise BundleContractError(
            "the extracted sion_translate.yaml differs from the GPU bundle training contract"
        )

    raw_records = tuple(record for record in records if record.origin == "data-jsonl")
    monolingual_records = tuple(
        record for record in records if record.origin == "monolingual-corpus"
    )
    if bool(raw_records) != raw_included:
        raise BundleContractError("the GPU training contract disagrees with its raw data payload")
    if not raw_included:
        if monolingual_records:
            raise BundleContractError("a prepared-only GPU bundle contains monolingual sources")
        for origin in ("tokenizer", "dataset"):
            if not any(record.origin == origin for record in records):
                raise BundleContractError(
                    f"a prepared-only GPU bundle is missing its {origin} payload"
                )
        if foundation_enabled and not any(
            record.origin == "foundation-dataset" for record in records
        ):
            raise BundleContractError(
                "a foundation-enabled prepared bundle is missing its foundation dataset"
            )

    dependency_environment = _validate_dependency_environment(
        contract.get("dependency_environment"),
        records_by_path,
    )
    return EmbeddedTrainingContract(
        root=bundle_root,
        manifest_sha256=manifest_sha256,
        records=records,
        config_path=config_path,
        config_sha256=config_sha256,
        config_content=config_content,
        raw_parallel_data_included=raw_included,
        foundation_enabled=foundation_enabled,
        dependency_environment=dependency_environment,
    )


def _runtime_extra_is_allowed(
    path: PurePosixPath,
    mutable_artifact_roots: Sequence[PurePosixPath],
) -> bool:
    if path.parts[0] in RUNTIME_MUTABLE_TOP_LEVEL:
        return True
    if path.name in RUNTIME_LOCK_FILENAMES:
        return True
    if any(part in RUNTIME_CACHE_PARTS for part in path.parts):
        return True
    if any(path == root or path.is_relative_to(root) for root in RUNTIME_EDITABLE_METADATA_ROOTS):
        return True
    return any(path == root or path.is_relative_to(root) for root in mutable_artifact_roots)


def _is_mutable_artifact_path(
    path: PurePosixPath,
    mutable_artifact_roots: Sequence[PurePosixPath],
) -> bool:
    return any(path == root or path.is_relative_to(root) for root in mutable_artifact_roots)


def _walk_runtime_payload(
    root: Path,
    mutable_artifact_roots: Sequence[PurePosixPath],
) -> tuple[set[str], list[str]]:
    files: set[str] = set()
    unsafe: list[str] = []
    for directory, raw_directories, raw_files in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        retained_directories: list[str] = []
        for name in raw_directories:
            candidate = directory_path / name
            relative = PurePosixPath(candidate.relative_to(root).as_posix())
            try:
                metadata = candidate.lstat()
            except OSError:
                unsafe.append(relative.as_posix())
                continue
            if _metadata_is_link_like(metadata) or not stat.S_ISDIR(metadata.st_mode):
                unsafe.append(relative.as_posix())
                continue
            if _is_mutable_artifact_path(relative, mutable_artifact_roots):
                # Generated artifacts are validated by their generation-specific
                # contracts below. Still descend so a nested symlink, junction,
                # reparse point, or special file cannot hide behind this exception.
                retained_directories.append(name)
                continue
            if _runtime_extra_is_allowed(relative, mutable_artifact_roots):
                continue
            retained_directories.append(name)
        raw_directories[:] = retained_directories
        for name in raw_files:
            candidate = directory_path / name
            relative = PurePosixPath(candidate.relative_to(root).as_posix())
            try:
                metadata = candidate.lstat()
            except OSError:
                unsafe.append(relative.as_posix())
                continue
            if _metadata_is_link_like(metadata) or not stat.S_ISREG(metadata.st_mode):
                unsafe.append(relative.as_posix())
                continue
            if _is_mutable_artifact_path(relative, mutable_artifact_roots):
                continue
            if _runtime_extra_is_allowed(relative, mutable_artifact_roots):
                continue
            files.add(_validated_relative_path(relative.as_posix()).as_posix())
    return files, unsafe


def verify_embedded_bundle_payload(contract: EmbeddedTrainingContract) -> None:
    """Hash every immutable payload and reject unexpected project-tree files."""

    _verify_embedded_metadata_identity(contract)
    expected = {record.path for record in contract.records} | {MANIFEST_NAME, CHECKSUMS_NAME}
    mutable_artifact_roots = tuple(
        PurePosixPath(root)
        for origin, root in ARTIFACT_ORIGIN_ROOTS.items()
        if not contract.records_for_origin(origin)
    )
    actual, unsafe = _walk_runtime_payload(contract.root, mutable_artifact_roots)
    if unsafe:
        raise BundleContractError(f"the GPU bundle contains unsafe paths: {sorted(unsafe)}")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BundleContractError(
            f"the GPU bundle runtime file set differs from its manifest; "
            f"missing={missing}, extra={extra}"
        )
    for record in contract.records:
        path = contract.root.joinpath(*PurePosixPath(record.path).parts)
        try:
            metadata = path.lstat()
        except FileNotFoundError as error:
            raise BundleContractError(
                f"the GPU bundle payload is missing: {record.path}"
            ) from error
        if _metadata_is_link_like(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise BundleContractError(f"the GPU bundle payload is not regular: {record.path}")
        if metadata.st_size != record.size or file_sha256(path) != record.sha256:
            raise BundleContractError(f"the GPU bundle payload hash differs: {record.path}")
    # Close the manifest-swap window across the potentially long payload hash
    # pass. The same cached contract must still authenticate the tree when the
    # verifier returns.
    _verify_embedded_metadata_identity(contract)


def _verify_embedded_metadata_identity(contract: EmbeddedTrainingContract) -> None:
    """Require live metadata to remain identical to one cached contract."""

    manifest_content = _read_regular_file(
        contract.root / MANIFEST_NAME,
        MANIFEST_NAME,
        limit=MAX_METADATA_BYTES,
    )
    manifest_sha256 = hashlib.sha256(manifest_content).hexdigest()
    if manifest_sha256 != contract.manifest_sha256:
        raise BundleContractError(
            "the GPU bundle manifest changed after its training contract was selected"
        )
    checksums_content = _read_regular_file(
        contract.root / CHECKSUMS_NAME,
        CHECKSUMS_NAME,
        limit=MAX_METADATA_BYTES,
    )
    checksums = _parse_checksums(checksums_content)
    expected_paths = [record.path for record in contract.records] + [MANIFEST_NAME]
    if list(checksums) != expected_paths:
        raise BundleContractError(
            "the GPU bundle checksum list changed after its training contract was selected"
        )
    for record in contract.records:
        if checksums.get(record.path) != record.sha256:
            raise BundleContractError(f"the GPU bundle checksum identity changed for {record.path}")
    if checksums.get(MANIFEST_NAME) != contract.manifest_sha256:
        raise BundleContractError(
            "the GPU bundle manifest checksum changed after its training contract was selected"
        )


def _source_identity_difference(
    expected: Mapping[str, tuple[int, str]],
    actual: Mapping[str, tuple[int, str]],
) -> str | None:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(
        path for path in set(expected).intersection(actual) if expected[path] != actual[path]
    )
    if not missing and not extra and not changed:
        return None
    return f"missing={missing}, extra={extra}, changed={changed}"


def validate_parallel_source_inventory(
    contract: EmbeddedTrainingContract,
    fingerprint: DatasetFingerprint,
) -> None:
    """Require the effective parallel files to match the manifest exactly."""

    expected = {
        record.path: (record.size, record.sha256)
        for record in contract.records_for_origin("data-jsonl")
    }
    actual = {f"data/{source.name}": (source.size, source.sha256) for source in fingerprint.files}
    difference = _source_identity_difference(expected, actual)
    if difference is not None:
        raise BundleContractError(
            "effective parallel training sources differ from the GPU bundle contract; "
            f"{difference}. Re-extract the reviewed bundle instead of rebuilding in place."
        )


def validate_monolingual_source_inventory(
    contract: EmbeddedTrainingContract,
    sources: Sequence[Any],
) -> None:
    """Require the effective foundation files to match the manifest exactly."""

    expected = {
        record.path: (record.size, record.sha256)
        for record in contract.records_for_origin("monolingual-corpus")
    }
    actual: dict[str, tuple[int, str]] = {}
    for source in sources:
        raw_path = getattr(source, "path", None)
        if not isinstance(raw_path, Path):
            raise BundleContractError("foundation discovery returned an invalid source path")
        path = raw_path.resolve()
        try:
            relative = path.relative_to(contract.root).as_posix()
        except ValueError as error:
            raise BundleContractError(
                f"foundation source escapes the extracted GPU bundle: {path}"
            ) from error
        canonical = _validated_relative_path(relative).as_posix()
        try:
            metadata = path.lstat()
        except FileNotFoundError as error:
            raise BundleContractError(f"foundation source disappeared: {canonical}") from error
        if _metadata_is_link_like(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise BundleContractError(f"foundation source is not a regular file: {canonical}")
        if canonical in actual:
            raise BundleContractError(f"foundation discovery repeated a source: {canonical}")
        actual[canonical] = (metadata.st_size, file_sha256(path))
    difference = _source_identity_difference(expected, actual)
    if difference is not None:
        raise BundleContractError(
            "effective monolingual training sources differ from the GPU bundle contract; "
            f"{difference}. Re-extract the reviewed bundle instead of rebuilding in place."
        )


def validate_tokenizer_source_inventory(
    contract: EmbeddedTrainingContract,
    metadata: Mapping[str, object] | None,
    fingerprint: DatasetFingerprint,
    monolingual_root: Path,
    monolingual_sources: Sequence[Any],
    *,
    expected_policy: Mapping[str, object] | None = None,
) -> None:
    """Bind a generated tokenizer to the exact authenticated source traversal."""

    if not contract.raw_parallel_data_included:
        return
    if metadata is None:
        raise BundleContractError(
            "a generated tokenizer in a raw GPU bundle has no authenticated training contract"
        )
    raw_training_contract = metadata.get("training_contract")
    if not isinstance(raw_training_contract, dict):
        raise BundleContractError(
            "a generated tokenizer in a raw GPU bundle has no authenticated training contract"
        )
    training_contract = cast(dict[str, object], raw_training_contract)
    if training_contract.get("schema") != TOKENIZER_TRAINING_SCHEMA:
        raise BundleContractError("the generated tokenizer training-contract schema is stale")
    if training_contract.get("input_traversal_policy") != TOKENIZER_INPUT_TRAVERSAL_POLICY:
        raise BundleContractError("the generated tokenizer input traversal policy is stale")
    canonical_contract = json.dumps(
        training_contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if metadata.get("training_contract_sha256") != hashlib.sha256(canonical_contract).hexdigest():
        raise BundleContractError("the generated tokenizer training-contract digest differs")
    for field, expected in (expected_policy or {}).items():
        if training_contract.get(field) != expected:
            raise BundleContractError(
                f"generated tokenizer policy {field} differs from the authenticated GPU config"
            )

    expected_sources: list[dict[str, object]] = [
        {
            "role": "parallel",
            "path": source.name,
            "size": source.size,
            "sha256": source.sha256,
        }
        for source in sorted(
            fingerprint.files,
            key=lambda item: (item.name.casefold(), item.name),
        )
    ]
    monolingual_root = monolingual_root.resolve()
    monolingual_identities = {
        record.path: (record.size, record.sha256)
        for record in contract.records_for_origin("monolingual-corpus")
    }
    for source in monolingual_sources:
        raw_path = getattr(source, "path", None)
        language = getattr(source, "language", None)
        if not isinstance(raw_path, Path) or not isinstance(language, str):
            raise BundleContractError("foundation discovery returned invalid tokenizer provenance")
        source_path = raw_path.resolve()
        try:
            bundle_path = source_path.relative_to(contract.root).as_posix()
            logical_path = source_path.relative_to(monolingual_root).as_posix()
        except ValueError as error:
            raise BundleContractError(
                f"tokenizer monolingual source escapes its authenticated root: {source_path}"
            ) from error
        canonical_bundle_path = _validated_relative_path(bundle_path).as_posix()
        canonical_logical_path = _validated_relative_path(logical_path).as_posix()
        identity = monolingual_identities.get(canonical_bundle_path)
        if identity is None:
            raise BundleContractError(
                f"tokenizer monolingual source is absent from the GPU bundle: {bundle_path}"
            )
        expected_sources.append(
            {
                "role": "monolingual",
                "path": canonical_logical_path,
                "size": identity[0],
                "sha256": identity[1],
                "language": language,
            }
        )
    if training_contract.get("sources") != expected_sources:
        raise BundleContractError(
            "generated tokenizer source provenance differs from the authenticated GPU bundle"
        )


def resolved_origin_identities(
    contract: EmbeddedTrainingContract,
    origin: str,
) -> dict[str, tuple[int, str]]:
    """Return absolute identities suitable for a transactional preparation API."""

    return {
        str(contract.root.joinpath(*PurePosixPath(record.path).parts).resolve()): (
            record.size,
            record.sha256,
        )
        for record in contract.records_for_origin(origin)
    }


__all__ = [
    "BundleContractError",
    "BundleFileRecord",
    "EmbeddedTrainingContract",
    "load_embedded_training_contract",
    "resolved_origin_identities",
    "validate_monolingual_source_inventory",
    "validate_parallel_source_inventory",
    "validate_tokenizer_source_inventory",
    "verify_embedded_bundle_payload",
]
