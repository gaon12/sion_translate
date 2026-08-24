"""Train the exact production tokenizer on Modal without replacing any artifact.

The local entrypoint fingerprints every raw file selected by the production
configuration.  The remote worker mounts the existing ``sion-dataset`` volume,
selects the inputs again, verifies every byte, and only then calls the project's
``train_tokenizer`` implementation.  Results are published to a fresh directory
in a separate output volume; existing tokenizer generations are never replaced.

Run from a clean repository checkout::

    python -m modal run scripts/modal_train_tokenizer.py
"""

# Modal's decorators and result payloads are dynamically typed.
# pyright: reportRedeclaration=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, cast


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPOSITORY_ROOT = "/opt/sion"
INPUT_MOUNT = "/input"
OUTPUT_MOUNT = "/output"
INPUT_VOLUME_NAME = "sion-dataset"
OUTPUT_VOLUME_NAME = "sion-tokenizer-production"
APP_NAME = "sion-tokenizer-production"
REMOTE_SCRIPT_PATH = f"{REMOTE_REPOSITORY_ROOT}/scripts/modal_train_tokenizer.py"
CHILD_MODE_FLAG = "--modal-tokenizer-child"
CHILD_HEARTBEAT_SECONDS = 45.0

SOURCE_MANIFEST_VERSION = 2
TRAINING_MANIFEST_VERSION = 2
EXPECTED_SENTENCEPIECE_VERSION = "0.2.1"
EXPECTED_VOCAB_SIZE = 48_000
REQUIRED_ARTIFACTS = frozenset(
    {"sion.model", "sion.vocab", "token_features.npz", "tokenizer_metadata.json"}
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_CHILD_MODE = CHILD_MODE_FLAG in sys.argv


@dataclass(frozen=True)
class SourceRecord:
    path: str
    size: int
    sha256: str


def _add_source_path(source_root: Path) -> None:
    source_directory = str((source_root / "src").resolve())
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)


def _load_production_modules(source_root: Path) -> tuple[Any, Any, Any]:
    """Import production modules only after their source tree is on ``sys.path``."""

    _add_source_path(source_root)
    config_module = importlib.import_module("sion_translate.config")
    monolingual_module = importlib.import_module("sion_translate.data.monolingual")
    tokenizer_module = importlib.import_module("sion_translate.tokenizer")
    return config_module, monolingual_module, tokenizer_module


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_identity(path: Path, *, role: str) -> tuple[int, str]:
    """Hash one regular file and reject replacement or mutation during the read."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file: {path}")
    before = path.stat()
    digest = _file_sha256(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{role} changed while it was hashed: {path}")
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"{role} changed while it was hashed: {path}")
    return after.st_size, digest


def _safe_relative_path(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise ValueError(f"manifest path must be non-empty POSIX text: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"manifest path must stay below the input root: {value!r}")
    return path


def _source_paths(
    input_root: Path,
    config_path: Path,
    *,
    source_root: Path,
) -> tuple[list[Path], Any]:
    """Select exactly the files consumed by the configured tokenizer path."""

    config_module, monolingual_module, _ = _load_production_modules(source_root)
    config = config_module.load_config(config_path)
    languages = config.foundation_languages()
    data_directory = input_root / config.data.raw_dir
    corpus_directory = input_root / config.foundation.corpus_dir
    parallel_paths = sorted(path for path in data_directory.glob("*.jsonl") if path.is_file())
    discovery = monolingual_module.discover_monolingual_sources(
        corpus_directory,
        languages,
    )
    if not parallel_paths:
        raise FileNotFoundError(f"no production JSONL inputs under {data_directory}")
    if not discovery.sources:
        raise FileNotFoundError(f"no configured monolingual inputs under {corpus_directory}")
    if discovery.languages_without_data:
        raise RuntimeError(
            "configured foundation languages have no corpus: "
            f"{list(discovery.languages_without_data)}"
        )
    paths = sorted(
        [*parallel_paths, *(source.path for source in discovery.sources)],
        key=lambda path: path.relative_to(input_root).as_posix(),
    )
    if len(paths) != len(set(paths)):
        raise RuntimeError("production source discovery returned duplicate paths")
    return paths, config


def _hash_source(path: Path, input_root: Path) -> SourceRecord:
    resolved_root = input_root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError(f"source escapes the input root: {path}") from error
    size, digest = _stable_file_identity(path, role=f"source {relative}")
    return SourceRecord(relative, size, digest)


def _records_digest(records: Sequence[SourceRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.path):
        digest.update(f"{record.path}\0{record.size}\0{record.sha256}\n".encode())
    return digest.hexdigest()


def _manifest_digest(manifest: Mapping[str, object]) -> str:
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_source_manifest(
    repository_root: Path,
    *,
    git_commit: str,
    workers: int = 4,
) -> dict[str, object]:
    """Hash the local production inputs selected by the current configuration."""

    if COMMIT_PATTERN.fullmatch(git_commit) is None:
        raise ValueError(f"invalid Git commit: {git_commit!r}")
    config_path = repository_root / "sion_translate.yaml"
    _, config_digest = _stable_file_identity(config_path, role="production configuration")
    paths, config = _source_paths(
        repository_root,
        config_path,
        source_root=repository_root,
    )

    def hash_source(path: Path) -> SourceRecord:
        return _hash_source(path, repository_root)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        records = list(executor.map(hash_source, paths))
    records.sort(key=lambda record: record.path)
    selected_after, _ = _source_paths(
        repository_root,
        config_path,
        source_root=repository_root,
    )
    if [path.resolve() for path in selected_after] != [path.resolve() for path in paths]:
        raise RuntimeError("production source selection changed while the manifest was built")
    _, config_digest_after = _stable_file_identity(
        config_path,
        role="production configuration",
    )
    if config_digest_after != config_digest:
        raise RuntimeError("production configuration changed while the manifest was built")
    ratio = float(config.foundation.tokenizer_sample_ratio)
    if not math.isfinite(ratio) or ratio < 0:
        raise RuntimeError("configured tokenizer sample ratio must be finite and non-negative")
    return {
        "version": SOURCE_MANIFEST_VERSION,
        "git_commit": git_commit,
        "config_path": "sion_translate.yaml",
        "config_sha256": config_digest,
        "tokenizer_sample_ratio": ratio,
        "file_count": len(records),
        "total_bytes": sum(record.size for record in records),
        "files_sha256": _records_digest(records),
        "files": [asdict(record) for record in records],
    }


def _parse_source_manifest(manifest: Mapping[str, object]) -> list[SourceRecord]:
    if manifest.get("version") != SOURCE_MANIFEST_VERSION:
        raise ValueError("unsupported source manifest version")
    commit = manifest.get("git_commit")
    if not isinstance(commit, str) or COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("source manifest has an invalid Git commit")
    ratio = manifest.get("tokenizer_sample_ratio")
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or ratio < 0
    ):
        raise ValueError("source manifest tokenizer sample ratio is invalid")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("source manifest files must be a list")
    records: list[SourceRecord] = []
    seen: set[str] = set()
    for raw_record in raw_files:
        if not isinstance(raw_record, Mapping):
            raise ValueError("source manifest file entries must be objects")
        path = raw_record.get("path")
        size = raw_record.get("size")
        digest = raw_record.get("sha256")
        if not isinstance(path, str):
            raise ValueError("source manifest file path must be text")
        _safe_relative_path(path)
        if path in seen:
            raise ValueError(f"duplicate source manifest path: {path}")
        seen.add(path)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid source size for {path}")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"invalid source SHA-256 for {path}")
        records.append(SourceRecord(path, size, digest))
    records.sort(key=lambda record: record.path)
    if manifest.get("file_count") != len(records):
        raise ValueError("source manifest file count does not match its records")
    if manifest.get("total_bytes") != sum(record.size for record in records):
        raise ValueError("source manifest byte count does not match its records")
    if manifest.get("files_sha256") != _records_digest(records):
        raise ValueError("source manifest aggregate digest does not match its records")
    return records


def verify_source_manifest(
    input_root: Path,
    config_path: Path,
    manifest: Mapping[str, object],
    *,
    source_root: Path,
    workers: int,
) -> tuple[list[Path], Any]:
    """Re-select and byte-verify every production source on the remote volume."""

    records = _parse_source_manifest(manifest)
    config_digest = manifest.get("config_sha256")
    _, observed_config_digest = _stable_file_identity(
        config_path,
        role="remote production configuration",
    )
    if not isinstance(config_digest, str) or config_digest != observed_config_digest:
        raise RuntimeError("remote production configuration differs from the local manifest")
    paths, config = _source_paths(input_root, config_path, source_root=source_root)
    selected = {path.relative_to(input_root).as_posix(): path for path in paths}
    expected = {record.path for record in records}
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise RuntimeError(
            f"remote production source set differs; missing={missing}, extra={extra}"
        )

    def verify(record: SourceRecord) -> None:
        path = selected[record.path]
        actual = _hash_source(path, input_root)
        if actual.size != record.size or actual.sha256 != record.sha256:
            raise RuntimeError(
                f"remote source identity differs for {record.path}: "
                f"expected size={record.size}, sha256={record.sha256}; "
                f"actual size={actual.size}, sha256={actual.sha256}"
            )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        list(executor.map(verify, records))
    selected_after, _ = _source_paths(input_root, config_path, source_root=source_root)
    if [path.resolve() for path in selected_after] != [path.resolve() for path in paths]:
        raise RuntimeError("remote production source selection changed during verification")
    _, config_digest_after = _stable_file_identity(
        config_path,
        role="remote production configuration",
    )
    if config_digest_after != config_digest:
        raise RuntimeError("remote production configuration changed during verification")
    return paths, config


def _artifact_records(directory: Path) -> dict[str, dict[str, object]]:
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != set(REQUIRED_ARTIFACTS):
        raise RuntimeError(
            "tokenizer output is incomplete or contains unexpected files: "
            f"expected={sorted(REQUIRED_ARTIFACTS)}, actual={sorted(actual)}"
        )
    records: dict[str, dict[str, object]] = {}
    for name in sorted(REQUIRED_ARTIFACTS):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"tokenizer artifact is not a regular file: {name}")
        records[name] = {"size": path.stat().st_size, "sha256": _file_sha256(path)}
    return records


def _validate_training_metadata(
    directory: Path,
    expected_sample_ratio: float,
    artifacts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    metadata_path = directory / "tokenizer_metadata.json"
    metadata_value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata_value, dict):
        raise RuntimeError("tokenizer metadata is not a JSON object")
    metadata = cast(dict[str, object], metadata_value)
    if metadata.get("sentencepiece_version") != EXPECTED_SENTENCEPIECE_VERSION:
        raise RuntimeError("tokenizer metadata records the wrong SentencePiece version")
    if metadata.get("monolingual_sample_ratio") != expected_sample_ratio:
        raise RuntimeError("tokenizer metadata records the wrong monolingual ratio")
    corpus_sentences = metadata.get("corpus_sentences")
    sampled_sentences = metadata.get("sampled_sentences")
    if (
        not isinstance(corpus_sentences, int)
        or isinstance(corpus_sentences, bool)
        or corpus_sentences < 1
    ):
        raise RuntimeError("tokenizer metadata has an invalid corpus sentence count")
    if (
        not isinstance(sampled_sentences, int)
        or isinstance(sampled_sentences, bool)
        or not 1 <= sampled_sentences <= corpus_sentences
    ):
        raise RuntimeError("tokenizer metadata has an invalid sampled sentence count")

    def sentence_counts(field: str, expected_total: int | None = None) -> dict[str, int]:
        raw_counts = metadata.get(field)
        if not isinstance(raw_counts, Mapping):
            raise RuntimeError(f"tokenizer metadata {field} must be an object")
        normalized: dict[str, int] = {}
        for raw_language, raw_count in raw_counts.items():
            if not isinstance(raw_language, str) or not raw_language:
                raise RuntimeError(f"tokenizer metadata {field} has an invalid language")
            if not isinstance(raw_count, int) or isinstance(raw_count, bool) or raw_count < 0:
                raise RuntimeError(f"tokenizer metadata {field} has an invalid count")
            normalized[raw_language] = raw_count
        if expected_total is not None and sum(normalized.values()) != expected_total:
            raise RuntimeError(f"tokenizer metadata {field} does not sum to its total")
        return normalized

    corpus_by_language = sentence_counts(
        "corpus_sentences_per_language",
        corpus_sentences,
    )
    sampled_by_language = sentence_counts(
        "sampled_sentences_per_language",
        sampled_sentences,
    )
    if set(corpus_by_language) != set(sampled_by_language):
        raise RuntimeError("sampled tokenizer languages differ from the full corpus")
    if any(sampled_by_language[language] > count for language, count in corpus_by_language.items()):
        raise RuntimeError("sampled tokenizer language count exceeds its corpus count")
    monolingual = sentence_counts("monolingual_sentences")
    if not set(monolingual).issubset(sampled_by_language):
        raise RuntimeError("monolingual tokenizer counts contain an unknown language")
    if any(monolingual[language] > sampled_by_language[language] for language in monolingual):
        raise RuntimeError("monolingual tokenizer count exceeds its sampled language count")
    contract = metadata.get("training_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("tokenizer metadata has no authenticated training contract")
    if metadata.get("training_contract_sha256") != _manifest_digest(contract):
        raise RuntimeError("tokenizer metadata training contract digest is invalid")
    if artifacts is not None:
        recorded_identities = {
            "sion.model": metadata.get("model_sha256"),
            "sion.vocab": metadata.get("vocab_sha256"),
            "token_features.npz": metadata.get("token_features_sha256"),
        }
        for name, recorded_digest in recorded_identities.items():
            if recorded_digest != artifacts[name]["sha256"]:
                raise RuntimeError(f"tokenizer metadata identity differs for {name}")
    return metadata


def _validate_training_source_identities(
    metadata: Mapping[str, object],
    expected_sources: Sequence[SourceRecord],
) -> None:
    """Prove that the child trained on the bytes authenticated by its parent."""

    contract = metadata.get("training_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("tokenizer metadata has no training contract")
    raw_sources = contract.get("sources")
    if not isinstance(raw_sources, list):
        raise RuntimeError("tokenizer training contract has no source identities")
    observed: list[tuple[int, str]] = []
    for raw_source in raw_sources:
        if not isinstance(raw_source, Mapping):
            raise RuntimeError("tokenizer training contract has an invalid source entry")
        size = raw_source.get("size")
        digest = raw_source.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError("tokenizer training contract has an invalid source size")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise RuntimeError("tokenizer training contract has an invalid source digest")
        observed.append((size, digest))
    expected = sorted((source.size, source.sha256) for source in expected_sources)
    if sorted(observed) != expected:
        raise RuntimeError(
            "tokenizer child trained on source bytes that differ from the parent manifest"
        )


def _publish_candidate(build_directory: Path, output_root: Path, candidate_name: str) -> Path:
    """Copy to a staging directory and atomically publish without replacement."""

    if RUN_ID_PATTERN.fullmatch(candidate_name) is None:
        raise ValueError(f"invalid candidate name: {candidate_name!r}")
    output_root.mkdir(parents=True, exist_ok=True)
    final_directory = output_root / candidate_name
    staging_directory = output_root / f".{candidate_name}.staging"
    if final_directory.exists() or staging_directory.exists():
        raise FileExistsError(f"refusing to replace an existing candidate: {candidate_name}")
    shutil.copytree(build_directory, staging_directory)
    os.rename(staging_directory, final_directory)
    return final_directory


def _git_commit(repository_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("tracked Git changes exist; commit them before starting Modal training")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError(f"Git returned an invalid commit identity: {commit!r}")
    return commit


def _new_run_id(commit: str, source_manifest: Mapping[str, object]) -> str:
    # Keep the generated identifier inside RUN_ID_PATTERN.  This is checked
    # again only after the expensive trainer returns, so an uppercase T/Z here
    # would waste the whole run before publication.
    timestamp = datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz")
    return (
        f"tokenizer-{commit[:12]}-{_manifest_digest(source_manifest)[:12]}-"
        f"{timestamp}-{secrets.token_hex(3)}"
    )


def _cpu_plan_payload(cpu_plan: Any) -> dict[str, int]:
    payload = {
        "available": int(cpu_plan.available),
        "preprocess_workers": int(cpu_plan.preprocess_workers),
        "sentencepiece_threads": int(cpu_plan.sentencepiece_threads),
    }
    if (
        payload["preprocess_workers"] < 1
        or payload["sentencepiece_threads"] < 1
        or payload["preprocess_workers"] + payload["sentencepiece_threads"] != payload["available"]
    ):
        raise RuntimeError(f"invalid production CPU plan: {payload}")
    return payload


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    if path.exists():
        raise FileExistsError(f"refusing to replace an existing result file: {path}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _train_child(build_directory: Path, result_path: Path) -> None:
    """Run production training in a process whose GIL cannot starve Modal."""

    remote_repository_root = Path(REMOTE_REPOSITORY_ROOT)
    input_mount = Path(INPUT_MOUNT)
    config_path = remote_repository_root / "sion_translate.yaml"
    _, monolingual_module, tokenizer_module = _load_production_modules(remote_repository_root)
    sentencepiece = importlib.import_module("sentencepiece")
    sentencepiece_version = str(getattr(sentencepiece, "__version__", "unknown"))
    if sentencepiece_version != EXPECTED_SENTENCEPIECE_VERSION:
        raise RuntimeError(
            f"expected SentencePiece {EXPECTED_SENTENCEPIECE_VERSION}, got {sentencepiece_version}"
        )
    if build_directory.exists() or result_path.exists():
        raise FileExistsError("child output paths must not exist before training")

    paths, config = _source_paths(
        input_mount,
        config_path,
        source_root=remote_repository_root,
    )
    ratio = float(config.foundation.tokenizer_sample_ratio)
    if not math.isfinite(ratio) or ratio < 0:
        raise RuntimeError("child tokenizer ratio must be finite and non-negative")
    pairs = config.data.configured_language_pairs()
    foundation_languages = config.foundation_languages()
    discovery = monolingual_module.discover_monolingual_sources(
        input_mount / config.foundation.corpus_dir,
        foundation_languages,
    )
    reasoning_languages = (
        tuple(
            dict.fromkeys(
                source.language
                for source in discovery.sources
                if source.path.suffix.lower() == ".jsonl"
                and source.path.name.lower().startswith("reasoning_")
            )
        )
        if config.foundation.enabled
        else ()
    )
    cpu_plan = tokenizer_module.build_cpu_plan(input_files=len(paths))
    cpu_plan_payload = _cpu_plan_payload(cpu_plan)

    tokenizer_module.train_tokenizer(
        [str(input_mount / config.data.raw_dir / "*.jsonl")],
        build_directory,
        vocab_size=EXPECTED_VOCAB_SIZE,
        language_pairs=pairs,
        translation_directions=config.data.configured_translation_directions(),
        monolingual=discovery,
        monolingual_sample_ratio=ratio,
        foundation_languages=foundation_languages,
        reasoning_languages=reasoning_languages,
        approximate_split=config.data.approximate_split,
        source_only_languages=config.data.configured_source_only_languages(),
        train_only_prefixes=config.data.configured_synthetic_prefixes(),
        num_workers=cpu_plan.preprocess_workers,
        num_threads=cpu_plan.sentencepiece_threads,
    )
    artifacts = _artifact_records(build_directory)
    metadata = _validate_training_metadata(build_directory, ratio, artifacts)
    _write_json_atomic(
        result_path,
        {
            "sentencepiece_version": sentencepiece_version,
            "tokenizer_sample_ratio": ratio,
            "corpus_sentences": metadata.get("corpus_sentences"),
            "corpus_sentences_per_language": metadata.get("corpus_sentences_per_language"),
            "sampled_sentences": metadata.get("sampled_sentences"),
            "sampled_sentences_per_language": metadata.get("sampled_sentences_per_language"),
            "monolingual_sentences": metadata.get("monolingual_sentences"),
            "training_contract_sha256": metadata.get("training_contract_sha256"),
            "cpu_plan": cpu_plan_payload,
        },
    )


def _child_failure_message(returncode: int) -> str:
    signal_number: int | None = None
    if returncode < 0:
        signal_number = -returncode
    elif returncode >= 128:
        signal_number = returncode - 128
    if signal_number is not None:
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"signal {signal_number}"
        return f"tokenizer child terminated by {signal_name} (return code {returncode})"
    return f"tokenizer child exited with return code {returncode}"


def _stop_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_training_subprocess(
    command: Sequence[str],
    *,
    heartbeat_seconds: float = CHILD_HEARTBEAT_SECONDS,
) -> None:
    """Wait without capturing child logs, periodically yielding the parent GIL."""

    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    started = time.monotonic()
    process = subprocess.Popen(list(command))
    print(f"[modal-parent] tokenizer child started: pid={process.pid}", flush=True)
    try:
        while True:
            try:
                returncode = process.wait(timeout=heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                print(
                    f"[modal-parent] tokenizer child alive: pid={process.pid}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    except BaseException:
        _stop_child(process)
        raise
    if returncode != 0:
        raise RuntimeError(_child_failure_message(returncode))
    print(
        f"[modal-parent] tokenizer child completed: pid={process.pid}, "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )


def _load_child_result(path: Path, *, expected_sample_ratio: float) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"tokenizer child succeeded without a result JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("tokenizer child result must be a JSON object")
    result = cast(dict[str, object], value)
    if result.get("sentencepiece_version") != EXPECTED_SENTENCEPIECE_VERSION:
        raise RuntimeError("tokenizer child reported the wrong SentencePiece version")
    if result.get("tokenizer_sample_ratio") != expected_sample_ratio:
        raise RuntimeError("tokenizer child reported the wrong sample ratio")
    corpus_sentences = result.get("corpus_sentences")
    sampled_sentences = result.get("sampled_sentences")
    if (
        not isinstance(corpus_sentences, int)
        or isinstance(corpus_sentences, bool)
        or corpus_sentences < 1
    ):
        raise RuntimeError("tokenizer child reported an invalid corpus sentence count")
    if (
        not isinstance(sampled_sentences, int)
        or isinstance(sampled_sentences, bool)
        or not 1 <= sampled_sentences <= corpus_sentences
    ):
        raise RuntimeError("tokenizer child reported an invalid sampled sentence count")
    for field in (
        "corpus_sentences_per_language",
        "sampled_sentences_per_language",
        "monolingual_sentences",
    ):
        if not isinstance(result.get(field), dict):
            raise RuntimeError(f"tokenizer child result has no {field}")
    contract_digest = result.get("training_contract_sha256")
    if not isinstance(contract_digest, str) or SHA256_PATTERN.fullmatch(contract_digest) is None:
        raise RuntimeError("tokenizer child result has no valid training contract digest")
    cpu_plan = result.get("cpu_plan")
    if not isinstance(cpu_plan, dict):
        raise RuntimeError("tokenizer child result has no CPU plan")
    return result


def _train_remote(source_manifest: Mapping[str, object], run_id: str) -> dict[str, object]:
    """Verify in the Modal parent, train in a child, and publish in the parent."""

    remote_repository_root = Path(REMOTE_REPOSITORY_ROOT)
    input_mount = Path(INPUT_MOUNT)
    output_mount = Path(OUTPUT_MOUNT)
    config_path = remote_repository_root / "sion_translate.yaml"
    _, _, tokenizer_module = _load_production_modules(remote_repository_root)
    sentencepiece = importlib.import_module("sentencepiece")
    sentencepiece_version = str(getattr(sentencepiece, "__version__", "unknown"))
    if sentencepiece_version != EXPECTED_SENTENCEPIECE_VERSION:
        raise RuntimeError(
            f"expected SentencePiece {EXPECTED_SENTENCEPIECE_VERSION}, got {sentencepiece_version}"
        )
    paths, config = verify_source_manifest(
        input_mount,
        config_path,
        source_manifest,
        source_root=remote_repository_root,
        workers=16,
    )
    ratio = float(config.foundation.tokenizer_sample_ratio)
    if ratio != source_manifest.get("tokenizer_sample_ratio"):
        raise RuntimeError("remote tokenizer ratio changed after source verification")
    parent_cpu_plan = _cpu_plan_payload(tokenizer_module.build_cpu_plan(input_files=len(paths)))

    with tempfile.TemporaryDirectory(prefix="sion-tokenizer-") as workspace:
        workspace_path = Path(workspace)
        build_directory = workspace_path / "tokenizer"
        child_result_path = workspace_path / "child-result.json"
        command = [
            sys.executable,
            REMOTE_SCRIPT_PATH,
            CHILD_MODE_FLAG,
            "--build-directory",
            str(build_directory),
            "--result",
            str(child_result_path),
        ]
        _run_training_subprocess(command)
        child_result = _load_child_result(
            child_result_path,
            expected_sample_ratio=ratio,
        )
        if child_result["cpu_plan"] != parent_cpu_plan:
            raise RuntimeError(
                "tokenizer child CPU plan differs from its verified parent: "
                f"parent={parent_cpu_plan}, child={child_result['cpu_plan']}"
            )
        artifacts = _artifact_records(build_directory)
        metadata = _validate_training_metadata(build_directory, ratio, artifacts)
        _validate_training_source_identities(
            metadata,
            _parse_source_manifest(source_manifest),
        )
        training_manifest: dict[str, object] = {
            "version": TRAINING_MANIFEST_VERSION,
            "run_id": run_id,
            "git_commit": source_manifest["git_commit"],
            "source_manifest_sha256": _manifest_digest(source_manifest),
            "source_files_sha256": source_manifest["files_sha256"],
            "source_file_count": source_manifest["file_count"],
            "source_total_bytes": source_manifest["total_bytes"],
            "config_sha256": source_manifest["config_sha256"],
            "sentencepiece_version": sentencepiece_version,
            "tokenizer_sample_ratio": ratio,
            "vocab_size": EXPECTED_VOCAB_SIZE,
            "sentence_counts": {
                "corpus": metadata.get("corpus_sentences"),
                "corpus_per_language": metadata.get("corpus_sentences_per_language"),
                "sampled": metadata.get("sampled_sentences"),
                "sampled_per_language": metadata.get("sampled_sentences_per_language"),
                "monolingual_sampled": metadata.get("monolingual_sentences"),
            },
            "cpu_plan": parent_cpu_plan,
            "training_contract_sha256": metadata.get("training_contract_sha256"),
            "required_character_count": metadata.get("required_character_count"),
            "required_characters_sha256": metadata.get("required_characters_sha256"),
            "artifacts": artifacts,
        }
        manifest_path = build_directory / "training_manifest.json"
        _write_json_atomic(manifest_path, training_manifest)
        final_directory = _publish_candidate(
            build_directory,
            output_mount / "tokenizer_candidates",
            run_id,
        )

    result = {
        "output_directory": str(final_directory),
        "training_manifest_sha256": _file_sha256(final_directory / "training_manifest.json"),
        **training_manifest,
    }
    return result


def _child_main(arguments: Sequence[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="SentencePiece child process")
    parser.add_argument(CHILD_MODE_FLAG, action="store_true", required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    _train_child(parsed.build_directory, parsed.result)
    return 0


if not _CHILD_MODE:
    try:
        import modal as _modal
    except ModuleNotFoundError:  # Unit tests do not require the optional Modal client.
        _modal = None
else:
    # A child must not construct Modal Images or Apps while re-importing this file.
    _modal = None


if _modal is not None:
    image = (
        _modal.Image.debian_slim(python_version="3.11")
        .pip_install("numpy>=2.0", "PyYAML>=6.0", "sentencepiece==0.2.1")
        .add_local_dir(
            str(REPOSITORY_ROOT / "src"),
            remote_path=f"{REMOTE_REPOSITORY_ROOT}/src",
            copy=True,
        )
        .add_local_file(
            str(REPOSITORY_ROOT / "sion_translate.yaml"),
            remote_path=f"{REMOTE_REPOSITORY_ROOT}/sion_translate.yaml",
            copy=True,
        )
        .add_local_file(
            str(REPOSITORY_ROOT / "scripts" / "modal_train_tokenizer.py"),
            remote_path=REMOTE_SCRIPT_PATH,
            copy=True,
        )
    )
    app = _modal.App(APP_NAME, image=image)
    input_volume = _modal.Volume.from_name(INPUT_VOLUME_NAME)
    output_volume = _modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

    @app.function(
        volumes={INPUT_MOUNT: input_volume, OUTPUT_MOUNT: output_volume},
        cpu=16.0,
        memory=262_144,
        timeout=8 * 60 * 60,
        retries=0,
    )
    def train_exact(source_manifest: dict[str, object], run_id: str) -> dict[str, object]:
        result = _train_remote(source_manifest, run_id)
        output_volume.commit()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return result

    @app.local_entrypoint()
    def main() -> None:
        commit = _git_commit(REPOSITORY_ROOT)
        source_manifest = build_source_manifest(REPOSITORY_ROOT, git_commit=commit)
        run_id = _new_run_id(commit, source_manifest)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "git_commit": commit,
                    "source_manifest_sha256": _manifest_digest(source_manifest),
                    "source_file_count": source_manifest["file_count"],
                    "source_total_bytes": source_manifest["total_bytes"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        result = cast(Any, train_exact).remote(source_manifest, run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

else:
    app = None

    def train_exact(source_manifest: dict[str, object], run_id: str) -> dict[str, object]:
        del source_manifest, run_id
        raise RuntimeError("install the Modal client to run remote tokenizer training")

    def main() -> None:
        raise RuntimeError("install the Modal client to run remote tokenizer training")


if __name__ == "__main__":
    if _CHILD_MODE:
        raise SystemExit(_child_main(sys.argv[1:]))
    main()
