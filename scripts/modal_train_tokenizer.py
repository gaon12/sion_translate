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

from collections.abc import Generator, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from typing import Any, cast


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPOSITORY_ROOT = "/opt/sion"
INPUT_MOUNT = "/input"
OUTPUT_MOUNT = "/output"
INPUT_VOLUME_NAME = "sion-dataset"
OUTPUT_VOLUME_NAME = "sion-tokenizer-production"
APP_NAME = "sion-tokenizer-production"

SOURCE_MANIFEST_VERSION = 1
TRAINING_MANIFEST_VERSION = 1
EXPECTED_SENTENCEPIECE_VERSION = "0.2.1"
EXPECTED_TOKENIZER_SAMPLE_RATIO = 0.40
EXPECTED_PARALLEL_SENTENCES = 18_177_344
EXPECTED_MONOLINGUAL_SENTENCES = {"ja": 3_445_471, "ko": 3_308_940}
EXPECTED_TOTAL_SENTENCES = 24_931_755
EXPECTED_VOCAB_SIZE = 48_000
REQUIRED_ARTIFACTS = frozenset(
    {"sion.model", "sion.vocab", "token_features.npz", "tokenizer_metadata.json"}
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


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

    config_module, monolingual_module, tokenizer_module = _load_production_modules(source_root)
    config = config_module.load_config(config_path)
    pairs = config.data.configured_language_pairs()
    languages = monolingual_module.foundation_languages(
        tokenizer_module.languages_from_pairs(pairs),
        config.data.configured_source_only_languages(),
    )
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
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source must be a regular non-symlink file: {relative}")
    return SourceRecord(relative, path.stat().st_size, _file_sha256(path))


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
    ratio = float(config.foundation.tokenizer_sample_ratio)
    if ratio != EXPECTED_TOKENIZER_SAMPLE_RATIO:
        raise RuntimeError(
            "refusing to train a non-production tokenizer ratio: "
            f"expected {EXPECTED_TOKENIZER_SAMPLE_RATIO}, got {ratio}"
        )
    return {
        "version": SOURCE_MANIFEST_VERSION,
        "git_commit": git_commit,
        "config_path": "sion_translate.yaml",
        "config_sha256": _file_sha256(config_path),
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
    if manifest.get("tokenizer_sample_ratio") != EXPECTED_TOKENIZER_SAMPLE_RATIO:
        raise ValueError("source manifest does not describe the production tokenizer ratio")
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
    if not isinstance(config_digest, str) or config_digest != _file_sha256(config_path):
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
    return paths, config


@contextmanager
def _observe_tokenizer_sentence_counts(tokenizer_module: Any) -> Generator[list[int]]:
    """Count both exhaustive production iterator passes without changing content."""

    production_iterator = tokenizer_module.iter_tokenizer_sentences
    observed: list[int] = []

    def counted_iterator(*args: Any, **kwargs: Any) -> Iterator[str]:
        count = 0
        for sentence in production_iterator(*args, **kwargs):
            count += 1
            yield sentence
        if count != EXPECTED_TOTAL_SENTENCES:
            raise RuntimeError(
                "production tokenizer iterator count differs from the measured input: "
                f"expected {EXPECTED_TOTAL_SENTENCES}, got {count}"
            )
        observed.append(count)

    tokenizer_module.iter_tokenizer_sentences = counted_iterator
    try:
        yield observed
    finally:
        tokenizer_module.iter_tokenizer_sentences = production_iterator


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
    observed_sentence_counts: Sequence[int],
    artifacts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    metadata_path = directory / "tokenizer_metadata.json"
    metadata_value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata_value, dict):
        raise RuntimeError("tokenizer metadata is not a JSON object")
    metadata = cast(dict[str, object], metadata_value)
    if metadata.get("sentencepiece_version") != EXPECTED_SENTENCEPIECE_VERSION:
        raise RuntimeError("tokenizer metadata records the wrong SentencePiece version")
    if metadata.get("monolingual_sample_ratio") != EXPECTED_TOKENIZER_SAMPLE_RATIO:
        raise RuntimeError("tokenizer metadata records the wrong monolingual ratio")
    if metadata.get("monolingual_sentences") != EXPECTED_MONOLINGUAL_SENTENCES:
        raise RuntimeError(
            "actual monolingual sample counts differ from the measured production counts: "
            f"{metadata.get('monolingual_sentences')}"
        )
    if list(observed_sentence_counts) != [EXPECTED_TOTAL_SENTENCES, EXPECTED_TOTAL_SENTENCES]:
        raise RuntimeError(
            "production tokenizer iterator counts differ between its two exhaustive passes: "
            f"{list(observed_sentence_counts)}"
        )
    actual_total = EXPECTED_PARALLEL_SENTENCES + sum(EXPECTED_MONOLINGUAL_SENTENCES.values())
    if actual_total != EXPECTED_TOTAL_SENTENCES:
        raise RuntimeError("configured production sentence-count assertions are inconsistent")
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
        f"ratio-040-{commit[:12]}-{_manifest_digest(source_manifest)[:12]}-"
        f"{timestamp}-{secrets.token_hex(3)}"
    )


def _train_remote(source_manifest: Mapping[str, object], run_id: str) -> dict[str, object]:
    remote_repository_root = Path(REMOTE_REPOSITORY_ROOT)
    input_mount = Path(INPUT_MOUNT)
    output_mount = Path(OUTPUT_MOUNT)
    config_path = remote_repository_root / "sion_translate.yaml"
    _, monolingual_module, tokenizer_module = _load_production_modules(remote_repository_root)
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
    if ratio != EXPECTED_TOKENIZER_SAMPLE_RATIO:
        raise RuntimeError(f"remote tokenizer ratio changed after verification: {ratio}")
    pairs = config.data.configured_language_pairs()
    foundation_languages = monolingual_module.foundation_languages(
        tokenizer_module.languages_from_pairs(pairs),
        config.data.configured_source_only_languages(),
    )
    discovery = monolingual_module.discover_monolingual_sources(
        input_mount / config.foundation.corpus_dir,
        foundation_languages,
    )
    cpu_plan = tokenizer_module.build_cpu_plan(input_files=len(paths))
    if (
        cpu_plan.preprocess_workers < 1
        or cpu_plan.sentencepiece_threads < 1
        or cpu_plan.preprocess_workers + cpu_plan.sentencepiece_threads != cpu_plan.available
    ):
        raise RuntimeError(f"invalid production CPU plan: {cpu_plan}")

    with tempfile.TemporaryDirectory(prefix="sion-tokenizer-") as workspace:
        build_directory = Path(workspace) / "tokenizer"
        with _observe_tokenizer_sentence_counts(tokenizer_module) as observed_counts:
            tokenizer_module.train_tokenizer(
                [str(input_mount / config.data.raw_dir / "*.jsonl")],
                build_directory,
                vocab_size=EXPECTED_VOCAB_SIZE,
                language_pairs=pairs,
                monolingual=discovery,
                monolingual_sample_ratio=ratio,
                approximate_split=config.data.approximate_split,
                source_only_languages=config.data.configured_source_only_languages(),
                train_only_prefixes=config.data.configured_synthetic_prefixes(),
                num_workers=cpu_plan.preprocess_workers,
                num_threads=cpu_plan.sentencepiece_threads,
            )
        artifacts = _artifact_records(build_directory)
        metadata = _validate_training_metadata(build_directory, observed_counts, artifacts)
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
                "parallel": EXPECTED_PARALLEL_SENTENCES,
                "monolingual": EXPECTED_MONOLINGUAL_SENTENCES,
                "total": EXPECTED_TOTAL_SENTENCES,
                "observed_passes": list(observed_counts),
            },
            "cpu_plan": {
                "available": cpu_plan.available,
                "preprocess_workers": cpu_plan.preprocess_workers,
                "sentencepiece_threads": cpu_plan.sentencepiece_threads,
            },
            "required_character_count": metadata.get("required_character_count"),
            "required_characters_sha256": metadata.get("required_characters_sha256"),
            "artifacts": artifacts,
        }
        manifest_path = build_directory / "training_manifest.json"
        manifest_path.write_text(
            json.dumps(training_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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


try:
    import modal as _modal
except ModuleNotFoundError:  # Unit tests do not require the optional Modal client.
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
    main()
