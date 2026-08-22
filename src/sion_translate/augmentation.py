"""Authenticated, direction-scoped backtranslation state and accounting."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
from typing import Protocol, cast

import numpy as np

from sion_translate.auto import stored_fingerprint
from sion_translate.config import DataConfig
from sion_translate.data.indexed import IndexedParallelDataset
from sion_translate.data.integrity import validate_dataset_artifact_inventory
from sion_translate.data.prepare import prepare_preprocessing_options
from sion_translate.data.quality import assess_pair, canonical_text
from sion_translate.fingerprint import DatasetFingerprint, file_sha256

AUGMENT_LEDGER_SCHEMA = "sion-augment-ledger-v1"
AUGMENT_ROW_SCHEMA = "sion-augment-row-v1"
AUGMENT_STATE_DIRECTORY = ".sion_augment"
_JOB_ID = re.compile(r"^[0-9a-f]{24}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TranslationBackend(Protocol):
    def translate(
        self,
        texts: Sequence[str],
        *,
        source_language: str,
        target_language: str,
        num_beams: int,
        max_new_tokens: int,
        batch_size: int,
    ) -> list[str]: ...


class _TextWriter(Protocol):
    def write(self, value: str, /) -> int: ...


@dataclass(frozen=True)
class FileSnapshot:
    filename: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return {"filename": self.filename, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: object) -> FileSnapshot:
        if not isinstance(value, dict):
            raise ValueError("augmentation input identity must be an object")
        fields = cast(dict[object, object], value)
        filename = fields.get("filename")
        size = fields.get("size")
        sha256 = fields.get("sha256")
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or Path(filename).name != filename
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(sha256, str)
            or _SHA256.fullmatch(sha256) is None
        ):
            raise ValueError("augmentation input identity is invalid")
        return cls(filename=filename, size=size, sha256=sha256)


@dataclass(frozen=True)
class DirectionCount:
    real: int = 0
    synthetic: int = 0


@dataclass(frozen=True)
class AugmentationIdentity:
    job_id: str
    pair: tuple[str, str]
    mono_language: str
    input: FileSnapshot
    model_identity: str
    generator_tokenizer_sha256: str
    generation_direction: tuple[str, str]
    training_direction: tuple[str, str]
    num_beams: int
    max_new_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "pair": list(self.pair),
            "mono_language": self.mono_language,
            "input": self.input.to_dict(),
            "model_identity": self.model_identity,
            "generator_tokenizer_sha256": self.generator_tokenizer_sha256,
            "generation_direction": list(self.generation_direction),
            "training_direction": list(self.training_direction),
            "num_beams": self.num_beams,
            "max_new_tokens": self.max_new_tokens,
        }

    @classmethod
    def from_dict(cls, value: object) -> AugmentationIdentity:
        if not isinstance(value, dict):
            raise ValueError("augmentation ledger identity must be an object")
        fields = cast(dict[object, object], value)

        def pair_field(name: str) -> tuple[str, str]:
            raw = fields.get(name)
            if not isinstance(raw, list):
                raise ValueError(f"augmentation ledger {name} must contain two languages")
            raw_values = cast(list[object], raw)
            if len(raw_values) != 2 or not all(
                isinstance(item, str) and item for item in raw_values
            ):
                raise ValueError(f"augmentation ledger {name} must contain two languages")
            values = cast(list[str], raw_values)
            return values[0], values[1]

        job_id = fields.get("job_id")
        mono_language = fields.get("mono_language")
        model_identity = fields.get("model_identity")
        tokenizer_sha = fields.get("generator_tokenizer_sha256")
        num_beams = fields.get("num_beams")
        max_new_tokens = fields.get("max_new_tokens")
        if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
            raise ValueError("augmentation ledger job_id is invalid")
        if not isinstance(mono_language, str) or not mono_language:
            raise ValueError("augmentation ledger mono_language is invalid")
        if not isinstance(model_identity, str) or _SHA256.fullmatch(model_identity) is None:
            raise ValueError("augmentation ledger model_identity is invalid")
        if not isinstance(tokenizer_sha, str) or _SHA256.fullmatch(tokenizer_sha) is None:
            raise ValueError("augmentation ledger generator tokenizer identity is invalid")
        if (
            not isinstance(num_beams, int)
            or isinstance(num_beams, bool)
            or num_beams < 1
            or not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 1
        ):
            raise ValueError("augmentation ledger generation parameters are invalid")
        pair = pair_field("pair")
        generation_direction = pair_field("generation_direction")
        training_direction = pair_field("training_direction")
        if (
            pair[0] == pair[1]
            or mono_language not in pair
            or frozenset(generation_direction) != frozenset(pair)
            or frozenset(training_direction) != frozenset(pair)
            or generation_direction != tuple(reversed(training_direction))
            or generation_direction[0] != mono_language
        ):
            raise ValueError("augmentation ledger direction contract is invalid")
        identity = cls(
            job_id=job_id,
            pair=pair,
            mono_language=mono_language,
            input=FileSnapshot.from_dict(fields.get("input")),
            model_identity=model_identity,
            generator_tokenizer_sha256=tokenizer_sha,
            generation_direction=generation_direction,
            training_direction=training_direction,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        if identity.job_id != augmentation_job_id(identity.pair, identity.input.filename):
            raise ValueError("augmentation ledger job_id does not match its pair/input")
        return identity


@dataclass(frozen=True)
class ShardSummary:
    name: str
    rows: int
    sha256: str
    first_input_line: int
    last_input_line: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "name": self.name,
            "rows": self.rows,
            "sha256": self.sha256,
            "first_input_line": self.first_input_line,
            "last_input_line": self.last_input_line,
        }

    @classmethod
    def from_dict(cls, value: object) -> ShardSummary:
        if not isinstance(value, dict):
            raise ValueError("augmentation shard summary must be an object")
        fields = cast(dict[object, object], value)
        name = fields.get("name")
        rows = fields.get("rows")
        sha256 = fields.get("sha256")
        first_line = fields.get("first_input_line")
        last_line = fields.get("last_input_line")
        integers = (rows, first_line, last_line)
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".jsonl")
            or not isinstance(sha256, str)
            or _SHA256.fullmatch(sha256) is None
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in integers)
        ):
            raise ValueError("augmentation shard summary is invalid")
        assert isinstance(rows, int) and isinstance(first_line, int) and isinstance(last_line, int)
        if rows < 1 or first_line < 0 or last_line < first_line:
            raise ValueError("augmentation shard summary ranges are invalid")
        return cls(name, rows, sha256, first_line, last_line)


@dataclass(frozen=True)
class JobProgress:
    identity: AugmentationIdentity
    cursor_line: int = 0
    eof: bool = False
    shards: tuple[ShardSummary, ...] = ()
    mono_text_hashes: frozenset[str] = frozenset()

    @property
    def accepted_rows(self) -> int:
        return sum(shard.rows for shard in self.shards)


@dataclass(frozen=True)
class AugmentationRegistry:
    jobs: Mapping[str, JobProgress]
    prepared_names: frozenset[str]

    def pending_direction_counts(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for progress in self.jobs.values():
            pending = sum(
                shard.rows for shard in progress.shards if shard.name not in self.prepared_names
            )
            direction = progress.identity.training_direction
            counts[direction] = counts.get(direction, 0) + pending
        return counts

    def mono_hashes_by_direction(self) -> dict[tuple[str, str], set[str]]:
        output: dict[tuple[str, str], set[str]] = {}
        for progress in self.jobs.values():
            output.setdefault(progress.identity.training_direction, set()).update(
                progress.mono_text_hashes
            )
        return output


@dataclass(frozen=True)
class JobRunResult:
    progress: JobProgress
    written: int
    quality_filtered: int
    duplicates: int
    too_long: int


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def snapshot_file(path: str | Path) -> FileSnapshot:
    """Hash one stable regular file, rejecting replacement during the read."""

    resolved = Path(path)
    before = resolved.stat()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = file_sha256(resolved)
    after = resolved.stat()
    identity_before = (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_dev,
        before.st_ino,
    )
    identity_after = (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_dev,
        after.st_ino,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"파일이 해시 계산 중 변경되었습니다: {resolved}")
    return FileSnapshot(resolved.name, after.st_size, digest)


def language_pair_slug(pair: Sequence[str]) -> str:
    canonical_pair = tuple(sorted(map(str, pair)))
    readable = "__".join(
        re.sub(r"[^A-Za-z0-9_.-]+", "_", language).strip("._-") or "language"
        for language in canonical_pair
    )
    digest = hashlib.sha256("\0".join(canonical_pair).encode("utf-8")).hexdigest()[:16]
    return f"{readable}__{digest}"


def augmentation_job_id(pair: Sequence[str], mono_filename: str) -> str:
    canonical_pair = tuple(sorted(map(str, pair)))
    payload = "\0".join((*canonical_pair, Path(mono_filename).name))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _safe_mono_stem(filename: str) -> str:
    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "mono"
    return safe[:48]


def augmentation_shard_path(
    data_dir: Path,
    synthetic_prefix: str,
    identity: AugmentationIdentity,
    sequence: int,
) -> Path:
    filename = (
        f"{synthetic_prefix}{language_pair_slug(identity.pair)}_"
        f"{_safe_mono_stem(identity.input.filename)}_{identity.job_id}_"
        f"{sequence:06d}.jsonl"
    )
    output = data_dir / filename
    if output.resolve().parent != data_dir.resolve():
        raise ValueError("augmentation output escaped the configured raw directory")
    return output


def build_job_identity(
    *,
    pair: tuple[str, str],
    mono_language: str,
    input_snapshot: FileSnapshot,
    model_identity: str,
    generator_tokenizer_sha256: str,
    num_beams: int,
    max_new_tokens: int,
) -> AugmentationIdentity:
    if mono_language not in pair:
        raise ValueError(f"monolingual language {mono_language!r} is outside pair {pair!r}")
    other_language = pair[0] if mono_language == pair[1] else pair[1]
    identity = AugmentationIdentity(
        job_id=augmentation_job_id(pair, input_snapshot.filename),
        pair=pair,
        mono_language=mono_language,
        input=input_snapshot,
        model_identity=model_identity,
        generator_tokenizer_sha256=generator_tokenizer_sha256,
        generation_direction=(mono_language, other_language),
        training_direction=(other_language, mono_language),
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )
    # Exercise the same strict parser used for persisted identities.
    return AugmentationIdentity.from_dict(identity.to_dict())


def _expected_preprocessing_options(data: DataConfig) -> dict[str, object]:
    pairs = data.configured_language_pairs()
    return prepare_preprocessing_options(
        approximate_split=data.approximate_split,
        source_only_languages=data.configured_source_only_languages(),
        translation_directions=data.configured_translation_directions(),
        train_only_prefixes=data.configured_synthetic_prefixes(),
        synthetic_sampling_weight=data.synthetic_sampling_weight,
        language_pair_count=len(pairs),
    )


def validate_prepared_raw_contract(
    data: DataConfig,
    *,
    augment_prefix: str,
) -> DatasetFingerprint:
    """Require an intact prepared baseline, allowing only managed new BT shards."""

    raw_dir = Path(data.raw_dir)
    dataset_dir = Path(data.dataset_dir)
    fingerprint = stored_fingerprint(dataset_dir)
    if not isinstance(fingerprint, DatasetFingerprint):
        raise RuntimeError(
            f"{dataset_dir}의 인증된 raw fingerprint가 없습니다. 먼저 sion-train을 실행하세요."
        )
    tokenizer_path = Path(data.tokenizer_model)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    if (
        fingerprint.language_pairs != data.configured_language_pairs()
        or fingerprint.tokenizer_sha256 != file_sha256(tokenizer_path)
        or fingerprint.preprocessing_options != _expected_preprocessing_options(data)
    ):
        raise RuntimeError(
            "준비된 데이터셋의 언어 그래프·토크나이저·전처리 계약이 현재 설정과 "
            "다릅니다. 먼저 sion-train으로 데이터셋을 다시 준비하세요."
        )
    if not raw_dir.is_dir():
        raise RuntimeError(f"준비된 데이터셋의 raw_dir가 사라졌습니다: {raw_dir}")

    prepared = {item.name: item for item in fingerprint.files}
    current = {path.name: path for path in raw_dir.glob("*.jsonl") if path.is_file()}
    missing = sorted(set(prepared) - set(current))
    if missing:
        raise RuntimeError(
            f"준비 뒤 raw 입력이 삭제되었습니다. 증강 전에 데이터셋을 다시 준비하세요: {missing}"
        )
    for name, expected in prepared.items():
        path = current[name]
        if path.stat().st_size != expected.size or file_sha256(path) != expected.sha256:
            raise RuntimeError(
                f"준비 뒤 raw 입력이 변경되었습니다. 증강 전에 데이터셋을 다시 준비하세요: {name}"
            )
    unexpected = sorted(
        name for name in set(current) - set(prepared) if not name.startswith(augment_prefix)
    )
    if unexpected:
        raise RuntimeError(
            "아직 준비되지 않은 일반/synthetic raw 입력이 있습니다. 증강 예산을 "
            f"계산하기 전에 sion-train을 실행하세요: {unexpected}"
        )
    return fingerprint


def count_prepared_direction_pairs(
    dataset_dir: str | Path,
    directions: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], DirectionCount]:
    """Count authenticated physical rows that supervise each requested edge."""

    root = Path(dataset_dir)
    validate_dataset_artifact_inventory(root)
    requested = tuple(dict.fromkeys(directions))
    counts = {direction: DirectionCount() for direction in requested}
    verified_graph = False
    # Synthetic rows are always train-only, so cap them against the real mass
    # that can actually be sampled by the same training direction.
    for split in ("train",):
        if not any((root / split).glob("*.idx.npy")):
            continue
        dataset = IndexedParallelDataset(root, split, verify_integrity=False)
        if not set(requested) <= set(dataset.translation_directions):
            raise ValueError("requested augmentation directions are absent from the dataset graph")
        verified_graph = True
        language_to_id = {language: index for index, language in enumerate(dataset.languages)}
        for index in dataset.indices:
            names = set(index.dtype.names or ())
            required = {
                "src_language_id",
                "tgt_language_id",
                "synthetic",
                "forward_only",
            }
            if not required <= names:
                raise ValueError("prepared dataset lacks authenticated direction/synthetic fields")
            sources = np.asarray(index["src_language_id"], dtype=np.int64)
            targets = np.asarray(index["tgt_language_id"], dtype=np.int64)
            synthetic = np.asarray(index["synthetic"], dtype=np.bool_)
            forward_only = np.asarray(index["forward_only"], dtype=np.bool_)
            for direction in requested:
                source_id = language_to_id[direction[0]]
                target_id = language_to_id[direction[1]]
                applies = ((sources == source_id) & (targets == target_id)) | (
                    (~forward_only) & (sources == target_id) & (targets == source_id)
                )
                previous = counts[direction]
                counts[direction] = DirectionCount(
                    real=previous.real + int(np.count_nonzero(applies & ~synthetic)),
                    synthetic=previous.synthetic + int(np.count_nonzero(applies & synthetic)),
                )
    if not verified_graph:
        raise ValueError(f"prepared dataset has no indexed splits: {root}")
    return counts


def _ledger_path(data_dir: Path, job_id: str) -> Path:
    return data_dir / AUGMENT_STATE_DIRECTORY / f"{job_id}.json"


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_job_progress(data_dir: Path, progress: JobProgress) -> None:
    payload = {
        "schema": AUGMENT_LEDGER_SCHEMA,
        "identity": progress.identity.to_dict(),
        "cursor_line": progress.cursor_line,
        "eof": progress.eof,
        "shards": [shard.to_dict() for shard in progress.shards],
    }
    _atomic_write_json(_ledger_path(data_dir, progress.identity.job_id), payload)


def _read_ledger(path: Path) -> tuple[AugmentationIdentity, int, bool, tuple[ShardSummary, ...]]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"augmentation ledger를 읽을 수 없습니다: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"augmentation ledger는 JSON object여야 합니다: {path}")
    values = cast(dict[object, object], raw)
    if values.get("schema") != AUGMENT_LEDGER_SCHEMA:
        raise ValueError(f"지원하지 않는 augmentation ledger schema입니다: {path}")
    identity = AugmentationIdentity.from_dict(values.get("identity"))
    cursor = values.get("cursor_line")
    eof = values.get("eof")
    raw_shards = values.get("shards")
    if (
        not isinstance(cursor, int)
        or isinstance(cursor, bool)
        or cursor < 0
        or not isinstance(eof, bool)
        or not isinstance(raw_shards, list)
    ):
        raise ValueError(f"augmentation ledger progress가 잘못되었습니다: {path}")
    shards = tuple(ShardSummary.from_dict(value) for value in cast(list[object], raw_shards))
    return identity, cursor, eof, shards


def _validate_shard(
    path: Path,
    identity: AugmentationIdentity,
    *,
    previous_input_line: int,
) -> tuple[ShardSummary, frozenset[str]]:
    rows = 0
    first_line: int | None = None
    last_line = previous_input_line
    mono_hashes: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    raise ValueError("empty line")
                raw_row: object = json.loads(raw_line)
                if not isinstance(raw_row, dict):
                    raise ValueError("row is not an object")
                row = cast(dict[object, object], raw_row)
                metadata = row.get("_sion_augment")
                if not isinstance(metadata, dict):
                    raise ValueError("row identity is missing")
                markers = cast(dict[object, object], metadata)
                input_line = markers.get("input_line")
                mono_hash = markers.get("mono_text_sha256")
                if (
                    markers.get("schema") != AUGMENT_ROW_SCHEMA
                    or markers.get("job_id") != identity.job_id
                    or markers.get("input_sha256") != identity.input.sha256
                    or not isinstance(input_line, int)
                    or isinstance(input_line, bool)
                    or input_line <= last_line
                    or not isinstance(mono_hash, str)
                    or _SHA256.fullmatch(mono_hash) is None
                    or mono_hash in mono_hashes
                    or row.get("synthetic") is not True
                    or row.get("training_direction") != list(identity.training_direction)
                    or not isinstance(row.get(identity.pair[0]), str)
                    or not isinstance(row.get(identity.pair[1]), str)
                    or _sha256_text(canonical_text(cast(str, row[identity.mono_language])))
                    != mono_hash
                ):
                    raise ValueError("row contract is invalid")
                first_line = input_line if first_line is None else first_line
                last_line = input_line
                mono_hashes.add(mono_hash)
                rows += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"손상되거나 소유권이 불명확한 augmentation shard입니다: {path}"
        ) from error
    if rows < 1 or first_line is None:
        raise ValueError(f"빈 augmentation shard는 완료 산출물이 아닙니다: {path}")
    return (
        ShardSummary(path.name, rows, file_sha256(path), first_line, last_line),
        frozenset(mono_hashes),
    )


def _shard_pattern(prefix: str, identity: AugmentationIdentity) -> re.Pattern[str]:
    base = augmentation_shard_path(Path("."), prefix, identity, 0).name
    stem = base[: -len("000000.jsonl")]
    return re.compile(rf"^{re.escape(stem)}([0-9]{{6}})\.jsonl$")


def _load_job_progress(
    data_dir: Path,
    prefix: str,
    ledger_path: Path,
) -> JobProgress:
    identity, cursor, eof, declared = _read_ledger(ledger_path)
    if ledger_path.stem != identity.job_id:
        raise ValueError(f"augmentation ledger filename과 job_id가 다릅니다: {ledger_path}")
    pattern = _shard_pattern(prefix, identity)
    discovered: list[tuple[int, Path]] = []
    for path in data_dir.glob(f"{prefix}*.jsonl"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            discovered.append((int(match.group(1)), path))
    discovered.sort()
    if [sequence for sequence, _ in discovered] != list(range(len(discovered))):
        raise ValueError(f"augmentation shard sequence에 빈 구간이 있습니다: {identity.job_id}")

    summaries: list[ShardSummary] = []
    mono_hashes: set[str] = set()
    previous_input_line = -1
    for _, path in discovered:
        summary, shard_hashes = _validate_shard(
            path,
            identity,
            previous_input_line=previous_input_line,
        )
        summaries.append(summary)
        mono_hashes.update(shard_hashes)
        previous_input_line = summary.last_input_line
    if len(declared) > len(summaries) or tuple(summaries[: len(declared)]) != declared:
        raise ValueError(f"augmentation ledger와 shard inventory가 다릅니다: {identity.job_id}")
    recovered = len(summaries) > len(declared)
    minimum_cursor = previous_input_line + 1
    if cursor < (declared[-1].last_input_line + 1 if declared else 0):
        raise ValueError(f"augmentation ledger cursor가 shard보다 뒤처졌습니다: {identity.job_id}")
    return JobProgress(
        identity=identity,
        cursor_line=max(cursor, minimum_cursor),
        eof=False if recovered else eof,
        shards=tuple(summaries),
        mono_text_hashes=frozenset(mono_hashes),
    )


def load_augmentation_registry(
    data_dir: Path,
    synthetic_prefix: str,
    prepared_names: Sequence[str],
) -> AugmentationRegistry:
    state_dir = data_dir / AUGMENT_STATE_DIRECTORY
    # Private temp files are never training inputs and can only be remnants of
    # a process that died before atomic publication.
    for partial in data_dir.glob(f".{synthetic_prefix}*.partial"):
        if partial.is_file():
            partial.unlink()
    if state_dir.is_dir():
        for partial in state_dir.glob(".*.partial"):
            if partial.is_file():
                partial.unlink()
    jobs: dict[str, JobProgress] = {}
    if state_dir.is_dir():
        for ledger_path in sorted(state_dir.glob("*.json")):
            progress = _load_job_progress(data_dir, synthetic_prefix, ledger_path)
            if progress.identity.job_id in jobs:
                raise ValueError(f"duplicate augmentation job_id: {progress.identity.job_id}")
            jobs[progress.identity.job_id] = progress
    owned = {shard.name for progress in jobs.values() for shard in progress.shards}
    actual = {path.name for path in data_dir.glob(f"{synthetic_prefix}*.jsonl") if path.is_file()}
    if actual != owned:
        raise ValueError(
            "ledger가 인증하지 않은 legacy/corrupt augmentation output이 있습니다: "
            f"unowned={sorted(actual - owned)}, missing={sorted(owned - actual)}"
        )
    return AugmentationRegistry(jobs=jobs, prepared_names=frozenset(prepared_names))


def _publish_shard(
    temporary_path: Path,
    output_path: Path,
    *,
    rows: int,
    first_input_line: int,
    last_input_line: int,
) -> ShardSummary:
    if output_path.exists():
        raise FileExistsError(f"augmentation shard already exists: {output_path}")
    os.replace(temporary_path, output_path)
    return ShardSummary(
        output_path.name,
        rows,
        file_sha256(output_path),
        first_input_line,
        last_input_line,
    )


def run_augmentation_job(
    translator: TranslationBackend,
    *,
    mono_path: Path,
    data_dir: Path,
    synthetic_prefix: str,
    progress: JobProgress,
    accepted_budget: int,
    batch_size: int,
    seen_mono_hashes: set[str],
    source_fits: Callable[[str], bool] | None = None,
) -> JobRunResult:
    """Stream one resumable job and publish at most one immutable shard."""

    if accepted_budget < 1 or batch_size < 1:
        raise ValueError("accepted_budget and batch_size must be positive")
    identity = progress.identity
    if snapshot_file(mono_path) != identity.input:
        raise RuntimeError(f"단일어 입력이 augmentation ledger 생성 뒤 변경되었습니다: {mono_path}")
    # Establish ownership before a shard can be published. If the process dies
    # after rename but before the final ledger update, registry loading can
    # validate and recover that immutable orphan shard.
    write_job_progress(data_dir, progress)
    output_path = augmentation_shard_path(
        data_dir,
        synthetic_prefix,
        identity,
        len(progress.shards),
    )

    def always_fits(_text: str) -> bool:
        return True

    fit: Callable[[str], bool] = source_fits or always_fits
    cursor = progress.cursor_line
    reached_eof = True
    written = 0
    quality_filtered = 0
    duplicates = 0
    too_long = 0
    first_written_line: int | None = None
    last_written_line: int | None = None
    run_hashes: set[str] = set()
    temporary_path: Path | None = None

    def translate_batch(
        batch: list[tuple[int, str, str]],
        out: _TextWriter,
    ) -> tuple[int, bool]:
        nonlocal cursor, written, quality_filtered, first_written_line, last_written_line
        translations = translator.translate(
            [text for _, text, _ in batch],
            source_language=identity.generation_direction[0],
            target_language=identity.generation_direction[1],
            num_beams=identity.num_beams,
            max_new_tokens=identity.max_new_tokens,
            batch_size=batch_size,
        )
        processed = 0
        for (line_number, mono_text, mono_hash), raw_translation in zip(
            batch, translations, strict=True
        ):
            synthetic_text = canonical_text(raw_translation)
            row: dict[str, object] = {
                identity.pair[0]: (
                    mono_text if identity.mono_language == identity.pair[0] else synthetic_text
                ),
                identity.pair[1]: (
                    mono_text if identity.mono_language == identity.pair[1] else synthetic_text
                ),
                "synthetic": True,
                "training_direction": list(identity.training_direction),
                "_sion_augment": {
                    "schema": AUGMENT_ROW_SCHEMA,
                    "job_id": identity.job_id,
                    "input_sha256": identity.input.sha256,
                    "input_line": line_number,
                    "mono_text_sha256": mono_hash,
                },
            }
            assessment = assess_pair(
                cast(str, row[identity.pair[0]]),
                cast(str, row[identity.pair[1]]),
                languages=identity.pair,
            )
            cursor = line_number + 1
            processed += 1
            run_hashes.add(mono_hash)
            if not assessment.accepted:
                quality_filtered += 1
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            first_written_line = line_number if first_written_line is None else first_written_line
            last_written_line = line_number
            if written >= accepted_budget:
                return processed, True
        return processed, False

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=data_dir,
            prefix=f".{output_path.name}.",
            suffix=".partial",
            delete=False,
        ) as out:
            temporary_path = Path(out.name)
            batch: list[tuple[int, str, str]] = []
            queued_hashes: set[str] = set()
            total_lines = 0
            with mono_path.open("r", encoding="utf-8") as source:
                for line_number, raw_line in enumerate(source):
                    total_lines = line_number + 1
                    if line_number < progress.cursor_line:
                        continue
                    mono_text = canonical_text(raw_line)
                    mono_hash = _sha256_text(mono_text)
                    if not mono_text or mono_hash in seen_mono_hashes or mono_hash in queued_hashes:
                        duplicates += bool(mono_text)
                        if not batch:
                            cursor = line_number + 1
                        continue
                    if not fit(mono_text):
                        too_long += 1
                        if not batch:
                            cursor = line_number + 1
                        continue
                    batch.append((line_number, mono_text, mono_hash))
                    queued_hashes.add(mono_hash)
                    if len(batch) < batch_size:
                        continue
                    processed, stop = translate_batch(batch, out)
                    for _, _, processed_hash in batch[:processed]:
                        seen_mono_hashes.add(processed_hash)
                    batch = batch[processed:]
                    queued_hashes = {item[2] for item in batch}
                    if stop:
                        reached_eof = False
                        break
                else:
                    if batch:
                        processed, stop = translate_batch(batch, out)
                        for _, _, processed_hash in batch[:processed]:
                            seen_mono_hashes.add(processed_hash)
                        if stop and processed < len(batch):
                            reached_eof = False
                        else:
                            cursor = total_lines
                    else:
                        cursor = total_lines
            out.flush()
            os.fsync(out.fileno())

        if snapshot_file(mono_path) != identity.input:
            raise RuntimeError(f"번역 중 단일어 입력이 변경되었습니다: {mono_path}")
        shards = progress.shards
        if written:
            assert temporary_path is not None
            assert first_written_line is not None and last_written_line is not None
            shard = _publish_shard(
                temporary_path,
                output_path,
                rows=written,
                first_input_line=first_written_line,
                last_input_line=last_written_line,
            )
            temporary_path = None
            shards = (*shards, shard)
        updated = JobProgress(
            identity=identity,
            cursor_line=cursor,
            eof=reached_eof,
            shards=shards,
            mono_text_hashes=frozenset((*progress.mono_text_hashes, *run_hashes)),
        )
        write_job_progress(data_dir, updated)
        return JobRunResult(updated, written, quality_filtered, duplicates, too_long)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def reconcile_job_identity(
    registry: AugmentationRegistry,
    identity: AugmentationIdentity,
) -> JobProgress:
    existing = registry.jobs.get(identity.job_id)
    if existing is None:
        return JobProgress(identity=identity)
    if existing.identity != identity:
        raise RuntimeError(
            "기존 augmentation job의 입력·모델·토크나이저·생성 설정이 현재와 "
            f"다릅니다: {identity.input.filename}. 기존 shard를 임의 재사용하지 않습니다."
        )
    return existing


def synthetic_budget(real_pairs: int, existing_synthetic: int, max_ratio: float) -> int:
    return max(0, int(real_pairs * max_ratio) - existing_synthetic)


__all__ = [
    "AUGMENT_LEDGER_SCHEMA",
    "AUGMENT_ROW_SCHEMA",
    "AugmentationIdentity",
    "AugmentationRegistry",
    "DirectionCount",
    "FileSnapshot",
    "JobProgress",
    "JobRunResult",
    "augmentation_job_id",
    "augmentation_shard_path",
    "build_job_identity",
    "count_prepared_direction_pairs",
    "language_pair_slug",
    "load_augmentation_registry",
    "reconcile_job_identity",
    "run_augmentation_job",
    "snapshot_file",
    "synthetic_budget",
    "validate_prepared_raw_contract",
    "write_job_progress",
]
