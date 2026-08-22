"""Resumable, auditable translation of monolingual JSONL queues.

Queue files are immutable inputs.  Each source row receives a result record,
while only rows that pass forward and round-trip checks are copied into
separate ``bt_*`` training shards.  Progress is committed after atomic shard
writes so a stopped multi-day run can safely resume from its byte offset.
Accepted shards are consumable only when their manifest part has
``published: true``; a two-phase pending publish prevents partial training data.
"""

# Queue manifests are persisted JSON objects and SacreBLEU exposes incomplete
# annotations. Validate their shape at runtime and contain Unknown types here.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast

from sacrebleu.metrics.chrf import CHRF

from sion_translate.data.quality import (
    assess_pair,
    canonical_text,
    japanese_kana_count,
    language_fraction,
)
from sion_translate.evaluation import multiset_f1, numeric_tokens
from sion_translate.scripts_registry import primary_language
from sion_translate.structured import structured_similarity


MANIFEST_SCHEMA = "sion-translation-queue-v1"
RESULT_SCHEMA = "sion-translation-result-v1"
ACCEPTED_OWNER_SCHEMA = "sion-accepted-namespace-owner-v1"
PIPELINE_VERSION = 1
SIGNATURE_VERSION = 2
RUN_LOCK_FILENAME = ".queue-translation.lock"
ACCEPTED_LOCK_FILENAME = RUN_LOCK_FILENAME
_CHRF = CHRF(word_order=0)


class TranslatorLike(Protocol):
    """The subset of :class:`Translator` used by the queue runner."""

    tokenizer: Any

    def translate(
        self,
        texts: Sequence[str],
        *,
        source_language: str,
        target_language: str,
        num_beams: int,
        max_new_tokens: int,
        batch_size: int,
        max_output_length_ratio: float,
        max_output_length_margin: int,
    ) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class QueueTranslationOptions:
    batch_size: int = 16
    shard_size: int = 1_000
    num_beams: int = 1
    max_new_tokens: int = 128
    max_output_length_ratio: float = 2.0
    max_output_length_margin: int = 12
    roundtrip_enabled: bool = True
    roundtrip_num_beams: int = 1
    roundtrip_max_new_tokens: int = 128
    roundtrip_max_output_length_ratio: float = 2.0
    roundtrip_max_output_length_margin: int = 12
    min_roundtrip_score: float = 0.65
    min_pair_score: int = 80
    min_target_language_fraction: float = 0.50
    min_japanese_kana_chars: int = 1
    min_structured_similarity: float = 1.0

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if self.num_beams <= 0 or self.roundtrip_num_beams <= 0:
            raise ValueError("beam counts must be positive")
        if self.max_new_tokens <= 0 or self.roundtrip_max_new_tokens <= 0:
            raise ValueError("generation limits must be positive")
        if self.max_output_length_ratio <= 0 or self.roundtrip_max_output_length_ratio <= 0:
            raise ValueError("output length ratios must be positive")
        if self.max_output_length_margin < 0 or self.roundtrip_max_output_length_margin < 0:
            raise ValueError("output length margins must be non-negative")
        if not 0.0 <= self.min_roundtrip_score <= 1.0:
            raise ValueError("min_roundtrip_score must be in [0, 1]")
        if not 0 <= self.min_pair_score <= 100:
            raise ValueError("min_pair_score must be in [0, 100]")
        if not 0.0 <= self.min_target_language_fraction <= 1.0:
            raise ValueError("min_target_language_fraction must be in [0, 1]")
        if self.min_japanese_kana_chars < 0:
            raise ValueError("min_japanese_kana_chars must be non-negative")
        if not 0.0 <= self.min_structured_similarity <= 1.0:
            raise ValueError("min_structured_similarity must be in [0, 1]")


def sha256_file(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _signature_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude volatile file metadata while retaining content identity."""

    source = configuration.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("queue configuration has no source identity")
    stable_source = {key: source.get(key) for key in ("path", "size", "sha256")}
    return {
        **dict(configuration),
        "source": stable_source,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _acquire_file_lock(handle: BinaryIO) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_file_lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager  # pyright: ignore[reportDeprecated]
def _directory_run_lock(
    directory: Path,
    *,
    lock_filename: str,
    label: str,
) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / lock_filename
    with lock_path.open("a+b") as handle:
        try:
            _acquire_file_lock(handle)
        except OSError as exc:
            raise RuntimeError(
                f"{label} is already being translated: {directory.resolve()}"
            ) from exc
        try:
            yield
        finally:
            _release_file_lock(handle)


@contextmanager  # pyright: ignore[reportDeprecated]
def _queue_run_lock(output_dir: Path) -> Iterator[None]:
    """Hold an OS-released single-writer lock for one queue output directory."""

    with _directory_run_lock(
        output_dir,
        lock_filename=RUN_LOCK_FILENAME,
        label="queue output",
    ):
        yield


@contextmanager  # pyright: ignore[reportDeprecated]
def _accepted_run_lock(accepted_dir: Path) -> Iterator[None]:
    """Serialize publishers sharing an accepted-shard namespace."""

    with _directory_run_lock(
        accepted_dir,
        lock_filename=ACCEPTED_LOCK_FILENAME,
        label="accepted queue namespace",
    ):
        yield


def _pending_accepted_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.pending")


def _publish_no_replace(pending_path: Path, accepted_path: Path) -> None:
    """Atomically publish without overwriting a target created by another host."""

    try:
        os.link(pending_path, accepted_path)
    except FileExistsError:
        raise
    except OSError as exc:
        raise OSError(
            f"filesystem does not support atomic no-clobber publication: {accepted_path}"
        ) from exc
    pending_path.unlink()


def _claim_accepted_namespace(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    accepted_dir: Path,
    input_stem: str,
) -> None:
    """Persistently bind one accepted run ID to its owning output manifest."""

    accepted_dir.mkdir(parents=True, exist_ok=True)
    owner_path = accepted_dir / f".{input_stem}_{manifest['run_id']}.owner.json"
    owner = {
        "schema": ACCEPTED_OWNER_SCHEMA,
        "run_id": manifest["run_id"],
        "output_dir": str(output_dir.resolve()),
        "manifest": str((output_dir / "manifest.json").resolve()),
    }
    if owner_path.is_file():
        try:
            existing = json.loads(owner_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid accepted namespace owner: {owner_path}") from exc
        if existing != owner:
            raise FileExistsError(
                f"accepted queue namespace is already owned by another output: {owner_path}"
            )
        return
    temporary = owner_path.with_name(f".{owner_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(owner, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        _publish_no_replace(temporary, owner_path)
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        try:
            existing = json.loads(owner_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as read_error:
            raise ValueError(f"invalid accepted namespace owner: {owner_path}") from read_error
        if existing != owner:
            raise FileExistsError(
                f"accepted queue namespace is already owned by another output: {owner_path}"
            ) from exc


def _jsonl_artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    final_byte = b""
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
            rows += block.count(b"\n")
            final_byte = block[-1:]
    size = path.stat().st_size
    if size and final_byte != b"\n":
        rows += 1
    return {
        "path": str(path.resolve()),
        "size": size,
        "rows": rows,
        "sha256": digest.hexdigest(),
    }


def _artifact_content_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    observed = _jsonl_artifact(path)
    return all(observed[field] == expected.get(field) for field in ("size", "rows", "sha256"))


def _recover_pending_accepted_parts(
    manifest: dict[str, Any],
    *,
    accepted_dir: Path,
    input_stem: str,
) -> bool:
    """Publish a committed accepted shard if a crash preceded its final rename."""

    next_part = int(manifest["progress"]["next_part"])
    parts = manifest.get("parts")
    changed = False
    for part_index in range(next_part):
        accepted_path = (
            accepted_dir / f"bt_{input_stem}_{manifest['run_id']}_{part_index:06d}.jsonl"
        )
        pending_path = _pending_accepted_path(accepted_path)
        if isinstance(parts, list) and part_index < len(parts):
            part = parts[part_index]
            expected = part.get("accepted") if isinstance(part, Mapping) else None
            if not isinstance(part, dict) or not isinstance(expected, Mapping):
                raise ValueError(f"accepted part {part_index:06d} has no valid manifest metadata")
            if part.get("published") is True and not pending_path.exists():
                continue
            final_matches = _artifact_content_matches(accepted_path, expected)
            pending_matches = _artifact_content_matches(pending_path, expected)
            if final_matches:
                if pending_path.exists():
                    pending_path.unlink()
                if part.get("published") is not True:
                    part["published"] = True
                    changed = True
                continue
            if pending_matches:
                try:
                    _publish_no_replace(pending_path, accepted_path)
                except FileExistsError as exc:
                    if _artifact_content_matches(accepted_path, expected):
                        pending_path.unlink()
                    else:
                        raise ValueError(
                            f"accepted part {part_index:06d} collided during recovery"
                        ) from exc
                if part.get("published") is not True:
                    part["published"] = True
                    changed = True
                continue
            if not accepted_path.exists() and not pending_path.exists():
                raise FileNotFoundError(
                    f"accepted part {part_index:06d} and its pending recovery are missing"
                )
            raise ValueError(f"accepted part {part_index:06d} does not match its manifest")
        if pending_path.is_file() and not accepted_path.exists():
            _publish_no_replace(pending_path, accepted_path)
        elif pending_path.exists():
            raise ValueError(f"ambiguous legacy accepted part recovery for {part_index:06d}")
    return changed


def _validate_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_path: Path,
    label: str,
) -> dict[str, Any]:
    recorded_path = Path(str(artifact.get("path", "")))
    if recorded_path.resolve() != expected_path.resolve():
        raise ValueError(f"{label} path does not match its queue manifest")
    if not expected_path.is_file():
        raise FileNotFoundError(f"{label} is missing: {expected_path}")
    observed = _jsonl_artifact(expected_path)
    for field in ("size", "rows", "sha256"):
        if artifact.get(field) != observed[field]:
            raise ValueError(
                f"{label} integrity mismatch for {field}: "
                f"expected {artifact.get(field)!r}, observed {observed[field]!r}"
            )
    return observed


def _part_semantics(
    result_path: Path,
    accepted_path: Path,
    *,
    source_start_index: int,
    run_id: str,
) -> tuple[dict[str, int], int]:
    """Validate row continuity and the exact accepted subset for legacy shards."""

    status_counts: Counter[str] = Counter()
    accepted_result_ids: list[str] = []
    generated_rows = 0
    with result_path.open("r", encoding="utf-8") as handle:
        for offset, line in enumerate(handle):
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {result_path}:{offset + 1}") from exc
            if not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA:
                raise ValueError(f"invalid result record in {result_path}:{offset + 1}")
            if result.get("source_index") != source_start_index + offset:
                raise ValueError(f"non-contiguous source_index in {result_path}:{offset + 1}")
            if result.get("run_id") != run_id:
                raise ValueError(f"run_id mismatch in {result_path}:{offset + 1}")
            status = result.get("status")
            if status not in {"accepted", "rejected", "error", "skipped_existing"}:
                raise ValueError(f"invalid result status in {result_path}:{offset + 1}")
            status_counts[status] += 1
            if result.get("translation") is not None and status != "skipped_existing":
                generated_rows += 1
            if status == "accepted":
                row_id = result.get("id")
                if not isinstance(row_id, str) or not row_id:
                    raise ValueError(f"accepted result has no id in {result_path}:{offset + 1}")
                accepted_result_ids.append(row_id)

    accepted_ids: list[str] = []
    with accepted_path.open("r", encoding="utf-8") as handle:
        for offset, line in enumerate(handle):
            try:
                accepted = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {accepted_path}:{offset + 1}") from exc
            if not isinstance(accepted, dict) or accepted.get("synthetic") is not True:
                raise ValueError(f"invalid accepted record in {accepted_path}:{offset + 1}")
            row_id = accepted.get("id")
            provenance = accepted.get("provenance")
            if (
                not isinstance(row_id, str)
                or not isinstance(provenance, dict)
                or provenance.get("queue_id") != row_id
                or provenance.get("run_id") != run_id
            ):
                raise ValueError(f"accepted provenance mismatch in {accepted_path}:{offset + 1}")
            accepted_ids.append(row_id)
    if accepted_ids != accepted_result_ids:
        raise ValueError("accepted shard is not the exact accepted subset of its result shard")
    return (
        {
            status: status_counts[status]
            for status in ("accepted", "rejected", "error", "skipped_existing")
        },
        generated_rows,
    )


def _translate_resilient(
    translator: TranslatorLike,
    texts: Sequence[str],
    *,
    source_language: str,
    target_language: str,
    num_beams: int,
    max_new_tokens: int,
    batch_size: int,
    max_output_length_ratio: float,
    max_output_length_margin: int,
) -> tuple[list[str | None], list[str | None]]:
    """Translate a batch and bisect failures so one row cannot stop a long run."""

    outputs: list[str | None] = [None] * len(texts)
    errors: list[str | None] = [None] * len(texts)

    def run(indices: list[int]) -> None:
        if not indices:
            return
        selected = [texts[index] for index in indices]
        try:
            translated = translator.translate(
                selected,
                source_language=source_language,
                target_language=target_language,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                batch_size=min(batch_size, len(selected)),
                max_output_length_ratio=max_output_length_ratio,
                max_output_length_margin=max_output_length_margin,
            )
            if len(translated) != len(selected):
                raise RuntimeError(
                    f"translator returned {len(translated)} rows for {len(selected)} inputs"
                )
        except Exception as exc:
            if len(indices) > 1:
                midpoint = len(indices) // 2
                run(indices[:midpoint])
                run(indices[midpoint:])
                return
            errors[indices[0]] = f"{type(exc).__name__}: {exc}"
            return
        for index, translation in zip(indices, translated, strict=True):
            outputs[index] = canonical_text(str(translation))

    ordered = sorted(
        range(len(texts)),
        key=lambda index: len(_content_tokens(translator, texts[index])),
    )
    for start in range(0, len(ordered), batch_size):
        run(ordered[start : start + batch_size])
    return outputs, errors


def _content_tokens(translator: TranslatorLike, text: str) -> list[object]:
    tokenizer = getattr(translator, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        typed_encode = cast(Callable[[str], Sequence[object]], encode)
        return list(typed_encode(text))
    return [character for character in canonical_text(text) if not character.isspace()]


def roundtrip_quality(
    source: str,
    roundtrip: str,
    *,
    tokenize: Callable[[str], Sequence[object]],
) -> dict[str, float | bool]:
    """Score whether a forward candidate can reconstruct its source."""

    cycle_chrf = _CHRF.sentence_score(roundtrip, [source]).score / 100.0
    token_f1 = multiset_f1(tokenize(source), tokenize(roundtrip))
    number_f1 = multiset_f1(numeric_tokens(source), numeric_tokens(roundtrip))
    structured_score, critical_mismatch = structured_similarity(source, roundtrip)
    source_length = sum(not character.isspace() for character in source)
    roundtrip_length = sum(not character.isspace() for character in roundtrip)
    length_score = math.exp(-abs(math.log((roundtrip_length + 1.0) / (source_length + 1.0))))
    score = (
        0.50 * cycle_chrf
        + 0.20 * token_f1
        + 0.15 * number_f1
        + 0.10 * structured_score
        + 0.05 * length_score
    )
    if critical_mismatch:
        score *= 0.5
    return {
        "score": score,
        "chrf": cycle_chrf,
        "token_f1": token_f1,
        "number_f1": number_f1,
        "structured": structured_score,
        "length": length_score,
        "critical_structured_mismatch": critical_mismatch,
    }


def _error_result(source_index: int, reason: str, *, raw_id: object = None) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "source_index": source_index,
        "id": raw_id if isinstance(raw_id, str) else None,
        "status": "error",
        "translation": None,
        "error": reason,
    }


def _parse_queue_line(raw: bytes, source_index: int) -> tuple[dict[str, Any], bool]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _error_result(source_index, f"invalid_utf8: {exc}"), False
    try:
        row = json.loads(text)
    except json.JSONDecodeError as exc:
        return _error_result(source_index, f"invalid_json: {exc}"), False
    if not isinstance(row, dict):
        return _error_result(source_index, "invalid_record: expected object"), False
    row_id = row.get("id")
    source_language = row.get("source_lang")
    target_language = row.get("target_lang")
    source = row.get("source")
    if not isinstance(row_id, str) or not row_id:
        return _error_result(source_index, "invalid_record: missing id"), False
    if (
        not isinstance(source_language, str)
        or not isinstance(target_language, str)
        or source_language == target_language
    ):
        return _error_result(
            source_index,
            "invalid_record: invalid source_lang/target_lang",
            raw_id=row_id,
        ), False
    if not isinstance(source, str) or not canonical_text(source):
        return _error_result(
            source_index,
            "invalid_record: missing source",
            raw_id=row_id,
        ), False
    result = {
        "schema": RESULT_SCHEMA,
        "source_index": source_index,
        "id": row_id,
        "source_lang": source_language,
        "target_lang": target_language,
        "source": canonical_text(source),
        "translation": row.get("translation"),
        "status": row.get("status", "pending"),
    }
    if result["translation"] is not None or result["status"] != "pending":
        result["status"] = "skipped_existing"
        return result, False
    return result, True


def _forward_quality(
    source: str,
    translation: str,
    *,
    source_language: str,
    target_language: str,
) -> tuple[dict[str, Any], list[str]]:
    assessment = assess_pair(
        source,
        translation,
        languages=(source_language, target_language),
    )
    number_exact = Counter(numeric_tokens(source)) == Counter(numeric_tokens(translation))
    structured_score, critical_mismatch = structured_similarity(source, translation)
    target_fraction = language_fraction(translation, target_language)
    quality = {
        "pair_score": assessment.score,
        "pair_warnings": list(assessment.warning_reasons),
        "number_exact": number_exact,
        "structured": structured_score,
        "critical_structured_mismatch": critical_mismatch,
        "target_language_fraction": target_fraction,
        "target_japanese_kana_chars": (
            japanese_kana_count(translation) if primary_language(target_language) == "ja" else None
        ),
    }
    reasons = list(assessment.rejection_reasons)
    return quality, reasons


def _source_identity(
    path: Path,
    previous: Mapping[str, Any] | None,
    *,
    force_hash: bool = False,
) -> dict[str, Any]:
    stat = path.stat()
    resolved = str(path.resolve())
    if (
        not force_hash
        and previous
        and previous.get("path") == resolved
        and previous.get("size") == stat.st_size
        and previous.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(previous.get("sha256"), str)
    ):
        digest = str(previous["sha256"])
    else:
        digest = sha256_file(path)
    return {
        "path": resolved,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def _new_manifest(
    *,
    source: Mapping[str, Any],
    options: QueueTranslationOptions,
    run_metadata: Mapping[str, Any],
    accepted_dir: Path,
    teacher_pilot_rows: int | None,
) -> dict[str, Any]:
    configuration = {
        "pipeline_version": PIPELINE_VERSION,
        "source": dict(source),
        "options": asdict(options),
        "run_metadata": dict(run_metadata),
        "accepted_dir": str(accepted_dir.resolve()),
    }
    signature = _stable_digest(_signature_configuration(configuration))
    return {
        "schema": MANIFEST_SCHEMA,
        "signature_version": SIGNATURE_VERSION,
        "run_id": signature[:16],
        "run_signature": signature,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "progress": {
            "completed_rows": 0,
            "source_byte_offset": 0,
            "next_part": 0,
            "complete": False,
        },
        "stats": {
            "processed": 0,
            "generated": 0,
            "accepted": 0,
            "rejected": 0,
            "errors": 0,
            "skipped_existing": 0,
        },
        "teacher_review": (
            {
                "pilot_rows": teacher_pilot_rows,
                "review_required": False,
                "approved": False,
                "approved_at": None,
                "approved_by": None,
            }
            if teacher_pilot_rows is not None
            else None
        ),
        "parts": [],
    }


def _validate_or_register_parts(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    accepted_dir: Path,
    input_stem: str,
) -> bool:
    """Verify committed shards, registering legacy v1 shards on first resume."""

    progress = manifest["progress"]
    next_part = int(progress["next_part"])
    parts = manifest.get("parts")
    changed = False
    if parts is None:
        parts = []
        manifest["parts"] = parts
        changed = True
        source_start = 0
        for part_index in range(next_part):
            result_path = output_dir / f"part-{part_index:06d}.jsonl"
            accepted_path = (
                accepted_dir / f"bt_{input_stem}_{manifest['run_id']}_{part_index:06d}.jsonl"
            )
            if not result_path.is_file():
                raise FileNotFoundError(f"legacy result shard is missing: {result_path}")
            if not accepted_path.is_file():
                raise FileNotFoundError(f"legacy accepted shard is missing: {accepted_path}")
            result_artifact = _jsonl_artifact(result_path)
            accepted_artifact = _jsonl_artifact(accepted_path)
            status_counts, generated_rows = _part_semantics(
                result_path,
                accepted_path,
                source_start_index=source_start,
                run_id=str(manifest["run_id"]),
            )
            parts.append(
                {
                    "part": part_index,
                    "source_start_index": source_start,
                    "source_rows": result_artifact["rows"],
                    "result": result_artifact,
                    "accepted": accepted_artifact,
                    "status_counts": status_counts,
                    "generated_rows": generated_rows,
                    "published": True,
                }
            )
            source_start += int(result_artifact["rows"])
    if not isinstance(parts, list) or len(parts) != next_part:
        raise ValueError("queue manifest part count does not match progress.next_part")

    total_result_rows = 0
    total_accepted_rows = 0
    total_generated_rows = 0
    total_status_counts: Counter[str] = Counter()
    expected_source_start = 0
    for part_index, part in enumerate(parts):
        if not isinstance(part, dict) or int(part.get("part", -1)) != part_index:
            raise ValueError("queue manifest contains an invalid or out-of-order part")
        if part.get("published") is not True:
            raise ValueError(f"accepted part {part_index:06d} is not published")
        result_path = output_dir / f"part-{part_index:06d}.jsonl"
        accepted_path = (
            accepted_dir / f"bt_{input_stem}_{manifest['run_id']}_{part_index:06d}.jsonl"
        )
        result = _validate_artifact(
            part.get("result", {}),
            expected_path=result_path,
            label=f"result part {part_index:06d}",
        )
        accepted = _validate_artifact(
            part.get("accepted", {}),
            expected_path=accepted_path,
            label=f"accepted part {part_index:06d}",
        )
        if int(part.get("source_rows", -1)) != result["rows"]:
            raise ValueError(f"result part {part_index:06d} source row count mismatch")
        if int(part.get("source_start_index", -1)) != expected_source_start:
            raise ValueError(f"result part {part_index:06d} source range is not contiguous")
        status_counts = part.get("status_counts")
        generated_rows = part.get("generated_rows")
        if not isinstance(status_counts, Mapping) or generated_rows is None:
            status_counts, generated_rows = _part_semantics(
                result_path,
                accepted_path,
                source_start_index=expected_source_start,
                run_id=str(manifest["run_id"]),
            )
            part["status_counts"] = status_counts
            part["generated_rows"] = generated_rows
            changed = True
        normalized_counts = {
            status: int(status_counts.get(status, 0))
            for status in ("accepted", "rejected", "error", "skipped_existing")
        }
        if sum(normalized_counts.values()) != result["rows"]:
            raise ValueError(f"result part {part_index:06d} status counts do not match rows")
        if normalized_counts["accepted"] != accepted["rows"]:
            raise ValueError(f"accepted part {part_index:06d} row count does not match statuses")
        if not 0 <= int(generated_rows) <= int(result["rows"]):
            raise ValueError(f"result part {part_index:06d} has invalid generated row count")
        total_result_rows += int(result["rows"])
        total_accepted_rows += int(accepted["rows"])
        total_generated_rows += int(generated_rows)
        total_status_counts.update(normalized_counts)
        expected_source_start += int(result["rows"])

    if total_result_rows != int(progress["completed_rows"]):
        raise ValueError("committed result rows do not match progress.completed_rows")
    stats = cast(dict[str, Any], manifest["stats"])
    if total_result_rows != int(stats["processed"]):
        raise ValueError("committed result rows do not match stats.processed")
    if total_accepted_rows != int(stats["accepted"]):
        raise ValueError("committed accepted rows do not match stats.accepted")
    for status, stat_name in (
        ("rejected", "rejected"),
        ("error", "errors"),
        ("skipped_existing", "skipped_existing"),
    ):
        if total_status_counts[status] != int(stats[stat_name]):
            raise ValueError(f"committed {status} rows do not match stats.{stat_name}")
    if "generated" not in stats:
        stats["generated"] = total_generated_rows
        changed = True
    elif total_generated_rows != int(stats["generated"]):
        raise ValueError("committed generated rows do not match stats.generated")
    return changed


def _configure_teacher_review(
    manifest: dict[str, Any],
    *,
    teacher_pilot_rows: int | None,
    approve_teacher: bool,
    approval_actor: str | None,
    existing_manifest: bool,
) -> bool:
    """Restore/freeze the pilot policy and record explicit post-pilot approval."""

    stats = manifest["stats"]
    changed = False
    if "generated" not in stats:
        stats["generated"] = int(stats.get("accepted", 0)) + int(stats.get("rejected", 0))
        changed = True
    review = manifest.get("teacher_review")
    if review is None and teacher_pilot_rows is not None:
        review = {
            "pilot_rows": teacher_pilot_rows,
            "review_required": False,
            "approved": False,
            "approved_at": None,
            "approved_by": None,
        }
        manifest["teacher_review"] = review
        changed = True
    elif isinstance(review, dict) and teacher_pilot_rows is not None:
        if int(review["pilot_rows"]) != teacher_pilot_rows:
            raise ValueError(
                "teacher pilot size is fixed by the manifest; "
                "use the original value or a new output directory"
            )
    elif review is not None and not isinstance(review, dict):
        raise ValueError("invalid teacher_review in queue manifest")

    if (
        isinstance(review, dict)
        and not review.get("approved")
        and (
            int(stats["generated"]) >= int(str(review["pilot_rows"]))
            or (manifest["progress"].get("complete") and int(stats["generated"]) > 0)
        )
        and not review.get("review_required")
    ):
        review = cast(dict[str, Any], review)
        review["review_required"] = True
        changed = True

    if approve_teacher:
        if not existing_manifest or not isinstance(review, dict):
            raise ValueError("a teacher cannot be approved before a pilot run")
        if not review.get("review_required"):
            raise ValueError("the teacher pilot is not complete or has no reviewable output")
        if not review.get("approved"):
            review = cast(dict[str, Any], review)
            actor = canonical_text(approval_actor or "")
            if not actor:
                raise ValueError("approval_actor is required to approve a teacher")
            review["approved"] = True
            review["approved_at"] = datetime.now(UTC).isoformat()
            review["approved_by"] = actor
            changed = True
    return changed


def _remaining_teacher_pilot_rows(manifest: Mapping[str, Any]) -> int | None:
    review = manifest.get("teacher_review")
    if not isinstance(review, Mapping) or review.get("approved"):
        return None
    return max(
        0,
        int(review["pilot_rows"]) - int(manifest["stats"].get("generated", 0)),
    )


def _translate_queue_unlocked(
    input_path: str | Path,
    output_dir: str | Path,
    translator: TranslatorLike,
    *,
    accepted_dir: str | Path,
    options: QueueTranslationOptions | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    max_rows: int | None = None,
    teacher_pilot_rows: int | None = None,
    approve_teacher: bool = False,
    approval_actor: str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Translate pending queue rows into atomic result and training shards."""

    options = options or QueueTranslationOptions()
    options.validate()
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive or None")
    if teacher_pilot_rows is not None and teacher_pilot_rows <= 0:
        raise ValueError("teacher_pilot_rows must be positive or None")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    accepted_dir = Path(accepted_dir)
    manifest_path = output_dir / "manifest.json"
    existing: dict[str, Any] | None = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict) or loaded.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"unsupported queue manifest: {manifest_path}")
        existing = loaded
    previous_source = (
        existing.get("configuration", {}).get("source") if existing is not None else None
    )
    force_source_hash = bool(
        approve_teacher or (existing is not None and existing.get("progress", {}).get("complete"))
    )
    source = _source_identity(
        input_path,
        previous_source,
        force_hash=force_source_hash,
    )
    candidate = _new_manifest(
        source=source,
        options=options,
        run_metadata=run_metadata or {},
        accepted_dir=accepted_dir,
        teacher_pilot_rows=teacher_pilot_rows,
    )
    resume_metadata_changed = False
    if existing is None:
        non_lock_entries = [
            entry
            for entry in output_dir.iterdir()
            if entry.name not in {RUN_LOCK_FILENAME, ACCEPTED_LOCK_FILENAME}
        ]
        if non_lock_entries:
            raise FileExistsError(f"{output_dir} is not empty and has no compatible queue manifest")
        manifest = candidate
    else:
        existing_configuration = existing.get("configuration")
        if not isinstance(existing_configuration, Mapping):
            raise ValueError("queue manifest has no valid configuration")
        stable_existing_signature = _stable_digest(_signature_configuration(existing_configuration))
        legacy_existing_signature = _stable_digest(existing_configuration)
        recorded_signature = existing.get("run_signature")
        if recorded_signature not in {
            stable_existing_signature,
            legacy_existing_signature,
        }:
            raise ValueError("queue manifest signature is invalid")
        if stable_existing_signature != candidate["run_signature"]:
            raise ValueError(
                "queue resume configuration changed; use a new output directory "
                "for a different source, model, or quality policy"
            )
        manifest = existing
        resume_metadata_changed = (
            recorded_signature != stable_existing_signature
            or manifest.get("signature_version") != SIGNATURE_VERSION
            or manifest["configuration"].get("source") != source
        )
        manifest["run_signature"] = stable_existing_signature
        manifest["signature_version"] = SIGNATURE_VERSION
        manifest["configuration"]["source"] = source
    _claim_accepted_namespace(
        manifest,
        output_dir=output_dir,
        accepted_dir=accepted_dir,
        input_stem=input_path.stem,
    )
    if existing is not None:
        recovery_changed = _recover_pending_accepted_parts(
            manifest,
            accepted_dir=accepted_dir,
            input_stem=input_path.stem,
        )
    else:
        recovery_changed = False
    validation_changed = _validate_or_register_parts(
        manifest,
        output_dir=output_dir,
        accepted_dir=accepted_dir,
        input_stem=input_path.stem,
    )
    parts_changed = recovery_changed or validation_changed
    review_changed = _configure_teacher_review(
        manifest,
        teacher_pilot_rows=teacher_pilot_rows,
        approve_teacher=approve_teacher,
        approval_actor=approval_actor,
        existing_manifest=existing is not None,
    )
    if existing is None or resume_metadata_changed or parts_changed or review_changed:
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_write_json(manifest_path, manifest)
    progress = manifest["progress"]
    if progress.get("complete"):
        return manifest

    processed_this_call = 0
    with input_path.open("rb") as source_handle:
        source_handle.seek(int(progress["source_byte_offset"]))
        while max_rows is None or processed_this_call < max_rows:
            pilot_remaining = _remaining_teacher_pilot_rows(manifest)
            if pilot_remaining == 0:
                review = manifest["teacher_review"]
                if not review["review_required"]:
                    review["review_required"] = True
                    manifest["updated_at"] = datetime.now(UTC).isoformat()
                    _atomic_write_json(manifest_path, manifest)
                break
            remaining = (
                options.shard_size
                if max_rows is None
                else min(options.shard_size, max_rows - processed_this_call)
            )
            if pilot_remaining is not None:
                remaining = min(remaining, pilot_remaining)
            start_index = int(progress["completed_rows"])
            raw_rows: list[bytes] = []
            for _ in range(remaining):
                raw = source_handle.readline()
                if not raw:
                    break
                raw_rows.append(raw)
            if not raw_rows:
                final_source = _source_identity(input_path, None, force_hash=True)
                if (
                    _signature_configuration({"source": final_source})["source"]
                    != _signature_configuration({"source": manifest["configuration"]["source"]})[
                        "source"
                    ]
                ):
                    raise ValueError("input source content changed during queue translation")
                manifest["configuration"]["source"] = final_source
                progress["complete"] = True
                review = manifest.get("teacher_review")
                if (
                    isinstance(review, dict)
                    and not review["approved"]
                    and manifest["stats"]["generated"] > 0
                ):
                    review["review_required"] = True
                manifest["updated_at"] = datetime.now(UTC).isoformat()
                _atomic_write_json(manifest_path, manifest)
                break

            results: list[dict[str, Any]] = []
            pending: list[dict[str, Any]] = []
            for offset, raw in enumerate(raw_rows):
                result, needs_translation = _parse_queue_line(raw, start_index + offset)
                results.append(result)
                if needs_translation:
                    pending.append(result)

            groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for result in pending:
                key = (result["source_lang"], result["target_lang"])
                groups.setdefault(key, []).append(result)
            for (source_language, target_language), jobs in groups.items():
                forward, errors = _translate_resilient(
                    translator,
                    [job["source"] for job in jobs],
                    source_language=source_language,
                    target_language=target_language,
                    num_beams=options.num_beams,
                    max_new_tokens=options.max_new_tokens,
                    batch_size=options.batch_size,
                    max_output_length_ratio=options.max_output_length_ratio,
                    max_output_length_margin=options.max_output_length_margin,
                )
                for job, translation, error in zip(jobs, forward, errors, strict=True):
                    if error is not None or not translation:
                        job["status"] = "error"
                        job["error"] = error or "empty_translation"
                        job["translation"] = None
                        continue
                    job["translation"] = translation
                    quality, reasons = _forward_quality(
                        job["source"],
                        translation,
                        source_language=source_language,
                        target_language=target_language,
                    )
                    job["quality"] = {"forward": quality}
                    if quality["pair_score"] < options.min_pair_score:
                        reasons.append("pair_score")
                    if not quality["number_exact"]:
                        reasons.append("number_mismatch")
                    if (
                        quality["critical_structured_mismatch"]
                        or quality["structured"] < options.min_structured_similarity
                    ):
                        reasons.append("structured_mismatch")
                    target_fraction = quality["target_language_fraction"]
                    if (
                        isinstance(target_fraction, (int, float))
                        and target_fraction < options.min_target_language_fraction
                    ):
                        reasons.append("target_language")
                    if (
                        primary_language(target_language) == "ja"
                        and isinstance(quality["target_japanese_kana_chars"], int)
                        and quality["target_japanese_kana_chars"] < options.min_japanese_kana_chars
                    ):
                        reasons.append("target_japanese_kana")
                    if reasons:
                        job["status"] = "rejected"
                        job["rejection_reasons"] = list(dict.fromkeys(reasons))
                    else:
                        job["status"] = "forward_pass"

            if options.roundtrip_enabled:
                cycle_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
                for job in pending:
                    if job["status"] == "forward_pass":
                        key = (job["target_lang"], job["source_lang"])
                        cycle_groups.setdefault(key, []).append(job)
                for (source_language, target_language), jobs in cycle_groups.items():
                    cycles, errors = _translate_resilient(
                        translator,
                        [str(job["translation"]) for job in jobs],
                        source_language=source_language,
                        target_language=target_language,
                        num_beams=options.roundtrip_num_beams,
                        max_new_tokens=options.roundtrip_max_new_tokens,
                        batch_size=options.batch_size,
                        max_output_length_ratio=(options.roundtrip_max_output_length_ratio),
                        max_output_length_margin=(options.roundtrip_max_output_length_margin),
                    )
                    for job, cycle, error in zip(jobs, cycles, errors, strict=True):
                        if error is not None or not cycle:
                            job["status"] = "error"
                            job["error"] = error or "empty_roundtrip"
                            continue
                        cycle_quality = roundtrip_quality(
                            job["source"],
                            cycle,
                            tokenize=lambda text: _content_tokens(translator, text),
                        )
                        job["roundtrip"] = cycle
                        job["quality"]["roundtrip"] = cycle_quality
                        reasons: list[str] = []
                        if cycle_quality["score"] < options.min_roundtrip_score:
                            reasons.append("roundtrip_score")
                        if cycle_quality["number_f1"] < 1.0:
                            reasons.append("roundtrip_number_mismatch")
                        if (
                            cycle_quality["critical_structured_mismatch"]
                            or cycle_quality["structured"] < options.min_structured_similarity
                        ):
                            reasons.append("roundtrip_structured_mismatch")
                        if reasons:
                            job["status"] = "rejected"
                            job["rejection_reasons"] = reasons
                        else:
                            job["status"] = "accepted"
            else:
                for job in pending:
                    if job["status"] == "forward_pass":
                        job["status"] = "accepted"

            part = int(progress["next_part"])
            run_id = str(manifest["run_id"])
            accepted_rows: list[dict[str, Any]] = []
            for result in results:
                result["run_id"] = run_id
                if result["status"] != "accepted":
                    continue
                cycle_score = (
                    result.get("quality", {}).get("roundtrip", {}).get("score")
                    if options.roundtrip_enabled
                    else None
                )
                accepted_rows.append(
                    {
                        result["source_lang"]: result["source"],
                        result["target_lang"]: result["translation"],
                        "id": result["id"],
                        "synthetic": True,
                        "provenance": {
                            "type": "machine_translation",
                            "queue_id": result["id"],
                            "run_id": run_id,
                            "roundtrip_score": cycle_score,
                        },
                    }
                )

            counts = Counter(result["status"] for result in results)
            status_counts = {
                status: counts[status]
                for status in ("accepted", "rejected", "error", "skipped_existing")
            }
            generated_rows = sum(
                result.get("translation") is not None and result["status"] != "skipped_existing"
                for result in results
            )
            result_path = output_dir / f"part-{part:06d}.jsonl"
            accepted_path = accepted_dir / f"bt_{input_path.stem}_{run_id}_{part:06d}.jsonl"
            pending_accepted_path = _pending_accepted_path(accepted_path)
            if accepted_path.exists():
                raise FileExistsError(
                    "uncommitted accepted shard target already exists; "
                    f"refusing to overwrite {accepted_path}"
                )
            _atomic_write_jsonl(result_path, results)
            _atomic_write_jsonl(pending_accepted_path, accepted_rows)
            accepted_artifact = _jsonl_artifact(pending_accepted_path)
            accepted_artifact["path"] = str(accepted_path.resolve())
            manifest["parts"].append(
                {
                    "part": part,
                    "source_start_index": start_index,
                    "source_rows": len(raw_rows),
                    "result": _jsonl_artifact(result_path),
                    "accepted": accepted_artifact,
                    "status_counts": status_counts,
                    "generated_rows": generated_rows,
                    "published": False,
                }
            )

            stats = manifest["stats"]
            stats["processed"] += len(results)
            stats["generated"] += generated_rows
            stats["accepted"] += counts["accepted"]
            stats["rejected"] += counts["rejected"]
            stats["errors"] += counts["error"]
            stats["skipped_existing"] += counts["skipped_existing"]
            progress["completed_rows"] += len(raw_rows)
            progress["source_byte_offset"] = source_handle.tell()
            progress["next_part"] += 1
            processed_this_call += len(raw_rows)
            position = source_handle.tell()
            source_exhausted = source_handle.read(1) == b""
            source_handle.seek(position)
            progress["complete"] = False
            if source_exhausted:
                final_source = _source_identity(input_path, None, force_hash=True)
                if (
                    _signature_configuration({"source": final_source})["source"]
                    != (
                        _signature_configuration({"source": manifest["configuration"]["source"]})[
                            "source"
                        ]
                    )
                ):
                    raise ValueError("input source content changed during queue translation")
                manifest["configuration"]["source"] = final_source
            review = manifest.get("teacher_review")
            if (
                isinstance(review, dict)
                and not review["approved"]
                and (
                    stats["generated"] >= review["pilot_rows"]
                    or (source_exhausted and stats["generated"] > 0)
                )
            ):
                review["review_required"] = True
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            _atomic_write_json(manifest_path, manifest)
            _publish_no_replace(pending_accepted_path, accepted_path)
            manifest["parts"][-1]["published"] = True
            progress["complete"] = source_exhausted
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            _atomic_write_json(manifest_path, manifest)
            if log is not None:
                log(
                    f"part {part:06d}: {len(results):,} rows, "
                    f"{counts['accepted']:,} accepted, {counts['rejected']:,} rejected, "
                    f"{counts['error']:,} errors; total {stats['processed']:,}"
                )
            if progress["complete"]:
                break
    return manifest


def translate_queue(
    input_path: str | Path,
    output_dir: str | Path,
    translator: TranslatorLike,
    *,
    accepted_dir: str | Path,
    options: QueueTranslationOptions | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    max_rows: int | None = None,
    teacher_pilot_rows: int | None = None,
    approve_teacher: bool = False,
    approval_actor: str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Translate a queue under a single-writer lock."""

    output_path = Path(output_dir)
    accepted_path = Path(accepted_dir)
    with ExitStack() as locks:
        locks.enter_context(_queue_run_lock(output_path))
        if output_path.resolve() != accepted_path.resolve():
            locks.enter_context(_accepted_run_lock(accepted_path))
        return _translate_queue_unlocked(
            input_path,
            output_dir,
            translator,
            accepted_dir=accepted_dir,
            options=options,
            run_metadata=run_metadata,
            max_rows=max_rows,
            teacher_pilot_rows=teacher_pilot_rows,
            approve_teacher=approve_teacher,
            approval_actor=approval_actor,
            log=log,
        )
