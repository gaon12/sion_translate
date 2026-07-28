"""Resumable, auditable translation of monolingual JSONL queues.

Queue files are immutable inputs.  Each source row receives a result record,
while only rows that pass forward and round-trip checks are copied into
separate ``bt_*`` training shards.  Progress is committed after atomic shard
writes so a stopped multi-day run can safely resume from its byte offset.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Protocol

from sacrebleu.metrics import CHRF

from sion_translate.data.quality import assess_pair, canonical_text, language_fraction
from sion_translate.evaluation import multiset_f1, numeric_tokens
from sion_translate.structured import structured_similarity


MANIFEST_SCHEMA = "sion-translation-queue-v1"
RESULT_SCHEMA = "sion-translation-result-v1"
PIPELINE_VERSION = 1
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
        return list(encode(text))
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
    }
    reasons = list(assessment.rejection_reasons)
    return quality, reasons


def _source_identity(path: Path, previous: Mapping[str, Any] | None) -> dict[str, Any]:
    stat = path.stat()
    resolved = str(path.resolve())
    if (
        previous
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
    signature = _stable_digest(configuration)
    return {
        "schema": MANIFEST_SCHEMA,
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
    }


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
            int(stats["generated"]) >= int(review["pilot_rows"])
            or (manifest["progress"].get("complete") and int(stats["generated"]) > 0)
        )
        and not review.get("review_required")
    ):
        review["review_required"] = True
        changed = True

    if approve_teacher:
        if not existing_manifest or not isinstance(review, dict):
            raise ValueError("a teacher cannot be approved before a pilot run")
        if not review.get("review_required"):
            raise ValueError("the teacher pilot is not complete or has no reviewable output")
        if not review.get("approved"):
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
    source = _source_identity(input_path, previous_source)
    candidate = _new_manifest(
        source=source,
        options=options,
        run_metadata=run_metadata or {},
        accepted_dir=accepted_dir,
        teacher_pilot_rows=teacher_pilot_rows,
    )
    if existing is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"{output_dir} is not empty and has no compatible queue manifest")
        manifest = candidate
    else:
        if existing.get("run_signature") != candidate["run_signature"]:
            raise ValueError(
                "queue resume configuration changed; use a new output directory "
                "for a different source, model, or quality policy"
            )
        manifest = existing
    review_changed = _configure_teacher_review(
        manifest,
        teacher_pilot_rows=teacher_pilot_rows,
        approve_teacher=approve_teacher,
        approval_actor=approval_actor,
        existing_manifest=existing is not None,
    )
    if existing is None or review_changed:
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
                progress["complete"] = True
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
                    if quality["target_language_fraction"] < options.min_target_language_fraction:
                        reasons.append("target_language")
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

            result_path = output_dir / f"part-{part:06d}.jsonl"
            accepted_path = accepted_dir / f"bt_{input_path.stem}_{run_id}_{part:06d}.jsonl"
            _atomic_write_jsonl(result_path, results)
            _atomic_write_jsonl(accepted_path, accepted_rows)

            counts = Counter(result["status"] for result in results)
            stats = manifest["stats"]
            stats["processed"] += len(results)
            stats["generated"] += sum(
                result.get("translation") is not None and result["status"] != "skipped_existing"
                for result in results
            )
            stats["accepted"] += counts["accepted"]
            stats["rejected"] += counts["rejected"]
            stats["errors"] += counts["error"]
            stats["skipped_existing"] += counts["skipped_existing"]
            progress["completed_rows"] += len(raw_rows)
            progress["source_byte_offset"] = source_handle.tell()
            progress["next_part"] += 1
            processed_this_call += len(raw_rows)
            position = source_handle.tell()
            progress["complete"] = source_handle.read(1) == b""
            source_handle.seek(position)
            review = manifest.get("teacher_review")
            if (
                isinstance(review, dict)
                and not review["approved"]
                and (
                    stats["generated"] >= review["pilot_rows"]
                    or (progress["complete"] and stats["generated"] > 0)
                )
            ):
                review["review_required"] = True
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
