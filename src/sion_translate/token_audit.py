"""Measure whether tokenizer pieces receive enough translation training signal.

Tokenizer *coverage* and model *exposure* are different problems. Byte fallback
guarantees that every string can be encoded, but a piece seen only a handful of
times as a decoder target still has an effectively untrained output embedding.
This module scans the same parallel JSONL schema as the training pipeline and
reports both failure modes by language and translation direction.
"""

# Audit reports consume heterogeneous JSON records validated at runtime.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from collections.abc import Mapping
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterator, Sequence, cast

import numpy as np

from sion_translate.data.integrity import validate_dataset_artifact_inventory
from sion_translate.data.monolingual import MonolingualDiscovery, iter_monolingual_lines
from sion_translate.data.prepare import (
    INDEX_DTYPE,
    INDEX_FORMAT,
    PREPARE_COMPLETION_FILENAME,
    PREPARE_COMPLETION_SCHEMA,
    RAW_FINGERPRINT_FILENAME,
    SHARED_TARGET_INDEX_DTYPE,
    prepare_stats_schema_from_manifest,
    validated_prepare_stats,
)
from sion_translate.data.quality import QualityPolicy, assess_pair, canonical_text
from sion_translate.data.record_metadata import (
    RECORD_METADATA_DATA_SUFFIX,
    RECORD_METADATA_FIELDS,
    RECORD_METADATA_FORMAT,
    RECORD_METADATA_INDEX_DTYPE,
    RECORD_METADATA_INDEX_SUFFIX,
    decode_record_metadata,
    resolve_record_training_direction,
)
from sion_translate.data.records import (
    expand_parallel_record,
    languages_from_pairs,
    normalize_language_pairs,
    normalize_translation_directions,
)
from sion_translate.fingerprint import (
    FINGERPRINT_SCHEMA,
    PREPROCESSING_SCHEMA,
    DatasetFingerprint,
    file_sha256,
)
from sion_translate.language_tags import (
    canonicalize_language_pair,
    canonicalize_language_tag,
    canonicalize_language_tags,
)
from sion_translate.synthetic import (
    normalize_synthetic_prefixes,
    synthetic_path,
    synthetic_record,
)
from sion_translate.tokenizer import SionTokenizer, expand_inputs


_KNOWN_LEGACY_PREPROCESSING_SCHEMAS = frozenset(
    {
        "sion-prepare-v4",
        "sion-prepare-v5",
        "sion-prepare-v6",
        "sion-prepare-v7",
        "sion-prepare-v8",
        "sion-prepare-v9",
    }
)
_SYNTHETIC_AUDIT_MARKER = "_sion_token_audit_synthetic_v1"


def _piece_is_special(piece: str) -> bool:
    return piece.startswith("<") and piece.endswith(">") and not piece.startswith("<0x")


def _direction_label(source: str, target: str) -> str:
    """Keep legacy labels for simple tags and delimit BCP 47 tags safely."""

    separator = "-" if "-" not in source and "-" not in target else "/"
    return f"{source}{separator}{target}"


def _annotate_synthetic_scopes(node: object, *, inherited: bool = False) -> object:
    """Attach a private provenance marker to each generated record subtree.

    ``expand_parallel_record`` deliberately supports one JSONL value containing
    multiple nested records.  A line-global boolean would therefore taint real
    siblings whenever only one child is generated.  The private provenance key
    follows the record expander's existing metadata inheritance without changing
    user data or the training-direction field.
    """

    if isinstance(node, (list, tuple)):
        return [
            _annotate_synthetic_scopes(item, inherited=inherited)
            for item in cast(Sequence[object], node)
        ]
    if not isinstance(node, Mapping):
        return deepcopy(node)

    current = inherited or synthetic_record(node)
    annotated: dict[object, object] = {}
    for key, value in node.items():
        if key in {"metadata", "provenance"}:
            annotated[key] = deepcopy(value)
        else:
            annotated[key] = _annotate_synthetic_scopes(value, inherited=current)
    if current:
        raw_provenance = annotated.get("provenance")
        if isinstance(raw_provenance, Mapping):
            provenance: dict[object, object] = deepcopy(dict(raw_provenance))
        elif raw_provenance is None:
            provenance = {}
        else:
            provenance = {"original": deepcopy(raw_provenance)}
        provenance[_SYNTHETIC_AUDIT_MARKER] = True
        annotated["provenance"] = provenance
    return annotated


def _metadata_is_synthetic(metadata: Mapping[str, object]) -> bool:
    provenance = metadata.get("provenance")
    return isinstance(provenance, Mapping) and provenance.get(_SYNTHETIC_AUDIT_MARKER) is True


def _raw_audit_synthetic_prefixes(
    configured: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Normalize shared train-only prefixes and select ambiguous generated files.

    ``concat_`` is the one documented exception: that namespace may contain
    ordinary concatenated real bitext and therefore needs row metadata, rather
    than its filename alone, to be classified as generated. Every other shared
    or caller-configured train-only prefix fails closed without a direction.
    """

    if isinstance(configured, (str, bytes)):
        raise ValueError("train_only_prefixes must be a sequence of filename prefixes")
    normalized = normalize_synthetic_prefixes(configured)
    direction_required = tuple(prefix for prefix in normalized if prefix != "concat_")
    return normalized, direction_required


def _frequency_summary(counts: np.ndarray, eligible: np.ndarray) -> dict[str, int | float]:
    values = counts[eligible]
    observed = values[values > 0]
    return {
        "total_occurrences": int(values.sum(dtype=np.uint64)),
        "eligible_pieces": int(values.size),
        "observed_pieces": int(observed.size),
        "unused_pieces": int(np.count_nonzero(values == 0)),
        "seen_once": int(np.count_nonzero(values == 1)),
        "seen_1_to_9": int(np.count_nonzero((values >= 1) & (values <= 9))),
        "seen_1_to_24": int(np.count_nonzero((values >= 1) & (values <= 24))),
        "median_observed_count": float(np.median(observed)) if observed.size else 0.0,
        "p10_observed_count": float(np.percentile(observed, 10)) if observed.size else 0.0,
    }


def _load_indexed_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Indexed dataset manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid indexed dataset manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Indexed dataset manifest must contain a JSON object: {manifest_path}")
    return manifest


def _normalize_explicit_language_pairs(
    value: object,
    *,
    field: str,
) -> tuple[tuple[str, str], ...]:
    """Canonicalize physical pairs and reject duplicate undirected identities."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{field} must be a non-empty sequence of two-language pairs")
    pairs: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for index, raw_pair in enumerate(value):
        pair = canonicalize_language_pair(raw_pair, field=f"{field}[{index}]")
        edge = frozenset(pair)
        if edge in seen:
            raise ValueError(
                f"{field} contains a duplicate or reversed physical pair after "
                f"canonicalization: {pair!r}"
            )
        seen.add(edge)
        pairs.append(pair)
    return tuple(pairs)


def _indexed_tokenizer_identity(
    manifest: dict[str, Any],
    tokenizer_model: Path,
) -> dict[str, object]:
    """Verify a pinned digest, or label a mutable legacy path match as unverified."""

    expected_sha256 = None
    identity_source = None
    pinned_identity = False
    fingerprint = manifest.get("fingerprint")
    if isinstance(fingerprint, dict) and isinstance(fingerprint.get("tokenizer_sha256"), str):
        expected_sha256 = fingerprint["tokenizer_sha256"].lower()
        identity_source = "manifest.fingerprint.tokenizer_sha256"
        pinned_identity = True
    else:
        recorded_path = manifest.get("tokenizer_model")
        if isinstance(recorded_path, str) and Path(recorded_path).is_file():
            expected_sha256 = file_sha256(recorded_path).lower()
            identity_source = "manifest.tokenizer_model (mutable path-time comparison)"

    actual_sha256 = file_sha256(tokenizer_model).lower()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            "Tokenizer SHA-256 does not match the indexed dataset: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return {
        "sha256": actual_sha256,
        "verified_against_manifest": pinned_identity,
        "mutable_path_match": expected_sha256 is not None and not pinned_identity,
        "assurance": (
            "pinned-sha256"
            if pinned_identity
            else "mutable-path-match-unverified"
            if expected_sha256 is not None
            else "unverified"
        ),
        "identity_source": identity_source,
    }


def _indexed_source_only_languages(
    manifest: Mapping[str, object],
    languages: Sequence[str],
) -> tuple[str, ...]:
    raw_source_only = manifest.get("source_only_languages", [])
    if not isinstance(raw_source_only, list):
        raise ValueError("Indexed dataset source_only_languages must be a list")
    source_only = canonicalize_language_tags(
        raw_source_only,
        field="indexed manifest source_only_languages",
        reject_duplicates=True,
    )
    unknown = sorted(set(source_only) - set(languages))
    if unknown:
        raise ValueError(f"Indexed dataset manifest has unknown source-only languages: {unknown}")
    return source_only


def _validate_indexed_language_to_id(
    manifest: Mapping[str, object],
    languages: Sequence[str],
) -> None:
    raw_mapping = manifest.get("language_to_id")
    if raw_mapping is None:
        return
    if not isinstance(raw_mapping, Mapping):
        raise ValueError("Indexed dataset language_to_id must be an object")
    actual: dict[str, int] = {}
    for raw_language, raw_id in raw_mapping.items():
        language = canonicalize_language_tag(
            raw_language,
            field="indexed manifest language_to_id key",
        )
        if language in actual:
            raise ValueError(
                "Indexed dataset language_to_id contains duplicate canonical language aliases"
            )
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise ValueError("Indexed dataset language_to_id values must be integer ids")
        actual[language] = raw_id
    expected = {language: index for index, language in enumerate(languages)}
    if actual != expected:
        raise ValueError(
            f"Indexed dataset language_to_id disagrees with languages: {actual!r} != {expected!r}"
        )


def _indexed_direction_contract(
    manifest: Mapping[str, object],
    *,
    current_schema: bool,
    legacy_storage_layout: bool,
    legacy_bidirectional: bool | None,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[str, ...],
    tuple[str, str] | None,
]:
    """Authenticate canonical language identities and trained direction edges."""

    legacy_storage_pair: tuple[str, str] | None = None
    if current_schema and legacy_bidirectional is not None:
        raise ValueError(
            "bidirectional is a legacy indexed-dataset override; current datasets "
            "authenticate directions in manifest.translation_directions"
        )

    if legacy_storage_layout:
        raw_pair = manifest.get("language_pair")
        pair = canonicalize_language_pair(
            raw_pair,
            field="legacy indexed manifest language_pair",
        )
        assert isinstance(raw_pair, Sequence) and not isinstance(raw_pair, (str, bytes))
        legacy_storage_pair = (cast(str, raw_pair[0]), cast(str, raw_pair[1]))
        pairs = (pair,)
    else:
        raw_pairs = manifest.get("language_pairs")
        if raw_pairs is None and not current_schema:
            raw_primary = manifest.get("language_pair")
            pairs = (
                canonicalize_language_pair(
                    raw_primary,
                    field="legacy indexed manifest language_pair",
                ),
            )
        elif manifest.get("stage") == "foundation":
            if not isinstance(raw_pairs, list) or not raw_pairs:
                raise ValueError("Foundation indexed manifest has no language tasks")
            foundation_pairs: list[tuple[str, str]] = []
            seen_foundation_languages: set[str] = set()
            for index, raw_pair in enumerate(cast(list[object], raw_pairs)):
                if not isinstance(raw_pair, list) or len(raw_pair) != 2:
                    raise ValueError("Foundation indexed language tasks must be self-pairs")
                pair_values = cast(list[object], raw_pair)
                source = canonicalize_language_tag(
                    pair_values[0],
                    field=f"indexed manifest language_pairs[{index}][0]",
                )
                target = canonicalize_language_tag(
                    pair_values[1],
                    field=f"indexed manifest language_pairs[{index}][1]",
                )
                if source != target or source in seen_foundation_languages:
                    raise ValueError(
                        "Foundation indexed language tasks must be unique canonical self-pairs"
                    )
                seen_foundation_languages.add(source)
                foundation_pairs.append((source, target))
            pairs = tuple(foundation_pairs)
        else:
            pairs = _normalize_explicit_language_pairs(
                raw_pairs,
                field="indexed manifest language_pairs",
            )
        raw_primary = manifest.get("language_pair")
        if raw_primary is not None:
            primary = canonicalize_language_pair(
                raw_primary,
                field="indexed manifest language_pair",
            )
            if primary != pairs[0]:
                raise ValueError(
                    "Indexed dataset language_pair must equal the first canonical language_pairs "
                    f"entry: {primary!r} != {pairs[0]!r}"
                )

    if manifest.get("stage") == "foundation":
        expected_languages = tuple(source for source, _target in pairs)
    else:
        expected_languages = languages_from_pairs(pairs)
    raw_languages = manifest.get("languages")
    if raw_languages is None and not current_schema:
        languages = expected_languages
    else:
        if not isinstance(raw_languages, list) or not raw_languages:
            raise ValueError("Indexed dataset manifest has no valid languages list")
        languages = canonicalize_language_tags(
            raw_languages,
            field="indexed manifest languages",
            reject_duplicates=True,
        )
    if languages != expected_languages:
        raise ValueError(
            "Indexed dataset languages must exactly match first appearance in language_pairs: "
            f"{languages!r} != {expected_languages!r}"
        )
    if not legacy_storage_layout and len(languages) > np.iinfo(np.uint16).max:
        raise ValueError("Indexed dataset has too many languages for uint16 language ids")
    _validate_indexed_language_to_id(manifest, languages)

    source_only = _indexed_source_only_languages(manifest, languages)
    raw_directions = manifest.get("translation_directions")
    if raw_directions is None:
        if manifest.get("stage") == "foundation":
            if any(source != target for source, target in pairs):
                raise ValueError("Foundation indexed datasets require self-pair tasks")
            directions = pairs
        else:
            if current_schema:
                raise ValueError("Current indexed dataset requires manifest.translation_directions")
            if legacy_bidirectional is None:
                raise ValueError(
                    "Legacy indexed dataset without translation_directions requires an explicit "
                    "bidirectional=True or bidirectional=False audit policy"
                )
            directions = normalize_translation_directions(
                pairs,
                bidirectional=legacy_bidirectional,
                source_only_languages=source_only,
            )
    else:
        if isinstance(raw_directions, (str, bytes)) or not isinstance(raw_directions, Sequence):
            raise ValueError("Indexed dataset translation_directions must be a sequence")
        directions = normalize_translation_directions(
            pairs,
            cast(Sequence[Sequence[str]], raw_directions),
            bidirectional=False,
            source_only_languages=source_only,
        )
        if not current_schema and legacy_bidirectional is not None:
            expected = normalize_translation_directions(
                pairs,
                bidirectional=legacy_bidirectional,
                source_only_languages=source_only,
            )
            if directions != expected:
                raise ValueError(
                    "Legacy bidirectional compatibility policy contradicts "
                    "manifest.translation_directions"
                )
    return languages, pairs, directions, source_only, legacy_storage_pair


def _read_indexed_json_object(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Indexed dataset {role} not found: {path}")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Indexed dataset {role} cannot be read: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"Indexed dataset {role} must be a JSON object: {path}")
    return cast(dict[str, Any], raw)


def _uses_current_indexed_schema(
    dataset_root: Path,
    manifest: Mapping[str, object],
) -> bool:
    """Match the loader's downgrade-resistant current-schema detection."""

    top_level = manifest.get("preprocessing_schema")
    raw_nested = manifest.get("fingerprint")
    nested = (
        cast(Mapping[object, object], raw_nested).get("preprocessing_schema")
        if isinstance(raw_nested, Mapping)
        else None
    )
    raw_fingerprint_path = dataset_root / RAW_FINGERPRINT_FILENAME
    raw_fingerprint_schema: object = None
    if raw_fingerprint_path.exists():
        raw_fingerprint_schema = _read_indexed_json_object(
            raw_fingerprint_path,
            role="raw fingerprint",
        ).get("preprocessing_schema")
    markers = (top_level, nested, raw_fingerprint_schema)
    if PREPROCESSING_SCHEMA not in markers:
        return False
    if any(marker != PREPROCESSING_SCHEMA for marker in markers):
        raise ValueError("Current dataset preprocessing schema markers disagree")
    return True


def _validate_legacy_indexed_identity(
    dataset_root: Path,
    manifest: Mapping[str, object],
) -> None:
    """Accept named legacy generations without treating a bare v6 label as proof."""

    dataset_format = manifest.get("format")
    if dataset_format in {"sion-foundation-indexed-v2", "sion-foundation-indexed-v3"}:
        if manifest.get("stage") != "foundation":
            raise ValueError("Foundation indexed dataset has an invalid stage identity")
        return
    if dataset_format in {
        "sion-indexed-parallel-v1",
        "sion-indexed-parallel-v2",
        "sion-indexed-parallel-v3",
        "sion-indexed-parallel-v4",
        "sion-indexed-parallel-v5",
    }:
        return
    if dataset_format != INDEX_FORMAT:
        raise ValueError(f"Unsupported indexed dataset format: {dataset_format!r}")

    fingerprint = manifest.get("fingerprint")
    nested = (
        cast(Mapping[object, object], fingerprint).get("preprocessing_schema")
        if isinstance(fingerprint, Mapping)
        else None
    )
    raw_path = dataset_root / RAW_FINGERPRINT_FILENAME
    raw_marker = (
        _read_indexed_json_object(raw_path, role="raw fingerprint").get("preprocessing_schema")
        if raw_path.exists()
        else None
    )
    markers = (manifest.get("preprocessing_schema"), nested, raw_marker)
    if not all(isinstance(marker, str) for marker in markers) or not (
        markers[0] == markers[1] == markers[2]
    ):
        raise ValueError(
            "Unauthenticated v6 dataset is neither a complete current artifact nor an "
            "explicit early-v6 preprocessing generation"
        )
    if markers[0] == PREPROCESSING_SCHEMA:
        raise ValueError("Current v6 preprocessing markers require full artifact authentication")
    if markers[0] not in _KNOWN_LEGACY_PREPROCESSING_SCHEMAS:
        raise ValueError(
            "Unauthenticated v6 dataset claims an unknown historical preprocessing "
            f"generation: {markers[0]!r}"
        )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _manifest_dtype(dtype: np.dtype[Any]) -> object:
    return json.loads(json.dumps(dtype.descr))


def _uses_shared_foundation_storage(manifest: Mapping[str, object]) -> bool:
    """Authenticate the v3 foundation row-alias contract before reading it."""

    enabled = manifest.get("format") == "sion-foundation-indexed-v3"
    target_storage = manifest.get("target_storage")
    if not enabled:
        if target_storage is not None:
            raise ValueError("Only foundation indexed v3 may declare shared-target storage")
        return False
    if manifest.get("stage") != "foundation":
        raise ValueError("Shared-target storage requires the foundation stage")
    if target_storage != "row-shared-source-v1":
        raise ValueError("Foundation shared-target storage contract is invalid")
    if manifest.get("preprocessing_schema") != "foundation-mixed-objectives-v6":
        raise ValueError("Foundation shared-target preprocessing schema is invalid")
    if manifest.get("index_dtype") != _manifest_dtype(SHARED_TARGET_INDEX_DTYPE):
        raise ValueError("Foundation shared-target index dtype is invalid")
    if manifest.get("storage_sides") != ["src", "tgt"]:
        raise ValueError("Foundation shared-target storage sides are invalid")
    return True


def _shared_foundation_source_tasks(manifest: Mapping[str, object]) -> tuple[str, ...]:
    """Return the authenticated task of each v3 foundation source id."""

    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Foundation shared-target manifest has no source tasks")
    tasks: list[str] = []
    for source_id, raw_source in enumerate(cast(list[object], raw_sources)):
        if not isinstance(raw_source, dict):
            raise ValueError("Foundation shared-target source metadata must be objects")
        source = cast(dict[object, object], raw_source)
        task = source.get("task")
        if source.get("id") != source_id or task not in {"denoising", "reasoning"}:
            raise ValueError("Foundation shared-target source metadata is invalid")
        tasks.append(cast(str, task))
    return tuple(tasks)


_LEGACY_GENERIC_REQUIRED_DTYPES = {
    "src_offset": np.dtype("<u8"),
    "src_length": np.dtype("<u4"),
    "tgt_offset": np.dtype("<u8"),
    "tgt_length": np.dtype("<u4"),
    "src_language_id": np.dtype("<u2"),
    "tgt_language_id": np.dtype("<u2"),
}
_LEGACY_GENERIC_OPTIONAL_DTYPES = {
    "src_register": np.dtype("u1"),
    "tgt_register": np.dtype("u1"),
    "source_id": np.dtype("<u2"),
    "quality_score": np.dtype("u1"),
    "synthetic": np.dtype("u1"),
    "forward_only": np.dtype("u1"),
}
_LEGACY_STORAGE_REQUIRED_DTYPES = {
    "ko_offset": np.dtype("<u8"),
    "ko_length": np.dtype("<u4"),
    "ja_offset": np.dtype("<u8"),
    "ja_length": np.dtype("<u4"),
}
_LEGACY_STORAGE_OPTIONAL_DTYPES = {
    "ko_register": np.dtype("u1"),
    "ja_register": np.dtype("u1"),
    "source_id": np.dtype("<u2"),
    "quality_score": np.dtype("u1"),
    "synthetic": np.dtype("u1"),
    "forward_only": np.dtype("u1"),
}


def _validate_legacy_index_dtype(
    index: np.ndarray,
    path: Path,
    *,
    generic: bool,
) -> None:
    """Reject coercible-but-malformed legacy fields before interpreting them."""

    if index.ndim != 1 or index.dtype.names is None:
        raise ValueError(f"Legacy indexed shard must be a one-dimensional structured array: {path}")
    required = _LEGACY_GENERIC_REQUIRED_DTYPES if generic else _LEGACY_STORAGE_REQUIRED_DTYPES
    optional = _LEGACY_GENERIC_OPTIONAL_DTYPES if generic else _LEGACY_STORAGE_OPTIONAL_DTYPES
    fields_by_name = cast(
        Mapping[str, tuple[np.dtype[Any], int]],
        index.dtype.fields or {},
    )
    names = set(index.dtype.names)
    missing = set(required) - names
    unknown = names - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(
            f"Legacy indexed shard fields are invalid at {path}: "
            f"missing={sorted(missing)}, unexpected={sorted(unknown)}"
        )
    for name, expected in {**required, **optional}.items():
        if name not in fields_by_name:
            continue
        actual = fields_by_name[name][0]
        if actual != expected:
            raise ValueError(
                f"Legacy indexed shard field {name!r} has invalid dtype at {path}: "
                f"{actual!r} != {expected!r}"
            )


def _validated_prepare_stats(
    value: object,
    *,
    stats_schema: str | None,
    role: str,
) -> dict[str, int]:
    return asdict(
        validated_prepare_stats(
            value,
            stats_schema=stats_schema,
            role=role,
        )
    )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Current dataset preprocessing {field} must be a positive integer")
    return value


def _strict_fraction(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Current dataset preprocessing {field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"Current dataset preprocessing {field} must be finite and non-negative")
    return normalized


def _validated_quality_policy(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Current dataset preprocessing quality_policy must be an object")
    policy = cast(Mapping[str, object], value)
    expected_fields = {field.name for field in fields(QualityPolicy)}
    if set(policy) != expected_fields:
        raise ValueError("Current dataset preprocessing quality_policy fields are invalid")
    integer_fields = {
        "min_chars_per_side",
        "min_language_check_chars",
        "long_ja_kana_warning_chars",
    }
    numeric_fields = {"max_length_ratio", "min_language_fraction"}
    boolean_fields = {
        "reject_identical",
        "reject_script_mismatch",
        "reject_controls",
        "reject_repetition",
    }
    for name in integer_fields:
        _strict_positive_integer(policy[name], field=f"quality_policy.{name}")
    for name in numeric_fields:
        raw_value = policy[name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"Current dataset preprocessing quality_policy.{name} must be numeric")
        if not math.isfinite(float(raw_value)):
            raise ValueError(f"Current dataset preprocessing quality_policy.{name} must be finite")
    for name in boolean_fields:
        if not isinstance(policy[name], bool):
            raise ValueError(f"Current dataset preprocessing quality_policy.{name} must be boolean")
    normalized = QualityPolicy(
        min_chars_per_side=cast(int, policy["min_chars_per_side"]),
        max_length_ratio=float(cast(int | float, policy["max_length_ratio"])),
        min_language_fraction=float(cast(int | float, policy["min_language_fraction"])),
        min_language_check_chars=cast(int, policy["min_language_check_chars"]),
        long_ja_kana_warning_chars=cast(int, policy["long_ja_kana_warning_chars"]),
        reject_identical=cast(bool, policy["reject_identical"]),
        reject_script_mismatch=cast(bool, policy["reject_script_mismatch"]),
        reject_controls=cast(bool, policy["reject_controls"]),
        reject_repetition=cast(bool, policy["reject_repetition"]),
    )
    normalized.validate()
    return normalized.to_dict()


def _validate_current_preprocessing_options(
    options: Mapping[str, object],
    *,
    language_pair_count: int,
) -> tuple[int, int]:
    """Validate self-described preprocessing values, not only their field names."""

    for name in ("approximate_split", "filter_quality", "prevent_target_leakage"):
        if not isinstance(options.get(name), bool):
            raise ValueError(f"Current dataset preprocessing {name} must be boolean")
    if options.get("dedup_backend") not in {"memory", "sqlite"}:
        raise ValueError("Current dataset preprocessing dedup_backend is invalid")
    shard_size = _strict_positive_integer(options.get("shard_size"), field="shard_size")
    maximum_tokens = _strict_positive_integer(
        options.get("max_tokens_per_side"),
        field="max_tokens_per_side",
    )
    validation_fraction = _strict_fraction(
        options.get("validation_fraction"),
        field="validation_fraction",
    )
    test_fraction = _strict_fraction(
        options.get("test_fraction"),
        field="test_fraction",
    )
    if validation_fraction + test_fraction >= 0.5:
        raise ValueError("Current dataset preprocessing split fractions are unexpectedly large")

    approximate = cast(bool, options["approximate_split"])
    endpoint_key = (
        "language-prefixed-minhash-char5-v1" if approximate else "language-prefixed-exact-v1"
    )
    split_key = "record-sha256-v1" if language_pair_count > 1 else endpoint_key
    if options.get("endpoint_leakage_guard") != "language-endpoint-bloom-v2":
        raise ValueError("Current dataset preprocessing endpoint leakage guard is invalid")
    if options.get("endpoint_leakage_key") != endpoint_key:
        raise ValueError("Current dataset preprocessing endpoint leakage key is invalid")
    if options.get("split_key") != split_key:
        raise ValueError("Current dataset preprocessing split key is invalid")
    _validated_quality_policy(options.get("quality_policy"))
    return shard_size, maximum_tokens


def _validate_current_manifest_contract(
    dataset_root: Path,
    manifest: Mapping[str, Any],
    *,
    language_pairs: tuple[tuple[str, str], ...],
    languages: tuple[str, ...],
    translation_directions: tuple[tuple[str, str], ...],
    source_only_languages: tuple[str, ...],
) -> tuple[
    dict[str, int],
    tuple[dict[str, int], ...],
    tuple[bool, ...],
    int,
    int,
    str,
]:
    """Authenticate the non-payload half of a published current dataset."""

    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ValueError(f"Current dataset root is not a regular directory: {dataset_root}")
    allowed_top_level = {
        "train",
        "validation",
        "test",
        RAW_FINGERPRINT_FILENAME,
        "manifest.json",
        PREPARE_COMPLETION_FILENAME,
    }
    actual_top_level = {candidate.name for candidate in dataset_root.iterdir()}
    if actual_top_level != allowed_top_level:
        raise ValueError(
            "Current dataset top-level artifacts differ from the complete contract: "
            f"missing={sorted(allowed_top_level - actual_top_level)}, "
            f"unexpected={sorted(actual_top_level - allowed_top_level)}"
        )
    for split in ("train", "validation", "test"):
        split_path = dataset_root / split
        if split_path.is_symlink() or not split_path.is_dir():
            raise ValueError(f"Current dataset split is not a regular directory: {split_path}")
    for filename in (
        RAW_FINGERPRINT_FILENAME,
        "manifest.json",
        PREPARE_COMPLETION_FILENAME,
    ):
        metadata_path = dataset_root / filename
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise ValueError(f"Current dataset metadata is not a regular file: {metadata_path}")

    if manifest.get("format") != INDEX_FORMAT:
        raise ValueError("Current dataset manifest format is unsupported")
    if manifest.get("preprocessing_schema") != PREPROCESSING_SCHEMA:
        raise ValueError("Current dataset manifest preprocessing schema is invalid")

    raw_fingerprint = _read_indexed_json_object(
        dataset_root / RAW_FINGERPRINT_FILENAME,
        role="raw fingerprint",
    )
    if manifest.get("fingerprint") != raw_fingerprint:
        raise ValueError("Current dataset manifest fingerprint differs from its raw sidecar")
    if raw_fingerprint.get("preprocessing_schema") != PREPROCESSING_SCHEMA:
        raise ValueError("Current dataset raw fingerprint preprocessing schema is invalid")
    if (
        set(raw_fingerprint)
        != {
            "schema",
            "preprocessing_schema",
            "language_pairs",
            "tokenizer_sha256",
            "preprocessing_options",
            "files",
        }
        or raw_fingerprint.get("schema") != FINGERPRINT_SCHEMA
    ):
        raise ValueError("Current dataset raw fingerprint schema is invalid")
    try:
        normalized_fingerprint = DatasetFingerprint.from_dict(raw_fingerprint).to_dict()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Current dataset raw fingerprint payload is invalid") from error
    if normalized_fingerprint != raw_fingerprint or not _valid_sha256(
        raw_fingerprint.get("tokenizer_sha256")
    ):
        raise ValueError("Current dataset raw fingerprint payload is invalid")

    raw_options = manifest.get("preprocessing_options")
    if not isinstance(raw_options, Mapping):
        raise ValueError("Current dataset preprocessing_options must be an object")
    options = cast(Mapping[str, object], raw_options)
    if raw_fingerprint.get("preprocessing_options") != dict(options):
        raise ValueError("Current dataset preprocessing options differ from its raw fingerprint")
    required_option_fields = {
        "approximate_split",
        "dedup_backend",
        "endpoint_leakage_guard",
        "endpoint_leakage_key",
        "filter_quality",
        "index_dtype",
        "max_tokens_per_side",
        "prevent_target_leakage",
        "quality_policy",
        "record_metadata_fields",
        "record_metadata_format",
        "record_metadata_index_dtype",
        "shard_size",
        "source_only_languages",
        "translation_directions",
        "split_key",
        "synthetic_sampling_weight",
        "test_fraction",
        "train_only_prefixes",
        "validation_fraction",
    }
    actual_option_fields = frozenset(options)
    if actual_option_fields not in {
        frozenset(required_option_fields),
        frozenset((*required_option_fields, "managed_augmentation_prefix")),
    }:
        raise ValueError("Current dataset preprocessing option fields are invalid")
    shard_size, maximum_tokens = _validate_current_preprocessing_options(
        options,
        language_pair_count=len(language_pairs),
    )

    canonical_pairs = [list(pair) for pair in language_pairs]
    canonical_directions = [list(direction) for direction in translation_directions]
    canonical_languages = list(languages)
    canonical_source_only = list(source_only_languages)
    if manifest.get("language_pairs") != canonical_pairs:
        raise ValueError("Current dataset manifest language_pairs are not canonical")
    if manifest.get("language_pair") != canonical_pairs[0]:
        raise ValueError("Current dataset manifest primary language_pair is invalid")
    if manifest.get("translation_directions") != canonical_directions:
        raise ValueError("Current dataset manifest translation_directions are not canonical")
    if manifest.get("languages") != canonical_languages:
        raise ValueError("Current dataset manifest languages are not canonical")
    if manifest.get("language_to_id") != {
        language: index for index, language in enumerate(languages)
    }:
        raise ValueError("Current dataset manifest language_to_id is invalid")
    if manifest.get("source_only_languages") != canonical_source_only:
        raise ValueError("Current dataset manifest source_only_languages are not canonical")
    if raw_fingerprint.get("language_pairs") != canonical_pairs:
        raise ValueError("Current dataset raw fingerprint language_pairs are invalid")
    fingerprint_files = raw_fingerprint.get("files")
    if not isinstance(fingerprint_files, Mapping):
        raise ValueError("Current dataset raw fingerprint files are invalid")
    for raw_name, raw_identity in fingerprint_files.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or not isinstance(raw_identity, Mapping)
            or set(raw_identity) != {"size", "sha256"}
        ):
            raise ValueError("Current dataset raw fingerprint file identity is invalid")
        raw_size = raw_identity.get("size")
        if (
            isinstance(raw_size, bool)
            or not isinstance(raw_size, int)
            or raw_size < 0
            or not _valid_sha256(raw_identity.get("sha256"))
        ):
            raise ValueError("Current dataset raw fingerprint file identity is invalid")

    raw_prefixes = manifest.get("train_only_prefixes")
    if not isinstance(raw_prefixes, list) or not all(
        isinstance(prefix, str) for prefix in raw_prefixes
    ):
        raise ValueError("Current dataset train_only_prefixes must be a string list")
    normalized_prefixes = normalize_synthetic_prefixes(cast(list[str], raw_prefixes))
    if tuple(raw_prefixes) != normalized_prefixes:
        raise ValueError("Current dataset train_only_prefixes are not normalized")

    expected_index_dtype = _manifest_dtype(INDEX_DTYPE)
    expected_metadata_dtype = _manifest_dtype(RECORD_METADATA_INDEX_DTYPE)
    if manifest.get("storage_sides") != ["src", "tgt"]:
        raise ValueError("Current dataset storage-side contract is invalid")
    if manifest.get("index_dtype") != expected_index_dtype:
        raise ValueError("Current dataset manifest index_dtype is invalid")
    if options.get("index_dtype") != expected_index_dtype:
        raise ValueError("Current dataset preprocessing index_dtype is invalid")
    expected_metadata = {
        "format": RECORD_METADATA_FORMAT,
        "fields": list(RECORD_METADATA_FIELDS),
        "optional": True,
        "index_suffix": RECORD_METADATA_INDEX_SUFFIX,
        "data_suffix": RECORD_METADATA_DATA_SUFFIX,
        "index_dtype": expected_metadata_dtype,
    }
    if manifest.get("record_metadata") != expected_metadata:
        raise ValueError("Current dataset record-metadata contract is invalid")
    if options.get("record_metadata_format") != RECORD_METADATA_FORMAT:
        raise ValueError("Current dataset preprocessing record-metadata format is invalid")
    if options.get("record_metadata_fields") != list(RECORD_METADATA_FIELDS):
        raise ValueError("Current dataset preprocessing record-metadata fields are invalid")
    if options.get("record_metadata_index_dtype") != expected_metadata_dtype:
        raise ValueError("Current dataset preprocessing record-metadata dtype is invalid")
    if options.get("translation_directions") != canonical_directions:
        raise ValueError("Current dataset preprocessing translation graph is invalid")
    if options.get("source_only_languages") != canonical_source_only:
        raise ValueError("Current dataset preprocessing source-only policy is invalid")
    if options.get("train_only_prefixes") != list(normalized_prefixes):
        raise ValueError("Current dataset preprocessing synthetic prefixes are invalid")
    managed_prefix = options.get("managed_augmentation_prefix")
    if managed_prefix is not None and (
        not isinstance(managed_prefix, str) or managed_prefix not in normalized_prefixes
    ):
        raise ValueError("Current dataset managed augmentation prefix is invalid")
    sampling_weight = options.get("synthetic_sampling_weight")
    if (
        isinstance(sampling_weight, bool)
        or not isinstance(sampling_weight, (int, float))
        or not 0.0 <= float(sampling_weight) <= 1.0
    ):
        raise ValueError("Current dataset synthetic sampling weight is invalid")
    expected_synthetic_policy = {
        "record_field": "synthetic",
        "train_only": True,
        "sampling_weight": options.get("synthetic_sampling_weight"),
        "prefixes": list(normalized_prefixes),
    }
    if manifest.get("synthetic_policy") != expected_synthetic_policy:
        raise ValueError("Current dataset synthetic policy contradicts preprocessing options")
    if manifest.get("atomic_build") is not True:
        raise ValueError("Current dataset is not marked as an atomic generation")

    direct_option_fields = {
        "approximate_split": "approximate_split",
        "dedup_backend": "dedup_backend",
        "endpoint_leakage_key": "endpoint_leakage_key",
        "shard_size": "shard_size",
        "test_fraction": "test_fraction",
        "validation_fraction": "validation_fraction",
    }
    for manifest_name, option_name in direct_option_fields.items():
        manifest_value = manifest.get(manifest_name)
        option_value = options.get(option_name)
        if manifest_value != option_value or type(manifest_value) is not type(option_value):
            raise ValueError(
                f"Current dataset manifest {manifest_name} contradicts preprocessing options"
            )
    if manifest.get("quality_filter_enabled") is not options.get("filter_quality"):
        raise ValueError("Current dataset quality filter contradicts preprocessing options")
    if manifest.get("quality_policy") != options.get("quality_policy") or _validated_quality_policy(
        manifest.get("quality_policy")
    ) != _validated_quality_policy(options.get("quality_policy")):
        raise ValueError("Current dataset quality policy contradicts preprocessing options")
    if manifest.get("target_leakage_guard_enabled") is not options.get("prevent_target_leakage"):
        raise ValueError("Current dataset leakage guard contradicts preprocessing options")
    if manifest.get("target_leakage_guard") != options.get("endpoint_leakage_guard"):
        raise ValueError("Current dataset leakage guard schema contradicts preprocessing options")
    if manifest.get("split_key") != options.get("split_key"):
        raise ValueError("Current dataset split key contradicts preprocessing options")

    stats_schema = prepare_stats_schema_from_manifest(
        manifest,
        role="Current dataset manifest",
    )
    stats = _validated_prepare_stats(
        manifest.get("stats"),
        stats_schema=stats_schema,
        role="Current dataset manifest",
    )
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Current dataset manifest sources must be a non-empty list")
    source_stats: list[dict[str, int]] = []
    source_paths: list[str] = []
    source_synthetic_files: list[bool] = []
    accumulated = {name: 0 for name in stats}
    for source_id, raw_source in enumerate(cast(list[object], raw_sources)):
        if not isinstance(raw_source, Mapping):
            raise ValueError(f"Current dataset manifest source {source_id} must be an object")
        source = cast(Mapping[object, object], raw_source)
        if set(source) != {
            "id",
            "name",
            "path",
            "synthetic_file",
            "stats",
            "mean_quality_score",
        }:
            raise ValueError(f"Current dataset manifest source {source_id} fields are invalid")
        raw_path = source.get("path")
        if not isinstance(raw_path, str):
            raise ValueError(f"Current dataset manifest source {source_id} path is invalid")
        synthetic_file = source.get("synthetic_file")
        if (
            source.get("id") != source_id
            or source.get("name") != Path(raw_path).name
            or not isinstance(synthetic_file, bool)
            or synthetic_file != synthetic_path(raw_path, normalized_prefixes)
        ):
            raise ValueError(f"Current dataset manifest source {source_id} identity is invalid")
        normalized_stats = _validated_prepare_stats(
            source.get("stats"),
            stats_schema=stats_schema,
            role=f"Current dataset manifest source {source_id}",
        )
        expected_mean = normalized_stats["quality_score_sum"] / max(
            normalized_stats["valid_pairs"], 1
        )
        raw_mean = source.get("mean_quality_score")
        if (
            isinstance(raw_mean, bool)
            or not isinstance(raw_mean, (int, float))
            or not math.isclose(float(raw_mean), expected_mean, rel_tol=1e-12, abs_tol=1e-15)
        ):
            raise ValueError(f"Current dataset manifest source {source_id} mean score is invalid")
        for name, count in normalized_stats.items():
            accumulated[name] += count
        source_stats.append(normalized_stats)
        source_paths.append(raw_path)
        source_synthetic_files.append(synthetic_file)
    if accumulated != stats:
        raise ValueError("Current dataset per-source stats do not add up to total stats")
    if manifest.get("inputs") != source_paths:
        raise ValueError("Current dataset input paths differ from its source identities")
    source_names = [Path(path).name for path in source_paths]
    if len(source_names) != len(set(source_names)) or set(
        cast(Mapping[object, object], fingerprint_files)
    ) != set(source_names):
        raise ValueError("Current dataset raw fingerprint files differ from its sources")
    if not isinstance(manifest.get("tokenizer_model"), str) or not manifest.get("tokenizer_model"):
        raise ValueError("Current dataset tokenizer_model is invalid")
    expected_mean = stats["quality_score_sum"] / max(stats["valid_pairs"], 1)
    raw_mean = manifest.get("mean_quality_score")
    if (
        isinstance(raw_mean, bool)
        or not isinstance(raw_mean, (int, float))
        or not math.isclose(float(raw_mean), expected_mean, rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise ValueError("Current dataset manifest mean quality score is invalid")

    inventory_digest = validate_dataset_artifact_inventory(dataset_root, manifest)
    if inventory_digest is None:
        raise ValueError("Current dataset artifact inventory did not produce an identity")
    completion = _read_indexed_json_object(
        dataset_root / PREPARE_COMPLETION_FILENAME,
        role="completion marker",
    )
    expected_completion = {
        "schema": PREPARE_COMPLETION_SCHEMA,
        "manifest_sha256": file_sha256(dataset_root / "manifest.json"),
        "raw_fingerprint_sha256": file_sha256(dataset_root / RAW_FINGERPRINT_FILENAME),
        "artifact_inventory_sha256": hashlib.sha256(
            _canonical_json_bytes(manifest.get("artifact_inventory"))
        ).hexdigest(),
    }
    if completion != expected_completion:
        raise ValueError("Current dataset completion marker does not authenticate its generation")
    return (
        stats,
        tuple(source_stats),
        tuple(source_synthetic_files),
        shard_size,
        maximum_tokens,
        inventory_digest,
    )


def _validate_current_indexed_payload(
    dataset_root: Path,
    *,
    stats: Mapping[str, int],
    source_stats: Sequence[Mapping[str, int]],
    source_synthetic_files: Sequence[bool],
    languages: tuple[str, ...],
    translation_directions: tuple[tuple[str, str], ...],
    source_only_languages: tuple[str, ...],
    shard_size: int,
    maximum_tokens_per_side: int,
    vocab_size: int,
) -> None:
    """Match prepare's exact INDEX_DTYPE, split, and metadata-sidecar checks."""

    source_count = len(source_stats)
    if len(source_synthetic_files) != source_count:
        raise ValueError("Current dataset source synthetic identities are incomplete")
    language_to_id = {language: index for index, language in enumerate(languages)}
    direction_set = frozenset(translation_directions)
    allowed_pairs = {
        (language_to_id[source], language_to_id[target])
        for source, target in translation_directions
    }
    source_only = frozenset(source_only_languages)
    source_rows = np.zeros(source_count, dtype=np.int64)
    source_synthetic = np.zeros(source_count, dtype=np.int64)
    source_forward_only = np.zeros(source_count, dtype=np.int64)
    source_quality = np.zeros(source_count, dtype=np.int64)
    source_src_tokens = np.zeros(source_count, dtype=np.int64)
    source_tgt_tokens = np.zeros(source_count, dtype=np.int64)
    source_split_rows = {
        split: np.zeros(source_count, dtype=np.int64) for split in ("train", "validation", "test")
    }
    split_rows: dict[str, int] = {}

    for split in ("train", "validation", "test"):
        split_root = dataset_root / split
        expected_artifacts: set[str] = set()
        split_total = 0
        for index_path in sorted(split_root.glob("*.idx.npy")):
            try:
                index = np.load(index_path, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ValueError(f"Cannot read current dataset index: {index_path}") from error
            if index.ndim != 1 or index.dtype != INDEX_DTYPE:
                raise ValueError(f"Current dataset index dtype is invalid: {index_path}")
            if len(index) > shard_size:
                raise ValueError(
                    f"Current dataset shard exceeds configured shard_size: {index_path}"
                )
            prefix = index_path.name.removesuffix(".idx.npy")
            src_path = split_root / f"{prefix}.src.bin"
            tgt_path = split_root / f"{prefix}.tgt.bin"
            src_lengths = np.asarray(index["src_length"], dtype=np.uint64)
            tgt_lengths = np.asarray(index["tgt_length"], dtype=np.uint64)
            if (src_lengths.size and int(src_lengths.max()) > maximum_tokens_per_side) or (
                tgt_lengths.size and int(tgt_lengths.max()) > maximum_tokens_per_side
            ):
                raise ValueError(
                    f"Current dataset row exceeds configured max_tokens_per_side: {index_path}"
                )
            src_store = _open_indexed_token_store(src_path, index["src_offset"], src_lengths)
            tgt_store = _open_indexed_token_store(tgt_path, index["tgt_offset"], tgt_lengths)
            for store, token_path in ((src_store, src_path), (tgt_store, tgt_path)):
                if store.size and int(store.max(initial=0)) >= vocab_size:
                    raise ValueError(
                        f"Current dataset token id exceeds tokenizer vocabulary: {token_path}"
                    )

            source_ids = np.asarray(index["source_id"], dtype=np.int64)
            src_language_ids = np.asarray(index["src_language_id"], dtype=np.int64)
            tgt_language_ids = np.asarray(index["tgt_language_id"], dtype=np.int64)
            synthetic = np.asarray(index["synthetic"], dtype=np.int64)
            forward_only = np.asarray(index["forward_only"], dtype=np.int64)
            quality = np.asarray(index["quality_score"], dtype=np.int64)
            if source_ids.size and (
                int(source_ids.min()) < 0 or int(source_ids.max()) >= source_count
            ):
                raise ValueError(f"Current dataset source_id is outside manifest: {index_path}")
            for values, name in (
                (src_language_ids, "src_language_id"),
                (tgt_language_ids, "tgt_language_id"),
            ):
                if values.size and (int(values.min()) < 0 or int(values.max()) >= len(languages)):
                    raise ValueError(f"Current dataset {name} is outside manifest: {index_path}")
            if not bool(np.isin(synthetic, (0, 1)).all()) or not bool(
                np.isin(forward_only, (0, 1)).all()
            ):
                raise ValueError(f"Current dataset boolean flags are invalid: {index_path}")
            if split != "train" and bool(np.count_nonzero(synthetic)):
                raise ValueError(f"Current dataset synthetic rows must be train-only: {index_path}")
            for source_id, synthetic_file in enumerate(source_synthetic_files):
                if not synthetic_file:
                    continue
                source_mask = source_ids == source_id
                if bool(source_mask.any()) and not bool((synthetic[source_mask] == 1).all()):
                    raise ValueError(
                        "Current dataset synthetic-file source contains a row without the "
                        f"synthetic flag: {index_path} source_id={source_id}"
                    )
            if quality.size and (int(quality.min()) < 0 or int(quality.max()) > 100):
                raise ValueError(f"Current dataset quality score is invalid: {index_path}")
            if not bool(
                np.isin(np.asarray(index["src_register"], dtype=np.int64), (0, 1, 2, 3)).all()
            ) or not bool(
                np.isin(np.asarray(index["tgt_register"], dtype=np.int64), (0, 1, 2, 3)).all()
            ):
                raise ValueError(f"Current dataset register is invalid: {index_path}")

            scoped_rows: set[int] = set()
            for row_id, (source_id, target_id, one_way) in enumerate(
                zip(src_language_ids, tgt_language_ids, forward_only, strict=True)
            ):
                pair = (int(source_id), int(target_id))
                if pair not in allowed_pairs:
                    raise ValueError(
                        f"Current dataset language pair is not configured: {index_path}"
                    )
                source_language = languages[pair[0]]
                target_language = languages[pair[1]]
                reverse_trained = (target_language, source_language) in direction_set
                if (not bool(one_way) and not reverse_trained) or target_language in source_only:
                    raise ValueError(
                        f"Current dataset translation direction is invalid: {index_path}"
                    )
                if bool(one_way) and reverse_trained:
                    scoped_rows.add(row_id)

            row_count = len(index)
            split_total += row_count
            counts = np.bincount(source_ids, minlength=source_count)[:source_count]
            source_rows += counts
            source_split_rows[split] += counts
            source_synthetic += np.bincount(source_ids, weights=synthetic, minlength=source_count)[
                :source_count
            ].astype(np.int64)
            source_forward_only += np.bincount(
                source_ids, weights=forward_only, minlength=source_count
            )[:source_count].astype(np.int64)
            source_quality += np.bincount(source_ids, weights=quality, minlength=source_count)[
                :source_count
            ].astype(np.int64)
            source_src_tokens += np.bincount(
                source_ids, weights=src_lengths, minlength=source_count
            )[:source_count].astype(np.int64)
            source_tgt_tokens += np.bincount(
                source_ids, weights=tgt_lengths, minlength=source_count
            )[:source_count].astype(np.int64)

            metadata_index_path = split_root / f"{prefix}{RECORD_METADATA_INDEX_SUFFIX}"
            metadata_data_path = split_root / f"{prefix}{RECORD_METADATA_DATA_SUFFIX}"
            metadata_present = metadata_index_path.exists() or metadata_data_path.exists()
            expected_artifacts.update({index_path.name, src_path.name, tgt_path.name})
            if metadata_present:
                if not metadata_index_path.is_file() or not metadata_data_path.is_file():
                    raise ValueError(f"Incomplete record metadata sidecar for {index_path}")
                metadata_index = np.load(metadata_index_path, allow_pickle=False)
                if (
                    metadata_index.ndim != 1
                    or metadata_index.dtype != RECORD_METADATA_INDEX_DTYPE
                    or len(metadata_index) != len(index)
                ):
                    raise ValueError(
                        f"Current dataset record metadata index is invalid: {metadata_index_path}"
                    )
                metadata_offsets = np.asarray(metadata_index["offset"], dtype=np.uint64)
                metadata_lengths = np.asarray(metadata_index["length"], dtype=np.uint64)
                expected_offsets = np.cumsum(
                    np.concatenate((np.zeros(1, dtype=np.uint64), metadata_lengths[:-1])),
                    dtype=np.uint64,
                )
                if not np.array_equal(metadata_offsets, expected_offsets):
                    raise ValueError(
                        "Current dataset record metadata offsets are not contiguous: "
                        f"{metadata_index_path}"
                    )
                if int(metadata_lengths.sum(dtype=np.uint64)) != metadata_data_path.stat().st_size:
                    raise ValueError(
                        "Current dataset record metadata offsets exceed payload: "
                        f"{metadata_index_path}"
                    )
                metadata_store = (
                    np.memmap(metadata_data_path, dtype=np.uint8, mode="r")
                    if metadata_data_path.stat().st_size
                    else np.empty(0, dtype=np.uint8)
                )
                for row_id, metadata_row in enumerate(metadata_index):
                    offset = int(metadata_row["offset"])
                    length = int(metadata_row["length"])
                    payload = np.asarray(
                        metadata_store[offset : offset + length],
                        dtype=np.uint8,
                    ).tobytes()
                    try:
                        metadata = decode_record_metadata(payload)
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                        raise ValueError(
                            f"Current dataset record metadata is invalid: {index_path} row={row_id}"
                        ) from error
                    stored_direction = (
                        languages[int(src_language_ids[row_id])],
                        languages[int(tgt_language_ids[row_id])],
                    )
                    try:
                        row_direction = resolve_record_training_direction(
                            metadata,
                            stored_direction,
                            direction_set,
                        )
                    except ValueError as error:
                        raise ValueError(
                            "Current dataset record metadata direction is invalid: "
                            f"{index_path} row={row_id}"
                        ) from error
                    if row_direction is not None:
                        if metadata.get("training_direction") != list(row_direction):
                            raise ValueError(
                                "Current dataset record metadata direction is not canonical: "
                                f"{index_path} row={row_id}"
                            )
                        if row_direction != stored_direction or not bool(forward_only[row_id]):
                            raise ValueError(
                                "Current dataset record metadata direction contradicts its "
                                f"index flags: {index_path} row={row_id}"
                            )
                    elif row_id in scoped_rows:
                        raise ValueError(
                            "Current dataset row-scoped direction lacks matching metadata: "
                            f"{index_path} row={row_id}"
                        )
                del metadata_store
                expected_artifacts.update({metadata_index_path.name, metadata_data_path.name})
            elif scoped_rows:
                raise ValueError(
                    f"Current dataset row-scoped directions require metadata: {index_path}"
                )
        actual_artifacts = {path.name for path in split_root.iterdir()}
        if actual_artifacts != expected_artifacts:
            raise ValueError(
                f"Current dataset split artifacts are incomplete or unexpected: {split}; "
                f"missing={sorted(expected_artifacts - actual_artifacts)}, "
                f"unexpected={sorted(actual_artifacts - expected_artifacts)}"
            )
        split_rows[split] = split_total

    if split_rows != {
        "train": stats["train"],
        "validation": stats["validation"],
        "test": stats["test"],
    }:
        raise ValueError("Current dataset manifest split counts differ from indexed payload rows")
    if stats["valid_pairs"] != stats["train"] + stats["validation"] + stats["test"]:
        raise ValueError("Current dataset valid_pairs differs from its split counts")
    for source_id, expected in enumerate(source_stats):
        derived = {
            "valid_pairs": int(source_rows[source_id]),
            "train": int(source_split_rows["train"][source_id]),
            "validation": int(source_split_rows["validation"][source_id]),
            "test": int(source_split_rows["test"][source_id]),
            "synthetic_pairs": int(source_synthetic[source_id]),
            "forward_only_pairs": int(source_forward_only[source_id]),
            "quality_score_sum": int(source_quality[source_id]),
            "src_tokens": int(source_src_tokens[source_id]),
            "tgt_tokens": int(source_tgt_tokens[source_id]),
        }
        for name, value in derived.items():
            if expected[name] != value:
                raise ValueError(
                    f"Current dataset source {source_id} {name} differs from indexed payload"
                )


def _row_blocks(lengths: np.ndarray, maximum_tokens: int = 4_000_000) -> Iterator[slice]:
    """Yield row slices whose expanded token metadata stays memory-bounded."""

    if len(lengths) == 0:
        return
    cumulative = np.cumsum(lengths, dtype=np.uint64)
    start = 0
    while start < len(lengths):
        before = 0 if start == 0 else int(cumulative[start - 1])
        end = int(np.searchsorted(cumulative, before + maximum_tokens, side="right"))
        end = max(start + 1, end)
        yield slice(start, min(end, len(lengths)))
        start = end


def _open_indexed_token_store(
    path: Path,
    offsets: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Indexed token shard not found: {path}")
    byte_size = path.stat().st_size
    if byte_size % np.dtype(np.uint32).itemsize:
        raise ValueError(f"Indexed token shard has a partial uint32 value: {path}")
    token_count = byte_size // np.dtype(np.uint32).itemsize
    expected_offsets = np.cumsum(
        np.concatenate((np.zeros(1, dtype=np.uint64), lengths[:-1].astype(np.uint64))),
        dtype=np.uint64,
    )
    if not np.array_equal(offsets.astype(np.uint64), expected_offsets):
        raise ValueError(f"Indexed token offsets are not contiguous in {path}")
    expected_tokens = int(lengths.sum(dtype=np.uint64))
    if token_count != expected_tokens:
        raise ValueError(
            f"Indexed token count does not match its index in {path}: "
            f"{token_count} != {expected_tokens}"
        )
    if token_count == 0:
        return np.empty(0, dtype=np.uint32)
    return np.memmap(path, dtype=np.uint32, mode="r")


def _accumulate_indexed_side(
    store: np.ndarray,
    lengths: np.ndarray,
    language_ids: np.ndarray,
    target_rows: np.ndarray,
    physical_counts: list[np.ndarray],
    target_counts: list[np.ndarray],
    *,
    vocab_size: int,
) -> None:
    """Count a stored side once physically and when it is a decoder target."""

    token_offset = 0
    for block in _row_blocks(lengths):
        block_lengths = lengths[block].astype(np.int64, copy=False)
        block_size = int(block_lengths.sum(dtype=np.int64))
        block_tokens = np.asarray(store[token_offset : token_offset + block_size])
        token_offset += block_size
        if block_size == 0:
            continue
        maximum_id = int(block_tokens.max(initial=0))
        if maximum_id >= vocab_size:
            raise ValueError(
                f"Indexed token id {maximum_id} exceeds tokenizer vocabulary size {vocab_size}"
            )
        row_languages = language_ids[block]
        row_targets = target_rows[block]
        unique_languages = np.unique(row_languages)
        if len(unique_languages) == 1:
            language_id = int(unique_languages[0])
            all_counts = np.bincount(
                block_tokens.astype(np.int64, copy=False),
                minlength=vocab_size,
            ).astype(np.uint64, copy=False)
            physical_counts[language_id] += all_counts
            if bool(row_targets.all()):
                target_counts[language_id] += all_counts
            elif bool(row_targets.any()):
                token_targets = np.repeat(row_targets, block_lengths)
                decoder_tokens = block_tokens[token_targets].astype(np.int64, copy=False)
                target_counts[language_id] += np.bincount(
                    decoder_tokens,
                    minlength=vocab_size,
                ).astype(np.uint64, copy=False)
            continue

        token_languages = np.repeat(row_languages, block_lengths)
        token_targets = None if bool(row_targets.all()) else np.repeat(row_targets, block_lengths)
        for language_id in unique_languages:
            language_mask = token_languages == language_id
            language_tokens = block_tokens[language_mask].astype(np.int64, copy=False)
            all_counts = np.bincount(
                language_tokens,
                minlength=vocab_size,
            ).astype(np.uint64, copy=False)
            physical_counts[int(language_id)] += all_counts
            if token_targets is None:
                target_counts[int(language_id)] += all_counts
                continue
            decoder_mask = language_mask & token_targets
            if bool(decoder_mask.any()):
                decoder_tokens = block_tokens[decoder_mask].astype(np.int64, copy=False)
                target_counts[int(language_id)] += np.bincount(
                    decoder_tokens,
                    minlength=vocab_size,
                ).astype(np.uint64, copy=False)


def _accumulate_shared_target_sides(
    source_store: np.ndarray,
    target_store: np.ndarray,
    source_lengths: np.ndarray,
    target_lengths: np.ndarray,
    target_shared: np.ndarray,
    source_languages: np.ndarray,
    target_languages: np.ndarray,
    source_target_rows: np.ndarray,
    target_target_rows: np.ndarray,
    physical_counts: list[np.ndarray],
    target_counts: list[np.ndarray],
    *,
    vocab_size: int,
) -> None:
    """Audit logical source/target rows while keeping aliased targets memory-bounded."""

    source_token_offset = 0
    stored_target_offset = 0
    logical_sizes = source_lengths.astype(np.uint64) + target_lengths.astype(np.uint64)
    for block in _row_blocks(logical_sizes):
        block_source_lengths = source_lengths[block].astype(np.int64, copy=False)
        block_target_lengths = target_lengths[block].astype(np.int64, copy=False)
        block_shared = target_shared[block].astype(np.bool_, copy=False)
        source_size = int(block_source_lengths.sum(dtype=np.int64))
        block_source = np.asarray(
            source_store[source_token_offset : source_token_offset + source_size]
        )
        source_token_offset += source_size
        source_row_offsets = np.cumsum(
            np.concatenate((np.zeros(1, dtype=np.int64), block_source_lengths[:-1])),
            dtype=np.int64,
        )
        target_chunks: list[np.ndarray] = []
        for row_offset, source_length, target_length, shared in zip(
            source_row_offsets,
            block_source_lengths,
            block_target_lengths,
            block_shared,
            strict=True,
        ):
            if bool(shared):
                if int(source_length) != int(target_length):
                    raise ValueError("Shared foundation target length differs from its source")
                target_chunks.append(
                    block_source[int(row_offset) : int(row_offset + source_length)]
                )
                continue
            target_chunks.append(
                np.asarray(
                    target_store[stored_target_offset : stored_target_offset + int(target_length)]
                )
            )
            stored_target_offset += int(target_length)
        logical_target = (
            np.concatenate(target_chunks) if target_chunks else np.empty(0, dtype=np.uint32)
        )
        _accumulate_indexed_side(
            block_source,
            block_source_lengths,
            source_languages[block],
            source_target_rows[block],
            physical_counts,
            target_counts,
            vocab_size=vocab_size,
        )
        _accumulate_indexed_side(
            logical_target,
            block_target_lengths,
            target_languages[block],
            target_target_rows[block],
            physical_counts,
            target_counts,
            vocab_size=vocab_size,
        )


def _add_direction_totals(
    totals: dict[str, Counter[str]],
    source_languages: np.ndarray,
    target_languages: np.ndarray,
    source_lengths: np.ndarray,
    target_lengths: np.ndarray,
    enabled: np.ndarray,
    languages: Sequence[str],
) -> int:
    examples = 0
    enabled_indices = np.flatnonzero(enabled)
    if not enabled_indices.size:
        return examples
    pair_keys = source_languages[enabled_indices].astype(np.uint64) * np.uint64(
        len(languages)
    ) + target_languages[enabled_indices].astype(np.uint64)
    for pair_key in np.unique(pair_keys):
        selected = enabled_indices[pair_keys == pair_key]
        source_id, target_id = divmod(int(pair_key), len(languages))
        direction = _direction_label(languages[source_id], languages[target_id])
        direction_totals = totals.setdefault(direction, Counter())
        direction_totals["examples"] += len(selected)
        direction_totals["source_tokens"] += int(source_lengths[selected].sum(dtype=np.uint64))
        direction_totals["target_tokens"] += int(target_lengths[selected].sum(dtype=np.uint64))
        examples += len(selected)
    return examples


def _rare_piece_examples(
    tokenizer: SionTokenizer,
    counts: np.ndarray,
    eligible: np.ndarray,
    *,
    maximum: int,
    include_unused: bool,
) -> list[dict[str, int | str]]:
    candidates = np.flatnonzero(eligible & ((counts >= 0) if include_unused else (counts > 0)))
    ordered = candidates[np.lexsort((candidates, counts[candidates]))]
    return [
        {
            "id": int(token_id),
            "piece": tokenizer.processor.id_to_piece(int(token_id)),
            "count": int(counts[token_id]),
        }
        for token_id in ordered[:maximum]
    ]


def audit_monolingual_token_exposure(
    discovery: MonolingualDiscovery,
    tokenizer_model: str | Path,
    *,
    minimum_characters: int = 8,
    maximum_characters: int = 4000,
    rare_threshold: int = 25,
    max_piece_examples: int = 50,
    max_lines_per_language: int = 0,
) -> dict[str, Any]:
    """Count decoder-target exposure contributed by the foundation corpus.

    Reconstruction targets contain the complete clean monolingual sentence, so
    every corpus token becomes a decoder target. This stage can therefore train
    output embeddings for pieces that never appear as parallel targets.

    A parallel-only audit can both overstate risk for pieces covered by
    foundation training and miss pieces introduced by monolingual vocabulary
    pressure but absent from translation targets.

    ``max_lines_per_language=0`` scans the complete corpus. A positive value is
    a deterministic prefix sample recorded as such in the report; it is useful
    for preflight but cannot establish vocabulary safety.
    """

    if minimum_characters < 1:
        raise ValueError("minimum_characters must be positive")
    if maximum_characters <= minimum_characters:
        raise ValueError("maximum_characters must be greater than minimum_characters")
    if rare_threshold < 1:
        raise ValueError("rare_threshold must be positive")
    if max_lines_per_language < 0:
        raise ValueError("max_lines_per_language must be non-negative")
    if not discovery.sources:
        raise ValueError(f"The monolingual corpus has no readable files: {discovery.root}")

    tokenizer = SionTokenizer(tokenizer_model)
    vocab_size = len(tokenizer)
    counts = {language: np.zeros(vocab_size, dtype=np.uint64) for language in discovery.languages}
    accepted = Counter()
    dropped = Counter()

    for language in discovery.languages:
        target = counts[language]
        for path in discovery.paths_for(language):
            if max_lines_per_language and accepted[language] >= max_lines_per_language:
                break
            for text in iter_monolingual_lines(path):
                if max_lines_per_language and accepted[language] >= max_lines_per_language:
                    break
                normalized = canonical_text(text)
                if len(normalized) < minimum_characters:
                    dropped[f"{language}:too_short"] += 1
                    continue
                if len(normalized) > maximum_characters:
                    dropped[f"{language}:too_long"] += 1
                    continue
                token_ids = tokenizer.encode(normalized)
                if token_ids:
                    target += np.bincount(token_ids, minlength=vocab_size).astype(
                        np.uint64, copy=False
                    )
                accepted[language] += 1

    combined = np.zeros(vocab_size, dtype=np.uint64)
    for value in counts.values():
        combined += value
    eligible = np.array(
        [
            not _piece_is_special(tokenizer.processor.id_to_piece(index))
            for index in range(vocab_size)
        ]
    )
    return {
        "scan": "monolingual-corpus",
        "root": str(discovery.root),
        "complete_scan": max_lines_per_language == 0,
        "max_lines_per_language": max_lines_per_language,
        "rare_threshold": rare_threshold,
        "vocab_size": vocab_size,
        "languages": list(discovery.languages),
        "accepted_lines": dict(accepted),
        "dropped_lines": dict(dropped),
        "decoder_target_totals": _frequency_summary(combined, eligible),
        "per_language": {
            language: _frequency_summary(counts[language], eligible) for language in counts
        },
        "lowest_target_exposure": _rare_piece_examples(
            tokenizer,
            combined,
            eligible,
            maximum=max_piece_examples,
            include_unused=True,
        ),
        "counts": combined,
    }


def combine_target_exposure(
    parallel_counts: np.ndarray,
    monolingual_counts: np.ndarray,
    tokenizer_model: str | Path,
    *,
    rare_threshold: int = 25,
    max_piece_examples: int = 50,
) -> dict[str, Any]:
    """Combine both stages before deciding whether a piece receives training.

    When foundation training runs first, output embeddings receive signal from
    both stages. Judging either stage alone can retain unsafe pieces or remove
    pieces that are adequately trained by the other stage.
    """

    if parallel_counts.shape != monolingual_counts.shape:
        raise ValueError("count vectors must describe the same vocabulary")
    tokenizer = SionTokenizer(tokenizer_model)
    vocab_size = len(tokenizer)
    if parallel_counts.shape[0] != vocab_size:
        raise ValueError("count vectors do not match the tokenizer vocabulary size")
    eligible = np.array(
        [
            not _piece_is_special(tokenizer.processor.id_to_piece(index))
            for index in range(vocab_size)
        ]
    )
    combined = parallel_counts.astype(np.uint64) + monolingual_counts.astype(np.uint64)
    rescued = int(
        np.count_nonzero(
            eligible & (parallel_counts < rare_threshold) & (combined >= rare_threshold)
        )
    )
    still_rare = int(np.count_nonzero(eligible & (combined < rare_threshold)))
    return {
        "scan": "combined-stages",
        "rare_threshold": rare_threshold,
        "totals": _frequency_summary(combined, eligible),
        # Count pieces whose insufficient parallel exposure is repaired by the
        # foundation stage.
        "rescued_by_foundation": rescued,
        "still_below_threshold": still_rare,
        "lowest_target_exposure": _rare_piece_examples(
            tokenizer,
            combined,
            eligible,
            maximum=max_piece_examples,
            include_unused=True,
        ),
    }


def audit_token_exposure(
    input_patterns: Sequence[str],
    tokenizer_model: str | Path,
    *,
    translation_directions: Sequence[Sequence[str]],
    language_pair: Sequence[str] | None = None,
    language_pairs: Sequence[Sequence[str]] | None = None,
    source_only_languages: Sequence[str] | None = None,
    bidirectional: bool | None = None,
    train_only_prefixes: Sequence[str] = (),
    max_physical_pairs: int = 0,
    rare_threshold: int = 25,
    max_piece_examples: int = 50,
    filter_quality: bool = True,
    return_counts: bool = False,
) -> dict[str, Any]:
    """Audit target-token exposure without materializing an indexed dataset.

    ``return_counts`` attaches the raw decoder-target count vector under
    ``global_target_counts`` so a caller can combine it with another stage's
    exposure. It is off by default because the value is a vocabulary-sized
    NumPy array and the report is otherwise JSON-serializable.

    ``max_physical_pairs=0`` performs an exact full scan. A positive value is a
    deterministic prefix sample and is labelled as such in the report; it is
    useful for a quick preflight, not for declaring a vocabulary safe.

    ``translation_directions`` is the authenticated ordered training graph.
    A row-scoped ``training_direction`` narrows one physical record to that
    edge; ordinary parallel rows without the annotation expand only over the
    configured edges for their physical pair.

    ``source_only_languages`` and ``bidirectional`` are migration-only policy
    assertions. They never manufacture an omitted graph; when supplied, their
    legacy-derived graph must exactly match ``translation_directions``.
    """

    if max_physical_pairs < 0:
        raise ValueError("max_physical_pairs must be non-negative")
    if rare_threshold < 1:
        raise ValueError("rare_threshold must be positive")
    if max_piece_examples < 0:
        raise ValueError("max_piece_examples must be non-negative")

    paths = expand_inputs(input_patterns)
    if not paths:
        raise FileNotFoundError(f"No JSONL files matched: {list(input_patterns)}")
    if language_pair is not None and language_pairs is not None:
        raise ValueError("language_pair and language_pairs are mutually exclusive")
    if language_pairs is not None:
        pairs = _normalize_explicit_language_pairs(
            language_pairs,
            field="language_pairs",
        )
    elif language_pair is not None:
        pairs = normalize_language_pairs(language_pair=language_pair)
    else:
        pairs = normalize_language_pairs(language_pairs=translation_directions)
    legacy_source_only = (
        ()
        if source_only_languages is None
        else canonicalize_language_tags(
            source_only_languages,
            field="source_only_languages",
            reject_duplicates=True,
        )
    )
    directions = normalize_translation_directions(
        pairs,
        translation_directions,
        bidirectional=False,
        source_only_languages=legacy_source_only,
    )
    if source_only_languages is not None or bidirectional is not None:
        compatibility_directions = normalize_translation_directions(
            pairs,
            bidirectional=True if bidirectional is None else bidirectional,
            source_only_languages=legacy_source_only,
        )
        if directions != compatibility_directions:
            raise ValueError(
                "legacy source_only_languages/bidirectional policy contradicts the explicit "
                "translation_directions graph; migrate by removing the legacy arguments"
            )
    normalized_prefixes, direction_required_prefixes = _raw_audit_synthetic_prefixes(
        train_only_prefixes
    )
    direction_set = frozenset(directions)
    directions_by_pair: dict[frozenset[str], tuple[tuple[str, str], ...]] = {}
    for pair in pairs:
        edge = frozenset(pair)
        directions_by_pair[edge] = tuple(
            direction for direction in directions if frozenset(direction) == edge
        )
    languages = languages_from_pairs(pairs)
    target_languages = frozenset(target for _, target in directions)

    tokenizer_path = Path(tokenizer_model)
    tokenizer_sha256 = file_sha256(tokenizer_path)
    tokenizer = SionTokenizer(tokenizer_path)
    missing_tags = sorted(set(languages) - set(tokenizer.languages))
    if missing_tags:
        raise ValueError(f"tokenizer is missing configured language tags: {missing_tags}")

    vocab_size = len(tokenizer)
    physical_counts = {language: np.zeros(vocab_size, dtype=np.uint64) for language in languages}
    target_counts = {language: np.zeros(vocab_size, dtype=np.uint64) for language in languages}
    language_totals: dict[str, Counter[str]] = {language: Counter() for language in languages}
    direction_totals: dict[str, Counter[str]] = {
        _direction_label(source, target): Counter() for source, target in directions
    }
    invalid = Counter()
    physical_pairs = 0
    virtual_examples = 0
    policy = QualityPolicy()

    def add_counts(target: np.ndarray, token_ids: list[int]) -> None:
        if token_ids:
            target += np.bincount(token_ids, minlength=vocab_size).astype(np.uint64, copy=False)

    stop = False
    for path in paths:
        path_is_synthetic = synthetic_path(path, direction_required_prefixes)
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                try:
                    row = json.loads(raw_line.decode("utf-8-sig"))
                except UnicodeDecodeError:
                    invalid["invalid_utf8"] += 1
                    continue
                except json.JSONDecodeError:
                    invalid["invalid_json"] += 1
                    continue
                annotated_row = _annotate_synthetic_scopes(
                    row,
                    inherited=path_is_synthetic,
                )
                expansion = expand_parallel_record(annotated_row, pairs)
                invalid.update(expansion.issues)
                for pair in expansion.pairs:
                    row_is_synthetic = _metadata_is_synthetic(pair.metadata)
                    try:
                        row_direction = resolve_record_training_direction(
                            pair.metadata,
                            (pair.language_a, pair.language_b),
                            direction_set,
                        )
                    except ValueError as error:
                        raise ValueError(f"{path}:{line_number}: {error}") from error
                    if row_is_synthetic and row_direction is None:
                        raise ValueError(
                            f"{path}:{line_number}: synthetic records require an explicit "
                            "training_direction for token exposure auditing"
                        )
                    text_a = canonical_text(pair.text_a)
                    text_b = canonical_text(pair.text_b)
                    if (
                        filter_quality
                        and not assess_pair(
                            text_a,
                            text_b,
                            policy,
                            languages=(pair.language_a, pair.language_b),
                        ).accepted
                    ):
                        invalid["quality_filtered"] += 1
                        continue
                    ids_a = tokenizer.encode(text_a)
                    ids_b = tokenizer.encode(text_b)
                    add_counts(physical_counts[pair.language_a], ids_a)
                    add_counts(physical_counts[pair.language_b], ids_b)
                    physical_pairs += 1

                    content = {
                        pair.language_a: (ids_a, text_a),
                        pair.language_b: (ids_b, text_b),
                    }
                    active_directions = (
                        (row_direction,)
                        if row_direction is not None
                        else directions_by_pair[frozenset((pair.language_a, pair.language_b))]
                    )
                    for src_lang, tgt_lang in active_directions:
                        src_ids, src_text = content[src_lang]
                        tgt_ids, tgt_text = content[tgt_lang]
                        add_counts(target_counts[tgt_lang], tgt_ids)
                        virtual_examples += 1
                        direction = _direction_label(src_lang, tgt_lang)
                        totals = direction_totals.setdefault(direction, Counter())
                        totals["examples"] += 1
                        totals["source_tokens"] += len(src_ids)
                        totals["target_tokens"] += len(tgt_ids)
                        totals["source_characters"] += sum(not c.isspace() for c in src_text)
                        totals["target_characters"] += sum(not c.isspace() for c in tgt_text)
                    for language, ids, text in (
                        (pair.language_a, ids_a, text_a),
                        (pair.language_b, ids_b, text_b),
                    ):
                        totals = language_totals[language]
                        totals["physical_sentences"] += 1
                        totals["physical_tokens"] += len(ids)
                        totals["physical_characters"] += sum(not c.isspace() for c in text)
                        totals["byte_fallback_tokens"] += sum(
                            tokenizer.processor.id_to_piece(token_id).startswith("<0x")
                            for token_id in ids
                        )
                        totals["unknown_tokens"] += sum(
                            token_id == tokenizer.unk_id for token_id in ids
                        )

                    if max_physical_pairs and physical_pairs >= max_physical_pairs:
                        stop = True
                        break
                if stop:
                    break
        if stop:
            break

    special = np.array(
        [_piece_is_special(tokenizer.processor.id_to_piece(i)) for i in range(vocab_size)],
        dtype=np.bool_,
    )
    byte = np.array(
        [tokenizer.processor.id_to_piece(i).startswith("<0x") for i in range(vocab_size)],
        dtype=np.bool_,
    )
    eligible = ~(special | byte)
    global_physical = np.zeros(vocab_size, dtype=np.uint64)
    global_target = np.zeros(vocab_size, dtype=np.uint64)
    for language in languages:
        global_physical += physical_counts[language]
        global_target += target_counts[language]

    language_reports: dict[str, Any] = {}
    for language in languages:
        totals = language_totals[language]
        tokens = totals["physical_tokens"]
        characters = totals["physical_characters"]
        target_enabled = language in target_languages
        target_eligible = (
            eligible & (physical_counts[language] > 0)
            if target_enabled
            else np.zeros(vocab_size, dtype=np.bool_)
        )
        target_summary = _frequency_summary(target_counts[language], target_eligible)
        target_summary["rare_threshold"] = rare_threshold
        target_summary["rare_observed_pieces"] = int(
            np.count_nonzero(
                target_eligible
                & (target_counts[language] > 0)
                & (target_counts[language] < rare_threshold)
            )
        )
        language_reports[language] = {
            **{name: int(value) for name, value in totals.items()},
            "target_enabled": target_enabled,
            "target_tokens": int(target_counts[language].sum(dtype=np.uint64)),
            "tokens_per_character": round(tokens / max(characters, 1), 6),
            "byte_fallback_rate": round(totals["byte_fallback_tokens"] / max(tokens, 1), 8),
            "target_frequency": target_summary,
            "lowest_target_exposure": _rare_piece_examples(
                tokenizer,
                target_counts[language],
                target_eligible,
                maximum=max_piece_examples,
                include_unused=True,
            ),
        }

    direction_report = {}
    for direction, totals in sorted(direction_totals.items()):
        target_tokens = totals["target_tokens"]
        target_characters = totals["target_characters"]
        direction_report[direction] = {
            **{
                name: int(totals[name])
                for name in (
                    "examples",
                    "source_tokens",
                    "target_tokens",
                    "source_characters",
                    "target_characters",
                )
            },
            "target_tokens_per_character": round(target_tokens / max(target_characters, 1), 6),
            "mean_target_tokens": round(target_tokens / max(totals["examples"], 1), 6),
        }

    global_corpus_observed = eligible & (global_physical > 0)
    # The global view deliberately covers the complete learnable vocabulary.
    # Restricting it to pieces present in this corpus would hide the most severe
    # failure mode: an ordinary SentencePiece token that never receives a decoder
    # update at all. Per-language summaries remain corpus-conditioned so Japanese
    # pieces are not misleadingly labelled as unused Korean targets (and vice versa).
    global_summary = _frequency_summary(global_target, eligible)
    global_summary["all_target_tokens"] = int(global_target.sum(dtype=np.uint64))
    global_summary["corpus_observed_pieces"] = int(np.count_nonzero(global_corpus_observed))
    global_summary["rare_threshold"] = rare_threshold
    global_summary["rare_observed_pieces"] = int(
        np.count_nonzero(eligible & (global_target > 0) & (global_target < rare_threshold))
    )
    global_summary["below_threshold_pieces"] = int(
        global_summary["unused_pieces"] + global_summary["rare_observed_pieces"]
    )
    report_counts = {"global_target_counts": global_target} if return_counts else {}
    report = {
        **report_counts,
        "schema": "sion-token-exposure-audit-v2",
        "complete_scan": max_physical_pairs == 0 or not stop,
        "parameters": {
            "tokenizer_model": str(Path(tokenizer_model).resolve()),
            "tokenizer_sha256": tokenizer_sha256,
            "tokenizer_scan_stability": "sha256-reverified-after-scan",
            "language_pairs": [list(pair) for pair in pairs],
            "translation_directions": [list(direction) for direction in directions],
            "source_only_languages": list(legacy_source_only),
            "legacy_bidirectional_validation": bidirectional,
            "train_only_prefixes": list(normalized_prefixes),
            "direction_required_synthetic_prefixes": list(direction_required_prefixes),
            "max_physical_pairs": max_physical_pairs,
            "rare_threshold": rare_threshold,
            "filter_quality": filter_quality,
        },
        "vocab_size": vocab_size,
        "physical_pairs": physical_pairs,
        "virtual_translation_examples": virtual_examples,
        "invalid_or_filtered": dict(sorted(invalid.items())),
        "directions": direction_report,
        "languages": language_reports,
        "global_target_frequency": global_summary,
        "lowest_global_target_exposure": _rare_piece_examples(
            tokenizer,
            global_target,
            eligible,
            maximum=max_piece_examples,
            include_unused=True,
        ),
    }
    if file_sha256(tokenizer_path) != tokenizer_sha256:
        raise RuntimeError("Tokenizer changed while raw token exposure was scanned")
    return report


def audit_indexed_token_exposure(
    dataset_root: str | Path,
    tokenizer_model: str | Path,
    *,
    split: str = "train",
    bidirectional: bool | None = None,
    rare_threshold: int = 25,
    max_piece_examples: int = 50,
) -> dict[str, Any]:
    """Audit exact decoder-target exposure from already indexed token shards.

    The scan follows the indexed dataset's recorded direction graph without
    decoding or re-tokenizing text. Current rows must store an allowed source to
    target edge, and expose the reverse only when ``forward_only`` is false and
    that reverse edge is also authenticated. Legacy datasets without a recorded
    graph require an explicit ``bidirectional`` compatibility policy. Runtime-added
    BOS/EOS/language control tokens are outside this content-piece audit.
    """

    if not split or split in {".", ".."} or Path(split).name != split:
        raise ValueError("split must be one directory name")
    if rare_threshold < 1:
        raise ValueError("rare_threshold must be positive")
    if max_piece_examples < 0:
        raise ValueError("max_piece_examples must be non-negative")

    root = Path(dataset_root)
    manifest_path = root / "manifest.json"
    manifest_sha256 = file_sha256(manifest_path)
    manifest = _load_indexed_manifest(root)
    if file_sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("Indexed dataset manifest changed while it was read")
    current_schema = _uses_current_indexed_schema(root, manifest)
    shared_foundation_storage = _uses_shared_foundation_storage(manifest)
    shared_foundation_tasks = (
        _shared_foundation_source_tasks(manifest) if shared_foundation_storage else ()
    )
    if not current_schema:
        _validate_legacy_indexed_identity(root, manifest)
    split_root = root / split
    index_paths = sorted(split_root.glob("*.idx.npy"))
    if not index_paths:
        raise FileNotFoundError(f"No index shards found under {split_root}")

    first_index = np.load(index_paths[0], mmap_mode="r", allow_pickle=False)
    first_fields = frozenset(first_index.dtype.names or ())
    src_tgt_layout = {"src_offset", "src_length", "tgt_offset", "tgt_length"}.issubset(first_fields)
    legacy_storage_layout = {"ko_offset", "ko_length", "ja_offset", "ja_length"}.issubset(
        first_fields
    )
    if src_tgt_layout == legacy_storage_layout:
        raise ValueError(
            f"Unsupported indexed shard layout in {index_paths[0]}: {first_index.dtype.descr!r}"
        )
    if current_schema and not src_tgt_layout:
        raise ValueError("Current indexed dataset must use the src/tgt storage layout")
    if not current_schema and not shared_foundation_storage:
        _validate_legacy_index_dtype(
            first_index,
            index_paths[0],
            generic=src_tgt_layout,
        )
    elif shared_foundation_storage and first_index.dtype != SHARED_TARGET_INDEX_DTYPE:
        raise ValueError("Foundation shared-target index dtype is invalid")

    (
        languages,
        language_pairs,
        translation_directions,
        source_only_languages,
        legacy_storage_pair,
    ) = _indexed_direction_contract(
        manifest,
        current_schema=current_schema,
        legacy_storage_layout=legacy_storage_layout,
        legacy_bidirectional=bidirectional,
    )
    current_inventory_digest: str | None = None
    current_metadata_sha256: dict[Path, str] = {}
    current_manifest_contract: (
        tuple[
            dict[str, int],
            tuple[dict[str, int], ...],
            tuple[bool, ...],
            int,
            int,
            str,
        ]
        | None
    ) = None
    current_payload_contract: (
        tuple[
            dict[str, int],
            tuple[dict[str, int], ...],
            tuple[bool, ...],
            int,
            int,
        ]
        | None
    ) = None
    if shared_foundation_storage:
        current_inventory_digest = validate_dataset_artifact_inventory(root, manifest)
    if current_schema:
        current_metadata_sha256 = {
            root / RAW_FINGERPRINT_FILENAME: file_sha256(root / RAW_FINGERPRINT_FILENAME),
            root / PREPARE_COMPLETION_FILENAME: file_sha256(root / PREPARE_COMPLETION_FILENAME),
        }
        current_manifest_contract = _validate_current_manifest_contract(
            root,
            manifest,
            language_pairs=language_pairs,
            languages=languages,
            translation_directions=translation_directions,
            source_only_languages=source_only_languages,
        )
        (
            stats,
            source_stats,
            source_synthetic_files,
            shard_size,
            maximum_tokens_per_side,
            current_inventory_digest,
        ) = current_manifest_contract
        current_payload_contract = (
            stats,
            source_stats,
            source_synthetic_files,
            shard_size,
            maximum_tokens_per_side,
        )
        for metadata_path, expected_sha256 in current_metadata_sha256.items():
            if file_sha256(metadata_path) != expected_sha256:
                raise RuntimeError(
                    f"Current dataset metadata changed while it was authenticated: {metadata_path}"
                )
    tokenizer_path = Path(tokenizer_model)
    tokenizer_identity = _indexed_tokenizer_identity(manifest, tokenizer_path)
    language_to_id = {language: index for index, language in enumerate(languages)}
    direction_set = frozenset(translation_directions)
    target_languages = frozenset(target for _, target in translation_directions)

    tokenizer = SionTokenizer(tokenizer_path)
    missing_tags = sorted(set(languages) - set(tokenizer.languages))
    if missing_tags:
        raise ValueError(f"tokenizer is missing indexed language tags: {missing_tags}")
    vocab_size = len(tokenizer)
    if current_payload_contract is not None:
        (
            stats,
            source_stats,
            source_synthetic_files,
            shard_size,
            maximum_tokens_per_side,
        ) = current_payload_contract
        _validate_current_indexed_payload(
            root,
            stats=stats,
            source_stats=source_stats,
            source_synthetic_files=source_synthetic_files,
            languages=languages,
            translation_directions=translation_directions,
            source_only_languages=source_only_languages,
            shard_size=shard_size,
            maximum_tokens_per_side=maximum_tokens_per_side,
            vocab_size=vocab_size,
        )
    physical_counts = [np.zeros(vocab_size, dtype=np.uint64) for _ in languages]
    target_counts = [np.zeros(vocab_size, dtype=np.uint64) for _ in languages]
    physical_sentences = np.zeros(len(languages), dtype=np.uint64)
    direction_totals: dict[str, Counter[str]] = {
        _direction_label(source, target): Counter() for source, target in translation_directions
    }
    physical_pairs = 0
    virtual_examples = 0
    forward_only_pairs = 0

    for index_path in index_paths:
        index = np.load(index_path, mmap_mode="r", allow_pickle=False)
        fields = frozenset(cast(tuple[str, ...], index.dtype.names or ()))
        shard_src_tgt = {"src_offset", "src_length", "tgt_offset", "tgt_length"}.issubset(fields)
        shard_legacy_storage = {
            "ko_offset",
            "ko_length",
            "ja_offset",
            "ja_length",
        }.issubset(fields)
        if shard_src_tgt != src_tgt_layout or shard_legacy_storage != legacy_storage_layout:
            raise ValueError(f"Indexed shard layouts are inconsistent at {index_path}")
        if not current_schema and not shared_foundation_storage:
            _validate_legacy_index_dtype(
                index,
                index_path,
                generic=src_tgt_layout,
            )
        elif shared_foundation_storage and index.dtype != SHARED_TARGET_INDEX_DTYPE:
            raise ValueError(f"Foundation shared-target index dtype is invalid: {index_path}")

        row_count = len(index)
        physical_pairs += row_count
        if src_tgt_layout:
            required_metadata = {"src_language_id", "tgt_language_id"}
            if not required_metadata.issubset(fields):
                raise ValueError(f"Generic indexed shard lacks language ids: {index_path}")
            side_a_offsets = index["src_offset"]
            side_a_lengths = index["src_length"]
            side_b_offsets = index["tgt_offset"]
            side_b_lengths = index["tgt_length"]
            side_a_languages = index["src_language_id"].astype(np.int64)
            side_b_languages = index["tgt_language_id"].astype(np.int64)
            prefix = index_path.name.removesuffix(".idx.npy")
            side_a_path = split_root / f"{prefix}.src.bin"
            side_b_path = split_root / f"{prefix}.tgt.bin"
            if shared_foundation_storage:
                if "target_shared" not in fields:
                    raise ValueError(f"Foundation shard lacks target_shared: {index_path}")
                raw_target_shared = np.asarray(index["target_shared"])
                if not bool(np.isin(raw_target_shared, (0, 1)).all()):
                    raise ValueError(f"Foundation target_shared flags are invalid: {index_path}")
                target_shared = raw_target_shared.astype(np.bool_)
                source_ids = np.asarray(index["source_id"], dtype=np.int64)
                if source_ids.size and (
                    int(source_ids.min()) < 0
                    or int(source_ids.max()) >= len(shared_foundation_tasks)
                ):
                    raise ValueError(f"Foundation source ids are invalid: {index_path}")
                expected_shared = np.fromiter(
                    (
                        shared_foundation_tasks[int(source_id)] == "denoising"
                        for source_id in source_ids
                    ),
                    dtype=np.bool_,
                    count=len(source_ids),
                )
                if not np.array_equal(target_shared, expected_shared):
                    raise ValueError(
                        f"Foundation shared targets disagree with source tasks: {index_path}"
                    )
                if bool(target_shared.any()) and (
                    not np.array_equal(
                        side_a_lengths[target_shared],
                        side_b_lengths[target_shared],
                    )
                    or not np.array_equal(
                        side_a_languages[target_shared],
                        side_b_languages[target_shared],
                    )
                    or not np.array_equal(
                        np.asarray(index["src_register"])[target_shared],
                        np.asarray(index["tgt_register"])[target_shared],
                    )
                    or not bool(
                        (np.asarray(index["synthetic"], dtype=np.uint8)[target_shared] == 0).all()
                    )
                    or not bool(
                        (
                            np.asarray(index["forward_only"], dtype=np.uint8)[target_shared] == 1
                        ).all()
                    )
                ):
                    raise ValueError(
                        f"Foundation shared targets contradict source rows: {index_path}"
                    )
            else:
                if "target_shared" in fields:
                    raise ValueError(
                        f"Unauthenticated dataset declares shared target rows: {index_path}"
                    )
                target_shared = np.zeros(row_count, dtype=np.bool_)
        else:
            assert legacy_storage_pair is not None
            legacy_pair = language_pairs[0]
            side_a_offsets = index["ko_offset"]
            side_a_lengths = index["ko_length"]
            side_b_offsets = index["ja_offset"]
            side_b_lengths = index["ja_length"]
            side_a_languages = np.full(
                row_count,
                language_to_id[legacy_pair[0]],
                dtype=np.int64,
            )
            side_b_languages = np.full(
                row_count,
                language_to_id[legacy_pair[1]],
                dtype=np.int64,
            )
            prefix = index_path.name.removesuffix(".idx.npy")
            side_a_path = split_root / f"{prefix}.{legacy_storage_pair[0]}.bin"
            side_b_path = split_root / f"{prefix}.{legacy_storage_pair[1]}.bin"
            target_shared = np.zeros(row_count, dtype=np.bool_)

        if row_count:
            minimum_language_id = min(
                int(side_a_languages.min(initial=0)),
                int(side_b_languages.min(initial=0)),
            )
            maximum_language_id = max(
                int(side_a_languages.max(initial=0)),
                int(side_b_languages.max(initial=0)),
            )
            if minimum_language_id < 0 or maximum_language_id >= len(languages):
                raise ValueError(
                    "Indexed language ids are outside manifest metadata at "
                    f"{index_path}: min={minimum_language_id}, max={maximum_language_id}"
                )
        if "forward_only" in fields:
            raw_forward_only = np.asarray(index["forward_only"])
            if not bool(np.isin(raw_forward_only, (0, 1)).all()):
                raise ValueError(f"Indexed forward_only flags are invalid at {index_path}")
            forward_only = raw_forward_only.astype(np.bool_)
        else:
            forward_only = np.zeros(row_count, dtype=np.bool_)
            for row_id, (source_id, target_id) in enumerate(
                zip(side_a_languages, side_b_languages, strict=True)
            ):
                forward = (languages[int(source_id)], languages[int(target_id)])
                forward_only[row_id] = (forward[1], forward[0]) not in direction_set
        forward_only_pairs += int(np.count_nonzero(forward_only))
        forward_enabled = np.zeros(row_count, dtype=np.bool_)
        reverse_enabled = np.zeros(row_count, dtype=np.bool_)
        if src_tgt_layout:
            for row_id, (source_id, target_id, one_way) in enumerate(
                zip(side_a_languages, side_b_languages, forward_only, strict=True)
            ):
                forward = (languages[int(source_id)], languages[int(target_id)])
                if forward not in direction_set:
                    raise ValueError(
                        "Indexed stored direction is absent from "
                        "manifest.translation_directions "
                        f"at {index_path} row {row_id}: {forward!r}"
                    )
                forward_enabled[row_id] = True
                if not bool(one_way):
                    reverse = (forward[1], forward[0])
                    if reverse not in direction_set:
                        raise ValueError(
                            "Indexed row exposes an unauthenticated reverse direction because "
                            f"forward_only is false at {index_path} row {row_id}: {reverse!r}"
                        )
                    reverse_enabled[row_id] = True
        else:
            legacy_pair = language_pairs[0]
            forward_enabled.fill(legacy_pair in direction_set)
            reverse_enabled[:] = ((legacy_pair[1], legacy_pair[0]) in direction_set) & ~forward_only

        side_a_store = _open_indexed_token_store(
            side_a_path,
            side_a_offsets,
            side_a_lengths,
        )
        side_b_store = _open_indexed_token_store(
            side_b_path,
            side_b_offsets,
            np.where(target_shared, 0, side_b_lengths),
        )
        if bool(target_shared.any()):
            _accumulate_shared_target_sides(
                side_a_store,
                side_b_store,
                side_a_lengths,
                side_b_lengths,
                target_shared,
                side_a_languages,
                side_b_languages,
                reverse_enabled,
                forward_enabled,
                physical_counts,
                target_counts,
                vocab_size=vocab_size,
            )
        else:
            _accumulate_indexed_side(
                side_a_store,
                side_a_lengths,
                side_a_languages,
                reverse_enabled,
                physical_counts,
                target_counts,
                vocab_size=vocab_size,
            )
            _accumulate_indexed_side(
                side_b_store,
                side_b_lengths,
                side_b_languages,
                forward_enabled,
                physical_counts,
                target_counts,
                vocab_size=vocab_size,
            )

        physical_sentences += np.bincount(
            np.concatenate((side_a_languages, side_b_languages)).astype(np.int64),
            minlength=len(languages),
        ).astype(np.uint64, copy=False)
        virtual_examples += _add_direction_totals(
            direction_totals,
            side_a_languages,
            side_b_languages,
            side_a_lengths,
            side_b_lengths,
            forward_enabled,
            languages,
        )
        virtual_examples += _add_direction_totals(
            direction_totals,
            side_b_languages,
            side_a_languages,
            side_b_lengths,
            side_a_lengths,
            reverse_enabled,
            languages,
        )

    special = np.array(
        [_piece_is_special(tokenizer.processor.id_to_piece(i)) for i in range(vocab_size)],
        dtype=np.bool_,
    )
    byte = np.array(
        [tokenizer.processor.id_to_piece(i).startswith("<0x") for i in range(vocab_size)],
        dtype=np.bool_,
    )
    eligible = ~(special | byte)
    global_physical = np.zeros(vocab_size, dtype=np.uint64)
    global_target = np.zeros(vocab_size, dtype=np.uint64)
    for language_id in range(len(languages)):
        global_physical += physical_counts[language_id]
        global_target += target_counts[language_id]

    language_reports: dict[str, Any] = {}
    for language_id, language in enumerate(languages):
        physical = physical_counts[language_id]
        target = target_counts[language_id]
        physical_tokens = int(physical.sum(dtype=np.uint64))
        target_enabled = language in target_languages
        target_eligible = (
            eligible & (physical > 0) if target_enabled else np.zeros(vocab_size, dtype=np.bool_)
        )
        target_summary = _frequency_summary(target, target_eligible)
        target_summary["rare_threshold"] = rare_threshold
        target_summary["rare_observed_pieces"] = int(
            np.count_nonzero(target_eligible & (target > 0) & (target < rare_threshold))
        )
        byte_tokens = int(physical[byte].sum(dtype=np.uint64))
        language_reports[language] = {
            "physical_sentences": int(physical_sentences[language_id]),
            "physical_tokens": physical_tokens,
            "byte_fallback_tokens": byte_tokens,
            "unknown_tokens": int(physical[tokenizer.unk_id]),
            "target_enabled": target_enabled,
            "target_tokens": int(target.sum(dtype=np.uint64)),
            "byte_fallback_rate": round(byte_tokens / max(physical_tokens, 1), 8),
            "target_frequency": target_summary,
            "lowest_target_exposure": _rare_piece_examples(
                tokenizer,
                target,
                target_eligible,
                maximum=max_piece_examples,
                include_unused=True,
            ),
        }

    direction_report = {
        direction: {
            **{name: int(totals[name]) for name in ("examples", "source_tokens", "target_tokens")},
            "mean_target_tokens": round(
                totals["target_tokens"] / max(totals["examples"], 1),
                6,
            ),
        }
        for direction, totals in sorted(direction_totals.items())
    }
    global_summary = _frequency_summary(global_target, eligible)
    global_summary["all_target_tokens"] = int(global_target.sum(dtype=np.uint64))
    global_summary["corpus_observed_pieces"] = int(
        np.count_nonzero(eligible & (global_physical > 0))
    )
    global_summary["rare_threshold"] = rare_threshold
    global_summary["rare_observed_pieces"] = int(
        np.count_nonzero(eligible & (global_target > 0) & (global_target < rare_threshold))
    )
    global_summary["below_threshold_pieces"] = int(
        global_summary["unused_pieces"] + global_summary["rare_observed_pieces"]
    )
    authenticated_indexed = current_schema or shared_foundation_storage
    report = {
        "schema": "sion-indexed-token-exposure-audit-v2",
        "complete_scan": True,
        "count_basis": "stored_target_content_tokens",
        "runtime_control_tokens_included": False,
        "parameters": {
            "dataset_root": str(root.resolve()),
            "dataset_format": str(manifest.get("format", "unknown")),
            "dataset_contract": (
                "current-integrity-verified"
                if authenticated_indexed
                else "legacy-unverified-explicit-policy"
            ),
            "integrity_assurance": {
                "level": (
                    "self-consistent-hashes-not-signed"
                    if authenticated_indexed
                    else "legacy-payload-unverified"
                ),
                "payload_sha256_reverified_after_scan": authenticated_indexed,
                "manifest_sha256_reverified_after_scan": True,
                "tokenizer_sha256_reverified_after_scan": True,
                "cryptographically_signed": False,
            },
            "manifest_sha256": manifest_sha256,
            "artifact_inventory_sha256": current_inventory_digest,
            "tokenizer_model": str(tokenizer_path.resolve()),
            "tokenizer_identity": tokenizer_identity,
            "split": split,
            "languages": list(languages),
            "language_pairs": [list(pair) for pair in language_pairs],
            "translation_directions": [list(direction) for direction in translation_directions],
            "source_only_languages": list(source_only_languages),
            "legacy_bidirectional_override": bidirectional if not authenticated_indexed else None,
            "rare_threshold": rare_threshold,
        },
        "vocab_size": vocab_size,
        "physical_pairs": physical_pairs,
        "forward_only_pairs": forward_only_pairs,
        "virtual_translation_examples": virtual_examples,
        "directions": direction_report,
        "languages": language_reports,
        "global_target_frequency": global_summary,
        "lowest_global_target_exposure": _rare_piece_examples(
            tokenizer,
            global_target,
            eligible,
            maximum=max_piece_examples,
            include_unused=True,
        ),
    }
    if current_manifest_contract is not None:
        try:
            post_manifest_contract = _validate_current_manifest_contract(
                root,
                manifest,
                language_pairs=language_pairs,
                languages=languages,
                translation_directions=translation_directions,
                source_only_languages=source_only_languages,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError("Current dataset contract changed during token audit") from error
        if post_manifest_contract != current_manifest_contract:
            raise RuntimeError("Current dataset contract changed during token audit")
        for metadata_path, expected_sha256 in current_metadata_sha256.items():
            if file_sha256(metadata_path) != expected_sha256:
                raise RuntimeError(
                    f"Current dataset metadata changed during token audit: {metadata_path}"
                )
    if shared_foundation_storage:
        post_inventory_digest = validate_dataset_artifact_inventory(root, manifest)
        if post_inventory_digest != current_inventory_digest:
            raise RuntimeError("Foundation dataset payload changed during token audit")
    if file_sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("Indexed dataset manifest changed during token audit")
    if file_sha256(tokenizer_path) != tokenizer_identity["sha256"]:
        raise RuntimeError("Tokenizer changed during indexed token audit")
    return report


__all__ = ["audit_indexed_token_exposure", "audit_token_exposure"]
