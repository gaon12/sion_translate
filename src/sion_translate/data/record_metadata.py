"""Row-level metadata shared by raw records and indexed sidecars."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Mapping, cast

import numpy as np


RECORD_METADATA_FIELDS = (
    "provenance",
    "domain",
    "category",
    "original_direction",
    "training_direction",
    "quality_profile",
)
RECORD_METADATA_FORMAT = "sion-record-metadata-json-offsets-v2"
RECORD_METADATA_INDEX_DTYPE = np.dtype(
    [
        ("offset", "<u8"),
        ("length", "<u4"),
    ]
)
RECORD_METADATA_INDEX_SUFFIX = ".meta.npy"
RECORD_METADATA_DATA_SUFFIX = ".meta.bin"


def inherit_record_metadata(
    record: Mapping[object, object],
    inherited: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Merge supported metadata from an ancestor, envelope, and current row.

    A common JSONL layout puts annotations inside a ``metadata`` object, while
    generated corpora usually place them directly beside the language fields.
    Descendants override ancestors and direct fields override the envelope.
    Values are copied so metadata attached to one expanded pair cannot mutate a
    sibling pair.
    """

    result = {
        field: deepcopy(value)
        for field, value in (inherited or {}).items()
        if field in RECORD_METADATA_FIELDS
    }
    envelope = record.get("metadata")
    if isinstance(envelope, Mapping):
        typed_envelope = cast(Mapping[object, object], envelope)
        for field in RECORD_METADATA_FIELDS:
            if field in typed_envelope:
                result[field] = deepcopy(typed_envelope[field])
    for field in RECORD_METADATA_FIELDS:
        if field in record:
            result[field] = deepcopy(record[field])
    return result


def encode_record_metadata(metadata: Mapping[str, object] | None) -> bytes:
    """Encode supported fields deterministically, using an empty payload for none."""

    if not metadata:
        return b""
    selected = {field: metadata[field] for field in RECORD_METADATA_FIELDS if field in metadata}
    if not selected:
        return b""
    return json.dumps(
        selected,
        # Escaping non-ASCII also keeps a lone surrogate from a permissive
        # ``json.loads`` input representable in the UTF-8 sidecar.
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_record_metadata(payload: bytes) -> dict[str, object]:
    """Decode and validate one indexed sidecar payload."""

    if not payload:
        return {}
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("record metadata payload must decode to an object")
    metadata = cast(dict[str, object], value)
    unknown = set(metadata) - set(RECORD_METADATA_FIELDS)
    if unknown:
        raise ValueError(f"record metadata contains unsupported fields: {sorted(unknown)}")
    return metadata
