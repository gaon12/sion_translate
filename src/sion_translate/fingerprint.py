# Fingerprint manifests contain recursively heterogeneous JSON values.
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FINGERPRINT_SCHEMA = "sion-dataset-fingerprint-v2"
PREPROCESSING_SCHEMA = "sion-prepare-v7"


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileFingerprint:
    name: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return {"size": self.size, "sha256": self.sha256}


@dataclass(frozen=True)
class DatasetFingerprint(Mapping[str, int]):
    """Strong dataset identity that remains compatible with size-map callers.

    Mapping operations expose ``{filename: byte_size}``, so existing code can
    still call ``len()``, ``sum(values())`` and iterate file names. Equality and
    JSON persistence include content hashes and preprocessing context.
    """

    files: tuple[FileFingerprint, ...]
    language_pairs: tuple[tuple[str, str], ...] = ()
    tokenizer_sha256: str | None = None
    preprocessing_schema: str = PREPROCESSING_SCHEMA
    preprocessing_options_json: str = "{}"
    schema: str = FINGERPRINT_SCHEMA

    def __post_init__(self) -> None:
        names = [file.name for file in self.files]
        if len(names) != len(set(names)):
            raise ValueError("fingerprint file names must be unique")

    def __getitem__(self, name: str) -> int:
        for file in self.files:
            if file.name == name:
                return file.size
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return (file.name for file in self.files)

    def __len__(self) -> int:
        return len(self.files)

    @property
    def preprocessing_options(self) -> dict[str, Any]:
        value = json.loads(self.preprocessing_options_json)
        if not isinstance(value, dict):
            raise ValueError("preprocessing options must decode to an object")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "preprocessing_schema": self.preprocessing_schema,
            "language_pairs": [list(pair) for pair in self.language_pairs],
            "tokenizer_sha256": self.tokenizer_sha256,
            "preprocessing_options": self.preprocessing_options,
            "files": {file.name: file.to_dict() for file in self.files},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DatasetFingerprint:
        if value.get("schema") != FINGERPRINT_SCHEMA:
            raise ValueError("unsupported fingerprint schema")
        raw_files = value.get("files")
        if not isinstance(raw_files, Mapping):
            raise ValueError("fingerprint files must be an object")
        files = tuple(
            FileFingerprint(
                name=str(name),
                size=int(details["size"]),
                sha256=str(details["sha256"]),
            )
            for name, details in sorted(raw_files.items())
            if isinstance(details, Mapping)
        )
        raw_pairs = value.get("language_pairs") or ()
        pairs = tuple(
            (str(pair[0]), str(pair[1]))
            for pair in raw_pairs
            if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2
        )
        options = value.get("preprocessing_options") or {}
        return cls(
            files=files,
            language_pairs=pairs,
            tokenizer_sha256=(
                str(value["tokenizer_sha256"])
                if value.get("tokenizer_sha256") is not None
                else None
            ),
            preprocessing_schema=str(value.get("preprocessing_schema", "")),
            preprocessing_options_json=_canonical_json(options),
            schema=str(value["schema"]),
        )


def _canonical_json(value: Mapping[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_dataset_fingerprint(
    paths: Sequence[str | Path],
    *,
    language_pairs: Sequence[Sequence[str]] = (),
    tokenizer_model: str | Path | None = None,
    preprocessing_schema: str = PREPROCESSING_SCHEMA,
    preprocessing_options: Mapping[str, Any] | None = None,
) -> DatasetFingerprint:
    resolved = [Path(path) for path in paths]
    files = tuple(
        FileFingerprint(path.name, path.stat().st_size, file_sha256(path))
        for path in sorted(resolved, key=lambda item: (item.name, str(item)))
    )
    tokenizer_path = Path(tokenizer_model) if tokenizer_model is not None else None
    tokenizer_hash = (
        file_sha256(tokenizer_path)
        if tokenizer_path is not None and tokenizer_path.is_file()
        else None
    )
    return DatasetFingerprint(
        files=files,
        language_pairs=tuple((str(pair[0]), str(pair[1])) for pair in language_pairs),
        tokenizer_sha256=tokenizer_hash,
        preprocessing_schema=preprocessing_schema,
        preprocessing_options_json=_canonical_json(preprocessing_options),
    )
