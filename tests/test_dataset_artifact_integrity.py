from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from sion_translate.data.integrity import (
    build_dataset_artifact_inventory,
    validate_dataset_artifact_inventory,
)
from sion_translate.data.indexed import IndexedParallelDataset
from sion_translate.data.prepare import INDEX_DTYPE


def _authenticated_dataset(root: Path) -> Path:
    train = root / "train"
    train.mkdir(parents=True)
    np.asarray([101, 102], dtype=np.uint32).tofile(train / "00000.src.bin")
    np.asarray([201, 202], dtype=np.uint32).tofile(train / "00000.tgt.bin")
    index = np.asarray(
        [(0, 2, 0, 2, 0, 0, 0, 1, 0, 100, 0, 0)],
        dtype=INDEX_DTYPE,
    )
    np.save(train / "00000.idx.npy", index, allow_pickle=False)
    manifest = {
        "format": "sion-indexed-parallel-v6",
        "language_pairs": [["ko", "ja"]],
        "languages": ["ko", "ja"],
        "artifact_inventory": build_dataset_artifact_inventory(root),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return train / "00000.src.bin"


def test_same_size_same_mtime_payload_mutation_is_rehashed(tmp_path: Path) -> None:
    payload = _authenticated_dataset(tmp_path)
    first_digest = validate_dataset_artifact_inventory(tmp_path)
    original_stat = payload.stat()

    with payload.open("r+b") as handle:
        handle.seek(0)
        handle.write((100).to_bytes(4, "little"))
        handle.flush()
        os.fsync(handle.fileno())
    os.utime(
        payload,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    assert payload.stat().st_size == original_stat.st_size
    assert payload.stat().st_mtime_ns == original_stat.st_mtime_ns
    with pytest.raises(ValueError, match=r"SHA-256 mismatch.*00000\.src\.bin"):
        validate_dataset_artifact_inventory(tmp_path)
    assert len(first_digest or "") == 64


def test_unmanifested_payload_file_is_rejected(tmp_path: Path) -> None:
    _authenticated_dataset(tmp_path)
    (tmp_path / "train" / "unexpected.bin").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="unexpected=.*unexpected.bin"):
        validate_dataset_artifact_inventory(tmp_path)


def test_authenticated_format_cannot_drop_its_inventory(tmp_path: Path) -> None:
    _authenticated_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("artifact_inventory")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="no artifact inventory"):
        validate_dataset_artifact_inventory(tmp_path, require_manifest=False)


def test_dataset_constructor_requires_explicit_legacy_opt_in(tmp_path: Path) -> None:
    _authenticated_dataset(tmp_path)
    (tmp_path / "manifest.json").unlink()

    with pytest.raises(FileNotFoundError, match="manifest is missing"):
        IndexedParallelDataset(tmp_path, "train")

    with pytest.raises(ValueError, match="pass legacy_language_pairs explicitly"):
        IndexedParallelDataset(
            tmp_path,
            "train",
            allow_unverified_legacy=True,
        )

    legacy = IndexedParallelDataset(
        tmp_path,
        "train",
        allow_unverified_legacy=True,
        legacy_language_pairs=(("fr", "de"),),
    )
    assert legacy.pair_count == 1
    assert legacy.language_pairs == (("fr", "de"),)
