"""CPU-only checks for bounded real-data benchmark cohort preparation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.prepare_training_probe import (
    bounded_representative_indices,
    json_value,
    length_statistics,
    prepare,
    stress_indices,
    verify_dataset_tokenizer,
)
from sion_translate.data.indexed import DistributedBucketBatchSampler


class TinyIndexedDataset:
    source_names = ["ordinary"]
    synthetic_sampling_weight = 1.0
    pair_synthetic_flags = None
    has_source_metadata = True
    bidirectional = False

    def __init__(self, count=100):
        self.pair_count = count
        self.pair_lengths = np.arange(count, dtype=np.uint32) + 1

    def __len__(self):
        return self.pair_count

    def lengths_for_indices(self, indices):
        return self.pair_lengths[indices]


def test_unbalanced_draw_uses_production_sampler_without_full_epoch(monkeypatch):
    dataset = TinyIndexedDataset()
    sampler = DistributedBucketBatchSampler(dataset, 4, bucket_size=16, seed=19)

    def forbidden_epoch(_sampler):
        raise AssertionError("the probe must not construct a full epoch")

    monkeypatch.setattr(DistributedBucketBatchSampler, "__iter__", forbidden_epoch)
    first = bounded_representative_indices(sampler, 24)
    assert first == bounded_representative_indices(sampler, 24)
    assert len(first) == len(set(first)) == 24
    assert min(first) >= 0 and max(first) < len(dataset)


def test_balanced_draw_delegates_to_production_weighted_primitive(monkeypatch):
    sampler = DistributedBucketBatchSampler(TinyIndexedDataset(), 4, bucket_size=16, seed=3)
    sampler._balance_sources = True
    requests = []

    def weighted_draw(rng, count):
        requests.append(count)
        return rng.integers(0, 100, size=count)

    monkeypatch.setattr(sampler, "_balanced_indices", weighted_draw)
    result = bounded_representative_indices(sampler, 24)
    assert requests == [64]
    assert len(result) == 24


def test_stress_cohort_uses_longest_positive_sampling_rows():
    dataset = TinyIndexedDataset()
    mask = np.arange(100) % 2 == 0
    sampler = SimpleNamespace(dataset=dataset, positive_sampling_pair_mask=lambda: mask)
    assert stress_indices(sampler, 4) == [98, 96, 94, 92]


def test_stress_direction_mapping_retains_forward_only_policy():
    dataset = TinyIndexedDataset()
    dataset.bidirectional = True
    seen = []

    def virtual(pairs, directions):
        seen.append((pairs.tolist(), directions.tolist()))
        return pairs * 2 + directions

    dataset._virtual_indices_for_pairs = virtual
    sampler = SimpleNamespace(
        dataset=dataset, positive_sampling_pair_mask=lambda: np.ones(100, dtype=bool)
    )
    assert stress_indices(sampler, 4) == [198, 197, 194, 193]
    assert seen == [([99, 98, 97, 96], [0, 1, 0, 1])]


def test_too_small_cohorts_fail_instead_of_fabricating_rows():
    sampler = DistributedBucketBatchSampler(TinyIndexedDataset(3), 1)
    with pytest.raises(ValueError, match="smaller"):
        bounded_representative_indices(sampler, 4)
    with pytest.raises(ValueError, match="not enough"):
        stress_indices(sampler, 4)


def test_json_conversion_preserves_generic_languages_and_metadata():
    item = json_value(
        {
            "src": np.asarray([10, 11], dtype=np.int64),
            "tgt": np.asarray([12], dtype=np.int64),
            "src_language": "de",
            "target_language": "sw",
            "pair_index": np.int64(91),
            "reverse_direction_trained": np.bool_(True),
            "metadata": {"task": "reasoning", "tags": ("one", "two")},
        }
    )
    assert item["src"] == [10, 11]
    assert item["pair_index"] == 91
    assert item["reverse_direction_trained"] is True
    assert item["metadata"] == {"task": "reasoning", "tags": ["one", "two"]}
    stats = length_statistics([item])
    assert stats["directions"] == {"de->sw": 1}
    assert stats["src"]["max"] == 2


def test_preparation_refuses_existing_output_and_onedrive(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare(config, tmp_path)
    with pytest.raises(ValueError, match="outside OneDrive"):
        prepare(config, tmp_path / "OneDrive - Private" / "probe")


@pytest.mark.parametrize("nested", [False, True])
def test_dataset_tokenizer_compatibility_is_required(tmp_path: Path, nested: bool):
    record = {"tokenizer_sha256": "a" * 64}
    manifest = {"fingerprint": record} if nested else record
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    verify_dataset_tokenizer(tmp_path, "a" * 64)
    with pytest.raises(ValueError, match="tokenizer SHA-256 mismatch"):
        verify_dataset_tokenizer(tmp_path, "b" * 64)


@pytest.mark.parametrize("representative,stress", [(1, 128), (4097, 128), (2048, 1)])
def test_preparation_rejects_unbounded_or_unrepresentative_requests(
    tmp_path: Path, representative: int, stress: int
):
    with pytest.raises(ValueError, match="cohort sizes"):
        prepare(
            tmp_path / "unused.yaml",
            tmp_path / "unused",
            representative_count=representative,
            stress_count=stress,
        )
