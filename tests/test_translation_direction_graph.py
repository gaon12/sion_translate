from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sion_translate.data import DistributedBucketBatchSampler, IndexedParallelDataset
from sion_translate.cli.train import preflight_effective_translation_training
from sion_translate.config import AppConfig, DataConfig
from sion_translate.data.prepare import prepare_dataset
from sion_translate.tokenizer import load_tokenizer_metadata, train_tokenizer


LANGUAGE_PAIRS = (("de", "fr"), ("ar", "sw"))
TRANSLATION_DIRECTIONS = (("de", "fr"), ("fr", "de"), ("sw", "ar"))


def _write_arbitrary_graph_corpus(path: Path, rows: int = 40) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "de": f"Deutscher Beispielsatz Nummer {index} für das Training.",
                    "fr": f"Phrase française numéro {index} pour l'entraînement.",
                    "sw": f"Sentensi ya Kiswahili nambari {index} kwa mafunzo.",
                    "ar": f"جملة عربية رقم {index} مخصصة للتدريب.",
                },
                ensure_ascii=False,
            )
            + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )


def test_mixed_direction_graph_is_materialized_exactly(tmp_path: Path) -> None:
    source = tmp_path / "parallel.jsonl"
    _write_arbitrary_graph_corpus(source)
    tokenizer_path = train_tokenizer(
        [str(source)],
        tmp_path / "tokenizer",
        vocab_size=640,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        validation_fraction=0.0,
        test_fraction=0.0,
        language_pairs=LANGUAGE_PAIRS,
        translation_directions=TRANSLATION_DIRECTIONS,
        num_workers=1,
        num_threads=1,
    )
    tokenizer_metadata = load_tokenizer_metadata(tokenizer_path)
    assert tokenizer_metadata is not None
    assert tokenizer_metadata["translation_directions"] == [
        list(direction) for direction in TRANSLATION_DIRECTIONS
    ]

    dataset_dir = tmp_path / "dataset"
    stats = prepare_dataset(
        [str(source)],
        tokenizer_path,
        dataset_dir,
        validation_fraction=0.0,
        test_fraction=0.0,
        dedup_backend="memory",
        language_pairs=LANGUAGE_PAIRS,
        translation_directions=TRANSLATION_DIRECTIONS,
        num_workers=1,
    )
    assert stats.valid_pairs == 80

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["language_pairs"] == [list(pair) for pair in LANGUAGE_PAIRS]
    assert manifest["translation_directions"] == [
        list(direction) for direction in TRANSLATION_DIRECTIONS
    ]

    dataset = IndexedParallelDataset(dataset_dir, "train", bidirectional=True)
    current_with_legacy_one_way = IndexedParallelDataset(
        dataset_dir,
        "train",
        bidirectional=True,
        legacy_bidirectional=False,
    )
    observed = Counter(
        (dataset[index]["src_language"], dataset[index]["target_language"])
        for index in range(len(dataset))
    )

    assert dataset.translation_directions == TRANSLATION_DIRECTIONS
    assert len(current_with_legacy_one_way) == len(dataset)
    assert dataset.observed_language_pairs == LANGUAGE_PAIRS
    assert dataset.direction_count == 3
    assert len(dataset) == 120
    assert observed == {
        ("de", "fr"): 40,
        ("fr", "de"): 40,
        ("sw", "ar"): 40,
    }
    for item in dataset:
        direction = (item["src_language"], item["target_language"])
        assert item["reverse_direction_trained"] is (direction != ("sw", "ar"))

    downgraded_graph = copy.deepcopy(manifest)
    downgraded_graph["preprocessing_schema"] = "sion-prepare-v8"
    downgraded_graph.pop("translation_directions")
    (dataset_dir / "manifest.json").write_text(
        json.dumps(downgraded_graph),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema markers disagree"):
        IndexedParallelDataset(dataset_dir, "train", bidirectional=True, verify_integrity=False)

    missing_graph = copy.deepcopy(manifest)
    missing_graph.pop("translation_directions")
    (dataset_dir / "manifest.json").write_text(
        json.dumps(missing_graph),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="translation_directions are invalid"):
        IndexedParallelDataset(dataset_dir, "train", bidirectional=True, verify_integrity=False)

    mismatched_graph = copy.deepcopy(manifest)
    mismatched_graph["translation_directions"].append(["ar", "sw"])
    (dataset_dir / "manifest.json").write_text(
        json.dumps(mismatched_graph),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="direction rows disagree"):
        IndexedParallelDataset(dataset_dir, "train", bidirectional=True, verify_integrity=False)


def test_balanced_sampling_preserves_mixed_graph_virtual_direction_mass(
    tmp_path: Path,
) -> None:
    source = tmp_path / "parallel.jsonl"
    _write_arbitrary_graph_corpus(source)
    tokenizer_path = train_tokenizer(
        [str(source)],
        tmp_path / "tokenizer",
        vocab_size=640,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        validation_fraction=0.0,
        test_fraction=0.0,
        language_pairs=LANGUAGE_PAIRS,
        translation_directions=TRANSLATION_DIRECTIONS,
        num_workers=1,
        num_threads=1,
    )
    dataset_dir = tmp_path / "dataset"
    prepare_dataset(
        [str(source)],
        tokenizer_path,
        dataset_dir,
        validation_fraction=0.0,
        test_fraction=0.0,
        dedup_backend="memory",
        language_pairs=LANGUAGE_PAIRS,
        translation_directions=TRANSLATION_DIRECTIONS,
        num_workers=1,
    )
    dataset = IndexedParallelDataset(dataset_dir, "train", bidirectional=True)
    sampler = DistributedBucketBatchSampler(
        dataset,
        batch_size=8,
        source_sampling_alpha=0.9,
        max_source_upsampling=10.0,
    )

    sampled = sampler._balanced_indices(  # pyright: ignore[reportPrivateUsage]
        np.random.default_rng(17),
        30_000,
    )
    observed = Counter(
        (dataset[int(index)]["src_language"], dataset[int(index)]["target_language"])
        for index in sampled
    )

    for direction in TRANSLATION_DIRECTIONS:
        assert observed[direction] / len(sampled) == pytest.approx(1 / 3, abs=0.02)

    physical_sources = np.concatenate([index["src_language_id"] for index in dataset.indices])
    de_id = dataset.languages.index("de")
    dataset.pair_source_ids = np.where(physical_sources == de_id, 0, 1).astype(np.uint16)
    dataset.source_names = ["de_fr.jsonl", "sw_ar.jsonl"]
    config = AppConfig(
        data=DataConfig(
            language_pairs=[list(pair) for pair in LANGUAGE_PAIRS],
            translation_directions=[list(direction) for direction in TRANSLATION_DIRECTIONS],
            source_sampling_weights={"sw_ar.jsonl": 0.0},
        )
    )
    excluding_sampler = DistributedBucketBatchSampler(
        dataset,
        batch_size=8,
        source_sampling_weights=config.data.source_sampling_weights,
        max_source_upsampling=10.0,
    )
    with pytest.raises(ValueError, match="zero probability"):
        preflight_effective_translation_training(config, excluding_sampler)

    config.data.source_sampling_weights = {}
    config.data.denoise_probability = 1.0
    denoise_only_sampler = DistributedBucketBatchSampler(dataset, batch_size=8)
    with pytest.raises(ValueError, match="no translation objective"):
        preflight_effective_translation_training(config, denoise_only_sampler)
