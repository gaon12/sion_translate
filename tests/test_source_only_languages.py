"""Source-only languages must never appear on the target side of a training pair.

한본어(kj) is a code-mixed input variety: the model has to read it, but every
translation it produces must be monolingual Korean or Japanese. Training with
``bidirectional=True`` and no restriction would also learn ko->kj and ja->kj,
which is how a model starts injecting kana into Korean output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sion_translate.config import AppConfig, DataConfig, config_from_raw
from sion_translate.data.collate import SionBatchCollator
from sion_translate.data.indexed import (
    DistributedBucketBatchSampler,
    IndexedParallelDataset,
)
from sion_translate.data.prepare import INDEX_DTYPE, prepare_dataset
from sion_translate.tokenizer import SionTokenizer, train_tokenizer


LANGUAGE_PAIRS = (("kj", "ko"), ("kj", "ja"), ("ko", "ja"))


@pytest.fixture(scope="module")
def hanboneo_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small kj/ko/ja corpus with enough distinct rows to train a tokenizer."""

    root = tmp_path_factory.mktemp("hanboneo-corpus")
    path = root / "hanboneo.jsonl"
    syllables = "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허"
    katakana = "カキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモ"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(400):
            high, low = divmod(index, len(syllables))
            ko_word = syllables[high % len(syllables)] + syllables[low]
            ja_word = katakana[high % len(katakana)] + katakana[low]
            handle.write(
                json.dumps(
                    {
                        "kj": f"{ko_word} 오늘 スケジュール 진짜 야바이데스네",
                        "ko": f"{ko_word} 오늘 일정 정말 위험하네요",
                        "ja": f"{ja_word} 今日のスケジュールは本当に危ないですね",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def _train(corpus: Path, directory: Path, pairs: tuple[tuple[str, str], ...]) -> Path:
    return train_tokenizer(
        [str(corpus)],
        directory,
        vocab_size=640,
        input_sentence_size=1000,
        seed_sentencepiece_size=1000,
        language_pairs=pairs,
        num_workers=1,
        num_threads=1,
    )


@pytest.fixture(scope="module")
def kj_tokenizer(hanboneo_corpus: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """SentencePiece training is slow, so share one kj/ko/ja model per module."""

    root = tmp_path_factory.mktemp("hanboneo-tokenizer")
    return _train(hanboneo_corpus, root, LANGUAGE_PAIRS)


@pytest.fixture(scope="module")
def prepared(
    hanboneo_corpus: Path,
    kj_tokenizer: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    dataset_dir = tmp_path_factory.mktemp("hanboneo-prepared") / "dataset"
    stats = prepare_dataset(
        [str(hanboneo_corpus)],
        kj_tokenizer,
        dataset_dir,
        language_pairs=LANGUAGE_PAIRS,
        source_only_languages=("kj",),
        validation_fraction=0.0,
        test_fraction=0.0,
        dedup_backend="memory",
        num_workers=1,
    )
    assert stats.valid_pairs > 0
    assert stats.forward_only_pairs > 0
    return dataset_dir


def test_index_dtype_carries_the_forward_only_flag() -> None:
    assert INDEX_DTYPE.names is not None
    assert "forward_only" in INDEX_DTYPE.names


def test_manifest_records_the_source_only_languages(prepared: Path) -> None:
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["source_only_languages"] == ["kj"]
    assert manifest["format"] == "sion-indexed-parallel-v5"


def test_kj_is_never_a_target_language(prepared: Path) -> None:
    dataset = IndexedParallelDataset(prepared, "train", bidirectional=True)

    directions = {
        (dataset[index]["src_language"], dataset[index]["target_language"])
        for index in range(len(dataset))
    }

    assert ("kj", "ko") in directions
    assert ("kj", "ja") in directions
    assert ("ko", "ja") in directions
    assert ("ja", "ko") in directions
    assert not any(target == "kj" for _, target in directions)


def test_direction_count_matches_the_reachable_directions(prepared: Path) -> None:
    dataset = IndexedParallelDataset(prepared, "train", bidirectional=True)

    directions = {
        (dataset[index]["src_language"], dataset[index]["target_language"])
        for index in range(len(dataset))
    }

    assert dataset.direction_count == len(directions) == 4


def test_length_accounts_for_the_suppressed_direction(prepared: Path) -> None:
    dataset = IndexedParallelDataset(prepared, "train", bidirectional=True)

    assert dataset.forward_only_count > 0
    assert len(dataset) == 2 * dataset.pair_count - dataset.forward_only_count


def test_every_virtual_index_resolves_and_is_reachable(prepared: Path) -> None:
    dataset = IndexedParallelDataset(prepared, "train", bidirectional=True)

    seen: set[tuple[int, int]] = set()
    for index in range(len(dataset)):
        seen.add(dataset._resolve_virtual(index))
    # Each bidirectional pair contributes two slots and each forward-only pair
    # exactly one, with no slot shared between two virtual indices.
    assert len(seen) == len(dataset)
    forward_only = {pair for pair, direction in seen if direction == 0}
    assert len(forward_only) == dataset.pair_count


def test_metadata_accessors_agree_with_getitem(prepared: Path) -> None:
    dataset = IndexedParallelDataset(prepared, "train", bidirectional=True)
    indices = np.arange(len(dataset))

    vector_lengths = dataset.lengths_for_indices(indices)
    for index in range(len(dataset)):
        item = dataset[index]
        assert dataset.length_at(index) == len(item["src"]) + len(item["tgt"]) + 4
        assert dataset.length_at(index) == int(vector_lengths[index]) + 4
        assert dataset.synthetic_at(index) == item["synthetic"]
        assert dataset.source_id_at(index) == 0


def test_balanced_sampling_maps_physical_pairs_to_valid_virtual_indices(
    prepared: Path,
) -> None:
    dataset = IndexedParallelDataset(prepared, "train", bidirectional=True)
    sampler = DistributedBucketBatchSampler(
        dataset,
        batch_size=17,
        bucket_size=68,
        source_sampling_alpha=0.5,
        seed=73,
        drop_last=False,
    )

    sampled = [index for batch in sampler for index in batch]

    assert sampled
    assert all(0 <= index < len(dataset) for index in sampled)
    assert all(dataset[index]["target_language"] != "kj" for index in sampled)

    # Exercise the inverse layout mapping directly for both row classes. A
    # requested reverse direction is honored only when that physical pair has
    # a trained reverse edge.
    physical = np.arange(dataset.pair_count, dtype=np.uint32)
    virtual = dataset._virtual_indices_for_pairs(
        physical,
        np.ones(dataset.pair_count, dtype=np.uint32),
    )
    resolved = [dataset._resolve_virtual(int(index)) for index in virtual]
    assert [pair for pair, _ in resolved] == physical.tolist()
    assert dataset._forward_only_pairs is not None
    forward_only_pairs = set(map(int, dataset._forward_only_pairs))
    assert all(
        direction == (0 if pair in forward_only_pairs else 1) for pair, direction in resolved
    )


def test_reverse_edge_metadata_tracks_source_only_and_dense_pairs(prepared: Path) -> None:
    dataset = IndexedParallelDataset(prepared, "train", bidirectional=True)

    for index in range(len(dataset)):
        item = dataset[index]
        assert item["reverse_direction_trained"] is (item["src_language"] != "kj")


def test_collator_preserves_the_dataset_reverse_edge_mask(
    prepared: Path,
    kj_tokenizer: Path,
) -> None:
    dataset = IndexedParallelDataset(prepared, "train", bidirectional=True)
    source_only = next(
        dataset[index]
        for index in range(len(dataset))
        if not dataset[index]["reverse_direction_trained"]
    )
    bidirectional = next(
        dataset[index]
        for index in range(len(dataset))
        if dataset[index]["reverse_direction_trained"]
    )
    collator = SionBatchCollator(
        SionTokenizer(kj_tokenizer),
        max_source_length=64,
        max_target_length=64,
    )

    batch = collator([source_only, bidirectional])

    assert batch["reverse_direction_trained"].tolist() == [False, True]


def test_worker_pickling_rebuilds_the_direction_maps(prepared: Path) -> None:
    dataset = IndexedParallelDataset(prepared, "train", bidirectional=True)
    restored = IndexedParallelDataset.__new__(IndexedParallelDataset)
    restored.__setstate__(dataset.__getstate__())

    assert len(restored) == len(dataset)
    for index in (0, 1, len(dataset) - 1):
        assert restored._resolve_virtual(index) == dataset._resolve_virtual(index)
        assert restored[index]["target_language"] == dataset[index]["target_language"]


def test_unidirectional_reading_is_unaffected(prepared: Path) -> None:
    dataset = IndexedParallelDataset(prepared, "train", bidirectional=False)

    assert len(dataset) == dataset.pair_count
    for index in range(len(dataset)):
        assert dataset[index]["target_language"] != "kj"
        assert dataset[index]["reverse_direction_trained"] is False


def test_corpus_without_source_only_languages_keeps_the_old_layout(
    hanboneo_corpus: Path,
    kj_tokenizer: Path,
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    prepare_dataset(
        [str(hanboneo_corpus)],
        kj_tokenizer,
        dataset_dir,
        language_pairs=(("ko", "ja"),),
        validation_fraction=0.0,
        test_fraction=0.0,
        dedup_backend="memory",
        num_workers=1,
    )
    dataset = IndexedParallelDataset(dataset_dir, "train", bidirectional=True)

    assert dataset.forward_only_count == 0
    assert dataset.source_only_languages == ()
    assert dataset.direction_count == 2
    assert len(dataset) == 2 * dataset.pair_count
    # No per-pair maps are allocated when nothing is forward-only.
    assert dataset._bidirectional_pairs is None
    assert dataset._forward_only_pairs is None
    assert dataset._resolve_virtual(5) == (2, 1)


@pytest.mark.parametrize(
    ("source_only", "message"),
    [
        (("zz",), "source_only_languages must appear"),
        (("kj", "ko", "ja"), "at most one side"),
    ],
)
def test_prepare_rejects_invalid_source_only_languages(
    hanboneo_corpus: Path,
    kj_tokenizer: Path,
    tmp_path: Path,
    source_only: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_dataset(
            [str(hanboneo_corpus)],
            kj_tokenizer,
            tmp_path / "dataset",
            language_pairs=LANGUAGE_PAIRS,
            source_only_languages=source_only,
            dedup_backend="memory",
            num_workers=1,
        )


def test_tokenizer_reserves_a_control_tag_for_every_language(kj_tokenizer: Path) -> None:
    tokenizer = SionTokenizer(kj_tokenizer)

    assert set(tokenizer.language_tags) == {"kj", "ko", "ja"}
    assert set(tokenizer.denoise_tags) == {"kj", "ko", "ja"}


def test_config_validates_source_only_languages() -> None:
    config = config_from_raw(
        {
            "model": {"vocab_size": 64},
            "data": {
                "language_pairs": [["kj", "ko"], ["kj", "ja"], ["ko", "ja"]],
                "source_only_languages": ["kj"],
            },
        }
    )
    config.validate()

    assert config.data.configured_source_only_languages() == ("kj",)


@pytest.mark.parametrize(
    ("source_only", "message"),
    [
        (["zz"], "must appear in the configured language"),
        (["kj", "ko", "ja"], "at most one side"),
    ],
)
def test_config_rejects_invalid_source_only_languages(
    source_only: list[str],
    message: str,
) -> None:
    config = config_from_raw(
        {
            "model": {"vocab_size": 64},
            "data": {
                "language_pairs": [["kj", "ko"], ["kj", "ja"], ["ko", "ja"]],
                "source_only_languages": source_only,
            },
        }
    )

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_config_rejects_a_pair_source_only_on_both_sides() -> None:
    config = AppConfig(
        data=DataConfig(
            language_pairs=[["kj", "ko"]],
            source_only_languages=["kj", "ko"],
        )
    )
    config.model.vocab_size = 64

    with pytest.raises(ValueError, match="at most one side"):
        config.validate()


def test_source_only_side_is_moved_to_the_source_position(
    tmp_path: Path,
) -> None:
    """Configuring the pair as (ko, kj) must still train kj->ko, not ko->kj."""

    corpus = tmp_path / "reversed.jsonl"
    syllables = "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허"
    with corpus.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(400):
            high, low = divmod(index, len(syllables))
            word = syllables[high % len(syllables)] + syllables[low]
            handle.write(
                json.dumps(
                    {
                        "ko": f"{word} 오늘 일정 정말 위험하네요",
                        "kj": f"{word} 오늘 スケジュール 야바이",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    tokenizer_path = _train(corpus, tmp_path / "tokenizer", (("ko", "kj"),))
    dataset_dir = tmp_path / "dataset"
    prepare_dataset(
        [str(corpus)],
        tokenizer_path,
        dataset_dir,
        language_pairs=(("ko", "kj"),),
        source_only_languages=("kj",),
        validation_fraction=0.0,
        test_fraction=0.0,
        dedup_backend="memory",
        num_workers=1,
    )
    dataset = IndexedParallelDataset(dataset_dir, "train", bidirectional=True)

    directions = {
        (dataset[index]["src_language"], dataset[index]["target_language"])
        for index in range(len(dataset))
    }

    assert directions == {("kj", "ko")}
    assert len(dataset) == dataset.pair_count
