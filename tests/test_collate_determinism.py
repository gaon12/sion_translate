from __future__ import annotations

import random
from collections.abc import Iterable

import pytest
import torch
from torch.utils.data import DataLoader

from sion_translate.data.collate import SionBatchCollator


class TinyTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    mask_id = 3
    language_tags = {"de": 10, "en": 11}
    denoise_tags = {"en": 12, "de": 13}
    slot_ids = [90]


class SourceOnlyTokenizer(TinyTokenizer):
    language_tags = {
        "de": 10,
        "en": 11,
        "kj": 14,
        "kd": 15,
        "jd": 16,
        "ko": 17,
        "ja": 18,
    }
    denoise_tags = {
        "en": 12,
        "de": 13,
        "kj": 24,
        "kd": 25,
        "jd": 26,
        "ko": 27,
        "ja": 28,
    }


def _items() -> list[dict]:
    return [
        {
            "src": [20 + index, *range(40, 60), 90],
            "tgt": [70, 71, 72],
            "src_language": "en",
            "target_language": "de",
            "src_register": 0,
            "target_register": 0,
            "pair_index": index,
        }
        for index in range(12)
    ]


def _collator() -> SionBatchCollator:
    return SionBatchCollator(
        TinyTokenizer(),
        max_source_length=64,
        max_target_length=64,
        denoise_probability=0.5,
        denoise_noise_density=0.35,
        denoise_mean_span=2.0,
        source_token_dropout=0.4,
        augmentation_seed=1729,
    )


def _collect(loader: Iterable[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    return [{name: value.detach().clone() for name, value in batch.items()} for batch in loader]


def _assert_batches_equal(
    actual: list[dict[str, torch.Tensor]],
    expected: list[dict[str, torch.Tensor]],
) -> None:
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        assert actual_batch.keys() == expected_batch.keys()
        for name in actual_batch:
            torch.testing.assert_close(actual_batch[name], expected_batch[name])


def test_augmentation_is_independent_of_global_rng_order_and_workers() -> None:
    items = _items()
    serial_collator = _collator()
    worker_collator = _collator()
    serial_collator.set_epoch(3)
    worker_collator.set_epoch(3)

    random.seed(1)
    serial = _collect(
        DataLoader(
            items,
            batch_size=1,
            num_workers=0,
            collate_fn=serial_collator,
        )
    )
    random.seed(999)
    worker_loader = DataLoader(
        items,
        batch_size=1,
        num_workers=2,
        persistent_workers=True,
        collate_fn=worker_collator,
    )
    workers = _collect(worker_loader)
    _assert_batches_equal(workers, serial)

    reversed_batches = _collect(
        DataLoader(
            list(reversed(items)),
            batch_size=1,
            num_workers=0,
            collate_fn=serial_collator,
        )
    )
    _assert_batches_equal(list(reversed(reversed_batches)), serial)

    task_ids = {int(batch["input_ids"][0, 0]) for batch in serial}
    assert task_ids == {
        TinyTokenizer.language_tags["de"],
        TinyTokenizer.denoise_tags["en"],
    }
    assert any(
        TinyTokenizer.mask_id in batch["input_ids"][0].tolist()
        for batch in serial
        if int(batch["input_ids"][0, 0]) == TinyTokenizer.denoise_tags["en"]
    )
    assert any(
        int(batch["attention_mask"][0].sum()) < len(items[index]["src"]) + 2
        for index, batch in enumerate(serial)
        if int(batch["input_ids"][0, 0]) == TinyTokenizer.language_tags["de"]
    )
    assert all(
        90 in batch["input_ids"][0].tolist()
        for batch in serial
        if int(batch["input_ids"][0, 0]) == TinyTokenizer.language_tags["de"]
    )

    serial_collator.set_augmentation_key(4)
    worker_collator.set_epoch(4)
    next_serial = _collect(
        DataLoader(
            items,
            batch_size=1,
            num_workers=0,
            collate_fn=serial_collator,
        )
    )
    next_workers = _collect(worker_loader)
    _assert_batches_equal(next_workers, next_serial)
    assert any(
        not torch.equal(previous["input_ids"], current["input_ids"])
        for previous, current in zip(serial, next_serial, strict=True)
    )


def test_denoise_target_only_contains_tokens_the_input_could_see() -> None:
    """The denoise target is the restored source, so it must be derived from the
    already-truncated source rather than from the full one."""

    tokenizer = TinyTokenizer()
    collator = SionBatchCollator(
        tokenizer,
        max_source_length=12,
        max_target_length=64,
        denoise_probability=1.0,
        denoise_noise_density=0.15,
        denoise_mean_span=2.0,
    )
    long_source = list(range(100, 160))
    item = {
        "src": long_source,
        "tgt": list(range(200, 230)),
        "src_language": "en",
        "target_language": "de",
        "src_register": 2,
        "target_register": 2,
        "pair_index": 0,
    }

    batch = collator([item])

    labels = [int(value) for value in batch["labels"][0] if int(value) != -100]
    assert labels[-1] == tokenizer.eos_id
    restored = labels[:-1]
    # Every restored token has to come from the window the encoder actually saw.
    assert set(restored) <= set(long_source[: 12 - 2])
    assert len(restored) <= 12 - 2


@pytest.mark.parametrize("source_language", ["kj", "kd", "jd"])
def test_source_only_inputs_never_become_denoise_tasks(source_language: str) -> None:
    tokenizer = SourceOnlyTokenizer()
    collator = SionBatchCollator(
        tokenizer,
        max_source_length=16,
        max_target_length=16,
        denoise_probability=1.0,
        source_only_languages=("kj", "kd", "jd"),
    )
    source_only_item = {
        "src": [30, 31],
        "tgt": [40, 41],
        "src_language": source_language,
        "target_language": "ko",
        "src_register": 0,
        "target_register": 1,
        "pair_index": 1,
        "reverse_direction_trained": False,
    }
    ordinary_item = {
        "src": [50, 51],
        "tgt": [60, 61],
        "src_language": "ko",
        "target_language": "ja",
        "src_register": 2,
        "target_register": 3,
        "pair_index": 2,
        "reverse_direction_trained": True,
    }

    batch = collator([source_only_item, ordinary_item])

    # A source-only row stays a translation row even when denoising is otherwise
    # certain: the labels remain the configured monolingual target.
    assert batch["input_ids"][0, 0].item() == tokenizer.language_tags["ko"]
    assert batch["labels"][0, :3].tolist() == [40, 41, tokenizer.eos_id]
    assert batch["source_language_tag_ids"][0].item() == tokenizer.language_tags[source_language]
    assert batch["reverse_direction_trained"][0].item() is False

    # An ordinary ko/ja row still follows the existing denoising path.
    assert batch["input_ids"][1, 0].item() == tokenizer.denoise_tags["ko"]
    assert batch["labels"][1, :3].tolist() == [50, 51, tokenizer.eos_id]
    assert batch["source_language_tag_ids"][1].item() == -1
    assert batch["reverse_direction_trained"][1].item() is False
