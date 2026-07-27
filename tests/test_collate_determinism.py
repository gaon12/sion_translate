from __future__ import annotations

import random
from collections.abc import Iterable

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
