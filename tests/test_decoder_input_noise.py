"""Teacher forcing never shows the decoder a wrong prefix, so it never learns one.

Post-training samples from the model and does address this, but it runs after the
fact for a few thousand steps. Corrupting the decoder input during the supervised
stage is the cheap in-training mitigation. The invariant that matters: the labels
never change, so the objective is the same and only the conditioning is noisy.
"""

from __future__ import annotations

import random

import pytest
import torch

from sion_translate.data.collate import SionBatchCollator


class FakeTokenizer:
    """Minimal stand-in: the collator only needs ids and a few tag maps."""

    pad_id = 0
    unk_id = 1
    bos_id = 2
    eos_id = 3
    mask_id = 4
    vocab_size = 200

    def __init__(self) -> None:
        self.slot_ids = [10, 11, 12]
        self.language_tags = {"ko": 20, "ja": 21}
        self.denoise_tags = {"ko": 30, "ja": 31}


def make_collator(noise: float) -> SionBatchCollator:
    return SionBatchCollator(
        FakeTokenizer(),
        max_source_length=64,
        max_target_length=64,
        decoder_input_noise=noise,
    )


def test_no_noise_leaves_the_decoder_input_untouched() -> None:
    collator = make_collator(0.0)
    target = [40, 41, 42, 43, 44]
    assert collator._noise_decoder_input(target, random.Random(0)) == target


def test_noise_changes_the_decoder_input() -> None:
    collator = make_collator(1.0)
    target = [40, 41, 42, 43, 44, 45, 46, 47]
    noised = collator._noise_decoder_input(target, random.Random(7))
    assert noised != target
    assert len(noised) == len(target)


def test_protected_slots_survive_the_noise() -> None:
    # A slot exists to guarantee a surface. Corrupting one would train the model
    # to break exactly the guarantee it is there to make.
    collator = make_collator(1.0)
    target = [10, 40, 11, 41, 12]
    noised = collator._noise_decoder_input(target, random.Random(3))
    assert noised[0] == 10
    assert noised[2] == 11
    assert noised[4] == 12


def test_every_noised_token_is_a_valid_id() -> None:
    collator = make_collator(1.0)
    target = list(range(40, 60))
    for seed in range(20):
        for token_id in collator._noise_decoder_input(target, random.Random(seed)):
            assert 0 <= token_id < FakeTokenizer.vocab_size


def test_the_noise_rate_is_roughly_honoured() -> None:
    collator = make_collator(0.30)
    target = [40] * 4000
    noised = collator._noise_decoder_input(target, random.Random(11))
    changed = sum(1 for value in noised if value != 40)
    # Some random draws land back on 40, so the observed rate sits just under the
    # configured one. A wide band still separates 0.30 from 0 and from 1.
    assert 0.20 < changed / len(target) < 0.35


def test_labels_are_never_noised() -> None:
    collator = make_collator(1.0)
    item = {
        "src": [40, 41, 42],
        "tgt": [50, 51, 52],
        "src_language": "ko",
        "target_language": "ja",
        "target_register": 0,
    }
    example = collator._make_example(item)
    # The label sequence is the clean target plus EOS, whatever the input became.
    assert example["labels"] == [50, 51, 52, FakeTokenizer.eos_id]
    assert example["decoder_input_ids"][0] == FakeTokenizer.bos_id
    assert len(example["decoder_input_ids"]) == len(example["labels"])


def test_the_batch_keeps_labels_clean_end_to_end() -> None:
    collator = make_collator(1.0)
    items = [
        {
            "src": [40, 41],
            "tgt": [50, 51],
            "src_language": "ko",
            "target_language": "ja",
            "target_register": 0,
        }
        for _ in range(4)
    ]
    batch = collator(items)
    labels = batch["labels"]
    for row in range(labels.shape[0]):
        kept = [int(value) for value in labels[row] if int(value) != -100]
        assert kept == [50, 51, FakeTokenizer.eos_id]


def test_the_config_rejects_an_out_of_range_rate() -> None:
    from sion_translate.config import AppConfig

    config = AppConfig()
    config.data.decoder_input_noise = 0.5
    with pytest.raises(ValueError, match="decoder_input_noise must be in"):
        config.validate()


def test_the_default_is_off() -> None:
    from sion_translate.config import AppConfig

    # Turning this on untested before a from-scratch run would be a guess.
    assert AppConfig().data.decoder_input_noise == 0.0


def test_noise_is_deterministic_for_a_seed() -> None:
    collator = make_collator(0.5)
    target = list(range(40, 80))
    first = collator._noise_decoder_input(target, random.Random(99))
    second = collator._noise_decoder_input(target, random.Random(99))
    assert first == second


def test_torch_is_available_for_the_batch_path() -> None:
    # Guards against the batch test silently skipping if torch changes shape.
    assert torch.tensor([1]).shape == (1,)
