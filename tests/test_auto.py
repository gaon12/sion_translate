"""Tests for automatic configuration and exponential moving averages."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sion_translate.auto as auto_module
from sion_translate.auto import (
    EnvironmentInfo,
    MODEL_SIZE_KEYS,
    _all_devices_support_native_bf16,
    apply_auto_settings,
    backup_stale_dataset,
    buffered_model_promotion_threshold,
    pick_model_preset,
    pick_parallel_strategy,
    pick_vocab_size,
    scan_raw_data,
    stored_fingerprint,
    synchronize_environment,
    write_fingerprint,
)
from sion_translate.config import AppConfig
from sion_translate.training.ema import EMAWeights


def cpu_environment() -> EnvironmentInfo:
    return EnvironmentInfo(
        cuda=False,
        world_size=1,
        device_count=0,
        device_name="CPU",
        min_vram_gib=0.0,
        bf16=False,
        cpu_count=8,
        os_name="Windows",
    )


def gpu_environment(vram: float = 24.0, world_size: int = 8) -> EnvironmentInfo:
    return EnvironmentInfo(
        cuda=True,
        world_size=world_size,
        device_count=world_size,
        device_name="Test GPU",
        min_vram_gib=vram,
        bf16=True,
        cpu_count=32,
        os_name="Linux",
    )


def explicit_architecture(config: AppConfig) -> dict[str, int]:
    """Return one complete architecture override for auto-setting tests."""

    return {key: int(getattr(config.model, key)) for key in MODEL_SIZE_KEYS}


def test_stale_dataset_backups_cannot_collide_or_nest(tmp_path: Path) -> None:
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    (dataset / "generation.txt").write_text("first", encoding="utf-8")
    first = backup_stale_dataset(dataset)
    dataset.mkdir()
    (dataset / "generation.txt").write_text("second", encoding="utf-8")
    second = backup_stale_dataset(dataset)

    assert first != second
    assert first.parent == second.parent == tmp_path
    assert (first / "generation.txt").read_text(encoding="utf-8") == "first"
    assert (second / "generation.txt").read_text(encoding="utf-8") == "second"
    assert not (first / second.name).exists()


def test_distributed_environment_uses_the_least_capable_rank(monkeypatch) -> None:
    context = SimpleNamespace(distributed=True, device=torch.device("cpu"))

    def reduce_to_cluster_minimum(values: torch.Tensor, **_kwargs: object) -> None:
        values.copy_(torch.tensor([40.0, 0.0], dtype=values.dtype))

    monkeypatch.setattr(auto_module.torch.distributed, "all_reduce", reduce_to_cluster_minimum)
    synchronized = synchronize_environment(
        gpu_environment(vram=80.0),
        context,
    )
    assert synchronized.min_vram_gib == 40.0
    assert synchronized.bf16 is False


def test_single_process_environment_needs_no_collective(monkeypatch) -> None:
    environment = gpu_environment(vram=80.0, world_size=1)
    monkeypatch.setattr(
        auto_module.torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single-process probing must not communicate")
        ),
    )
    assert (
        synchronize_environment(
            environment,
            SimpleNamespace(distributed=False, device=torch.device("cpu")),
        )
        is environment
    )


def test_model_preset_scales_with_data() -> None:
    assert pick_model_preset(50_000)[1]["d_model"] == 512
    assert pick_model_preset(1_000_000)[1]["d_model"] == 640
    assert pick_model_preset(11_000_000)[1]["d_model"] == 768
    assert pick_model_preset(50_000_000)[1]["d_model"] == 1024
    assert pick_model_preset(200_000_000)[1]["d_model"] == 1280
    assert pick_vocab_size(50_000) == 16_000
    assert pick_vocab_size(11_000_000) == 48_000


@pytest.mark.parametrize(
    ("pair_count", "expected_name"),
    [
        (209_999, "small"),
        (210_000, "medium"),
        (3_149_999, "medium"),
        (3_150_000, "base"),
        (31_499_999, "base"),
        (31_500_000, "large"),
        (104_999_999, "large"),
        (105_000_000, "xlarge"),
    ],
)
def test_model_preset_promotes_only_after_the_five_percent_buffer(
    pair_count: int,
    expected_name: str,
) -> None:
    assert pick_model_preset(pair_count)[0] == expected_name


def test_half_million_pair_growth_near_the_nominal_base_boundary_does_not_promote() -> None:
    assert pick_model_preset(29_750_000)[0] == "base"
    assert pick_model_preset(30_250_000)[0] == "base"
    assert buffered_model_promotion_threshold(30_000_000) == 31_500_000


@pytest.mark.parametrize("invalid", [-1, True, False, 1.5])
def test_model_preset_rejects_invalid_pair_counts(invalid: object) -> None:
    expected = ValueError if invalid == -1 else TypeError
    with pytest.raises(expected):
        pick_model_preset(invalid)  # type: ignore[arg-type]


def test_model_preset_has_an_unbounded_deterministic_final_tier() -> None:
    first_name, first_values = pick_model_preset(10**30)
    first_values["d_model"] = -1
    second_name, second_values = pick_model_preset(10**30)

    assert first_name == second_name == "xlarge"
    assert second_values["d_model"] == 1280


def test_auto_settings_fill_unspecified_fields() -> None:
    config = AppConfig()
    config.data.language_pair = ["ko", "ja"]
    decisions = apply_auto_settings(
        config,
        raw={},
        env=gpu_environment(),
        train_examples=22_000_000,  # Includes both directions for 11M physical pairs.
        validation_examples=110_000,
    )
    assert decisions
    assert config.model.d_model == 768  # 11M pairs select the base preset.
    assert config.training.precision == "bf16"
    assert config.model.gradient_checkpointing is True
    assert config.training.batch_size_per_gpu == 8  # Selected for 24 GiB.
    # Accumulation keeps the effective batch at the target of 256.
    effective = config.training.batch_size_per_gpu * 8 * config.training.gradient_accumulation_steps
    assert effective == 256
    assert config.training.num_train_epochs == 3
    assert config.training.max_steps is None
    assert config.training.warmup_steps <= 4000
    config.validate()


def test_auto_sizing_uses_physical_pairs_for_a_mixed_direction_graph() -> None:
    config = AppConfig()

    apply_auto_settings(
        config,
        raw={},
        env=cpu_environment(),
        train_examples=20_000_000,
        validation_examples=100_000,
        physical_train_pairs=2_000_000,
    )

    # 2M physical pairs select the medium preset. Inferring pair count from the
    # 20M virtual directions would incorrectly select the larger base preset.
    assert config.model.d_model == 640


def test_auto_settings_respect_user_values() -> None:
    config = AppConfig()
    config.training.max_steps = 777
    config.model.d_model = 512
    raw = {
        "training": {"max_steps": 777},
        "model": explicit_architecture(config),
    }
    apply_auto_settings(
        config,
        raw=raw,
        env=cpu_environment(),
        train_examples=1000,
        validation_examples=100,
    )
    # Automatic settings never overwrite explicit user values.
    assert config.training.max_steps == 777
    assert config.model.d_model == 512


def test_auto_settings_reject_partial_manual_architecture_overrides() -> None:
    config = AppConfig()
    with pytest.raises(ValueError, match="every preset-defining key"):
        apply_auto_settings(
            config,
            raw={"model": {"d_model": 768}},
            env=cpu_environment(),
            train_examples=1_000_000,
            validation_examples=10_000,
        )


def test_h100_defaults_leave_memory_headroom_without_checkpointing() -> None:
    config = AppConfig()
    apply_auto_settings(
        config,
        raw={},
        env=gpu_environment(vram=80.0, world_size=1),
        train_examples=22_000_000,
        validation_examples=110_000,
    )
    assert config.model.gradient_checkpointing is False
    assert config.training.batch_size_per_gpu == 32
    assert config.training.gradient_accumulation_steps == 8


def test_a100_40g_defaults_trade_compute_for_memory_headroom() -> None:
    config = AppConfig()
    apply_auto_settings(
        config,
        raw={},
        env=gpu_environment(vram=40.0, world_size=1),
        train_examples=22_000_000,
        validation_examples=110_000,
    )
    assert config.model.gradient_checkpointing is True
    assert config.training.batch_size_per_gpu == 16
    assert config.training.gradient_accumulation_steps == 16
    assert config.training.compile is False


def test_native_bf16_requires_every_visible_device_to_support_it() -> None:
    ampere = SimpleNamespace(major=8)
    hopper = SimpleNamespace(major=9)
    turing = SimpleNamespace(major=7)

    assert _all_devices_support_native_bf16([ampere, hopper])
    assert not _all_devices_support_native_bf16([ampere, turing])
    assert not _all_devices_support_native_bf16([])


def test_multi_h100_prefers_ddp_and_bf16_collectives() -> None:
    env = gpu_environment(vram=80.0, world_size=4)
    assert pick_parallel_strategy(env, d_model=768) == "ddp"
    config = AppConfig()
    apply_auto_settings(
        config,
        raw={},
        env=env,
        train_examples=22_000_000,
        validation_examples=110_000,
    )
    assert config.training.parallel_strategy == "ddp"
    assert config.training.fsdp_reduce_dtype == "bf16"
    assert config.data.num_workers <= 16


def test_memory_constrained_large_model_selects_fsdp2() -> None:
    env = gpu_environment(vram=24.0, world_size=4)
    assert pick_parallel_strategy(env, d_model=1024) == "fsdp2"


def test_explicit_auto_parallel_strategy_uses_environment_picker() -> None:
    config = AppConfig()
    config.model.d_model = 1024
    apply_auto_settings(
        config,
        raw={
            "model": explicit_architecture(config),
            "training": {"parallel_strategy": "auto"},
        },
        env=gpu_environment(vram=24.0, world_size=4),
        train_examples=22_000_000,
        validation_examples=110_000,
    )
    assert config.training.parallel_strategy == "fsdp2"


def test_multi_h100_fsdp2_keeps_resharding_unless_explicitly_disabled() -> None:
    env = gpu_environment(vram=80.0, world_size=4)
    config = AppConfig()
    config.model.d_model = 1280
    apply_auto_settings(
        config,
        raw={"model": explicit_architecture(config)},
        env=env,
        train_examples=400_000_000,
        validation_examples=110_000,
    )
    assert config.training.parallel_strategy == "fsdp2"
    assert config.training.reshard_after_forward is True

    explicit = AppConfig()
    explicit.model.d_model = 1280
    explicit.training.reshard_after_forward = False
    apply_auto_settings(
        explicit,
        raw={
            "model": explicit_architecture(explicit),
            "training": {"reshard_after_forward": False},
        },
        env=env,
        train_examples=400_000_000,
        validation_examples=110_000,
    )
    assert explicit.training.parallel_strategy == "fsdp2"
    assert explicit.training.reshard_after_forward is False


def test_auto_downweights_synthetic_sources() -> None:
    config = AppConfig()
    decisions = apply_auto_settings(
        config,
        raw={},
        env=cpu_environment(),
        train_examples=2000,
        validation_examples=100,
        source_names=[
            "real.jsonl",
            "bt_news.jsonl",
            "concat_dialogue.jsonl",
            "revise_legal.jsonl",
            "synthetic_numeric_data38.jsonl",
        ],
    )
    assert config.data.source_sampling_weights == {
        "bt_news.jsonl": 0.5,
        "concat_dialogue.jsonl": 0.5,
        "revise_legal.jsonl": 0.5,
        "synthetic_numeric_data38.jsonl": 0.5,
    }
    assert any("Synthetic sampling" in line for line in decisions)
    # Preserve an explicit user sampling policy.
    config2 = AppConfig()
    config2.data.source_sampling_weights = {"bt_news.jsonl": 0.9}
    apply_auto_settings(
        config2,
        raw={"data": {"source_sampling_weights": {"bt_news.jsonl": 0.9}}},
        env=cpu_environment(),
        train_examples=2000,
        validation_examples=100,
        source_names=["real.jsonl", "bt_news.jsonl"],
    )
    assert config2.data.source_sampling_weights == {"bt_news.jsonl": 0.9}


def test_epoch_budget_shrinks_with_data() -> None:
    from sion_translate.auto import target_epochs

    assert target_epochs(100_000) > target_epochs(10_000_000) > target_epochs(200_000_000)
    assert target_epochs(200_000_000) >= 2


def test_fingerprint_detects_data_changes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.jsonl").write_text("x\n", encoding="utf-8")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    tokenizer = tmp_path / "sion.model"
    tokenizer.write_bytes(b"tokenizer-a")
    first = scan_raw_data(
        data_dir,
        language_pairs=(("ko", "ja"),),
        tokenizer_model=tokenizer,
        preprocessing_options={"max_tokens_per_side": 510},
    )
    write_fingerprint(dataset_dir, first)
    assert stored_fingerprint(dataset_dir) == first
    assert dict(first) == {"a.jsonl": (data_dir / "a.jsonl").stat().st_size}
    assert not (dataset_dir / "raw_fingerprint.json.tmp").exists()

    # A same-size content change still changes SHA-256 and forces preparation.
    (data_dir / "a.jsonl").write_text("y\n", encoding="utf-8")
    changed = scan_raw_data(
        data_dir,
        language_pairs=(("ko", "ja"),),
        tokenizer_model=tokenizer,
        preprocessing_options={"max_tokens_per_side": 510},
    )
    assert changed != stored_fingerprint(dataset_dir)


def test_fingerprint_covers_tokenizer_languages_and_preprocessing_schema(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.jsonl").write_text("{}\n", encoding="utf-8")
    tokenizer = tmp_path / "sion.model"
    tokenizer.write_bytes(b"first")
    base = scan_raw_data(
        data_dir,
        language_pairs=(("ko", "ja"),),
        tokenizer_model=tokenizer,
        preprocessing_schema="prepare-v1",
    )
    assert base != scan_raw_data(
        data_dir,
        language_pairs=(("en", "ru"),),
        tokenizer_model=tokenizer,
        preprocessing_schema="prepare-v1",
    )
    assert base != scan_raw_data(
        data_dir,
        language_pairs=(("ko", "ja"),),
        tokenizer_model=tokenizer,
        preprocessing_schema="prepare-v2",
    )
    tokenizer.write_bytes(b"other")
    assert base != scan_raw_data(
        data_dir,
        language_pairs=(("ko", "ja"),),
        tokenizer_model=tokenizer,
        preprocessing_schema="prepare-v1",
    )


def test_legacy_size_only_fingerprint_forces_a_safe_rebuild(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "a.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "raw_fingerprint.json").write_text(
        json.dumps({"a.jsonl": source.stat().st_size}),
        encoding="utf-8",
    )

    current = scan_raw_data(data_dir)
    assert stored_fingerprint(dataset_dir) != current


def test_ema_tracks_and_swaps() -> None:
    model = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = EMAWeights(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    # shadow = 0.5*0 + 0.5*1 = 0.5
    shadow = ema.shadow["weight"]
    assert torch.allclose(shadow, torch.full_like(shadow, 0.5))
    # The swap block exposes EMA values and restores the original values on exit.
    with ema.swap(model):
        assert torch.allclose(model.weight, torch.full_like(model.weight, 0.5))
    assert torch.allclose(model.weight, torch.full_like(model.weight, 1.0))


def test_ema_swap_restores_model_and_shadow_after_an_exception() -> None:
    model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = EMAWeights(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    original_shadow = ema.shadow["weight"].clone()

    with pytest.raises(RuntimeError, match="evaluation failed"):
        with ema.swap(model):
            torch.testing.assert_close(
                model.weight,
                torch.full_like(model.weight, 2.0),
            )
            torch.testing.assert_close(
                ema.shadow["weight"],
                torch.full_like(model.weight, 3.0),
            )
            raise RuntimeError("evaluation failed")

    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 3.0))
    torch.testing.assert_close(ema.shadow["weight"], original_shadow)
    ema.copy_to(model)
    torch.testing.assert_close(model.weight, original_shadow)
