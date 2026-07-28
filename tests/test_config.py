"""설정 검증 로직 확인."""

from __future__ import annotations

import warnings

import pytest

from sion_translate.config import AppConfig, DataConfig, ExperimentalConfig, config_from_raw


def _warnings_from(config: ExperimentalConfig) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.validate()
    return [str(entry.message) for entry in caught]


def test_warns_when_bats_is_enabled_without_any_loss_weight() -> None:
    # 순전파 비용과 파라미터만 늘고 학습 신호는 없는 조합이다.
    messages = _warnings_from(
        ExperimentalConfig(bats_enabled=True, bats_loss_weight=0.0, bats_coverage_weight=0.0)
    )
    assert len(messages) == 1
    assert "bats_enabled" in messages[0]


@pytest.mark.parametrize(
    "weights",
    [{"bats_loss_weight": 0.05}, {"bats_coverage_weight": 0.05}],
)
def test_no_bats_warning_when_either_weight_is_set(weights: dict[str, float]) -> None:
    assert _warnings_from(ExperimentalConfig(bats_enabled=True, **weights)) == []


def test_no_bats_warning_when_module_is_disabled() -> None:
    # 꺼져 있으면 가중치가 0인 것이 정상이다.
    assert _warnings_from(ExperimentalConfig(bats_enabled=False)) == []


def test_warns_when_core_is_enabled_without_register_loss_weight() -> None:
    messages = _warnings_from(ExperimentalConfig(core_enabled=True, register_loss_weight=0.0))
    assert len(messages) == 1
    assert "core_enabled" in messages[0]


def test_no_core_warning_with_default_register_loss_weight() -> None:
    assert _warnings_from(ExperimentalConfig(core_enabled=True)) == []


def test_negative_loss_weight_is_still_an_error() -> None:
    with pytest.raises(ValueError, match="bats_loss_weight"):
        ExperimentalConfig(bats_loss_weight=-0.1).validate()


@pytest.mark.parametrize(
    ("field", "value", "minimum"),
    [
        ("tetm_types", 8, 9),
        ("tetm_modes", 4, 5),
    ],
)
def test_tetm_requires_protected_slot_type_and_mode_capacity(
    field: str,
    value: int,
    minimum: int,
) -> None:
    config = ExperimentalConfig(tetm_enabled=True)
    setattr(config, field, value)
    with pytest.raises(ValueError, match=rf"{field} must be at least {minimum}"):
        config.validate()


def test_disabled_tetm_allows_smaller_positive_embedding_tables() -> None:
    ExperimentalConfig(
        tetm_enabled=False,
        tetm_types=1,
        tetm_modes=1,
    ).validate()


def test_multilingual_pairs_are_validated_and_expose_language_union() -> None:
    config = AppConfig(data=DataConfig(language_pairs=[["ko", "ja"], ["en", "ru"]]))
    config.validate()
    assert config.data.configured_language_pairs() == (("ko", "ja"), ("en", "ru"))
    assert config.data.languages == ("ko", "ja", "en", "ru")


def test_single_and_multi_pair_keys_cannot_be_configured_together() -> None:
    with pytest.raises(ValueError, match="cannot both"):
        config_from_raw(
            {
                "data": {
                    "language_pair": ["ko", "ja"],
                    "language_pairs": [["en", "ru"]],
                }
            }
        )


def test_reversed_multilingual_pair_is_rejected() -> None:
    config = AppConfig(data=DataConfig(language_pairs=[["ko", "ja"], ["ja", "ko"]]))
    with pytest.raises(ValueError, match="duplicate or reversed"):
        config.validate()


@pytest.mark.parametrize(
    "field",
    ("candidate_micro_batch", "eval_batch_size_per_gpu"),
)
def test_posttraining_memory_batch_sizes_must_be_positive(field: str) -> None:
    config = AppConfig()
    setattr(config.posttraining, field, 0)
    with pytest.raises(ValueError, match=field):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("roundtrip_reward_weight", -0.1),
        ("roundtrip_failure_penalty", 1.1),
        ("roundtrip_min_score", -0.1),
        ("roundtrip_num_beams", 0),
        ("roundtrip_max_new_tokens", 0),
    ],
)
def test_roundtrip_posttraining_parameters_are_validated(field: str, value: float) -> None:
    config = AppConfig()
    setattr(config.posttraining, field, value)
    with pytest.raises(ValueError, match=field):
        config.validate()


def test_enabled_roundtrip_requires_a_positive_reward_weight() -> None:
    config = AppConfig()
    config.posttraining.roundtrip_enabled = True
    config.posttraining.roundtrip_reward_weight = 0.0
    with pytest.raises(ValueError, match="roundtrip_reward_weight"):
        config.validate()


def test_padding_multiple_must_be_positive() -> None:
    config = AppConfig()
    config.data.pad_to_multiple_of = 0
    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        config.validate()


def test_final_export_formats_cover_all_requested_deployment_precisions() -> None:
    config = AppConfig()
    assert config.training.final_export_formats == [
        "fp32",
        "fp16",
        "bf16",
        "int8",
        "int4",
        "gguf_q4_k_m",
        "transformers",
    ]
    config.validate()


@pytest.mark.parametrize(
    "formats, message",
    [
        ([], "at least one"),
        (["fp32", "fp32"], "duplicates"),
        (["onnx"], "unsupported"),
    ],
)
def test_final_export_formats_are_validated(
    formats: list[str],
    message: str,
) -> None:
    config = AppConfig()
    config.training.final_export_formats = formats
    with pytest.raises(ValueError, match=message):
        config.validate()
