"""설정 검증 로직 확인."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from sion_translate.config import (
    AppConfig,
    DataConfig,
    ExperimentalConfig,
    config_from_raw,
    load_config,
)


def _warnings_from(config: ExperimentalConfig) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.validate()
    return [str(entry.message) for entry in caught]


def test_default_data_paths_use_the_current_compatible_artifact_layout() -> None:
    config = DataConfig()
    assert config.tokenizer_model == "artifacts/sion-v6/tokenizer/sion.model"
    assert config.tokenizer_features == "artifacts/sion-v6/tokenizer/token_features.npz"
    assert config.dataset_dir == "artifacts/sion-v6/dataset"


def test_shipped_configs_never_point_at_the_legacy_artifact_layout() -> None:
    config_root = Path(__file__).resolve().parents[1] / "configs"
    for config_path in sorted(config_root.glob("*.yaml")):
        data = load_config(config_path).data
        assert data.tokenizer_model.startswith("artifacts/sion-v6/"), config_path
        assert data.tokenizer_features.startswith("artifacts/sion-v6/"), config_path
        assert data.dataset_dir.startswith("artifacts/sion-v6/"), config_path


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


def test_builtin_training_contract_rejects_unsupplied_bats_alignments() -> None:
    config = AppConfig()
    config.model.experimental = ExperimentalConfig(
        bats_enabled=True,
        bats_loss_weight=0.05,
        bats_coverage_weight=0.0,
    )

    with pytest.raises(ValueError, match="alignment_targets"):
        config.validate_training_supervision(alignment_targets_available=False)


def test_custom_alignment_provider_can_enable_bats_alignment_loss() -> None:
    config = AppConfig()
    config.model.experimental = ExperimentalConfig(
        bats_enabled=True,
        bats_loss_weight=0.05,
        bats_coverage_weight=0.0,
    )

    config.validate_training_supervision(alignment_targets_available=True)


def test_bats_coverage_only_needs_no_external_alignment_labels() -> None:
    config = AppConfig()
    config.model.experimental = ExperimentalConfig(
        bats_enabled=True,
        bats_loss_weight=0.0,
        bats_coverage_weight=0.01,
    )

    config.validate_training_supervision(alignment_targets_available=False)


def test_warns_when_core_is_enabled_without_register_loss_weight() -> None:
    messages = _warnings_from(ExperimentalConfig(core_enabled=True, register_loss_weight=0.0))
    assert len(messages) == 1
    assert "core_enabled" in messages[0]


def test_no_core_warning_with_default_register_loss_weight() -> None:
    assert _warnings_from(ExperimentalConfig(core_enabled=True)) == []


def test_root_config_gives_every_enabled_module_a_training_signal() -> None:
    """A module that is on must have a non-zero weight, or it is pure cost.

    This is the invariant worth pinning. Which modules are on is a per-run
    decision - the from-scratch run narrowed it to CoRe so a change in quality
    has one candidate cause - but any module that *is* on has to be learning
    something.
    """

    root_config = Path(__file__).resolve().parents[1] / "sion_translate.yaml"
    experimental = load_config(root_config).model.experimental

    if experimental.bats_enabled:
        assert experimental.bats_coverage_weight > 0 or experimental.bats_loss_weight > 0
    if experimental.core_enabled:
        assert experimental.register_loss_weight > 0
    if experimental.semantic_parity_enabled:
        assert experimental.semantic_parity_loss_weight > 0


def test_root_config_keeps_the_experimental_surface_small() -> None:
    """Several modules at once makes a quality change impossible to attribute.

    The yaml says to enable one at a time and used to contradict itself by
    enabling five. SiTU-GLU is excluded from the count: it reshapes an existing
    activation rather than adding a module, so it cannot be the thing that moved
    a score on its own.
    """

    root_config = Path(__file__).resolve().parents[1] / "sion_translate.yaml"
    experimental = load_config(root_config).model.experimental

    enabled = [
        name
        for name, active in (
            ("bats", experimental.bats_enabled),
            ("core", experimental.core_enabled),
            ("tetm", experimental.tetm_enabled),
            ("morphoscript", experimental.morphoscript_enabled),
            ("evidence_repair", experimental.evidence_repair_enabled),
            ("semantic_parity", experimental.semantic_parity_enabled),
        )
        if active
    ]
    assert len(enabled) <= 2, enabled


def test_negative_loss_weight_is_still_an_error() -> None:
    with pytest.raises(ValueError, match="bats_loss_weight"):
        ExperimentalConfig(bats_loss_weight=-0.1).validate()


@pytest.mark.parametrize(
    "field",
    (
        "evidence_uncertainty_loss_weight",
        "evidence_budget_loss_weight",
        "evidence_repair_gain_loss_weight",
        "semantic_parity_loss_weight",
    ),
)
def test_new_auxiliary_loss_weights_must_be_non_negative(field: str) -> None:
    config = ExperimentalConfig()
    setattr(config, field, -0.1)
    with pytest.raises(ValueError, match=field):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("semantic_parity_dim", 0),
        ("semantic_parity_temperature", 0.0),
    ),
)
def test_semantic_parity_shape_and_temperature_must_be_positive(
    field: str,
    value: float,
) -> None:
    config = ExperimentalConfig()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        config.validate()


@pytest.mark.parametrize("target", (-0.01, 1.01))
def test_evidence_budget_target_must_be_a_probability(target: float) -> None:
    with pytest.raises(ValueError, match="evidence_budget_target"):
        ExperimentalConfig(evidence_budget_target=target).validate()


def test_evidence_minimum_gain_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="evidence_minimum_gain"):
        ExperimentalConfig(evidence_minimum_gain=-0.01).validate()


@pytest.mark.parametrize("field", ("situglu_gate_beta", "situglu_up_beta"))
def test_situglu_caps_must_be_positive(field: str) -> None:
    config = ExperimentalConfig(situglu_enabled=True)
    setattr(config, field, 0.0)
    with pytest.raises(ValueError, match="SiTU-GLU"):
        config.validate()


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


@pytest.mark.parametrize(
    "metric",
    ("global_nll", "macro_direction_nll", "worst_direction_nll"),
)
def test_supported_sft_selection_metrics_are_valid(metric: str) -> None:
    config = AppConfig()
    config.training.sft_selection_metric = metric
    config.validate()


def test_unknown_sft_selection_metric_is_rejected() -> None:
    config = AppConfig()
    config.training.sft_selection_metric = "bleu"
    with pytest.raises(ValueError, match="sft_selection_metric"):
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
