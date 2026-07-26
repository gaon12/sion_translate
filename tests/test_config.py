"""설정 검증 로직 확인."""

from __future__ import annotations

import warnings

import pytest

from sion_translate.config import ExperimentalConfig


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
    messages = _warnings_from(
        ExperimentalConfig(core_enabled=True, register_loss_weight=0.0)
    )
    assert len(messages) == 1
    assert "core_enabled" in messages[0]


def test_no_core_warning_with_default_register_loss_weight() -> None:
    assert _warnings_from(ExperimentalConfig(core_enabled=True)) == []


def test_negative_loss_weight_is_still_an_error() -> None:
    with pytest.raises(ValueError, match="bats_loss_weight"):
        ExperimentalConfig(bats_loss_weight=-0.1).validate()
