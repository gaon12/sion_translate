from __future__ import annotations

import pytest
import torch

from sion_translate.gpu_runtime import _require_finite


def test_cuda_canary_finite_guard_names_the_failed_phase() -> None:
    _require_finite("finite tensor", torch.ones(2))

    with pytest.raises(FloatingPointError, match="attention output"):
        _require_finite("attention output", torch.tensor([1.0, float("nan")]))


def test_cuda_canary_finite_guard_rejects_an_empty_tensor_list() -> None:
    with pytest.raises(FloatingPointError, match="gradient"):
        _require_finite("gradient")
