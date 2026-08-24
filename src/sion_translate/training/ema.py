"""Exponential moving average (EMA) of model weights.

Training weights fluctuate from step to step. EMA smooths those fluctuations by
maintaining a shadow copy that follows the weight trajectory:

    shadow = decay × shadow + (1 - decay) × current weights   (each step)

For translation models, evaluation and deployment with EMA (or checkpoint
averaging) commonly improves validation loss and BLEU more consistently than
raw weights. It is therefore enabled by default through training.ema_decay;
setting that value to 0 disables it.

In FSDP2, parameters are sharded DTensors. Shadow tensors use the same sharding,
so each rank allocates only its local shard. Elementwise operations such as
lerp_ and copy_ work directly on DTensors.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from typing import Iterator, cast

import torch
from torch import nn


def _named_parameters_below_compile(model: nn.Module):
    """Yield stable names from below any torch.compile wrapper."""

    unwrapped = model
    while True:
        original = getattr(unwrapped, "_orig_mod", None)
        if not isinstance(original, nn.Module):
            break
        unwrapped = original
    return unwrapped.named_parameters()


class EMAWeights:
    """Maintain an EMA shadow copy of trainable model parameters.

    Typical use:
        ema = EMAWeights(model, decay=0.999)
        ...  # after each successful optimizer.step()
        ema.update(model)
        ...  # when evaluating or exporting EMA weights
        with ema.swap(model):
            evaluate(model, ...)   # model uses EMA weights in this block
    """

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        # Track trainable parameters only; this model has no learned-state buffers.
        self.shadow: dict[str, torch.Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in _named_parameters_below_compile(model)
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update each shadow after a successful optimizer step."""
        one_minus_decay = 1.0 - self.decay
        for name, parameter in _named_parameters_below_compile(model):
            shadow = self.shadow.get(name)
            if shadow is not None:
                shadow.lerp_(parameter.detach(), one_minus_decay)

    @staticmethod
    @torch.no_grad()
    def _exchange(parameter: torch.Tensor, shadow: torch.Tensor) -> None:
        """Exchange one parameter with its shadow using one-tensor scratch space."""

        current = parameter.detach().clone()
        parameter.copy_(shadow)
        shadow.copy_(current)

    @contextmanager  # pyright: ignore[reportDeprecated]
    def swap(self, model: nn.Module) -> Iterator[None]:
        """Use EMA weights only inside the context and restore raw weights afterward.

        Parameters and shadows are exchanged one at a time instead of allocating
        a model-sized backup dictionary. Additional peak memory is therefore one
        largest parameter, and raw weights are restored even if evaluation or
        export raises an exception.
        """
        swapped: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            for name, parameter in _named_parameters_below_compile(model):
                shadow = self.shadow.get(name)
                if shadow is not None:
                    self._exchange(parameter, shadow)
                    swapped.append((parameter, shadow))
            yield
        finally:
            for parameter, shadow in reversed(swapped):
                self._exchange(parameter, shadow)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Permanently copy EMA weights into a model without allocating a backup."""

        for name, parameter in _named_parameters_below_compile(model):
            shadow = self.shadow.get(name)
            if shadow is not None:
                parameter.copy_(shadow)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return shadows for checkpointing so resume preserves EMA history."""
        return dict(self.shadow)

    def _validated_state_dict(self, state: object) -> dict[str, torch.Tensor]:
        if not isinstance(state, Mapping):
            raise ValueError("EMA checkpoint state must be an object")
        normalized: dict[str, torch.Tensor] = {}
        for raw_name, raw_tensor in cast(Mapping[object, object], state).items():
            if not isinstance(raw_name, str) or not isinstance(raw_tensor, torch.Tensor):
                raise ValueError("EMA checkpoint state is malformed")
            canonical_name = raw_name
            while canonical_name.startswith("_orig_mod."):
                canonical_name = canonical_name.removeprefix("_orig_mod.")
            if canonical_name in normalized:
                raise ValueError(f"EMA checkpoint contains duplicate key: {canonical_name}")
            normalized[canonical_name] = raw_tensor
        missing = sorted(set(self.shadow) - set(normalized))
        unexpected = sorted(set(normalized) - set(self.shadow))
        if missing or unexpected:
            raise ValueError(
                "EMA checkpoint does not match the model parameters: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        for name, target in self.shadow.items():
            source = normalized[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(
                    "EMA checkpoint tensor metadata mismatch for "
                    f"{name}: checkpoint=(shape={tuple(source.shape)}, dtype={source.dtype}), "
                    f"model=(shape={tuple(target.shape)}, dtype={target.dtype})"
                )
        return normalized

    def validate_state_dict(self, state: object) -> None:
        """Validate a complete EMA snapshot without changing any shadow tensor."""

        self._validated_state_dict(state)

    @torch.no_grad()
    def load_state_dict(self, state: object) -> None:
        normalized = self._validated_state_dict(state)
        for name, tensor in normalized.items():
            canonical_name = name
            self.shadow[canonical_name].copy_(tensor)
