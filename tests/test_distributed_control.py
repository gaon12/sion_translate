from __future__ import annotations

import torch

import sion_translate.training.distributed as distributed_module
from sion_translate.training.distributed import DistributedContext


def _local_context() -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        distributed=False,
        backend=None,
    )


def test_local_failure_scope_and_control_broadcasts_are_exact() -> None:
    context = _local_context()

    assert distributed_module.distributed_failure_scope(False, context) == "none"
    assert distributed_module.distributed_failure_scope(True, context) == "all"
    assert distributed_module.broadcast_int(17, context) == 17
    assert distributed_module.broadcast_text("generation-a", context) == "generation-a"


def test_failure_scope_reports_rank_divergence(monkeypatch) -> None:
    context = _local_context()
    observed = iter((torch.tensor(1), torch.tensor(1)))
    monkeypatch.setattr(
        distributed_module,
        "reduce_max",
        lambda _tensor, _context: next(observed),
    )

    assert distributed_module.distributed_failure_scope(True, context) == "partial"
