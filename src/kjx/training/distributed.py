from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator

import torch
import torch.distributed as dist
from torch import nn

from kjx.model.layers import DecoderLayer, EncoderLayer


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool
    backend: str | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed:
        backend = (
            "nccl"
            if device.type == "cuda" and dist.is_nccl_available()
            else "gloo"
        )
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    else:
        backend = None
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    return DistributedContext(rank, local_rank, world_size, device, distributed, backend)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()


def reduce_sum(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast_bool(value: bool, context: DistributedContext, *, source: int = 0) -> bool:
    """Broadcast a control-flow decision so every rank exits at the same point."""

    if not context.distributed:
        return value
    decision = torch.tensor(
        int(value) if context.rank == source else 0,
        device=context.device,
        dtype=torch.int32,
    )
    dist.broadcast(decision, src=source)
    return bool(decision.item())


def precision_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    normalized = name.lower()
    if normalized == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 precision was requested, but this CUDA device does not support it")
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    if normalized == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {name}")


def parallelize_model(
    model: nn.Module,
    context: DistributedContext,
    *,
    use_fsdp2: bool,
    precision: str,
    reshard_after_forward: bool,
    materialize_meta: bool,
    find_unused_parameters: bool = False,
) -> nn.Module:
    dtype = precision_dtype(precision, context.device)
    if context.distributed and use_fsdp2:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        policy = MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            output_dtype=dtype,
            cast_forward_inputs=True,
        )
        for module in model.modules():
            if isinstance(module, (EncoderLayer, DecoderLayer)):
                fully_shard(
                    module,
                    reshard_after_forward=reshard_after_forward,
                    mp_policy=policy,
                )
        fully_shard(
            model,
            reshard_after_forward=reshard_after_forward,
            mp_policy=policy,
        )
        if materialize_meta:
            model.to_empty(device=context.device)
            model.init_weights()
        return model

    if materialize_meta:
        model.to_empty(device=context.device)
        model.init_weights()
    else:
        # Keep full-precision master parameters for ordinary single-GPU/DDP
        # training. The trainer applies CUDA autocast for bf16/fp16 compute.
        model.to(device=context.device)
    if context.distributed:
        from torch.nn.parallel import DistributedDataParallel

        # find_unused_parameters 는 forward 에서 안 쓰인 파라미터를 매 step
        # 탐색하는 비용이 있으므로, 실험 모듈처럼 조건부로만 쓰이는 파라미터가
        # 있을 때만 켭니다 (호출부에서 설정 기준으로 결정).
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            find_unused_parameters=find_unused_parameters,
        )
    return model


@contextmanager
def maybe_no_sync(model: nn.Module, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    if hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            model.set_requires_gradient_sync(True)
        return
    if hasattr(model, "no_sync"):
        with model.no_sync():
            yield
        return
    yield
