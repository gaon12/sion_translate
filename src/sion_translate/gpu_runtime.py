"""Bounded CUDA kernel preflight used before paid training starts."""

# CUDA, NCCL, and fused optimizers expose runtime-generated extension types.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import torch

from sion_translate.model.layers import GQAAttention, RotaryEmbedding
from sion_translate.training.trainer import build_optimizer_param_groups


def _require_finite(name: str, *tensors: torch.Tensor) -> None:
    """Raise with the failing phase name instead of allowing corrupt state."""

    if not tensors or any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise FloatingPointError(f"CUDA canary produced a non-finite {name}")


def _run_nccl_canary(device: torch.device) -> bool:
    """Initialize NCCL and complete one all-reduce when this build provides it."""

    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        return False
    with tempfile.TemporaryDirectory(prefix="sion-nccl-canary-") as temporary:
        rendezvous = (Path(temporary) / "store").resolve().as_uri()
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=rendezvous,
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=15),
        )
        try:
            value = torch.ones((), device=device)
            torch.distributed.all_reduce(value)
            torch.cuda.synchronize(device)
            if float(value.item()) != 1.0:
                raise RuntimeError("NCCL canary returned the wrong all-reduce value")
        finally:
            torch.distributed.destroy_process_group()
    return True


def run_cuda_canary(device_index: int) -> dict[str, Any]:
    """Exercise the production GQA, BF16, backward, AdamW, and NCCL paths."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    gpu_count = torch.cuda.device_count()
    if not 0 <= device_index < gpu_count:
        raise ValueError(f"CUDA device index {device_index} is outside 0..{gpu_count - 1}")
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError(f"CUDA device {device_index} does not support BF16")

    torch.manual_seed(20260831 + device_index)
    torch.cuda.manual_seed_all(20260831 + device_index)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    attention = GQAAttention(
        d_model=864,
        num_heads=12,
        num_kv_heads=6,
        dropout=0.0,
        qk_norm=True,
        norm_eps=1e-6,
        rope=RotaryEmbedding(head_dim=72, max_seq_len=64),
    ).to(device)
    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(attention, weight_decay=0.1),
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        fused=True,
    )
    if optimizer.defaults.get("fused") is not True:
        raise RuntimeError("the CUDA canary could not enable fused AdamW")
    hidden = torch.randn(2, 32, 864, device=device)
    key_padding_mask = torch.ones(2, 32, dtype=torch.bool, device=device)
    key_padding_mask[0, -5:] = False
    before = attention.q_proj.weight.detach().clone()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = attention(
            hidden,
            key_padding_mask=key_padding_mask,
            is_causal=True,
        )
        matrix = output[:, :8].transpose(1, 2) @ output[:, :8]
        loss = output.float().square().mean() + matrix.float().square().mean() * 1e-4
    _require_finite("attention output or BF16 matrix product", output, matrix, loss)
    loss.backward()
    gradients = [
        parameter.grad for parameter in attention.parameters() if parameter.grad is not None
    ]
    _require_finite("gradient", *gradients)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        attention.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    _require_finite("updated parameter", *attention.parameters())
    optimizer_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    _require_finite("fused AdamW state", *optimizer_tensors)
    if torch.equal(before, attention.q_proj.weight):
        raise RuntimeError("fused AdamW did not update the production GQA parameters")
    nccl_completed = _run_nccl_canary(device)
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    elapsed = time.perf_counter() - started
    return {
        "schema": "sion-cuda-canary-v1",
        "status": "passed",
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": int(properties.total_memory),
        "torch": str(torch.__version__),
        "compiled_cuda": str(torch.version.cuda),
        "bf16": True,
        "gqa_query_heads": 12,
        "gqa_kv_heads": 6,
        "head_dimension": 72,
        "gradient_norm": float(gradient_norm.item()),
        "nccl_all_reduce": nccl_completed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(run_cuda_canary(args.device_index), sort_keys=True, allow_nan=False), flush=True
    )


if __name__ == "__main__":
    main()
