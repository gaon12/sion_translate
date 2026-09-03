"""Verify CPU meta initialization without unsupported DTensor random operators."""

from datetime import timedelta
from pathlib import Path
import warnings

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sion_translate.config import ModelConfig
from sion_translate.model.transformer import SionForConditionalGeneration
from sion_translate.training.distributed import DistributedContext, parallelize_model


def _cpu_initialization_worker(rank: int, rendezvous: str) -> None:
    warnings.simplefilter("error", UserWarning)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        config = ModelConfig(
            vocab_size=32,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            num_heads=4,
            num_kv_heads=2,
            d_ff=32,
            max_seq_len=16,
            dropout=0.0,
        )
        reference = SionForConditionalGeneration(config)
        torch.manual_seed(1729)
        reference.init_weights()
        expected = reference.state_dict()
        with torch.device("meta"):
            model = SionForConditionalGeneration(config)
        # Deliberately different rank seeds must not produce mismatched shards.
        torch.manual_seed(1729 + rank)
        model = parallelize_model(
            model,
            DistributedContext(rank, rank, 2, torch.device("cpu"), True, "gloo"),
            strategy="fsdp2",
            precision="fp32",
            reshard_after_forward=True,
            materialize_meta=True,
        )
        from torch.distributed.tensor import DTensor

        for name, value in model.state_dict().items():
            actual = value.full_tensor() if isinstance(value, DTensor) else value
            torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_cpu_fsdp_meta_initialization_synchronizes_different_rank_seeds(tmp_path: Path) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU FSDP initialization requires Gloo")
    rendezvous = (tmp_path / "initialization-rendezvous").resolve().as_uri()
    context = mp.spawn(_cpu_initialization_worker, args=(rendezvous,), nprocs=2, join=False)
    try:
        for process in context.processes:
            process.join(timeout=60)
        assert all(not process.is_alive() for process in context.processes), "CPU workers hung"
        context.join()
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
