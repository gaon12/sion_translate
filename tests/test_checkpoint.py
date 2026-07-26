from __future__ import annotations

from pathlib import Path

import torch

from sion_translate.config import ExperimentalConfig, ModelConfig
from sion_translate.model import SionForConditionalGeneration
from sion_translate.training.checkpoint import load_checkpoint, save_checkpoint
from sion_translate.training.distributed import DistributedContext
from sion_translate.training.trainer import cosine_scheduler


class FakeScaler:
    def __init__(self, scale: float):
        self.scale = scale

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = state["scale"]


def test_local_checkpoint_round_trip(tmp_path: Path) -> None:
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        max_seq_len=16,
        gradient_checkpointing=False,
        experimental=ExperimentalConfig(
            bats_enabled=False,
            core_enabled=False,
            tetm_enabled=False,
            morphoscript_enabled=False,
        ),
    )
    model = SionForConditionalGeneration(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = cosine_scheduler(optimizer, warmup_steps=1, max_steps=10, min_ratio=0.1)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    checkpoint = tmp_path / "checkpoint"
    expected = model.token_embedding.weight.detach().clone()
    scaler = FakeScaler(1024.0)
    progress = {
        "best_validation_loss": 1.25,
        "early_stopping_bad_evals": 2,
        "epoch": 3,
    }
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        7,
        context,
        scaler=scaler,
        training_state=progress,
    )
    with torch.no_grad():
        model.token_embedding.weight.add_(1.0)
    scaler.scale = 1.0
    restored_progress: dict = {}
    step = load_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        context,
        scaler=scaler,
        training_state=restored_progress,
    )
    assert step == 7
    assert scaler.scale == 1024.0
    assert restored_progress == progress
    torch.testing.assert_close(model.token_embedding.weight, expected)
