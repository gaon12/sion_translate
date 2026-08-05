from __future__ import annotations

from pathlib import Path

import torch

from sion_translate.config import (
    AppConfig,
    DataConfig,
    ExperimentalConfig,
    ModelConfig,
    TrainingConfig,
)
from sion_translate.model import SionForConditionalGeneration
from sion_translate.training.distributed import DistributedContext
from sion_translate.training.trainer import train


def _config(tmp_path: Path) -> AppConfig:
    model = ModelConfig(
        vocab_size=48,
        d_model=24,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=48,
        max_seq_len=12,
        dropout=0.0,
        gradient_checkpointing=False,
        experimental=ExperimentalConfig(
            bats_enabled=False,
            core_enabled=False,
            tetm_enabled=False,
            morphoscript_enabled=False,
        ),
    )
    training = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=2,
        batch_size_per_gpu=1,
        learning_rate=1e-3,
        warmup_steps=0,
        precision="fp32",
        fsdp2=False,
        log_every=1,
        eval_every=1,
        eval_batches=1,
        save_every=10,
        tensorboard=False,
        ema_decay=0.5,
    )
    return AppConfig(
        model=model,
        data=DataConfig(max_source_length=12, max_target_length=12),
        training=training,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[4, 10, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "decoder_input_ids": torch.tensor([[2, 20, 21]]),
        "labels": torch.tensor([[20, 21, 3]]),
        "register_labels": torch.zeros(1, dtype=torch.long),
        "memory_token_ids": torch.zeros(1, 1, 1, dtype=torch.long),
        "memory_mask": torch.zeros(1, 1, dtype=torch.bool),
        "memory_type_ids": torch.zeros(1, 1, dtype=torch.long),
        "memory_mode_ids": torch.zeros(1, 1, dtype=torch.long),
    }


def test_training_returns_best_ema_weights_for_the_next_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Global NLL improves on step 2, while the direction-balanced NLL gets
    # worse. The configured macro metric must therefore keep step 1 as best.
    evaluations = iter(((2.2, 1.2), (2.0, 1.0), (1.2, 2.2), (1.0, 2.0)))

    def fake_evaluate(*args, **kwargs) -> dict[str, float]:
        nll, macro_direction_nll = next(evaluations)
        return {
            "validation_loss": nll,
            "validation_nll": nll,
            "validation_macro_direction_nll": macro_direction_nll,
            "validation_perplexity": float(torch.exp(torch.tensor(nll))),
            "validation_auxiliary_loss": 0.0,
            "validation_tokens": 1.0,
        }

    monkeypatch.setattr("sion_translate.training.trainer.evaluate", fake_evaluate)
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: None,
    )

    config = _config(tmp_path)
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    train(model, [_batch()], [_batch()], config, context)

    best = torch.load(
        tmp_path / "run" / "checkpoints" / "best" / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    final = torch.load(
        tmp_path / "run" / "checkpoints" / "final" / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    live = dict(model.named_parameters())
    for name, expected in best["ema"].items():
        torch.testing.assert_close(live[name], expected)
    assert any(not torch.equal(live[name], final["model"][name]) for name in live)
    assert best["step"] == 1
