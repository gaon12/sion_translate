from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import sion_translate.training.trainer as trainer_module
from sion_translate.config import (
    AppConfig,
    DataConfig,
    ExperimentalConfig,
    ModelConfig,
    PostTrainingConfig,
    TrainingConfig,
)
from sion_translate.inference import find_exported_model
from sion_translate.model import SionForConditionalGeneration
from sion_translate.training.distributed import DistributedContext
from sion_translate.training.export import load_exported_model
from sion_translate.training.objectives import MinimumRiskObjective
from sion_translate.training.trainer import (
    build_optimizer_param_groups,
    cosine_scheduler,
    evaluate,
    train,
)


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
        gradient_checkpointing=False,
        experimental=ExperimentalConfig(
            bats_enabled=False,
            core_enabled=False,
            tetm_enabled=False,
            morphoscript_enabled=False,
        ),
    )


def tiny_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[4, 10, 3], [4, 11, 3]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.bool),
        "decoder_input_ids": torch.tensor([[2, 20, 21], [2, 22, 23]]),
        "labels": torch.tensor([[20, 21, 3], [22, 23, 3]]),
        "register_labels": torch.zeros(2, dtype=torch.long),
        "memory_token_ids": torch.zeros(2, 1, 1, dtype=torch.long),
        "memory_mask": torch.zeros(2, 1, dtype=torch.bool),
        "memory_type_ids": torch.zeros(2, 1, dtype=torch.long),
        "memory_mode_ids": torch.zeros(2, 1, dtype=torch.long),
    }


class FakeTokenizer:
    pad_id = 0
    bos_id = 2
    eos_id = 3
    mask_id = 6
    language_tags = {"ja": 4, "ko": 5}
    denoise_tags = {"ja": 7, "ko": 8}
    slot_ids: list[int] = []

    @staticmethod
    def decode(ids) -> str:
        return "".join(chr(0x3041 + int(token_id) % 80) for token_id in ids)


def tiny_app_config(tmp_path: Path, **training_overrides) -> AppConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"training export tokenizer fixture")
    training_values = {
        "output_dir": str(tmp_path / "run"),
        "max_steps": 1,
        "batch_size_per_gpu": 2,
        "learning_rate": 1e-3,
        "warmup_steps": 0,
        "precision": "fp32",
        "fsdp2": False,
        "log_every": 1,
        "eval_every": 1,
        "eval_batches": 1,
        "save_every": 1,
        "tensorboard": False,
    }
    training_values.update(training_overrides)
    return AppConfig(
        model=tiny_model_config(),
        data=DataConfig(
            tokenizer_model=str(tokenizer_path),
            max_source_length=16,
            max_target_length=16,
            num_workers=0,
        ),
        training=TrainingConfig(**training_values),
    )


def test_single_step_training_loop(tmp_path: Path) -> None:
    config = tiny_app_config(tmp_path)
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    result = train(model, [tiny_batch()], [tiny_batch()], config, context)
    assert result["step"] == 1
    assert (tmp_path / "run" / "checkpoints" / "best" / "checkpoint.pt").exists()
    assert (tmp_path / "run" / "checkpoints" / "latest" / "checkpoint.pt").exists()
    assert (tmp_path / "run" / "checkpoints" / "final" / "checkpoint.pt").exists()
    # raw 가중치는 checkpoint에 있으므로 중간 inference export는 EMA 하나뿐이다.
    for name in ("best", "latest"):
        assert (tmp_path / "run" / "exports" / name / "model_ema.pt").exists()
        assert not (tmp_path / "run" / "exports" / name / "model.pt").exists()


def test_posttraining_validation_metrics_share_the_autocast_context(monkeypatch) -> None:
    active = False

    @contextmanager
    def recording_autocast(precision: str, device: torch.device):
        nonlocal active
        assert precision == "bf16"
        assert device.type == "cpu"
        active = True
        try:
            yield
        finally:
            active = False

    class ProbeObjective:
        def validation_metrics(self, model, batch):
            del model, batch
            assert active, "generation metrics escaped the validation autocast context"
            return {"reward": torch.tensor(0.75)}

    monkeypatch.setattr(trainer_module, "_autocast_context", recording_autocast)
    model = SionForConditionalGeneration(tiny_model_config())
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    metrics = evaluate(
        model,
        [tiny_batch()],
        context,
        max_batches=1,
        precision="bf16",
        objective=ProbeObjective(),
    )

    assert metrics["validation_reward"] == pytest.approx(0.75)
    assert not active


def test_mid_epoch_resume_uses_saved_batch_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: None,
    )
    first_batch = tiny_batch()
    second_batch = {name: value.clone() for name, value in first_batch.items()}
    second_batch["input_ids"][:, 1] = torch.tensor([30, 31])
    second_batch["labels"][:, 0] = torch.tensor([32, 33])
    batches = [first_batch, second_batch, tiny_batch()]
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    torch.manual_seed(20260730)
    initial = SionForConditionalGeneration(tiny_model_config())
    initial_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}

    baseline = SionForConditionalGeneration(tiny_model_config())
    baseline.load_state_dict(initial_state)
    baseline_config = tiny_app_config(
        tmp_path / "baseline",
        max_steps=2,
        eval_every=100,
        save_every=100,
        ema_decay=0.0,
        min_learning_rate_ratio=1.0,
    )
    train(baseline, batches, [tiny_batch()], baseline_config, context)

    interrupted = SionForConditionalGeneration(tiny_model_config())
    interrupted.load_state_dict(initial_state)
    resumed_config = tiny_app_config(
        tmp_path / "resumed",
        max_steps=1,
        eval_every=100,
        save_every=100,
        ema_decay=0.0,
        min_learning_rate_ratio=1.0,
    )
    train(interrupted, batches, [tiny_batch()], resumed_config, context)
    resume_path = tmp_path / "resumed" / "run" / "checkpoints" / "latest"
    interrupted_payload = torch.load(
        resume_path / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert interrupted_payload["training_state"]["batch_in_epoch"] == 1
    assert interrupted_payload["training_state"]["epoch"] == 0

    resumed_config.training.max_steps = 2
    resumed_config.training.resume_from = str(resume_path)
    resumed = SionForConditionalGeneration(tiny_model_config())
    train(resumed, batches, [tiny_batch()], resumed_config, context)

    baseline_payload = torch.load(
        tmp_path / "baseline" / "run" / "checkpoints" / "final" / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    resumed_payload = torch.load(
        tmp_path / "resumed" / "run" / "checkpoints" / "final" / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    for name, expected in baseline_payload["model"].items():
        torch.testing.assert_close(resumed_payload["model"][name], expected)


def test_exported_models_reload_and_run(tmp_path: Path) -> None:
    config = tiny_app_config(tmp_path)
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    train(model, [tiny_batch()], [tiny_batch()], config, context)

    # 학습 종료 시 live model은 best EMA로 복원되므로 같은 export와 비교한다.
    rebuilt, rebuilt_config, pad_id = load_exported_model(
        tmp_path / "run" / "exports" / "best" / "model_ema.pt",
    )
    assert rebuilt_config == config.model
    assert pad_id == 0
    torch.testing.assert_close(rebuilt.token_embedding.weight, model.token_embedding.weight)


def test_early_stopping_saves_best_checkpoint(tmp_path: Path) -> None:
    config = tiny_app_config(
        tmp_path,
        max_steps=5,
        save_every=5,
        early_stopping_patience=1,
        early_stopping_min_delta=1_000_000.0,
    )
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    result = train(model, [tiny_batch()], [tiny_batch()], config, context)
    assert result["stopped_early"] is True
    assert result["step"] == 2
    assert result["best_step"] == 1
    assert result["selected_step"] == 1
    assert result["early_stopping_bad_evals"] == 1
    assert (tmp_path / "run" / "checkpoints" / "best" / "checkpoint.pt").exists()


def test_tensorboard_writes_main_rank_scalars(tmp_path: Path) -> None:
    config = tiny_app_config(tmp_path, tensorboard=True)
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    train(model, [tiny_batch()], [tiny_batch()], config, context)
    event_files = list((tmp_path / "run" / "tensorboard").glob("events.out.tfevents.*"))
    assert event_files
    assert all(path.stat().st_size > 0 for path in event_files)


def test_empty_training_loader_fails_fast(tmp_path: Path) -> None:
    config = tiny_app_config(tmp_path)
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    with pytest.raises(ValueError, match="training loader is empty"):
        train(model, [], [tiny_batch()], config, context)


def test_adamw_groups_exclude_norms_biases_and_one_dimensional_parameters() -> None:
    model = SionForConditionalGeneration(tiny_model_config())
    groups = build_optimizer_param_groups(model, weight_decay=0.1)
    decayed = {id(parameter) for parameter in groups[0]["params"]}
    not_decayed = {id(parameter) for parameter in groups[1]["params"]}
    assert id(model.token_embedding.weight) in decayed
    assert id(model.encoder_norm.weight) in not_decayed
    for name, parameter in model.named_parameters():
        if parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            assert id(parameter) in not_decayed


def test_cosine_scheduler_warms_up_then_decays() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = cosine_scheduler(optimizer, warmup_steps=2, max_steps=6, min_ratio=0.1)
    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    assert learning_rates[0] == pytest.approx(0.5)
    assert learning_rates[1] == pytest.approx(1.0)
    assert learning_rates[-1] == pytest.approx(0.1)


def test_gradient_accumulation_matches_combined_token_normalization(
    tmp_path: Path,
) -> None:
    # 초기 가중치에 따라 두 학습 경로의 부동소수점 차이가 허용 오차 근처까지
    # 커질 수 있으므로, 테스트가 항상 같은 가중치에서 출발하도록 시드를 고정한다.
    torch.manual_seed(20260711)
    combined = tiny_batch()
    micros = [
        {name: value[index : index + 1].clone() for name, value in combined.items()}
        for index in range(2)
    ]
    first = SionForConditionalGeneration(tiny_model_config())
    second = SionForConditionalGeneration(tiny_model_config())
    second.load_state_dict(first.state_dict())
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    accumulated_config = tiny_app_config(
        tmp_path / "accumulated",
        gradient_accumulation_steps=2,
    )
    combined_config = tiny_app_config(tmp_path / "combined")
    train(first, micros, [combined], accumulated_config, context)
    train(second, [combined], [combined], combined_config, context)

    for accumulated_parameter, combined_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(
            accumulated_parameter,
            combined_parameter,
            rtol=5e-4,
            atol=3e-6,
        )


def test_mrt_posttraining_objective_and_stage_save(tmp_path: Path) -> None:
    torch.manual_seed(11)
    config = tiny_app_config(tmp_path)
    config.training.output_dir = str(tmp_path / "run" / "posttrain")
    config.posttraining = PostTrainingConfig(
        max_steps=1,
        batch_size_per_gpu=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        warmup_steps=0,
        samples_per_source=2,
        top_k=8,
        max_new_tokens=4,
        validation_num_beams=1,
        eval_every=1,
        save_every=1,
    )
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    objective = MinimumRiskObjective(FakeTokenizer(), config.posttraining)
    result = train(
        model,
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
        objective=objective,
        stage_name="posttrain/MRT",
    )
    assert result["step"] == 1
    assert 0.0 <= result["best_validation_reward"] <= 1.0
    assert (tmp_path / "run" / "posttrain" / "checkpoints" / "final" / "checkpoint.pt").exists()
    assert (tmp_path / "run" / "posttrain" / "exports" / "latest" / "model_ema.pt").exists()


def test_inference_prefers_posttrain_then_pretrain(tmp_path: Path) -> None:
    pretrain = tmp_path / "run" / "pretrain" / "exports" / "best" / "model.pt"
    posttrain = tmp_path / "run" / "posttrain" / "exports" / "best" / "model.pt"
    pretrain.parent.mkdir(parents=True)
    pretrain.touch()
    assert find_exported_model(tmp_path / "run") == pretrain
    posttrain.parent.mkdir(parents=True)
    posttrain.touch()
    assert find_exported_model(tmp_path / "run") == posttrain
