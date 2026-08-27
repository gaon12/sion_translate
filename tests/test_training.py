from __future__ import annotations

import hashlib
import json
import math
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch import nn

import sion_translate.training.trainer as trainer_module
from sion_translate.artifacts import RELEASE_INELIGIBLE_FILENAME
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
    resolve_training_budget,
    train as _train,
)

TRANSLATION_PIPELINE_IDENTITY = {
    "schema": "sion-translation-pipeline-v2",
    "branch": "translation-only",
}


def train(*args, **kwargs):
    """Run translation tests under the exact public 1.5 ancestry contract."""

    kwargs.setdefault("pipeline_identity", TRANSLATION_PIPELINE_IDENTITY)
    return _train(*args, **kwargs)


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


class FixedValidationModel(nn.Module):
    """Emit controlled token probabilities while exposing the trainer output contract."""

    def __init__(
        self,
        target_probabilities: torch.Tensor,
        *,
        smoothed_loss_sum: float = 30.0,
        refinement_token_gain: torch.Tensor | None = None,
    ):
        super().__init__()
        self.register_buffer("target_probabilities", target_probabilities)
        self.register_buffer("refinement_token_gain", refinement_token_gain)
        self.smoothed_loss_sum = smoothed_loss_sum

    def forward(self, **batch):
        labels = batch["labels"]
        probabilities = self.target_probabilities[: labels.shape[0], : labels.shape[1]]
        logits = torch.stack((probabilities.log(), (1.0 - probabilities).log()), dim=-1)
        return type(
            "ValidationOutput",
            (),
            {
                "logits": logits,
                "lm_loss_sum": logits.new_tensor(self.smoothed_loss_sum),
                "token_count": labels.ne(-100).sum(),
                "auxiliary_loss": logits.new_zeros(()),
                "candidate_refinement_token_nll_gain": self.refinement_token_gain,
            },
        )()


def direction_validation_batch() -> dict[str, torch.Tensor]:
    return {
        # Row 0 is ja->ko and row 1 is ko->ja. The first input token is target.
        "input_ids": torch.tensor([[5, 10], [4, 11]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.bool),
        "decoder_input_ids": torch.zeros(2, 2, dtype=torch.long),
        "labels": torch.tensor([[0, 0], [0, -100]]),
        "source_language_tag_ids": torch.tensor([4, 5]),
    }


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
            language_pair=["ko", "ja"],
            tokenizer_model=str(tokenizer_path),
            max_source_length=16,
            max_target_length=16,
            num_workers=0,
        ),
        training=TrainingConfig(**training_values),
    )


@pytest.mark.parametrize(
    "pipeline_identity",
    [
        None,
        {"schema": "sion-translation-pipeline-v1", "branch": "translation-only"},
    ],
)
def test_translation_training_rejects_invalid_ancestry_before_optimizer_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_identity: dict[str, str] | None,
) -> None:
    config = tiny_app_config(tmp_path)
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    monkeypatch.setattr(
        trainer_module.torch.optim,
        "AdamW",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("optimizer must not be allocated")
        ),
    )

    with pytest.raises(ValueError, match="pipeline"):
        _train(
            model,
            [tiny_batch()],
            [tiny_batch()],
            config,
            context,
            pipeline_identity=pipeline_identity,
        )


def test_programmatic_training_rejects_an_unauthenticated_revision_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_app_config(tmp_path)
    config.data.revision_directions = [["ko", "ja"]]
    config.data.revision_examples = True
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    monkeypatch.setattr(
        trainer_module.torch.optim,
        "AdamW",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("optimizer must not be allocated")
        ),
    )

    with pytest.raises(ValueError, match="authenticated_revision_directions"):
        _train(
            model,
            [tiny_batch()],
            [tiny_batch()],
            config,
            context,
            pipeline_identity=TRANSLATION_PIPELINE_IDENTITY,
        )


def test_single_step_training_loop(tmp_path: Path) -> None:
    config = tiny_app_config(tmp_path)
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    pipeline_identity = TRANSLATION_PIPELINE_IDENTITY
    result = train(
        model,
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
        pipeline_identity=pipeline_identity,
    )
    assert result["step"] == 1
    assert (tmp_path / "run" / "checkpoints" / "best" / "checkpoint.pt").exists()
    assert (tmp_path / "run" / "checkpoints" / "latest" / "checkpoint.pt").exists()
    assert (tmp_path / "run" / "checkpoints" / "final" / "checkpoint.pt").exists()
    # Raw weights remain in checkpoints, so intermediate inference exports need only EMA.
    for name in ("best", "latest"):
        assert (tmp_path / "run" / "exports" / name / "model_ema.pt").exists()
        assert not (tmp_path / "run" / "exports" / name / "model.pt").exists()
    checkpoint = torch.load(
        tmp_path / "run" / "checkpoints" / "final" / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["identity"]["pipeline"] == pipeline_identity
    resolved = json.loads((tmp_path / "run" / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved == config.to_dict()


@pytest.mark.parametrize("failure_phase", ("preflight", "load"))
def test_failed_resume_preserves_the_previous_resolved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    config = tiny_app_config(tmp_path, ema_decay=0.0)
    config.training.resume_from = str(tmp_path / "checkpoint")
    resolved_path = Path(config.training.output_dir) / "resolved_config.json"
    resolved_path.parent.mkdir(parents=True)
    original = b'{"generation":"known-good"}\n'
    resolved_path.write_bytes(original)

    if failure_phase == "preflight":
        monkeypatch.setattr(
            trainer_module,
            "preflight_checkpoint_identity",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("incompatible resume")),
        )
    else:
        monkeypatch.setattr(
            trainer_module,
            "preflight_checkpoint_identity",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            trainer_module,
            "load_checkpoint",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("corrupt resume")),
        )

    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    with pytest.raises((ValueError, RuntimeError), match="resume"):
        train(
            SionForConditionalGeneration(config.model),
            [tiny_batch()],
            [tiny_batch()],
            config,
            context,
        )

    assert resolved_path.read_bytes() == original


def test_resolved_config_replace_failure_preserves_the_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "resolved_config.json"
    original = b'{"generation":"known-good"}\n'
    path.write_bytes(original)

    def reject_replace(_source: Path, _destination: Path) -> None:
        raise PermissionError("injected resolved config replace failure")

    monkeypatch.setattr(trainer_module.os, "replace", reject_replace)

    with pytest.raises(PermissionError, match="injected resolved config replace failure"):
        trainer_module._atomic_write_resolved_config(path, {"generation": "retry"})

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".resolved_config.json.*.tmp"))


def test_epoch_budget_traverses_every_batch_and_flushes_partial_accumulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: None,
    )

    class TrackingLoader:
        def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
            self.batches = batches
            self.visits = [0] * len(batches)

        def __len__(self) -> int:
            return len(self.batches)

        def __iter__(self):
            for index, batch in enumerate(self.batches):
                self.visits[index] += 1
                yield batch

    loader = TrackingLoader([tiny_batch(), tiny_batch(), tiny_batch()])
    config = tiny_app_config(
        tmp_path,
        max_steps=None,
        num_train_epochs=2,
        gradient_accumulation_steps=2,
        eval_every=1,
        save_every=100,
        ema_decay=0.0,
    )
    budget = resolve_training_budget(loader, config.training)
    assert budget.max_optimizer_steps == 4
    assert budget.target_epochs == 2

    result = train(
        SionForConditionalGeneration(config.model),
        loader,
        [tiny_batch()],
        config,
        DistributedContext(0, 0, 1, torch.device("cpu"), False),
    )

    assert loader.visits == [2, 2, 2]
    assert result["step"] == 4
    assert result["epoch"] == 2
    assert result["stopped_early"] is False


def test_epoch_early_stopping_waits_for_complete_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: None,
    )

    class CountingLoader:
        def __init__(self) -> None:
            self.visits = 0

        def __len__(self) -> int:
            return 3

        def __iter__(self):
            for _ in range(3):
                self.visits += 1
                yield tiny_batch()

    loader = CountingLoader()
    config = tiny_app_config(
        tmp_path,
        max_steps=None,
        num_train_epochs=5,
        early_stopping_min_epochs=3,
        eval_every=1,
        save_every=100,
        ema_decay=0.0,
        early_stopping_patience=1,
        early_stopping_min_delta=1_000_000.0,
    )
    result = train(
        SionForConditionalGeneration(config.model),
        loader,
        [tiny_batch()],
        config,
        DistributedContext(0, 0, 1, torch.device("cpu"), False),
    )

    assert loader.visits == 9
    assert result["epoch"] == 3
    assert result["step"] == 9
    assert result["stopped_early"] is True
    assert result["best_step"] == 3


def test_intermediate_exports_advertise_only_trained_directions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tiny_app_config(tmp_path, ema_decay=0.0)
    config.data.language_pair = []
    config.data.language_pairs = [
        ["kj", "ko"],
        ["kj", "ja"],
        ["kd", "ko"],
        ["kd", "ja"],
        ["jd", "ko"],
        ["jd", "ja"],
        ["ko", "ja"],
    ]
    config.data.source_only_languages = ["kj", "kd", "jd"]
    captured: list[dict[str, object]] = []

    def capture(*_args: object, **kwargs: object) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(trainer_module, "export_inference_models", capture)
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    train(model, [tiny_batch()], [tiny_batch()], config, context)

    expected = (
        ("kj", "ko"),
        ("kj", "ja"),
        ("kd", "ko"),
        ("kd", "ja"),
        ("jd", "ko"),
        ("jd", "ja"),
        ("ko", "ja"),
        ("ja", "ko"),
    )
    assert captured
    assert all(item["translation_directions"] == expected for item in captured)
    assert all(item["release_name"] == "sion_translate" for item in captured)
    assert all(item["translation_capable"] is True for item in captured)


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


def test_validation_reports_true_nll_for_each_target_and_direction() -> None:
    model = FixedValidationModel(torch.tensor([[0.8, 0.8], [0.25, 0.5]]))
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    metrics = evaluate(
        model,
        [direction_validation_batch()],
        context,
        max_batches=1,
        language_tags=FakeTokenizer.language_tags,
    )

    ja_to_ko_nll = -math.log(0.8)
    ko_to_ja_nll = -math.log(0.25)
    global_nll = (2 * ja_to_ko_nll + ko_to_ja_nll) / 3
    macro_nll = (ja_to_ko_nll + ko_to_ja_nll) / 2
    # The model's reported training loss was deliberately 30 / 3 = 10. PPL
    # must instead come from the independently recomputed, unsmoothed NLL.
    assert metrics["validation_loss"] == pytest.approx(10.0)
    assert metrics["validation_nll"] == pytest.approx(global_nll)
    assert metrics["validation_perplexity"] == pytest.approx(math.exp(global_nll))
    assert metrics["validation_target_ko_nll"] == pytest.approx(ja_to_ko_nll)
    assert metrics["validation_target_ko_tokens"] == 2
    assert metrics["validation_target_ja_nll"] == pytest.approx(ko_to_ja_nll)
    assert metrics["validation_target_ja_tokens"] == 1
    assert metrics["validation_direction_ja_to_ko_nll"] == pytest.approx(ja_to_ko_nll)
    assert metrics["validation_direction_ko_to_ja_nll"] == pytest.approx(ko_to_ja_nll)
    assert metrics["validation_macro_direction_nll"] == pytest.approx(macro_nll)
    assert metrics["validation_worst_direction_nll"] == pytest.approx(ko_to_ja_nll)
    assert metrics["validation_direction_count"] == 2


def test_validation_reports_direction_balanced_candidate_refinement_gain() -> None:
    model = FixedValidationModel(
        torch.tensor([[0.8, 0.8], [0.25, 0.5]]),
        refinement_token_gain=torch.tensor([[0.2, 0.4], [-0.3, 999.0]]),
    )
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    metrics = evaluate(
        model,
        [direction_validation_batch()],
        context,
        max_batches=1,
        language_tags=FakeTokenizer.language_tags,
    )

    assert metrics["validation_candidate_refinement_nll_gain"] == pytest.approx(0.1)
    assert metrics["validation_candidate_refinement_tokens"] == 3
    assert metrics["validation_direction_ja_to_ko_candidate_refinement_nll_gain"] == pytest.approx(
        0.3
    )
    assert metrics["validation_direction_ko_to_ja_candidate_refinement_nll_gain"] == pytest.approx(
        -0.3
    )
    assert metrics["validation_macro_direction_candidate_refinement_nll_gain"] == pytest.approx(
        0.0,
        abs=1e-8,
    )
    assert metrics["validation_worst_direction_candidate_refinement_nll_gain"] == pytest.approx(
        -0.3
    )
    assert metrics["validation_candidate_refinement_direction_count"] == 2


def test_validation_restores_training_mode_after_refinement_contract_error() -> None:
    model = FixedValidationModel(
        torch.tensor([[0.8, 0.8], [0.25, 0.5]]),
        refinement_token_gain=torch.tensor([[0.2]]),
    ).train()
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    with pytest.raises(ValueError, match="must match labels shape"):
        evaluate(
            model,
            [direction_validation_batch()],
            context,
            max_batches=1,
            language_tags=FakeTokenizer.language_tags,
        )

    assert model.training


def test_direction_statistics_have_a_fixed_ddp_reduction_layout(monkeypatch) -> None:
    model = FixedValidationModel(
        torch.tensor([[0.8, 0.8]]),
        smoothed_loss_sum=2.0,
        refinement_token_gain=torch.tensor([[0.2, 0.4]]),
    )
    local_batch = {name: value[:1].clone() for name, value in direction_validation_batch().items()}
    context = DistributedContext(0, 0, 2, torch.device("cpu"), True, "gloo")
    packed_reductions = 0
    refinement_reductions = 0
    remote_nll = -math.log(0.25)
    remote_refinement_gain = -0.5

    def simulate_all_reduce(tensor: torch.Tensor, _context: DistributedContext) -> torch.Tensor:
        nonlocal packed_reductions, refinement_reductions
        # Sorted layout is [ja target, ko target, ja->ja, ja->ko,
        # ko->ja, ko->ko]. Rank 0 has only ja->ko; inject rank 1's
        # ko->ja row into the same preallocated tensor.
        if tensor.shape == (2,):
            refinement_reductions += 1
            tensor += tensor.new_tensor([remote_refinement_gain, 1.0])
        if tensor.shape == (6, 4):
            packed_reductions += 1
            remote = tensor.new_tensor([remote_nll, 1.0, remote_refinement_gain, 1.0])
            tensor[0] += remote
            tensor[4] += remote
        return tensor

    monkeypatch.setattr(trainer_module, "reduce_sum", simulate_all_reduce)
    metrics = evaluate(
        model,
        [local_batch],
        context,
        max_batches=1,
        language_tags=FakeTokenizer.language_tags,
    )

    assert packed_reductions == 1
    assert refinement_reductions == 1
    assert metrics["validation_direction_ja_to_ko_tokens"] == 2
    assert metrics["validation_direction_ko_to_ja_tokens"] == 1
    assert metrics["validation_direction_count"] == 2
    assert metrics["validation_worst_direction_nll"] == pytest.approx(remote_nll)
    assert metrics["validation_candidate_refinement_nll_gain"] == pytest.approx(1.0 / 30.0)
    assert metrics["validation_direction_ja_to_ko_candidate_refinement_nll_gain"] == pytest.approx(
        0.3
    )
    assert metrics["validation_direction_ko_to_ja_candidate_refinement_nll_gain"] == pytest.approx(
        remote_refinement_gain
    )
    assert metrics["validation_worst_direction_candidate_refinement_nll_gain"] == pytest.approx(
        remote_refinement_gain
    )


def test_sparse_objective_metrics_use_one_rank_stable_ddp_layout(monkeypatch) -> None:
    """Rank-local direction/custom keys must not alter collective order or meaning."""

    class SparseObjective:
        @staticmethod
        def validation_metrics(model, batch):
            del model, batch
            return {
                "reward": torch.tensor(0.2),
                "direction_ja_to_ko_reward_sum": torch.tensor(0.2),
                "direction_ja_to_ko_rows": torch.tensor(1.0),
                "local_metric": torch.tensor(2.0),
            }

    model = FixedValidationModel(torch.tensor([[0.8, 0.8]]), smoothed_loss_sum=2.0)
    local_batch = {name: value[:1].clone() for name, value in direction_validation_batch().items()}
    context = DistributedContext(0, 0, 2, torch.device("cpu"), True, "gloo")
    scalar_reductions = 0
    packed_reductions = 0

    def simulate_name_gather(
        gathered: list[tuple[str, ...] | None], local_names: tuple[str, ...]
    ) -> None:
        # Direction accumulator names come from language_tags and do not need
        # object exchange. Arbitrary objective metrics still participate in the
        # sorted union even when only one rank emits them.
        assert local_names == ("local_metric", "reward")
        gathered[0] = local_names
        gathered[1] = ("remote_metric", "reward")

    def simulate_all_reduce(tensor: torch.Tensor, _context: DistributedContext) -> torch.Tensor:
        nonlocal scalar_reductions, packed_reductions
        if tensor.ndim == 0:
            scalar_reductions += 1
            if scalar_reductions == 6:  # objective_count
                tensor += 1.0
        elif tensor.shape == (7,):
            packed_reductions += 1
            # Sorted layout: two accumulators for each of ja->ko and ko->ja,
            # followed by local_metric, remote_metric, and reward.
            torch.testing.assert_close(
                tensor,
                tensor.new_tensor([0.2, 1.0, 0.0, 0.0, 2.0, 0.0, 0.2]),
            )
            tensor += tensor.new_tensor([0.0, 0.0, 0.9, 1.0, 0.0, 6.0, 0.9])
        return tensor

    monkeypatch.setattr(trainer_module.dist, "all_gather_object", simulate_name_gather)
    monkeypatch.setattr(trainer_module, "reduce_sum", simulate_all_reduce)
    metrics = evaluate(
        model,
        [local_batch],
        context,
        max_batches=1,
        objective=SparseObjective(),
        language_tags=FakeTokenizer.language_tags,
    )

    assert packed_reductions == 1
    assert metrics["validation_reward"] == pytest.approx(0.55)
    assert metrics["validation_local_metric"] == pytest.approx(1.0)
    assert metrics["validation_remote_metric"] == pytest.approx(3.0)
    assert metrics["validation_direction_ja_to_ko_reward"] == pytest.approx(0.2)
    assert metrics["validation_direction_ko_to_ja_reward"] == pytest.approx(0.9)
    assert metrics["validation_worst_direction_reward"] == pytest.approx(0.2)
    assert metrics["validation_macro_direction_reward"] == pytest.approx(0.55)


def test_sft_direction_selection_falls_back_to_a_finite_global_nll() -> None:
    metrics = {
        "validation_ema_macro_direction_nll": float("nan"),
        "validation_macro_direction_nll": 0.1,
        "validation_ema_nll": 0.75,
        "validation_nll": 0.8,
        "validation_ema_loss": 0.5,
        "validation_loss": 0.6,
    }

    value, key, used_fallback = trainer_module._select_sft_validation_metric(
        metrics,
        "macro_direction_nll",
        prefer_ema=True,
    )

    assert value == pytest.approx(0.75)
    assert key == "validation_ema_nll"
    assert used_fallback is True


def refinement_release_metrics(
    directions: tuple[tuple[str, str], ...],
    gains: tuple[float, ...],
    *,
    ema: bool = False,
) -> dict[str, float]:
    prefix = "validation_ema_" if ema else "validation_"
    metrics = {
        f"{prefix}worst_direction_candidate_refinement_nll_gain": min(gains),
        f"{prefix}candidate_refinement_direction_count": float(len(directions)),
    }
    for (source_language, target_language), gain in zip(directions, gains, strict=True):
        direction_prefix = f"{prefix}direction_{source_language}_to_{target_language}"
        metrics[f"{direction_prefix}_candidate_refinement_nll_gain"] = gain
        metrics[f"{direction_prefix}_candidate_refinement_tokens"] = 10.0
    return metrics


def test_candidate_refinement_release_check_uses_the_deployed_ema_family() -> None:
    directions = (("de", "fr"), ("sw", "ar"))
    metrics = refinement_release_metrics(directions, (0.5, 0.6))
    metrics.update(refinement_release_metrics(directions, (-5e-7, 0.2), ema=True))

    passed, key, worst_gain = trainer_module._check_candidate_refinement_release(
        metrics,
        prefer_ema=True,
        expected_directions=directions,
    )

    assert passed
    assert key == "validation_ema_worst_direction_candidate_refinement_nll_gain"
    assert worst_gain == pytest.approx(-5e-7)


def test_candidate_refinement_release_check_rejects_incomplete_evidence() -> None:
    directions = (("de", "fr"), ("sw", "ar"))
    with pytest.raises(RuntimeError, match="evidence is incomplete"):
        trainer_module._check_candidate_refinement_release(
            {},
            prefer_ema=False,
            expected_directions=directions,
        )

    malformed_cases = (
        (
            "validation_worst_direction_candidate_refinement_nll_gain",
            float("nan"),
            "must be finite",
        ),
        ("validation_candidate_refinement_direction_count", 1.0, "observed 1 directions"),
        ("validation_candidate_refinement_direction_count", 1.5, "must be an integer"),
        ("validation_direction_de_to_fr_candidate_refinement_tokens", 0.0, "target token"),
        (
            "validation_direction_de_to_fr_candidate_refinement_nll_gain",
            float("nan"),
            "must be finite",
        ),
    )
    for key, value, message in malformed_cases:
        metrics = refinement_release_metrics(directions, (0.1, 0.2))
        metrics[key] = value
        with pytest.raises(RuntimeError, match=message):
            trainer_module._check_candidate_refinement_release(
                metrics,
                prefer_ema=False,
                expected_directions=directions,
            )


def test_candidate_refinement_release_check_rejects_same_size_wrong_graph() -> None:
    expected_directions = (("de", "fr"), ("sw", "ar"))
    wrong_directions = (("ko", "ja"), ("en", "es"))
    metrics = refinement_release_metrics(wrong_directions, (0.1, 0.2))

    with pytest.raises(RuntimeError, match="evidence is incomplete"):
        trainer_module._check_candidate_refinement_release(
            metrics,
            prefer_ema=False,
            expected_directions=expected_directions,
        )


def test_sft_selection_keeps_direction_balance_ahead_of_lower_global_nll() -> None:
    metrics = {
        "validation_ema_macro_direction_nll": 1.1,
        "validation_macro_direction_nll": 1.0,
        "validation_ema_nll": 0.2,
        "validation_nll": 0.1,
    }

    value, key, used_fallback = trainer_module._select_sft_validation_metric(
        metrics,
        "macro_direction_nll",
        prefer_ema=True,
    )

    assert value == pytest.approx(1.1)
    assert key == "validation_ema_macro_direction_nll"
    assert used_fallback is False


def test_posttraining_direction_selection_prefers_the_ema_model_that_is_deployed() -> None:
    metrics = {
        "validation_worst_direction_reward": 0.9,
        "validation_ema_worst_direction_reward": 0.2,
        "validation_reward": 0.95,
        "validation_ema_reward": 0.8,
    }

    value, key, used_fallback = trainer_module._select_posttraining_validation_metric(
        metrics,
        "worst_direction_reward",
        prefer_ema=True,
    )

    assert value == pytest.approx(0.2)
    assert key == "validation_ema_worst_direction_reward"
    assert used_fallback is False


def test_posttraining_direction_selection_falls_back_within_the_ema_family() -> None:
    metrics = {
        "validation_worst_direction_reward": 0.99,
        "validation_reward": 0.95,
        "validation_ema_reward": 0.8,
    }

    value, key, used_fallback = trainer_module._select_posttraining_validation_metric(
        metrics,
        "worst_direction_reward",
        prefer_ema=True,
    )

    assert value == pytest.approx(0.8)
    assert key == "validation_ema_reward"
    assert used_fallback is True


@pytest.mark.parametrize(
    "checkpoint_metric",
    (None, "global_nll"),
    ids=("legacy-missing", "configured-mismatch"),
)
def test_resume_resets_an_incompatible_best_selection_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_metric: str | None,
) -> None:
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: None,
    )
    config = tiny_app_config(tmp_path, max_steps=1, ema_decay=0.0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
    )
    latest_path = tmp_path / "run" / "checkpoints" / "latest"
    checkpoint_file = latest_path / "checkpoint.pt"
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    state = payload["training_state"]
    assert state["configured_selection_metric"] == "macro_direction_nll"
    assert state["best_selection_metric"] == "validation_nll"
    state["best_validation_loss"] = -100.0
    state["best_step"] = 777
    state["early_stopping_bad_evals"] = 99
    if checkpoint_metric is None:
        state.pop("configured_selection_metric", None)
        state.pop("best_selection_metric", None)
    else:
        state["configured_selection_metric"] = checkpoint_metric
        state["best_selection_metric"] = "validation_nll"
    torch.save(payload, checkpoint_file)

    config.training.max_steps = 2
    config.training.resume_from = str(latest_path)
    result = train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
    )

    assert result["configured_selection_metric"] == "macro_direction_nll"
    assert result["best_selection_metric"] == "validation_nll"
    assert result["best_step"] == 2
    assert result["early_stopping_bad_evals"] == 0


def test_resume_reselects_best_when_recorded_artifact_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: None,
    )
    config = tiny_app_config(tmp_path, max_steps=1, ema_decay=0.0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
    )
    checkpoint_root = tmp_path / "run" / "checkpoints"
    latest = checkpoint_root / "latest"
    latest_file = latest / "checkpoint.pt"
    payload = torch.load(latest_file, map_location="cpu", weights_only=True)
    payload["training_state"]["best_validation_loss"] = -100.0
    payload["training_state"]["best_step"] = 1
    payload["training_state"]["best_selection_metric"] = "validation_nll"
    torch.save(payload, latest_file)
    shutil.rmtree(checkpoint_root / "best")

    config.training.resume_from = str(latest)
    result = train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
    )

    assert (checkpoint_root / "best" / "checkpoint.pt").is_file()
    assert result["best_step"] == 1
    assert math.isfinite(float(result["best_validation_loss"]))
    assert float(result["best_validation_loss"]) > -100.0


def test_resume_reselects_best_when_recorded_artifact_step_is_not_authentic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: None,
    )
    config = tiny_app_config(tmp_path, max_steps=1, ema_decay=0.0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
    )
    checkpoint_root = tmp_path / "run" / "checkpoints"
    latest_file = checkpoint_root / "latest" / "checkpoint.pt"
    latest_payload = torch.load(latest_file, map_location="cpu", weights_only=True)
    latest_payload["training_state"]["best_validation_loss"] = -100.0
    latest_payload["training_state"]["best_step"] = 1
    latest_payload["training_state"]["best_selection_metric"] = "validation_nll"
    torch.save(latest_payload, latest_file)
    best_file = checkpoint_root / "best" / "checkpoint.pt"
    best_payload = torch.load(best_file, map_location="cpu", weights_only=True)
    best_payload["step"] = 999
    torch.save(best_payload, best_file)

    config.training.max_steps = 2
    config.training.resume_from = str(checkpoint_root / "latest")
    result = train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
    )

    selected_source = Path(str(result["selected_checkpoint_source"]))
    assert selected_source == checkpoint_root / "best"
    assert result["best_step"] == 2
    assert float(result["best_validation_loss"]) > -100.0
    assert (
        result["selected_checkpoint_artifact_sha256"]
        == hashlib.sha256((selected_source / "checkpoint.pt").read_bytes()).hexdigest()
    )


def test_resume_reselects_best_when_same_step_artifact_digest_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: None,
    )
    config = tiny_app_config(tmp_path, max_steps=1, ema_decay=0.0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
    )
    checkpoint_root = tmp_path / "run" / "checkpoints"
    best_file = checkpoint_root / "best" / "checkpoint.pt"
    best_payload = torch.load(best_file, map_location="cpu", weights_only=True)
    first_parameter = next(iter(best_payload["model"]))
    best_payload["model"][first_parameter] = best_payload["model"][first_parameter].clone() + 1
    torch.save(best_payload, best_file)

    latest_file = checkpoint_root / "latest" / "checkpoint.pt"
    latest_payload = torch.load(latest_file, map_location="cpu", weights_only=True)
    latest_payload["training_state"]["best_validation_loss"] = -100.0
    torch.save(latest_payload, latest_file)

    config.training.max_steps = 2
    config.training.resume_from = str(checkpoint_root / "latest")
    result = train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
    )

    selected_source = Path(str(result["selected_checkpoint_source"]))
    assert result["best_step"] == 2
    assert float(result["best_validation_loss"]) > -100.0
    assert (
        result["selected_checkpoint_artifact_sha256"]
        == hashlib.sha256((selected_source / "checkpoint.pt").read_bytes()).hexdigest()
    )


def test_resume_reselects_legacy_best_without_an_artifact_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: None,
    )
    config = tiny_app_config(tmp_path, max_steps=1, ema_decay=0.0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
    )
    checkpoint_root = tmp_path / "run" / "checkpoints"
    latest_file = checkpoint_root / "latest" / "checkpoint.pt"
    latest_payload = torch.load(latest_file, map_location="cpu", weights_only=True)
    latest_payload["training_state"].pop("best_checkpoint_artifact_sha256")
    latest_payload["training_state"]["best_validation_loss"] = -100.0
    torch.save(latest_payload, latest_file)

    config.training.max_steps = 2
    config.training.resume_from = str(checkpoint_root / "latest")
    result = train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
    )

    assert result["best_step"] == 2
    assert float(result["best_validation_loss"]) > -100.0


def test_best_checkpoint_binding_is_resumable_before_fallible_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_export(*args, **kwargs):
        raise RuntimeError("simulated export failure")

    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        fail_export,
    )
    config = tiny_app_config(tmp_path, max_steps=1, ema_decay=0.0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    with pytest.raises(RuntimeError, match="simulated export failure"):
        train(
            SionForConditionalGeneration(config.model),
            [tiny_batch()],
            [tiny_batch()],
            config,
            context,
        )

    checkpoint_root = tmp_path / "run" / "checkpoints"
    best_file = checkpoint_root / "best" / "checkpoint.pt"
    latest_file = checkpoint_root / "latest" / "checkpoint.pt"
    latest_payload = torch.load(latest_file, map_location="cpu", weights_only=True)
    assert latest_payload["step"] == 1
    assert (
        latest_payload["training_state"]["best_checkpoint_artifact_sha256"]
        == hashlib.sha256(best_file.read_bytes()).hexdigest()
    )


def test_nonfinite_validation_cannot_be_published_as_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        trainer_module,
        "evaluate",
        lambda *args, **kwargs: {
            "validation_loss": float("nan"),
            "validation_nll": float("nan"),
            "validation_perplexity": float("nan"),
            "validation_auxiliary_loss": 0.0,
            "validation_tokens": 1.0,
        },
    )
    config = tiny_app_config(tmp_path, max_steps=1, ema_decay=0.0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    with pytest.raises(RuntimeError, match="no finite validation selection metric"):
        train(
            SionForConditionalGeneration(config.model),
            [tiny_batch()],
            [tiny_batch()],
            config,
            context,
        )

    assert not (tmp_path / "run" / "checkpoints" / "best").exists()


def test_refinement_regression_cannot_be_published_as_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_calls: list[Path] = []
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda path, *args, **kwargs: export_calls.append(Path(path)),
    )
    validation_metrics = {
        "validation_loss": 0.5,
        "validation_nll": 0.5,
        "validation_perplexity": math.exp(0.5),
        "validation_auxiliary_loss": 0.0,
        "validation_tokens": 6.0,
        "validation_candidate_refinement_nll_gain": 0.1,
    }
    validation_metrics.update(
        refinement_release_metrics((("ko", "ja"), ("ja", "ko")), (0.1, -0.01))
    )
    monkeypatch.setattr(
        trainer_module,
        "evaluate",
        lambda *args, **kwargs: dict(validation_metrics),
    )
    config = tiny_app_config(tmp_path, max_steps=1)
    config.model.experimental.candidate_refinement_enabled = True
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    with pytest.raises(RuntimeError, match="release-safe best checkpoint"):
        train(
            SionForConditionalGeneration(config.model),
            [tiny_batch()],
            [tiny_batch()],
            config,
            context,
            language_tags={"ko": 4, "ja": 5},
        )

    checkpoint_root = tmp_path / "run" / "checkpoints"
    assert not (checkpoint_root / "best").exists()
    assert export_calls == []
    assert (tmp_path / "run" / "exports" / "best" / RELEASE_INELIGIBLE_FILENAME).is_file()
    assert (tmp_path / "run" / "exports" / "latest" / RELEASE_INELIGIBLE_FILENAME).is_file()
    with pytest.raises(FileNotFoundError, match="No exported model"):
        find_exported_model(tmp_path / "run")
    latest = torch.load(
        checkpoint_root / "latest" / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert latest["training_state"]["best_candidate_refinement_release_guard_passed"] is False


def test_refinement_baseline_failure_closes_the_summary_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWriter:
        closed = False

        def close(self) -> None:
            self.closed = True

    writer = FakeWriter()
    monkeypatch.setattr(trainer_module, "_make_summary_writer", lambda *args: writer)
    monkeypatch.setattr(
        trainer_module,
        "evaluate",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("baseline failed")),
    )
    config = tiny_app_config(tmp_path, max_steps=1)
    config.model.experimental.candidate_refinement_enabled = True
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    with pytest.raises(RuntimeError, match="baseline failed"):
        train(
            SionForConditionalGeneration(config.model),
            [tiny_batch()],
            [tiny_batch()],
            config,
            context,
            language_tags={"ko": 4, "ja": 5},
        )

    assert writer.closed


def test_guard_approved_export_failure_keeps_the_release_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: {
            "formats": {"fp32": {"status": "error", "message": "simulated failure"}}
        },
    )
    validation_metrics = {
        "validation_loss": 0.5,
        "validation_nll": 0.5,
        "validation_perplexity": math.exp(0.5),
        "validation_auxiliary_loss": 0.0,
        "validation_tokens": 6.0,
        "validation_candidate_refinement_nll_gain": 0.05,
        **refinement_release_metrics((("ko", "ja"), ("ja", "ko")), (0.02, 0.01)),
    }
    monkeypatch.setattr(
        trainer_module,
        "evaluate",
        lambda *args, **kwargs: dict(validation_metrics),
    )
    config = tiny_app_config(tmp_path, max_steps=1)
    config.model.experimental.candidate_refinement_enabled = True
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    with pytest.raises(RuntimeError, match="could not clear its release block"):
        train(
            SionForConditionalGeneration(config.model),
            [tiny_batch()],
            [tiny_batch()],
            config,
            context,
            language_tags={"ko": 4, "ja": 5},
        )

    assert (tmp_path / "run" / "exports" / "best" / RELEASE_INELIGIBLE_FILENAME).is_file()
    latest = torch.load(
        tmp_path / "run" / "checkpoints" / "latest" / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert latest["training_state"]["best_candidate_refinement_release_guard_passed"] is True


def test_refinement_release_attestation_is_resumable_and_legacy_best_is_reselected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_calls: list[Path] = []

    def record_successful_export(path: Path, *args, **kwargs) -> dict[str, object]:
        export_calls.append(Path(path))
        return {"formats": {"fp32": {"status": "ok"}}}

    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        record_successful_export,
    )
    validation_metrics = {
        "validation_loss": 0.5,
        "validation_nll": 0.5,
        "validation_perplexity": math.exp(0.5),
        "validation_auxiliary_loss": 0.0,
        "validation_tokens": 6.0,
        "validation_candidate_refinement_nll_gain": 0.05,
    }
    validation_metrics.update(
        refinement_release_metrics((("ko", "ja"), ("ja", "ko")), (0.02, 0.01))
    )
    monkeypatch.setattr(
        trainer_module,
        "evaluate",
        lambda *args, **kwargs: dict(validation_metrics),
    )
    config = tiny_app_config(tmp_path, max_steps=1)
    config.model.experimental.candidate_refinement_enabled = True
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    first = train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
        language_tags={"ko": 4, "ja": 5},
    )

    assert first["best_step"] == 0
    assert first["candidate_refinement_release_guard_passed"] is True
    assert export_calls == [tmp_path / "run" / "exports" / "best"]
    assert not (tmp_path / "run" / "exports" / "best" / RELEASE_INELIGIBLE_FILENAME).exists()
    assert (tmp_path / "run" / "exports" / "latest" / RELEASE_INELIGIBLE_FILENAME).is_file()
    latest_file = tmp_path / "run" / "checkpoints" / "latest" / "checkpoint.pt"
    payload = torch.load(latest_file, map_location="cpu", weights_only=True)
    attestation_keys = {
        key for key in payload["training_state"] if key.startswith("best_candidate_refinement_")
    }
    assert len(attestation_keys) == 6
    for key in attestation_keys:
        payload["training_state"].pop(key)
    torch.save(payload, latest_file)

    export_calls.clear()
    config.training.max_steps = 2
    config.training.resume_from = str(latest_file.parent)
    resumed = train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
        language_tags={"ko": 4, "ja": 5},
    )

    assert resumed["step"] == 2
    assert resumed["best_step"] == 1
    assert resumed["candidate_refinement_release_guard_passed"] is True
    assert export_calls == [tmp_path / "run" / "exports" / "best"]


def test_guard_ineligible_resume_keeps_the_attested_best_and_selection_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export_calls: list[Path] = []

    def record_successful_export(path: Path, *args, **kwargs) -> dict[str, object]:
        export_calls.append(Path(path))
        return {"formats": {"fp32": {"status": "ok"}}}

    active_metrics: dict[str, float] = {
        "validation_loss": 0.5,
        "validation_nll": 0.5,
        "validation_perplexity": math.exp(0.5),
        "validation_auxiliary_loss": 0.0,
        "validation_tokens": 6.0,
        "validation_candidate_refinement_nll_gain": 0.05,
        **refinement_release_metrics((("ko", "ja"), ("ja", "ko")), (0.02, 0.01)),
    }
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        record_successful_export,
    )
    monkeypatch.setattr(
        trainer_module,
        "evaluate",
        lambda *args, **kwargs: dict(active_metrics),
    )
    config = tiny_app_config(tmp_path, max_steps=1)
    config.model.experimental.candidate_refinement_enabled = True
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    first = train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
        language_tags={"ko": 4, "ja": 5},
    )
    best_file = tmp_path / "run" / "checkpoints" / "best" / "checkpoint.pt"
    original_best_digest = hashlib.sha256(best_file.read_bytes()).hexdigest()
    assert first["best_step"] == 0
    assert first["best_selection_metric"] == "validation_ema_nll"

    active_metrics.update(
        {
            "validation_macro_direction_nll": 0.4,
            "validation_candidate_refinement_nll_gain": -0.05,
            **refinement_release_metrics(
                (("ko", "ja"), ("ja", "ko")),
                (-0.02, -0.01),
            ),
        }
    )
    export_calls.clear()
    config.training.max_steps = 2
    config.training.resume_from = str(tmp_path / "run" / "checkpoints" / "latest")
    resumed = train(
        SionForConditionalGeneration(config.model),
        [tiny_batch()],
        [tiny_batch()],
        config,
        context,
        language_tags={"ko": 4, "ja": 5},
    )

    assert resumed["step"] == 2
    assert resumed["best_step"] == 0
    assert resumed["best_selection_metric"] == "validation_ema_nll"
    assert resumed["selected_checkpoint_artifact_sha256"] == original_best_digest
    assert export_calls == []


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

    # Training restores the live model from best EMA, so compare it with that export.
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


def test_sft_json_log_exposes_native_auxiliary_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(trainer_module.tqdm, "write", lambda message: messages.append(str(message)))
    monkeypatch.setattr(
        "sion_translate.training.trainer.export_inference_models",
        lambda *args, **kwargs: None,
    )
    config = tiny_app_config(tmp_path, ema_decay=0.0)
    model = SionForConditionalGeneration(config.model)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    train(model, [tiny_batch()], [tiny_batch()], config, context)

    json_records = [json.loads(message) for message in messages if message.startswith("{")]
    train_record = next(record for record in json_records if "learning_rate" in record)
    assert {
        "register_loss",
        "alignment_loss",
        "coverage_loss",
        "uncertainty_loss",
        "evidence_budget_loss",
        "evidence_request_rate",
        "evidence_repair_gain_loss",
        "evidence_repair_gain",
        "semantic_parity_loss",
        "semantic_parity_score",
    } <= train_record.keys()


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
    # Initial weights can make floating-point differences between the two paths
    # approach the tolerance, so fix the seed to start both from identical weights.
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
    assert result["configured_selection_metric"] == "validation_reward"
    assert result["best_selection_metric"] == "validation_ema_reward"
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


# ── MRT no-regression checks by direction ───────────────────────────────


class _DirectionRewardObjective:
    """A fake objective that gives ko->ja low reward and ja->ko high reward."""

    def __init__(self, tags: dict[str, int]):
        self.tags = tags

    def validation_metrics(self, model, batch):
        del model
        source = batch["source_language_tag_ids"]
        target = batch["input_ids"][:, 0]
        rows = float(target.shape[0])
        rewards = torch.where(
            target.eq(self.tags["ja"]),
            torch.full((int(rows),), 0.2),
            torch.full((int(rows),), 0.9),
        )
        metrics = {"reward": rewards.mean()}
        for source_name, source_id in self.tags.items():
            for target_name, target_id in self.tags.items():
                if source_name == target_name:
                    continue
                selected = source.eq(source_id) & target.eq(target_id)
                if not bool(selected.any()):
                    continue
                name = f"direction_{source_name}_to_{target_name}"
                metrics[f"{name}_reward_sum"] = rewards[selected].sum() / rows
                metrics[f"{name}_rows"] = selected.sum().float() / rows
        return metrics


def test_direction_rewards_are_weighted_by_rows_not_by_batch_size() -> None:
    """One mean reward can hide a regression in a single direction.

    This also explains why the objective must not emit direction means. Validation
    aggregation weights every metric by **batch size**, so a mean from one ko->ja
    row receives the full batch weight. Emitting sums and row counts separately
    applies equal weighting to both values, which cancels when they are divided.
    """
    tags = {"ko": 4, "ja": 5}
    model = SionForConditionalGeneration(tiny_model_config())
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    batch = tiny_batch()
    # One ko->ja row and one ja->ko row.
    batch["input_ids"] = torch.tensor([[5, 10, 3], [4, 11, 3]])
    batch["source_language_tag_ids"] = torch.tensor([4, 5])

    metrics = evaluate(
        model,
        [batch],
        context,
        max_batches=1,
        precision="fp32",
        objective=_DirectionRewardObjective(tags),
        language_tags=tags,
    )

    assert metrics["validation_direction_ko_to_ja_reward"] == pytest.approx(0.2)
    assert metrics["validation_direction_ja_to_ko_reward"] == pytest.approx(0.9)
    assert metrics["validation_worst_direction_reward"] == pytest.approx(0.2)
    assert metrics["validation_macro_direction_reward"] == pytest.approx(0.55)
    assert metrics["validation_reward_direction_count"] == 2.0
    # Mean reward is much higher than the worst direction and would hide its regression.
    assert metrics["validation_reward"] > metrics["validation_worst_direction_reward"]


def test_the_intermediate_direction_sums_do_not_leak_into_the_report() -> None:
    """Sums and row counts are calculation details, not report metrics."""
    tags = {"ko": 4, "ja": 5}
    model = SionForConditionalGeneration(tiny_model_config())
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    batch = tiny_batch()
    batch["input_ids"] = torch.tensor([[5, 10, 3], [4, 11, 3]])
    batch["source_language_tag_ids"] = torch.tensor([4, 5])

    metrics = evaluate(
        model,
        [batch],
        context,
        max_batches=1,
        precision="fp32",
        objective=_DirectionRewardObjective(tags),
        language_tags=tags,
    )

    assert not [name for name in metrics if name.endswith("_reward_sum")]
    assert not [name for name in metrics if name.endswith("_rows")]


def test_the_default_posttraining_selection_metric_protects_the_worst_direction() -> None:
    from sion_translate.config import AppConfig

    assert AppConfig().posttraining.selection_metric == "worst_direction_reward"


def test_an_unknown_posttraining_selection_metric_is_rejected() -> None:
    from sion_translate.config import AppConfig

    config = AppConfig()
    config.posttraining.selection_metric = "average"
    with pytest.raises(ValueError, match="posttraining.selection_metric"):
        config.validate()
