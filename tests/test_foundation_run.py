"""foundation 단계를 실제로 한 번 돌려 본다.

계획·설정 유도는 `test_foundation_stage.py` 가 봅니다. 여기서는 그 설정으로
정말 학습이 돌고, **두 번째 실행이 다시 학습하지 않는지** 를 봅니다. 이
단계는 파이프라인에서 가장 오래 걸리는 구간이라, 번역 학습이 실패해 다시
실행할 때마다 며칠짜리 사전학습을 반복하면 안 됩니다.
"""

from __future__ import annotations

import json

import pytest
import torch

from sion_translate.cli.train import FOUNDATION_COMPLETION_FILENAME, run_foundation_stage
from sion_translate.config import AppConfig, ExperimentalConfig, ModelConfig
from sion_translate.data.prepare_foundation import prepare_foundation_dataset
from sion_translate.foundation import foundation_run_directory, plan_foundation_stage
from sion_translate.model import SionForConditionalGeneration
from sion_translate.tokenizer import SionTokenizer, train_tokenizer
from sion_translate.training.distributed import DistributedContext


@pytest.fixture(scope="module")
def tokenizer_model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("foundation_tokenizer")
    shard = directory / "pairs.jsonl"
    with shard.open("w", encoding="utf-8") as handle:
        for index in range(400):
            handle.write(
                json.dumps(
                    {
                        "ko": f"한국어 문장 {index} 입니다 그리고 조금 더 깁니다",
                        "ja": f"日本語の文 {index} です そしてもう少し長いです",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return train_tokenizer(
        [str(shard)],
        directory / "out",
        vocab_size=700,
        num_workers=1,
        language_pairs=[["kj", "ko"], ["kj", "ja"], ["ko", "ja"]],
        source_only_languages=["kj"],
    )


def _prepared(tmp_path, tokenizer_model):
    """토크나이저·코퍼스·데이터셋이 준비된 설정과 모델을 만든다."""

    corpus = tmp_path / "corpus"
    for language, template in (
        ("ko", "한국어 단일어 문장 {} 입니다 조금 더 깁니다"),
        ("ja", "日本語の単言語文 {} です もう少し長いです"),
    ):
        (corpus / language).mkdir(parents=True)
        (corpus / language / "a.txt").write_text(
            "\n".join(template.format(index) for index in range(240)) + "\n",
            encoding="utf-8",
        )

    config = AppConfig()
    config.data.language_pairs = [["ko", "ja"]]
    config.data.tokenizer_model = str(tokenizer_model)
    config.data.tokenizer_features = str(tokenizer_model.parent / "token_features.npz")
    config.data.dataset_dir = str(tmp_path / "dataset")
    config.data.num_workers = 0
    config.data.bucket_size = 16
    config.data.max_source_length = 32
    config.data.max_target_length = 32
    config.foundation.corpus_dir = str(corpus)
    config.foundation.dataset_dir = str(tmp_path / "foundation_dataset")
    config.foundation.max_steps = 2
    config.foundation.warmup_steps = 1
    config.foundation.batch_size_per_gpu = 2
    config.foundation.eval_every = 1
    config.foundation.eval_batches = 1
    config.foundation.save_every = 1
    config.foundation.shard_size = 64
    config.foundation.validation_fraction = 0.1
    config.training.output_dir = str(tmp_path / "runs")
    config.training.tensorboard = False
    config.training.ema_decay = 0.0
    config.training.precision = "fp32"

    tokenizer = SionTokenizer(tokenizer_model)
    config.model = ModelConfig(
        vocab_size=len(tokenizer),
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
        experimental=ExperimentalConfig(),
    )
    config.validate()

    plan = plan_foundation_stage(config)
    prepare_foundation_dataset(
        plan.discovery,
        tokenizer_model,
        config.foundation.dataset_dir,
        shard_size=config.foundation.shard_size,
        validation_fraction=config.foundation.validation_fraction,
        minimum_characters=4,
    )
    model = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    return config, plan, model, tokenizer, context


def test_the_stage_trains_and_marks_itself_complete(tmp_path, tokenizer_model) -> None:
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    before = model.token_embedding.weight.detach().clone()

    outcome = run_foundation_stage(config, plan, model, tokenizer, context)

    assert outcome.ran
    assert outcome.selected_step is not None
    run_root = foundation_run_directory(config)
    assert (run_root / FOUNDATION_COMPLETION_FILENAME).is_file()
    assert (run_root / "checkpoints" / "best").exists()
    # 실제로 학습이 일어났다면 가중치가 움직여야 한다.
    assert not torch.allclose(model.token_embedding.weight, before)

    marker = json.loads((run_root / FOUNDATION_COMPLETION_FILENAME).read_text(encoding="utf-8"))
    assert marker["stage"] == "foundation"
    assert marker["release_name"] == "sion"
    assert sorted(marker["languages"]) == ["ja", "ko"]


def test_a_second_run_reuses_the_weights_instead_of_retraining(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    """가장 비싼 단계를 반복하지 않는 것이 이 표시의 존재 이유다."""
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    run_foundation_stage(config, plan, model, tokenizer, context)
    trained = model.token_embedding.weight.detach().clone()

    import sion_translate.cli.train as train_module

    def _fail(*args, **kwargs):
        raise AssertionError("완료된 foundation 단계를 다시 학습하려 했습니다")

    monkeypatch.setattr(train_module, "train", _fail)

    fresh = SionForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
    outcome = run_foundation_stage(config, plan, fresh, tokenizer, context)

    assert not outcome.ran
    assert "재사용" in outcome.reason
    assert outcome.best_checkpoint is not None
    assert torch.allclose(fresh.token_embedding.weight, trained)


def test_a_disabled_plan_does_nothing_and_says_why(tmp_path, tokenizer_model) -> None:
    config, _, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    config.foundation.enabled = False
    plan = plan_foundation_stage(config)

    outcome = run_foundation_stage(config, plan, model, tokenizer, context)

    assert not outcome.ran
    assert "foundation.enabled=false" in outcome.reason
    assert outcome.best_checkpoint is None
    assert not (foundation_run_directory(config) / FOUNDATION_COMPLETION_FILENAME).exists()


def test_a_missing_corpus_never_trains_or_exports_a_foundation_model(
    tmp_path,
    tokenizer_model,
    monkeypatch,
) -> None:
    config, _, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    config.foundation.corpus_dir = str(tmp_path / "missing-corpus")
    plan = plan_foundation_stage(config)
    run_root = foundation_run_directory(config)

    import sion_translate.cli.train as train_module

    def _fail(*_args, **_kwargs):
        raise AssertionError("missing corpus must not train or export sion")

    monkeypatch.setattr(train_module, "train", _fail)
    monkeypatch.setattr(train_module, "export_final_model", _fail)

    outcome = run_foundation_stage(config, plan, model, tokenizer, context)

    assert not outcome.ran
    assert outcome.best_checkpoint is None
    assert not run_root.exists()


def test_the_stage_publishes_under_the_foundation_release_name(tmp_path, tokenizer_model) -> None:
    """foundation 산출물은 번역 모델이 아니라 그 파운데이션이다."""
    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    # The translation graph also contains a source-only variety, but the
    # foundation plan deliberately targets only ko/ja.
    config.data.language_pairs = [["kj", "ko"], ["kj", "ja"], ["ko", "ja"]]
    config.data.source_only_languages = ["kj"]
    config.foundation.final_export_formats = ["fp32", "transformers"]
    run_foundation_stage(config, plan, model, tokenizer, context)

    export = foundation_run_directory(config) / "exports" / "best"
    assert export.is_dir()
    # `best` is replaced by the final multi-format export. `latest` is the
    # training-loop export that would remain if the run were interrupted after
    # a save, so it must carry the same non-translation contract too.
    latest = foundation_run_directory(config) / "exports" / "latest" / "model.pt"
    latest_payload = torch.load(latest, map_location="cpu", weights_only=True)
    latest_metadata = latest_payload["metadata"]
    assert latest_metadata["release_name"] == "sion"
    assert latest_metadata["translation_capable"] is False
    assert latest_metadata["languages"] == ["ko", "ja"]
    assert "translation_directions" not in latest_metadata
    assert "language_pair" not in latest_metadata
    assert "language_pairs" not in latest_metadata

    payload = torch.load(export / "model.pt", map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    assert metadata["release_name"] == "sion"
    assert metadata["translation_capable"] is False
    assert metadata["languages"] == ["ko", "ja"]
    # 번역 방향을 적지 않는다. 번역할 수 없는 가중치이기 때문이다.
    assert "translation_directions" not in metadata
    assert "language_pair" not in metadata
    assert "language_pairs" not in metadata

    transformers_config = json.loads(
        (export / "transformers" / "config.json").read_text(encoding="utf-8")
    )
    assert transformers_config["languages"] == ["ko", "ja"]
    assert transformers_config["language_pairs"] == []
    assert transformers_config["translation_directions"] == []
    assert transformers_config["translation_capable"] is False

    tokenizer_config = json.loads(
        (export / "transformers" / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert tokenizer_config["translation_capable"] is False

    manifest = json.loads((export / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["formats"]["transformers"]["languages"] == metadata["languages"]
    assert manifest["formats"]["transformers"]["translation_capable"] is False

    from transformers import AutoConfig, AutoTokenizer

    hf_config = AutoConfig.from_pretrained(export / "transformers", trust_remote_code=True)
    hf_tokenizer = AutoTokenizer.from_pretrained(
        export / "transformers",
        trust_remote_code=True,
    )
    assert hf_config.translation_capable is False
    assert hf_tokenizer.translation_capable is False
    with pytest.raises(ValueError, match="foundation model.*not translation-capable"):
        hf_tokenizer._build_translation_inputs(
            "번역을 시도하면 안 됩니다.",
            return_tensors="pt",
            src_lang="ko",
            tgt_lang="ja",
        )


def test_a_foundation_export_is_refused_by_the_translator(tmp_path, tokenizer_model) -> None:
    """막지 않으면 방향 태그를 받아들이고 그럴듯한 쓰레기를 낸다."""
    from sion_translate.inference import Translator

    config, plan, model, tokenizer, context = _prepared(tmp_path, tokenizer_model)
    config.foundation.final_export_formats = ["fp32"]
    run_foundation_stage(config, plan, model, tokenizer, context)
    export = foundation_run_directory(config) / "exports" / "best" / "model.pt"

    with pytest.raises(ValueError, match="번역 모델이 아닙니다"):
        Translator(str(export), str(tokenizer_model))
