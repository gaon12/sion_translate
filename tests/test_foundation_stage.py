"""foundation 단계의 실행 여부 판단과 설정 유도.

"폴더가 있으면 자동 실행" 이므로 건너뛰는 것이 정상 경로입니다. 그래서 여기
테스트의 절반은 **왜 건너뛰었는지가 문장으로 남는가** 를 봅니다 — 조용히
건너뛰면 사용자는 5 GB 코퍼스가 학습에 들어갔다고 믿은 채로 번역 학습을
끝냅니다.
"""

from __future__ import annotations

import pytest

from sion_translate.config import AppConfig
from sion_translate.foundation import (
    build_foundation_config,
    build_translation_pipeline_identity,
    foundation_run_directory,
    plan_foundation_stage,
)


def _config(tmp_path, **foundation):
    config = AppConfig()
    config.data.language_pairs = [["kj", "ko"], ["kj", "ja"], ["ko", "ja"]]
    config.data.source_only_languages = ["kj"]
    config.foundation.corpus_dir = str(tmp_path / "corpus")
    config.foundation.dataset_dir = str(tmp_path / "foundation_dataset")
    config.training.output_dir = str(tmp_path / "runs")
    for name, value in foundation.items():
        setattr(config.foundation, name, value)
    config.validate()
    return config


def _corpus(tmp_path, languages=("ko", "ja")):
    root = tmp_path / "corpus"
    for language in languages:
        (root / language).mkdir(parents=True, exist_ok=True)
        (root / language / "a.txt").write_text(
            "\n".join(f"{language} 문장 {index} 입니다" for index in range(20)) + "\n",
            encoding="utf-8",
        )
    return root


def test_the_stage_runs_when_the_corpus_has_data(tmp_path) -> None:
    _corpus(tmp_path)
    plan = plan_foundation_stage(_config(tmp_path))
    assert plan.enabled
    assert plan.languages == ("ko", "ja")


def test_a_missing_corpus_skips_with_an_actionable_reason(tmp_path) -> None:
    plan = plan_foundation_stage(_config(tmp_path))
    assert not plan.enabled
    # 무엇을 어디에 두면 되는지가 문장 안에 있어야 한다.
    assert "언어 코드 폴더" in plan.reason
    assert ".txt" in plan.reason and ".jsonl" in plan.reason


def test_disabling_the_stage_says_so_explicitly(tmp_path) -> None:
    _corpus(tmp_path)
    plan = plan_foundation_stage(_config(tmp_path, enabled=False))
    assert not plan.enabled
    assert "foundation.enabled=false" in plan.reason


def test_pipeline_identity_tracks_model_ancestry_not_the_skip_reason(tmp_path) -> None:
    missing = plan_foundation_stage(_config(tmp_path))
    _corpus(tmp_path)
    disabled = plan_foundation_stage(_config(tmp_path, enabled=False))
    enabled = plan_foundation_stage(_config(tmp_path))

    assert build_translation_pipeline_identity(missing) == {
        "schema": "sion-translation-pipeline-v2",
        "branch": "translation-only",
    }
    assert build_translation_pipeline_identity(disabled) == build_translation_pipeline_identity(
        missing
    )
    lineage = {
        "schema": "sion-foundation-lineage-v1",
        "release_name": "sion",
        "release_version": "1.5",
        "languages": list(enabled.languages),
        "selected_step": 7,
        "foundation_manifest_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "checkpoint_identity_sha256": "c" * 64,
        "checkpoint_artifact_sha256": "d" * 64,
    }
    identity = build_translation_pipeline_identity(enabled, foundation_lineage=lineage)
    assert identity["branch"] == "foundation-then-translation"
    assert identity["foundation"] == lineage
    with pytest.raises(ValueError, match="requires resolved lineage"):
        build_translation_pipeline_identity(enabled)


def test_source_only_languages_never_reach_the_plan(tmp_path) -> None:
    """혼용문 kj 로 복원 학습을 하면 디코더가 kj 를 생성하도록 배운다."""
    _corpus(tmp_path, languages=("ko", "ja", "kj"))
    plan = plan_foundation_stage(_config(tmp_path))
    assert "kj" not in plan.languages
    assert all(source.language != "kj" for source in plan.discovery.sources)


def test_a_language_without_data_warns_but_still_runs(tmp_path) -> None:
    """현재 저장소의 실제 상태: ko 만 있고 ja 는 없다."""
    _corpus(tmp_path, languages=("ko",))
    plan = plan_foundation_stage(_config(tmp_path))
    assert plan.enabled
    assert any("ja" in warning for warning in plan.warnings)


def test_require_all_languages_turns_that_warning_into_a_stop(tmp_path) -> None:
    _corpus(tmp_path, languages=("ko",))
    with pytest.raises(RuntimeError, match="require_all_languages"):
        plan_foundation_stage(_config(tmp_path, require_all_languages=True))


def test_the_plan_carries_the_discovery_report(tmp_path) -> None:
    root = _corpus(tmp_path)
    (root / "stray.py").write_text("print()\n", encoding="utf-8")
    plan = plan_foundation_stage(_config(tmp_path))
    assert any("stray.py" in line for line in plan.report)


# ── 설정 유도 ───────────────────────────────────────────────────────────


def test_the_derived_config_is_a_pure_denoising_run(tmp_path) -> None:
    """src == tgt 인 shard 에서 복원을 걸지 않으면 '입력을 베껴라' 가 된다."""
    derived = build_foundation_config(_config(tmp_path))
    assert derived.data.denoise_probability == 1.0
    # 검증만 0 이면 검증 손실이 복사 과제의 손실이 되어 best 선택이 무의미해진다.
    assert derived.data.validation_denoise_probability == 1.0
    assert derived.data.source_token_dropout == 0.0
    assert derived.data.decoder_input_noise == 0.0


def test_the_derived_config_points_at_the_foundation_artifacts(tmp_path) -> None:
    config = _config(tmp_path)
    derived = build_foundation_config(config)
    assert derived.data.dataset_dir == config.foundation.dataset_dir
    assert derived.data.dataset_dir != config.data.dataset_dir
    assert derived.training.output_dir == str(foundation_run_directory(config))
    assert derived.training.output_dir.endswith("foundation")


def test_the_derived_config_inherits_the_model_and_precision(tmp_path) -> None:
    config = _config(tmp_path)
    config.model.d_model = 256
    config.model.encoder_layers = 4
    config.training.precision = "bf16"
    config.training.seed = 4321
    derived = build_foundation_config(config)
    assert derived.model.d_model == 256
    assert derived.model.encoder_layers == 4
    assert derived.training.precision == "bf16"
    assert derived.training.seed == 4321


def test_the_derived_config_takes_its_schedule_from_the_foundation_section(tmp_path) -> None:
    config = _config(
        tmp_path,
        num_train_epochs=4,
        max_steps=777,
        learning_rate=1e-4,
        warmup_steps=11,
        early_stopping_min_epochs=3,
    )
    config.training.num_train_epochs = 2
    config.training.max_steps = 5
    config.training.learning_rate = 9e-9
    derived = build_foundation_config(config)
    assert derived.training.num_train_epochs == 4
    assert derived.training.max_steps == 777
    assert derived.training.early_stopping_min_epochs == 3
    assert derived.training.learning_rate == 1e-4
    assert derived.training.warmup_steps == 11


def test_the_derived_config_has_no_posttraining_and_no_recursion(tmp_path) -> None:
    """복원 단계는 번역을 하지 않으므로 MRT 도, 또 다른 foundation 도 없다."""
    derived = build_foundation_config(_config(tmp_path))
    assert derived.posttraining.enabled is False
    assert derived.foundation.enabled is False


def test_the_derived_config_drops_source_only_languages(tmp_path) -> None:
    derived = build_foundation_config(_config(tmp_path))
    assert derived.data.configured_source_only_languages() == ()


def test_the_derived_config_uses_a_direction_free_selection_metric(tmp_path) -> None:
    """복원 과제에는 번역 방향이 없다."""
    derived = build_foundation_config(_config(tmp_path))
    assert derived.training.sft_selection_metric == "global_nll"


def test_the_derived_config_still_validates(tmp_path) -> None:
    build_foundation_config(_config(tmp_path)).validate()


def test_deriving_does_not_mutate_the_translation_config(tmp_path) -> None:
    config = _config(tmp_path)
    original_dataset = config.data.dataset_dir
    original_denoise = config.data.denoise_probability
    build_foundation_config(config)
    assert config.data.dataset_dir == original_dataset
    assert config.data.denoise_probability == original_denoise
    assert config.posttraining.enabled is True
