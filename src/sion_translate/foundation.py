"""foundation(단일어 사전학습) 단계의 계획·설정 유도.

CLI 밖에 두는 이유는 두 가지입니다. 이 단계를 **돌릴지 말지** 판단하는 규칙과
그 이유를 사람에게 설명하는 문장은 CLI 배선과 별개로 검증할 수 있어야 하고,
번역 설정에서 foundation 설정을 유도하는 규칙(무엇을 덮어쓰고 무엇을 물려받는지)
은 조용히 틀리기 쉬운 종류이기 때문입니다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from sion_translate.artifacts import FOUNDATION_STAGE_DIRECTORY
from sion_translate.config import AppConfig
from sion_translate.data.monolingual import (
    MonolingualDiscovery,
    assess_language_balance,
    discover_monolingual_sources,
    render_discovery_report,
)


PIPELINE_IDENTITY_SCHEMA = "sion-translation-pipeline-v1"


@dataclass(frozen=True)
class FoundationPlan:
    """이 실행에서 foundation 단계를 돌릴지, 그리고 그 이유."""

    enabled: bool
    reason: str
    discovery: MonolingualDiscovery
    languages: tuple[str, ...] = ()
    report: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.enabled


def build_translation_pipeline_identity(plan: FoundationPlan) -> dict[str, str]:
    """Return the stable ancestry branch required for translation-stage resume.

    ``ran`` is deliberately not part of this identity: a completed foundation
    stage that is reused and one that was trained in the current process have
    the same translation-model ancestry. Paths and human-readable skip reasons
    are likewise runtime details rather than compatibility inputs.
    """

    return {
        "schema": PIPELINE_IDENTITY_SCHEMA,
        "branch": "foundation-then-translation" if plan.enabled else "translation-only",
    }


def plan_foundation_stage(config: AppConfig) -> FoundationPlan:
    """코퍼스를 훑어 단계 실행 여부와 그 이유를 정한다.

    "폴더가 있으면 자동 실행" 이므로 건너뛰는 경우가 정상 경로이고, 그래서
    **왜 건너뛰는지가 항상 문장으로 남아야 합니다.** 조용히 건너뛰면 사용자는
    5 GB 코퍼스가 학습에 들어갔다고 믿은 채로 번역 학습을 끝내게 됩니다.
    """

    languages = config.foundation_languages()
    discovery = discover_monolingual_sources(config.foundation.corpus_dir, languages)
    report = tuple(render_discovery_report(discovery))

    if not config.foundation.enabled:
        return FoundationPlan(
            enabled=False,
            reason="foundation.enabled=false — 설정에서 껐습니다.",
            discovery=discovery,
            languages=languages,
            report=report,
        )
    if not discovery.sources:
        return FoundationPlan(
            enabled=False,
            reason=(
                f"{config.foundation.corpus_dir} 에 학습 가능한 단일어 파일이 없어 "
                "건너뜁니다. 언어 코드 폴더(예: "
                f"{config.foundation.corpus_dir}/{languages[0] if languages else 'ko'}/) "
                "안에 .txt 또는 .jsonl 을 두면 다음 실행에서 자동으로 잡습니다."
            ),
            discovery=discovery,
            languages=languages,
            report=report,
        )

    balance = assess_language_balance(
        {
            language: sum(
                source.size_bytes for source in discovery.sources if source.language == language
            )
            for language in languages
        },
        alpha=config.foundation.language_sampling_alpha,
        minimum_share=config.foundation.minimum_language_share,
    )
    if config.foundation.require_all_languages and discovery.languages_without_data:
        missing = ", ".join(discovery.languages_without_data)
        raise RuntimeError(
            f"foundation.require_all_languages=true 인데 단일어 데이터가 없는 언어가 "
            f"있습니다: {missing}. 데이터를 채우거나 require_all_languages 를 끄십시오."
        )
    return FoundationPlan(
        enabled=True,
        reason=f"단일어 코퍼스 {len(discovery.sources)}개 파일을 찾았습니다.",
        discovery=discovery,
        languages=languages,
        report=report,
        warnings=balance.warnings,
    )


def foundation_run_directory(config: AppConfig) -> Path:
    return Path(config.training.output_dir) / FOUNDATION_STAGE_DIRECTORY


def build_foundation_config(config: AppConfig) -> AppConfig:
    """번역 설정에서 foundation 단계 설정을 유도한다.

    물려받는 것: 모델 구조, 정밀도, 병렬 전략, seed, 토크나이저.
    덮어쓰는 것: 데이터셋 경로, 목적(100% 복원), 학습 일정, 산출 경로.

    복원 확률을 학습·검증 **양쪽 다** 1.0 으로 둡니다. 이 단계의 shard 는
    ``src == tgt`` 라, 복원을 걸지 않은 예제는 "입력을 그대로 베껴라" 가 되어
    아무것도 가르치지 않습니다. 검증만 0 으로 두면 검증 손실이 복사 과제의
    손실이 되어 best 선택이 무의미해집니다.
    """

    foundation = config.foundation
    derived = copy.deepcopy(config)
    derived.data.dataset_dir = foundation.dataset_dir
    derived.data.denoise_probability = 1.0
    derived.data.validation_denoise_probability = 1.0
    derived.data.denoise_noise_density = foundation.noise_density
    derived.data.denoise_mean_span = foundation.mean_span
    # 손상 자체가 이 단계의 잡음입니다. 원문 토큰 dropout 을 겹치면 복원해야
    # 할 근거까지 지워집니다.
    derived.data.source_token_dropout = 0.0
    derived.data.decoder_input_noise = 0.0
    # foundation shard 에는 source-only 언어가 없습니다(설계상 제외). 남겨 두면
    # collator 가 그 언어의 복원 예제를 건너뛰려 하다가 아무것도 못 찾습니다.
    derived.data.source_only_languages = []
    # ``data.language_pairs`` 는 번역 설정 그대로 둡니다. 복원 과제의 "쌍"은
    # (ko, ko) 처럼 자기 자신인데, 공용 검증기는 두 키가 서로 달라야 한다고
    # 요구합니다 — 번역 설정에서는 옳은 규칙이므로 여기서 예외를 만들지
    # 않습니다. 이 단계가 실제로 읽는 언어는 foundation 데이터셋 manifest 가
    # 정하고(``IndexedParallelDataset`` 이 거기서 읽습니다), collator 는
    # record 의 ``src_language``/``target_language`` 만 봅니다. 설정의 쌍은
    # 이 경로에서 쓰이지 않습니다.

    training = derived.training
    training.output_dir = str(foundation_run_directory(config))
    training.max_steps = foundation.max_steps
    training.batch_size_per_gpu = foundation.batch_size_per_gpu
    training.gradient_accumulation_steps = foundation.gradient_accumulation_steps
    training.learning_rate = foundation.learning_rate
    training.min_learning_rate_ratio = foundation.min_learning_rate_ratio
    training.warmup_steps = foundation.warmup_steps
    training.eval_every = foundation.eval_every
    training.eval_batches = foundation.eval_batches
    training.save_every = foundation.save_every
    training.early_stopping_patience = foundation.early_stopping_patience
    training.early_stopping_min_delta = foundation.early_stopping_min_delta
    training.final_export_formats = list(foundation.final_export_formats)
    training.resume_from = None
    training.tensorboard_dir = None
    # 방향별 지표는 번역 단계의 개념입니다. 복원 과제에는 방향이 없습니다.
    training.sft_selection_metric = "global_nll"

    # 이 단계는 번역을 하지 않으므로 사후학습도 없습니다.
    derived.posttraining.enabled = False
    derived.foundation.enabled = False
    return derived


@dataclass
class FoundationOutcome:
    """단계 실행 결과. 번역 단계가 provenance 로 기록합니다."""

    ran: bool
    reason: str
    best_checkpoint: str | None = None
    selected_step: int | None = None
    languages: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "ran": self.ran,
            "reason": self.reason,
            "best_checkpoint": self.best_checkpoint,
            "selected_step": self.selected_step,
            "languages": list(self.languages),
            "warnings": list(self.warnings),
        }


__all__ = [
    "FoundationOutcome",
    "FoundationPlan",
    "PIPELINE_IDENTITY_SCHEMA",
    "build_foundation_config",
    "build_translation_pipeline_identity",
    "foundation_run_directory",
    "plan_foundation_stage",
]
