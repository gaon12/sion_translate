# YAML constructors and raw configuration objects are dynamically typed.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Sequence

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from sion_translate.artifacts import (
    DEFAULT_DATASET_DIRECTORY,
    DEFAULT_FOUNDATION_DATASET_DIRECTORY,
    DEFAULT_MONOLINGUAL_CORPUS_DIRECTORY,
    DEFAULT_RUN_DIRECTORY,
    DEFAULT_TOKENIZER_FEATURES,
    DEFAULT_TOKENIZER_MODEL,
    FOUNDATION_RELEASE_NAME,
    TRANSLATION_RELEASE_NAME,
)
from sion_translate.data.monolingual import (
    DEFAULT_LANGUAGE_SAMPLING_ALPHA,
    foundation_languages,
)
from sion_translate.data.records import (
    normalize_language_pairs,
    normalize_translation_directions,
)
from sion_translate.language_tags import canonicalize_language_pair, canonicalize_language_tags
from sion_translate.synthetic import (
    DEFAULT_SYNTHETIC_PREFIXES,
    DEFAULT_SYNTHETIC_SAMPLING_WEIGHT,
    normalize_synthetic_prefixes,
)


@dataclass
class ExperimentalConfig:
    bats_enabled: bool = False
    bats_dim: int = 256
    bats_loss_weight: float = 0.0
    bats_coverage_weight: float = 0.0
    bats_stride: int = 2
    bats_max_positions: int = 1024
    core_enabled: bool = False
    register_classes: int = 4
    register_loss_weight: float = 0.05
    tetm_enabled: bool = False
    tetm_types: int = 10
    tetm_modes: int = 5
    morphoscript_enabled: bool = False
    morphoscript_interval: int = 4
    script_classes: int = 9
    # Decoder가 불확실한 위치에서 encoder evidence를 한 번 더 조회하고,
    # 요청 비율이 무제한으로 커지지 않도록 uncertainty/budget loss로 제어합니다.
    evidence_repair_enabled: bool = False
    evidence_uncertainty_loss_weight: float = 0.02
    evidence_budget_loss_weight: float = 0.001
    evidence_budget_target: float = 0.25
    evidence_repair_gain_loss_weight: float = 0.005
    evidence_minimum_gain: float = 0.01
    # 첫 next-token 분포 전체를 token embedding 기대값으로 압축해 decoder
    # hidden state에 되먹인 뒤, 전체 vocabulary logits를 다시 계산합니다.
    # 기존 checkpoint는 모듈 자체가 없도록 기본값을 꺼 둡니다.
    candidate_refinement_enabled: bool = False
    candidate_refinement_steps: int = 1
    candidate_refinement_temperature: float = 1.0
    candidate_refinement_loss_weight: float = 0.25
    candidate_refinement_vocab_chunk_size: int = 2048
    # 원문/정답의 pooled semantic representation을 대조 학습해 직역 표면형이
    # 달라도 의미가 보존되도록 하는 보조 목적입니다.
    semantic_parity_enabled: bool = False
    semantic_parity_dim: int = 256
    semantic_parity_temperature: float = 0.07
    semantic_parity_loss_weight: float = 0.05
    # 공유 블록 반복(latent reasoning): 인코더 마지막 N개 층을 같은 가중치로
    # 여러 번 통과시켜, 파라미터를 늘리지 않고 깊이만 늘립니다. 명시적 사고
    # 토큰 없이 hidden state 안에서 추가 계산을 하게 하는 구조입니다.
    # 0 이면 꺼짐 — 기본이 꺼져 있어야 기존 체크포인트가 그대로 동작합니다.
    recurrent_block_layers: int = 0
    recurrent_steps: int = 1
    # Kimi K3의 SiTU-GLU: SwiGLU의 gate/up pre-activation을 부드럽게
    # 제한해 큰 activation과 저정밀도 overflow를 줄입니다. projection
    # 파라미터 모양은 같아서 기능을 끈 기존 체크포인트와 호환됩니다.
    situglu_enabled: bool = False
    situglu_gate_beta: float = 4.0
    situglu_up_beta: float = 25.0

    def validate(self) -> None:
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.candidate_refinement_enabled, bool
        ):
            raise ValueError("experimental.candidate_refinement_enabled must be a boolean")
        if isinstance(self.candidate_refinement_steps, bool) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.candidate_refinement_steps, int
        ):
            raise ValueError("experimental.candidate_refinement_steps must be an integer")
        if isinstance(self.candidate_refinement_vocab_chunk_size, bool) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            self.candidate_refinement_vocab_chunk_size, int
        ):
            raise ValueError(
                "experimental.candidate_refinement_vocab_chunk_size must be an integer"
            )
        for name, value in (
            ("candidate_refinement_temperature", self.candidate_refinement_temperature),
            ("candidate_refinement_loss_weight", self.candidate_refinement_loss_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))  # pyright: ignore[reportUnnecessaryIsInstance]
                or not math.isfinite(value)
            ):
                raise ValueError(f"experimental.{name} must be a finite real number")
        if self.recurrent_block_layers < 0:
            raise ValueError("experimental.recurrent_block_layers must be non-negative")
        if self.recurrent_steps < 1:
            raise ValueError("experimental.recurrent_steps must be at least 1")
        if not 1 <= self.candidate_refinement_steps <= 4:
            raise ValueError("experimental.candidate_refinement_steps must be between 1 and 4")
        if self.candidate_refinement_temperature <= 0:
            raise ValueError("experimental.candidate_refinement_temperature must be positive")
        if self.candidate_refinement_vocab_chunk_size <= 0:
            raise ValueError("experimental.candidate_refinement_vocab_chunk_size must be positive")
        if self.situglu_gate_beta <= 0 or self.situglu_up_beta <= 0:
            raise ValueError("experimental SiTU-GLU beta values must be positive")
        if self.recurrent_block_layers and self.recurrent_steps == 1:
            warnings.warn(
                "experimental.recurrent_block_layers 가 설정됐지만 recurrent_steps 가 "
                "1 입니다. 한 번만 통과하면 일반 층과 같으므로 반복 계산이 없습니다 — "
                "recurrent_steps 를 2 이상으로 두거나 recurrent_block_layers 를 0 으로 "
                "두십시오.",
                RuntimeWarning,
                stacklevel=2,
            )
        for name, value in (
            ("bats_dim", self.bats_dim),
            ("bats_stride", self.bats_stride),
            ("bats_max_positions", self.bats_max_positions),
            ("register_classes", self.register_classes),
            ("tetm_types", self.tetm_types),
            ("tetm_modes", self.tetm_modes),
            ("morphoscript_interval", self.morphoscript_interval),
            ("script_classes", self.script_classes),
            ("semantic_parity_dim", self.semantic_parity_dim),
        ):
            if value <= 0:
                raise ValueError(f"experimental.{name} must be positive")
        if self.semantic_parity_temperature <= 0:
            raise ValueError("experimental.semantic_parity_temperature must be positive")
        if self.tetm_enabled and self.tetm_types < 9:
            raise ValueError(
                "experimental.tetm_types must be at least 9 when TETM is enabled "
                "because protected slots use type ID 8"
            )
        if self.tetm_enabled and self.tetm_modes < 5:
            raise ValueError(
                "experimental.tetm_modes must be at least 5 when TETM is enabled "
                "because protected slots use mode ID 4"
            )
        for name, value in (
            ("bats_loss_weight", self.bats_loss_weight),
            ("bats_coverage_weight", self.bats_coverage_weight),
            ("register_loss_weight", self.register_loss_weight),
            ("evidence_uncertainty_loss_weight", self.evidence_uncertainty_loss_weight),
            ("evidence_budget_loss_weight", self.evidence_budget_loss_weight),
            ("evidence_repair_gain_loss_weight", self.evidence_repair_gain_loss_weight),
            ("candidate_refinement_loss_weight", self.candidate_refinement_loss_weight),
            ("semantic_parity_loss_weight", self.semantic_parity_loss_weight),
        ):
            if value < 0:
                raise ValueError(f"experimental.{name} must be non-negative")
        if not 0.0 <= self.evidence_budget_target <= 1.0:
            raise ValueError("experimental.evidence_budget_target must be in [0, 1]")
        if self.evidence_minimum_gain < 0:
            raise ValueError("experimental.evidence_minimum_gain must be non-negative")
        if self.candidate_refinement_enabled and self.candidate_refinement_loss_weight == 0:
            raise ValueError(
                "experimental.candidate_refinement_loss_weight must be positive when "
                "candidate refinement is enabled"
            )

        # 모듈을 켜 두고 그 보조 손실 가중치를 모두 0으로 두면 파라미터와 순전파
        # 비용만 늘고 학습 신호는 없습니다. 조용히 낭비되므로 알려 줍니다.
        if self.bats_enabled and not (self.bats_loss_weight or self.bats_coverage_weight):
            warnings.warn(
                "experimental.bats_enabled 가 켜져 있지만 bats_loss_weight 와 "
                "bats_coverage_weight 가 모두 0 입니다. 연산과 파라미터만 늘고 "
                "학습되는 것은 없습니다 — 가중치를 주거나 bats_enabled 를 끄십시오.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.core_enabled and not self.register_loss_weight:
            warnings.warn(
                "experimental.core_enabled 가 켜져 있지만 register_loss_weight 가 "
                "0 입니다. register 분류기가 학습되지 않습니다 — 가중치를 주거나 "
                "core_enabled 를 끄십시오.",
                RuntimeWarning,
                stacklevel=2,
            )


@dataclass
class ModelConfig:
    vocab_size: int = 0
    d_model: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 2
    d_ff: int = 1536
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_embeddings: bool = True
    label_smoothing: float = 0.10
    z_loss_weight: float = 1e-4
    # H100-class GPUs have enough memory for the baseline without activation
    # recomputation. Disable by default for throughput; users can re-enable it
    # for larger models or memory-constrained devices.
    gradient_checkpointing: bool = False
    init_std: float = 0.02
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    def validate(self) -> None:
        self.experimental.validate()
        if self.vocab_size < 0:
            raise ValueError("vocab_size must be non-negative")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.encoder_layers <= 0 or self.decoder_layers <= 0:
            raise ValueError("encoder_layers and decoder_layers must be positive")
        if self.num_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("num_heads and num_kv_heads must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.max_seq_len < 8:
            raise ValueError("max_seq_len is unexpectedly small")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.z_loss_weight < 0:
            raise ValueError("z_loss_weight must be non-negative")
        if self.rms_norm_eps <= 0 or self.rope_base <= 0 or self.init_std <= 0:
            raise ValueError("normalization, RoPE, and initialization values must be positive")


@dataclass
class DataConfig:
    # 원천 JSONL 이 들어 있는 폴더. 자동 파이프라인이 이 폴더를 스캔해
    # 파일 추가/변경을 감지하고 토크나이저·데이터셋을 자동 준비합니다.
    raw_dir: str = "data"
    # 학습할 언어쌍 = JSONL 의 키 이름. 예: ["ko", "ja"] 또는 ["en", "de"].
    # 토크나이저의 <2xx>/<denoise_xx> 제어 토큰, 전처리, 방향 태그가
    # 모두 이 값을 따라가므로 다른 언어쌍도 설정만 바꾸면 학습됩니다.
    language_pair: list[str] = field(default_factory=list)
    # 여러 언어쌍을 한 모델에서 학습할 때 사용합니다. 비어 있으면 위의
    # language_pair 한 쌍만 사용합니다. YAML에서는 둘 중 하나만 적습니다.
    language_pairs: list[list[str]] = field(default_factory=list)
    # 실제로 학습할 directed edge 목록입니다. 예를 들어 de↔fr와 sw→ar를
    # 함께 학습하려면 [[de, fr], [fr, de], [sw, ar]]로 둡니다. 비어 있으면
    # 기존 bidirectional/source_only_languages 정책에서 자동으로 계산합니다.
    translation_directions: list[list[str]] = field(default_factory=list)
    # ``원문 <draft> 초안 -> 정답`` revision 학습이 실제로 이루어진 directed
    # edge 목록입니다. 비어 있으면 indexed row provenance에서 자동 검출합니다.
    # translation_directions와 별개로 기록해 revision 능력을 다른 edge로
    # 확장해 버리는 것을 막습니다.
    revision_directions: list[list[str]] = field(default_factory=list)
    # 원문으로만 등장하고 번역 결과로는 절대 나오면 안 되는 언어입니다.
    # 한본어(kj)처럼 "한국어와 일본어가 섞인 입력"을 받아 각각 단일어로
    # 번역하는 경우에 씁니다. 여기 등재된 언어가 포함된 쌍은 단방향으로만
    # 학습되어(kj->ko, kj->ja), 역방향(ko->kj, ja->kj)이 생성되지 않습니다.
    # 이것을 비워 두면 bidirectional=True 가 혼용문을 target 으로도 학습해
    # 한국어 출력에 가나를 주입하는 모델이 됩니다.
    source_only_languages: list[str] = field(default_factory=list)
    # 원문(입력) 쪽 토큰을 낮은 확률로 무작위 탈락시키는 온라인 증강.
    # 소량(0.05 안팎)은 과적합을 줄여 주지만, 큰 값은 오히려 해롭습니다.
    # 검증에는 절대 적용되지 않습니다. 0 이면 끕니다.
    source_token_dropout: float = 0.05
    # 디코더 입력 토큰을 낮은 확률로 교란하는 exposure bias 완화 장치.
    # teacher forcing 은 항상 정답 접두사만 보여 주므로, 추론에서 첫 오류가
    # 나면 모델이 학습한 적 없는 상태에 놓이고 오류가 누적됩니다. 사후학습(MRT)
    # 이 이 문제를 다루지만 그건 본학습이 끝난 뒤 수천 스텝짜리 미세조정이고,
    # 본학습 자체에는 대책이 없습니다.
    #
    # 정답(labels)은 건드리지 않습니다. 바뀌는 것은 디코더가 무엇을 보고
    # 다음 토큰을 예측하느냐뿐이라, 목적함수는 그대로입니다.
    #
    # 기본값 0(끔). 측정 전에는 켜지 않습니다 — 디코더의 조건부를 바꾸는
    # 개입이라 처음부터 학습하는 run 에서 검증 없이 켜는 것은 도박입니다.
    # 시도한다면 0.1 부터 보고 검증 loss 로 A/B 하십시오.
    decoder_input_noise: float = 0.0
    # 역번역(backtranslation) 등 합성 데이터 파일의 이름 접두사.
    # 이 접두사로 시작하는 파일은 ① 항상 train split 에만 들어가고
    # ② 샘플링 가중치가 자동으로 낮아집니다 (합성 데이터 과다 방지).
    # 기존 단일 설정은 계속 지원하며 synthetic_prefixes와 합쳐집니다.
    synthetic_prefix: str = "bt_"
    synthetic_prefixes: list[str] = field(default_factory=lambda: list(DEFAULT_SYNTHETIC_PREFIXES))
    synthetic_sampling_weight: float = DEFAULT_SYNTHETIC_SAMPLING_WEIGHT
    # ``원문 <draft> 초안 -> 정답`` revision 예제가 실제 학습 데이터에
    # 포함됐음을 산출물 metadata에 기록합니다. ``revise_`` 접두사의 원천은
    # 학습 CLI가 자동 감지하며, 다른 파일명을 썼을 때만 직접 켭니다.
    revision_examples: bool = False
    # 글로서리(용어집) JSON 경로. 지정하면 sion-translate/evaluate 가 이 파일을
    # 기본으로 불러와 지정한 용어를 정해진 대응어로 강제합니다. 빈 문자열이면 끔.
    glossary: str = ""
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    tokenizer_features: str = DEFAULT_TOKENIZER_FEATURES
    dataset_dir: str = DEFAULT_DATASET_DIRECTORY
    train_split: str = "train"
    validation_split: str = "validation"
    bidirectional: bool = True
    # split 배정과 누출 방지를 완전일치 문자열 대신 문자 5-gram MinHash
    # 버킷으로 수행합니다. 조사 하나만 다른 근사 중복이 train 과 holdout 을
    # 넘나들지 못하게 하는 안전한 기본값입니다. 과거 exact split을 재현해야
    # 할 때만 명시적으로 끄십시오. 두 방식의 holdout 점수는 직접 비교할 수 없습니다.
    approximate_split: bool = True
    max_source_length: int = 512
    max_target_length: int = 512
    # Tensor Core 친화적인 sequence shape을 위해 동적 padding 길이를
    # 이 배수로 올립니다. 1이면 배수 padding을 끕니다.
    pad_to_multiple_of: int = 8
    denoise_probability: float = 0.10
    validation_denoise_probability: float = 0.0
    denoise_noise_density: float = 0.15
    denoise_mean_span: float = 3.0
    num_workers: int = 4
    bucket_size: int = 4096
    source_sampling_alpha: float = 1.0
    source_sampling_weights: dict[str, float] = field(default_factory=dict)
    max_source_upsampling: float = 3.0

    def configured_language_pairs(self) -> tuple[tuple[str, str], ...]:
        has_single = bool(self.language_pair)
        has_multiple = bool(self.language_pairs)
        if has_single == has_multiple:
            raise ValueError("configure exactly one of data.language_pair or data.language_pairs")
        raw_pairs = self.language_pairs if has_multiple else [self.language_pair]
        return normalize_language_pairs(language_pairs=raw_pairs)

    def configured_synthetic_prefixes(self) -> tuple[str, ...]:
        return normalize_synthetic_prefixes(
            self.synthetic_prefixes,
            legacy_prefix=self.synthetic_prefix,
        )

    def configured_source_only_languages(self) -> tuple[str, ...]:
        return canonicalize_language_tags(
            self.source_only_languages,
            field="data.source_only_languages",
            reject_duplicates=False,
        )

    def configured_translation_directions(self) -> tuple[tuple[str, str], ...]:
        """Return the directed edges the indexed dataset can actually train."""

        return normalize_translation_directions(
            self.configured_language_pairs(),
            self.translation_directions or None,
            bidirectional=self.bidirectional,
            source_only_languages=self.configured_source_only_languages(),
        )

    def configured_revision_directions(self) -> tuple[tuple[str, str], ...]:
        """Return explicitly authenticated revision edges, if configured."""

        trained = set(self.configured_translation_directions())
        directions: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for index, raw_direction in enumerate(self.revision_directions):
            direction = canonicalize_language_pair(
                raw_direction,
                field=f"data.revision_directions[{index}]",
            )
            if direction in seen:
                raise ValueError(
                    "duplicate data.revision_directions after BCP 47 canonicalization: "
                    f"{raw_direction!r}"
                )
            if direction not in trained:
                raise ValueError(
                    "data.revision_directions must be a subset of "
                    f"data.translation_directions; got {direction!r}"
                )
            seen.add(direction)
            directions.append(direction)
        return tuple(directions)

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                language for pair in self.configured_language_pairs() for language in pair
            )
        )


@dataclass
class TrainingConfig:
    # 학습 산출물(체크포인트/로그/exports) 위치. sion-translate 와 sion-augment 도
    # 같은 기본값을 보고 모델을 자동으로 찾으므로 함께 움직입니다.
    output_dir: str = DEFAULT_RUN_DIRECTORY
    seed: int = 20260710
    # 공식 학습은 corpus를 중간에서 자르지 않고 이 횟수만큼 완주합니다.
    # max_steps는 짧은 디버그와 구버전 설정을 위한 명시적 override입니다.
    num_train_epochs: int = 3
    max_steps: int | None = None
    batch_size_per_gpu: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    precision: str = "bf16"
    compile: bool = False
    # auto는 모델이 GPU 한 장에 충분히 들어가면 통신이 적은 DDP를,
    # 메모리가 부족할 때만 FSDP2를 선택합니다.
    parallel_strategy: str = "auto"
    # 이전 설정 파일 호환용. 새 설정에서는 parallel_strategy를 사용합니다.
    fsdp2: bool | None = None
    fsdp_reduce_dtype: str = "auto"
    reshard_after_forward: bool = True
    log_every: int = 10
    eval_every: int = 250
    eval_batches: int = 20
    save_every: int = 500
    # 초반 validation 변동으로 너무 일찍 끝나지 않도록 이만큼은 완주합니다.
    early_stopping_min_epochs: int = 2
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    # SFT best/early stopping 기준. 방향별 token NLL을 같은 비중으로 평균하면
    # 데이터가 많은 방향이 적은 방향의 품질 저하를 평균값으로 가리지 못합니다.
    # 방향 메타데이터가 없는 custom caller에서는 trainer가 global NLL로 fallback합니다.
    sft_selection_metric: str = "macro_direction_nll"
    # 가중치 지수이동평균(EMA). 매 step 뒤 shadow 가중치를
    # shadow = decay*shadow + (1-decay)*param 으로 갱신합니다.
    # 번역 모델에서 검증 loss/BLEU 를 안정적으로 개선하는 검증된 기법입니다.
    # 0.0 이면 비활성화됩니다.
    ema_decay: float = 0.999
    tensorboard: bool = True
    tensorboard_dir: str | None = None
    resume_from: str | None = None
    # 전체 학습이 끝난 뒤 선택된 best 가중치에서 한 번만 생성합니다.
    # 중간 best/latest 저장은 학습을 오래 멈추지 않도록 FP32만 유지합니다.
    final_export_formats: list[str] = field(
        default_factory=lambda: [
            "fp32",
            "fp16",
            "bf16",
            "int8",
            "int4",
            "gguf_q4_k_m",
            "transformers",
        ]
    )


SUPPORTED_EXPORT_FORMATS = frozenset(
    {
        "fp32",
        "fp16",
        "bf16",
        "int8",
        "int4",
        "fp8",
        "gguf_q4_k_m",
        "transformers",
    }
)


def _validate_export_formats(formats: Sequence[str], *, field_name: str) -> None:
    """Both training stages publish their own artifacts, so both need this."""

    if not formats:
        raise ValueError(f"{field_name}.final_export_formats must contain at least one format")
    if len(set(formats)) != len(formats):
        raise ValueError(f"{field_name}.final_export_formats must not contain duplicates")
    unknown = sorted(set(formats) - SUPPORTED_EXPORT_FORMATS)
    if unknown:
        raise ValueError(
            f"unsupported {field_name}.final_export_formats: "
            f"{unknown}; supported={sorted(SUPPORTED_EXPORT_FORMATS)}"
        )


@dataclass
class FoundationConfig:
    """번역 학습 **이전**의 단일어 사전학습(pre-pre-train) 설정.

    이 단계는 언어별 단일어 텍스트로 span-corruption 복원만 학습합니다
    (``<denoise_xx>`` 과제). 번역쌍을 전혀 쓰지 않으므로 결과물은 번역
    모델이 아니라 encoder-decoder **파운데이션 모델**이며, 그래서 이후
    번역 단계와 다른 이름(``sion``)으로 따로 저장·배포됩니다.

    단계 순서: foundation → SFT(번역) → MRT(사후학습).
    """

    # 코퍼스 폴더에 유효한 데이터가 있으면 자동 실행합니다. 폴더가 없거나
    # 비어 있으면 이유를 출력하고 건너뜁니다 — 토크나이저·데이터셋 자동
    # 감지와 같은 방식입니다. false 로 두면 데이터가 있어도 건너뜁니다.
    enabled: bool = True
    corpus_dir: str = DEFAULT_MONOLINGUAL_CORPUS_DIRECTORY
    dataset_dir: str = DEFAULT_FOUNDATION_DATASET_DIRECTORY
    # 배포 이름. 이 단계의 산출물은 번역 모델이 아니라 그 파운데이션입니다.
    release_name: str = FOUNDATION_RELEASE_NAME
    # 비어 언어를 번역 edge에 추가하지 않고 foundation에만 넣을 때
    # 명시합니다. 비워 두면 번역 언어에서 source-only variety를 뺀
    # 기존 목록을 사용합니다. 예: [ko, ja, en].
    languages: list[str] = field(default_factory=list)

    # ── 코퍼스 구성 ──────────────────────────────────────────────────
    # 언어 간 온도 샘플링. 1.0 은 분량 정비례, 낮출수록 균등에 가깝습니다.
    language_sampling_alpha: float = DEFAULT_LANGUAGE_SAMPLING_ALPHA
    # 이 비중 미만인 언어가 있으면 경고합니다(0 이면 경고 안 함).
    minimum_language_share: float = 0.05
    # 학습 대상 언어 중 단일어 데이터가 아예 없는 언어가 있을 때 중단할지.
    # 기본은 false: 경고하고 있는 언어로 진행합니다. 언어를 나중에 채우는
    # 것이 정상적인 작업 흐름이기 때문입니다.
    require_all_languages: bool = False
    minimum_characters: int = 8
    maximum_characters: int = 4000
    deduplicate: bool = True
    # 구조화 reasoning 파일이 있을 때 foundation 배치에서 차지할 목표 행
    # 비중입니다. trace는 일반 복원 target보다 길어서 작은 행 비중도 decoder
    # token 기준으로는 충분한 보조 신호가 됩니다. 파일이 없으면 사용되지 않습니다.
    reasoning_sample_share: float = 0.05

    # ── 복원 과제 ────────────────────────────────────────────────────
    # 일반 단일어 행은 100% denoising 입니다. 번역 SFT 의
    # denoise_probability와 달리 확률이 아니라 과제 자체이므로 별도 값을 두지
    # 않습니다. reasoning_*.jsonl 행은 collator가 이 손상을 명시적으로 우회합니다.
    noise_density: float = 0.15
    mean_span: float = 3.0

    # ── 토크나이저 ───────────────────────────────────────────────────
    # 토크나이저 학습에 단일어 코퍼스를 넣되, 언어별로 "병렬 코퍼스의 해당
    # 언어 문장 수 × 이 배수" 까지만 샘플링합니다. 넣지 않으면 foundation
    # 단계가 자기 코퍼스에 없는 어휘로 학습하고, 전량 넣으면 분량이 큰
    # 언어가 vocab 을 독식해 다른 언어 토큰화가 나빠집니다. 0 이면 단일어
    # 코퍼스를 토크나이저 학습에서 제외합니다.
    tokenizer_sample_ratio: float = 0.4

    # ── 학습 ─────────────────────────────────────────────────────────
    num_train_epochs: int = 3
    max_steps: int | None = None
    batch_size_per_gpu: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 2_000
    eval_every: int = 1_000
    eval_batches: int = 50
    save_every: int = 2_000
    early_stopping_min_epochs: int = 2
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.0
    shard_size: int = 200_000
    validation_fraction: float = 0.002
    # 이 단계 산출물의 export 형식. 파운데이션 모델은 이어서 미세조정하는
    # 것이 용도이므로 기본은 학습을 이어갈 수 있는 형식만 냅니다.
    final_export_formats: list[str] = field(
        default_factory=lambda: ["fp32", "bf16", "transformers"]
    )

    def validate(self) -> None:
        if not self.corpus_dir:
            raise ValueError("foundation.corpus_dir must be non-empty")
        if not self.dataset_dir:
            raise ValueError("foundation.dataset_dir must be non-empty")
        if (
            not self.release_name
            or self.release_name != self.release_name.strip()
            or not self.release_name.isascii()
        ):
            raise ValueError("foundation.release_name must be a normalized non-empty ASCII name")
        if self.release_name == TRANSLATION_RELEASE_NAME:
            raise ValueError(
                "foundation.release_name must differ from the translation release name "
                f"({TRANSLATION_RELEASE_NAME!r}); the two stages are published separately"
            )
        self.languages = list(
            canonicalize_language_tags(
                self.languages,
                field="foundation.languages",
            )
        )
        if not 0.0 < self.language_sampling_alpha <= 1.0:
            raise ValueError("foundation.language_sampling_alpha must be in (0, 1]")
        if not 0.0 <= self.minimum_language_share < 1.0:
            raise ValueError("foundation.minimum_language_share must be in [0, 1)")
        if self.minimum_characters < 1:
            raise ValueError("foundation.minimum_characters must be positive")
        if self.maximum_characters <= self.minimum_characters:
            raise ValueError(
                "foundation.maximum_characters must be greater than minimum_characters"
            )
        if not 0.0 <= self.reasoning_sample_share <= 0.10:
            raise ValueError("foundation.reasoning_sample_share must be in [0, 0.10]")
        if not 0.0 < self.noise_density < 1.0:
            raise ValueError("foundation.noise_density must be in (0, 1)")
        if self.mean_span <= 0:
            raise ValueError("foundation.mean_span must be positive")
        if self.tokenizer_sample_ratio < 0:
            raise ValueError("foundation.tokenizer_sample_ratio must be non-negative")
        for name, value in (
            ("num_train_epochs", self.num_train_epochs),
            ("batch_size_per_gpu", self.batch_size_per_gpu),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("eval_every", self.eval_every),
            ("eval_batches", self.eval_batches),
            ("save_every", self.save_every),
            ("shard_size", self.shard_size),
        ):
            if value <= 0:
                raise ValueError(f"foundation.{name} must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("foundation.max_steps must be positive when specified")
        if self.learning_rate <= 0:
            raise ValueError("foundation.learning_rate must be positive")
        if not 0.0 <= self.min_learning_rate_ratio <= 1.0:
            raise ValueError("foundation.min_learning_rate_ratio must be in [0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("foundation.warmup_steps must be non-negative")
        if self.max_steps is not None and self.warmup_steps > self.max_steps:
            raise ValueError("foundation.warmup_steps cannot exceed max_steps")
        if self.early_stopping_patience < 0:
            raise ValueError("foundation.early_stopping_patience must be non-negative")
        if self.early_stopping_min_epochs <= 0:
            raise ValueError("foundation.early_stopping_min_epochs must be positive")
        if self.max_steps is None and self.early_stopping_min_epochs > self.num_train_epochs:
            raise ValueError("foundation.early_stopping_min_epochs cannot exceed num_train_epochs")
        if self.early_stopping_min_delta < 0:
            raise ValueError("foundation.early_stopping_min_delta must be non-negative")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("foundation.validation_fraction must be in (0, 0.5)")
        _validate_export_formats(self.final_export_formats, field_name="foundation")


@dataclass
class PostTrainingConfig:
    """SFT 뒤 실행하는 복합 최소위험 + 다중 후보 선호학습 설정."""

    enabled: bool = True
    method: str = "mrt"
    num_train_epochs: int = 2
    max_steps: int | None = None
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-5
    warmup_steps: int = 200
    samples_per_source: int = 2
    # 후보 전체의 vocabulary logits를 한 번에 만들지 않고, source마다
    # 이 개수씩 나눠 점수를 계산합니다. 1이 VRAM 사용량이 가장 낮습니다.
    candidate_micro_batch: int = 1
    # 후보 scoring graph의 큰 activation은 저장하지 않고 backward 때 재계산합니다.
    candidate_gradient_checkpointing: bool = True
    sampling_temperature: float = 1.0
    top_k: int = 64
    max_new_tokens: int = 256
    risk_weight: float = 0.20
    mrt_alpha: float = 1.0
    # reward 차이가 있는 모든 후보쌍의 순서를 직접 학습합니다.
    preference_weight: float = 0.10
    preference_min_gap: float = 0.05
    preference_temperature: float = 1.0
    # 단일 metric reward hacking을 줄이는 복합 보상 가중치입니다.
    reward_chrf_weight: float = 0.55
    reward_token_f1_weight: float = 0.15
    reward_number_weight: float = 0.10
    reward_structured_weight: float = 0.05
    reward_slot_weight: float = 0.05
    reward_language_weight: float = 0.05
    reward_length_weight: float = 0.05
    reward_repetition_penalty: float = 0.15
    reward_copy_penalty: float = 0.10
    # 값 변조는 가중 성분이 아니라 하드 페널티입니다. `reward_number_weight`
    # 는 비율이라, 값 하나를 지어낸 후보도 chrF 가 조금 높으면 이깁니다.
    # 배포 홀드아웃 10문장 중 8문장의 숫자가 바뀐 것이 그 결과입니다.
    # 변조 하나당 이 값을 빼며, 기본값은 chrF 가중치(0.55)로 만회할 수 있는
    # 현실적인 chrF 차이보다 크게 잡았습니다 — 숫자를 틀리고 이기는 후보가
    # 없어야 한다는 뜻입니다. 0 으로 두면 이전의 가중치 전용 동작이 됩니다.
    reward_number_corruption_penalty: float = 0.35
    # 후보를 원문 언어로 다시 번역해 원문 복원 여부를 확인합니다. 정답과
    # 우연히 비슷한 후보가 보상을 독식하지 못하게 하는 순환 일관성 검증입니다.
    roundtrip_enabled: bool = False
    roundtrip_reward_weight: float = 0.20
    roundtrip_failure_penalty: float = 0.15
    roundtrip_min_score: float = 0.55
    roundtrip_num_beams: int = 1
    roundtrip_max_new_tokens: int = 256
    # best/early stopping은 이 beam 수로 생성한 번역의 복합 보상을 사용합니다.
    validation_num_beams: int = 4
    # MRT best 선택 지표. 평균 reward 하나로 고르면 한 방향이 후퇴해도 다른
    # 방향이 더 오르면 그 체크포인트가 best 가 됩니다. 이 저장소는 이미
    # ko→ja 59.81 대 ja→ko 49.87 로 방향 격차가 있어서, 평균만 보면 격차가
    # 벌어지는 것을 놓칩니다. 기본은 가장 낮은 방향을 올리는 쪽입니다.
    # 방향 메타데이터가 없으면 trainer 가 평균 reward 로 되돌아갑니다.
    selection_metric: str = "worst_direction_reward"
    eval_batch_size_per_gpu: int = 1
    eval_every: int = 250
    save_every: int = 1_000
    early_stopping_min_epochs: int = 2
    early_stopping_patience: int = 5


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    # 단계 순서대로: foundation(단일어) → training(번역 SFT) → posttraining(MRT).
    foundation: FoundationConfig = field(default_factory=FoundationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    posttraining: PostTrainingConfig = field(default_factory=PostTrainingConfig)

    def foundation_languages(self) -> tuple[str, ...]:
        """foundation 단계가 실제로 학습할 언어 (source-only 제외)."""

        return foundation_languages(
            self.foundation.languages or self.data.languages,
            self.data.configured_source_only_languages(),
        )

    def validate(self) -> None:
        self.model.validate()
        if not 0.0 <= self.data.source_token_dropout < 0.5:
            raise ValueError("source_token_dropout must be in [0, 0.5)")
        if not 0.0 <= self.data.decoder_input_noise < 0.5:
            raise ValueError("decoder_input_noise must be in [0, 0.5)")
        if not self.data.synthetic_prefix:
            raise ValueError("synthetic_prefix must be non-empty")
        self.data.configured_synthetic_prefixes()
        if not 0.0 <= self.data.synthetic_sampling_weight <= 1.0:
            raise ValueError("synthetic_sampling_weight must be in [0, 1]")
        if not 0.0 <= self.data.denoise_probability <= 1.0:
            raise ValueError("denoise_probability must be in [0, 1]")
        if not 0.0 <= self.data.validation_denoise_probability <= 1.0:
            raise ValueError("validation_denoise_probability must be in [0, 1]")
        if not 0.0 <= self.data.denoise_noise_density < 1.0:
            raise ValueError("denoise_noise_density must be in [0, 1)")
        if self.data.denoise_mean_span <= 0:
            raise ValueError("denoise_mean_span must be positive")
        if self.data.max_source_length < 2 or self.data.max_target_length < 2:
            raise ValueError("maximum source and target lengths must be at least 2")
        if self.data.max_source_length > self.model.max_seq_len:
            raise ValueError("max_source_length cannot exceed model.max_seq_len")
        if self.data.max_target_length > self.model.max_seq_len:
            raise ValueError("max_target_length cannot exceed model.max_seq_len")
        if self.data.pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be at least 1")
        if self.data.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.data.bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if self.data.source_sampling_alpha <= 0.0:
            raise ValueError("source_sampling_alpha must be positive")
        if self.data.max_source_upsampling < 1.0:
            raise ValueError("max_source_upsampling must be at least 1")
        if any(
            not name or weight < 0 for name, weight in self.data.source_sampling_weights.items()
        ):
            raise ValueError(
                "source_sampling_weights must have non-empty names and non-negative values"
            )
        if self.training.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.training.max_steps is not None and self.training.max_steps <= 0:
            raise ValueError("max_steps must be positive when specified")
        if self.training.batch_size_per_gpu <= 0:
            raise ValueError("batch_size_per_gpu must be positive")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.training.min_learning_rate_ratio <= 1.0:
            raise ValueError("min_learning_rate_ratio must be in [0, 1]")
        if self.training.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if (
            self.training.max_steps is not None
            and self.training.warmup_steps > self.training.max_steps
        ):
            raise ValueError("warmup_steps cannot exceed max_steps")
        if self.training.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.training.adam_beta1 < 1.0:
            raise ValueError("adam_beta1 must be in [0, 1)")
        if not 0.0 <= self.training.adam_beta2 < 1.0:
            raise ValueError("adam_beta2 must be in [0, 1)")
        if self.training.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if self.training.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.training.precision.lower() not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be one of: fp32, bf16, fp16")
        if self.training.parallel_strategy.lower() not in {"auto", "ddp", "fsdp2"}:
            raise ValueError("parallel_strategy must be one of: auto, ddp, fsdp2")
        if self.training.fsdp_reduce_dtype.lower() not in {"auto", "fp32", "bf16"}:
            raise ValueError("fsdp_reduce_dtype must be one of: auto, fp32, bf16")
        if self.training.fsdp2 is not None and self.training.parallel_strategy.lower() != "auto":
            raise ValueError("training.fsdp2 and training.parallel_strategy cannot both be set")
        for name, value in (
            ("log_every", self.training.log_every),
            ("eval_every", self.training.eval_every),
            ("eval_batches", self.training.eval_batches),
            ("save_every", self.training.save_every),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.training.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.training.early_stopping_min_epochs <= 0:
            raise ValueError("early_stopping_min_epochs must be positive")
        if (
            self.training.max_steps is None
            and self.training.early_stopping_min_epochs > self.training.num_train_epochs
        ):
            raise ValueError("early_stopping_min_epochs cannot exceed num_train_epochs")
        if self.training.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        supported_sft_selection_metrics = {
            "global_nll",
            "macro_direction_nll",
            "worst_direction_nll",
        }
        if self.training.sft_selection_metric.lower() not in supported_sft_selection_metrics:
            raise ValueError(
                "sft_selection_metric must be one of: "
                + ", ".join(sorted(supported_sft_selection_metrics))
            )
        if not 0.0 <= self.training.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        _validate_export_formats(self.training.final_export_formats, field_name="training")
        self.foundation.validate()
        post = self.posttraining
        if post.method != "mrt":
            raise ValueError("posttraining.method must be 'mrt'")
        for name, value in (
            ("num_train_epochs", post.num_train_epochs),
            ("batch_size_per_gpu", post.batch_size_per_gpu),
            ("gradient_accumulation_steps", post.gradient_accumulation_steps),
            ("samples_per_source", post.samples_per_source),
            ("candidate_micro_batch", post.candidate_micro_batch),
            ("max_new_tokens", post.max_new_tokens),
            ("eval_batch_size_per_gpu", post.eval_batch_size_per_gpu),
            ("eval_every", post.eval_every),
            ("save_every", post.save_every),
            ("validation_num_beams", post.validation_num_beams),
            ("roundtrip_num_beams", post.roundtrip_num_beams),
            ("roundtrip_max_new_tokens", post.roundtrip_max_new_tokens),
        ):
            if value <= 0:
                raise ValueError(f"posttraining.{name} must be positive")
        if post.max_steps is not None and post.max_steps <= 0:
            raise ValueError("posttraining.max_steps must be positive when specified")
        if post.samples_per_source < 2:
            raise ValueError("posttraining.samples_per_source must be at least 2")
        if post.learning_rate <= 0 or post.sampling_temperature <= 0 or post.mrt_alpha <= 0:
            raise ValueError(
                "posttraining learning rate, temperature, and mrt_alpha must be positive"
            )
        if not 0.0 <= post.risk_weight <= 1.0:
            raise ValueError("posttraining.risk_weight must be in [0, 1]")
        if not 0.0 <= post.preference_weight <= 1.0:
            raise ValueError("posttraining.preference_weight must be in [0, 1]")
        if not 0.0 <= post.preference_min_gap < 1.0:
            raise ValueError("posttraining.preference_min_gap must be in [0, 1)")
        if post.preference_temperature <= 0:
            raise ValueError("posttraining.preference_temperature must be positive")
        reward_weights = {
            name: getattr(post, name)
            for name in (
                "reward_chrf_weight",
                "reward_token_f1_weight",
                "reward_number_weight",
                "reward_structured_weight",
                "reward_slot_weight",
                "reward_language_weight",
                "reward_length_weight",
            )
        }
        if any(value < 0 for value in reward_weights.values()):
            raise ValueError("posttraining reward weights must be non-negative")
        if post.reward_number_corruption_penalty < 0:
            raise ValueError("posttraining.reward_number_corruption_penalty must be non-negative")
        if sum(reward_weights.values()) <= 0:
            raise ValueError("at least one posttraining reward weight must be positive")
        if not 0.0 <= post.reward_repetition_penalty <= 1.0:
            raise ValueError("posttraining.reward_repetition_penalty must be in [0, 1]")
        if not 0.0 <= post.reward_copy_penalty <= 1.0:
            raise ValueError("posttraining.reward_copy_penalty must be in [0, 1]")
        if post.roundtrip_reward_weight < 0:
            raise ValueError("posttraining.roundtrip_reward_weight must be non-negative")
        if not 0.0 <= post.roundtrip_failure_penalty <= 1.0:
            raise ValueError("posttraining.roundtrip_failure_penalty must be in [0, 1]")
        if not 0.0 <= post.roundtrip_min_score <= 1.0:
            raise ValueError("posttraining.roundtrip_min_score must be in [0, 1]")
        if post.roundtrip_enabled and post.roundtrip_reward_weight == 0:
            raise ValueError(
                "posttraining.roundtrip_reward_weight must be positive when roundtrip is enabled"
            )
        if post.top_k < 0:
            raise ValueError("posttraining.top_k must be non-negative")
        supported_post_selection = {"reward", "macro_direction_reward", "worst_direction_reward"}
        if post.selection_metric not in supported_post_selection:
            raise ValueError(
                "posttraining.selection_metric must be one of: "
                + ", ".join(sorted(supported_post_selection))
            )
        if post.warmup_steps < 0:
            raise ValueError("posttraining.warmup_steps must be non-negative")
        if post.max_steps is not None and post.warmup_steps > post.max_steps:
            raise ValueError("posttraining.warmup_steps cannot exceed max_steps")
        if post.early_stopping_patience < 0:
            raise ValueError("posttraining.early_stopping_patience must be non-negative")
        if post.early_stopping_min_epochs <= 0:
            raise ValueError("posttraining.early_stopping_min_epochs must be positive")
        if post.max_steps is None and post.early_stopping_min_epochs > post.num_train_epochs:
            raise ValueError(
                "posttraining.early_stopping_min_epochs cannot exceed num_train_epochs"
            )
        raw_pairs = (
            self.data.language_pairs if self.data.language_pairs else [self.data.language_pair]
        )
        pairs = self.data.configured_language_pairs()
        if len(pairs) != len(raw_pairs):
            raise ValueError("duplicate or reversed language pair after BCP 47 canonicalization")
        if self.data.language_pairs:
            self.data.language_pairs = [list(pair) for pair in pairs]
        else:
            self.data.language_pair = list(pairs[0])
        source_only = self.data.configured_source_only_languages()
        self.data.source_only_languages = list(source_only)
        if source_only:
            known = set(self.data.languages)
            unknown = sorted(set(source_only) - known)
            if unknown:
                raise ValueError(
                    "data.source_only_languages must appear in the configured language "
                    f"pairs; {unknown} do not (configured languages: {sorted(known)})"
                )
            for pair in pairs:
                if pair[0] in source_only and pair[1] in source_only:
                    raise ValueError(
                        "at most one side of a language pair may be source-only; "
                        f"both sides of {list(pair)!r} are listed in "
                        "data.source_only_languages"
                    )
        directions = self.data.configured_translation_directions()
        if self.data.translation_directions:
            self.data.translation_directions = [list(direction) for direction in directions]
        revision_directions = self.data.configured_revision_directions()
        if self.data.revision_directions:
            self.data.revision_directions = [list(direction) for direction in revision_directions]
            self.data.revision_examples = True

    def validate_training_supervision(
        self,
        *,
        alignment_targets_available: bool,
    ) -> None:
        """Validate losses whose labels are supplied by the training pipeline.

        ``ModelConfig`` deliberately permits a positive BATS alignment weight:
        research callers can pass ``alignment_targets`` directly to the model.
        The built-in collator does not create those dense alignment matrices,
        however, so the standard CLI must reject that configuration instead of
        silently optimizing a permanently-zero alignment loss.
        """

        experimental = self.model.experimental
        if (
            experimental.bats_enabled
            and experimental.bats_loss_weight > 0
            and not alignment_targets_available
        ):
            raise ValueError(
                "experimental.bats_loss_weight is positive, but this training pipeline "
                "does not provide alignment_targets. Set bats_loss_weight=0, add an "
                "alignment-aware collator/training caller, or explicitly validate the "
                "custom pipeline with alignment_targets_available=True."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_dataclass(cls: type, values: dict[str, Any] | None):
    values = dict(values or {})
    return cls(**values)


_CONFIG_TOP_LEVEL_KEYS = frozenset({"model", "data", "foundation", "training", "posttraining"})


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys instead of keeping the last one."""

    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """Read a config file and reject ambiguous or misspelled top-level keys."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=_UniqueKeySafeLoader) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    unknown = set(raw) - _CONFIG_TOP_LEVEL_KEYS
    if unknown:
        rendered = ", ".join(sorted(repr(key) for key in unknown))
        expected = ", ".join(sorted(_CONFIG_TOP_LEVEL_KEYS))
        raise ValueError(f"unknown top-level config key(s): {rendered}; expected only: {expected}")
    return raw


def config_from_raw(raw: dict[str, Any]) -> AppConfig:
    """raw dict 에서 AppConfig 를 만듭니다. 빠진 키는 dataclass 기본값을 씁니다."""
    model_values = dict(raw.get("model") or {})
    data_values = dict(raw.get("data") or {})
    if "language_pair" in data_values and "language_pairs" in data_values:
        raise ValueError("data.language_pair and data.language_pairs cannot both be set")
    training_values = dict(raw.get("training") or {})
    if "fsdp2" in training_values and "parallel_strategy" in training_values:
        raise ValueError("training.fsdp2 and training.parallel_strategy cannot both be set")
    experimental = _construct_dataclass(ExperimentalConfig, model_values.pop("experimental", {}))
    model = ModelConfig(experimental=experimental, **model_values)
    return AppConfig(
        model=model,
        data=_construct_dataclass(DataConfig, data_values),
        foundation=_construct_dataclass(FoundationConfig, raw.get("foundation")),
        training=_construct_dataclass(TrainingConfig, training_values),
        posttraining=_construct_dataclass(PostTrainingConfig, raw.get("posttraining")),
    )


def load_config(path: str | Path) -> AppConfig:
    config = config_from_raw(load_raw_config(path))
    config.validate()
    return config
