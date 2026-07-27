from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
    # 공유 블록 반복(latent reasoning): 인코더 마지막 N개 층을 같은 가중치로
    # 여러 번 통과시켜, 파라미터를 늘리지 않고 깊이만 늘립니다. 명시적 사고
    # 토큰 없이 hidden state 안에서 추가 계산을 하게 하는 구조입니다.
    # 0 이면 꺼짐 — 기본이 꺼져 있어야 기존 체크포인트가 그대로 동작합니다.
    recurrent_block_layers: int = 0
    recurrent_steps: int = 1

    def validate(self) -> None:
        if self.recurrent_block_layers < 0:
            raise ValueError("experimental.recurrent_block_layers must be non-negative")
        if self.recurrent_steps < 1:
            raise ValueError("experimental.recurrent_steps must be at least 1")
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
        ):
            if value <= 0:
                raise ValueError(f"experimental.{name} must be positive")
        for name, value in (
            ("bats_loss_weight", self.bats_loss_weight),
            ("bats_coverage_weight", self.bats_coverage_weight),
            ("register_loss_weight", self.register_loss_weight),
        ):
            if value < 0:
                raise ValueError(f"experimental.{name} must be non-negative")

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
    language_pair: list[str] = field(default_factory=lambda: ["ko", "ja"])
    # 여러 언어쌍을 한 모델에서 학습할 때 사용합니다. 비어 있으면 위의
    # language_pair 한 쌍만 사용합니다. YAML에서는 둘 중 하나만 적습니다.
    language_pairs: list[list[str]] = field(default_factory=list)
    # 원문(입력) 쪽 토큰을 낮은 확률로 무작위 탈락시키는 온라인 증강.
    # 소량(0.05 안팎)은 과적합을 줄여 주지만, 큰 값은 오히려 해롭습니다.
    # 검증에는 절대 적용되지 않습니다. 0 이면 끕니다.
    source_token_dropout: float = 0.05
    # 역번역(backtranslation) 등 합성 데이터 파일의 이름 접두사.
    # 이 접두사로 시작하는 파일은 ① 항상 train split 에만 들어가고
    # ② 샘플링 가중치가 자동으로 낮아집니다 (합성 데이터 과다 방지).
    synthetic_prefix: str = "bt_"
    # 글로서리(용어집) JSON 경로. 지정하면 sion-translate/evaluate 가 이 파일을
    # 기본으로 불러와 지정한 용어를 정해진 대응어로 강제합니다. 빈 문자열이면 끔.
    glossary: str = ""
    tokenizer_model: str = "artifacts/tokenizer/sion.model"
    tokenizer_features: str = "artifacts/tokenizer/token_features.npz"
    dataset_dir: str = "artifacts/dataset"
    train_split: str = "train"
    validation_split: str = "validation"
    bidirectional: bool = True
    max_source_length: int = 512
    max_target_length: int = 512
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
        raw_pairs = self.language_pairs or [self.language_pair]
        return tuple((str(pair[0]), str(pair[1])) for pair in raw_pairs)

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
    output_dir: str = "runs/auto"
    seed: int = 20260710
    max_steps: int = 1000
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
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    # 가중치 지수이동평균(EMA). 매 step 뒤 shadow 가중치를
    # shadow = decay*shadow + (1-decay)*param 으로 갱신합니다.
    # 번역 모델에서 검증 loss/BLEU 를 안정적으로 개선하는 검증된 기법입니다.
    # 0.0 이면 비활성화됩니다.
    ema_decay: float = 0.999
    tensorboard: bool = True
    tensorboard_dir: str | None = None
    resume_from: str | None = None


@dataclass
class PostTrainingConfig:
    """SFT 뒤 실행하는 복합 최소위험 + 다중 후보 선호학습 설정."""

    enabled: bool = True
    method: str = "mrt"
    max_steps: int = 5_000
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
    # best/early stopping은 이 beam 수로 생성한 번역의 복합 보상을 사용합니다.
    validation_num_beams: int = 4
    eval_batch_size_per_gpu: int = 1
    eval_every: int = 250
    save_every: int = 1_000
    early_stopping_patience: int = 5


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    posttraining: PostTrainingConfig = field(default_factory=PostTrainingConfig)

    def validate(self) -> None:
        self.model.validate()
        pairs = self.data.language_pairs or [self.data.language_pair]
        if not pairs:
            raise ValueError("at least one language pair is required")
        seen_edges: set[frozenset[str]] = set()
        for pair in pairs:
            if (
                len(pair) != 2
                or pair[0] == pair[1]
                or any(
                    not lang or not lang.isascii() or not lang.isalnum() or not lang[0].isalpha()
                    for lang in pair
                )
            ):
                raise ValueError(
                    "each language pair must contain two distinct ASCII language "
                    "keys, e.g. ['ko', 'ja']"
                )
            edge = frozenset(pair)
            if edge in seen_edges:
                raise ValueError(f"duplicate or reversed language pair: {pair!r}")
            seen_edges.add(edge)
        if not 0.0 <= self.data.source_token_dropout < 0.5:
            raise ValueError("source_token_dropout must be in [0, 0.5)")
        if not self.data.synthetic_prefix:
            raise ValueError("synthetic_prefix must be non-empty")
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
        if self.training.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.training.batch_size_per_gpu <= 0:
            raise ValueError("batch_size_per_gpu must be positive")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.training.min_learning_rate_ratio <= 1.0:
            raise ValueError("min_learning_rate_ratio must be in [0, 1]")
        if self.training.warmup_steps < 0 or self.training.warmup_steps > self.training.max_steps:
            raise ValueError("warmup_steps must be between 0 and max_steps")
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
        if self.training.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if not 0.0 <= self.training.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        post = self.posttraining
        if post.method != "mrt":
            raise ValueError("posttraining.method must be 'mrt'")
        for name, value in (
            ("max_steps", post.max_steps),
            ("batch_size_per_gpu", post.batch_size_per_gpu),
            ("gradient_accumulation_steps", post.gradient_accumulation_steps),
            ("samples_per_source", post.samples_per_source),
            ("candidate_micro_batch", post.candidate_micro_batch),
            ("max_new_tokens", post.max_new_tokens),
            ("eval_batch_size_per_gpu", post.eval_batch_size_per_gpu),
            ("eval_every", post.eval_every),
            ("save_every", post.save_every),
            ("validation_num_beams", post.validation_num_beams),
        ):
            if value <= 0:
                raise ValueError(f"posttraining.{name} must be positive")
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
        if sum(reward_weights.values()) <= 0:
            raise ValueError("at least one posttraining reward weight must be positive")
        if not 0.0 <= post.reward_repetition_penalty <= 1.0:
            raise ValueError("posttraining.reward_repetition_penalty must be in [0, 1]")
        if not 0.0 <= post.reward_copy_penalty <= 1.0:
            raise ValueError("posttraining.reward_copy_penalty must be in [0, 1]")
        if post.top_k < 0:
            raise ValueError("posttraining.top_k must be non-negative")
        if post.warmup_steps < 0 or post.warmup_steps > post.max_steps:
            raise ValueError("posttraining.warmup_steps must be between 0 and max_steps")
        if post.early_stopping_patience < 0:
            raise ValueError("posttraining.early_stopping_patience must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_dataclass(cls: type, values: dict[str, Any] | None):
    values = dict(values or {})
    return cls(**values)


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """YAML 파일을 dict 그대로 읽습니다 (없는 키 = 자동 결정 대상)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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
        training=_construct_dataclass(TrainingConfig, training_values),
        posttraining=_construct_dataclass(PostTrainingConfig, raw.get("posttraining")),
    )


def load_config(path: str | Path) -> AppConfig:
    config = config_from_raw(load_raw_config(path))
    config.validate()
    return config
