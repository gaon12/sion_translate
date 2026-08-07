"""자동 설정(auto-configuration).

목표: 인자 없이 ``sion-train`` 만 실행하면
  ① 실행 환경(GPU 수·VRAM·bf16 지원·CPU 코어)을 스스로 인식하고,
  ② ``data/`` 의 원천 JSONL 을 스스로 찾아(추가·변경도 감지)
     토크나이저 학습과 데이터 준비까지 자동으로 수행하며,
  ③ 데이터 규모에 맞춰 모델 크기·step 수·배치 등 수치를 자동으로 정하도록
합니다.

사용자가 프로젝트 루트의 ``sion_translate.yaml`` 에 적은 값은 항상 자동값보다
우선합니다. 즉 sion_translate.yaml 은 '바꾸고 싶은 것만 적는 얇은 override 파일'입니다.
"""

# CUDA device properties and YAML payloads are dynamically typed boundaries.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

import json
import math
import os
import platform as platform_module
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from sion_translate.config import AppConfig
from sion_translate.fingerprint import (
    PREPROCESSING_SCHEMA,
    DatasetFingerprint,
    build_dataset_fingerprint,
)
from sion_translate.performance import available_cpu_count

# ──────────────────────────────────────────────────────────────────────────
# ① 실행 환경 인식
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EnvironmentInfo:
    """학습 수치를 정하는 데 필요한 하드웨어/실행 환경 요약."""

    cuda: bool  # CUDA GPU 사용 가능 여부
    world_size: int  # torchrun 이 띄운 프로세스 수 (단일 실행이면 1)
    device_count: int  # 이 머신에서 보이는 GPU 수
    device_name: str  # 대표 GPU 이름 (없으면 "CPU")
    min_vram_gib: float  # 가장 작은 GPU 의 메모리(GiB) — 배치 크기 기준
    bf16: bool  # bfloat16 연산 지원 여부
    cpu_count: int  # 논리 CPU 코어 수
    os_name: str  # "Windows" / "Linux" / "Darwin"


def _all_devices_support_native_bf16(properties: Sequence[Any]) -> bool:
    """Use BF16 only when every visible accelerator supports it natively."""

    if not properties:
        return False
    if torch.version.hip is not None:
        return True
    return all(int(getattr(device, "major", 0)) >= 8 for device in properties)


def probe_environment() -> EnvironmentInfo:
    """현재 머신의 하드웨어를 조사합니다."""
    cuda = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if cuda:
        properties = [torch.cuda.get_device_properties(index) for index in range(device_count)]
        min_vram_gib = min(p.total_memory for p in properties) / (1024**3)
        device_names = tuple(dict.fromkeys(str(device.name) for device in properties))
        device_name = (
            device_names[0] if len(device_names) == 1 else "mixed: " + " / ".join(device_names)
        )
        bf16 = _all_devices_support_native_bf16(properties)
    else:
        min_vram_gib = 0.0
        device_name = "CPU"
        bf16 = False
    return EnvironmentInfo(
        cuda=cuda,
        world_size=world_size,
        device_count=device_count,
        device_name=device_name,
        min_vram_gib=min_vram_gib,
        bf16=bf16,
        cpu_count=available_cpu_count(),
        os_name=platform_module.system(),
    )


def synchronize_environment(
    env: EnvironmentInfo,
    context: Any,
) -> EnvironmentInfo:
    """Use the least-capable rank for settings shared by a distributed job."""

    if not context.distributed or not env.cuda:
        return env
    minimums = torch.tensor(
        [env.min_vram_gib, float(env.bf16)],
        device=context.device,
        dtype=torch.float64,
    )
    torch.distributed.all_reduce(minimums, op=torch.distributed.ReduceOp.MIN)
    return replace(
        env,
        min_vram_gib=float(minimums[0].item()),
        bf16=bool(minimums[1].item()),
    )


def describe_environment(env: EnvironmentInfo) -> str:
    """사람이 읽기 좋은 한 줄 요약."""
    if env.cuda:
        return (
            f"GPU {env.device_count}개 ({env.device_name}, "
            f"최소 {env.min_vram_gib:.0f}GiB), 프로세스 {env.world_size}개, "
            f"bf16 {'지원' if env.bf16 else '미지원'}, CPU {env.cpu_count}코어"
        )
    return f"GPU 없음 (CPU {env.cpu_count}코어, {env.os_name})"


# ──────────────────────────────────────────────────────────────────────────
# ② 원천 데이터 인식 (추가/변경 감지)
# ──────────────────────────────────────────────────────────────────────────

FINGERPRINT_FILENAME = "raw_fingerprint.json"


def scan_raw_data(
    data_dir: str | Path,
    *,
    language_pairs: Sequence[Sequence[str]] = (),
    tokenizer_model: str | Path | None = None,
    preprocessing_schema: str = PREPROCESSING_SCHEMA,
    preprocessing_options: Mapping[str, Any] | None = None,
) -> DatasetFingerprint:
    """Build a content-addressed fingerprint for every raw JSONL input.

    The return value still behaves as ``Mapping[str, int]`` for legacy callers:
    iteration yields file names and values are byte sizes. Equality additionally
    covers file SHA-256, language pairs, tokenizer SHA-256, preprocessing schema,
    and normalized preprocessing options.
    """
    data_dir = Path(data_dir)
    return build_dataset_fingerprint(
        sorted(data_dir.glob("*.jsonl")),
        language_pairs=language_pairs,
        tokenizer_model=tokenizer_model,
        preprocessing_schema=preprocessing_schema,
        preprocessing_options=preprocessing_options,
    )


def stored_fingerprint(
    dataset_dir: str | Path,
) -> DatasetFingerprint | dict[str, int] | None:
    """이전 준비 때 기록해 둔 데이터 지문을 읽습니다 (없으면 None)."""
    path = Path(dataset_dir) / FINGERPRINT_FILENAME
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict) and value.get("schema"):
        try:
            return DatasetFingerprint.from_dict(value)
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, dict):
        # v1 fingerprints only tracked byte sizes. Returning the legacy mapping
        # makes it compare unequal to a v2 DatasetFingerprint and forces one
        # safe rebuild.
        try:
            return {str(key): int(size) for key, size in value.items()}
        except (TypeError, ValueError):
            return None
    return None


def write_fingerprint(
    dataset_dir: str | Path,
    fingerprint: DatasetFingerprint | Mapping[str, int],
) -> None:
    """준비가 끝난 데이터셋 디렉터리에 지문을 기록합니다."""
    payload = (
        fingerprint.to_dict() if isinstance(fingerprint, DatasetFingerprint) else dict(fingerprint)
    )
    path = Path(dataset_dir) / FINGERPRINT_FILENAME
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def backup_stale_dataset(dataset_dir: str | Path) -> Path:
    """원천 데이터가 바뀌어 못 쓰게 된 기존 데이터셋을 옆으로 치워 둡니다.

    삭제하지 않고 이름을 바꿔 보관하므로, 필요하면 수동으로 되돌릴 수 있습니다.
    """
    dataset_dir = Path(dataset_dir)
    backup = dataset_dir.with_name(f"{dataset_dir.name}.stale-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.move(str(dataset_dir), str(backup))
    return backup


def estimate_pair_count(files: Mapping[str, int], data_dir: str | Path) -> int:
    """원천 JSONL 의 행 수(≒ 번역쌍 수)를 빠르게 셉니다.

    토크나이저 vocab 크기를 정할 때 한 번만 사용합니다. 줄바꿈 문자를
    큰 버퍼 단위로 세므로 수 GiB 도 수십 초 안에 끝납니다.
    """
    data_dir = Path(data_dir)
    total = 0
    for name in files:
        with (data_dir / name).open("rb") as handle:
            while chunk := handle.read(1 << 22):  # 4 MiB 씩
                total += chunk.count(b"\n")
    return total


# ──────────────────────────────────────────────────────────────────────────
# ③ 데이터 규모·환경 기반 자동 수치 결정
# ──────────────────────────────────────────────────────────────────────────

# 데이터 양에 맞는 모델 크기 프리셋.
# 데이터가 적은데 모델이 크면 과적합, 반대면 과소적합이 되므로
# '번역쌍 수' 구간별로 적정 크기를 고릅니다.
MODEL_PRESETS: list[tuple[int, str, dict[str, int]]] = [
    # (이 값 미만의 번역쌍 수, 프리셋 이름, 모델 설정)
    (
        200_000,
        "small(~60M)",
        dict(
            d_model=512,
            encoder_layers=8,
            decoder_layers=4,
            num_heads=8,
            num_kv_heads=2,
            d_ff=1536,
        ),
    ),
    (
        3_000_000,
        "medium(~120M)",
        dict(
            d_model=640,
            encoder_layers=12,
            decoder_layers=6,
            num_heads=10,
            num_kv_heads=2,
            d_ff=1792,
        ),
    ),
    (
        30_000_000,
        "base(~200M)",
        dict(
            d_model=768,
            encoder_layers=16,
            decoder_layers=8,
            num_heads=12,
            num_kv_heads=4,
            d_ff=2048,
        ),
    ),
    (
        100_000_000,
        "large(~450M)",
        dict(
            d_model=1024,
            encoder_layers=20,
            decoder_layers=10,
            num_heads=16,
            num_kv_heads=4,
            d_ff=2816,
        ),
    ),
    (
        10**12,
        "xlarge(~900M)",
        dict(
            d_model=1280,
            encoder_layers=24,
            decoder_layers=12,
            num_heads=20,
            num_kv_heads=4,
            d_ff=3584,
        ),
    ),
]

# update 당 목표 시퀀스 수 (배치 × GPU 수 × accumulation)
TARGET_EFFECTIVE_BATCH = 256


def target_epochs(pair_count: int) -> float:
    """corpus 통과 목표 횟수. 데이터가 커질수록 적은 epoch 으로 충분하므로
    (매 step 새 데이터를 보게 됨) step 예산이 데이터에 선형으로 폭주하지
    않도록 낮춥니다. early stopping 이 있어 넉넉해도 낭비되지 않습니다."""
    if pair_count < 500_000:
        return 8.0
    if pair_count < 5_000_000:
        return 5.0
    if pair_count < 30_000_000:
        return 3.0
    if pair_count < 100_000_000:
        return 2.0
    return 1.2


def pick_vocab_size(pair_estimate: int) -> int:
    """corpus 크기에 맞는 SentencePiece vocab 크기."""
    if pair_estimate < 200_000:
        return 16_000
    if pair_estimate < 3_000_000:
        return 32_000
    if pair_estimate < 100_000_000:
        return 48_000
    return 64_000


def pick_model_preset(pair_count: int) -> tuple[str, dict[str, int]]:
    for threshold, name, preset in MODEL_PRESETS:
        if pair_count < threshold:
            return name, preset
    raise AssertionError("unreachable")


def pick_batch_size(env: EnvironmentInfo, d_model: int) -> int:
    """GPU 메모리(GiB)와 모델 폭으로 GPU당 배치 크기를 고릅니다.

    max_seq_len 512, gradient checkpointing 활성 기준의 보수적인 값입니다.
    OOM 이 나면 sion_translate.yaml 의 training.batch_size_per_gpu 로 낮추면 됩니다.
    """
    if not env.cuda:
        return 2  # CPU 는 스모크 테스트 용도
    vram = env.min_vram_gib
    if vram >= 70:
        # 80 GiB-class cards run the baseline without checkpointing by default.
        # Keep headroom for rare 512-token buckets instead of selecting 64 from
        # short-sentence averages and failing late in the run.
        base = 32
    elif vram >= 40:
        base = 16
    elif vram >= 22:
        base = 8
    elif vram >= 14:
        base = 4
    elif vram >= 10:
        base = 2
    else:
        base = 1
    # base 프리셋(768)보다 큰 모델이면 배치를 줄입니다.
    if d_model > 1024:
        base = max(1, base // 4)
    elif d_model > 768:
        base = max(1, base // 2)
    return base


def pick_parallel_strategy(env: EnvironmentInfo, d_model: int) -> str:
    """Prefer lower-overhead DDP whenever one GPU has enough training memory."""

    if env.world_size <= 1:
        return "auto"
    if env.min_vram_gib >= 70 and d_model <= 1024:
        return "ddp"
    if env.min_vram_gib >= 40 and d_model <= 768:
        return "ddp"
    return "fsdp2"


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def apply_auto_settings(
    config: AppConfig,
    raw: dict[str, Any],
    env: EnvironmentInfo,
    *,
    train_examples: int,
    validation_examples: int,
    source_names: list[str] | None = None,
) -> list[str]:
    """사용자가 sion_translate.yaml 에 적지 않은 항목을 자동값으로 채웁니다.

    ``raw`` 는 sion_translate.yaml 을 그대로 읽은 dict 로, '어떤 키를 사용자가 직접
    적었는지'를 판단하는 데 씁니다 (적은 값은 절대 덮어쓰지 않습니다).
    반환값은 자동으로 결정된 항목의 사람이 읽을 요약 목록입니다.
    """
    raw_model = dict(raw.get("model") or {})
    raw_training = dict(raw.get("training") or {})
    raw_data = dict(raw.get("data") or {})
    decisions: list[str] = []

    def auto(section: dict[str, Any], key: str) -> bool:
        """해당 키를 사용자가 적지 않았으면 True (= 자동 결정 대상)."""
        return key not in section

    pair_count = train_examples // (2 if config.data.bidirectional else 1)

    # ── 모델 크기: 데이터 양 기준 프리셋 ────────────────────────────────
    size_keys = ("d_model", "encoder_layers", "decoder_layers", "num_heads", "num_kv_heads", "d_ff")
    if all(auto(raw_model, key) for key in size_keys):
        name, preset = pick_model_preset(pair_count)
        for key, value in preset.items():
            setattr(config.model, key, value)
        decisions.append(f"모델 크기: {name} — 학습쌍 {pair_count:,}개 기준")
    if auto(raw_model, "gradient_checkpointing"):
        config.model.gradient_checkpointing = env.cuda and env.min_vram_gib < 70
        if config.model.gradient_checkpointing:
            decisions.append("activation checkpointing: 활성 (70GiB 미만 GPU 메모리 보호)")

    # ── 정밀도: bf16 > fp16 > fp32 ──────────────────────────────────────
    if auto(raw_training, "precision"):
        config.training.precision = "bf16" if env.bf16 else ("fp16" if env.cuda else "fp32")
        decisions.append(f"정밀도: {config.training.precision}")

    # ── 배치/accumulation: VRAM 과 목표 effective batch 기준 ───────────
    if auto(raw_training, "batch_size_per_gpu"):
        config.training.batch_size_per_gpu = pick_batch_size(env, config.model.d_model)
        basis = f"VRAM {env.min_vram_gib:.0f}GiB 기준" if env.cuda else "CPU 스모크 기준"
        decisions.append(f"GPU당 배치: {config.training.batch_size_per_gpu} ({basis})")
    if auto(raw_training, "gradient_accumulation_steps"):
        per_update = config.training.batch_size_per_gpu * env.world_size
        config.training.gradient_accumulation_steps = max(
            1, round(TARGET_EFFECTIVE_BATCH / per_update)
        )
        effective = per_update * config.training.gradient_accumulation_steps
        decisions.append(
            f"accumulation: {config.training.gradient_accumulation_steps} "
            f"(effective batch {effective} sequences/update)"
        )

    # ── step 예산: 데이터 규모별 목표 epoch 수만큼 corpus 통과 ─────────
    effective_batch = (
        config.training.batch_size_per_gpu
        * env.world_size
        * config.training.gradient_accumulation_steps
    )
    steps_per_epoch = max(1, train_examples // max(1, effective_batch))
    epochs = target_epochs(pair_count)
    if auto(raw_training, "max_steps"):
        config.training.max_steps = _clamp(int(epochs * steps_per_epoch), 200, 1_000_000)
        decisions.append(
            f"max_steps: {config.training.max_steps:,} "
            f"(epoch당 약 {steps_per_epoch:,} step × {epochs:g}회, early stopping 있음)"
        )
    if auto(raw_training, "warmup_steps"):
        config.training.warmup_steps = _clamp(int(0.025 * config.training.max_steps), 10, 4000)
        # warmup 은 max_steps 를 넘을 수 없습니다.
        config.training.warmup_steps = min(config.training.warmup_steps, config.training.max_steps)
        decisions.append(f"warmup_steps: {config.training.warmup_steps:,}")

    # ── 검증/저장 주기: epoch 길이에 비례 ───────────────────────────────
    if auto(raw_training, "eval_every"):
        config.training.eval_every = _clamp(steps_per_epoch // 8, 50, 2500)
        decisions.append(f"eval_every: {config.training.eval_every:,}")
    if auto(raw_training, "save_every"):
        config.training.save_every = config.training.eval_every * 2
        decisions.append(f"save_every: {config.training.save_every:,}")
    if auto(raw_training, "eval_batches"):
        per_rank = config.training.batch_size_per_gpu * env.world_size
        needed = math.ceil(validation_examples / max(1, per_rank))
        config.training.eval_batches = _clamp(needed, 8, 200)
        decisions.append(f"eval_batches: {config.training.eval_batches}")

    # ── 실행 방식 ───────────────────────────────────────────────────────
    # ``parallel_strategy: auto`` is an explicit request for the environment
    # picker, not a request to leave the generic DDP fallback unresolved.
    if config.training.parallel_strategy.lower() == "auto" and auto(raw_training, "fsdp2"):
        config.training.parallel_strategy = pick_parallel_strategy(
            env,
            config.model.d_model,
        )
        if env.world_size > 1:
            decisions.append(f"다중 GPU 병렬화: {config.training.parallel_strategy.upper()}")
    if auto(raw_training, "fsdp_reduce_dtype"):
        config.training.fsdp_reduce_dtype = "bf16" if env.bf16 else "fp32"
    if auto(raw_training, "reshard_after_forward"):
        # Keep FSDP2's memory-bounded default. Disabling resharding retains
        # full parameters after every forward and can erase sharding's VRAM
        # benefit even on an 80 GiB H100; users may still opt out explicitly.
        config.training.reshard_after_forward = True
    if auto(raw_training, "compile"):
        # Compiler/backend support varies across CUDA architectures and container
        # builds. Reliability-first automatic runs stay eager; measured profiles
        # can still opt in with ``training.compile: true``.
        config.training.compile = False
    if auto(raw_data, "num_workers"):
        per_rank = max(1, env.cpu_count // max(1, env.world_size))
        config.data.num_workers = min(16, max(0, per_rank - 1))
        decisions.append(f"DataLoader workers: {config.data.num_workers}")

    # ── 합성(역번역) 데이터 자동 다운웨이트 ────────────────────────────
    # 합성 데이터가 실데이터와 같은 비중으로 섞이면 모델이 자기 출력의
    # 오류 패턴을 학습할 수 있으므로, 사용자가 직접 가중치를 정하지 않은 한
    # 합성 출처의 샘플링 비중을 절반으로 낮춥니다.
    if auto(raw_data, "source_sampling_weights") and source_names:
        prefixes = config.data.configured_synthetic_prefixes()
        synthetic = [name for name in source_names if name.startswith(prefixes)]
        if synthetic:
            weight = config.data.synthetic_sampling_weight
            config.data.source_sampling_weights = {name: weight for name in synthetic}
            decisions.append(
                f"합성 데이터 가중치: {len(synthetic)}개 출처"
                f"({', '.join(prefixes)}*) × {weight:g} 다운웨이트"
            )

    return decisions
