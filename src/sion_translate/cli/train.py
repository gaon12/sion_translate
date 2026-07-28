"""sion_translate 학습 실행 진입점(CLI) — 인자 없이 실행하는 완전 자동 파이프라인.

    sion-train            ← 이것만 실행하면 됩니다.

동작 순서:
    ① 실행 환경(GPU 수·VRAM·bf16·CPU) 자동 인식
    ② 설정 로드 — 프로젝트 루트의 ``sion_translate.yaml`` 이 있으면 그 값을 우선 적용
       (없거나 비워 두면 전부 자동). ``--config`` 로 다른 파일도 지정 가능.
    ③ ``data/*.jsonl`` 자동 인식 — 토크나이저가 없으면 학습하고,
       파일이 추가/변경되었으면 데이터셋을 자동으로 다시 준비
    ④ 데이터 규모에 맞춰 모델 크기·step 수·배치 등 수치 자동 결정
    ⑤ 이전 학습이 있으면 단계별 checkpoints/latest 에서 자동 재개
    ⑥ SFT 사전학습 후 pretrain/에 저장
    ⑦ 복합 보상 MRT + 다중 후보 선호학습 후 posttrain/에 별도 저장

각 단계가 시작될 때마다 "[sion] …" 텍스트가 출력되므로 현재 어디까지
진행됐는지 터미널에서 바로 확인할 수 있습니다.
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sion_translate.auto import (
    apply_auto_settings,
    backup_stale_dataset,
    describe_environment,
    estimate_pair_count,
    pick_vocab_size,
    probe_environment,
    scan_raw_data,
    stored_fingerprint,
    write_fingerprint,
)
from sion_translate.config import AppConfig, config_from_raw, load_raw_config
from sion_translate.console import configure_stdio
from sion_translate.data import (
    DistributedBucketBatchSampler,
    IndexedParallelDataset,
    SionBatchCollator,
)
from sion_translate.fingerprint import DatasetFingerprint, file_sha256
from sion_translate.model import SionForConditionalGeneration
from sion_translate.tokenizer import (
    SionTokenizer,
    load_tokenizer_metadata,
    tokenizer_split_digits_policy,
    write_tokenizer_metadata,
)
from sion_translate.training.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    initialize_distributed,
    parallelize_model,
    resolve_parallel_strategy,
)
from sion_translate.training.export import export_inference_models
from sion_translate.training.objectives import MinimumRiskObjective
from sion_translate.training.trainer import announce, train
from sion_translate.performance import build_cpu_plan

DEFAULT_CONFIG_FILE = "sion_translate.yaml"


def scan_configured_raw_data(
    config: AppConfig,
    data_dir: Path,
    tokenizer_path: Path,
) -> DatasetFingerprint:
    """Fingerprint every input that can change the prepared dataset."""

    return scan_raw_data(
        data_dir,
        language_pairs=config.data.configured_language_pairs(),
        tokenizer_model=tokenizer_path,
        preprocessing_options={
            "synthetic_sampling_weight": config.data.synthetic_sampling_weight,
            "train_only_prefixes": list(config.data.configured_synthetic_prefixes()),
        },
    )


def dataloader_runtime_kwargs(
    num_workers: int,
    device: torch.device,
    *,
    training: bool,
) -> dict[str, Any]:
    """Build stage-specific loader settings without retaining idle worker pools."""

    workers = max(0, num_workers)
    options: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        options.update(
            {
                "persistent_workers": training,
                "prefetch_factor": 4 if training else 2,
            }
        )
    return options


def shutdown_dataloader(loader: DataLoader | None) -> None:
    """Stop a persistent DataLoader pool before constructing the next stage."""

    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if iterator is not None:
        loader._iterator = None


def release_stage_resources(
    context: DistributedContext,
    *loaders: DataLoader | None,
) -> dict[str, float]:
    """Release CPU workers and CUDA cache at a pretrain/posttrain boundary."""

    for loader in loaders:
        shutdown_dataloader(loader)
    gc.collect()
    if context.device.type != "cuda":
        return {}
    torch.cuda.synchronize(context.device)
    before_allocated = torch.cuda.memory_allocated(context.device) / 2**30
    before_reserved = torch.cuda.memory_reserved(context.device) / 2**30
    torch.cuda.empty_cache()
    after_allocated = torch.cuda.memory_allocated(context.device) / 2**30
    after_reserved = torch.cuda.memory_reserved(context.device) / 2**30
    torch.cuda.reset_peak_memory_stats(context.device)
    return {
        "before_allocated_gib": before_allocated,
        "before_reserved_gib": before_reserved,
        "after_allocated_gib": after_allocated,
        "after_reserved_gib": after_reserved,
    }


def requires_ddp_unused_parameter_detection(config: AppConfig) -> bool:
    """Return whether one DDP wrapper spans changing parameter-use graphs."""

    experimental = config.model.experimental
    if not experimental.bats_enabled:
        return False
    unused_during_sft = (
        experimental.bats_loss_weight == 0 and experimental.bats_coverage_weight == 0
    )
    # MRT scores candidates through label-free forwards, so BATS parameters
    # used by supervised losses become unused after the SFT stage. The same
    # DDP wrapper spans both stages and therefore cannot use a static graph.
    return unused_during_sft or config.posttraining.enabled


def validate_training_capacity(
    parameter_count: int,
    context: DistributedContext,
    *,
    parallel_strategy: str,
    ema_enabled: bool,
    per_gpu_vram_gib: float | None = None,
) -> dict[str, float | int] | None:
    """Fail before allocation when persistent training state consumes H100 headroom."""

    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    if per_gpu_vram_gib is None:
        if context.device.type != "cuda":
            return None
        per_gpu_vram_gib = torch.cuda.get_device_properties(context.device).total_memory / 2**30
    if per_gpu_vram_gib <= 0:
        raise ValueError("per_gpu_vram_gib must be positive")

    # FP32 master parameter + gradient + AdamW first/second moments = 16 B.
    # EMA adds another FP32 shard. Reserve just over half of VRAM for BF16 layer
    # all-gathers, activations, temporary kernels, and the CUDA context.
    bytes_per_parameter = 16 + (4 if ema_enabled else 0)
    sharding_factor = context.world_size if parallel_strategy == "fsdp2" else 1
    total_state_gib = parameter_count * bytes_per_parameter / 2**30
    per_rank_state_gib = total_state_gib / sharding_factor
    state_budget_gib = per_gpu_vram_gib * 0.49
    minimum_world_size = (
        math.ceil(total_state_gib / state_budget_gib)
        if parallel_strategy == "fsdp2"
        else context.world_size
    )
    report: dict[str, float | int] = {
        "bytes_per_parameter": bytes_per_parameter,
        "total_state_gib": total_state_gib,
        "per_rank_state_gib": per_rank_state_gib,
        "state_budget_gib": state_budget_gib,
        "minimum_world_size": minimum_world_size,
    }
    if per_rank_state_gib > state_budget_gib:
        strategy_hint = (
            f"Use at least {minimum_world_size} GPUs with FSDP2"
            if parallel_strategy == "fsdp2"
            else "Switch to FSDP2 or use a smaller model"
        )
        ema_hint = ", disable EMA (training.ema_decay=0)" if ema_enabled else ""
        raise RuntimeError(
            "Estimated persistent training state leaves insufficient accelerator "
            f"headroom: {per_rank_state_gib:.1f} GiB/rank versus a "
            f"{state_budget_gib:.1f} GiB safety budget on {per_gpu_vram_gib:.1f} GiB GPUs. "
            f"{strategy_hint}{ema_hint}, or use an explicitly validated lower-memory "
            "optimizer/offload policy."
        )
    return report


def construct_training_model(
    config: AppConfig,
    context: DistributedContext,
    *,
    pad_id: int,
    parallel_strategy: str,
) -> tuple[
    SionForConditionalGeneration,
    int,
    dict[str, float | int] | None,
    bool,
]:
    """Count and capacity-check CUDA models before allocating parameter storage."""

    materialize_meta = context.device.type == "cuda"
    construction_device: torch.device | str = "meta" if materialize_meta else context.device
    with torch.device(construction_device):
        model = SionForConditionalGeneration(config.model, pad_id=pad_id)
    parameter_count = model.parameter_count()
    capacity = validate_training_capacity(
        parameter_count,
        context,
        parallel_strategy=parallel_strategy,
        ema_enabled=config.training.ema_decay > 0,
    )
    return model, parameter_count, capacity, materialize_meta


def export_final_model(
    model: torch.nn.Module,
    config: AppConfig,
    context: DistributedContext,
    run_root: Path,
    *,
    stage: str,
    step: int,
) -> Path:
    """Create the required final format set from the restored best weights."""

    export_dir = run_root / stage / "exports" / "best"
    export_inference_models(
        export_dir,
        model,
        config.model,
        context,
        step=step,
        formats=tuple(config.training.final_export_formats),
        tokenizer_path=config.data.tokenizer_model,
        token_features_path=(
            config.data.tokenizer_features
            if config.model.experimental.morphoscript_enabled
            else None
        ),
        language_pairs=config.data.configured_language_pairs(),
        bidirectional=config.data.bidirectional,
        revision_trained=config.data.revision_examples,
        strict=True,
    )
    return export_dir


def find_existing_checkpoint(config: AppConfig) -> Path | None:
    """Find any checkpoint that constrains the tokenizer vocabulary identity."""

    if config.training.resume_from:
        explicit = Path(config.training.resume_from)
        if explicit.exists():
            return explicit
    run_root = Path(config.training.output_dir)
    for stage_root in (run_root, run_root / "pretrain", run_root / "posttrain"):
        checkpoint_root = stage_root / "checkpoints"
        if not checkpoint_root.is_dir():
            continue
        for candidate in sorted(checkpoint_root.iterdir()):
            if (candidate / "checkpoint.pt").is_file() or (candidate / ".metadata").exists():
                return candidate
    return None


def tokenizer_policy_problem(
    tokenizer_path: str | Path,
    language_pairs: tuple[tuple[str, str], ...],
) -> str | None:
    """Return a concrete compatibility problem for a tokenizer, if any."""

    tokenizer_path = Path(tokenizer_path)
    try:
        tokenizer = SionTokenizer(tokenizer_path)
        metadata = load_tokenizer_metadata(tokenizer_path)
        recorded_policy = tokenizer_split_digits_policy(tokenizer_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return f"토크나이저 정책/메타데이터를 읽을 수 없습니다: {exc}"

    if not tokenizer.splits_digits:
        return "토크나이저가 여러 자리 숫자를 한 자리씩 분리하지 않습니다 (split_digits=False 동작)"
    if recorded_policy is False:
        return "tokenizer_metadata.json에 split_digits=false가 기록되어 있습니다"
    if recorded_policy is None or metadata is None:
        return "버전 2 이상의 tokenizer_metadata.json이 없습니다"

    recorded_hash = metadata.get("model_sha256")
    if recorded_hash != file_sha256(tokenizer_path):
        return "tokenizer_metadata.json의 model_sha256이 실제 모델과 다릅니다"
    vocab_path = tokenizer_path.with_suffix(".vocab")
    if not vocab_path.is_file() or metadata.get("vocab_sha256") != file_sha256(vocab_path):
        return "tokenizer_metadata.json의 vocab_sha256이 실제 vocabulary와 다릅니다"
    raw_pairs = metadata.get("language_pairs")
    recorded_pairs = (
        tuple((str(pair[0]), str(pair[1])) for pair in raw_pairs)
        if isinstance(raw_pairs, list)
        and all(isinstance(pair, list) and len(pair) == 2 for pair in raw_pairs)
        else ()
    )
    if recorded_pairs != language_pairs:
        return (
            "tokenizer_metadata.json의 language_pairs가 현재 설정과 다릅니다 "
            f"(metadata={recorded_pairs}, config={language_pairs})"
        )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train sion_translate. 인자 없이 실행하면 환경/데이터를 자동 인식합니다."
    )
    parser.add_argument(
        "--config", help=f"설정 파일 (기본: 루트의 {DEFAULT_CONFIG_FILE}, 없으면 전부 자동)"
    )
    parser.add_argument("--max-steps", type=int, help="최대 step 수동 지정 (자동값 무시)")
    parser.add_argument("--posttrain-steps", type=int, help="MRT 사후학습 step 수동 지정")
    parser.add_argument("--skip-posttraining", action="store_true", help="SFT 사전학습까지만 실행")
    parser.add_argument(
        "--resume-from", help="재개할 체크포인트 수동 지정 (기본: latest 자동 감지)"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="토크나이저와 dataset shard만 준비하고 학습 전에 종료",
    )
    return parser


def seed_everything(seed: int, rank: int) -> None:
    """재현 가능한 학습을 위해 모든 난수 생성기에 시드를 고정합니다.

    rank 를 더해 주는 이유: 분산 학습에서 rank 마다 dropout 등
    실행 시점 난수가 서로 달라야 하기 때문입니다.
    """
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32))
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def resolve_config(args: argparse.Namespace) -> tuple[AppConfig, dict, str]:
    """설정 파일을 찾아 (config, raw dict, 출처 설명) 을 돌려줍니다.

    raw dict 는 '사용자가 어떤 키를 직접 적었는지'를 기억하는 용도입니다.
    자동 설정은 사용자가 적지 않은 키만 채웁니다.
    """
    if args.config:
        raw = load_raw_config(args.config)
        source = args.config
    elif Path(DEFAULT_CONFIG_FILE).exists():
        raw = load_raw_config(DEFAULT_CONFIG_FILE)
        source = DEFAULT_CONFIG_FILE
    else:
        raw = {}
        source = "내장 기본값 (전부 자동)"

    # 커맨드라인 인자는 파일보다 우선하며, '사용자 지정'으로 취급합니다.
    if args.max_steps is not None:
        raw.setdefault("training", {})["max_steps"] = args.max_steps
    if args.resume_from is not None:
        raw.setdefault("training", {})["resume_from"] = args.resume_from
    if args.posttrain_steps is not None:
        post = raw.setdefault("posttraining", {})
        post["max_steps"] = args.posttrain_steps
        post["warmup_steps"] = min(int(post.get("warmup_steps", 200)), args.posttrain_steps)
    if args.skip_posttraining:
        raw.setdefault("posttraining", {})["enabled"] = False
    return config_from_raw(raw), raw, source


def ensure_artifacts(config: AppConfig, context: DistributedContext) -> None:
    """토크나이저와 준비된 데이터셋이 없거나 낡았으면 자동으로 만듭니다.

    - 토크나이저: 없을 때만 학습합니다. (다시 학습하면 vocab 이 바뀌어
      기존 체크포인트와 호환되지 않으므로, 데이터가 바뀌어도 유지합니다.)
    - 데이터셋: ``data/`` 의 파일 이름+크기 지문을 기록해 두고, 지문이
      달라지면(파일 추가/변경) 기존 데이터셋을 옆으로 보관한 뒤 다시 만듭니다.

    무거운 작업이므로 rank 0 만 수행하고 나머지 rank 는 barrier 에서
    기다립니다. 다중 GPU 로 처음 실행하기 전에 단일 프로세스로 한 번
    실행해 준비를 끝내 두는 편이 통신 타임아웃 걱정이 없습니다.
    """
    if context.is_main:
        data_dir = Path(config.data.raw_dir)
        tokenizer_path = Path(config.data.tokenizer_model)
        dataset_dir = Path(config.data.dataset_dir)
        files = scan_configured_raw_data(config, data_dir, tokenizer_path)
        dataset_ready = (dataset_dir / "manifest.json").exists()
        existing_checkpoint = find_existing_checkpoint(config)

        if not files and not dataset_ready:
            raise FileNotFoundError(
                f"원천 데이터({data_dir}/*.jsonl)도 준비된 데이터셋({dataset_dir})도 없습니다."
            )
        if not tokenizer_path.is_file() and dataset_ready and not files:
            raise FileNotFoundError(
                f"준비된 데이터셋은 있지만 대응하는 토크나이저가 없습니다: {tokenizer_path}. "
                "원천 데이터와 새 출력 경로를 지정해 새 run을 시작하세요."
            )

        if files:
            cpu_plan = build_cpu_plan(input_files=len(files))
            announce(
                f"원천 데이터 인식: {len(files)}개 파일, "
                f"총 {sum(files.values()) / 2**30:.2f} GiB ({data_dir}/)",
                context,
            )
            announce(
                f"CPU 자동 배분: 할당 {cpu_plan.available}개 → "
                f"입력 정제 {cpu_plan.preprocess_workers}개 + "
                f"SentencePiece {cpu_plan.sentencepiece_threads}개; "
                f"dataset 준비 {cpu_plan.dataset_workers}개",
                context,
            )
            # ── 토크나이저 ────────────────────────────────────────────
            if not tokenizer_path.exists():
                if existing_checkpoint is not None:
                    raise RuntimeError(
                        "기존 체크포인트가 있지만 대응하는 토크나이저가 없습니다. "
                        f"checkpoint={existing_checkpoint}. 기존 vocabulary를 추측해 "
                        "새 토크나이저로 덮어쓸 수 없습니다. tokenizer_model, dataset_dir, "
                        "training.output_dir을 새 경로로 지정해 새 run을 시작하세요."
                    )
                from sion_translate.tokenizer import train_tokenizer

                pair_estimate = estimate_pair_count(files, data_dir)
                vocab_size = pick_vocab_size(pair_estimate)
                announce(
                    f"토크나이저가 없어 새로 학습합니다 "
                    f"(약 {pair_estimate:,}행 → vocab {vocab_size:,}) — 시간이 걸립니다.",
                    context,
                )
                train_tokenizer(
                    [str(data_dir / "*.jsonl")],
                    tokenizer_path.parent,
                    vocab_size=vocab_size,
                    language_pairs=config.data.configured_language_pairs(),
                    num_workers=cpu_plan.preprocess_workers,
                    num_threads=cpu_plan.sentencepiece_threads,
                )
                announce("토크나이저 학습 완료.", context)
                # 토크나이저 파일의 SHA-256도 데이터셋 지문에 포함됩니다.
                files = scan_configured_raw_data(config, data_dir, tokenizer_path)

            # ── 데이터셋 (지문 기반 변경 감지) ─────────────────────────
            policy_problem = tokenizer_policy_problem(
                tokenizer_path,
                config.data.configured_language_pairs(),
            )
            if policy_problem is not None:
                existing_tokenizer = SionTokenizer(tokenizer_path)
                if (
                    existing_checkpoint is None
                    and existing_tokenizer.splits_digits
                    and load_tokenizer_metadata(tokenizer_path) is None
                ):
                    write_tokenizer_metadata(
                        tokenizer_path,
                        split_digits=True,
                        language_pairs=config.data.configured_language_pairs(),
                    )
                    files = scan_configured_raw_data(config, data_dir, tokenizer_path)
                    policy_problem = tokenizer_policy_problem(
                        tokenizer_path,
                        config.data.configured_language_pairs(),
                    )
                if policy_problem is not None:
                    checkpoint_detail = (
                        f" 기존 checkpoint={existing_checkpoint}와 vocabulary 호환성을 "
                        "깨뜨리는 자동 재학습은 수행하지 않습니다."
                        if existing_checkpoint is not None
                        else ""
                    )
                    raise RuntimeError(
                        f"{policy_problem}.{checkpoint_detail} tokenizer_model, dataset_dir, "
                        "training.output_dir을 새 경로로 지정하고 split_digits=True로 "
                        "토크나이저부터 재학습하는 새 run을 시작하세요."
                    )
            existing_tokenizer = SionTokenizer(tokenizer_path)
            if set(existing_tokenizer.languages) != set(config.data.languages):
                raise RuntimeError(
                    "기존 토크나이저의 언어 태그가 현재 data.language_pairs와 "
                    "다릅니다. 기존 체크포인트와 vocab 호환성을 확인한 뒤 "
                    "tokenizer_model과 dataset_dir을 새 경로로 지정해 재학습하세요. "
                    f"tokenizer={sorted(existing_tokenizer.languages)}, "
                    f"config={sorted(config.data.languages)}"
                )
            stored = stored_fingerprint(dataset_dir) if dataset_ready else None
            if not dataset_ready or stored != files:
                from sion_translate.data.prepare import prepare_dataset

                if dataset_ready:
                    backup = backup_stale_dataset(dataset_dir)
                    reason = (
                        "호환 가능한 지문 없음" if stored is None else "원천/토크나이저/전처리 변경"
                    )
                    announce(
                        f"{reason} 감지 → 기존 데이터셋을 {backup.name}/ 으로 보관합니다.",
                        context,
                    )
                announce(
                    "데이터셋 준비 시작 (품질 필터 + 중복 제거 + 토큰화) — 시간이 걸립니다.",
                    context,
                )
                prepare_dataset(
                    [str(data_dir / "*.jsonl")],
                    tokenizer_path,
                    dataset_dir,
                    language_pairs=config.data.configured_language_pairs(),
                    train_only_prefixes=config.data.configured_synthetic_prefixes(),
                    synthetic_sampling_weight=config.data.synthetic_sampling_weight,
                    num_workers=cpu_plan.dataset_workers,
                )
                files = scan_configured_raw_data(config, data_dir, tokenizer_path)
                write_fingerprint(dataset_dir, files)
                announce("데이터셋 준비 완료.", context)
            else:
                announce("데이터셋 최신 상태 확인 (원천 데이터 변경 없음).", context)
    # 준비가 끝날 때까지 다른 rank 들이 기다립니다.
    barrier(context)


def find_auto_resume(config: AppConfig) -> str | None:
    """이전 학습의 latest 체크포인트가 있으면 그 경로를 돌려줍니다."""
    latest = Path(config.training.output_dir) / "checkpoints" / "latest"
    if (latest / "checkpoint.pt").exists() or (latest / ".metadata").exists():
        return str(latest)
    return None


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    context = initialize_distributed()
    try:
        # ── 단계 ①: 환경 자동 인식 ──────────────────────────────────────
        env = probe_environment()
        announce(f"준비 ①: 실행 환경 — {describe_environment(env)}", context)

        # ── 단계 ②: 설정 로드 ───────────────────────────────────────────
        config, raw, source = resolve_config(args)
        announce(f"준비 ②: 설정 로드 — {source}", context)

        # ── 단계 ③: 원천 데이터 인식 + 토크나이저/데이터셋 자동 준비 ──
        announce("준비 ③: 원천 데이터를 확인합니다.", context)
        ensure_artifacts(config, context)
        if args.prepare_only:
            announce("전처리 전용 실행 완료.", context)
            return

        # 모델 파라미터 초기화는 world size 와 무관하게 같은 시드(rank 0 기준)로
        # 수행합니다. 실행 시점 난수는 모델 생성 후에 rank 별로 다시 시드합니다.
        seed_everything(config.training.seed, 0)
        tokenizer = SionTokenizer(config.data.tokenizer_model)
        config.model.vocab_size = len(tokenizer)

        train_dataset = IndexedParallelDataset(
            config.data.dataset_dir,
            config.data.train_split,
            bidirectional=config.data.bidirectional,
        )
        validation_dataset = IndexedParallelDataset(
            config.data.dataset_dir,
            config.data.validation_split,
            bidirectional=config.data.bidirectional,
        )
        revision_sources = [
            source
            for source in train_dataset.source_names
            if Path(source).name.startswith("revise_")
        ]
        if revision_sources and not config.data.revision_examples:
            config.data.revision_examples = True
            announce(
                "revision 예제 원천을 자동 감지했습니다: "
                + ", ".join(revision_sources[:3])
                + (" …" if len(revision_sources) > 3 else ""),
                context,
            )
        announce(
            f"데이터 규모: 학습 {len(train_dataset):,}개 / 검증 {len(validation_dataset):,}개 "
            f"(양방향 포함)",
            context,
        )

        # ── 단계 ④: 데이터 규모·환경 기반 자동 수치 결정 ────────────────
        decisions = apply_auto_settings(
            config,
            raw,
            env,
            train_examples=len(train_dataset),
            validation_examples=len(validation_dataset),
            source_names=train_dataset.source_names,
        )
        if decisions:
            announce("준비 ④: 자동 결정된 설정 —", context)
            for line in decisions:
                announce(f"  · {line}", context)
        config.validate()

        # 실행 루트 아래에 사전학습/사후학습 산출물을 서로 분리합니다.
        run_root = Path(config.training.output_dir)
        pretrain_config = copy.deepcopy(config)
        pretrain_config.training.output_dir = str(run_root / "pretrain")

        # ── 단계 ⑤: 이전 사전학습 자동 재개 ────────────────────────────
        if not pretrain_config.training.resume_from:
            resume = find_auto_resume(pretrain_config)
            if resume:
                pretrain_config.training.resume_from = resume
                announce(f"준비 ⑤: 이전 사전학습 발견 → {resume} 에서 자동 재개합니다.", context)

        # ── DataLoader 구성 ──────────────────────────────────────────────
        # collator: 원문/번역문을 토큰화하고 패딩해 텐서 배치로 만듭니다.
        collator_args = dict(
            tokenizer=tokenizer,
            max_source_length=config.data.max_source_length,
            max_target_length=config.data.max_target_length,
            pad_to_multiple_of=config.data.pad_to_multiple_of,
            denoise_noise_density=config.data.denoise_noise_density,
            denoise_mean_span=config.data.denoise_mean_span,
            augmentation_seed=config.training.seed,
            token_features=config.data.tokenizer_features,
        )
        train_collator = SionBatchCollator(
            **collator_args,
            denoise_probability=config.data.denoise_probability,
            # 온라인 증강(원문 토큰 dropout)은 학습에만 적용합니다.
            source_token_dropout=config.data.source_token_dropout,
        )
        validation_collator = SionBatchCollator(
            **collator_args,
            denoise_probability=config.data.validation_denoise_probability,
            source_token_dropout=0.0,  # 검증은 항상 깨끗한 입력으로
        )
        # sampler: 비슷한 길이끼리 묶어(bucket) 패딩 낭비를 줄이고,
        # 분산 학습에서 rank 별로 겹치지 않게 배치를 나눕니다.
        train_sampler = DistributedBucketBatchSampler(
            train_dataset,
            config.training.batch_size_per_gpu,
            rank=context.rank,
            world_size=context.world_size,
            bucket_size=config.data.bucket_size,
            seed=config.training.seed,
            source_sampling_alpha=config.data.source_sampling_alpha,
            source_sampling_weights=config.data.source_sampling_weights,
            max_source_upsampling=config.data.max_source_upsampling,
        )
        validation_sampler = DistributedBucketBatchSampler(
            validation_dataset,
            config.training.batch_size_per_gpu,
            rank=context.rank,
            world_size=context.world_size,
            bucket_size=config.data.bucket_size,
            seed=config.training.seed + 1,
        )
        train_loader_args = dataloader_runtime_kwargs(
            config.data.num_workers,
            context.device,
            training=True,
        )
        validation_workers = (
            0 if config.data.num_workers == 0 else min(4, max(1, config.data.num_workers // 4))
        )
        validation_loader_args = dataloader_runtime_kwargs(
            validation_workers,
            context.device,
            training=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=train_collator,
            **train_loader_args,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_sampler=validation_sampler,
            collate_fn=validation_collator,
            **validation_loader_args,
        )

        # ── 모델 생성과 분산 배치 ────────────────────────────────────────
        announce("모델을 생성하고 장치에 배치합니다.", context)
        # 모든 CUDA 전략은 meta device에서 파라미터 수와 영구 상태 용량을 먼저
        # 검사합니다. 통과한 뒤에만 single/DDP는 전체 모델을, FSDP2는 shard를
        # 실제 GPU에 할당하므로 과대 구성도 constructor OOM보다 명확히 실패합니다.
        parallel_strategy = resolve_parallel_strategy(
            config.training.parallel_strategy,
            context,
            legacy_fsdp2=config.training.fsdp2,
        )
        model, parameter_count, capacity, materialize_meta = construct_training_model(
            config,
            context,
            pad_id=tokenizer.pad_id,
            parallel_strategy=parallel_strategy,
        )
        # SFT와 MRT가 같은 DDP wrapper를 공유하므로 단계 전환 뒤의 파라미터
        # 사용 집합까지 고려해 static_graph 사용 여부를 정합니다.
        detect_unused_parameters = requires_ddp_unused_parameter_detection(config)
        model = parallelize_model(
            model,
            context,
            strategy=config.training.parallel_strategy,
            use_fsdp2=config.training.fsdp2,
            precision=config.training.precision,
            reduce_dtype=config.training.fsdp_reduce_dtype,
            reshard_after_forward=config.training.reshard_after_forward,
            materialize_meta=materialize_meta,
            find_unused_parameters=detect_unused_parameters,
        )
        if config.training.compile:
            model = torch.compile(model)
        # 여기서부터의 난수(dropout, denoising 등)는 rank 별로 달라야 합니다.
        seed_everything(config.training.seed, context.rank)
        announce(
            f"모델 파라미터 수: {parameter_count:,}; 병렬 전략: {parallel_strategy}",
            context,
        )
        if capacity is not None:
            announce(
                "영구 학습 상태 추정: "
                f"rank당 {capacity['per_rank_state_gib']:.1f} GiB / "
                f"안전 예산 {capacity['state_budget_gib']:.1f} GiB",
                context,
            )

        # ── 단계 ⑥: SFT 사전학습 ───────────────────────────────────────
        announce("1단계 SFT 사전학습을 시작합니다.", context)
        pretrain_result = train(
            model,
            train_loader,
            validation_loader,
            pretrain_config,
            context,
            stage_name="pretrain/SFT",
        )
        barrier(context)
        memory = release_stage_resources(context, train_loader, validation_loader)
        del train_loader, validation_loader
        del train_sampler, validation_sampler
        del train_collator, validation_collator
        if memory:
            announce(
                "사전학습 메모리 정리: "
                f"allocated {memory['before_allocated_gib']:.2f}→"
                f"{memory['after_allocated_gib']:.2f} GiB, "
                f"reserved {memory['before_reserved_gib']:.2f}→"
                f"{memory['after_reserved_gib']:.2f} GiB",
                context,
            )

        # ── 단계 ⑦: MRT 사후학습 ───────────────────────────────────────
        if config.posttraining.enabled:
            post = config.posttraining
            post_config = copy.deepcopy(config)
            post_config.training.output_dir = str(run_root / "posttrain")
            post_config.training.max_steps = post.max_steps
            post_config.training.batch_size_per_gpu = post.batch_size_per_gpu
            post_config.training.gradient_accumulation_steps = post.gradient_accumulation_steps
            post_config.training.learning_rate = post.learning_rate
            post_config.training.warmup_steps = post.warmup_steps
            post_config.training.eval_every = post.eval_every
            post_config.training.save_every = post.save_every
            post_config.training.early_stopping_patience = post.early_stopping_patience
            post_config.training.resume_from = None
            post_config.training.tensorboard_dir = None
            resume = find_auto_resume(post_config)
            if resume:
                post_config.training.resume_from = resume
                announce(f"이전 사후학습 발견 → {resume} 에서 자동 재개합니다.", context)

            # 보상 계산은 깨끗한 원문/정답을 기준으로 해야 하므로 증강을 끕니다.
            post_collator = SionBatchCollator(
                **collator_args,
                denoise_probability=0.0,
                source_token_dropout=0.0,
            )
            post_sampler = DistributedBucketBatchSampler(
                train_dataset,
                post.batch_size_per_gpu,
                rank=context.rank,
                world_size=context.world_size,
                bucket_size=config.data.bucket_size,
                seed=config.training.seed + 2,
                source_sampling_alpha=config.data.source_sampling_alpha,
                source_sampling_weights=config.data.source_sampling_weights,
                max_source_upsampling=config.data.max_source_upsampling,
            )
            post_validation_sampler = DistributedBucketBatchSampler(
                validation_dataset,
                post.eval_batch_size_per_gpu,
                rank=context.rank,
                world_size=context.world_size,
                bucket_size=config.data.bucket_size,
                seed=config.training.seed + 3,
            )
            post_loader = DataLoader(
                train_dataset,
                batch_sampler=post_sampler,
                collate_fn=post_collator,
                **train_loader_args,
            )
            post_validation_loader = DataLoader(
                validation_dataset,
                batch_sampler=post_validation_sampler,
                collate_fn=post_collator,
                **validation_loader_args,
            )
            objective = MinimumRiskObjective(tokenizer, post)
            announce(
                f"2단계 복합 MRT/선호 사후학습을 시작합니다: "
                f"후보 {post.samples_per_source}개, risk {post.risk_weight:.2f}, "
                f"preference {post.preference_weight:.2f}, "
                f"검증 beam {post.validation_num_beams}",
                context,
            )
            posttrain_result = train(
                model,
                post_loader,
                post_validation_loader,
                post_config,
                context,
                objective=objective,
                stage_name="posttrain/composite-MRT+preference",
            )
            barrier(context)
            memory = release_stage_resources(
                context,
                post_loader,
                post_validation_loader,
            )
            if memory:
                announce(
                    "사후학습 메모리 정리: "
                    f"allocated {memory['before_allocated_gib']:.2f}→"
                    f"{memory['after_allocated_gib']:.2f} GiB, "
                    f"reserved {memory['before_reserved_gib']:.2f}→"
                    f"{memory['after_reserved_gib']:.2f} GiB",
                    context,
                )
            final_step = int(posttrain_result["selected_step"])
        else:
            announce("posttraining.enabled=false — 사후학습을 건너뜁니다.", context)
            final_step = int(pretrain_result["selected_step"])

        # 중간 best/latest에서는 학습 재개와 빠른 확인에 필요한 경량 포맷만
        # 저장합니다. 모든 학습 단계가 끝난 지금 선택된 best 가중치에서 7종을
        # 한 번만 생성해, 매 평가 때 대형 CPU 양자화/I/O로 H100을 세우지 않습니다.
        final_stage = "posttrain" if config.posttraining.enabled else "pretrain"
        announce(
            "선택된 best 가중치 최종 내보내기: " + ", ".join(config.training.final_export_formats),
            context,
        )
        final_export_dir = export_final_model(
            model,
            config,
            context,
            run_root,
            stage=final_stage,
            step=final_step,
        )
        announce(f"최종 모델 내보내기 검증 완료: {final_export_dir}", context)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
