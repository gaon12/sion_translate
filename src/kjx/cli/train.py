"""KJ-X 학습 실행 진입점(CLI) — 인자 없이 실행하는 완전 자동 파이프라인.

    kjx-train            ← 이것만 실행하면 됩니다.

동작 순서:
    ① 실행 환경(GPU 수·VRAM·bf16·CPU) 자동 인식
    ② 설정 로드 — 프로젝트 루트의 ``kjx.yaml`` 이 있으면 그 값을 우선 적용
       (없거나 비워 두면 전부 자동). ``--config`` 로 다른 파일도 지정 가능.
    ③ ``data/*.jsonl`` 자동 인식 — 토크나이저가 없으면 학습하고,
       파일이 추가/변경되었으면 데이터셋을 자동으로 다시 준비
    ④ 데이터 규모에 맞춰 모델 크기·step 수·배치 등 수치 자동 결정
    ⑤ 이전 학습이 있으면 단계별 checkpoints/latest 에서 자동 재개
    ⑥ SFT 사전학습 후 pretrain/에 저장
    ⑦ 복합 보상 MRT + 다중 후보 선호학습 후 posttrain/에 별도 저장

각 단계가 시작될 때마다 "[KJ-X] …" 텍스트가 출력되므로 현재 어디까지
진행됐는지 터미널에서 바로 확인할 수 있습니다.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kjx.auto import (
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
from kjx.config import AppConfig, config_from_raw, load_raw_config
from kjx.console import configure_stdio
from kjx.data import DistributedBucketBatchSampler, IndexedParallelDataset, KJBatchCollator
from kjx.model import KJXForConditionalGeneration
from kjx.tokenizer import KJTokenizer
from kjx.training.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    initialize_distributed,
    parallelize_model,
)
from kjx.training.objectives import MinimumRiskObjective
from kjx.training.trainer import announce, train
from kjx.performance import build_cpu_plan

DEFAULT_CONFIG_FILE = "kjx.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train KJ-X. 인자 없이 실행하면 환경/데이터를 자동 인식합니다."
    )
    parser.add_argument(
        "--config", help=f"설정 파일 (기본: 루트의 {DEFAULT_CONFIG_FILE}, 없으면 전부 자동)"
    )
    parser.add_argument("--max-steps", type=int, help="최대 step 수동 지정 (자동값 무시)")
    parser.add_argument("--posttrain-steps", type=int, help="MRT 사후학습 step 수동 지정")
    parser.add_argument(
        "--skip-posttraining", action="store_true", help="SFT 사전학습까지만 실행"
    )
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
        files = scan_raw_data(data_dir)
        dataset_ready = (dataset_dir / "manifest.json").exists()

        if not files and not dataset_ready:
            raise FileNotFoundError(
                f"원천 데이터({data_dir}/*.jsonl)도 준비된 데이터셋({dataset_dir})도 없습니다."
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
                from kjx.tokenizer import train_tokenizer

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
                    language_pair=config.data.language_pair,
                    num_workers=cpu_plan.preprocess_workers,
                    num_threads=cpu_plan.sentencepiece_threads,
                )
                announce("토크나이저 학습 완료.", context)

            # ── 데이터셋 (지문 기반 변경 감지) ─────────────────────────
            stored = stored_fingerprint(dataset_dir) if dataset_ready else None
            if dataset_ready and stored is None:
                # 수동으로 준비한 데이터셋: 현재 파일 목록을 지문으로 채택합니다.
                announce(
                    "기존 데이터셋에 지문이 없어 현재 원천 데이터 기준으로 기록합니다.", context
                )
                write_fingerprint(dataset_dir, files)
            elif not dataset_ready or stored != files:
                from kjx.data.prepare import prepare_dataset

                if dataset_ready:
                    backup = backup_stale_dataset(dataset_dir)
                    announce(
                        f"원천 데이터 변경 감지 → 기존 데이터셋을 {backup.name}/ 으로 보관합니다.",
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
                    language_pair=config.data.language_pair,
                    train_only_prefixes=(config.data.synthetic_prefix,),
                    num_workers=cpu_plan.dataset_workers,
                )
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
        tokenizer = KJTokenizer(config.data.tokenizer_model)
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
            denoise_noise_density=config.data.denoise_noise_density,
            denoise_mean_span=config.data.denoise_mean_span,
            token_features=config.data.tokenizer_features,
        )
        train_collator = KJBatchCollator(
            **collator_args,
            denoise_probability=config.data.denoise_probability,
            # 온라인 증강(원문 토큰 dropout)은 학습에만 적용합니다.
            source_token_dropout=config.data.source_token_dropout,
        )
        validation_collator = KJBatchCollator(
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
        loader_args = dict(
            num_workers=config.data.num_workers,
            pin_memory=context.device.type == "cuda",
            persistent_workers=config.data.num_workers > 0,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=train_collator,
            **loader_args,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_sampler=validation_sampler,
            collate_fn=validation_collator,
            **loader_args,
        )

        # ── 모델 생성과 분산 배치 ────────────────────────────────────────
        announce("모델을 생성하고 장치에 배치합니다.", context)
        # FSDP2 에서는 meta device 로 '빈 모델'을 먼저 만들고, shard 이후에
        # 실제 메모리를 할당해 대형 모델도 rank 당 메모리 한도 안에서 생성합니다.
        materialize_meta = context.distributed and config.training.fsdp2
        construction_device = "meta" if materialize_meta else context.device
        with torch.device(construction_device):
            model = KJXForConditionalGeneration(config.model, pad_id=tokenizer.pad_id)
        parameter_count = model.parameter_count()
        # 실험 모듈이 켜져 있으면 일부 파라미터가 조건부로만 쓰이므로
        # DDP 의 unused-parameter 탐색이 필요합니다 (꺼져 있으면 비활성 = 더 빠름).
        experimental = config.model.experimental
        has_conditional_parameters = any(
            (
                experimental.bats_enabled,
                experimental.core_enabled,
                experimental.tetm_enabled,
                experimental.morphoscript_enabled,
            )
        )
        model = parallelize_model(
            model,
            context,
            use_fsdp2=config.training.fsdp2,
            precision=config.training.precision,
            reshard_after_forward=config.training.reshard_after_forward,
            materialize_meta=materialize_meta,
            find_unused_parameters=has_conditional_parameters,
        )
        if config.training.compile:
            model = torch.compile(model)
        # 여기서부터의 난수(dropout, denoising 등)는 rank 별로 달라야 합니다.
        seed_everything(config.training.seed, context.rank)
        announce(f"모델 파라미터 수: {parameter_count:,}", context)

        # ── 단계 ⑥: SFT 사전학습 ───────────────────────────────────────
        announce("1단계 SFT 사전학습을 시작합니다.", context)
        train(
            model,
            train_loader,
            validation_loader,
            pretrain_config,
            context,
            stage_name="pretrain/SFT",
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
            post_collator = KJBatchCollator(
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
            post_loader = DataLoader(
                train_dataset,
                batch_sampler=post_sampler,
                collate_fn=post_collator,
                **loader_args,
            )
            objective = MinimumRiskObjective(tokenizer, post)
            announce(
                f"2단계 복합 MRT/선호 사후학습을 시작합니다: "
                f"후보 {post.samples_per_source}개, risk {post.risk_weight:.2f}, "
                f"preference {post.preference_weight:.2f}, "
                f"검증 beam {post.validation_num_beams}",
                context,
            )
            train(
                model,
                post_loader,
                validation_loader,
                post_config,
                context,
                objective=objective,
                stage_name="posttrain/composite-MRT+preference",
            )
        else:
            announce("posttraining.enabled=false — 사후학습을 건너뜁니다.", context)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
