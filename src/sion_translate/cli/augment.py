"""Authenticated backtranslation data augmentation CLI.

Monolingual files use ``<name>.<language>.txt``. The model artifact owns the
generation edge ``T→S``; the destination dataset configuration must own the
opposite training edge ``S→T``. Every published row is scoped to that one
training direction so a generated pseudo-target is never learned in reverse.
"""

from __future__ import annotations

import argparse
from collections.abc import Collection, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import cast

from sion_translate.augmentation import (
    AugmentationIdentity,
    AugmentationRegistry,
    FileSnapshot,
    JobProgress,
    build_job_identity,
    count_prepared_direction_pairs,
    load_augmentation_registry,
    reconcile_job_identity,
    run_augmentation_job,
    snapshot_file,
    synthetic_budget,
    validate_prepared_raw_contract,
)
from sion_translate.config import AppConfig, config_from_raw, load_raw_config
from sion_translate.console import configure_stdio
from sion_translate.data.quality import canonical_text
from sion_translate.inference import Translator, find_exported_model
from sion_translate.locking import artifact_locks

DEFAULT_CONFIG_FILE = "sion_translate.yaml"
_SHA256_LENGTH = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtranslation data augmentation")
    parser.add_argument(
        "--mono-dir", default="data_mono", help="단일어 텍스트 폴더 (기본: data_mono)"
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.0,
        help="방향별 합성 train 행 상한 = 같은 방향 real train 행 × 이 값",
    )
    parser.add_argument("--model", help="내보낸 생성 모델 경로 (기본: exports 자동 탐색)")
    parser.add_argument(
        "--tokenizer",
        help=(
            "생성 모델의 토크나이저 경로. 생략하면 목적 학습 설정의 토크나이저를 "
            "사용하며, 별도 reverse-generator artifact에는 이 옵션을 지정해야 합니다"
        ),
    )
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--language-pair",
        nargs=2,
        metavar=("LANG_A", "LANG_B"),
        help="다중 언어쌍 모델에서 증강할 물리 언어쌍",
    )
    parser.add_argument("--config", help=f"목적 학습 설정 파일 (기본: {DEFAULT_CONFIG_FILE})")
    return parser


def log(message: str) -> None:
    print(f"[sion] {message}", flush=True)


def resolve_augmentation_pair(
    requested: Sequence[str] | None,
    trained_pairs: Sequence[Sequence[str]],
) -> tuple[str, str]:
    """Choose a physical pair from the generator artifact, never local YAML."""

    pairs = [(str(pair[0]), str(pair[1])) for pair in trained_pairs]
    if requested is not None:
        edge = frozenset(map(str, requested))
        matches = [pair for pair in pairs if frozenset(pair) == edge]
        if len(matches) != 1:
            raise SystemExit(
                f"모델에 없는 --language-pair 입니다: {tuple(requested)} (지원: {pairs})"
            )
        return matches[0]
    if len(pairs) == 1:
        return pairs[0]
    raise SystemExit(
        f"다중 언어쌍 모델에서는 --language-pair LANG_A LANG_B를 지정하세요 (지원: {pairs})"
    )


def resolve_augmentation_destination(
    model_pair: Sequence[str],
    destination_pairs: Sequence[Sequence[str]],
) -> tuple[str, str]:
    """Bind the generator pair to exactly one destination physical pair."""

    edge = frozenset(map(str, model_pair))
    matches = [
        (str(pair[0]), str(pair[1]))
        for pair in destination_pairs
        if frozenset(map(str, pair)) == edge
    ]
    if len(matches) != 1:
        configured = [tuple(map(str, pair)) for pair in destination_pairs]
        raise SystemExit(
            "증강 모델의 언어쌍이 현재 학습 설정에 정확히 한 번 존재해야 합니다: "
            f"model={tuple(model_pair)}, config={configured}"
        )
    return matches[0]


def preflight_backtranslation_directions(
    pair: tuple[str, str],
    jobs: Sequence[tuple[Path, str]],
    generation_directions: Sequence[Sequence[str]],
    training_directions: Sequence[Sequence[str]],
) -> None:
    """Validate every true-BT generator/destination edge before generation."""

    generated = {tuple(map(str, direction)) for direction in generation_directions}
    trained = {tuple(map(str, direction)) for direction in training_directions}
    required_generation = {
        (mono_language, pair[0] if mono_language == pair[1] else pair[1])
        for _, mono_language in jobs
    }
    required_training = {(target, source) for source, target in required_generation}
    missing_generation = sorted(required_generation - generated)
    missing_training = sorted(required_training - trained)
    if not missing_generation and not missing_training:
        return

    failures: list[str] = []
    if missing_generation:
        needed = ", ".join(f"{source}→{target}" for source, target in missing_generation)
        supported = ", ".join(f"{source}→{target}" for source, target in sorted(generated))
        failures.append(f"모델 생성 방향 누락: {needed} (모델 지원: {supported or '없음'})")
    if missing_training:
        needed = ", ".join(f"{source}→{target}" for source, target in missing_training)
        supported = ", ".join(f"{source}→{target}" for source, target in sorted(trained))
        failures.append(f"목적 학습 방향 누락: {needed} (설정 지원: {supported or '없음'})")
    raise SystemExit("진짜 역번역 방향 계약을 만족하지 않습니다: " + "; ".join(failures))


def _valid_sha256(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return None
    try:
        int(value, 16)
    except ValueError:
        return None
    return value.lower()


def generator_identity(
    translator: Translator,
    model_snapshot: FileSnapshot,
) -> tuple[str, str]:
    """Return strong model and tokenizer identities authenticated by the export."""

    metadata = translator.export_metadata
    raw_source: object = metadata.get("source")
    source: Mapping[object, object] = (
        cast(Mapping[object, object], raw_source)
        if isinstance(raw_source, Mapping)
        else cast(Mapping[object, object], {})
    )
    source_sha = _valid_sha256(source.get("sha256"))
    raw_tokenizer: object = metadata.get("tokenizer")
    tokenizer: Mapping[object, object] = (
        cast(Mapping[object, object], raw_tokenizer)
        if isinstance(raw_tokenizer, Mapping)
        else cast(Mapping[object, object], {})
    )
    tokenizer_sha = _valid_sha256(tokenizer.get("sha256"))
    if tokenizer_sha is None and translator.tokenizer_metadata is not None:
        tokenizer_sha = _valid_sha256(translator.tokenizer_metadata.get("model_sha256"))
    if source_sha is None or tokenizer_sha is None:
        raise ValueError(
            "증강 생성 모델 export에 source/tokenizer SHA-256 신원이 없습니다. "
            "현재 1.5 export를 사용하세요."
        )
    identity_payload = {
        "loaded_artifact": {
            "size": model_snapshot.size,
            "sha256": model_snapshot.sha256,
        },
        "source_sha256": source_sha,
        "tokenizer_sha256": tokenizer_sha,
        "release_name": metadata.get("release_name"),
        "release_version": metadata.get("release_version"),
        "step": metadata.get("step"),
        "language_pairs": [list(pair) for pair in translator.language_pairs],
        "translation_directions": [list(edge) for edge in translator.translation_directions],
        "pipeline": metadata.get("pipeline"),
        "feature_flags": metadata.get("feature_flags"),
        "capabilities": metadata.get("capabilities"),
        "quantization": metadata.get("quantization"),
        "generation_defaults": metadata.get("generation_defaults"),
    }
    serialized = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), tokenizer_sha


def _discover_mono_files(mono_dir: Path, pair: tuple[str, str]) -> list[tuple[Path, str]]:
    jobs: list[tuple[Path, str]] = []
    for path in sorted(mono_dir.glob("*.txt")) if mono_dir.exists() else []:
        parts = path.name.split(".")
        if len(parts) >= 3 and parts[-2] in pair:
            jobs.append((path, parts[-2]))
    if not jobs:
        raise SystemExit(
            f"{mono_dir}/ 에 단일어 파일이 없습니다. '이름.<언어>.txt' 형식으로 "
            f"넣어 주세요 (언어: {'/'.join(pair)}). 예: news.{pair[1]}.txt"
        )
    return jobs


def _source_has_remaining_text(
    path: Path,
    progress: JobProgress,
    seen_mono_hashes: Collection[str] = frozenset(),
) -> bool:
    total_lines = 0
    has_text = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            total_lines = line_number + 1
            text = canonical_text(line)
            if (
                line_number >= progress.cursor_line
                and text
                and hashlib.sha256(text.encode("utf-8")).hexdigest() not in seen_mono_hashes
            ):
                has_text = True
    if progress.cursor_line > total_lines:
        raise ValueError(
            "augmentation ledger cursor가 단일어 입력 행 수보다 큽니다: "
            f"{progress.cursor_line} > {total_lines}"
        )
    if progress.eof and progress.cursor_line != total_lines:
        raise ValueError(
            "augmentation ledger eof 상태와 단일어 입력 행 수가 다릅니다: "
            f"cursor={progress.cursor_line}, lines={total_lines}"
        )
    return has_text


def _build_jobs(
    mono_files: Sequence[tuple[Path, str]],
    *,
    pair: tuple[str, str],
    synthetic_prefix: str,
    model_identity: str,
    tokenizer_identity: str,
    num_beams: int,
    max_new_tokens: int,
    registry: AugmentationRegistry,
) -> list[tuple[Path, AugmentationIdentity, JobProgress]]:
    jobs: list[tuple[Path, AugmentationIdentity, JobProgress]] = []
    for path, mono_language in mono_files:
        identity = build_job_identity(
            synthetic_prefix=synthetic_prefix,
            pair=pair,
            mono_language=mono_language,
            input_snapshot=snapshot_file(path),
            model_identity=model_identity,
            generator_tokenizer_sha256=tokenizer_identity,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        jobs.append((path, identity, reconcile_job_identity(registry, identity)))
    return jobs


def _run_locked(args: argparse.Namespace, config: AppConfig) -> None:
    data_dir = Path(config.data.raw_dir)
    prefix = config.data.synthetic_prefix
    fingerprint = validate_prepared_raw_contract(config.data, augment_prefix=prefix)
    registry = load_augmentation_registry(
        data_dir,
        prefix,
        [item.name for item in fingerprint.files],
    )

    model_path = Path(args.model or find_exported_model(config.training.output_dir)).resolve()
    generator_tokenizer = args.tokenizer or config.data.tokenizer_model
    log(f"생성 모델 로드: {model_path}")
    model_stat_before = model_path.stat()
    translator = Translator(model_path, generator_tokenizer)
    model_snapshot = snapshot_file(model_path)
    model_stat_after = model_path.stat()
    model_file_identity_before = (
        model_stat_before.st_size,
        model_stat_before.st_mtime_ns,
        model_stat_before.st_ctime_ns,
        model_stat_before.st_dev,
        model_stat_before.st_ino,
    )
    model_file_identity_after = (
        model_stat_after.st_size,
        model_stat_after.st_mtime_ns,
        model_stat_after.st_ctime_ns,
        model_stat_after.st_dev,
        model_stat_after.st_ino,
    )
    if model_file_identity_before != model_file_identity_after:
        raise RuntimeError("생성 모델 파일이 로드·해시 계산 중 변경되었습니다")
    model_pair = resolve_augmentation_pair(args.language_pair, translator.language_pairs)
    pair = resolve_augmentation_destination(
        model_pair,
        config.data.configured_language_pairs(),
    )
    model_max_seq_len = int(translator.model_config.max_seq_len)
    if args.max_new_tokens > model_max_seq_len:
        raise SystemExit(
            f"--max-new-tokens {args.max_new_tokens}이 모델 최대 길이 "
            f"{model_max_seq_len}보다 큽니다. ledger를 만들기 전에 중단합니다."
        )
    model_identity, tokenizer_identity = generator_identity(translator, model_snapshot)

    mono_files = _discover_mono_files(Path(args.mono_dir), pair)
    jobs = _build_jobs(
        mono_files,
        pair=pair,
        synthetic_prefix=prefix,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        registry=registry,
    )
    seen_by_direction = registry.mono_hashes_by_direction()
    actionable: list[tuple[Path, AugmentationIdentity, JobProgress]] = []
    finalized_empty = 0
    for path, identity, progress in jobs:
        direction_seen = seen_by_direction.setdefault(identity.training_direction, set())
        has_remaining_text = _source_has_remaining_text(path, progress, direction_seen)
        if progress.eof:
            continue
        if has_remaining_text:
            actionable.append((path, identity, progress))
            continue
        result = run_augmentation_job(
            translator,
            mono_path=path,
            data_dir=data_dir,
            synthetic_prefix=prefix,
            progress=progress,
            accepted_budget=1,
            batch_size=args.batch_size,
            seen_mono_hashes=direction_seen,
        )
        if result.written:
            raise RuntimeError("blank-only augmentation finalization unexpectedly wrote a row")
        finalized_empty += 1
    if not actionable:
        if finalized_empty:
            log(f"빈 입력 또는 기게시 문장 {finalized_empty:,}개 job을 EOF로 확정했습니다.")
        else:
            log("처리할 새 단일어 문장이 없습니다.")
        return

    preflight_backtranslation_directions(
        pair,
        [(path, identity.mono_language) for path, identity, _ in actionable],
        translator.translation_directions,
        config.data.configured_translation_directions(),
    )
    directions = tuple(dict.fromkeys(identity.training_direction for _, identity, _ in actionable))
    prepared = count_prepared_direction_pairs(config.data.dataset_dir, directions)
    pending = registry.pending_direction_counts()
    total_written = 0

    max_source_tokens = model_max_seq_len - 2

    def source_fits(text: str) -> bool:
        return len(translator.tokenizer.encode(text)) <= max_source_tokens

    for path, identity, progress in actionable:
        direction = identity.training_direction
        counts = prepared[direction]
        existing_synthetic = counts.synthetic + pending.get(direction, 0)
        budget = synthetic_budget(counts.real, existing_synthetic, args.max_ratio)
        log(
            f"{direction[0]}→{direction[1]}: real train {counts.real:,} / "
            f"기존·미준비 synthetic {existing_synthetic:,} / 비율 {args.max_ratio:g} "
            f"→ 최대 {budget:,}행"
        )
        if budget <= 0:
            continue
        result = run_augmentation_job(
            translator,
            mono_path=path,
            data_dir=data_dir,
            synthetic_prefix=prefix,
            progress=progress,
            accepted_budget=budget,
            batch_size=args.batch_size,
            seen_mono_hashes=seen_by_direction.setdefault(direction, set()),
            source_fits=source_fits,
        )
        pending[direction] = pending.get(direction, 0) + result.written
        total_written += result.written
        log(
            f"{path.name}: {result.written:,}행 게시 / 품질 탈락 "
            f"{result.quality_filtered:,} / 중복 {result.duplicates:,} / 길이 초과 "
            f"{result.too_long:,}"
        )

    if total_written == 0:
        log("방향별 합성 상한 또는 입력 종료로 새로 게시된 행이 없습니다.")
        return
    log(
        f"완료: 총 {total_written:,}행. 다음 sion-train에서 train 전용, "
        f"합성 sampling weight {config.data.synthetic_sampling_weight:g}로 반영됩니다."
    )


def main() -> None:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.max_ratio) or args.max_ratio < 0:
        parser.error("--max-ratio must be finite and non-negative")
    if args.num_beams < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--num-beams, --batch-size, and --max-new-tokens must be positive")

    config_path = args.config or (
        DEFAULT_CONFIG_FILE if Path(DEFAULT_CONFIG_FILE).exists() else None
    )
    config = config_from_raw(load_raw_config(config_path) if config_path else {})
    config.validate()
    if config.data.synthetic_sampling_weight == 0:
        raise SystemExit(
            "data.synthetic_sampling_weight가 0이라 생성해도 학습되지 않습니다. "
            "증강을 실행하지 않습니다."
        )

    # Training/prepare holds the same dataset-parent lease. Acquire it before
    # constructing Translator so two augment processes cannot load large models
    # onto one GPU and only then discover that one must stop.
    lock_roots = (
        Path(config.data.raw_dir).resolve(),
        Path(config.data.dataset_dir).resolve().parent,
    )
    with artifact_locks(lock_roots):
        _run_locked(args, config)


if __name__ == "__main__":
    main()
