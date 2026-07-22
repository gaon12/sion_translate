"""역번역(backtranslation) 데이터 증강 CLI.

원리: 단일어(한쪽 언어만 있는) 텍스트를 학습된 모델로 반대 언어로 번역해
합성 번역쌍을 만듭니다. '진짜 문장이 target 쪽에 오는' 방향의 학습 신호가
늘어나 번역 품질(특히 자연스러움)이 올라가는, 상용 번역기들도 쓰는
검증된 기법입니다.

사용법:
    1. data_mono/ 폴더에 단일어 텍스트를 넣습니다 (한 줄 = 한 문장).
       파일 이름으로 언어를 표시합니다: 예) news.ja.txt, wiki.ko.txt
    2. kjx-augment 실행 → data/bt_*.jsonl 로 합성쌍이 저장됩니다.
    3. 다음 kjx-train 실행 때 자동 인식되어 재준비/학습에 반영됩니다.

과증강 방지 안전장치 (모델 성능 하락 방지):
    - 합성쌍 총량을 실데이터의 --max-ratio 배(기본 1.0)로 제한합니다.
    - 품질 필터를 통과하지 못한 합성쌍(너무 짧음/원문 복사/스크립트 불일치
      등)은 버립니다.
    - 합성 파일(bt_*)은 데이터 준비 때 train split 에만 들어가고
      (검증/테스트 오염 방지), 샘플링 가중치도 자동으로 0.5 로 낮아집니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kjx.config import config_from_raw, load_raw_config
from kjx.data.quality import assess_pair, canonical_text
from kjx.inference import Translator, find_exported_model

DEFAULT_CONFIG_FILE = "kjx.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtranslation data augmentation")
    parser.add_argument("--mono-dir", default="data_mono", help="단일어 텍스트 폴더 (기본: data_mono)")
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.0,
        help="합성쌍 총량 상한 = 실데이터 쌍 수 × 이 값 (기본 1.0 = 실데이터와 같은 양까지)",
    )
    parser.add_argument("--model", help="내보낸 모델 경로 (기본: exports 자동 탐색)")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--config", help=f"설정 파일 (기본: {DEFAULT_CONFIG_FILE})")
    return parser


def synthetic_budget(real_pairs: int, existing_synthetic: int, max_ratio: float) -> int:
    """앞으로 더 만들 수 있는 합성쌍 수. 실데이터 × max_ratio 가 총량 상한입니다."""
    return max(0, int(real_pairs * max_ratio) - existing_synthetic)


def count_dataset_pairs(dataset_dir: Path, synthetic_prefix: str) -> tuple[int, int]:
    """준비된 데이터셋 manifest 에서 (실데이터 쌍 수, 기존 합성쌍 수)를 읽습니다."""
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{dataset_dir} 에 준비된 데이터셋이 없습니다. 먼저 kjx-train 을 실행하세요."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    real = 0
    synthetic = 0
    for source in manifest.get("sources", []):
        pairs = int(source.get("stats", {}).get("valid_pairs", 0))
        if str(source.get("name", "")).startswith(synthetic_prefix):
            synthetic += pairs
        else:
            real += pairs
    return real, synthetic


def log(message: str) -> None:
    print(f"[KJ-X] {message}", flush=True)


def main() -> None:
    args = build_parser().parse_args()

    # ── 설정과 언어쌍 ───────────────────────────────────────────────────
    config_path = args.config or (
        DEFAULT_CONFIG_FILE if Path(DEFAULT_CONFIG_FILE).exists() else None
    )
    config = config_from_raw(load_raw_config(config_path) if config_path else {})
    pair = tuple(config.data.language_pair)
    prefix = config.data.synthetic_prefix

    # ── 과증강 방지: 예산 계산 ──────────────────────────────────────────
    real_pairs, existing_synthetic = count_dataset_pairs(
        Path(config.data.dataset_dir), prefix
    )
    budget = synthetic_budget(real_pairs, existing_synthetic, args.max_ratio)
    log(
        f"실데이터 {real_pairs:,}쌍 / 기존 합성 {existing_synthetic:,}쌍 / "
        f"상한 비율 {args.max_ratio:g} → 이번에 최대 {budget:,}쌍 생성 가능"
    )
    if budget <= 0:
        log("합성 데이터가 이미 상한에 도달했습니다. --max-ratio 를 올리지 않는 한 추가 생성하지 않습니다.")
        return

    # ── 단일어 파일 탐색: <이름>.<언어>.txt ─────────────────────────────
    mono_dir = Path(args.mono_dir)
    mono_files: list[tuple[Path, str]] = []
    for path in sorted(mono_dir.glob("*.txt")) if mono_dir.exists() else []:
        parts = path.name.split(".")
        if len(parts) >= 3 and parts[-2] in pair:
            mono_files.append((path, parts[-2]))
    if not mono_files:
        raise SystemExit(
            f"{mono_dir}/ 에 단일어 파일이 없습니다. "
            f"'이름.<언어>.txt' 형식으로 넣어 주세요 (언어: {'/'.join(pair)}). 예: news.{pair[1]}.txt"
        )

    # ── 모델 로드 ───────────────────────────────────────────────────────
    model_path = args.model or find_exported_model(config.training.output_dir)
    log(f"모델 로드: {model_path}")
    translator = Translator(model_path, config.data.tokenizer_model)

    # ── 파일별로 역번역 수행 ────────────────────────────────────────────
    data_dir = Path(config.data.raw_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    seen: set[str] = set()  # 이번 실행 내 중복 문장 제거
    for path, mono_language in mono_files:
        if total_written >= budget:
            break
        # 단일어 문장이 '진짜 target' 이 되도록, 반대 언어로 번역해
        # 합성 source 를 만듭니다.
        other_language = pair[0] if mono_language == pair[1] else pair[1]
        output_path = data_dir / f"{prefix}{path.stem}.jsonl"
        if output_path.exists():
            log(f"건너뜀: {output_path.name} 이미 존재 (재생성하려면 삭제 후 실행)")
            continue

        lines: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = canonical_text(line)
                if text and text not in seen:
                    seen.add(text)
                    lines.append(text)
                if total_written + len(lines) >= budget:
                    break
        if not lines:
            continue
        log(f"{path.name}: {len(lines):,}문장 → {other_language} 로 역번역 중...")

        written = 0
        filtered = 0
        with output_path.open("w", encoding="utf-8") as out:
            for start in range(0, len(lines), args.batch_size):
                chunk = lines[start : start + args.batch_size]
                translations = translator.translate(
                    chunk,
                    target_language=other_language,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                )
                for mono_text, synthetic_text in zip(chunk, translations, strict=True):
                    row = {
                        pair[0]: synthetic_text if mono_language == pair[1] else mono_text,
                        pair[1]: mono_text if mono_language == pair[1] else synthetic_text,
                    }
                    # 품질 필터: 실데이터와 같은 기준으로 손상된 합성쌍을 버립니다.
                    assessment = assess_pair(
                        row[pair[0]], row[pair[1]], languages=pair
                    )
                    if not assessment.accepted:
                        filtered += 1
                        continue
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
        if written == 0:
            # 빈 파일을 남기면 다음 학습 때 불필요한 데이터셋 재준비를
            # 유발하므로 지웁니다.
            output_path.unlink(missing_ok=True)
            log(f"{path.name}: 품질 필터를 통과한 합성쌍이 없어 저장하지 않음 (탈락 {filtered:,}쌍)")
            continue
        total_written += written
        log(f"{output_path.name}: {written:,}쌍 저장 (품질 탈락 {filtered:,}쌍)")

    if total_written == 0:
        log("저장된 합성쌍이 없습니다.")
        return
    log(
        f"완료: 총 {total_written:,}쌍. 다음 kjx-train 실행 때 자동 인식되어 "
        f"train 전용 + 가중치 0.5 로 반영됩니다."
    )


if __name__ == "__main__":
    main()
