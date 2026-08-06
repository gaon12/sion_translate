"""단일어 코퍼스를 foundation 학습용 indexed shard 로 변환한다.

병렬 데이터셋과 **같은 shard 규격**을 씁니다. 복원 과제는 "원문을 망가뜨린
것"이 입력이고 "원문"이 정답이라, ``src`` 와 ``tgt`` 에 같은 토큰열을 쓰고
망가뜨리는 일은 collator 가 배치마다 새로 합니다. 매 epoch 다른 span 이
가려지므로 미리 손상시켜 저장하는 것보다 신호가 많고, 디스크도 덜 씁니다.

두 가지가 병렬 준비와 다릅니다.

- ``forward_only=True``. 양방향 확장은 (a→b, b→a) 를 만드는 장치인데 여기서는
  두 방향이 같은 예제라 그대로 두면 모든 문장이 두 번 학습됩니다.
- ``src_language == tgt_language``. collator 가 이 값을 보고 ``<denoise_xx>``
  과제 태그를 고릅니다.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sion_translate.artifacts import FOUNDATION_RELEASE_NAME
from sion_translate.data.monolingual import (
    DEFAULT_LANGUAGE_SAMPLING_ALPHA,
    MonolingualDiscovery,
    ReadStats,
    assess_language_balance,
    iter_monolingual_lines,
    segment_text,
)
from sion_translate.data.prepare import INDEX_DTYPE, ShardWriter, infer_register
from sion_translate.fingerprint import file_sha256
from sion_translate.splitting import choose_split_for_key
from sion_translate.tokenizer import SionTokenizer, normalize_text

FOUNDATION_INDEX_FORMAT = "sion-foundation-indexed-v1"
FOUNDATION_PREPROCESSING_SCHEMA = "foundation-monolingual-v1"


@dataclass
class LanguageStats:
    lines_read: int = 0
    accepted: int = 0
    too_short: int = 0
    # 상한을 넘어 폐기한 행. 이제 나누므로 0 이어야 정상입니다.
    too_long: int = 0
    # 여러 조각으로 나뉜 문서 수와, 그 결과로 생긴 총 조각 수.
    segmented_documents: int = 0
    segments: int = 0
    duplicate: int = 0
    empty_after_tokenization: int = 0
    read_rejects: dict[str, int] = field(default_factory=dict)

    def merge_read(self, stats: ReadStats) -> None:
        for reason, count in stats.reasons().items():
            self.read_rejects[reason] = self.read_rejects.get(reason, 0) + count


@dataclass
class FoundationPrepareStats:
    languages: dict[str, LanguageStats] = field(default_factory=dict)
    train_records: int = 0
    validation_records: int = 0

    @property
    def total_records(self) -> int:
        return self.train_records + self.validation_records

    def accepted_per_language(self) -> dict[str, int]:
        return {language: stats.accepted for language, stats in self.languages.items()}


def _text_digest(language: str, text: str) -> bytes:
    return hashlib.blake2b(f"{language}\0{text}".encode("utf-8"), digest_size=16).digest()


def _is_usable(text: str) -> bool:
    """제어 문자만 있거나 눈에 보이는 글자가 없는 줄을 거른다."""

    return any(not unicodedata.category(char).startswith("C") for char in text)


def prepare_foundation_dataset(
    discovery: MonolingualDiscovery,
    tokenizer_model: str | Path,
    output_dir: str | Path,
    *,
    minimum_characters: int = 8,
    maximum_characters: int = 4000,
    max_tokens: int = 510,
    deduplicate: bool = True,
    shard_size: int = 200_000,
    validation_fraction: float = 0.002,
    language_sampling_alpha: float = DEFAULT_LANGUAGE_SAMPLING_ALPHA,
    minimum_language_share: float = 0.05,
    release_name: str = FOUNDATION_RELEASE_NAME,
) -> FoundationPrepareStats:
    """단일어 파일들을 ``output_dir`` 아래 train/validation shard 로 쓴다."""

    if not discovery.sources:
        raise ValueError(
            "단일어 코퍼스에 학습 가능한 파일이 없습니다. "
            f"루트={discovery.root} — 언어 코드 폴더 안에 .txt 또는 .jsonl 을 두십시오."
        )
    if minimum_characters < 1:
        raise ValueError("minimum_characters must be positive")
    if maximum_characters <= minimum_characters:
        raise ValueError("maximum_characters must be greater than minimum_characters")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Remove it or choose another foundation.dataset_dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = SionTokenizer(tokenizer_model)
    languages = discovery.languages
    missing_tags = sorted(set(languages) - set(tokenizer.languages))
    if missing_tags:
        raise ValueError(
            "Tokenizer is missing denoise tags for the monolingual languages: "
            f"{missing_tags}; retrain it with these languages configured"
        )
    language_to_id = {language: index for index, language in enumerate(languages)}

    writers = {
        split: ShardWriter(output_dir, split, shard_size, language_to_id)
        for split in ("train", "validation")
    }
    stats = FoundationPrepareStats(languages={language: LanguageStats() for language in languages})
    seen: set[bytes] = set()
    source_ids = {source.path: index for index, source in enumerate(discovery.sources)}

    def record_segment(
        text: str,
        *,
        language: str,
        language_stats: LanguageStats,
        source_id: int,
    ) -> None:
        """조각 하나를 shard 에 넣는다 (중복·빈 토큰은 여기서 걸러냄)."""

        nonlocal seen
        if deduplicate:
            digest = _text_digest(language, text)
            if digest in seen:
                language_stats.duplicate += 1
                return
            seen.add(digest)
        token_ids = tokenizer.encode(text)[:max_tokens]
        if not token_ids:
            language_stats.empty_after_tokenization += 1
            return
        # 복원 과제에는 test split 이 없습니다. 이 단계의 선택 지표는 복원
        # 손실뿐이고, 최종 품질 판정은 번역 단계의 holdout 이 합니다.
        split = choose_split_for_key(f"{language}\0{text}", validation_fraction, 0.0)
        if split == "test":
            split = "train"
        register = infer_register(text, language)
        writers[split].add(
            src_ids=token_ids,
            tgt_ids=token_ids,
            src_register=register,
            tgt_register=register,
            src_language=language,
            tgt_language=language,
            source_id=source_id,
            quality_score=100,
            synthetic=False,
            # 양방향 확장을 끕니다. 복원 과제는 두 방향이 같은 예제라
            # 켜 두면 모든 문장이 정확히 두 번 학습됩니다.
            forward_only=True,
        )
        language_stats.accepted += 1
        if split == "train":
            stats.train_records += 1
        else:
            stats.validation_records += 1

    for source in discovery.sources:
        language = source.language
        language_stats = stats.languages[language]
        read_stats = ReadStats()
        for raw_text in iter_monolingual_lines(source.path, stats=read_stats):
            language_stats.lines_read += 1
            document = normalize_text(raw_text)
            if not _is_usable(document):
                language_stats.too_short += 1
                continue
            # 긴 문서는 버리지 않고 나눕니다. 자르지도 않습니다. 실측으로
            # e_gov 는 문자의 97.3%, aozora 는 92.8%, kowiki 는 68.0% 가
            # "상한 초과" 한 줄이라 통째로 폐기됐고, 전체로는 25.8% 였습니다.
            segments = segment_text(
                document,
                maximum_characters=maximum_characters,
                minimum_characters=minimum_characters,
            )
            if not segments:
                language_stats.too_short += 1
                continue
            if len(segments) > 1:
                language_stats.segmented_documents += 1
            language_stats.segments += len(segments)
            for text in segments:
                record_segment(
                    text,
                    language=language,
                    language_stats=language_stats,
                    source_id=source_ids[source.path],
                )
        language_stats.merge_read(read_stats)

    for writer in writers.values():
        writer.close()

    if stats.total_records == 0:
        raise ValueError(
            "단일어 코퍼스에서 학습 가능한 문장이 하나도 나오지 않았습니다. "
            "minimum_characters/maximum_characters 와 파일 형식을 확인하십시오."
        )

    balance = assess_language_balance(
        stats.accepted_per_language(),
        alpha=language_sampling_alpha,
        minimum_share=minimum_language_share,
    )
    manifest = {
        "format": FOUNDATION_INDEX_FORMAT,
        "stage": "foundation",
        "release_name": release_name,
        "objective": "span-corruption-denoising",
        "languages": list(languages),
        "language_to_id": language_to_id,
        # 복원 과제는 언어쌍이 아니라 언어 하나짜리 과제입니다. 같은 언어를
        # 양쪽에 적어 두면 indexed reader 가 방향 해석을 그대로 할 수 있고,
        # forward_only 플래그가 역방향 복제를 막습니다.
        "language_pairs": [[language, language] for language in languages],
        "source_only_languages": [],
        "storage_sides": ["src", "tgt"],
        "index_dtype": INDEX_DTYPE.descr,
        "tokenizer_model": str(Path(tokenizer_model).resolve()),
        "tokenizer_sha256": file_sha256(tokenizer_model),
        "preprocessing_schema": FOUNDATION_PREPROCESSING_SCHEMA,
        "preprocessing_options": {
            "deduplicate": deduplicate,
            "maximum_characters": maximum_characters,
            "max_tokens": max_tokens,
            "minimum_characters": minimum_characters,
            "shard_size": shard_size,
            "validation_fraction": validation_fraction,
        },
        "language_sampling": {
            "alpha": language_sampling_alpha,
            "weights": balance.weights,
            "counts": balance.counts,
            "warnings": list(balance.warnings),
        },
        "sources": [
            {
                "id": source_ids[source.path],
                "language": source.language,
                "name": source.path.name,
                "path": str(source.path),
                "size_bytes": source.size_bytes,
            }
            for source in discovery.sources
        ],
        "skipped": [
            {"path": str(entry.path), "reason": entry.reason} for entry in discovery.skipped
        ],
        "stats": {
            "train_records": stats.train_records,
            "validation_records": stats.validation_records,
            "languages": {
                language: asdict(language_stats)
                for language, language_stats in stats.languages.items()
            },
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stats


def render_prepare_report(stats: FoundationPrepareStats) -> list[str]:
    """준비 결과 요약. 버려진 줄을 이유별로 드러내는 것이 목적입니다."""

    lines = [
        f"foundation 데이터셋: train {stats.train_records:,} / "
        f"validation {stats.validation_records:,}"
    ]
    for language in sorted(stats.languages):
        language_stats = stats.languages[language]
        dropped = {
            "too_short": language_stats.too_short,
            "too_long": language_stats.too_long,
            "duplicate": language_stats.duplicate,
            "empty_after_tokenization": language_stats.empty_after_tokenization,
            **language_stats.read_rejects,
        }
        rendered = ", ".join(f"{name} {count:,}" for name, count in dropped.items() if count)
        lines.append(
            f"  {language}: 읽음 {language_stats.lines_read:,} → "
            f"채택 {language_stats.accepted:,}" + (f" (제외: {rendered})" if rendered else "")
        )
    return lines


__all__ = [
    "FOUNDATION_INDEX_FORMAT",
    "FOUNDATION_PREPROCESSING_SCHEMA",
    "FoundationPrepareStats",
    "LanguageStats",
    "prepare_foundation_dataset",
    "render_prepare_report",
]
