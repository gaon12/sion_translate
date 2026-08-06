"""국립국어원 모두의 말뭉치 ZIP 을 foundation 코퍼스 JSONL 로 변환한다.

모두의 말뭉치는 배포본마다 폴더 이름이 다르지만 JSON 구조는 같습니다::

    {"id": ..., "metadata": {...},
     "document": [{"id": ..., "utterance"|"paragraph"|"sentence": [{"form": "..."}]}]}

**발화/문단 단위로 한 줄씩** 씁니다. 문서 통째로 한 줄에 넣으면 foundation
준비 단계가 문장 경계로 다시 나눠야 하고, 그 전 단계에서 잘리거나 버려질 수
있습니다 — 실측으로 문서 단위 코퍼스는 문자의 25.9%를 잃었습니다.

빈 ``form`` 은 건너뜁니다. 구어 말뭉치에는 "배경 음악 있음" 처럼 note 만 있고
발화가 없는 항목이 실제로 들어 있습니다.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterator

from sion_translate.console import configure_stdio

# 발화·문단·문장 중 어느 이름으로든 들어옵니다.
SEGMENT_KEYS = ("utterance", "paragraph", "sentence")

# 구어 전사에 붙는 표기. 남겨 두면 모델이 이 기호를 배웁니다.
_TRANSCRIPTION_NOISE = re.compile(
    r"&[a-zA-Z-]+\d*&"  # &name&, &address& 같은 비식별 태그
    r"|\([^)]{0,40}\)/\([^)]{0,40}\)"  # (철자)/(발음) 이중 표기
    r"|[{}<>@#*~]"
)
_WHITESPACE = re.compile(r"\s+")


def clean(text: str) -> str:
    """전사 기호를 걷어내고 공백을 정규화한다."""

    text = unicodedata.normalize("NFC", text)
    text = _TRANSCRIPTION_NOISE.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def iter_segments(payload: object) -> Iterator[str]:
    """하나의 JSON 문서에서 발화/문단 텍스트를 낸다."""

    if not isinstance(payload, dict):
        return
    documents = payload.get("document")
    if not isinstance(documents, list):
        documents = [payload]
    for document in documents:
        if not isinstance(document, dict):
            continue
        for key in SEGMENT_KEYS:
            segments = document.get(key)
            if not isinstance(segments, list):
                continue
            for segment in segments:
                if isinstance(segment, dict):
                    form = segment.get("form")
                elif isinstance(segment, str):
                    form = segment
                else:
                    form = None
                if isinstance(form, str) and form.strip():
                    yield form


def extract_archive(
    archive: Path,
    sink,
    *,
    minimum_characters: int,
    maximum_characters: int,
    seen: set[int] | None,
) -> Counter:
    stats: Counter = Counter()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if not member.lower().endswith(".json"):
                continue
            try:
                payload = json.loads(bundle.read(member).decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                stats["unreadable_members"] += 1
                continue
            stats["members"] += 1
            for form in iter_segments(payload):
                stats["segments"] += 1
                text = clean(form)
                if len(text) < minimum_characters:
                    stats["too_short"] += 1
                    continue
                if len(text) > maximum_characters:
                    stats["too_long"] += 1
                    continue
                if seen is not None:
                    digest = hash(text)
                    if digest in seen:
                        stats["duplicate"] += 1
                        continue
                    seen.add(digest)
                sink.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                stats["written"] += 1
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="모두의 말뭉치 ZIP → foundation JSONL")
    parser.add_argument("--archive", nargs="+", required=True, help="ZIP 경로")
    parser.add_argument("--output", required=True, help="출력 JSONL")
    parser.add_argument("--minimum-characters", type=int, default=8)
    parser.add_argument("--maximum-characters", type=int, default=4000)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="중복 제거를 끕니다 (기본은 파일 내 중복 제거)",
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_parser().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[int] | None = None if args.keep_duplicates else set()
    totals: Counter = Counter()
    with output.open("w", encoding="utf-8") as sink:
        for name in args.archive:
            archive = Path(name)
            if not archive.is_file():
                print(f"  건너뜀 (없음): {archive}", flush=True)
                continue
            stats = extract_archive(
                archive,
                sink,
                minimum_characters=args.minimum_characters,
                maximum_characters=args.maximum_characters,
                seen=seen,
            )
            totals.update(stats)
            print(
                f"  {archive.name}: 발화 {stats['segments']:,} → 기록 {stats['written']:,}"
                f" (짧음 {stats['too_short']:,} / 김 {stats['too_long']:,}"
                f" / 중복 {stats['duplicate']:,})",
                flush=True,
            )
    print(
        f"{output}: 총 {totals['written']:,}행, {output.stat().st_size / 1e9:.2f} GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
