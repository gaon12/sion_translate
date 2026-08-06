"""open2ch(일본어 웹 포럼 대화)를 foundation 코퍼스로 받아 변환한다.

일본어 쪽 실측 공백을 메우기 위한 것입니다. 현재 ja 코퍼스는 fineweb2 ·
wikipedia · kokkai · aozora · e_gov 로 **전부 문어/문서**이고 구어체가 사실상
0 인 반면, ko 는 community_corpus 가 13.7% 를 담당합니다. 그 비대칭을 없앱니다.

``all-corpus-cleaned`` 설정만 씁니다. 원본 ``all-corpus`` 는 필터 전 판본이라
같은 대화가 노이즈까지 포함해 들어옵니다.

발화 단위로 한 줄씩 씁니다 — 대화 전체를 한 줄에 넣으면 foundation 준비가
문장 경계로 다시 나눠야 합니다.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

from sion_translate.console import configure_stdio

REPO = "p1atdev/open2ch"
CONFIG = "all-corpus-cleaned"
_WHITESPACE = re.compile(r"\s+")
# 게시판 인용 기호와 앵커. 남기면 모델이 이 표기를 배웁니다.
_FORUM_NOISE = re.compile(r">>\d+|^>+|https?://\S+")


def clean(text: str) -> str:
    return _WHITESPACE.sub(" ", _FORUM_NOISE.sub(" ", text)).strip()


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="open2ch → foundation JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-characters", type=int, default=8)
    parser.add_argument("--maximum-characters", type=int, default=4000)
    parser.add_argument("--max-bytes", type=float, default=1.0e9, help="목표 분량 상한")
    args = parser.parse_args()

    files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith(CONFIG + "/")]
    if not files:
        raise SystemExit(f"{CONFIG} 파일을 찾지 못했습니다")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = seen_dupes = 0
    seen: set[int] = set()
    with output.open("w", encoding="utf-8") as sink:
        for name in sorted(files):
            if output.exists() and output.stat().st_size >= args.max_bytes:
                print("  목표 분량 도달, 남은 파일 건너뜀", flush=True)
                break
            local = hf_hub_download(REPO, name, repo_type="dataset")
            table = pq.read_table(local)
            column = "dialogue" if "dialogue" in table.column_names else table.column_names[0]
            for value in table.column(column).to_pylist():
                # `dialogue` 는 {"speaker": [...], "content": [...]} 구조체입니다.
                # 문자열 리스트로 가정하면 조용히 0행이 나옵니다 — 실제로 그랬습니다.
                if isinstance(value, dict):
                    turns = value.get("content") or []
                elif isinstance(value, list):
                    turns = value
                else:
                    turns = [value]
                for turn in turns:
                    if not isinstance(turn, str):
                        continue
                    text = clean(turn)
                    if not args.minimum_characters <= len(text) <= args.maximum_characters:
                        continue
                    digest = hash(text)
                    if digest in seen:
                        seen_dupes += 1
                        continue
                    seen.add(digest)
                    sink.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
            sink.flush()
            print(
                f"  {name}: 누적 {written:,}행, {output.stat().st_size / 1e9:.2f} GB",
                flush=True,
            )
    print(f"{output}: {written:,}행, {output.stat().st_size / 1e9:.2f} GB (중복 {seen_dupes:,})")


if __name__ == "__main__":
    main()
